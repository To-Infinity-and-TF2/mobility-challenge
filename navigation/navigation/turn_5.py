#!/usr/bin/env python3
"""
ArtPark Arena Bot Navigation v4 — ROS2 Node (Simulation)
=========================================================
CHANGES FROM v3:
  - ORANGE HSV reverted to tight range matching the ArtPark logo golden-amber
    (H=18-30, S>=130, V>=150) so yellow tape walls are NOT detected as orange.
  - RED blob detection added (HSV wraps: H=0-8 OR H=170-180, S>=140, V>=80).
    When red blob depth < 0.2 m → publish Twist(0,0) STOP and enter S_STOP state.
  - ArUco detections now published to CSV file (aruco_log.csv) with columns:
    id, value, timestamp  — ready for pandas DataFrame ingestion.
  - GREEN HSV kept at the expanded range from v3 (no change).

Subscribes : /r1_mini/camera/image_raw       (sensor_msgs/Image)
             /r1_mini/depth_cam/image_raw    (sensor_msgs/Image)
             /r1_mini/lidar                  (sensor_msgs/LaserScan)
Publishes  : /cmd_vel                        (geometry_msgs/Twist)

State machine
─────────────
APPROACH_MARKER    – Drives toward detected ArUco until <= 0.2 m depth.
INTERSECTION_TURN  – Waits for LiDAR-detected side opening, then turns.
EXPLORATION        – Default: finds least-dense forward path.
FOLLOW_GREEN       – Go toward farther of green/orange (depth); log green depth.
FOLLOW_ORANGE      – Go toward farther of green/orange (depth); log orange depth.
GREEN_ONLY         – Only green visible; move forward, log depth, timeout→EXPLORATION.
ORANGE_ONLY        – Only orange visible; move forward, log depth, timeout→EXPLORATION.
TURNING            – Timed turn in progress.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

import cv2
import numpy as np
import csv
import time
import os
import math
from typing import Optional
from dataclasses import dataclass

# ── Topics ────────────────────────────────────────────────────────────────
RGB_TOPIC   = '/r1_mini/camera/image_raw'
DEPTH_TOPIC = '/r1_mini/depth_cam/image_raw'
LIDAR_TOPIC = '/r1_mini/lidar'

# ── Velocity ──────────────────────────────────────────────────────────────
LINEAR_SPEED_FWD  =  0.6
ANGULAR_TURN      =  0.4
ANGULAR_U_TURN    =  0.8
ANGULAR_SEARCH    =  0.3

TURN_90_DURATION  = 1.5
TURN_180_DURATION = 3.0

# ── Timeouts ──────────────────────────────────────────────────────────────
SINGLE_COLOUR_TIMEOUT = 5.0
APPROACH_TIMEOUT      = 3.0

# ── ArUco ─────────────────────────────────────────────────────────────────
MIN_MARKER_AREA                = 200
UPSCALE_FACTOR                 = 2.0
MARKER_COOLDOWN_S              = 8.0    # Per-ID cooldown after registration
MARKER_APPROACH_DIST_THRESHOLD = 0.50   # Stop + register when depth <= 0.5 m
MARKER_APPROACH_AREA_FALLBACK  = 15000  # Area fallback if depth cam unavailable

# ── HSV — tuned to ArtPark reference map colours ─────────────────────────
# Green (START blob): hue 35-95, good saturation
GREEN_LOW   = np.array([35,  60,  60], dtype=np.uint8)
GREEN_HIGH  = np.array([95, 255, 255], dtype=np.uint8)

# Orange: original reference values from the nav reference file — proven to
# detect the ArtPark orange blob reliably in simulation.
ORANGE_LOW  = np.array([ 5, 120, 120], dtype=np.uint8)
ORANGE_HIGH = np.array([22, 255, 255], dtype=np.uint8)

# Red (STOP blob): hue wraps around 0°; two sub-ranges merged
RED_LOW1    = np.array([  0, 140,  80], dtype=np.uint8)
RED_HIGH1   = np.array([  8, 255, 255], dtype=np.uint8)
RED_LOW2    = np.array([170, 140,  80], dtype=np.uint8)
RED_HIGH2   = np.array([180, 255, 255], dtype=np.uint8)

RED_STOP_DIST = 0.30    # metres — stop if red blob closer than this
MIN_BLOB_AREA = 400

# ── Depth thresholds ──────────────────────────────────────────────────────
COLOUR_FOLLOW_LOG_DIST = 0.50   # Log depth only when blob is within this distance
DEPTH_PATCH_R          = 8      # px radius for median depth sample at centroid

# ── LiDAR ─────────────────────────────────────────────────────────────────
LIDAR_STOP_DIST          = 0.30
LIDAR_WARN_FRONT         = 0.50
LIDAR_CENTERING_GAIN     = 0.8
LIDAR_MAX_CORRECTION     = 0.6
LIDAR_MIN_VALID          = 0.05
LIDAR_INTERSECTION_OPEN  = 0.80   # Side clearance > this → intersection detected
LIDAR_INTERSECTION_ZONE  = 30     # degrees from ±90° checked for side opening

# ── Logging ───────────────────────────────────────────────────────────────
# All three paths are stamped ONCE at import time so each run gets its own
# files and no previous-run data is ever overwritten or mixed in.
# e.g.  ~/aruco_logs/aruco_run_2026-04-18_10-23-01.csv
_LOG_DIR        = os.path.expanduser('~/aruco_logs')
os.makedirs(_LOG_DIR, exist_ok=True)
_RUN_TS         = time.strftime('%Y-%m-%d_%H-%M-%S')          # frozen at startup
LOG_PATH        = os.path.join(_LOG_DIR, f'aruco_run_{_RUN_TS}.txt')
CSV_LOG_PATH    = os.path.join(_LOG_DIR, f'aruco_run_{_RUN_TS}.csv')
COLOUR_LOG_PATH = os.path.join(_LOG_DIR, f'colour_depth_{_RUN_TS}.txt')

# ── States ────────────────────────────────────────────────────────────────
S_EXPLORATION        = 'EXPLORATION'
S_TURNING            = 'TURNING'
S_INTERSECTION_TURN  = 'INTERSECTION_TURN'
S_FOLLOW_GREEN       = 'FOLLOW_GREEN'
S_FOLLOW_ORANGE      = 'FOLLOW_ORANGE'
S_FOLLOW_RED         = 'FOLLOW_RED'
S_GREEN_ONLY         = 'GREEN_ONLY'
S_ORANGE_ONLY        = 'ORANGE_ONLY'
S_APPROACH_MARKER    = 'APPROACH_MARKER'
S_STOP               = 'STOP'          # red blob within 0.2 m → hard stop

# ── ArUco command map ─────────────────────────────────────────────────────
ARUCO_CMD = {
    0: 'TURN_RIGHT',
    1: 'TURN_LEFT',
    2: 'FOLLOW_GREEN',
    3: 'U_TURN',
    4: 'FOLLOW_ORANGE',
}


@dataclass
class Blob:
    cx: int
    cy: int
    area: float


@dataclass
class MarkerEvent:
    aruco_id:  int
    timestamp: float
    command:   str


# ─────────────────────────────────────────────────────────────────────────────
class ArtParkNavNode(Node):

    def __init__(self):
        super().__init__('artpark_nav_node')

        self.declare_parameter('show_debug', True)
        self.show_debug = self.get_parameter('show_debug').value

        self.bridge       = CvBridge()
        self.depth_frame: Optional[np.ndarray] = None
        self._scan:       Optional[LaserScan]  = None

        # ── State ──────────────────────────────────────────────────────
        self._state = S_EXPLORATION

        # Timed turn
        self._turn_end_time   = 0.0
        self._turn_angular_z  = 0.0
        self._turn_next_state = S_EXPLORATION

        # Approach state
        self._target_aruco_id       = None
        self._pre_approach_state    = S_EXPLORATION
        self._last_seen_target_time = 0.0
        self._approach_stopped      = False   # True after stop-tick, before registration

        # Intersection-turn state
        # Pending turn direction: +1 = left, -1 = right
        self._pending_intersection_turn: Optional[float] = None

        # Single-colour timer
        self._single_colour_since = time.time()

        # Cross-function references
        self._last_tag          = None
        self._last_process_time = 0.0

        # ArUco cooldowns (per ID)
        self._marker_last_seen: dict[int, float] = {}
        self._marker_log: list[MarkerEvent]      = []
        self._logged_ids: set[int]               = set()

        # Counters
        self.green_counter = 0
        self.orange_counter = 0
        self.last_green_depth = None
        self.last_orange_depth = None

        # ArUco detector — DICT_APRILTAG_36h11 supports IDs 0-4
        aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.adaptiveThreshWinSizeMin  = 3
        aruco_params.adaptiveThreshWinSizeMax  = 53
        aruco_params.adaptiveThreshWinSizeStep = 4
        aruco_params.minMarkerPerimeterRate    = 0.01
        aruco_params.errorCorrectionRate       = 1.0
        self._det = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Image,     RGB_TOPIC,   self._rgb_cb,   10)
        self.create_subscription(Image,     DEPTH_TOPIC, self._depth_cb, 10)
        self.create_subscription(LaserScan, LIDAR_TOPIC, self._lidar_cb, 10)

        # ── Initialise log files (create dir, write headers) ───────────
        try:
            os.makedirs(_LOG_DIR, exist_ok=True)
        except Exception as ex:
            self.get_logger().warn(f"Could not create log dir {_LOG_DIR}: {ex}")

        # Text log — written fresh for this run (append per detection later)
        try:
            with open(LOG_PATH, 'w') as f:
                f.write(f"ArUco Marker Log — run started {_RUN_TS}\n")
                f.write("=" * 60 + "\n")
                f.write(f"{'ArUco ID':<10} {'Command':<25} {'Source':<15} {'Timestamp'}\n")
                f.write("-" * 60 + "\n")
        except Exception as ex:
            self.get_logger().warn(f"ArUco text log init failed: {ex}")

        # CSV log — write header once; each detection appends a row
        try:
            with open(CSV_LOG_PATH, 'w', newline='') as f:
                csv.writer(f).writerow(['id', 'value', 'timestamp', 'source'])
        except Exception as ex:
            self.get_logger().warn(f"ArUco CSV log init failed: {ex}")

        # Colour depth log — written fresh for this run
        try:
            with open(COLOUR_LOG_PATH, 'w') as f:
                f.write(f"Colour Depth Log — run started {_RUN_TS}\n")
                f.write("=" * 60 + "\n")
        except Exception as ex:
            self.get_logger().warn(f"Colour log init failed: {ex}")

        self.get_logger().info(
            f"ArtPark Nav v4 ready  state={self._state}\n"
            f"  RGB={RGB_TOPIC}  Depth={DEPTH_TOPIC}  LiDAR={LIDAR_TOPIC}\n"
            f"  Logs → {_LOG_DIR}  (run={_RUN_TS})"
        )

    # ── Sensor callbacks ──────────────────────────────────────────────────

    def _lidar_cb(self, msg: LaserScan):
        self._scan = msg

    def _depth_cb(self, msg: Image):
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            self.get_logger().warn(f"Depth decode: {e}")

    def _rgb_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB decode: {e}")
            return

        twist = self._process(frame)
        twist = self._lidar_avoidance(twist)
        self.cmd_pub.publish(twist)
        if self.show_debug:
            cv2.waitKey(1)

    # ── Main pipeline ─────────────────────────────────────────────────────

    def _process(self, frame: np.ndarray) -> Twist:
        dbg  = frame.copy() if self.show_debug else None
        now  = time.time()
        self._last_process_time = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        green  = self._blob(hsv, GREEN_LOW,  GREEN_HIGH,  'GREEN',  dbg)
        orange = self._blob(hsv, ORANGE_LOW, ORANGE_HIGH, 'ORANGE', dbg)
        red    = self._blob_red(hsv, dbg)

        # ── GLOBAL INTERCEPT: RED detection ──────
        if red is not None:
            if self._state == S_FOLLOW_ORANGE:
                self.get_logger().info("Detected red while following orange → FOLLOW_RED")
                self._state = S_FOLLOW_RED
            elif self._state != S_STOP:
                red_depth = self._depth_at(red.cx, red.cy)
                if red_depth is not None and red_depth < RED_STOP_DIST:
                    self.get_logger().warn(
                        f"RED blob at {red_depth:.3f}m < {RED_STOP_DIST}m → STOP"
                    )
                    self._state = S_STOP

        if self._state == S_STOP:
            stop_twist = Twist()   # all zeros = full stop
            self._overlay(dbg, f'STOP (RED DETECTED)', stop_twist)
            self._show(dbg)
            return stop_twist

        # Detect ALL ArUco IDs 0-4; returns best match (highest area, cooldown respected)
        tag = self._aruco(gray, dbg)
        self._last_tag = tag

        twist = Twist()

        # ── GLOBAL INTERCEPT: Act on any valid ArUco (IDs 0-4) ────────
        # Do not interrupt a turn already in progress or an approach already locked on.
        if tag is not None and self._state not in (S_TURNING, S_APPROACH_MARKER):
            tid = tag['id']
            if self._marker_ready(tid, now):
                self.get_logger().info(
                    f"ArUco [{tid}] acquired — cmd={ARUCO_CMD.get(tid,'?')} — entering APPROACH"
                )
                self._pre_approach_state    = self._state
                self._state                 = S_APPROACH_MARKER
                self._target_aruco_id       = tid
                self._last_seen_target_time = now
                self._approach_stopped      = False   # always reset for new target

        # ── APPROACH_MARKER ──────────────────────────────────────────
        if self._state == S_APPROACH_MARKER:
            twist = self._handle_approach(tag, frame.shape[1], now, dbg)
            return twist

        # ── TURNING (timed) ──────────────────────────────────────────
        if self._state == S_TURNING:
            if now < self._turn_end_time:
                twist.angular.z = self._turn_angular_z
                self._overlay(dbg, f"TURNING → {self._turn_next_state}", twist)
                self._show(dbg)
                return twist
            else:
                self.get_logger().info(f"Turn done → {self._turn_next_state}")
                self._state = self._turn_next_state
                self._single_colour_since = now

        # ── INTERSECTION_TURN ────────────────────────────────────────
        if self._state == S_INTERSECTION_TURN:
            twist = self._handle_intersection_turn(now, dbg)
            return twist

        # ── EXPLORATION ──────────────────────────────────────────────
        if self._state == S_EXPLORATION:
            twist = self._handle_exploration(now, dbg)
            return twist

        # ── FOLLOW_GREEN ─────────────────────────────────────────────
        if self._state == S_FOLLOW_GREEN:
            twist = self._handle_follow_depth(
                primary=green, secondary=orange,
                primary_name='GREEN', secondary_name='ORANGE',
                only_state=S_GREEN_ONLY, log_primary=True, now=now
            )
            self._overlay(dbg, 'FOLLOW_GREEN', twist)
            self._show(dbg)
            return twist

        # ── FOLLOW_ORANGE ────────────────────────────────────────────
        if self._state == S_FOLLOW_ORANGE:
            twist = self._handle_follow_depth(
                primary=orange, secondary=green,
                primary_name='ORANGE', secondary_name='GREEN',
                only_state=S_ORANGE_ONLY, log_primary=True, now=now
            )
            self._overlay(dbg, 'FOLLOW_ORANGE', twist)
            self._show(dbg)
            return twist

        # ── FOLLOW_RED ───────────────────────────────────────────────
        if self._state == S_FOLLOW_RED:
            if red is None:
                twist.angular.z = ANGULAR_SEARCH
            else:
                red_depth = self._depth_at(red.cx, red.cy)
                if red_depth is not None and red_depth < RED_STOP_DIST:
                    twist = Twist()  # stop
                else:
                    twist.linear.x = LINEAR_SPEED_FWD
                    err = (frame.shape[1] / 2) - red.cx
                    twist.angular.z = 0.002 * err
            self._overlay(dbg, 'FOLLOW_RED', twist)
            self._show(dbg)
            return twist

        # ── GREEN_ONLY ───────────────────────────────────────────────
        if self._state == S_GREEN_ONLY:
            twist = self._handle_only(
                present=green, other=orange,
                follow_state=S_FOLLOW_GREEN,
                colour_name='GREEN', now=now
            )
            self._overlay(dbg, 'GREEN_ONLY', twist)
            self._show(dbg)
            return twist

        # ── ORANGE_ONLY ──────────────────────────────────────────────
        if self._state == S_ORANGE_ONLY:
            twist = self._handle_only(
                present=orange, other=green,
                follow_state=S_FOLLOW_ORANGE,
                colour_name='ORANGE', now=now
            )
            self._overlay(dbg, 'ORANGE_ONLY', twist)
            self._show(dbg)
            return twist

        self._show(dbg)
        return twist

    # ── State handlers ────────────────────────────────────────────────────

    def _register_marker(self, tid: int, area: float, depth: Optional[float], now: float) -> Twist:
        """
        Log the marker, update cooldown, and execute its command.
        Always called AFTER a stop-twist has already been published
        (guaranteed by _handle_approach's two-tick protocol).
        Returns the first motion twist for the new mode (or zero for turns).
        """
        depth_str = f"{depth:.3f}m" if depth is not None else "N/A"
        cmd = ARUCO_CMD.get(tid, '?')
        self.get_logger().info(
            f"[REGISTER] ArUco {tid} | cmd={cmd} | depth={depth_str} | area={area:.0f}"
        )
        # Stamp cooldown + log
        self._marker_last_seen[tid] = now
        self._approach_stopped      = False
        self._log_marker(tid, now)

        twist = Twist()   # start from rest

        if tid == 0:   # ── TURN RIGHT at next intersection ─────────────
            self.get_logger().info("Cmd: TURN_RIGHT — moving to INTERSECTION_TURN state.")
            self._pending_intersection_turn = -1.0
            self._state = S_INTERSECTION_TURN

        elif tid == 1: # ── TURN LEFT at next intersection ──────────────
            self.get_logger().info("Cmd: TURN_LEFT — moving to INTERSECTION_TURN state.")
            self._pending_intersection_turn = 1.0
            self._state = S_INTERSECTION_TURN

        elif tid == 2: # ── START FOLLOWING GREEN ─────────────────────
            self.get_logger().info("Cmd: FOLLOW_GREEN")
            self._state = S_FOLLOW_GREEN
            self._single_colour_since = now

        elif tid == 3: # ── U-TURN ───────────────────────────────────
            self.get_logger().info("Cmd: U_TURN")
            self._start_turn(ANGULAR_U_TURN, TURN_180_DURATION, S_EXPLORATION)
            twist.angular.z = ANGULAR_U_TURN

        elif tid == 4: # ── START FOLLOWING ORANGE ───────────────────
            self.get_logger().info("Cmd: FOLLOW_ORANGE")
            self._state = S_FOLLOW_ORANGE
            self._single_colour_since = now

        return twist

    def _handle_approach(self, tag, frame_width: int, now: float, dbg) -> Twist:
        """
        Approach the target ArUco marker.

        Phase 1 — CLOSING: drive forward slowly, steering to keep the marker
                  centred, until depth <= MARKER_APPROACH_DIST_THRESHOLD (0.5 m).
        Phase 2 — STOPPED: publish a full stop for one tick, log the marker,
                  then immediately execute the command and switch state.
                  This guarantees a zero-velocity Twist is published before
                  any mode-switch twist so the robot physically halts first.
        """
        twist = Twist()   # default = full stop

        if tag is not None and tag['id'] == self._target_aruco_id:
            self._last_seen_target_time = now
            area       = tag['area']
            cx, cy     = tag['center']
            depth      = self._depth_at(cx, cy)   # metres from depth channel

            # Determine whether we have reached the stop distance
            is_close_enough = False
            if depth is not None and depth > 0.0:
                is_close_enough = depth <= MARKER_APPROACH_DIST_THRESHOLD
            elif area >= MARKER_APPROACH_AREA_FALLBACK:
                is_close_enough = True

            if is_close_enough:
                # ── Phase 2: STOP first, then register ───────────────
                # If we haven't published the stop yet this registration,
                # do it now and flag that we are in the stop tick.
                if not self._approach_stopped:
                    self._approach_stopped = True
                    depth_disp = f"{depth:.3f}m" if depth is not None else "N/A"
                    self.get_logger().info(
                        f"ArUco [{self._target_aruco_id}] within {MARKER_APPROACH_DIST_THRESHOLD}m "
                        f"(depth={depth_disp}) — publishing STOP then registering."
                    )
                    # Publish explicit zero-velocity stop
                    self.cmd_pub.publish(Twist())
                    self._overlay(dbg, f"STOP @ ArUco [{self._target_aruco_id}] D:{depth_disp}", twist)
                    self._show(dbg)
                    return twist   # return stop twist; next tick will register

                # Second tick after stop: log + execute command
                return self._register_marker(self._target_aruco_id, area, depth, now)

            else:
                # ── Phase 1: CLOSING ─────────────────────────────────
                self._approach_stopped = False   # reset stop flag while still closing
                err = (frame_width / 2) - cx
                twist.angular.z = 0.002 * err          # proportional centre steering
                # Slow down proportionally as we close in
                if depth is not None and depth > 0.0:
                    speed_scale = min(1.0, max(0.3, (depth - MARKER_APPROACH_DIST_THRESHOLD) / 1.0))
                else:
                    speed_scale = 0.5
                twist.linear.x = LINEAR_SPEED_FWD * speed_scale
                depth_disp = f"{depth:.3f}m" if depth is not None else "N/A"
                self._overlay(
                    dbg,
                    f"APPROACH [{self._target_aruco_id}] D:{depth_disp} A:{area:.0f} "
                    f"spd={twist.linear.x:.2f}",
                    twist
                )
                self._show(dbg)
                return twist

        else:
            # ── Target not in frame ───────────────────────────────────
            self._approach_stopped = False
            lost = now - self._last_seen_target_time
            if lost > APPROACH_TIMEOUT:
                self.get_logger().warn(
                    f"Lost ArUco [{self._target_aruco_id}] for {lost:.1f}s → EXPLORATION"
                )
                self._state = S_EXPLORATION
            else:
                # Creep forward hoping to re-acquire
                twist.linear.x = LINEAR_SPEED_FWD * 0.3
            self._overlay(dbg, f"APPROACH LOST [{self._target_aruco_id}] ({lost:.1f}s)", twist)
            self._show(dbg)
            return twist

    def _handle_intersection_turn(self, now: float, dbg) -> Twist:
        """
        Drive forward until LiDAR detects a side opening (intersection),
        then execute the pending turn direction.
        """
        twist = Twist()
        twist.linear.x = LINEAR_SPEED_FWD * 0.5   # slow approach to intersection

        direction = self._pending_intersection_turn  # +1=left, -1=right

        # Check the relevant side for a substantial opening
        if direction is not None:
            if direction > 0:  # Expecting left opening
                side_dist = self._range_zone(
                    90 - LIDAR_INTERSECTION_ZONE,
                    90 + LIDAR_INTERSECTION_ZONE,
                    use_avg=True
                )
                side_label = "LEFT"
            else:              # Expecting right opening
                side_dist = self._range_zone(
                    -90 - LIDAR_INTERSECTION_ZONE,
                    -90 + LIDAR_INTERSECTION_ZONE,
                    use_avg=True
                )
                side_label = "RIGHT"

            if side_dist > LIDAR_INTERSECTION_OPEN:
                self.get_logger().info(
                    f"Intersection detected ({side_label} open {side_dist:.2f}m) — turning."
                )
                self._pending_intersection_turn = None
                self._start_turn(
                    ANGULAR_TURN * direction,
                    TURN_90_DURATION,
                    S_EXPLORATION
                )
                twist.angular.z = ANGULAR_TURN * direction
                twist.linear.x  = 0.0
            else:
                self._overlay(dbg, f"WAIT INTERSECTION ({side_label} {side_dist:.2f}m)", twist)
        else:
            # No pending turn; fall back
            self._state = S_EXPLORATION

        self._show(dbg)
        return twist

    def _handle_exploration(self, now: float, dbg) -> Twist:
        twist = Twist()
        twist.linear.x = LINEAR_SPEED_FWD

        best_angle = 0.0
        max_dist   = -1.0
        for ang in [-45, 0, 45]:
            dist = self._range_zone(ang - 15, ang + 15, use_avg=True)
            if dist > max_dist:
                max_dist   = dist
                best_angle = ang

        twist.angular.z = math.radians(best_angle) * 0.5
        self._overlay(dbg, f'EXPLORATION (bias {best_angle}°)', twist)
        self._show(dbg)
        return twist

    def _handle_follow_depth(
        self,
        primary: Optional[Blob],
        secondary: Optional[Blob],
        primary_name: str,
        secondary_name: str,
        only_state: str,
        log_primary: bool,
        now: float
    ) -> Twist:
        """
        Follow whichever colour blob is FARTHER away (depth in metres).
        If both are at the sides (no forward depth reading usable),
        enter EXPLORATION mode.
        Log the primary colour's depth if it is within COLOUR_FOLLOW_LOG_DIST.
        """
        twist = Twist()

        if primary is None and secondary is None:
            # Nothing visible — spin slowly
            twist.angular.z = ANGULAR_SEARCH
            return twist

        # Get depth for each blob (metres via depth channel)
        primary_d   = self._depth_at(primary.cx,   primary.cy)   if primary   else None
        secondary_d = self._depth_at(secondary.cx, secondary.cy) if secondary else None

        # ── Only primary visible ─────────────────────────────────────
        if secondary is None:
            if primary_d is None:
                # Blob visible but no depth → sides/noise → EXPLORATION
                self.get_logger().info(f"Only {primary_name} visible but no depth → EXPLORATION")
                self._state = S_EXPLORATION
                return Twist()
            if log_primary:
                self._log_colour(primary_name, primary_d, now)
            # Check side-only heuristic: blob centroid far from horizontal centre
            # If depth is small the blob might be beside us rather than ahead
            self.get_logger().info(f"Only {primary_name} → {only_state} (depth={primary_d:.3f}m)")
            self._state = only_state
            self._single_colour_since = now
            twist.linear.x = LINEAR_SPEED_FWD
            return twist

        # ── Only secondary visible ───────────────────────────────────
        if primary is None:
            twist.angular.z = ANGULAR_SEARCH
            return twist

        # ── Both visible — go toward the FARTHER one ─────────────────
        # Fallback to blob Y if depth unavailable (larger Y = closer in image coords)
        if primary_d is None:
            primary_d   = 1.0 / max(primary.cy,   1)   # invert pixel Y as proxy
        if secondary_d is None:
            secondary_d = 1.0 / max(secondary.cy, 1)

        # Log primary depth if within threshold
        if log_primary:
            actual_d = self._depth_at(primary.cx, primary.cy)
            if actual_d is not None:
                self._log_colour(primary_name, actual_d, now)
                # Update counters
                if primary_name == 'GREEN':
                    if self.last_green_depth is not None and abs(actual_d - self.last_green_depth) > 0.5:
                        self.green_counter += 1
                        self.get_logger().info(f"Green counter incremented to {self.green_counter}")
                    self.last_green_depth = actual_d
                elif primary_name == 'ORANGE':
                    if self.last_orange_depth is not None and abs(actual_d - self.last_orange_depth) > 0.5:
                        self.orange_counter += 1
                        self.get_logger().info(f"Orange counter incremented to {self.orange_counter}")
                    self.last_orange_depth = actual_d

        # Side-blob check: if BOTH blobs have no meaningful forward depth,
        # they are likely to the sides → EXPLORATION
        actual_p = self._depth_at(primary.cx,   primary.cy)
        actual_s = self._depth_at(secondary.cx, secondary.cy)
        if actual_p is None and actual_s is None:
            self.get_logger().info("Both blobs lack depth (side-only) → EXPLORATION")
            self._state = S_EXPLORATION
            return Twist()

        if primary_d >= secondary_d:
            # Primary is farther → move toward primary (straight ahead or slight steer)
            twist.linear.x = LINEAR_SPEED_FWD
            self.get_logger().debug(
                f"Following {primary_name} (farther: {primary_d:.3f}m > {secondary_d:.3f}m)"
            )
        else:
            # Secondary is farther → steer toward secondary
            twist.linear.x = LINEAR_SPEED_FWD
            # Proportional horizontal steering toward secondary centroid
            # (frame_width unknown here; use sign of cx difference)
            if secondary.cx < primary.cx:
                twist.angular.z =  0.3   # secondary is to the left
            else:
                twist.angular.z = -0.3   # secondary is to the right
            self.get_logger().debug(
                f"{secondary_name} is farther ({secondary_d:.3f}m) → steering toward it"
            )

        return twist

    def _handle_only(
        self,
        present: Optional[Blob],
        other:   Optional[Blob],
        follow_state: str,
        colour_name: str,
        now: float
    ) -> Twist:
        twist = Twist()

        if other is not None:
            self.get_logger().info(f"Second colour back → {follow_state}")
            self._state = follow_state
            self._single_colour_since = now
            twist.linear.x = LINEAR_SPEED_FWD
            return twist

        if present is not None:
            depth = self._depth_at(present.cx, present.cy)
            if depth is not None:
                self._log_colour(colour_name, depth, now)
                if depth > COLOUR_FOLLOW_LOG_DIST:
                    self.get_logger().info(
                        f"{colour_name} depth {depth:.3f}m > {COLOUR_FOLLOW_LOG_DIST}m → EXPLORATION"
                    )
                    self._state = S_EXPLORATION
                    return Twist()

        elapsed = now - self._single_colour_since
        if elapsed >= SINGLE_COLOUR_TIMEOUT:
            self.get_logger().warn(f"Single colour timeout ({elapsed:.1f}s) → EXPLORATION")
            self._state = S_EXPLORATION
            return Twist()

        twist.linear.x = LINEAR_SPEED_FWD
        return twist

    # ── Timed turn helper ─────────────────────────────────────────────────

    def _start_turn(self, angular_z: float, duration: float, next_state: str):
        self._state           = S_TURNING
        self._turn_end_time   = time.time() + duration
        self._turn_angular_z  = angular_z
        self._turn_next_state = next_state
        direction = 'LEFT' if angular_z > 0 else 'RIGHT' if angular_z < 0 else 'U'
        self.get_logger().info(f"Turn {direction} {duration:.2f}s → {next_state}")

    # ── ArUco detection ───────────────────────────────────────────────────

    def _aruco(self, gray: np.ndarray, dbg) -> Optional[dict]:
        """
        Detects ALL ArUco IDs in {0,1,2,3,4}.
        Returns the largest-area valid (cooldown OK) marker found.
        """
        s    = UPSCALE_FACTOR
        big  = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        corners_list, ids, _ = self._det.detectMarkers(big)

        if ids is None or len(ids) == 0:
            return None

        valid_ids    = {0, 1, 2, 3, 4}
        now          = time.time()
        best         = None
        best_area    = -1.0

        for i, raw_id in enumerate(ids):
            tid = int(raw_id[0])
            if tid not in valid_ids:
                continue
            if tid not in self._logged_ids:
                self._logged_ids.add(tid)
                self._log_marker(tid, now, source='DETECTION')
            if not self._marker_ready(tid, now):
                continue
            corners_orig = corners_list[i] / s
            area = cv2.contourArea(corners_orig.astype(np.float32))
            if area < MIN_MARKER_AREA:
                continue
            if area > best_area:
                best_area = area
                bc = corners_orig[0]
                cx = int(bc[:, 0].mean())
                cy = int(bc[:, 1].mean())
                best = {
                    'id':     tid,
                    'center': (cx, cy),
                    'area':   area,
                    'corners': corners_orig,
                }

        if best is None:
            return None

        if dbg is not None:
            cv2.aruco.drawDetectedMarkers(
                dbg,
                [best['corners'].reshape(1, 4, 2).astype(np.float32)],
                np.array([[best['id']]])
            )
            cx, cy = best['center']
            label = f"[{best['id']}] {ARUCO_CMD.get(best['id'], '?')}"
            cv2.putText(dbg, label, (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        return best

    def _marker_ready(self, tid: int, now: float) -> bool:
        return (now - self._marker_last_seen.get(tid, 0.0)) > MARKER_COOLDOWN_S

    # ── Red blob (dual-range, wraps around hue=0) ─────────────────────────

    def _blob_red(self, hsv, dbg) -> Optional[Blob]:
        """
        Red hue wraps around 0° in OpenCV HSV.
        Combine two sub-ranges and find the largest contour.
        """
        mask1 = cv2.inRange(hsv, RED_LOW1, RED_HIGH1)
        mask2 = cv2.inRange(hsv, RED_LOW2, RED_HIGH2)
        mask  = cv2.bitwise_or(mask1, mask2)

        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        largest = max(cnts, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        if area < MIN_BLOB_AREA:
            return None

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if dbg is not None:
            col = (0, 0, 220)
            cv2.drawContours(dbg, [largest], -1, col, 2)
            cv2.circle(dbg, (cx, cy), 9, col, -1)
            depth = self._depth_at(cx, cy)
            depth_str = f"{depth:.3f}m" if depth else "N/A"
            cv2.putText(dbg, f"RED {area:.0f}px D:{depth_str}",
                        (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        return Blob(cx=cx, cy=cy, area=area)

    # ── Colour blob ───────────────────────────────────────────────────────

    def _blob(self, hsv, lo, hi, label: str, dbg) -> Optional[Blob]:
        mask = cv2.inRange(hsv, lo, hi)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not cnts:
            return None
        largest = max(cnts, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        if area < MIN_BLOB_AREA:
            return None

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        if dbg is not None:
            col = (0, 220, 0) if label == 'GREEN' else (0, 140, 255)
            cv2.drawContours(dbg, [largest], -1, col, 2)
            cv2.circle(dbg, (cx, cy), 9, col, -1)
            depth = self._depth_at(cx, cy)
            depth_str = f"{depth:.3f}m" if depth else "N/A"
            cv2.putText(dbg, f"{label} {area:.0f}px D:{depth_str}",
                        (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        return Blob(cx=cx, cy=cy, area=area)

    # ── Depth sampling (DEPTH CHANNEL) ────────────────────────────────────

    def _depth_at(self, cx: int, cy: int) -> Optional[float]:
        """
        Returns the median depth (metres) in a patch around (cx, cy)
        using the 32FC1 depth image from the depth camera.
        Returns None if depth frame is unavailable or no valid readings.
        """
        if self.depth_frame is None:
            return None
        dh, dw = self.depth_frame.shape[:2]
        r  = DEPTH_PATCH_R
        x0, x1 = max(cx - r, 0), min(cx + r, dw)
        y0, y1 = max(cy - r, 0), min(cy + r, dh)
        patch = self.depth_frame[y0:y1, x0:x1]
        valid = patch[(patch > LIDAR_MIN_VALID) & np.isfinite(patch)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    # ── LiDAR helpers ─────────────────────────────────────────────────────

    def _range_zone(self, lo_deg: float, hi_deg: float, use_avg: bool = False) -> float:
        if self._scan is None:
            return 999.0
        scan = self._scan

        def to_idx(a: float) -> int:
            i = int(round((a - scan.angle_min) / scan.angle_increment))
            return max(0, min(i, len(scan.ranges) - 1))

        i0 = to_idx(math.radians(lo_deg))
        i1 = to_idx(math.radians(hi_deg))
        if i0 > i1:
            i0, i1 = i1, i0

        vals = []
        for r in scan.ranges[i0:i1 + 1]:
            if math.isfinite(r) and r > LIDAR_MIN_VALID:
                vals.append(min(r, scan.range_max))
            else:
                vals.append(scan.range_max)

        if not vals:
            return 999.0
        return float(np.mean(vals)) if use_avg else float(min(vals))

    def _find_least_dense_direction(self) -> float:
        if self._scan is None:
            return 1.0
        max_avg   = -1.0
        best_angle = 0.0
        for angle_deg in range(-180, 180, 15):
            avg = self._range_zone(angle_deg - 15, angle_deg + 15, use_avg=True)
            if avg > max_avg:
                max_avg    = avg
                best_angle = angle_deg
        return float(best_angle)

    def _lidar_avoidance(self, twist: Twist) -> Twist:
        # Never override a stop command (linear.x == 0 and we're approaching or stopped)
        if self._scan is None:
            return twist

        # While approaching a marker, the approach logic itself controls stopping.
        # Only apply the centering correction, not the hard-stop override.
        in_approach = self._state == S_APPROACH_MARKER

        if twist.linear.x <= 0.0 and not in_approach:
            return twist

        # Use a tighter effective stop only when NOT in approach (let approach handle it)
        effective_stop = LIDAR_STOP_DIST if not in_approach else 0.12
        front = self._range_zone(-20, 20)

        if front < effective_stop and not in_approach:
            tag = self._last_tag
            now = self._last_process_time

            # Wall recovery
            if self._state != S_TURNING:
                best_angle = self._find_least_dense_direction()
                if best_angle == 0.0:
                    best_angle = 180.0
                duration = abs(math.radians(best_angle)) / ANGULAR_TURN
                turn_vel = ANGULAR_TURN if best_angle > 0 else -ANGULAR_TURN
                self.get_logger().warn(
                    f"LIDAR HARD STOP front={front:.2f}m → recovery turn {best_angle:.0f}°"
                )
                self._start_turn(turn_vel, duration, S_EXPLORATION)

            twist.linear.x = 0.0
            return twist

        # Dynamic centering (applies in all moving states including approach)
        left_dist  = self._range_zone( 20,  90, use_avg=True)
        right_dist = self._range_zone(-90, -20, use_avg=True)

        if left_dist < 1.5 or right_dist < 1.5:
            l_cap = min(left_dist, 1.5)
            r_cap = min(right_dist, 1.5)
            correction = LIDAR_CENTERING_GAIN * (l_cap - r_cap)
            correction = float(np.clip(correction, -LIDAR_MAX_CORRECTION, LIDAR_MAX_CORRECTION))
            twist.angular.z += correction

        if front < LIDAR_WARN_FRONT and not in_approach:
            slow = (front - effective_stop) / (LIDAR_WARN_FRONT - effective_stop)
            twist.linear.x = max(0.0, twist.linear.x * slow)

        return twist

    # ── Logging ───────────────────────────────────────────────────────────

    def _log_marker(self, tid: int, ts: float, source: str = 'REGISTRATION'):
        """
        Append a single ArUco event to the text log and CSV.
        `source` marks whether this was a raw detection or a registration.
        The in-memory _marker_log list is kept for any runtime queries.
        """
        cmd = ARUCO_CMD.get(tid, '?')
        self._marker_log.append(MarkerEvent(tid, ts, cmd))
        ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
        self.get_logger().info(f"[ARUCO LOG] source={source} ID={tid} cmd={cmd} @{ts_str}")

        VALUE_MAP = {
            0: 'Take right',
            1: 'Take left',
            2: 'Start following green',
            3: 'Take u turn',
            4: 'Start following orange',
        }
        value = VALUE_MAP.get(tid, f'Unknown tag {tid}')

        # ── Append one line to text log ───────────────────────────────
        try:
            with open(LOG_PATH, 'a') as f:
                f.write(f"{tid:<10} {cmd:<25} {source:<15} {ts_str}\n")
        except Exception as ex:
            self.get_logger().warn(f"ArUco text log append failed: {ex}")

        # ── Append one row to CSV ─────────────────────────────────────
        try:
            with open(CSV_LOG_PATH, 'a', newline='') as f:
                csv.writer(f).writerow([tid, value, ts_str, source])
        except Exception as ex:
            self.get_logger().warn(f"ArUco CSV log append failed: {ex}")

    def _log_colour(self, colour: str, depth: float, ts: float):
        """Append a colour-depth reading to the colour log file."""
        ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
        line   = f"{colour:<8} depth={depth:.4f}m  @{ts_str}\n"
        self.get_logger().debug(f"[COLOUR LOG] {line.strip()}")
        try:
            with open(COLOUR_LOG_PATH, 'a') as f:
                f.write(line)
        except Exception as ex:
            self.get_logger().warn(f"Colour log write failed: {ex}")

    # ── Debug overlay / display ───────────────────────────────────────────

    def _overlay(self, dbg, label: str, twist: Twist):
        if dbg is None:
            return
        f = self._range_zone(-20,  20)            if self._scan else 0.0
        l = self._range_zone( 20,  90, use_avg=True) if self._scan else 0.0
        r = self._range_zone(-90, -20, use_avg=True) if self._scan else 0.0
        text = (f"{label} | lin={twist.linear.x:+.2f} ang={twist.angular.z:+.2f}"
                f" | LiDAR F:{f:.2f} L:{l:.2f} R:{r:.2f}")
        col = (0, 255, 0) if twist.linear.x > 0 else (0, 0, 255) if twist.linear.x < 0 else (0, 200, 255)
        cv2.rectangle(dbg, (0, 0), (dbg.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(dbg, text, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    def _show(self, dbg):
        if dbg is not None:
            cv2.imshow("ArtPark Nav v4", dbg)


# ── Entry point ───────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = ArtParkNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.get_logger().info("Safety stop. Shutting down.")
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
