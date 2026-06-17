#!/usr/bin/env python3
"""
ArtPark Arena Bot Navigation — ROS2 Node (Simulation)
======================================================
Subscribes : /r1_mini/camera/image_raw      (sensor_msgs/Image)   ← RGB
             /r1_mini/depth_cam/image_raw   (sensor_msgs/Image)   ← Depth
             /r1_mini/scan                  (sensor_msgs/LaserScan) ← LiDAR
Publishes  : /cmd_vel                       (geometry_msgs/Twist)

AprilTag → Action mapping (36h11 dictionary):
  Tag 0 → TURN RIGHT   (−90°)
  Tag 1 → TURN LEFT    (+90°)
  Tag 2 → FOLLOW GREEN (colour mode switch)
  Tag 3 → U-TURN       (180°)
  Tag 4 → FOLLOW ORANGE (colour mode switch)

Action execution:
  - Rotational goals (tags 0, 1, 3) spin the bot by a target angle,
    checking LiDAR on every Twist publish to abort if something is too close
  - Colour-switch goals (tags 2, 4) are instantaneous state changes
  - A tag is only acted on ONCE per sighting — a cooldown prevents
    re-triggering while the bot is still in frame of the same tag
"""

import math
import threading
import time
from enum import Enum, auto
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


# ---------------------------------------------------------------------------
# TOPICS
# ---------------------------------------------------------------------------
RGB_TOPIC   = '/r1_mini/camera/image_raw'
DEPTH_TOPIC = '/r1_mini/depth_cam/image_raw'
LIDAR_TOPIC = '/r1_mini/scan'
CMD_TOPIC   = '/cmd_vel'

# ---------------------------------------------------------------------------
# VELOCITY CONSTANTS
# ---------------------------------------------------------------------------
LINEAR_SPEED_FWD  =  0.2     # m/s
LINEAR_SPEED_BWD  = -0.15    # m/s
ANGULAR_TURN      =  0.5     # rad/s — used during action turns
ANGULAR_U_TURN    =  0.5     # rad/s — same speed, just longer duration

# ---------------------------------------------------------------------------
# ACTION GOAL CONFIG
# ---------------------------------------------------------------------------
TURN_RIGHT_DEG  = -90.0   # degrees
TURN_LEFT_DEG   =  90.0
U_TURN_DEG      =  180.0

# How long to wait after an action finishes before re-enabling tag detection
# Prevents the same tag from firing again immediately
TAG_COOLDOWN_SEC = 3.0

# LiDAR safety — stop action if anything within this radius (metres)
# Applies only during rotational actions, NOT during colour following
LIDAR_STOP_DIST  = 0.25   # m

# Sectors (degrees) to check for LiDAR obstacles during each action
# RIGHT turn: check right arc; LEFT: check left arc; U-TURN: check rear
LIDAR_SECTOR_RIGHT  = (-120, -30)   # degrees relative to forward (0°)
LIDAR_SECTOR_LEFT   = (  30, 120)
LIDAR_SECTOR_UTURN  = ( 150, 210)   # rear

# ---------------------------------------------------------------------------
# TEMPLATE MATCHING CONFIG
# ---------------------------------------------------------------------------
TEMPLATE_PATH            = Path(__file__).parent / 'artpark_logo.png'
TEMPLATE_MATCH_THRESHOLD = 0.55
TEMPLATE_SCALES          = np.arange(0.3, 1.2, 0.1)
EXIT_DEBOUNCE_FRAMES     = 8

# ---------------------------------------------------------------------------
# HSV COLOUR RANGES
# ---------------------------------------------------------------------------
GREEN_HSV_LOW   = np.array([40,  80,  80],  dtype=np.uint8)
GREEN_HSV_HIGH  = np.array([85, 255, 255],  dtype=np.uint8)
ORANGE_HSV_LOW  = np.array([5,  120, 120],  dtype=np.uint8)
ORANGE_HSV_HIGH = np.array([22, 255, 255],  dtype=np.uint8)

MIN_BLOB_AREA        = 500
SIDE_THRESHOLD_RATIO = 0.15


# ---------------------------------------------------------------------------
# State machine states
# ---------------------------------------------------------------------------
class BotState(Enum):
    FOLLOWING   = auto()   # normal colour-follow mode
    TURNING     = auto()   # executing a rotational action goal
    COOLDOWN    = auto()   # waiting after an action before re-enabling tags


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ColourBlob:
    centroid:   Tuple[int, int]
    area:       float
    distance_m: Optional[float] = None


@dataclass
class TemplateResult:
    on_tile:    bool
    confidence: float
    match_loc:  Optional[Tuple[int, int]] = None
    match_size: Optional[Tuple[int, int]] = None


@dataclass
class TileTracker:
    tiles_passed:        int  = 0
    on_tile:             bool = False
    tile_done:           bool = False
    _below_thresh_count: int  = field(default=0, repr=False)

    def update(self, result: TemplateResult) -> None:
        self.tile_done = False
        if result.on_tile:
            self.on_tile = True
            self._below_thresh_count = 0
        else:
            if self.on_tile:
                self._below_thresh_count += 1
                if self._below_thresh_count >= EXIT_DEBOUNCE_FRAMES:
                    self.on_tile             = False
                    self._below_thresh_count = 0
                    self.tiles_passed       += 1
                    self.tile_done           = True


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------
class ArtParkNavNode(Node):

    def __init__(self):
        super().__init__('artpark_nav_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('active_colour', 'GREEN')
        self.declare_parameter('show_debug',    True)
        self.declare_parameter('template_path', str(TEMPLATE_PATH))

        self.active_colour = self.get_parameter('active_colour').value.upper()
        self.show_debug    = self.get_parameter('show_debug').value

        # ── State machine ──────────────────────────────────────────────
        self.bot_state      = BotState.FOLLOWING
        self._state_lock    = threading.Lock()
        self._cooldown_until: float = 0.0   # epoch time

        # ── Shared sensor data ─────────────────────────────────────────
        self._depth_lock  = threading.Lock()
        self._depth_frame: Optional[np.ndarray] = None

        self._lidar_lock  = threading.Lock()
        self._lidar_msg:  Optional[LaserScan]   = None

        self._debug_lock  = threading.Lock()
        self._debug_frame: Optional[np.ndarray] = None

        # ── Subsystems ─────────────────────────────────────────────────
        self.bridge       = CvBridge()
        self.tile_tracker = TileTracker()

        tmpl_path = self.get_parameter('template_path').value
        self._template_gray, self._template_scales_cache = self._load_template(tmpl_path)

        self._apriltag_detector = None
        self._init_apriltag_detector()

        # ── Publisher ──────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, CMD_TOPIC, 10)

        # ── Subscribers ────────────────────────────────────────────────
        self.create_subscription(Image,     RGB_TOPIC,   self._rgb_cb,   10)
        self.create_subscription(Image,     DEPTH_TOPIC, self._depth_cb, 10)
        self.create_subscription(LaserScan, LIDAR_TOPIC, self._lidar_cb, 10)

        # ── Debug display timer ────────────────────────────────────────
        if self.show_debug:
            self.create_timer(1.0 / 30.0, self._display_timer_cb)

        self.get_logger().info(
            f'\nArtPark Nav Node started'
            f'\n  RGB   : {RGB_TOPIC}'
            f'\n  Depth : {DEPTH_TOPIC}'
            f'\n  LiDAR : {LIDAR_TOPIC}'
            f'\n  Mode  : {self.active_colour}'
        )

    # ======================================================================
    # Sensor callbacks
    # ======================================================================

    def _depth_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            with self._depth_lock:
                self._depth_frame = frame
        except Exception as e:
            self.get_logger().warn(f'Depth decode error: {e}')

    def _lidar_cb(self, msg: LaserScan):
        with self._lidar_lock:
            self._lidar_msg = msg

    def _rgb_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB decode error: {e}')
            return

        self.active_colour = self.get_parameter('active_colour').value.upper()

        with self._depth_lock:
            depth_snapshot = (
                self._depth_frame.copy() if self._depth_frame is not None else None
            )
        with self._lidar_lock:
            lidar_snapshot = self._lidar_msg   # LaserScan is read-only here, no copy needed

        twist, debug_frame = self._process_frame(frame, depth_snapshot, lidar_snapshot)

        # Only publish Twist when in FOLLOWING state — action turns publish internally
        with self._state_lock:
            current_state = self.bot_state
            if current_state == BotState.COOLDOWN and time.time() > self._cooldown_until:
                self.bot_state = BotState.FOLLOWING
                current_state  = BotState.FOLLOWING
                self.get_logger().info('Cooldown complete — resuming colour follow')

        if current_state == BotState.FOLLOWING:
            self.cmd_pub.publish(twist)

        if self.show_debug and debug_frame is not None:
            with self._debug_lock:
                self._debug_frame = debug_frame

    # ======================================================================
    # Display timer
    # ======================================================================

    def _display_timer_cb(self):
        with self._debug_lock:
            frame = self._debug_frame
        if frame is not None:
            cv2.imshow('ArtPark Nav [sim]', frame)
            cv2.waitKey(1)

    # ======================================================================
    # Main per-frame pipeline
    # ======================================================================

    def _process_frame(
        self,
        frame:  np.ndarray,
        depth:  Optional[np.ndarray],
        lidar:  Optional[LaserScan],
    ) -> Tuple[Twist, Optional[np.ndarray]]:

        debug = frame.copy() if self.show_debug else None
        h, w  = frame.shape[:2]

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Template matching ──────────────────────────────────────────
        tmpl_result = self._run_template_match(gray)
        self.tile_tracker.update(tmpl_result)
        if self.tile_tracker.tile_done:
            self.get_logger().info(
                f'✓ Tile DONE — total: {self.tile_tracker.tiles_passed}'
            )
        if debug is not None:
            self._draw_tile_info(debug, tmpl_result)

        # ── AprilTag detection ─────────────────────────────────────────
        with self._state_lock:
            can_act = self.bot_state == BotState.FOLLOWING

        if can_act:
            tag = self._detect_apriltag(gray, debug)
            if tag is not None:
                # Fire the action in a separate thread so RGB callbacks keep running
                threading.Thread(
                    target=self._execute_tag_action,
                    args=(tag['id'], lidar),
                    daemon=True,
                ).start()

        # ── Colour blobs (for FOLLOWING state) ────────────────────────
        green_blob  = self._detect_blob(hsv, GREEN_HSV_LOW,  GREEN_HSV_HIGH,  'GREEN',  debug, depth)
        orange_blob = self._detect_blob(hsv, ORANGE_HSV_LOW, ORANGE_HSV_HIGH, 'ORANGE', debug, depth)

        twist = self._blobs_to_twist(green_blob, orange_blob, h)

        if depth is not None and twist.linear.x > 0:
            twist = self._apply_depth_guard(twist, depth)

        # ── Debug overlays ─────────────────────────────────────────────
        if debug is not None:
            with self._state_lock:
                state_str = self.bot_state.name
            self._draw_overlay(debug, f'{state_str} | {self.active_colour}', twist)
            if lidar is not None:
                self._draw_lidar_arcs(debug, lidar)

        return twist, debug

    # ======================================================================
    # AprilTag Action Executor  (runs in its own thread)
    # ======================================================================

    def _execute_tag_action(self, tag_id: int, lidar: Optional[LaserScan]):
        """
        Interprets the tag ID and executes the corresponding action.
        Blocks until the action is complete, then sets COOLDOWN.
        Runs in a daemon thread so it never blocks the ROS executor.
        """
        with self._state_lock:
            if self.bot_state != BotState.FOLLOWING:
                return   # another action already running
            self.bot_state = BotState.TURNING

        action_name = {0: 'TURN RIGHT', 1: 'TURN LEFT',
                       2: 'FOLLOW GREEN', 3: 'U-TURN', 4: 'FOLLOW ORANGE'}.get(tag_id, '?')
        self.get_logger().info(f'[TAG {tag_id}] Action → {action_name}')

        if tag_id == 0:
            self._action_rotate(TURN_RIGHT_DEG, lidar)

        elif tag_id == 1:
            self._action_rotate(TURN_LEFT_DEG, lidar)

        elif tag_id == 2:
            self._action_set_colour('GREEN')

        elif tag_id == 3:
            self._action_rotate(U_TURN_DEG, lidar)

        elif tag_id == 4:
            self._action_set_colour('ORANGE')

        # Enter cooldown
        with self._state_lock:
            self.bot_state       = BotState.COOLDOWN
            self._cooldown_until = time.time() + TAG_COOLDOWN_SEC

        self.get_logger().info(
            f'[TAG {tag_id}] Action complete — cooldown {TAG_COOLDOWN_SEC}s'
        )

    # ------------------------------------------------------------------
    # Action: rotate by target_deg degrees
    # ------------------------------------------------------------------
    def _action_rotate(self, target_deg: float, lidar: Optional[LaserScan]):
        """
        Spin the bot by target_deg degrees.
        Positive = left (CCW), negative = right (CW).
        LiDAR is checked before each Twist publish — aborts if blocked.
        """
        target_rad  = math.radians(abs(target_deg))
        direction   = 1.0 if target_deg > 0 else -1.0
        speed       = ANGULAR_TURN if abs(target_deg) < 180 else ANGULAR_U_TURN

        # Determine which LiDAR sector to monitor
        if target_deg < 0:
            sector = LIDAR_SECTOR_RIGHT
        elif abs(target_deg) >= 180:
            sector = LIDAR_SECTOR_UTURN
        else:
            sector = LIDAR_SECTOR_LEFT

        # Time-based rotation — dt * angular_speed = angle covered
        # Not as accurate as an IMU but works well in simulation
        twist       = Twist()
        twist.angular.z = direction * speed

        elapsed    = 0.0
        dt         = 0.05   # 20 Hz control loop
        start_time = time.time()

        self.get_logger().info(
            f'Rotating {target_deg:+.0f}° at {speed:.2f} rad/s '
            f'(est. {target_rad/speed:.1f}s)'
        )

        while elapsed < target_rad / speed:
            # Refresh lidar snapshot
            with self._lidar_lock:
                current_lidar = self._lidar_msg

            # LiDAR safety check
            if current_lidar is not None:
                min_d = self._lidar_sector_min(current_lidar, sector[0], sector[1])
                if min_d < LIDAR_STOP_DIST:
                    self.get_logger().warn(
                        f'LiDAR abort! obstacle at {min_d:.2f}m in sector {sector} — stopping rotation'
                    )
                    self._publish_stop()
                    return

            self.cmd_pub.publish(twist)
            time.sleep(dt)
            elapsed = time.time() - start_time

        self._publish_stop()
        time.sleep(0.1)   # brief settle pause

    # ------------------------------------------------------------------
    # Action: switch colour mode
    # ------------------------------------------------------------------
    def _action_set_colour(self, colour: str):
        """Instantly switches active colour for blob following."""
        self.active_colour = colour
        # Also update the ROS parameter so ros2 param get reflects it
        self.set_parameters([
            rclpy.parameter.Parameter(
                'active_colour',
                rclpy.parameter.Parameter.Type.STRING,
                colour,
            )
        ])
        self.get_logger().info(f'Colour mode → {colour}')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _publish_stop(self):
        self.cmd_pub.publish(Twist())   # all-zero Twist = stop

    def _lidar_sector_min(
        self,
        scan: LaserScan,
        sector_start_deg: float,
        sector_end_deg: float,
    ) -> float:
        """
        Return the minimum range reading within a degree arc.
        sector_start/end are in degrees relative to forward (0° = straight ahead).
        Handles wrap-around (e.g. 150° to 210° crosses the 180° rear point).
        """
        angle_min = scan.angle_min        # radians
        angle_inc = scan.angle_increment  # radians per index
        ranges    = np.array(scan.ranges, dtype=np.float32)

        # Replace inf/nan with max range
        ranges = np.where(np.isfinite(ranges), ranges, scan.range_max)

        start_rad = math.radians(sector_start_deg)
        end_rad   = math.radians(sector_end_deg)

        # Convert to indices
        def angle_to_idx(a_rad):
            idx = int((a_rad - angle_min) / angle_inc)
            return max(0, min(len(ranges) - 1, idx))

        # Handle wrap-around (sector crosses ±180°)
        if sector_start_deg <= sector_end_deg:
            i0 = angle_to_idx(start_rad)
            i1 = angle_to_idx(end_rad)
            sector_ranges = ranges[i0:i1 + 1]
        else:
            # e.g. 150° to 210° → split into [150°, 180°] ∪ [-180°, -150°]
            i0a = angle_to_idx(start_rad)
            i1a = angle_to_idx(math.radians(180))
            i0b = angle_to_idx(math.radians(-180))
            i1b = angle_to_idx(end_rad - math.radians(360))
            sector_ranges = np.concatenate([ranges[i0a:i1a + 1], ranges[i0b:i1b + 1]])

        if sector_ranges.size == 0:
            return float('inf')

        return float(np.min(sector_ranges))

    # ======================================================================
    # Template matching
    # ======================================================================

    def _load_template(self, path: str) -> Tuple[Optional[np.ndarray], List[np.ndarray]]:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self.get_logger().warn(f'Template not found at {path} — tile counting disabled.')
            return None, []

        if len(img.shape) == 3 and img.shape[2] == 4:
            alpha   = img[:, :, 3:4].astype(np.float32) / 255.0
            rgb     = img[:, :, :3].astype(np.float32)
            white   = np.ones_like(rgb) * 255.0
            img_bgr = (rgb * alpha + white * (1 - alpha)).astype(np.uint8)
        else:
            img_bgr = img[:, :, :3] if len(img.shape) == 3 else img

        base_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = base_gray.shape
        scaled_cache = []
        for scale in TEMPLATE_SCALES:
            sw, sh = int(w * scale), int(h * scale)
            if sw >= 10 and sh >= 10:
                scaled_cache.append(cv2.resize(base_gray, (sw, sh)))

        self.get_logger().info(f'Template: {w}×{h}px, {len(scaled_cache)} scales')
        return base_gray, scaled_cache

    def _run_template_match(self, gray: np.ndarray) -> TemplateResult:
        if not self._template_scales_cache:
            return TemplateResult(on_tile=False, confidence=0.0)

        fh, fw  = gray.shape
        best    = (0.0, None, None)

        for tmpl in self._template_scales_cache:
            th, tw = tmpl.shape
            if th > fh or tw > fw:
                continue
            result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best[0]:
                best = (max_val, max_loc, (tw, th))

        conf, loc, size = best
        on_tile = conf >= TEMPLATE_MATCH_THRESHOLD
        return TemplateResult(
            on_tile=on_tile, confidence=conf,
            match_loc=loc if on_tile else None,
            match_size=size if on_tile else None,
        )

    # ======================================================================
    # AprilTag detection  (36h11 dictionary)
    # ======================================================================

    def _init_apriltag_detector(self):
        try:
            from pupil_apriltags import Detector
            self._apriltag_detector = Detector(
                families='tag36h11', nthreads=2, quad_decimate=1.0
            )
            self.get_logger().info('AprilTag: pupil_apriltags')
        except ImportError:
            self._apriltag_detector = None
            self.get_logger().warn('pupil_apriltags not found — ArUco fallback')

    def _detect_apriltag(self, gray: np.ndarray, debug: Optional[np.ndarray]) -> Optional[dict]:
        if self._apriltag_detector is not None:
            try:
                detections = self._apriltag_detector.detect(gray)
                if detections:
                    best = max(
                        detections,
                        key=lambda d: cv2.contourArea(d.corners.astype(np.float32)),
                    )
                    cx, cy = int(best.center[0]), int(best.center[1])
                    if debug is not None:
                        pts = best.corners.astype(int)
                        cv2.polylines(debug, [pts], True, (255, 0, 255), 2)
                        cv2.circle(debug, (cx, cy), 6, (255, 0, 255), -1)
                        action_name = {
                            0: 'TURN RIGHT', 1: 'TURN LEFT',
                            2: 'FOLLOW GREEN', 3: 'U-TURN', 4: 'FOLLOW ORANGE'
                        }.get(best.tag_id, f'TAG {best.tag_id}')
                        cv2.putText(debug, action_name,
                                    (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 0, 255), 2)
                    return {'id': best.tag_id, 'center': (cx, cy)}
            except Exception as e:
                self.get_logger().warn(f'AprilTag error: {e}')
            return None

        # ArUco fallback
        try:
            aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            aruco_params = cv2.aruco.DetectorParameters()
            det          = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            corners_list, ids, _ = det.detectMarkers(gray)
            if ids is None:
                return None
            best_idx     = int(np.argmax([cv2.contourArea(c) for c in corners_list]))
            best_id      = int(ids[best_idx][0])
            best_corners = corners_list[best_idx][0]
            cx = int(best_corners[:, 0].mean())
            cy = int(best_corners[:, 1].mean())
            if debug is not None:
                cv2.aruco.drawDetectedMarkers(debug, corners_list, ids)
                action_name = {
                    0: 'TURN RIGHT', 1: 'TURN LEFT',
                    2: 'FOLLOW GREEN', 3: 'U-TURN', 4: 'FOLLOW ORANGE'
                }.get(best_id, f'TAG {best_id}')
                cv2.putText(debug, action_name,
                            (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            return {'id': best_id, 'center': (cx, cy)}
        except Exception as e:
            self.get_logger().warn(f'ArUco error: {e}')
            return None

    # ======================================================================
    # Colour blob detection
    # ======================================================================

    def _detect_blob(
        self, hsv: np.ndarray, low: np.ndarray, high: np.ndarray,
        label: str, debug: Optional[np.ndarray], depth: Optional[np.ndarray] = None,
    ) -> Optional[ColourBlob]:

        mask = cv2.inRange(hsv, low, high)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        if area < MIN_BLOB_AREA:
            return None

        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        distance_m: Optional[float] = None
        if depth is not None and debug is not None:
            dh, dw = depth.shape[:2]
            sx = int(cx * dw / debug.shape[1])
            sy = int(cy * dh / debug.shape[0])
            sx = max(0, min(sx, dw - 1))
            sy = max(0, min(sy, dh - 1))
            patch = depth[max(0, sy-2):sy+3, max(0, sx-2):sx+3]
            valid = patch[np.isfinite(patch)]
            if valid.size > 0:
                distance_m = float(np.median(valid))

        if debug is not None:
            colour_bgr = (0, 220, 0) if label == 'GREEN' else (0, 140, 255)
            cv2.drawContours(debug, [largest], -1, colour_bgr, 2)
            cv2.circle(debug, (cx, cy), 9, colour_bgr, -1)
            dist_text = (f'{distance_m:.2f}m'
                         if distance_m and np.isfinite(distance_m)
                         else f'{area:.0f}px')
            cv2.putText(debug, f'{label} {dist_text}',
                        (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour_bgr, 2)

        return ColourBlob(centroid=(cx, cy), area=area, distance_m=distance_m)

    # ======================================================================
    # Blob → Twist
    # ======================================================================

    def _blobs_to_twist(
        self, green: Optional[ColourBlob], orange: Optional[ColourBlob], frame_h: int
    ) -> Twist:

        twist     = Twist()
        primary   = green  if self.active_colour == 'GREEN'  else orange
        secondary = orange if self.active_colour == 'GREEN'  else green
        threshold = frame_h * SIDE_THRESHOLD_RATIO

        if primary is None and secondary is None:
            twist.linear.x = LINEAR_SPEED_FWD * 0.5
            return twist
        if primary is None:
            twist.linear.x = LINEAR_SPEED_FWD * 0.4
            return twist
        if secondary is None:
            twist.linear.x = LINEAR_SPEED_FWD
            return twist

        dy = primary.centroid[1] - secondary.centroid[1]
        if abs(dy) < threshold:
            twist.linear.x = LINEAR_SPEED_FWD
        elif dy > 0:
            twist.linear.x = LINEAR_SPEED_FWD
        else:
            twist.linear.x = LINEAR_SPEED_BWD
        return twist

    def _apply_depth_guard(self, twist: Twist, depth: np.ndarray) -> Twist:
        h, w  = depth.shape[:2]
        strip = depth[int(h*0.4):int(h*0.7), int(w*0.4):int(w*0.6)]
        strip = np.where(np.isfinite(strip), strip, 10.0)
        if float(np.nanmin(strip)) < 0.3:
            return Twist()
        return twist

    # ======================================================================
    # Debug drawing
    # ======================================================================

    def _draw_tile_info(self, frame: np.ndarray, result: TemplateResult):
        h = frame.shape[0]
        if result.on_tile and result.match_loc and result.match_size:
            x, y   = result.match_loc
            mw, mh = result.match_size
            cv2.rectangle(frame, (x, y), (x+mw, y+mh), (255, 255, 0), 2)
            cv2.putText(frame, f'TILE {result.confidence:.2f}',
                        (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        status_col = (0, 255, 255) if self.tile_tracker.on_tile else (180, 180, 180)
        done_str   = '  ← DONE!' if self.tile_tracker.tile_done else ''
        on_str     = 'ON TILE' if self.tile_tracker.on_tile else 'between tiles'
        cv2.putText(frame,
                    f'Tiles: {self.tile_tracker.tiles_passed}  [{on_str}]{done_str}',
                    (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

    def _draw_overlay(self, frame: np.ndarray, label: str, twist: Twist):
        text = (f'{label} | lin={twist.linear.x:+.2f}  ang={twist.angular.z:+.2f}')
        col  = ((0,255,0) if twist.linear.x > 0 else
                (0,0,255) if twist.linear.x < 0 else (0,200,255))
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0,0,0), -1)
        cv2.putText(frame, text, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    def _draw_lidar_arcs(self, frame: np.ndarray, scan: LaserScan):
        """
        Draw a miniature LiDAR sweep indicator in the bottom-right corner.
        Shows obstacle proximity in each monitored sector.
        """
        h, w   = frame.shape[:2]
        cx, cy = w - 80, h - 80
        radius = 60

        sectors = {
            'R': (LIDAR_SECTOR_RIGHT,  (0, 0, 255)),
            'L': (LIDAR_SECTOR_LEFT,   (0, 255, 0)),
            'U': (LIDAR_SECTOR_UTURN,  (0, 165, 255)),
        }

        cv2.circle(frame, (cx, cy), radius, (60, 60, 60), 1)

        for label, (sector, colour) in sectors.items():
            min_d = self._lidar_sector_min(scan, sector[0], sector[1])
            # Brightness encodes distance — brighter = closer
            alpha = max(0.0, 1.0 - min_d / 2.0)
            if alpha > 0.1:
                # Draw arc wedge
                start_a = int(-sector[1])   # cv2 angles: clockwise, 0=right
                end_a   = int(-sector[0])
                cv2.ellipse(frame, (cx, cy), (radius-5, radius-5),
                            0, start_a, end_a, colour, 3)
            # Distance text
            angle_mid = math.radians((sector[0] + sector[1]) / 2.0)
            tx = cx + int((radius + 10) * math.sin(angle_mid))
            ty = cy - int((radius + 10) * math.cos(angle_mid))
            cv2.putText(frame, f'{min_d:.1f}m',
                        (tx - 15, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = ArtParkNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.get_logger().info('Safety stop. Shutting down.')
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()