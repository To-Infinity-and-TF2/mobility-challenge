#!/usr/bin/env python3
"""
ArtPark Arena Bot Navigation v2 — ROS2 Node (Simulation)
=========================================================
Subscribes : /r1_mini/camera/image_raw       (sensor_msgs/Image)
             /r1_mini/depth_cam/image_raw    (sensor_msgs/Image)
             /r1_mini/lidar                  (sensor_msgs/LaserScan)
Publishes  : /cmd_vel                        (geometry_msgs/Twist)

State machine
─────────────
APPROACH_MARKER
  • Triggers globally ONLY when the STRICTLY NEXT expected numerical ArUco is seen.
  • Drives towards the centroid of the marker while avoiding walls.
  • Registers ONLY when depth to marker is <= 0.2 meters (or LiDAR forces a stop against it). 
  • Once registered, executes the command and advances sequence.

EXPLORATION (Replaces STRAIGHT / SCANNING)
  • Default state when searching for the +1 ArUco.
  • Evaluates the front 120 degrees dynamically to find the least dense path.
  • Maintains a central position between left and right walls.

FOLLOW_GREEN / FOLLOW_ORANGE
  • Triggered by specific ArUco commands. Uses depth to stay behind the correct colour.
  • Falls back to EXPLORATION if colours are lost.

LiDAR Post-Processing & Avoidance
  • Dynamic Centering: Adjusts heading to maximize distance from left/right walls.
  • Hard Obstacle Avoidance: Prevents crashing into walls.
  • ArUco Priority: Wall recovery/stop is completely overridden if the +1 ArUco is reached.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

import cv2
import numpy as np
import time
import os
import math
from typing import Optional, Tuple
from dataclasses import dataclass

# ── Topics ────────────────────────────────────────────────────────────────
RGB_TOPIC   = '/r1_mini/camera/image_raw'
DEPTH_TOPIC = '/r1_mini/depth_cam/image_raw'
LIDAR_TOPIC = '/r1_mini/lidar'

# ── Velocity ──────────────────────────────────────────────────────────────
LINEAR_SPEED_FWD  =  0.6
ANGULAR_TURN      =  0.4
ANGULAR_U_TURN    =  0.8
ANGULAR_SEARCH    =  0.3    # slow spin when only one colour visible

TURN_90_DURATION  = 1.5
TURN_180_DURATION = 3.0

# ── Timeouts ──────────────────────────────────────────────────────────────
SINGLE_COLOUR_TIMEOUT = 5.0   # Seconds to wait before falling back to EXPLORATION
APPROACH_TIMEOUT      = 3.0   # Seconds to keep searching if marker is lost during approach

# ── ArUco ─────────────────────────────────────────────────────────────────
MIN_MARKER_AREA   = 200
UPSCALE_FACTOR    = 2.0
MARKER_COOLDOWN_S = 5.0
MARKER_APPROACH_DIST_THRESHOLD = 0.20   # Register when <= 0.2 meters away from marker
MARKER_APPROACH_AREA_FALLBACK  = 30000  # Fallback registration if depth camera reads N/A

# ── HSV ───────────────────────────────────────────────────────────────────
GREEN_LOW   = np.array([40,  80,  80],  dtype=np.uint8)
GREEN_HIGH  = np.array([85, 255, 255],  dtype=np.uint8)
ORANGE_LOW  = np.array([5,  120, 120],  dtype=np.uint8)
ORANGE_HIGH = np.array([22, 255, 255],  dtype=np.uint8)
MIN_BLOB_AREA = 500

# ── LiDAR ─────────────────────────────────────────────────────────────────
LIDAR_STOP_DIST      = 0.30
LIDAR_WARN_FRONT     = 0.50
LIDAR_CENTERING_GAIN = 0.8
LIDAR_MAX_CORRECTION = 0.6
LIDAR_MIN_VALID      = 0.05

# ── Depth comparison ──────────────────────────────────────────────────────
DEPTH_PATCH_R = 8    # px radius for median depth sample at centroid

# ── Logging ───────────────────────────────────────────────────────────────
LOG_PATH = os.path.expanduser('~/aruco_log.txt')

# ── States ────────────────────────────────────────────────────────────────
S_EXPLORATION        = 'EXPLORATION'
S_TURNING            = 'TURNING'
S_FOLLOW_GREEN       = 'FOLLOW_GREEN'
S_FOLLOW_ORANGE      = 'FOLLOW_ORANGE'
S_GREEN_ONLY         = 'GREEN_ONLY'
S_ORANGE_ONLY        = 'ORANGE_ONLY'
S_APPROACH_MARKER    = 'APPROACH_MARKER'


@dataclass
class Blob:
    cx: int
    cy: int
    area: float

@dataclass
class MarkerEvent:
    aruco_id:  int
    log_id:    int
    timestamp: float
    command:   str


# ─────────────────────────────────────────────────────────────────────────────
class ArtParkNavNode(Node):

    def __init__(self):
        super().__init__('artpark_nav_node')

        self.declare_parameter('show_debug', True)
        self.show_debug = self.get_parameter('show_debug').value

        self.bridge      = CvBridge()
        self.depth_frame: Optional[np.ndarray] = None
        self._scan:       Optional[LaserScan]  = None

        # ── State ──────────────────────────────────────────────────────
        self._state = S_EXPLORATION

        # timed turn
        self._turn_end_time    = 0.0
        self._turn_angular_z   = 0.0
        self._turn_next_state  = S_EXPLORATION

        # ArUco Sequence & Approach
        self._next_expected_aruco: int = 0
        self._pre_approach_state       = S_EXPLORATION
        self._target_aruco_id          = None
        self._last_seen_target_time    = 0.0
        
        # Cross-function tracking for Lidar overrides
        self._last_tag                 = None
        self._last_process_time        = 0.0

        # single-colour timer (GREEN_ONLY / ORANGE_ONLY)
        self._single_colour_since = time.time()

        # Logging
        self._marker_last_seen: dict[int, float] = {}
        self._marker_log: list[MarkerEvent]      = []

        # ArUco detector
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

        self.get_logger().info(
            f"ArtPark Nav v2 ready  state={self._state}\n"
            f"  RGB={RGB_TOPIC}  Depth={DEPTH_TOPIC}  LiDAR={LIDAR_TOPIC}"
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
        tag    = self._aruco(gray, dbg)  # Will ONLY return the +1 marker
        
        # Save tag for potential LiDAR overrides
        self._last_tag = tag

        twist = Twist()

        # ── GLOBAL INTERCEPT: See exact expected marker -> Lock On ────────
        if tag is not None and self._state != S_TURNING:
            tid = tag['id']
            # Safety check, although _aruco() already filters this
            if tid == self._next_expected_aruco and self._marker_ready(tid, now):
                if self._state != S_APPROACH_MARKER:
                    self.get_logger().info(f"Target Acquired: +1 ArUco [{tid}], locking on...")
                    self._pre_approach_state = self._state
                    self._state              = S_APPROACH_MARKER
                    self._target_aruco_id    = tid
                self._last_seen_target_time = now

        # ── APPROACH_MARKER ───────────────────────────────────────────
        if self._state == S_APPROACH_MARKER:
            twist = self._handle_approach(tag, frame.shape[1], now, dbg)
            return twist

        # ── TURNING ───────────────────────────────────────────────────
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

        # ── EXPLORATION ───────────────────────────────────────────────
        if self._state == S_EXPLORATION:
            twist = self._handle_exploration(now, dbg)
            return twist

        # ── FOLLOW_GREEN ──────────────────────────────────────────────
        if self._state == S_FOLLOW_GREEN:
            twist = self._handle_follow(
                primary=green, secondary=orange,
                primary_name='GREEN', secondary_name='ORANGE',
                only_state=S_GREEN_ONLY, now=now
            )
            self._overlay(dbg, 'FOLLOW_GREEN', twist)
            self._show(dbg)
            return twist

        # ── FOLLOW_ORANGE ─────────────────────────────────────────────
        if self._state == S_FOLLOW_ORANGE:
            twist = self._handle_follow(
                primary=orange, secondary=green,
                primary_name='ORANGE', secondary_name='GREEN',
                only_state=S_ORANGE_ONLY, now=now
            )
            self._overlay(dbg, 'FOLLOW_ORANGE', twist)
            self._show(dbg)
            return twist

        # ── GREEN_ONLY ────────────────────────────────────────────────
        if self._state == S_GREEN_ONLY:
            twist = self._handle_only(
                present=green, other=orange,
                follow_state=S_FOLLOW_GREEN, now=now
            )
            self._overlay(dbg, 'GREEN_ONLY', twist)
            self._show(dbg)
            return twist

        # ── ORANGE_ONLY ───────────────────────────────────────────────
        if self._state == S_ORANGE_ONLY:
            twist = self._handle_only(
                present=orange, other=green,
                follow_state=S_FOLLOW_ORANGE, now=now
            )
            self._overlay(dbg, 'ORANGE_ONLY', twist)
            self._show(dbg)
            return twist

        self._show(dbg)
        return twist

    # ── State handlers ────────────────────────────────────────────────────

    def _register_marker(self, tid: int, area: float, depth: float, now: float) -> Twist:
        """Centralized method for executing an ArUco sequence marker."""
        depth_str = f"{depth:.2f}m" if depth is not None else "N/A"
        self.get_logger().info(f"ArUco {tid} registered! (Depth: {depth_str}, Area: {area:.0f})")
        
        self._marker_last_seen[tid] = now
        self._log_marker(tid, now)
        self._advance_sequence()
        
        twist = Twist()
        # Command Execution - Default back to EXPLORATION instead of STRAIGHT
        if tid == 0:
            self._start_turn(-ANGULAR_TURN, TURN_90_DURATION, S_EXPLORATION)
            twist.angular.z = -ANGULAR_TURN
        elif tid == 1:
            self._start_turn(ANGULAR_TURN, TURN_90_DURATION, S_EXPLORATION)
            twist.angular.z = ANGULAR_TURN
        elif tid == 2:
            self._start_turn(-ANGULAR_TURN, TURN_90_DURATION, S_FOLLOW_GREEN)
            twist.angular.z = -ANGULAR_TURN
        elif tid == 3:
            self._start_turn(ANGULAR_U_TURN, TURN_180_DURATION, S_FOLLOW_GREEN)
            twist.angular.z = ANGULAR_U_TURN
        elif tid == 4:
            self._start_turn(ANGULAR_TURN, TURN_90_DURATION, S_FOLLOW_ORANGE)
            twist.angular.z = ANGULAR_TURN
            
        return twist

    def _handle_approach(self, tag, frame_width, now, dbg) -> Twist:
        twist = Twist()
        
        if tag is not None and tag['id'] == self._target_aruco_id:
            self._last_seen_target_time = now
            area = tag['area']
            cx, cy = tag['center']
            
            depth = self._depth_at(cx, cy)
            
            # Check if we reached 0.2 meters (or use area fallback)
            is_close_enough = False
            if depth is not None and depth > 0.0:
                if depth <= MARKER_APPROACH_DIST_THRESHOLD:
                    is_close_enough = True
            elif area >= MARKER_APPROACH_AREA_FALLBACK:
                is_close_enough = True

            if is_close_enough:
                # ── REGISTERED ──
                return self._register_marker(self._target_aruco_id, area, depth, now)
            else:
                # ── STILL APPROACHING ──
                # Drive forward while steering to lock the ArUco in the exact center of the frame
                err = (frame_width / 2) - cx
                twist.angular.z = 0.002 * err  # Proportional steering
                twist.linear.x  = LINEAR_SPEED_FWD * 0.7  
                depth_disp = f"{depth:.2f}m" if depth else "N/A"
                self._overlay(dbg, f"APPROACH {self._target_aruco_id} (D:{depth_disp} A:{area:.0f})", twist)
                self._show(dbg)
                return twist
        else:
            # ── TARGET LOST DURING APPROACH ──
            lost_duration = now - self._last_seen_target_time
            if lost_duration > APPROACH_TIMEOUT:
                self.get_logger().warn(f"Lost target marker {self._target_aruco_id}, reverting to EXPLORATION.")
                self._state = S_EXPLORATION
            else:
                twist.linear.x = LINEAR_SPEED_FWD * 0.4
                
            self._overlay(dbg, f"APPROACH LOST (Timeout in {APPROACH_TIMEOUT - lost_duration:.1f}s)", twist)
            self._show(dbg)
            return twist

    def _handle_exploration(self, now: float, dbg) -> Twist:
        """
        Dynamically drives forward while checking the front 120-degree sector 
        to bias its steering towards the least dense/most open path.
        """
        twist = Twist()
        twist.linear.x = LINEAR_SPEED_FWD
        
        # Scan sectors: Left (-45), Center (0), Right (+45) to find the most open path
        best_angle = 0.0
        max_dist = -1.0
        
        for ang in [-45, 0, 45]:
            dist = self._range_zone(ang - 15, ang + 15, use_avg=True)
            if dist > max_dist:
                max_dist = dist
                best_angle = ang
                
        # Apply a mild steering bias towards the open path. 
        # The dynamic centering in _lidar_avoidance will refine this.
        twist.angular.z = math.radians(best_angle) * 0.5

        self._overlay(dbg, f'EXPLORATION (Heading Bias: {best_angle}°)', twist)
        self._show(dbg)
        return twist

    def _handle_follow(self, primary: Optional[Blob], secondary: Optional[Blob],
                       primary_name: str, secondary_name: str,
                       only_state: str, now: float) -> Twist:
        twist = Twist()
        if primary is None and secondary is None:
            twist.angular.z = ANGULAR_SEARCH
            return twist

        if secondary is None:
            self.get_logger().info(f"Only {primary_name} visible → {only_state}")
            self._state = only_state
            self._single_colour_since = now
            twist.linear.x = LINEAR_SPEED_FWD
            return twist

        if primary is None:
            twist.angular.z = ANGULAR_SEARCH
            return twist

        primary_d   = self._depth_at(primary.cx,   primary.cy)
        secondary_d = self._depth_at(secondary.cx, secondary.cy)

        if primary_d is None or secondary_d is None:
            primary_d   = float(primary.cy)
            secondary_d = float(secondary.cy)

        if primary_d < secondary_d:
            twist.linear.x = LINEAR_SPEED_FWD
        else:
            self.get_logger().info(f"{secondary_name} is ahead of {primary_name} → U-turn")
            self._start_turn(ANGULAR_U_TURN, TURN_180_DURATION,
                             S_FOLLOW_GREEN if primary_name == 'GREEN' else S_FOLLOW_ORANGE)
            twist.angular.z = ANGULAR_U_TURN

        return twist

    def _handle_only(self, present: Optional[Blob], other: Optional[Blob],
                     follow_state: str, now: float) -> Twist:
        twist = Twist()
        if other is not None:
            self.get_logger().info(f"Second colour back in frame → {follow_state}")
            self._state = follow_state
            self._single_colour_since = now
            twist.linear.x = LINEAR_SPEED_FWD
            return twist

        elapsed = now - self._single_colour_since
        if elapsed >= SINGLE_COLOUR_TIMEOUT:
            self.get_logger().warn(f"Single colour timeout ({elapsed:.1f}s) → Returning to EXPLORATION")
            self._state = S_EXPLORATION
            return Twist() # Stop for a tick

        twist.linear.x = LINEAR_SPEED_FWD
        return twist

    # ── Timed Moves ───────────────────────────────────────────────────────

    def _start_turn(self, angular_z: float, duration: float, next_state: str):
        self._state           = S_TURNING
        self._turn_end_time   = time.time() + duration
        self._turn_angular_z  = angular_z
        self._turn_next_state = next_state
        direction = 'RIGHT' if angular_z < 0 else 'LEFT' if angular_z > 0 else 'U'
        self.get_logger().info(f"Turn {direction} {duration:.2f}s → {next_state}")

    # ── ArUco ─────────────────────────────────────────────────────────────

    def _aruco(self, gray: np.ndarray, dbg) -> Optional[dict]:
        """
        Detects markers but strictly ignores ALL markers except the +1 expected ID.
        """
        s    = UPSCALE_FACTOR
        big  = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
        corners_list, ids, _ = self._det.detectMarkers(big)
        
        if ids is None or len(ids) == 0:
            return None
            
        target_idx = None
        # Loop through found IDs to strictly filter for the +1 target
        for i, tid in enumerate(ids):
            if tid[0] == self._next_expected_aruco:
                target_idx = i
                break
                
        # If the +1 expected marker is not in the frame, return None (ignoring all others)
        if target_idx is None:
            return None
            
        corners_orig = [c / s for c in corners_list]
        area = cv2.contourArea(corners_orig[target_idx].astype(np.float32))
        
        if area < MIN_MARKER_AREA:
            return None
            
        best_id = int(ids[target_idx][0])
        bc      = corners_orig[target_idx][0]
        cx, cy  = int(bc[:, 0].mean()), int(bc[:, 1].mean())
        
        if dbg is not None:
            cv2.aruco.drawDetectedMarkers(
                dbg,
                [corners_orig[target_idx].reshape(1, 4, 2).astype(np.float32)], 
                np.array([[best_id]])
            )
            cv2.putText(dbg, f"TARGET +1 [{best_id}] {self._cmd_name(best_id)}",
                        (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                        
        return {'id': best_id, 'center': (cx, cy), 'area': area}

    def _marker_ready(self, tid: int, now: float) -> bool:
        return (now - self._marker_last_seen.get(tid, 0)) > MARKER_COOLDOWN_S

    def _advance_sequence(self):
        self._next_expected_aruco += 1
        self.get_logger().info(f'Sequence advanced -- Searching strictly for NEXT expected: {self._next_expected_aruco}')

    # ── Colour blob ───────────────────────────────────────────────────────

    def _blob(self, hsv, lo, hi, label: str, dbg) -> Optional[Blob]:
        mask = cv2.inRange(hsv, lo, hi)
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not cnts: return None
        largest = max(cnts, key=cv2.contourArea)
        area    = cv2.contourArea(largest)
        if area < MIN_BLOB_AREA: return None
        
        M = cv2.moments(largest)
        if M['m00'] == 0: return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        if dbg is not None:
            col = (0, 220, 0) if label == 'GREEN' else (0, 140, 255)
            cv2.drawContours(dbg, [largest], -1, col, 2)
            cv2.circle(dbg, (cx, cy), 9, col, -1)
            cv2.putText(dbg, f"{label} {area:.0f}px",
                        (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        return Blob(cx=cx, cy=cy, area=area)

    # ── Depth sampling ────────────────────────────────────────────────────

    def _depth_at(self, cx: int, cy: int) -> Optional[float]:
        if self.depth_frame is None: return None
        dh, dw = self.depth_frame.shape[:2]
        r  = DEPTH_PATCH_R
        x0, x1 = max(cx - r, 0), min(cx + r, dw)
        y0, y1 = max(cy - r, 0), min(cy + r, dh)
        patch = self.depth_frame[y0:y1, x0:x1]
        valid = patch[(patch > LIDAR_MIN_VALID) & np.isfinite(patch)]
        if valid.size == 0: return None
        return float(np.median(valid))

    # ── LiDAR avoidance (post-processing layer) ───────────────────────────

    def _range_zone(self, lo_deg: float, hi_deg: float, use_avg: bool = False) -> float:
        if self._scan is None: return 999.0
        scan = self._scan
        def to_idx(a):
            i = int(round((a - scan.angle_min) / scan.angle_increment))
            return max(0, min(i, len(scan.ranges) - 1))
            
        i0 = to_idx(math.radians(lo_deg))
        i1 = to_idx(math.radians(hi_deg))
        if i0 > i1: i0, i1 = i1, i0
        
        vals = []
        for r in scan.ranges[i0:i1+1]:
            if math.isfinite(r) and r > LIDAR_MIN_VALID:
                vals.append(min(r, scan.range_max))
            else:
                vals.append(scan.range_max) # Treat inf/out-of-range as fully open

        if not vals: return 999.0
        
        if use_avg:
            return float(np.mean(vals))
        return float(min(vals))

    def _find_least_dense_direction(self) -> float:
        """Evaluates the instantaneous 360 LiDAR local map to find the most open heading."""
        if self._scan is None: return 1.0
        max_avg = -1.0
        best_angle = 0.0
        
        # Check every 15 degrees around the robot to find the highest average clearance
        for angle_deg in range(-180, 180, 15):
            avg_dist = self._range_zone(angle_deg - 15, angle_deg + 15, use_avg=True)
            if avg_dist > max_avg:
                max_avg = avg_dist
                best_angle = angle_deg
                
        return float(best_angle)

    def _lidar_avoidance(self, twist: Twist) -> Twist:
        # If we are in the middle of a deliberate turn, ignore walls so the turn completes
        if self._scan is None or twist.linear.x <= 0.0: return twist
        
        # Dynamically lower the physical safety boundary when actively approaching an ArUco
        effective_stop_dist = 0.15 if self._state == S_APPROACH_MARKER else LIDAR_STOP_DIST
        
        front = self._range_zone(-20,  20)

        # ── WALL RECOVERY & ARUCO OVERRIDE ──
        if front < effective_stop_dist:
            tag = self._last_tag
            now = self._last_process_time
            
            # 1. Absolute Priority: If blocked by wall but we see the target ArUco, force register it!
            if self._state == S_APPROACH_MARKER and tag is not None and tag['id'] == self._target_aruco_id:
                depth = self._depth_at(tag['center'][0], tag['center'][1])
                self.get_logger().warn(f"LIDAR STOP OVERRIDDEN! ArUco {tag['id']} blocking at {front:.2f}m. Forcing registration.")
                return self._register_marker(tag['id'], tag['area'], depth, now)

            # 2. Wall Hit Recovery: Instant local map evaluation
            if self._state != S_TURNING:
                best_angle = self._find_least_dense_direction()
                if best_angle == 0.0: best_angle = 180.0  # Fallback if entirely surrounded
                
                # Calculate required duration to hit that exact angle
                turn_rad = math.radians(best_angle)
                duration = abs(turn_rad) / ANGULAR_TURN
                turn_vel = ANGULAR_TURN if best_angle > 0 else -ANGULAR_TURN
                
                self.get_logger().warn(f"LIDAR HARD STOP front={front:.2f}m -> CALCULATED RECOVERY Turn {best_angle:.0f}°")
                self._start_turn(turn_vel, duration, S_EXPLORATION)
                
            # Stop forward motion while the turn is scheduled/starting
            twist.linear.x = 0.0
            return twist

        # ── DYNAMIC HEADING / CENTERING (Max distance from walls) ──
        # Check the sides (e.g. 20 to 90 degrees left and right)
        left_dist  = self._range_zone(20,  90, use_avg=True)
        right_dist = self._range_zone(-90, -20, use_avg=True)

        # Apply centering to maintain a safe path down the middle of a corridor
        if left_dist < 1.5 or right_dist < 1.5:
            l_capped = min(left_dist, 1.5)
            r_capped = min(right_dist, 1.5)
            
            # If left is closer (e.g. 0.5) and right is open (1.5), error is negative.
            # Negative angular.z turns the bot right, away from the left wall.
            centering_err = l_capped - r_capped
            correction = LIDAR_CENTERING_GAIN * centering_err
            
            # Cap maximum correction to prevent wild wobbling
            correction = float(np.clip(correction, -LIDAR_MAX_CORRECTION, LIDAR_MAX_CORRECTION))
            twist.angular.z += correction

        # Hard front-slowdown
        if front < LIDAR_WARN_FRONT:
            slow = (front - effective_stop_dist) / (LIDAR_WARN_FRONT - effective_stop_dist)
            twist.linear.x = max(0.0, twist.linear.x * slow)

        return twist

    # ── Helpers ───────────────────────────────────────────────────────────

    def _cmd_name(self, tid: int) -> str:
        return {0:'TURN_R', 1:'TURN_L', 2:'R->GREEN',
                3:'U->GREEN', 4:'L->ORANGE'}.get(tid, '?')

    def _get_log_id(self, tid: int) -> int:
        return {0:2, 1:1, 2:3, 3:4, 4:5}.get(tid, 99)

    def _log_marker(self, tid: int, ts: float):
        cmd    = self._cmd_name(tid)
        log_id = self._get_log_id(tid)
        self._marker_log.append(MarkerEvent(tid, log_id, ts, cmd))
        self._marker_log.sort(key=lambda e: e.log_id)
        ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))
        self.get_logger().info(f"[ARUCO] ID={tid} logID={log_id} cmd={cmd} @{ts_str}")
        try:
            with open(LOG_PATH, 'w') as f:
                f.write("ArUco Marker Log — Sorted by Log ID\n")
                f.write("=" * 60 + "\n")
                for e in self._marker_log:
                    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e.timestamp))
                    f.write(f"Log ID: {e.log_id:>2}  |  ArUco ID: {e.aruco_id:>2}  "
                            f"|  {e.command:<16}  |  {t}\n")
        except Exception as ex:
            self.get_logger().warn(f"Log write failed: {ex}")

    def _overlay(self, dbg, label: str, twist: Twist):
        if dbg is None: return
        f  = self._range_zone(-20,  20) if self._scan else 0
        l  = self._range_zone( 20,  90, use_avg=True) if self._scan else 0
        r  = self._range_zone(-90, -20, use_avg=True) if self._scan else 0
        text = (f"{label} | lin={twist.linear.x:+.2f} ang={twist.angular.z:+.2f}"
                f" | LiDAR F:{f:.2f} L:{l:.2f} R:{r:.2f}")
        col = (0,255,0) if twist.linear.x > 0 else (0,0,255) if twist.linear.x < 0 else (0,200,255)
        cv2.rectangle(dbg, (0,0), (dbg.shape[1], 42), (0,0,0), -1)
        cv2.putText(dbg, text, (8,29), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    def _show(self, dbg):
        if dbg is not None:
            cv2.imshow("ArtPark Nav v2", dbg)


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