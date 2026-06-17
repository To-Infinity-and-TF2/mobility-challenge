#!/usr/bin/env python3
"""
ArtPark Arena Bot Navigation — ROS2 Node (Simulation) — Segfault-safe
======================================================================
Subscribes : /r1_mini/camera/image_raw      (sensor_msgs/Image)  ← RGB
             /r1_mini/depth_cam/image_raw   (sensor_msgs/Image)  ← Depth
Publishes  : /cmd_vel                       (geometry_msgs/Twist)

Segfault fixes applied:
  1. AprilTag Detector instantiated ONCE at __init__, not every frame
  2. cv2.imshow / cv2.waitKey moved to a dedicated Timer callback that
     runs on the main ROS thread — never called from inside a subscriber cb
  3. depth_frame guarded with a threading.Lock to avoid torn reads
  4. CvBridge instantiated once and reused
  5. Graceful OpenCV window cleanup on shutdown

Run:
    python3 artpark_navigation_node.py
"""

import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from dataclasses import dataclass
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# SIMULATION CAMERA TOPICS
# ---------------------------------------------------------------------------
RGB_TOPIC   = '/r1_mini/camera/image_raw'
DEPTH_TOPIC = '/r1_mini/depth_cam/image_raw'

# ---------------------------------------------------------------------------
# VELOCITY CONSTANTS
# ---------------------------------------------------------------------------
LINEAR_SPEED_FWD  =  0.2
LINEAR_SPEED_BWD  = -0.15
ANGULAR_TURN      =  0.4
ANGULAR_U_TURN    =  0.8

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
# AprilTag ID → (linear.x, angular.z)
# ---------------------------------------------------------------------------
TAG_TWIST_MAP = {
    1: (LINEAR_SPEED_FWD, 0.0),
    2: (0.0,  ANGULAR_U_TURN),
    3: (0.0, -ANGULAR_TURN),
    4: (0.0,  ANGULAR_U_TURN),
    5: (LINEAR_SPEED_FWD, 0.0),
}


@dataclass
class ColourBlob:
    centroid: Tuple[int, int]
    area: float


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------
class ArtParkNavNode(Node):

    def __init__(self):
        super().__init__('artpark_nav_node')

        # ── Parameters ────────────────────────────────────────────────
        self.declare_parameter('active_colour', 'GREEN')
        self.declare_parameter('show_debug',    True)

        self.active_colour = self.get_parameter('active_colour').value.upper()
        self.show_debug    = self.get_parameter('show_debug').value

        # ── Shared state (written by callbacks, read by timer) ─────────
        self._depth_lock  = threading.Lock()
        self._depth_frame: Optional[np.ndarray] = None

        self._debug_lock  = threading.Lock()
        self._debug_frame: Optional[np.ndarray] = None   # for display only

        # ── CvBridge — ONE instance, never re-created ──────────────────
        self.bridge = CvBridge()

        # ── AprilTag detector — ONE instance, never re-created ─────────
        #    Re-creating it every frame is the #1 cause of segfaults
        self._apriltag_detector = None
        self._init_apriltag_detector()

        # ── Publisher ──────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Subscribers ────────────────────────────────────────────────
        self.create_subscription(Image, RGB_TOPIC,   self._rgb_cb,   10)
        self.create_subscription(Image, DEPTH_TOPIC, self._depth_cb, 10)

        # ── Debug display timer — runs on main thread, 30 Hz ───────────
        #    cv2.imshow MUST NOT be called from a subscriber callback
        if self.show_debug:
            self.create_timer(1.0 / 30.0, self._display_timer_cb)

        self.get_logger().info(
            f'\nArtPark Nav Node started'
            f'\n  RGB  : {RGB_TOPIC}'
            f'\n  Depth: {DEPTH_TOPIC}'
            f'\n  Mode : {self.active_colour}'
        )

    # ------------------------------------------------------------------
    # AprilTag detector init (called once)
    # ------------------------------------------------------------------
    def _init_apriltag_detector(self):
        try:
            from pupil_apriltags import Detector
            self._apriltag_detector = Detector(
                families='tag36h11',
                nthreads=2,
                quad_decimate=1.0,
            )
            self.get_logger().info('AprilTag detector: pupil_apriltags')
        except ImportError:
            # Will fall back to ArUco inside _detect_apriltag
            self._apriltag_detector = None
            self.get_logger().warn(
                'pupil_apriltags not found — using OpenCV ArUco fallback'
            )

    # ------------------------------------------------------------------
    # Depth callback — cache with lock
    # ------------------------------------------------------------------
    def _depth_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            with self._depth_lock:
                self._depth_frame = frame
        except Exception as e:
            self.get_logger().warn(f'Depth decode error: {e}')

    # ------------------------------------------------------------------
    # RGB callback — full pipeline, publish Twist
    # ------------------------------------------------------------------
    def _rgb_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'RGB decode error: {e}')
            return

        # Refresh param (supports runtime changes)
        self.active_colour = self.get_parameter('active_colour').value.upper()

        # Grab depth snapshot under lock
        with self._depth_lock:
            depth_snapshot = (
                self._depth_frame.copy() if self._depth_frame is not None else None
            )

        twist, debug_frame = self._process_frame(frame, depth_snapshot)
        self.cmd_pub.publish(twist)

        # Hand the debug frame to the display timer — don't imshow here
        if self.show_debug and debug_frame is not None:
            with self._debug_lock:
                self._debug_frame = debug_frame

    # ------------------------------------------------------------------
    # Display timer — ONLY place cv2.imshow is called
    # ------------------------------------------------------------------
    def _display_timer_cb(self):
        with self._debug_lock:
            frame = self._debug_frame
        if frame is not None:
            cv2.imshow('ArtPark Nav [sim]', frame)
            cv2.waitKey(1)   # must be called from same thread as imshow

    # ------------------------------------------------------------------
    # Full per-frame pipeline
    # ------------------------------------------------------------------
    def _process_frame(
        self,
        frame: np.ndarray,
        depth: Optional[np.ndarray],
    ) -> Tuple[Twist, Optional[np.ndarray]]:

        debug = frame.copy() if self.show_debug else None
        h, _  = frame.shape[:2]

        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. AprilTag — highest priority
        tag = self._detect_apriltag(gray, debug)
        if tag is not None:
            twist = self._tag_to_twist(tag['id'])
            self.get_logger().info(
                f"[TAG {tag['id']}] lin={twist.linear.x:.2f} ang={twist.angular.z:.2f}"
            )
            if debug is not None:
                self._draw_overlay(debug, f"APRILTAG {tag['id']}", twist)
            return twist, debug

        # 2. Colour blobs
        green_blob  = self._detect_blob(hsv, GREEN_HSV_LOW,  GREEN_HSV_HIGH,  'GREEN',  debug)
        orange_blob = self._detect_blob(hsv, ORANGE_HSV_LOW, ORANGE_HSV_HIGH, 'ORANGE', debug)

        # 3. Twist decision
        twist = self._blobs_to_twist(green_blob, orange_blob, h)

        # 4. Depth obstacle guard
        if depth is not None and twist.linear.x > 0:
            twist = self._apply_depth_guard(twist, depth)

        if debug is not None:
            self._draw_overlay(debug, f'COLOUR:{self.active_colour}', twist)

        return twist, debug

    # ------------------------------------------------------------------
    # Colour blob detection
    # ------------------------------------------------------------------
    def _detect_blob(
        self,
        hsv:   np.ndarray,
        low:   np.ndarray,
        high:  np.ndarray,
        label: str,
        debug: Optional[np.ndarray],
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

        if debug is not None:
            colour_bgr = (0, 220, 0) if label == 'GREEN' else (0, 140, 255)
            cv2.drawContours(debug, [largest], -1, colour_bgr, 2)
            cv2.circle(debug, (cx, cy), 9, colour_bgr, -1)
            cv2.putText(debug, f'{label} {area:.0f}px',
                        (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour_bgr, 2)

        return ColourBlob(centroid=(cx, cy), area=area)

    # ------------------------------------------------------------------
    # Blob geometry → Twist
    # ------------------------------------------------------------------
    def _blobs_to_twist(
        self,
        green:   Optional[ColourBlob],
        orange:  Optional[ColourBlob],
        frame_h: int,
    ) -> Twist:

        twist     = Twist()
        primary   = green  if self.active_colour == 'GREEN'  else orange
        secondary = orange if self.active_colour == 'GREEN'  else green
        threshold = frame_h * SIDE_THRESHOLD_RATIO

        if primary is None and secondary is None:
            twist.linear.x = LINEAR_SPEED_FWD * 0.5
            self.get_logger().warn('No blobs — creeping forward')
            return twist

        if primary is None:
            twist.linear.x = LINEAR_SPEED_FWD * 0.4
            self.get_logger().warn(f'Lost {self.active_colour} — creeping')
            return twist

        if secondary is None:
            twist.linear.x = LINEAR_SPEED_FWD
            return twist

        dy = primary.centroid[1] - secondary.centroid[1]

        if abs(dy) < threshold:
            twist.linear.x = LINEAR_SPEED_FWD
            self.get_logger().info('Side-by-side → STRAIGHT')
        elif dy > 0:
            twist.linear.x = LINEAR_SPEED_FWD
            self.get_logger().info(f'{self.active_colour} near-side → FORWARD')
        else:
            twist.linear.x = LINEAR_SPEED_BWD
            self.get_logger().warn(f'{self.active_colour} far-side → BACKWARD')

        return twist

    # ------------------------------------------------------------------
    # Depth obstacle guard (separated from blob logic)
    # ------------------------------------------------------------------
    def _apply_depth_guard(self, twist: Twist, depth: np.ndarray) -> Twist:
        h, w  = depth.shape[:2]
        strip = depth[int(h * 0.4):int(h * 0.7), int(w * 0.4):int(w * 0.6)]
        strip = np.where(np.isfinite(strip), strip, 10.0)
        min_d = float(np.nanmin(strip))

        if min_d < 0.3:
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            self.get_logger().warn(f'Obstacle at {min_d:.2f} m — STOP')

        return twist

    # ------------------------------------------------------------------
    # AprilTag detection — uses the pre-created detector
    # ------------------------------------------------------------------
    def _detect_apriltag(
        self,
        gray:  np.ndarray,
        debug: Optional[np.ndarray],
    ) -> Optional[dict]:

        # ── pupil_apriltags path ───────────────────────────────────────
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
                        cv2.putText(debug, f'TAG {best.tag_id}',
                                    (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255, 0, 255), 2)
                    return {'id': best.tag_id, 'center': (cx, cy)}
            except Exception as e:
                self.get_logger().warn(f'AprilTag detect error: {e}')
            return None

        # ── OpenCV ArUco fallback ──────────────────────────────────────
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
                cv2.putText(debug, f'TAG {best_id}',
                            (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 0, 255), 2)

            return {'id': best_id, 'center': (cx, cy)}
        except Exception as e:
            self.get_logger().warn(f'ArUco detect error: {e}')
            return None

    # ------------------------------------------------------------------
    # AprilTag ID → Twist
    # ------------------------------------------------------------------
    def _tag_to_twist(self, tag_id: int) -> Twist:
        twist = Twist()
        lin, ang = TAG_TWIST_MAP.get(tag_id, (LINEAR_SPEED_FWD, 0.0))
        twist.linear.x  = lin
        twist.angular.z = ang
        return twist

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------
    def _draw_overlay(self, frame: np.ndarray, label: str, twist: Twist):
        text = (f'{label} | lin={twist.linear.x:+.2f} m/s  '
                f'ang={twist.angular.z:+.2f} rad/s')
        colour = (
            (0, 255,   0) if twist.linear.x > 0 else
            (0,   0, 255) if twist.linear.x < 0 else
            (0, 200, 255)
        )
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
        cv2.putText(frame, text, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2)


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
        stop = Twist()
        node.cmd_pub.publish(stop)
        node.get_logger().info('Safety stop published. Shutting down.')
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()