#!/usr/bin/env python3
"""
ArtPark Arena Bot Navigation — ROS2 Node (Simulation)
======================================================
Subscribes : /r1_mini/camera/image_raw      (sensor_msgs/Image)  ← RGB
             /r1_mini/depth_cam/image_raw   (sensor_msgs/Image)  ← Depth
Publishes  : /cmd_vel                       (geometry_msgs/Twist)

Run:
    python3 artpark_navigation_node.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# SIMULATION CAMERA TOPICS
# ---------------------------------------------------------------------------
RGB_TOPIC   = '/r1_mini/camera/image_raw'
DEPTH_TOPIC = '/r1_mini/depth_cam/image_raw'

# ---------------------------------------------------------------------------
# VELOCITY CONSTANTS — tune for your bot
# ---------------------------------------------------------------------------
LINEAR_SPEED_FWD  =  0.2    # m/s  forward
LINEAR_SPEED_BWD  = -0.15   # m/s  backward
ANGULAR_TURN      =  0.4    # rad/s  90° turn
ANGULAR_U_TURN    =  0.8    # rad/s  U-turn

# ---------------------------------------------------------------------------
# HSV COLOUR RANGES — tune with the hsv_tuner if needed
# ---------------------------------------------------------------------------
GREEN_HSV_LOW   = np.array([40,  80,  80],  dtype=np.uint8)
GREEN_HSV_HIGH  = np.array([85, 255, 255],  dtype=np.uint8)

ORANGE_HSV_LOW  = np.array([5,  120, 120],  dtype=np.uint8)
ORANGE_HSV_HIGH = np.array([22, 255, 255],  dtype=np.uint8)

MIN_BLOB_AREA        = 500    # px²
SIDE_THRESHOLD_RATIO = 0.15   # fraction of frame height for "side-by-side"

# ---------------------------------------------------------------------------
# AprilTag ID → (linear.x, angular.z)
# Edit to match your arena's physical tag IDs
# ---------------------------------------------------------------------------
TAG_TWIST_MAP = {
    1: (LINEAR_SPEED_FWD, 0.0),
    2: (0.0,  ANGULAR_U_TURN),
    3: (0.0, -ANGULAR_TURN),
    4: (0.0,  ANGULAR_U_TURN),
    5: (LINEAR_SPEED_FWD, 0.0),
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
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

        # Parameters (settable via: ros2 param set /artpark_nav_node active_colour ORANGE)
        self.declare_parameter('active_colour', 'GREEN')
        self.declare_parameter('show_debug',    True)

        self.active_colour = self.get_parameter('active_colour').value.upper()
        self.show_debug    = self.get_parameter('show_debug').value

        self.bridge      = CvBridge()
        self.depth_frame = None   # latest depth image, kept in sync

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # RGB subscriber — main navigation loop
        self.create_subscription(
            Image, RGB_TOPIC, self._rgb_cb, 10
        )

        # Depth subscriber — stored for optional use (obstacle check etc.)
        self.create_subscription(
            Image, DEPTH_TOPIC, self._depth_cb, 10
        )

        self.get_logger().info(
            f"ArtPark Nav Node started\n"
            f"  RGB  topic : {RGB_TOPIC}\n"
            f"  Depth topic: {DEPTH_TOPIC}\n"
            f"  Colour mode: {self.active_colour}"
        )

    # ------------------------------------------------------------------
    # Depth callback — just cache the latest frame
    # ------------------------------------------------------------------
    def _depth_cb(self, msg: Image):
        try:
            # 32FC1 = 32-bit float depth in metres (standard Gazebo output)
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().warn(f"Depth decode error: {e}")

    # ------------------------------------------------------------------
    # RGB callback — runs navigation pipeline every frame
    # ------------------------------------------------------------------
    def _rgb_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"RGB decode error: {e}")
            return

        # Re-read parameter in case it was changed at runtime
        self.active_colour = self.get_parameter('active_colour').value.upper()

        twist = self._process_frame(frame)
        self.cmd_pub.publish(twist)

        if self.show_debug:
            cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Full per-frame pipeline
    # ------------------------------------------------------------------
    def _process_frame(self, frame):

        debug = frame.copy() if self.show_debug else None
        h, w = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        twist = Twist()

        # ==============================
        # 1. APRILTAG PRIORITY
        # ==============================
        tag = self._detect_apriltag(gray, debug)
        if tag is not None:
            twist.linear.x = 0.2
            twist.angular.z = 0.0

            if debug is not None:
                cv2.putText(debug, "APRILTAG MODE", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

            return twist

        # ==============================
        # 2. DETECT BLOBS + FEATURES
        # ==============================
        green = self._detect_blob_advanced(hsv, frame, "GREEN")
        orange = self._detect_blob_advanced(hsv, frame, "ORANGE")

        # ==============================
        # 3. COMPUTE SCORES
        # ==============================
        candidates = []

        for blob, label in [(green, "GREEN"), (orange, "ORANGE")]:
            if blob is None:
                continue

            cx, cy = blob["center"]
            conf = blob["confidence"]
            z = blob["depth"]

            if z is None:
                continue

            score = conf / (z + 1e-3)

            candidates.append((score, label, blob))

        # ==============================
        # 4. MODE SELECTION
        # ==============================
        if candidates:
            candidates.sort(reverse=True)
            _, mode, best_blob = candidates[0]

            cx, cy = best_blob["center"]
            z = best_blob["depth"]

            error_x = cx - w//2

            twist.angular.z = -0.003 * error_x
            twist.linear.x = 0.2

            if debug is not None:
                cv2.putText(debug, f"{mode} MODE", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        else:
            # ==============================
            # 5. EXPLORATION MODE
            # ==============================
            twist.linear.x = 0.05
            twist.angular.z = 0.4

            if debug is not None:
                cv2.putText(debug, "EXPLORING", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # ==============================
        # STORE DEBUG FRAME
        # ==============================
        if self.show_debug:
            with self.lock:
                self.debug_frame = debug.copy()

        return twist

    # ------------------------------------------------------------------
    # Colour blob detection
    # ------------------------------------------------------------------
    def _detect_blob(self,
                     hsv: np.ndarray,
                     low: np.ndarray,
                     high: np.ndarray,
                     label: str,
                     debug: Optional[np.ndarray]) -> Optional[ColourBlob]:

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
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if debug is not None:
            colour_bgr = (0, 220, 0) if label == "GREEN" else (0, 140, 255)
            cv2.drawContours(debug, [largest], -1, colour_bgr, 2)
            cv2.circle(debug, (cx, cy), 9, colour_bgr, -1)
            cv2.putText(debug, f"{label} {area:.0f}px",
                        (cx + 12, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour_bgr, 2)

        return ColourBlob(centroid=(cx, cy), area=area)

    # ------------------------------------------------------------------
    # Blob geometry → Twist  (RGB only)
    # ------------------------------------------------------------------
    def _blobs_to_twist(self,
                         green:  Optional[ColourBlob],
                         orange: Optional[ColourBlob],
                         frame_h: int) -> Twist:

        twist     = Twist()
        primary   = green  if self.active_colour == "GREEN"  else orange
        secondary = orange if self.active_colour == "GREEN"  else green
        threshold = frame_h * SIDE_THRESHOLD_RATIO

        if primary is None and secondary is None:
            twist.linear.x = LINEAR_SPEED_FWD * 0.5
            self.get_logger().warn("No blobs visible — creeping forward")
            return twist

        if primary is None:
            twist.linear.x = LINEAR_SPEED_FWD * 0.4
            self.get_logger().warn(f"Lost {self.active_colour} — creeping")
            return twist

        if secondary is None:
            twist.linear.x = LINEAR_SPEED_FWD
            return twist

        dy = primary.centroid[1] - secondary.centroid[1]
        # dy > 0 → primary lower in frame (closer to bot) → facing bot → FORWARD
        # dy < 0 → primary higher (far side)              → overshot  → BACKWARD
        # |dy| small → side-by-side                                    → STRAIGHT

        if abs(dy) < threshold:
            twist.linear.x = LINEAR_SPEED_FWD
            self.get_logger().info("Side-by-side → STRAIGHT")
        elif dy > 0:
            twist.linear.x = LINEAR_SPEED_FWD
            self.get_logger().info(f"{self.active_colour} near-side → FORWARD")
        else:
            twist.linear.x = LINEAR_SPEED_BWD
            self.get_logger().warn(f"{self.active_colour} far-side → BACKWARD")

        return twist

    # ------------------------------------------------------------------
    # Blob geometry → Twist  (with depth obstacle check)
    # ------------------------------------------------------------------
    def _blobs_to_twist_with_depth(self,
                                    green:  Optional[ColourBlob],
                                    orange: Optional[ColourBlob],
                                    frame_h: int) -> Twist:

        twist = self._blobs_to_twist(green, orange, frame_h)

        # Only check depth when about to go forward
        if twist.linear.x > 0 and self.depth_frame is not None:
            h, w  = self.depth_frame.shape[:2]
            # Sample a central foreground strip (middle 20% width, rows 40-70% down)
            strip = self.depth_frame[int(h * 0.4):int(h * 0.7),
                                     int(w * 0.4):int(w * 0.6)]
            strip = np.where(np.isfinite(strip), strip, 10.0)
            min_d = float(np.min(strip))

            if min_d < 0.3:   # obstacle closer than 30 cm
                twist.linear.x  = 0.0
                twist.angular.z = 0.0
                self.get_logger().warn(f"Obstacle at {min_d:.2f} m — STOP")

        return twist

    # ------------------------------------------------------------------
    # AprilTag detection
    # ------------------------------------------------------------------
    def _detect_apriltag(self,
                          gray: np.ndarray,
                          debug: Optional[np.ndarray]) -> Optional[dict]:
        # Try pupil-apriltags first
        try:
            from pupil_apriltags import Detector
            detector   = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)
            detections = detector.detect(gray)
            if detections:
                best = max(detections,
                           key=lambda d: cv2.contourArea(d.corners.astype(np.float32)))
                cx, cy = int(best.center[0]), int(best.center[1])
                if debug is not None:
                    pts = best.corners.astype(int)
                    cv2.polylines(debug, [pts], True, (255, 0, 255), 2)
                    cv2.circle(debug, (cx, cy), 6, (255, 0, 255), -1)
                    cv2.putText(debug, f"TAG {best.tag_id}",
                                (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 0, 255), 2)
                return {"id": best.tag_id, "center": (cx, cy)}
        except ImportError:
            pass

        # OpenCV ArUco fallback
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
            cv2.putText(debug, f"TAG {best_id}",
                        (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        return {"id": best_id, "center": (cx, cy)}

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
        text   = (f"{label} | lin={twist.linear.x:+.2f} m/s  "
                  f"ang={twist.angular.z:+.2f} rad/s")
        colour = (0, 255, 0) if twist.linear.x > 0 else \
                 (0, 0, 255) if twist.linear.x < 0 else \
                 (0, 200, 255)
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
        node.get_logger().info("Safety stop published. Shutting down.")
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()