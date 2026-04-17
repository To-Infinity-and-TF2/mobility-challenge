#!/usr/bin/env python3

import math

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan


class MicroMouseWallFollower(Node):
    def __init__(self) -> None:
        super().__init__("micro_mouse_wall_follower")

        self.declare_parameter("scan_topic", "/r1_mini/lidar")
        self.declare_parameter("camera_topic", "/r1_mini/camera/image_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("forward_speed", 0.18)
        self.declare_parameter("turn_speed", 0.9)
        self.declare_parameter("turn_forward_speed", 0.04)
        self.declare_parameter("wall_distance", 0.55)
        self.declare_parameter("left_open_distance", 0.95)
        self.declare_parameter("front_clearance", 0.70)
        self.declare_parameter("front_stop_distance", 0.40)
        self.declare_parameter("side_clearance", 0.45)
        self.declare_parameter("follow_gain", 1.8)
        self.declare_parameter("corner_gain", 1.1)
        self.declare_parameter("max_angular_speed", 1.2)
        self.declare_parameter("min_turn_time", 0.35)
        self.declare_parameter("left_turn_timeout", 1.8)
        self.declare_parameter("right_turn_timeout", 1.4)
        self.declare_parameter("uturn_timeout", 2.8)
        self.declare_parameter("aruco_ids", [0, 1, 2, 3, 4, 5])
        self.declare_parameter("aruco_dictionary", "DICT_6X6_250")
        self.declare_parameter("completion_hold_time", 1.5)

        self.scan_topic = self.get_parameter("scan_topic").value
        self.camera_topic = self.get_parameter("camera_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.turn_speed = float(self.get_parameter("turn_speed").value)
        self.turn_forward_speed = float(self.get_parameter("turn_forward_speed").value)
        self.wall_distance = float(self.get_parameter("wall_distance").value)
        self.left_open_distance = float(self.get_parameter("left_open_distance").value)
        self.front_clearance = float(self.get_parameter("front_clearance").value)
        self.front_stop_distance = float(self.get_parameter("front_stop_distance").value)
        self.side_clearance = float(self.get_parameter("side_clearance").value)
        self.follow_gain = float(self.get_parameter("follow_gain").value)
        self.corner_gain = float(self.get_parameter("corner_gain").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.min_turn_time = float(self.get_parameter("min_turn_time").value)
        self.left_turn_timeout = float(self.get_parameter("left_turn_timeout").value)
        self.right_turn_timeout = float(self.get_parameter("right_turn_timeout").value)
        self.uturn_timeout = float(self.get_parameter("uturn_timeout").value)
        self.aruco_ids = set(int(i) for i in self.get_parameter("aruco_ids").value)
        self.aruco_dictionary = self.get_parameter("aruco_dictionary").value
        self.completion_hold_time = float(self.get_parameter("completion_hold_time").value)

        self.latest_scan = None
        self.turn_mode = None
        self.turn_started_at = 0.0
        self.seen_markers = set()
        self.completion_time = None

        self.bridge = CvBridge()
        self.aruco_dict = self._load_aruco_dictionary(self.aruco_dictionary)
        self.aruco_params = self._create_aruco_parameters()
        self.aruco_detector = self._create_aruco_detector()

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.image_sub = self.create_subscription(Image, self.camera_topic, self._image_callback, 10)
        self.control_timer = self.create_timer(1.0 / self.control_rate, self._control_loop)

        self.get_logger().info(
            f"Micro mouse wall follower listening on {self.scan_topic} and {self.camera_topic}, publishing to {self.cmd_vel_topic}"
        )
        self.get_logger().info(f"Expected ArUco IDs: {sorted(self.aruco_ids)}")

    def _load_aruco_dictionary(self, dictionary_name: str):
        dictionary_attr = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_attr is None:
            self.get_logger().warn(
                f"Aruco dictionary '{dictionary_name}' not found, falling back to DICT_6X6_250"
            )
            dictionary_attr = cv2.aruco.DICT_6X6_250
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            return cv2.aruco.getPredefinedDictionary(dictionary_attr)
        return cv2.aruco.Dictionary_get(dictionary_attr)

    def _create_aruco_parameters(self):
        if hasattr(cv2.aruco, "DetectorParameters"):
            return cv2.aruco.DetectorParameters()
        return cv2.aruco.DetectorParameters_create()

    def _create_aruco_detector(self):
        if hasattr(cv2.aruco, "ArucoDetector"):
            return cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        return None

    def _detect_markers(self, gray_image):
        if self.aruco_detector is not None:
            return self.aruco_detector.detectMarkers(gray_image)
        return cv2.aruco.detectMarkers(gray_image, self.aruco_dict, parameters=self.aruco_params)

    def _scan_callback(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def _image_callback(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert image: {exc}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detect_markers(gray)
        if ids is None:
            return

        for marker_id in ids.flatten():
            marker_id = int(marker_id)
            if marker_id not in self.aruco_ids:
                continue
            if marker_id not in self.seen_markers:
                self.seen_markers.add(marker_id)
                self.get_logger().info(
                    f"Detected ArUco marker {marker_id} ({len(self.seen_markers)}/{len(self.aruco_ids)})"
                )

        if len(self.seen_markers) >= len(self.aruco_ids) and self.completion_time is None:
            self.completion_time = self._now_seconds()
            self.get_logger().info("All ArUco markers have been observed. Preparing to stop.")

    def _control_loop(self) -> None:
        if self.latest_scan is None:
            self._publish_cmd(0.0, 0.0)
            return

        if self.completion_time is not None:
            if self._now_seconds() - self.completion_time >= self.completion_hold_time:
                self.get_logger().info("Micro mouse completed: stopping robot.")
                self._publish_cmd(0.0, 0.0)
                return

        front = self._window_min(self.latest_scan, 0.0, 20.0)
        front_left = self._window_min(self.latest_scan, 45.0, 20.0)
        left = self._window_min(self.latest_scan, 90.0, 25.0)
        right = self._window_min(self.latest_scan, -90.0, 25.0)

        if self.turn_mode is not None:
            if self._turn_is_complete(front, left):
                self.turn_mode = None
            else:
                self._publish_turn()
                return

        if front < self.front_stop_distance and left < self.side_clearance and right < self.side_clearance:
            self._start_turn("uturn")
            self._publish_turn()
            return

        if left > self.left_open_distance and front > self.front_stop_distance:
            self._start_turn("left")
            self._publish_turn()
            return

        if front < self.front_clearance:
            self._start_turn("right")
            self._publish_turn()
            return

        linear_x = self.forward_speed
        if left < math.inf:
            wall_error = self.wall_distance - left
            angular_z = -self.follow_gain * wall_error
        else:
            angular_z = self.turn_speed * 0.5

        if front_left < self.front_clearance:
            angular_z -= self.corner_gain * (self.front_clearance - front_left)

        if front < (self.front_clearance + 0.25):
            slowdown = max(
                0.45,
                (front - self.front_stop_distance)
                / max(self.front_clearance - self.front_stop_distance, 0.05),
            )
            linear_x *= min(1.0, slowdown)

        angular_z = self._clamp(angular_z, -self.max_angular_speed, self.max_angular_speed)
        self._publish_cmd(linear_x, angular_z)

    def _start_turn(self, mode: str) -> None:
        if self.turn_mode == mode:
            return

        self.turn_mode = mode
        self.turn_started_at = self._now_seconds()
        self.get_logger().info(f"Starting {mode} maneuver")

    def _turn_is_complete(self, front: float, left: float) -> bool:
        elapsed = self._now_seconds() - self.turn_started_at
        if elapsed < self.min_turn_time:
            return False

        if self.turn_mode == "left":
            return elapsed >= self.left_turn_timeout or (
                front > self.front_clearance and left <= self.left_open_distance
            )

        if self.turn_mode == "right":
            return elapsed >= self.right_turn_timeout or (
                front > self.front_clearance and left <= self.left_open_distance
            )

        if self.turn_mode == "uturn":
            return elapsed >= self.uturn_timeout and front > self.front_clearance

        return True

    def _publish_turn(self) -> None:
        if self.turn_mode == "left":
            self._publish_cmd(self.turn_forward_speed, self.turn_speed)
            return

        if self.turn_mode == "right":
            self._publish_cmd(0.0, -self.turn_speed)
            return

        if self.turn_mode == "uturn":
            self._publish_cmd(0.0, -self.turn_speed)
            return

        self._publish_cmd(0.0, 0.0)

    def _publish_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_pub.publish(msg)

    def _window_min(self, scan: LaserScan, center_deg: float, half_width_deg: float) -> float:
        center = math.radians(center_deg)
        half_width = math.radians(half_width_deg)
        valid_ranges = []

        for index, distance in enumerate(scan.ranges):
            if not math.isfinite(distance):
                continue
            if distance < scan.range_min or distance > scan.range_max:
                continue

            angle = scan.angle_min + index * scan.angle_increment
            angle_delta = math.atan2(math.sin(angle - center), math.cos(angle - center))
            if abs(angle_delta) <= half_width:
                valid_ranges.append(distance)

        if not valid_ranges:
            return math.inf

        return min(valid_ranges)

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MicroMouseWallFollower()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
