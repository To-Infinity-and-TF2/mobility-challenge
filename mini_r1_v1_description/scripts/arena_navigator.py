#!/usr/bin/env python3

import math
import time
import json

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
import tf2_ros
from tf2_ros import TransformException


class ArenaNavigator(Node):
    """Complete arena navigation node: detects ArUco markers, follows directions, visits all markers, goes to goal."""

    def __init__(self) -> None:
        super().__init__("arena_navigator")

        self.declare_parameter("camera_topic", "/r1_mini/camera/image_raw")
        self.declare_parameter("scan_topic", "/r1_mini/lidar")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("navigate_to_pose_action", "navigate_to_pose")
        self.declare_parameter("aruco_dictionary", "DICT_6X6_250")
        self.declare_parameter("all_aruco_ids", [0, 1, 2, 3, 4, 5])
        self.declare_parameter("direction_aruco_ids", [1, 2, 3])
        self.declare_parameter("navigation_distance", 1.8)
        self.declare_parameter("min_approach_distance", 0.10)
        self.declare_parameter("max_approach_distance", 2.0)
        self.declare_parameter("forward_speed", 0.14)
        self.declare_parameter("rotation_speed", 0.6)
        self.declare_parameter("alignment_tolerance", 0.08)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("goal_x", 5.0)
        self.declare_parameter("goal_y", 0.0)
        self.declare_parameter("goal_yaw", 0.0)
        self.declare_parameter("control_rate", 10.0)
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

        self.camera_topic = self.get_parameter("camera_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.action_name = self.get_parameter("navigate_to_pose_action").value
        self.aruco_dictionary = self.get_parameter("aruco_dictionary").value
        self.all_aruco_ids = set(int(i) for i in self.get_parameter("all_aruco_ids").value)
        self.direction_aruco_ids = set(int(i) for i in self.get_parameter("direction_aruco_ids").value)
        self.navigation_distance = float(self.get_parameter("navigation_distance").value)
        self.min_approach_distance = float(self.get_parameter("min_approach_distance").value)
        self.max_approach_distance = float(self.get_parameter("max_approach_distance").value)
        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.rotation_speed = float(self.get_parameter("rotation_speed").value)
        self.alignment_tolerance = float(self.get_parameter("alignment_tolerance").value)
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.goal_yaw = float(self.get_parameter("goal_yaw").value)
        self.control_rate = float(self.get_parameter("control_rate").value)
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

        self.bridge = CvBridge()
        self.latest_scan = None
        self.visited_markers = set()
        self.current_direction_id = None
        self.current_direction_name = None
        self.current_marker_center_x = 0.5
        self.last_marker_seen_time = 0.0
        self.maze_mode_active = False
        self.goal_active = False
        self.exploring = True
        self.turn_mode = None
        self.turn_started_at = 0.0

        self.aruco_dict = self._load_aruco_dictionary(self.aruco_dictionary)
        self.aruco_params = self._create_aruco_parameters()
        self.aruco_detector = self._create_aruco_detector()

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.image_sub = self.create_subscription(Image, self.camera_topic, self._image_callback, 10)
        self.control_timer = self.create_timer(0.05, self._control_loop)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, self.action_name)

        self.get_logger().info(f"Arena navigator listening on {self.camera_topic} and {self.scan_topic}")
        self.get_logger().info(f"All ArUco IDs: {sorted(self.all_aruco_ids)}, directions: {sorted(self.direction_aruco_ids)}")

    def _load_aruco_dictionary(self, dictionary_name: str):
        dictionary_attr = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_attr is None:
            self.get_logger().warning(
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

        image_width = gray.shape[1]
        maze_triggered = False
        direction_candidates = []

        for marker_id, marker_corners in zip(ids.flatten(), corners):
            marker_id = int(marker_id)
            if marker_id not in self.all_aruco_ids:
                continue
            if marker_id not in self.visited_markers:
                self.visited_markers.add(marker_id)
                self.get_logger().info(f"Visited ArUco marker {marker_id} ({len(self.visited_markers)}/{len(self.all_aruco_ids)})")
            if marker_id == 0:
                maze_triggered = True
            if marker_id in self.direction_aruco_ids:
                center_x = self._compute_marker_center_x(marker_corners, image_width)
                direction_candidates.append((marker_id, center_x))

        if maze_triggered and not self.maze_mode_active:
            self.maze_mode_active = True
            self.exploring = False
            self.get_logger().info("Maze mode activated: switching to left-hand algorithm.")

        if direction_candidates:
            marker_id, center_x = self._select_nearest_marker(direction_candidates)
            self.current_direction_id = marker_id
            self.current_direction_name = self._direction_name(marker_id)
            self.current_marker_center_x = center_x
            self.last_marker_seen_time = time.time()
            self.exploring = False
            self._send_direction_goal(self.current_direction_name)

        if len(self.visited_markers) >= len(self.all_aruco_ids) and not self.goal_active:
            self._send_goal_zone_goal()

    def _select_nearest_marker(self, direction_candidates):
        return min(direction_candidates, key=lambda item: abs(item[1] - 0.5))

    def _direction_name(self, marker_id: int) -> str:
        if marker_id == 1:
            return "LEFT"
        if marker_id == 2:
            return "RIGHT"
        return "FORWARD"

    def _compute_marker_center_x(self, corners, image_width: int) -> float:
        flat = corners.reshape((4, 2))
        center_x = float(flat[:, 0].mean())
        return float(center_x) / float(image_width)

    def _control_loop(self) -> None:
        if self.maze_mode_active:
            self._maze_control_loop()
            return

        if self.goal_active:
            # Goal is active, let Nav2 handle it
            return

        if time.time() - self.last_marker_seen_time > 0.5:
            # No recent marker; explore
            if self.exploring:
                self._explore()
            return

        if self.latest_scan is None:
            return

        front_distance = self._window_min(self.latest_scan, 0.0, 15.0)
        if front_distance == math.inf:
            return

        if front_distance > self.max_approach_distance:
            self._align_and_approach(marker_in_sight=True)
            return

        if front_distance < self.min_approach_distance:
            self._publish_cmd(0.0, 0.0)
            self.get_logger().info("Reached close enough to marker.")
            self.exploring = True
            return

        self._align_and_approach(marker_in_sight=True)

    def _maze_control_loop(self) -> None:
        if self.latest_scan is None:
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
            angular_z = self.rotation_speed * 0.5

        if front_left < self.front_clearance:
            angular_z -= self.corner_gain * (self.front_clearance - front_left)

        if front < (self.front_clearance + 0.25):
            slowdown = max(
                0.45,
                (front - self.front_stop_distance) / max(self.front_clearance - self.front_stop_distance, 0.05),
            )
            linear_x *= min(1.0, slowdown)

        angular_z = self._clamp(angular_z, -self.max_angular_speed, self.max_angular_speed)
        self._publish_cmd(linear_x, angular_z)

    def _align_and_approach(self, marker_in_sight: bool) -> None:
        if not marker_in_sight:
            self._publish_cmd(0.0, 0.0)
            return

        error = self.current_marker_center_x - 0.5
        angular_z = -self.rotation_speed * error
        angular_z = self._clamp(angular_z, -self.rotation_speed, self.rotation_speed)

        if abs(error) < self.alignment_tolerance:
            linear_x = self.forward_speed
        else:
            linear_x = 0.0

        self._publish_cmd(linear_x, angular_z)

    def _explore(self) -> None:
        # Simple exploration: move forward slowly
        self._publish_cmd(0.1, 0.0)

    def _send_direction_goal(self, direction_name: str) -> None:
        if not self.nav_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warning("Nav2 action server not available, cannot send direction goal.")
            return

        if self.goal_active:
            self.get_logger().info("A goal is already active.")
            return

        transform = self._lookup_transform(self.map_frame, self.base_frame)
        if transform is None:
            self.get_logger().warning("Could not get robot pose from TF, goal not sent.")
            return

        current_x = transform.transform.translation.x
        current_y = transform.transform.translation.y
        current_yaw = self._yaw_from_quaternion(transform.transform.rotation)

        dx, dy = self._relative_offset(direction_name)
        goal_x = current_x + dx * math.cos(current_yaw) - dy * math.sin(current_yaw)
        goal_y = current_y + dx * math.sin(current_yaw) + dy * math.cos(current_yaw)

        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = goal_x
        pose.pose.position.y = goal_y
        pose.pose.position.z = 0.0
        qx, qy, qz, qw = self._quaternion_from_yaw(current_yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(f"Sending Nav2 goal for direction {direction_name} to ({goal_x:.2f}, {goal_y:.2f}).")
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._goal_response_callback)
        self.goal_active = True

    def _send_goal_zone_goal(self) -> None:
        if not self.nav_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warning("Nav2 action server not available, cannot send goal zone goal.")
            return

        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = self.goal_x
        pose.pose.position.y = self.goal_y
        pose.pose.position.z = 0.0
        qx, qy, qz, qw = self._quaternion_from_yaw(self.goal_yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        self.get_logger().info(f"All markers visited! Sending goal to goal zone at ({self.goal_x:.2f}, {self.goal_y:.2f}).")
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._goal_response_callback)
        self.goal_active = True

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning("Goal rejected by Nav2.")
            self.goal_active = False
            return

        self.get_logger().info("Goal accepted, waiting for result.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        self.goal_active = False
        if status in [4, 5, 9]:
            self.get_logger().warn(f"Goal failed with status {status}.")
        else:
            self.get_logger().info("Goal completed.")
            self.exploring = True

    def _lookup_transform(self, target_frame: str, source_frame: str):
        try:
            return self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        except TransformException as exc:
            self.get_logger().warning(f"TF lookup failed: {exc}")
            return None

    def _relative_offset(self, direction_name: str) -> tuple[float, float]:
        distance = min(max(self.navigation_distance, self.min_approach_distance), self.max_approach_distance)
        if direction_name == "LEFT":
            return 0.0, distance
        if direction_name == "RIGHT":
            return 0.0, -distance
        return distance, 0.0

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
            self._publish_cmd(self.forward_speed * 0.5, self.rotation_speed)
            return

        if self.turn_mode == "right":
            self._publish_cmd(0.0, -self.rotation_speed)
            return

        if self.turn_mode == "uturn":
            self._publish_cmd(0.0, -self.rotation_speed)
            return

        self._publish_cmd(0.0, 0.0)

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _yaw_from_quaternion(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
        return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)

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
        return min(valid_ranges) if valid_ranges else math.inf

    def _publish_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        self.cmd_pub.publish(msg)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))


def main(args=None):
    rclpy.init(args=args)
    node = ArenaNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
