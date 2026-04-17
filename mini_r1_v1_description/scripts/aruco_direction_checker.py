#!/usr/bin/env python3

import math
import time

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


class ArucoDirectionChecker(Node):
    """Template node for ArUco direction-marker detection and action goals."""

    def __init__(self) -> None:
        super().__init__("aruco_direction_checker")

        self.declare_parameter("camera_topic", "/r1_mini/camera/image_raw")
        self.declare_parameter("scan_topic", "/r1_mini/lidar")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("direction_topic", "/perception/direction_marker")
        self.declare_parameter("maze_topic", "/perception/maze_mode")
        self.declare_parameter("navigate_to_pose_action", "navigate_to_pose")
        self.declare_parameter("aruco_dictionary", "DICT_6X6_250")
        self.declare_parameter("maze_aruco_id", 0)
        self.declare_parameter("direction_aruco_ids", [1, 2, 3])
        self.declare_parameter("navigation_distance", 1.8)
        self.declare_parameter("min_approach_distance", 0.10)
        self.declare_parameter("max_approach_distance", 2.0)
        self.declare_parameter("forward_speed", 0.14)
        self.declare_parameter("rotation_speed", 0.6)
        self.declare_parameter("alignment_tolerance", 0.08)
        self.declare_parameter("use_nav2_action", True)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_link")

        self.camera_topic = self.get_parameter("camera_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.direction_topic = self.get_parameter("direction_topic").value
        self.maze_topic = self.get_parameter("maze_topic").value
        self.action_name = self.get_parameter("navigate_to_pose_action").value
        self.aruco_dictionary = self.get_parameter("aruco_dictionary").value
        self.maze_aruco_id = int(self.get_parameter("maze_aruco_id").value)
        self.direction_aruco_ids = set(int(i) for i in self.get_parameter("direction_aruco_ids").value)
        self.navigation_distance = float(self.get_parameter("navigation_distance").value)
        self.min_approach_distance = float(self.get_parameter("min_approach_distance").value)
        self.max_approach_distance = float(self.get_parameter("max_approach_distance").value)
        self.forward_speed = float(self.get_parameter("forward_speed").value)
        self.rotation_speed = float(self.get_parameter("rotation_speed").value)
        self.alignment_tolerance = float(self.get_parameter("alignment_tolerance").value)
        self.use_nav2_action = bool(self.get_parameter("use_nav2_action").value)
        self.map_frame = self.get_parameter("map_frame").value
        self.base_frame = self.get_parameter("base_frame").value

        self.bridge = CvBridge()
        self.latest_scan = None
        self.last_marker_seen_time = 0.0
        self.current_direction_id = None
        self.current_direction_name = None
        self.current_marker_center_x = 0.5
        self.goal_sent_for_marker = None
        self.goal_active = False

        self.aruco_dict = self._load_aruco_dictionary(self.aruco_dictionary)
        self.aruco_params = self._create_aruco_parameters()
        self.aruco_detector = self._create_aruco_detector()

        self.direction_pub = self.create_publisher(String, self.direction_topic, 10)
        self.maze_pub = self.create_publisher(String, self.maze_topic, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self._scan_callback, 10)
        self.image_sub = self.create_subscription(Image, self.camera_topic, self._image_callback, 10)
        self.control_timer = self.create_timer(0.05, self._control_loop)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, self.action_name)

        self.get_logger().info(f"Aruco direction checker listening on {self.camera_topic} and {self.scan_topic}")
        self.get_logger().info(f"Maze ArUco ID: {self.maze_aruco_id}, direction IDs: {sorted(self.direction_aruco_ids)}")

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
        maze_seen = False
        direction_candidates = []

        for marker_id, marker_corners in zip(ids.flatten(), corners):
            marker_id = int(marker_id)
            if marker_id == self.maze_aruco_id:
                maze_seen = True
            if marker_id in self.direction_aruco_ids:
                center_x = self._compute_marker_center_x(marker_corners, image_width)
                direction_candidates.append((marker_id, center_x))

        if maze_seen:
            self._publish_maze_mode()

        if direction_candidates:
            marker_id, center_x = self._select_nearest_marker(direction_candidates)
            self.current_direction_id = marker_id
            self.current_direction_name = self._direction_name(marker_id)
            self.current_marker_center_x = center_x
            self.last_marker_seen_time = time.time()
            self._publish_direction_marker(self.current_direction_name)
            if self.current_direction_id != self.goal_sent_for_marker:
                self.goal_sent_for_marker = self.current_direction_id
                self.goal_active = False
                self._send_direction_goal(self.current_direction_name)

    def _publish_direction_marker(self, direction: str) -> None:
        msg = String()
        msg.data = direction
        self.direction_pub.publish(msg)
        self.get_logger().info(f"Detected direction marker: {direction}")

    def _publish_maze_mode(self) -> None:
        msg = String()
        msg.data = "LEFT_HAND_MAZE"
        self.maze_pub.publish(msg)
        self.get_logger().info("Maze mode marker 0 detected: activating left-hand maze behaviour.")

    def _select_nearest_marker(self, direction_candidates):
        # Prefer the marker closest to the image center, which is usually the easiest to approach.
        return min(direction_candidates, key=lambda item: abs(item[1] - 0.5))

    def _direction_name(self, marker_id: int) -> str:
        if marker_id == 1:
            return "LEFT"
        if marker_id == 2:
            return "RIGHT"
        if marker_id == 3:
            return "FORWARD"
        return f"UNKNOWN_{marker_id}"

    def _compute_marker_center_x(self, corners, image_width: int) -> float:
        flat = corners.reshape((4, 2))
        center_x = float(flat[:, 0].mean())
        return float(center_x) / float(image_width)

    def _control_loop(self) -> None:
        if time.time() - self.last_marker_seen_time > 0.5:
            # No marker seen recently; stop until a new marker appears
            self._publish_cmd(0.0, 0.0)
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
            return

        self._align_and_approach(marker_in_sight=True)

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

    def _send_direction_goal(self, direction_name: str) -> None:
        if not self.use_nav2_action:
            self.get_logger().info("Nav2 action disabled; skipping goal send.")
            return

        if not self.nav_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warning("Nav2 action server not available, cannot send direction goal.")
            return

        if self.goal_active:
            self.get_logger().info("A direction goal is already active.")
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

    def _goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warning("Direction goal rejected by Nav2.")
            self.goal_active = False
            return

        self.get_logger().info("Direction goal accepted, waiting for result.")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        result = future.result().result
        status = future.result().status
        self.goal_active = False
        if status in [4, 5, 9]:
            self.get_logger().warn(f"Direction goal failed with status {status}.")
        else:
            self.get_logger().info("Direction goal completed.")

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
    node = ArucoDirectionChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
