#!/usr/bin/env python3
"""
Navigation Decision Engine — ROS 2 Humble
Subscribes to /perception/sign_detected and /perception/aruco_visited.
Sends Nav2 NavigateToPose goals and triggers recovery behaviours.

Priority logic:
  1. GOAL sign → navigate to goal zone
  2. All 4 ArUco markers visited → navigate to goal zone
  3. Sign detected → execute turn / forward / rotate-in-place
  4. No cue → explore (random frontier-like forward nudge)
  5. Stuck > 8 s → recovery (rotate, backtrack)
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose


TOTAL_MARKERS = 4
STUCK_TIMEOUT = 8.0          # seconds before triggering recovery
EXPLORE_STEP = 0.8           # metres to nudge forward during exploration
BACKTRACK_DIST = 0.5         # metres to reverse during backtrack recovery


class NavigationEngine(Node):
    def __init__(self):
        super().__init__("navigation_engine")

        self._visited: list[int] = []
        self._last_sign: str | None = None
        self._mission_complete = False
        self._last_progress_time = time.time()

        # ── Subscriptions ──────────────────────────────────────────────────
        self.create_subscription(
            String, "/perception/sign_detected", self._sign_cb, 10
        )
        self.create_subscription(
            String, "/perception/aruco_visited", self._aruco_cb, 10
        )

        # ── Publishers ─────────────────────────────────────────────────────
        self._cmd_vel = self.create_publisher(Twist, "/cmd_vel", 10)

        # ── Nav2 action client ─────────────────────────────────────────────
        self._nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # ── Stuck detector timer ───────────────────────────────────────────
        self.create_timer(1.0, self._check_stuck)

        self.get_logger().info("Navigation engine ready.")

    # ──────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _sign_cb(self, msg: String):
        sign = msg.data
        self.get_logger().info(f"Sign received: {sign}")
        self._last_sign = sign
        self._last_progress_time = time.time()
        self._act_on_sign(sign)

    def _aruco_cb(self, msg: String):
        import json
        self._visited = json.loads(msg.data)
        self.get_logger().info(f"ArUco visited: {self._visited}")
        self._last_progress_time = time.time()

        if len(self._visited) >= TOTAL_MARKERS and not self._mission_complete:
            self.get_logger().info("All 4 markers visited — heading to goal zone!")
            self._navigate_to_goal()

    # ──────────────────────────────────────────────────────────────────────────
    # Sign actions
    # ──────────────────────────────────────────────────────────────────────────

    def _act_on_sign(self, sign: str):
        if self._mission_complete:
            return

        if sign == "GOAL":
            self._navigate_to_goal()
        elif sign == "STOP":
            self._stop()
        elif sign == "LEFT":
            self._turn(math.pi / 2)          # 90° left
        elif sign == "RIGHT":
            self._turn(-math.pi / 2)         # 90° right
        elif sign == "FORWARD":
            self._move_forward(EXPLORE_STEP)
        elif sign == "INPLACE_ROTATION":
            self._turn(math.pi)              # 180° spin

    # ──────────────────────────────────────────────────────────────────────────
    # Motion helpers (direct cmd_vel; swap for Nav2 goal if preferred)
    # ──────────────────────────────────────────────────────────────────────────

    def _stop(self):
        self.get_logger().info("STOP — halting robot.")
        twist = Twist()
        self._cmd_vel.publish(twist)

    def _move_forward(self, distance: float, speed: float = 0.2):
        """Publish forward velocity for the time needed to cover `distance`."""
        duration = distance / speed
        twist = Twist()
        twist.linear.x = speed
        end = time.time() + duration
        # Use a one-shot timer to stop after duration
        def _stop_cb():
            self._cmd_vel.publish(Twist())
        self._cmd_vel.publish(twist)
        self.create_timer(duration, lambda: (self._cmd_vel.publish(Twist()),))

    def _turn(self, angle_rad: float, speed: float = 0.5):
        """Rotate in place by angle_rad (positive = CCW)."""
        duration = abs(angle_rad) / speed
        twist = Twist()
        twist.angular.z = math.copysign(speed, angle_rad)
        self._cmd_vel.publish(twist)
        self.create_timer(duration, lambda: (self._cmd_vel.publish(Twist()),))

    # ──────────────────────────────────────────────────────────────────────────
    # Nav2 goal
    # ──────────────────────────────────────────────────────────────────────────

    def _navigate_to_goal(self):
        """
        Send a Nav2 NavigateToPose goal to the true goal zone.
        REPLACE the pose coordinates below with your world's actual goal position.
        """
        if self._mission_complete:
            return
        self._mission_complete = True

        if not self._nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available!")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        # ── TODO: Set your actual goal zone coordinates here ──────────────
        goal_msg.pose.pose.position.x = 5.0
        goal_msg.pose.pose.position.y = 0.0
        goal_msg.pose.pose.orientation.w = 1.0
        # ─────────────────────────────────────────────────────────────────

        self.get_logger().info("Sending Nav2 goal to goal zone...")
        self._nav_client.send_goal_async(goal_msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Recovery: Mandatory behaviours (Rule 5)
    # ──────────────────────────────────────────────────────────────────────────

    def _check_stuck(self):
        """Triggered every second. If no progress in STUCK_TIMEOUT seconds → recover."""
        if self._mission_complete:
            return
        elapsed = time.time() - self._last_progress_time
        if elapsed > STUCK_TIMEOUT:
            self.get_logger().warn(
                f"Stuck for {elapsed:.1f}s — triggering recovery!"
            )
            self._recover()
            self._last_progress_time = time.time()

    def _recover(self):
        """
        Recovery sequence (satisfies mandatory requirement):
          1. Rotate in place 180° (escape dead-end / reorient)
          2. Backtrack 0.5 m
        """
        self.get_logger().info("Recovery: rotate in place 180°")
        self._turn(math.pi)

        # Schedule backtrack after rotation completes (≈ π/0.5 ≈ 6.3 s)
        rotation_duration = math.pi / 0.5 + 0.5
        self.create_timer(
            rotation_duration,
            lambda: self._backtrack()
        )

    def _backtrack(self):
        self.get_logger().info(f"Recovery: backtrack {BACKTRACK_DIST}m")
        twist = Twist()
        twist.linear.x = -0.15          # reverse slowly
        duration = BACKTRACK_DIST / 0.15
        self._cmd_vel.publish(twist)
        self.create_timer(duration, lambda: (self._cmd_vel.publish(Twist()),))


# ──────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = NavigationEngine()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()