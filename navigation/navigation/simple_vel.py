import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class Vel_pub(Node):
    def __init__(self):
        super().__init__("vel_pub_node")

        self.declare_parameter("linear_speed", 0.20)
        self.declare_parameter("angular_speed", 0.0)
        self.declare_parameter("publish_period", 0.1)

        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_speed = float(self.get_parameter("angular_speed").value)
        publish_period = float(self.get_parameter("publish_period").value)

        self.pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.timer_1 = self.create_timer(publish_period, self.straight)

        self.get_logger().info(
            f"Publishing /cmd_vel with linear.x={self.linear_speed:.2f}, "
            f"angular.z={self.angular_speed:.2f}"
        )

    def straight(self):
        go_straight = Twist()
        go_straight.linear.x = self.linear_speed
        go_straight.angular.z = self.angular_speed
        self.pub.publish(go_straight)

    """
    def right_turn(self):
        right_turn=Twist()
        right_turn.angular.z=1.57079632679
        self.pub.publish(right_turn)
    """

def main():
    rclpy.init()
    velocity_publisher = Vel_pub()

    rclpy.spin(velocity_publisher)
    rclpy.shutdown()
