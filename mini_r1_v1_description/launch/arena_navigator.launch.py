from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_topic = LaunchConfiguration("camera_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    goal_x = LaunchConfiguration("goal_x")
    goal_y = LaunchConfiguration("goal_y")
    goal_yaw = LaunchConfiguration("goal_yaw")

    launch_args = [
        DeclareLaunchArgument(
            "camera_topic",
            default_value="/r1_mini/camera/image_raw",
            description="Camera image topic for ArUco detection",
        ),
        DeclareLaunchArgument(
            "scan_topic",
            default_value="/r1_mini/lidar",
            description="LaserScan topic for approach control",
        ),
        DeclareLaunchArgument(
            "cmd_vel_topic",
            default_value="/cmd_vel",
            description="Velocity command topic for the robot base",
        ),
        DeclareLaunchArgument(
            "goal_x",
            default_value="5.0",
            description="X coordinate of the goal zone",
        ),
        DeclareLaunchArgument(
            "goal_y",
            default_value="0.0",
            description="Y coordinate of the goal zone",
        ),
        DeclareLaunchArgument(
            "goal_yaw",
            default_value="0.0",
            description="Yaw orientation at the goal zone",
        ),
    ]

    arena_navigator = Node(
        package="mini_r1_v1_description",
        executable="arena_navigator",
        output="screen",
        parameters=[
            {
                "camera_topic": camera_topic,
                "scan_topic": scan_topic,
                "cmd_vel_topic": cmd_vel_topic,
                "goal_x": goal_x,
                "goal_y": goal_y,
                "goal_yaw": goal_yaw,
                "control_rate": 10.0,
                "wall_distance": 0.55,
                "left_open_distance": 0.95,
                "front_clearance": 0.70,
                "front_stop_distance": 0.40,
                "side_clearance": 0.45,
                "follow_gain": 1.8,
                "corner_gain": 1.1,
                "max_angular_speed": 1.2,
                "min_turn_time": 0.35,
                "left_turn_timeout": 1.8,
                "right_turn_timeout": 1.4,
                "uturn_timeout": 2.8,
            }
        ],
    )

    return LaunchDescription([*launch_args, arena_navigator])
