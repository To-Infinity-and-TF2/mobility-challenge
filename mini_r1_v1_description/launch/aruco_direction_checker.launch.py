from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    camera_topic = LaunchConfiguration("camera_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    use_nav2_action = LaunchConfiguration("use_nav2_action")

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
            "use_nav2_action",
            default_value="true",
            description="Whether to send Nav2 NavigateToPose goals",
        ),
    ]

    aruco_direction_checker = Node(
        package="mini_r1_v1_description",
        executable="aruco_direction_checker",
        output="screen",
        parameters=[
            {
                "camera_topic": camera_topic,
                "scan_topic": scan_topic,
                "cmd_vel_topic": cmd_vel_topic,
                "use_nav2_action": use_nav2_action,
            }
        ],
    )

    return LaunchDescription([*launch_args, aruco_direction_checker])
