from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scan_topic = LaunchConfiguration("scan_topic")
    camera_topic = LaunchConfiguration("camera_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    use_sim_time = LaunchConfiguration("use_sim_time")

    launch_args = [
        DeclareLaunchArgument(
            "scan_topic",
            default_value="/r1_mini/lidar",
            description="LaserScan topic used by the micro mouse navigator",
        ),
        DeclareLaunchArgument(
            "camera_topic",
            default_value="/r1_mini/camera/image_raw",
            description="Camera image topic for ArUco detection",
        ),
        DeclareLaunchArgument(
            "cmd_vel_topic",
            default_value="/cmd_vel",
            description="Velocity command topic for the robot base",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation time",
        ),
    ]

    micro_mouse = Node(
        package="mini_r1_v1_description",
        executable="micro_mouse_wall_follower",
        output="screen",
        parameters=[
            {
                "scan_topic": scan_topic,
                "camera_topic": camera_topic,
                "cmd_vel_topic": cmd_vel_topic,
                "use_sim_time": use_sim_time,
            }
        ],
    )

    return LaunchDescription([*launch_args, micro_mouse])
