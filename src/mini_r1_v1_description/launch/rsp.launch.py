import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, AppendEnvironmentVariable, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():

    pkg_name = 'mini_r1_v1_description'
    pkg_path = get_package_share_directory(pkg_name)

    use_sim_time = LaunchConfiguration('use_sim_time')
    use_control  = LaunchConfiguration('use_control')
    x_pos        = LaunchConfiguration('x')
    y_pos        = LaunchConfiguration('y')
    z_pos        = LaunchConfiguration('z')

    xacro_file = os.path.join(pkg_path, 'urdf', 'mini_r1.urdf.xacro')

    robot_description = ParameterValue(
        Command(['xacro ', xacro_file, ' use_control:=', use_control]),
        value_type=str
    )

    # Make the package's models visible to Gazebo
    install_dir = os.path.dirname(pkg_path)
    set_ign_resource_path = AppendEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=install_dir
    )
    set_gz_resource_path = AppendEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=install_dir
    )

    # Launch Ignition Gazebo with an empty world — swap out the world file if you have one
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': '-r -v 4 empty.sdf',
            'on_exit_shutdown': 'true'
        }.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description,
            'use_sim_time': use_sim_time
        }]
    )

    # Spawns the robot into the running Gazebo instance using the robot_description topic
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'mini_r1',
            '-topic', 'robot_description',
            '-x', x_pos,
            '-y', y_pos,
            '-z', z_pos,
        ],
        output='screen'
    )

    launch_args = [
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use simulation clock if true'),
        DeclareLaunchArgument('use_control', default_value='false',
                              description='Toggle ros2_control vs gz control'),
        DeclareLaunchArgument('x', default_value='0.0', description='Spawn X position'),
        DeclareLaunchArgument('y', default_value='0.0', description='Spawn Y position'),
        DeclareLaunchArgument('z', default_value='0.5', description='Spawn Z position'),
    ]

    return LaunchDescription([
        *launch_args,
        set_ign_resource_path,
        set_gz_resource_path,
        gazebo,
        robot_state_publisher,
        spawn_robot,
    ])