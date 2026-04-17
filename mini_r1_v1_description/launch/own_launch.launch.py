from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, SetEnvironmentVariable
#from launch_ros.actions import 
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.descriptions import ParameterValue
import os

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    ld=LaunchDescription()

    spawn_x = "-1.423120"
    spawn_y = "1.790000"
    spawn_z = "0.038178"
    spawn_yaw = "0.0"
   
    gz_sim=get_package_share_directory('ros_gz_sim')
    pkg_path=get_package_share_directory('mini_r1_v1_description')

    xacro_file = os.path.join(pkg_path, 'urdf', 'mini_r1.urdf.xacro')

    # Set the GZ_SIM_RESOURCE_PATH to include the worlds directory where arena model is located
    worlds_path = os.path.join(pkg_path, 'worlds')
    set_env = SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', worlds_path)


    model = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )


    gz_runner=IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gz_sim,'launch','gz_sim.launch.py')
        ),
        launch_arguments={
            "gz_args":os.path.join(pkg_path,"worlds","grid_world","worlds","grid_world_FINAL.sdf")
        }.items()
    )

    bridge_params=os.path.join(
        pkg_path,"params","bridge_params.yaml"
    )

    start_bridge=Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ],
        output='screen'
    )

    rob_bod_pub=Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': model,
                     'use_sim_time': True}],
        #output='screen',
    )

    birth=Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name","gefier",
            "-topic","/robot_description",
            "-x", spawn_x,
            "-y", spawn_y,
            "-z", spawn_z,
            "-Y", spawn_yaw,
        ],
        #arguments={"-name":"robo_v1","-topic":"/robot_description"}.items()

        #arguments=[{"name"}]
    )

    
    
    ld.add_action(set_env)
    ld.add_action(gz_runner)
    ld.add_action(rob_bod_pub)
    ld.add_action(birth)
    ld.add_action(start_bridge)

    return ld
