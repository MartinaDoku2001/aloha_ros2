from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import TimerAction  # Ensure this import is at the top of the file!


def generate_launch_description():
    # Declare launch arguments
    declare_arguments = [
        DeclareLaunchArgument('robot_model_master', default_value='wx250s'),
        DeclareLaunchArgument('robot_model_puppet', default_value='vx300s'),
        DeclareLaunchArgument('base_link_master', default_value='base_link'),
        DeclareLaunchArgument('base_link_puppet', default_value='base_link'),
        DeclareLaunchArgument('master_modes_left', default_value=PathJoinSubstitution([FindPackageShare('snn_aloha'), 'config', 'master_modes_left.yaml'])),
        DeclareLaunchArgument('puppet_modes_left', default_value=PathJoinSubstitution([FindPackageShare('snn_aloha'), 'config', 'puppet_modes_left.yaml'])),
        DeclareLaunchArgument('master_modes_right', default_value=PathJoinSubstitution([FindPackageShare('snn_aloha'), 'config', 'master_modes_right.yaml'])),
        DeclareLaunchArgument('puppet_modes_right', default_value=PathJoinSubstitution([FindPackageShare('snn_aloha'), 'config', 'puppet_modes_right.yaml'])),
        DeclareLaunchArgument('launch_driver', default_value='true'),
        DeclareLaunchArgument('use_sim', default_value='false'),
        DeclareLaunchArgument('robot_name_master_left', default_value='master_left'),
        DeclareLaunchArgument('robot_name_puppet_left', default_value='puppet_left'),
        DeclareLaunchArgument('robot_name_master_right', default_value='master_right'),
        DeclareLaunchArgument('robot_name_puppet_right', default_value='puppet_right')
    ]

    # xsarm_control launch inclusions with conditions
    includes = []
    for side, mode_config, robot_model, robot_name, base_link in [
        ('master_left', 'master_modes_left', 'robot_model_master', 'robot_name_master_left', 'base_link_master'),
        ('master_right', 'master_modes_right', 'robot_model_master', 'robot_name_master_right', 'base_link_master'),
        ('puppet_left', 'puppet_modes_left', 'robot_model_puppet', 'robot_name_puppet_left', 'base_link_puppet'),
        ('puppet_right', 'puppet_modes_right', 'robot_model_puppet', 'robot_name_puppet_right', 'base_link_puppet')
    ]:
        includes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    PathJoinSubstitution([
                        FindPackageShare('interbotix_xsarm_control'),
                        'launch',
                        'xsarm_control.launch.py'
                    ])
                ]),
                condition=IfCondition(LaunchConfiguration('launch_driver')),
                launch_arguments={
                    'robot_model': LaunchConfiguration(robot_model),
                    'robot_name': LaunchConfiguration(robot_name),
                    'base_link_frame': LaunchConfiguration(base_link),
                    'use_world_frame': 'false',
                    'use_rviz': 'false',
                    'mode_configs': LaunchConfiguration(mode_config),
                    'use_sim': LaunchConfiguration('use_sim')
                }.items()
            )
        )

    # Transform broadcasters
    transform_nodes = [
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='master_left_transform_broadcaster',
            arguments=['0', '-0.25', '0', '0', '0', '0', '/world', f'/{LaunchConfiguration("robot_name_master_left")}/base_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='master_right_transform_broadcaster',
            arguments=['0', '-0.25', '0', '0', '0', '0', '/world', f'/{LaunchConfiguration("robot_name_master_right")}/base_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='puppet_left_transform_broadcaster',
            arguments=['0', '0.25', '0', '0', '0', '0', '/world', f'/{LaunchConfiguration("robot_name_puppet_left")}/base_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='puppet_right_transform_broadcaster',
            arguments=['0', '0.25', '0', '0', '0', '0', '/world', f'/{LaunchConfiguration("robot_name_puppet_right")}/base_link']
        )
    ]

    # Realsense camera nodes 
    camera_package = 'snn_aloha' 
    
    realsense_nodes = [
        # Main realsense publisher
        Node(
            package=camera_package,
            executable='realsense_publisher.py', 
            name='realsense_publisher',
            output='screen',
            respawn=True
        )
    ]

    # Generate realsense_publisher_0 through realsense_publisher_3 dynamically
   

    # Ensure these imports exist at the very top of your file
    # from launch.actions import TimerAction
    # from launch_ros.actions import Node

    realsense_nodes = []
    camera_package = 'snn_aloha'  # Double-check this matches your package name

    for i in range(4):
        camera_node = Node(
            package=camera_package,
            executable='realsense_publisher.py',  # Ensure .py matches your installation choice
            name=f'realsense_publisher_{i}',
            arguments=[str(i)],
            output='screen',
            respawn=True
        )
        
        # Stagger the camera node execution to protect the USB bandwidth limits
        staggered_camera = TimerAction(
            period=float(i * 2.0),
            actions=[camera_node]
        )
        
        realsense_nodes.append(staggered_camera)
    return LaunchDescription(declare_arguments + includes + transform_nodes + realsense_nodes)