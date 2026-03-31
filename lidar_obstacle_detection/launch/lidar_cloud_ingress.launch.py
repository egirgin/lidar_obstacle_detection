"""
Launch: ``lidar_cloud_ingress`` (transform + FOV + voxel) + optional RViz.

Loads default parameters from ``config/lidar_cloud_ingress.yaml``; launch
argument ``verbose`` overrides the YAML value. Use ``ingress_params_file`` to
point at an alternate YAML. Pass ``headless:=true`` to skip RViz2.

Published cloud: ``/lidar_obstacle_detection/cloud_in_base``. Static TF
``base_link``→``lidar_link`` is for RViz only.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue


def generate_launch_description() -> LaunchDescription:
    """Wire ingress node + RViz; YAML params + optional ``verbose`` override."""
    pkg_share = get_package_share_directory('lidar_obstacle_detection')
    rviz_cfg = os.path.join(pkg_share, 'rviz', 'lidar_ingress.rviz')
    default_ingress_yaml = os.path.join(pkg_share, 'config', 'lidar_cloud_ingress.yaml')

    verbose_arg = DeclareLaunchArgument(
        'verbose',
        default_value='false',
        description='Extra INFO logs from the ingress node (overrides YAML)',
    )
    ingress_params_arg = DeclareLaunchArgument(
        'ingress_params_file',
        default_value=default_ingress_yaml,
        description='Path to lidar_cloud_ingress parameters YAML',
    )
    headless_arg = DeclareLaunchArgument(
        'headless',
        default_value='false',
        description='If true, do not start RViz2 (for robots / CI / SSH without display)',
    )

    ingress_node = Node(
        package='lidar_obstacle_detection',
        executable='lidar_cloud_ingress',
        name='lidar_cloud_ingress',
        output='screen',
        parameters=[
            ParameterFile(LaunchConfiguration('ingress_params_file'), allow_substs=False),
            {
                'verbose': ParameterValue(
                    LaunchConfiguration('verbose'),
                    value_type=bool,
                ),
            },
        ],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        condition=UnlessCondition(LaunchConfiguration('headless')),
    )

    return LaunchDescription(
        [
            verbose_arg,
            ingress_params_arg,
            headless_arg,
            ingress_node,
            rviz_node,
        ],
    )
