from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory("apoptotic_loader")
    cfg = os.path.join(pkg, "config", "default.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("ttl_hours",    default_value="24"),
        DeclareLaunchArgument("kl_threshold", default_value="0.05"),
        DeclareLaunchArgument("stop_type",    default_value="velocity_ramp"),
        Node(package="apoptotic_loader", executable="checkpoint_registry",  name="checkpoint_registry",  output="screen", parameters=[cfg]),
        Node(package="apoptotic_loader", executable="apoptotic_manager",    name="apoptotic_manager",    output="screen", parameters=[cfg]),
        Node(package="apoptotic_loader", executable="drift_observer",       name="drift_observer",       output="screen", parameters=[cfg]),
        Node(package="apoptotic_loader", executable="safe_stop_controller", name="safe_stop_controller", output="screen", parameters=[cfg]),
    ])
