#!/usr/bin/env python3
"""Collect observations from a grid of workspace positions for policy visualization.

This script moves the robot to a grid of positions in the workspace and collects
observations (point clouds, joint states, etc.) at each position. The data is saved
for offline policy trajectory visualization.

Example usage:
    python collect_workspace_obs.py \
        --x_range 0.3 0.7 \
        --y_range -0.3 0.3 \
        --z_range 0.1 0.5 \
        --spacing 0.04 \
        --output workspace_obs_data.pkl
"""

import argparse
import pickle
import time
import sys
import os
import threading
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy

from crisp_py.camera.pointcloud import PointCloudManager
from crisp_py.robot import Robot
from crisp_py.gripper.gripper import Gripper, GripperConfig

# Add external dependencies
sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')

from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from robotool.robot_filter.arm_segmentor import RobotArmSegmentation

# Import our utilities
from diff_eval_utils.diffusion_transforms import (
    get_pose_from_robot,
    franka_obs_to_diff_obs
)
from diff_eval_utils.diffusion_clients import PcdProcessingClient
from diff_eval_utils.ros_utils import JointStateSubscriber
from diff_eval_utils.diffusion_constants import START_POSITION


class WorkspaceDataCollector:
    """Collects observations from a grid of workspace positions."""

    def __init__(self, robot, gripper, obs_manager, joint_state_subscriber,
                 pcd_client, ctrl_freq=10.0):
        """Initialize the workspace data collector.

        Args:
            robot: Robot instance
            gripper: Gripper instance
            obs_manager: PointCloudManager instance
            joint_state_subscriber: JointStateSubscriber instance
            pcd_client: PcdProcessingClient instance
            ctrl_freq: Control frequency in Hz
        """
        self.robot = robot
        self.gripper = gripper
        self.obs_manager = obs_manager
        self.joint_state_subscriber = joint_state_subscriber
        self.pcd_client = pcd_client
        self.ctrl_freq = ctrl_freq

        self.target_pose = robot.end_effector_pose.copy()
        self.gripper_value = 0.0  # Open gripper

    def generate_grid_points(self, x_range, y_range, z_range, spacing):
        """Generate grid points in the workspace.

        Args:
            x_range: Tuple of (min_x, max_x)
            y_range: Tuple of (min_y, max_y)
            z_range: Tuple of (min_z, max_z)
            spacing: Distance between grid points in meters

        Returns:
            List of (x, y, z) tuples
        """
        x_points = np.arange(x_range[0], x_range[1] + spacing/2, spacing)
        y_points = np.arange(y_range[0], y_range[1] + spacing/2, spacing)
        z_points = np.arange(z_range[0], z_range[1] + spacing/2, spacing)

        grid_points = []
        for x in x_points:
            for y in y_points:
                for z in z_points:
                    grid_points.append((x, y, z))

        print(f"Generated {len(grid_points)} grid points")
        print(f"X: {len(x_points)} points from {x_range[0]} to {x_range[1]}")
        print(f"Y: {len(y_points)} points from {y_range[0]} to {y_range[1]}")
        print(f"Z: {len(z_points)} points from {z_range[0]} to {z_range[1]}")

        return grid_points

    def move_to_position(self, position):
        """Move robot end-effector to specified position.

        Args:
            position: Tuple of (x, y, z) in meters
        """
        self.target_pose.position = np.array(position)
        # Keep downward-facing orientation
        self.target_pose.orientation = R.from_euler('XYZ', [np.pi, 0, 0])

        self.robot.move_to(position=self.target_pose.position, speed=0.15)
        time.sleep(0.5)  # Wait for robot to stabilize

    def get_observation(self):
        """Get current observation from sensors.

        Returns:
            Dictionary containing observation data
        """
        joint_state = self.joint_state_subscriber.joint_values
        gripper_state = self.gripper_value

        time.sleep(0.05)  # Wait for robot stability

        eef_pose = get_pose_from_robot(self.robot.end_effector_pose)

        obs_dict = franka_obs_to_diff_obs(
            self.obs_manager, eef_pose, gripper_state,
            self.pcd_client, joint_state, visualize=False
        )

        return obs_dict

    def demonstrate_bounding_box(self, x_range, y_range, z_range):
        """Move robot between diagonal corners to show workspace region.

        Args:
            x_range: Tuple of (min_x, max_x)
            y_range: Tuple of (min_y, max_y)
            z_range: Tuple of (min_z, max_z)
        """
        print("\n" + "=" * 60)
        print("Demonstrating workspace bounding box...")
        print(f"  X: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
        print(f"  Y: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
        print(f"  Z: [{z_range[0]:.3f}, {z_range[1]:.3f}]")
        print("=" * 60)

        # Move between two diagonal corners
        # Top-left: max Z, min X, min Y
        top_left = (x_range[0], y_range[0], z_range[1])
        # Bottom-right: min Z, max X, max Y
        bottom_right = (x_range[1], y_range[1], z_range[0])

        print(f"\nMoving to top-left corner (min X, min Y, max Z)")
        print(f"  Position: ({top_left[0]:.3f}, {top_left[1]:.3f}, {top_left[2]:.3f})")
        self.move_to_position(top_left)
        time.sleep(1.0)

        print(f"\nMoving to bottom-right corner (max X, max Y, min Z)")
        print(f"  Position: ({bottom_right[0]:.3f}, {bottom_right[1]:.3f}, {bottom_right[2]:.3f})")
        self.move_to_position(bottom_right)
        time.sleep(1.0)

        print("\n" + "=" * 60)
        print("Bounding box demonstration complete!")
        print("=" * 60 + "\n")

    def collect_workspace_data(self, x_range, y_range, z_range, spacing):
        """Collect observations from grid of workspace positions.

        Args:
            x_range: Tuple of (min_x, max_x)
            y_range: Tuple of (min_y, max_y)
            z_range: Tuple of (min_z, max_z)
            spacing: Distance between grid points in meters

        Returns:
            List of dictionaries with 'position' and 'observation' keys
        """
        # First, demonstrate the bounding box
        self.demonstrate_bounding_box(x_range, y_range, z_range)

        grid_points = self.generate_grid_points(x_range, y_range, z_range, spacing)

        print("\nStarting workspace data collection...")
        print(f"Total points to visit: {len(grid_points)}")
        print("=" * 60)

        collected_data = []

        for idx, position in enumerate(grid_points):
            print(f"\n[{idx+1}/{len(grid_points)}] Moving to position: "
                  f"({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")

            try:
                # Move to position
                self.move_to_position(position)

                # Collect observation
                obs_dict = self.get_observation()

                # Store data
                data_point = {
                    'position': position,
                    'observation': obs_dict,
                    'timestamp': time.time()
                }
                collected_data.append(data_point)

                print(f"✓ Observation collected successfully")

            except Exception as e:
                print(f"✗ Error at position {position}: {e}")
                continue

        print("\n" + "=" * 60)
        print(f"Data collection complete! Collected {len(collected_data)}/{len(grid_points)} observations")

        return collected_data


def setup_point_cloud_manager(toolbox_path: str):
    """Initialize point cloud manager and joint state subscriber.

    Args:
        toolbox_path: Path to robot-vision-toolbox

    Returns:
        Tuple of (PointCloudManager, JointStateSubscriber, spin_thread)
    """
    config_path = os.path.join(toolbox_path, "configs", "camera_info.yaml")
    manager = PointCloudManager(config_path)

    # Get joint names from URDF
    temp_robot_seg = RobotArmSegmentation()
    joint_names = sorted([j.name for j in temp_robot_seg.robot_urdf.actuated_joints])
    joint_state_subscriber = JointStateSubscriber(manager, joint_names, topic="/joint_states")

    # Spin in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(manager,), daemon=True)
    spin_thread.start()

    # Wait for joint states
    while not joint_state_subscriber.is_ready:
        time.sleep(0.1)
    print(f"Joint state subscriber ready. Joint names: {joint_names}")

    return manager, joint_state_subscriber, spin_thread


def main():
    parser = argparse.ArgumentParser(
        description='Collect workspace observations for policy visualization'
    )
    parser.add_argument('--x_range', nargs=2, type=float, default=[0.55, 0.65],
                        help='X workspace range (min max) in meters')
    parser.add_argument('--y_range', nargs=2, type=float, default=[-0.1, 0.1],
                        help='Y workspace range (min max) in meters')
    parser.add_argument('--z_range', nargs=2, type=float, default=[0.24, 0.39],
                        help='Z workspace range (min max) in meters')
    parser.add_argument('--spacing', type=float, default=0.05,
                        help='Grid spacing in meters (default: 0.05)')
    parser.add_argument('--output', type=str, default='workspace_obs_data.pkl',
                        help='Output file path for collected data')
    parser.add_argument('--ctrl_freq', type=float, default=10.0,
                        help='Control frequency in Hz')
    parser.add_argument('--pcd-port', type=int, default=5001,
                        help='Point cloud processing server port')

    args = parser.parse_args()

    # Initialize ROS
    rclpy.init()

    # Initialize robot and sensors
    print("Initializing robot and sensors...")
    robot = Robot(namespace="")
    robot.wait_until_ready()
    print(f"Robot ready. Joint values: {robot.joint_values}")
    print("Going to home position...")
    robot.home()

    gripper_config = GripperConfig.from_yaml("./config/gripper_right.yaml")
    gripper = Gripper(gripper_config=gripper_config, namespace="/right/gripper")
    gripper.wait_until_ready()
    print("Gripper ready.")

    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path="config/control/default_cartesian_impedance.yaml"
    )

    print("Moving to start position...")
    robot.move_to(position=START_POSITION, speed=0.15)
    print(f"Robot ready at start position. Joint values: {robot.joint_values}")
    # [1.15398143e-03, 3.16230784e-02, 1.72399025e-03, -1.89068288e+00, 3.19050939e-03, 1.94383945e+00, 7.96259513e-01]

    # Setup point cloud manager and joint state subscriber
    print("\nInitializing sensors...")
    manager, joint_state_subscriber, _ = setup_point_cloud_manager(
        
    )

    # Setup PCD processing client
    print("\nConnecting to PCD processing server...")
    pcd_client = PcdProcessingClient(f"http://localhost:{args.pcd_port}")

    time.sleep(2.0)  # Wait for initialization

    # Create collector
    collector = WorkspaceDataCollector(
        robot=robot,
        gripper=gripper,
        obs_manager=manager,
        joint_state_subscriber=joint_state_subscriber,
        pcd_client=pcd_client,
        ctrl_freq=args.ctrl_freq
    )

    # Collect data
    collected_data = collector.collect_workspace_data(
        x_range=tuple(args.x_range),
        y_range=tuple(args.y_range),
        z_range=tuple(args.z_range),
        spacing=args.spacing
    )

    # Save data
    output_path = Path(args.output)
    print(f"\nSaving data to {output_path.absolute()}...")

    with open(output_path, 'wb') as f:
        pickle.dump({
            'data': collected_data,
            'metadata': {
                'x_range': args.x_range,
                'y_range': args.y_range,
                'z_range': args.z_range,
                'spacing': args.spacing,
                'n_points': len(collected_data),
                'collection_time': time.time()
            }
        }, f)

    print(f"✓ Data saved successfully!")
    print(f"\nNext step: Run visualization with:")
    print(f"  python visualize_policy_trajectories.py --input {args.output}")


if __name__ == '__main__':
    main()
