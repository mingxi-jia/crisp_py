#!/usr/bin/env python3
"""Unified ROS2 diffusion policy evaluation script.

Supports three control modes:
- simple: Sequential action execution without buffering
- chunking: Action buffering with policy_delay offset
- blending: Action blending in overlap regions
"""

import argparse
import sys
import os
import time
import threading
import rclpy

from crisp_py.camera.pointcloud import PointCloudManager
from crisp_py.robot import Robot
from crisp_py.gripper.gripper import Gripper, GripperConfig

# Add external dependencies
sys.path.append('/home/mingxi/mingxi_ws/handpi/robot-vision-toolbox')
sys.path.append('/home/mingxi/mingxi_ws/handpi/diffusion_policy')

from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from robot_filter.arm_segmentor import RobotArmSegmentation

# Import our utilities
from diff_eval_utils.diffusion_constants import DPEvalConfig, START_POSITION
from diff_eval_utils.diffusion_clients import PolicyClient, PcdProcessingClient
from diff_eval_utils.diffusion_controllers import create_controller
from diff_eval_utils.ros_utils import JointStateSubscriber


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified diffusion policy evaluation script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['simple', 'chunking', 'blending'],
        default='simple',
        help='Control mode: simple (sequential), chunking (buffered), or blending (merged)'
    )

    parser.add_argument(
        '--n-steps',
        type=int,
        default=40,
        help='Number of control steps to execute'
    )

    parser.add_argument(
        '--ctrl-freq',
        type=float,
        default=10.0,
        help='Control frequency in Hz'
    )

    parser.add_argument(
        '--debug-plotting',
        action='store_true',
        help='Enable debug plotting (only for chunking/blending modes)'
    )

    parser.add_argument(
        '--policy-port',
        type=int,
        default=5000,
        help='Policy server port'
    )

    parser.add_argument(
        '--pcd-port',
        type=int,
        default=5001,
        help='Point cloud processing server port'
    )

    return parser.parse_args()


def setup_robot(config: DPEvalConfig):
    """Initialize robot and move to home position.

    Args:
        config: DPEvalConfig instance

    Returns:
        Initialized Robot instance
    """
    robot = Robot(namespace="")
    robot.wait_until_ready()
    print(f"Robot ready. Joint values: {robot.joint_values}")
    print(f"End effector pose: {robot.end_effector_pose}")

    print("Going to home position...")
    robot.home()

    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path="config/control/default_cartesian_impedance.yaml"
    )

    print("Moving to start position...")
    robot.move_to(position=START_POSITION, speed=0.15)

    return robot


def setup_gripper():
    """Initialize gripper.

    Returns:
        Initialized Gripper instance
    """
    gripper_config = GripperConfig.from_yaml("./config/gripper_right.yaml")
    gripper = Gripper(gripper_config=gripper_config, namespace="/right/gripper")
    gripper.wait_until_ready()
    gripper.set_target(1.0)
    print("Gripper ready")

    return gripper


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
    """Main entry point."""
    args = parse_args()
    config = DPEvalConfig.from_cli_args(args)

    print("="*60)
    print(f"Diffusion Policy Control - Mode: {config.mode.upper()}")
    print("="*60)
    print(f"Steps: {config.n_steps}")
    print(f"Control frequency: {config.ctrl_freq} Hz")
    print(f"Debug plotting: {config.debug_plotting}")
    print(f"Policy server: localhost:{config.policy_server_port}")
    print(f"PCD server: localhost:{config.pcd_server_port}")
    print("="*60)

    # Initialize ROS2
    rclpy.init()

    try:
        # Setup clients
        print("\nConnecting to servers...")
        policy_client = PolicyClient(f"http://localhost:{config.policy_server_port}")
        pcd_client = PcdProcessingClient(f"http://localhost:{config.pcd_server_port}")
        rotation_transformer = RotationTransformer(from_rep='rotation_6d', to_rep='matrix')

        # Setup hardware
        print("\nInitializing robot...")
        robot = setup_robot(config)

        print("\nInitializing gripper...")
        gripper = setup_gripper()

        print("\nInitializing sensors...")
        manager, joint_state_subscriber, _ = setup_point_cloud_manager(config.toolbox_path)

        # Create controller
        print(f"\nCreating {config.mode} controller...")
        controller = create_controller(
            mode=config.mode,
            robot=robot,
            gripper=gripper,
            obs_manager=manager,
            joint_state_subscriber=joint_state_subscriber,
            policy_client=policy_client,
            pcd_client=pcd_client,
            rotation_transformer=rotation_transformer,
            config=config,
            ctrl_freq=config.ctrl_freq,
        )

        # Run controller
        print("\n" + "="*60)
        print("STARTING CONTROL LOOP")
        print("="*60 + "\n")
        controller.run(n_steps=config.n_steps)

        # Cleanup
        print("\n" + "="*60)
        print("Control loop completed. Cleaning up...")
        print("="*60)
        robot.home()
        robot.shutdown()
        manager.destroy_node()

    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt received, shutting down...")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
