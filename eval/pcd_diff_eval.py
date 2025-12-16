#!/usr/bin/env python3
"""Unified ROS2 diffusion policy evaluation script.

Supports three control modes:
- simple: Sequential action execution without buffering
- chunking: Action buffering with policy_delay offset
- blending: Action blending in overlap regions

Also supports inspection mode:
- --vis-pcd: Visualize observations without running policy or actions

Debug mode:
- --direct-policy: Use policy directly without server (requires --ckpt-path)

Examples:
python eval/pcd_diff_eval.py --mode simple --n-steps 9999
python eval/pcd_diff_eval.py --mode simple --n-steps 60 --debug-plotting
python eval/pcd_diff_eval.py --vis-pcd  # Inspection mode
python eval/pcd_diff_eval.py --mode simple --n-steps 60 --direct-policy --ckpt-path /path/to/checkpoint.ckpt  # Debug mode
"""

import argparse
import sys
import os
import time
import threading
import rclpy

import numpy as np

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
from diff_eval_utils.diffusion_clients import PolicyClient, PcdProcessingClient, DirectPolicyWrapper
from diff_eval_utils.diffusion_controllers import create_controller
from diff_eval_utils.ros_utils import JointStateSubscriber
from diff_eval_utils.diffusion_visualization import visualize_robot_pcd, np2o3d


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

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Enable visualization of observations (images and point clouds)'
    )

    parser.add_argument(
        '--vis-pcd',
        action='store_true',
        help='Point cloud inspection mode: visualize observations without running policy or actions'
    )

    parser.add_argument(
        '--direct-policy',
        action='store_true',
        help='Debug mode: use policy directly without server (requires --ckpt-path)'
    )

    parser.add_argument(
        '--ckpt-path',
        type=str,
        default=None,
        help='Path to policy checkpoint (required when using --direct-policy)'
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
    print(f"Loading point cloud manager with config: {config_path}")
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


def pcd_inspect(config: DPEvalConfig):
    """Point cloud inspection mode: visualize observations without running policy.

    Args:
        config: DPEvalConfig instance with visualize=True
    """
    print("\n" + "="*60)
    print("POINT CLOUD INSPECTION MODE")
    print("="*60)
    print("Visualizing observations without running policy or actions")
    print("Press Ctrl+C to exit")
    print("="*60 + "\n")

    # Setup hardware (robot for joint states only, no movement)
    print("Initializing robot (read-only for joint states)...")
    robot = Robot(namespace="")
    robot.wait_until_ready()
    print(f"Robot ready. Joint values: {robot.joint_values}")
    print(f"End effector pose: {robot.end_effector_pose}")

    robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
    robot.cartesian_controller_parameters_client.load_param_config(
        file_path="config/control/spacemouse_cartesian_impedance.yaml"
    )
    time.sleep(2.0)

    # Setup sensors
    print("\nInitializing sensors...")
    print(config.toolbox_path)
    manager, joint_state_subscriber, _ = setup_point_cloud_manager(config.toolbox_path)

    # Setup point cloud processing client
    print(f"\nConnecting to PCD processing server at localhost:{config.pcd_server_port}...")
    pcd_client = PcdProcessingClient(f"http://localhost:{config.pcd_server_port}")

    # Create rate for observation loop
    rate = robot.node.create_rate(1.0)  # 1 Hz for inspection

    robot.move_to(position=[0.6, 0.0, 0.25], speed=0.1)

    print("\n" + "="*60)
    print("Starting observation visualization loop...")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")

    step = 0
    try:
        while True:
            print(f"\n--- Observation {step} ---")

            # Get current joint positions and robot pose
            joint_positions = joint_state_subscriber.joint_values
            eef_pose_raw = robot.end_effector_pose
            from diff_eval_utils.diffusion_transforms import get_pose_from_robot
            eef_pose = get_pose_from_robot(eef_pose_raw)

            print(f"Joint positions: {joint_positions[:3]}...")  # Print first 3 for brevity
            print(f"EEF position: {eef_pose[:3]}")

            # Get raw point cloud
            print("Capturing point cloud...")
            raw_pcd = manager.get_latest_pointcloud()
            print(f"Raw point cloud shape: {raw_pcd.shape if raw_pcd is not None else 'None'}")

            # Get RGB-D images
            print("Capturing RGB-D images...")
            inhand_cam = 'cam4'
            rgb, depth = manager.get_latest_rgbd(inhand_cam)
            print(f"RGB shape: {rgb.shape if rgb is not None else 'None'}")
            print(f"Depth shape: {depth.shape if depth is not None else 'None'}")
            
            # Process point cloud with visualization
            if raw_pcd is not None:
                print("Processing point cloud...")
                processed_pcd, render_pcd = pcd_client.process_pcd(raw_pcd, eef_pose, joint_positions)
                print(f"Processed PCD shape: {processed_pcd.shape}")
                print(f"Render PCD shape: {render_pcd.shape}")

                # Process images with visualization
                rgb_dict = {inhand_cam: rgb}
                depth_dict = {inhand_cam: depth}
                processed_rgb_dict, processed_depth_dict = pcd_client.process_images(rgb_dict, depth_dict)

                # Visualize if requested
                if config.visualize:
                    import matplotlib.pyplot as plt
                    import open3d as o3d

                    # Visualize RGB image
                    plt.figure(figsize=(8, 6))
                    plt.imshow(processed_rgb_dict[inhand_cam].astype(np.uint8))
                    plt.title(f"Step {step}: {inhand_cam} RGB")
                    plt.axis('off')
                    plt.show()

                    # Visualize point cloud
                    pcd_vis = o3d.geometry.PointCloud()
                    pcd_vis.points = o3d.utility.Vector3dVector(processed_pcd[:, :3])
                    pcd_vis.colors = o3d.utility.Vector3dVector(processed_pcd[:, 3:])
                    
                    robot_pcd = visualize_robot_pcd(processed_pcd, None, joint_state=joint_positions[1:])
                    # robot_pcd = np2o3d(robot_pcd)
                    o3d.visualization.draw_geometries([pcd_vis], window_name=f"Step {step}: Point Cloud")

            step += 1
            rate.sleep()

    except KeyboardInterrupt:
        print("\n\nInspection stopped by user")
    finally:
        print("\nCleaning up...")
        manager.destroy_node()
        robot.shutdown()


def main():
    """Main entry point."""
    args = parse_args()
    config = DPEvalConfig.from_cli_args(args)

    # Validate arguments
    if args.direct_policy and not args.ckpt_path:
        print("Error: --ckpt-path is required when using --direct-policy")
        sys.exit(1)

    # Initialize ROS2
    rclpy.init()

    try:
        # Check if in inspection mode
        if args.vis_pcd:
            # Force visualization on in inspection mode
            config.visualize = True
            pcd_inspect(config)
            return

        # Normal control mode
        print("="*60)
        print(f"Diffusion Policy Control - Mode: {config.mode.upper()}")
        print("="*60)
        print(f"Steps: {config.n_steps}")
        print(f"Control frequency: {config.ctrl_freq} Hz")
        print(f"Debug plotting: {config.debug_plotting}")
        print(f"Visualize observations: {config.visualize}")
        if args.direct_policy:
            print(f"Policy mode: DIRECT (debug mode)")
            print(f"Checkpoint: {args.ckpt_path}")
        else:
            print(f"Policy mode: SERVER")
            print(f"Policy server: localhost:{config.policy_server_port}")
        print(f"PCD server: localhost:{config.pcd_server_port}")
        print("="*60)

        # Setup policy client
        print("\nInitializing policy...")
        if args.direct_policy:
            policy_client = DirectPolicyWrapper(args.ckpt_path)
        else:
            print("Connecting to policy server...")
            policy_client = PolicyClient(f"http://localhost:{config.policy_server_port}")
        # pcd_client = PcdProcessingClient(f"http://localhost:{config.pcd_server_port}")
        rotation_transformer = RotationTransformer(from_rep='rotation_6d', to_rep='matrix')

        # Setup hardware
        print("\nInitializing gripper...")
        gripper = setup_gripper()

        print("\nInitializing robot...")
        robot = setup_robot(config)

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
            # pcd_client=pcd_client,
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
