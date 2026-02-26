#!/usr/bin/env python3
"""Unified ROS2 diffusion policy evaluation script.

Supports six control modes:
- simple: Sequential action execution without buffering
- chunking: Action buffering with policy_delay offset
- blending: Action blending in overlap regions
- intv: Recording control with keyboard ('x' to reset, 'r' to toggle recording)
- controlnet: Sequential control with intervention detection and re-inference
- gello: Joint control from gello_control_signal ROS2 topic

Also supports inspection mode:
- --vis-pcd: Visualize observations without running policy or actions

Debug mode:
- --direct-policy: Use policy directly without server (requires --ckpt-path)

Examples:
python eval/eval.py --mode simple --ctrl-space cartesian --n-steps 9999
python eval/eval.py --mode simple --ctrl-space joint --n-steps 60 --debug-plotting
python eval/eval.py --mode controlnet --ctrl-space cartesian --n-steps 100  # ControlNet mode
python eval/eval.py --mode intv --ctrl-space cartesian  # Recording mode (no n-steps needed)
python eval/eval.py --mode gello --ctrl-space joint  # Gello joint control mode (no n-steps needed)
python eval/eval.py --vis-pcd --ctrl-space cartesian  # Inspection mode
python eval/eval.py --mode simple --ctrl-space cartesian --n-steps 60 --direct-policy --ckpt-path /path/to/ckpt  # Debug mode
"""

import argparse
import sys
import os
import time
import threading
import rclpy

import numpy as np

from crisp_py.camera.pointcloud import PointCloudManager
from crisp_py.robot import Robot, Pose
from crisp_py.robot_config import FrankaConfig
from crisp_py.gripper.gripper import Gripper, GripperConfig

from scipy.spatial.transform import Rotation as R


# Import our utilities
from diff_eval_utils.diffusion_constants import DPEvalConfig, START_POSITION, ROBOTIQ_ROTATION_OFFSET
from diff_eval_utils.diffusion_clients import PolicyClient, PcdProcessingClient, DirectPolicyWrapper, PinkIKClient
from diff_eval_utils.diffusion_controllers import create_controller
from diff_eval_utils.ros_utils import JointStateSubscriber
# from diff_eval_utils.diffusion_visualization import visualize_robot_pcd, np2o3d


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Unified diffusion policy evaluation script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['simple', 'chunking', 'blending', 'intv', 'controlnet', 'teleop', 'gello', 'test', 'test_teleop'],
        default='simple',
        help='Control mode: simple (sequential), chunking (buffered), blending (merged), intv (recording control), controlnet (intervention detection with re-inference), teleop (spacemouse teleoperation only), gello (joint control from gello_control_signal topic), test (impedance tracking test), or test_teleop (teleop with tracking plots)'
    )

    parser.add_argument(
        '--n-steps',
        type=int,
        default=40,
        help='Number of control steps to execute'
    )
    
    parser.add_argument(
        '--debug-plotting',
        action='store_true',
        help='Enable debug plotting (only for chunking/blending modes)'
    )

    parser.add_argument(
        '--policy-port',
        type=int,
        default=6666,
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

    parser.add_argument(
        '--img-policy',
        action='store_true',
        help='Use image-only policy (excludes pcd, depth, joint_pos from observations)'
    )

    parser.add_argument(
        '--ctrl-space',
        type=str,
        choices=['joint', 'cartesian'],
        required=True,
        help='Control space: joint (uses IK + joint impedance) or cartesian (Cartesian impedance)'
    )

    parser.add_argument(
        '--ik-port',
        type=int,
        default=5002,
        help='Pink IK server port (only used when --ctrl-space joint)'
    )

    return parser.parse_args()

class MyRobot(Robot):
    def __init__(self, gripper, **kwargs):
        super().__init__(**kwargs)
        self.my_gripper = gripper
 
    def home(self):
        super().home()
        self.my_gripper.set_target(1.0)

        # self.controller_switcher_client.switch_controller("cartesian_impedance_controller")
        # self.cartesian_controller_parameters_client.load_param_config(
        #     file_path="config/control/default_cartesian_impedance.yaml"
        # )
        # init_pose = Pose(
        #     position=START_POSITION,
        #     orientation=R.from_euler('XYZ',  np.array([np.pi, 0, -np.pi / 4]) ),
        # )
        # self.move_to(pose=init_pose, speed=0.05)
        # time.sleep(10)


def setup_robot(config: DPEvalConfig):
    """Initialize robot and move to home position.

    Args:
        config: DPEvalConfig instance

    Returns:
        Initialized Robot instance
    """
    gripper_config = GripperConfig.from_yaml("./config/gripper_robotiq.yaml")
    gripper = Gripper(gripper_config=gripper_config)
    gripper.wait_until_ready()
    gripper.set_target(1.0)  # Open gripper

    config_franka = FrankaConfig()
    config_franka.home_config = config.home_joint_position
    robot = MyRobot(gripper=gripper, namespace="", robot_config=config_franka)
    robot.wait_until_ready()
    print(f"Robot ready. Joint values: {robot.joint_values}")
    print(f"End effector pose: {robot.end_effector_pose}")

    print("Going to home position...")
    robot.home()
    if config.ctrl_space == 'joint':
        robot.controller_switcher_client.switch_controller("joint_impedance_controller")
        robot.joint_controller_parameters_client.load_param_config(
            file_path="config/control/joint_impedance_controller.yaml"
        )
    else:
        robot.controller_switcher_client.switch_controller("cartesian_impedance_controller")
        robot.cartesian_controller_parameters_client.load_param_config(
            file_path="config/control/default_cartesian_impedance.yaml"
        )

    return robot, gripper

def setup_point_cloud_manager(toolbox_path: str):
    """Initialize point cloud manager and joint state subscriber.

    Args:
        toolbox_path: Path to robot-vision-toolbox

    Returns:
        Tuple of (PointCloudManager, JointStateSubscriber, spin_thread)
    """
    config_path = os.path.join(toolbox_path, "robot_configs", "camera_info.yaml")
    print(f"Loading point cloud manager with config: {config_path}")
    manager = PointCloudManager(config_path)

    joint_state_subscriber = JointStateSubscriber(manager, topic="/joint_states")

    # Spin in background thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(manager,), daemon=True)
    spin_thread.start()

    # Wait for joint states
    while not joint_state_subscriber.is_ready:
        time.sleep(0.1)

    return manager, joint_state_subscriber, spin_thread


def main():
    """Main entry point."""
    args = parse_args()
    config = DPEvalConfig.from_cli_args(args)
    sys.path.append(config.toolbox_path)
    sys.path.append(config.diffusion_policy_path)

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
        if config.mode not in ['intv', 'teleop', 'gello', 'test_teleop']:
            print(f"Steps: {config.n_steps}")
        elif config.mode == 'intv':
            print(f"Mode: Recording control (runs until Ctrl+C)")
        elif config.mode == 'gello':
            print(f"Mode: Gello joint control (runs until Ctrl+C)")
        elif config.mode == 'test_teleop':
            print(f"Mode: Test teleoperation with tracking (runs until Ctrl+C)")
        else:
            print(f"Mode: Teleoperation only (runs until Ctrl+C)")
        print(f"Control frequency: {config.ctrl_freq} Hz")
        print(f"Control space: {config.ctrl_space}")
        print(f"Debug plotting: {config.debug_plotting}")
        print(f"Visualize observations: {config.visualize}")
        if config.mode in ['teleop', 'gello', 'test_teleop']:
            print(f"Policy mode: NONE ({config.mode} only)")
        elif args.direct_policy:
            print(f"Policy mode: DIRECT (debug mode)")
            print(f"Checkpoint: {args.ckpt_path}")
        else:
            print(f"Policy mode: SERVER")
            print(f"Policy server: localhost:{config.policy_server_port}")
        print(f"PCD server: localhost:{config.pcd_server_port}")
        if config.ctrl_space == 'joint':
            print(f"IK server: localhost:{config.ik_server_port}")
        print("="*60)

        # Setup IK client (only for joint control space)
        ik_client = None
        if config.ctrl_space == 'joint':
            print("\nConnecting to Pink IK server...")
            ik_client = PinkIKClient(f"http://localhost:{config.ik_server_port}")

        # Setup policy client (None for teleop/gello/test_teleop mode)
        if config.mode in ['teleop', 'gello', 'test_teleop']:
            print(f"\n{config.mode.replace('_', ' ').capitalize()} mode: No policy client needed")
            policy_client = None
        elif args.direct_policy:
            print("\nInitializing policy...")
            policy_client = DirectPolicyWrapper(args.ckpt_path, img_policy=config.img_policy)
        else:
            print("\nInitializing policy...")
            print("Connecting to policy server...")
            policy_client = PolicyClient(f"http://localhost:{config.policy_server_port}", img_policy=config.img_policy)
        # pcd_client = PcdProcessingClient(f"http://localhost:{config.pcd_server_port}")

        # Setup hardware
        print("\nInitializing robot...")
        robot, gripper = setup_robot(config)

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
            ik_client=ik_client,
            config=config,
            ctrl_freq=config.ctrl_freq,
        )

        # Run controller
        print("\n" + "="*60)
        print("STARTING CONTROL LOOP")
        print("="*60 + "\n")
        if config.mode in ['intv', 'teleop', 'gello', 'test_teleop']:
            # Recording/teleop/gello/test_teleop mode - runs indefinitely until Ctrl+C
            controller.run()
        else:
            # Normal modes - run for n_steps
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
