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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

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
        choices=['simple', 'chunking', 'blending', 'intv', 'intv_party', 'controlnet', 'teleop', 'teleop_intv', 'gello', 'test', 'test_teleop'],
        default='simple',
        help='Control mode: simple (sequential), chunking (buffered), blending (merged), intv (recording control), intv_party (party host intervention control), controlnet (intervention detection with re-inference), teleop (spacemouse teleoperation only), teleop_intv (spacemouse teleoperation with async intervention monitor), gello (joint control from gello_control_signal topic), test (impedance tracking test), or test_teleop (teleop with tracking plots)'
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
        '--ckpt-path',
        type=str,
        default=None,
        help='Path to policy checkpoint'
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

    parser.add_argument(
        '--no-home',
        action='store_true',
        help='Never call robot.home() (skips homing during setup and after control loop)'
    )

    parser.add_argument(
        '--no-reset',
        action='store_true',
        help='Skip robot.home() during setup_robot initialization only'
    )

    parser.add_argument(
        '--flip-gripper-obs',
        action='store_true',
        help='In get_observation, override gripper_qpos with self.prev_grasp_value (ignore is_contact flip)'
    )

    parser.add_argument(
        '--home-pos',
        type=int,
        default=None,
        help='Home position index: None (default) uses home_joint_position, 1 uses home_joint_position_1'
    )

    freq_override_group = parser.add_mutually_exclusive_group()
    freq_override_group.add_argument(
        '--intv_freq',
        action='store_true',
        help='Force policy actions to use n_steps_per_unit_intv_action regardless of intervention prediction'
    )
    freq_override_group.add_argument(
        '--hand_freq',
        action='store_true',
        help='Force policy actions to use n_steps_per_unit_hand_action regardless of intervention prediction'
    )

    return parser.parse_args()

class MyRobot(Robot):
    def __init__(self, gripper, no_home: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.my_gripper = gripper
        self.no_home = no_home
        self.ctrl_controller = None  # set after controller is created

        self.home_pub = self.node.create_publisher(
            JointTrajectory,
            '/joint_trajectory_controller/joint_trajectory',
            10,
        )
 
    def home(self):
        if self.no_home:
            print("Skipping home() (--no-home set)")
            return

        # Lift 0.05 m along z before homing (only when controller is available)
        if self.ctrl_controller is not None:
            print("Lifting 0.05 m along z before home...")
            prev_n_interp = self.ctrl_controller.n_interpolation
            self.ctrl_controller.n_interpolation = 100
            current_pose = self.end_effector_pose
            lift_pos = np.array(current_pose.position) + np.array([0.0, 0.0, 0.05])
            self.ctrl_controller._execute_joint_action(lift_pos, current_pose.orientation)
            # time.sleep(0.5)
            self.ctrl_controller.n_interpolation = prev_n_interp
        
        self.controller_switcher_client.switch_controller("joint_trajectory_controller")

        

        msg = JointTrajectory()
        msg.joint_names = self.config.joint_names
        point = JointTrajectoryPoint()
        point.positions = list(self.config.home_config)
        point.time_from_start = Duration(sec=5, nanosec=0)
        msg.points = [point]

        # Wait for the controller to subscribe, then publish
        while self.home_pub.get_subscription_count() == 0:
            time.sleep(0.05)
        self.home_pub.publish(msg)
        print("Published home position, waiting for robot to reach it...")


        time.sleep(5.5)
        self._target_pose = None
        self._target_joint = None
        self.wait_until_ready()
        print("ready")

        self.my_gripper.set_target(1.0)
        self.prev_grasp_value = 0.0
        # self.controller_switcher_client.switch_controller("cartesian_impedance_controller")
        # self.cartesian_controller_parameters_client.load_param_config(
        #     file_path="config/control/default_cartesian_impedance.yaml"
        # )
        # init_pose = Pose(
        #     position=START_POSITION,
        #     orientation=R.from_euler('XYZ',  np.array([np.pi + np.pi / 12, 0, -np.pi / 4]) ),
        # )
        # self.move_to(pose=init_pose, speed=0.05)
        # time.sleep(10)


def setup_robot(config: DPEvalConfig, no_home: bool = False, no_reset: bool = False, home_pos: int = None):
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

    home_pos_map = {
        None: config.home_joint_position,
        1: config.home_joint_position_1,
    }
    if home_pos not in home_pos_map:
        raise ValueError(f"Unknown --home-pos value: {home_pos}. Valid values: {list(home_pos_map.keys())}")
    config_franka = FrankaConfig()
    config_franka.home_config = home_pos_map[home_pos]
    robot = MyRobot(gripper=gripper, no_home=no_home, namespace="", robot_config=config_franka)
    robot.wait_until_ready()
    if not (no_home or no_reset):
        print("Going to home position...")
        robot.home()
    else:
        print("Skipping home position (--no-home or --no-reset set)")
    print(f"Robot ready. Joint values: {robot.joint_values}")
    print(f"End effector pose: {robot.end_effector_pose}")
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

def setup_point_cloud_manager(toolbox_path: str, config: DPEvalConfig = None):
    """Initialize point cloud manager and joint state subscriber.

    Args:
        toolbox_path: Path to robot-vision-toolbox
        config: DPEvalConfig instance (used to pass ft_sensor_on to JointStateSubscriber)

    Returns:
        Tuple of (PointCloudManager, JointStateSubscriber, spin_thread)
    """
    ft_sensor_on = config.ft_sensor_on if config is not None else True
    config_path = os.path.join(toolbox_path, "robot_configs", "camera_info.yaml")
    print(f"Loading point cloud manager with config: {config_path}")
    manager = PointCloudManager(config_path)

    joint_state_subscriber = JointStateSubscriber(manager, topic="/joint_states", ft_sensor_on=ft_sensor_on)

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
        if config.mode not in ['intv', 'intv_party', 'teleop', 'teleop_intv', 'gello', 'test_teleop']:
            print(f"Steps: {config.n_steps}")
        elif config.mode == 'intv':
            print(f"Mode: Recording control (runs until Ctrl+C)")
        elif config.mode == 'intv_party':
            print(f"Mode: Party host intervention control (runs until Ctrl+C)")
        elif config.mode == 'gello':
            print(f"Mode: Gello joint control (runs until Ctrl+C)")
        elif config.mode == 'test_teleop':
            print(f"Mode: Test teleoperation with tracking (runs until Ctrl+C)")
        else:
            print(f"Mode: Teleoperation only (runs until Ctrl+C)")
        print(f"Control frequency: {config.ctrl_freq} Hz")
        print(f"Policy action frequency override: {config.policy_action_freq_override or 'auto'}")
        print(f"Control space: {config.ctrl_space}")
        print(f"Debug plotting: {config.debug_plotting}")
        print(f"Visualize observations: {config.visualize}")
        if config.mode in ['teleop', 'gello', 'test_teleop']:
            print(f"Policy mode: NONE ({config.mode} only)")
        elif args.ckpt_path is not None:
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
        elif args.ckpt_path is not None:
            print("\nInitializing policy...")
            policy_client = DirectPolicyWrapper(args.ckpt_path, img_policy=config.img_policy)
        else:
            print("\nInitializing policy...")
            print("Connecting to policy server...")
            policy_client = PolicyClient(f"http://localhost:{config.policy_server_port}", img_policy=config.img_policy)
        # pcd_client = PcdProcessingClient(f"http://localhost:{config.pcd_server_port}")

        # Setup hardware
        print("\nInitializing robot...")
        robot, gripper = setup_robot(config, no_home=args.no_home, no_reset=args.no_reset, home_pos=args.home_pos)

        print("\nInitializing sensors...")
        manager, joint_state_subscriber, _ = setup_point_cloud_manager(config.toolbox_path, config=config)

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

        robot.ctrl_controller = controller

        # Warn if teleop_intv mode but server doesn't support intervention prediction
        if config.mode == 'teleop_intv' and not getattr(controller, 'predict_contact', False):
            print("\nWARNING: teleop_intv mode selected but policy server does not support")
            print("         intervention prediction (predict_contact=False).")
            print("         Intervention monitor will show N/A. Use a checkpoint trained")
            print("         with predict_contact enabled for intervention predictions.\n")

        # Run controller
        print("\n" + "="*60)
        print("STARTING CONTROL LOOP")
        print("="*60 + "\n")
        controller._pre_run()
        if config.mode in ['intv', 'intv_party', 'teleop', 'teleop_intv', 'gello', 'test_teleop']:
            # Recording/teleop/gello/test_teleop mode - runs indefinitely until Ctrl+C
            controller.run()
        else:
            # Normal modes - run for n_steps
            controller.run(n_steps=config.n_steps)

        # Cleanup
        print("\n" + "="*60)
        print("Control loop completed. Cleaning up...")
        print("="*60)
        if not args.no_home:
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
