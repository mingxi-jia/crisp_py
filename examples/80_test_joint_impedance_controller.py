"""Test Pink IK solver with joint impedance control.

This script:
1. Loads trajectory from HDF5 (same as replay_hand_poses.py)
2. Sends poses to Pink IK server to get joint solutions (per-step IK)
3. Executes joint trajectory using joint_impedance_controller
4. Interpolates between poses with 4 steps
5. Plots trajectory tracking results

Usage:
    1. Start the Pink IK server (in pink_solver env):
       conda activate pink_solver
       python pink_ik_server.py --port 5002

    2. Run this script (in ROS2/crisp_py env):
       python 80_test_joint_impedance_controller.py
"""

import time
import requests
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys

from scipy.spatial.transform import Rotation as R

from crisp_py.robot import Robot
from crisp_py.robot_config import FrankaConfig

# Add path for diffusion policy imports
sys.path.append('/home/mingxi/mingxi_ws/crisp/crisp_py/eval_policy')
from diff_eval_utils.diffusion_constants import DPEvalConfig
from diff_eval_utils.diffusion_transforms import ten_d_action_to_pose, convert_action_from_fingertip_to_gripper, mat_to_rot6d

# Configuration
PINK_SERVER_URL = "http://localhost:5002"
HDF5_FILE = "/media/mingxi/T7/XEMB_Experiment/coffee_prep/replay_hand_test/test_2_smoothed.hdf5"
HDF5_FILE = "/media/mingxi/T7/XEMB_Experiment/nutella_sort/nutella_d1_59_realworld_pretrain.hdf5"
# NUM_INTERP_STEPS = 4  # Number of interpolation steps between poses


class PinkIKClient:
    """Client for the Pink IK server."""

    def __init__(self, server_url: str):
        self.server_url = server_url

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=2)
            return resp.status_code == 200
        except Exception:
            return False

    def solve_ik(self, position, orientation_quat, q_init=None):
        """Solve IK for a single pose.

        Args:
            position: [x, y, z]
            orientation_quat: [qx, qy, qz, qw]
            q_init: Initial joint configuration (optional)

        Returns:
            q: Joint configuration (7 DOF for Panda arm)
            success: Whether IK converged
            timing: Timing statistics from the server
        """
        payload = {
            "position": position.tolist() if isinstance(position, np.ndarray) else position,
            "orientation": orientation_quat.tolist() if isinstance(orientation_quat, np.ndarray) else orientation_quat,
        }
        if q_init is not None:
            payload["q_init"] = q_init.tolist() if isinstance(q_init, np.ndarray) else q_init

        resp = requests.post(f"{self.server_url}/solve_ik", json=payload)
        data = resp.json()
        return (
            np.array(data.get("q", [])),
            data.get("success", False),
            data.get("timing", {})
        )


def setup_robot(config: DPEvalConfig):
    """Initialize robot with joint impedance controller."""
    # Setup robot
    config_franka = FrankaConfig()
    config_franka.home_config = config.home_joint_position
    robot = Robot(namespace="", robot_config=config_franka)
    robot.wait_until_ready()

    print(f"Robot ready. Joint values: {robot.joint_values}")
    print(f"End effector pose: {robot.end_effector_pose}")

    # Home the robot first using joint trajectory controller
    print("Homing robot...")
    robot.home()

    # Switch to joint impedance controller for compliant motion
    print("Switching to joint_impedance_controller...")
    robot.controller_switcher_client.switch_controller("joint_impedance_controller")
    robot.joint_controller_parameters_client.load_param_config(
        "./config/control/joint_impedance_controller.yaml"
    )
    time.sleep(0.5)

    return robot


def load_trajectory_from_hdf5(file_path: str):
    """Load hand poses from HDF5 file."""
    dataset = h5py.File(file_path, "r")
    data = dataset['data']['demo_2']

    hand_pos = data['obs']['robot0_eef_pos'][:]
    hand_quat = data['obs']['robot0_eef_quat'][:]
    hand_grasp = data['obs']['robot0_gripper_qpos'][:, 0]

    hand_poses = np.hstack((hand_pos, hand_quat)).astype(float)

    assert len(hand_poses) == len(hand_grasp), f"Length mismatch: {len(hand_poses)} vs {len(hand_grasp)}"

    dataset.close()
    return hand_poses, hand_grasp


def convert_poses_to_gripper_frame(hand_poses):
    """Convert fingertip poses to gripper frame."""

    gripper_poses = []

    for hand_pose in hand_poses:
        # Build 10D action from hand pose
        action = np.concatenate([
            hand_pose[:3],
            mat_to_rot6d.forward(
                R.from_quat(hand_pose[3:7]).as_matrix().reshape(1, 3, 3).astype(np.float32)
            )[0],
            [0.0]  # grasp placeholder
        ])
        print(f"hand_pose[:3]: {hand_pose[:3]}")
        # Convert to Pose then to gripper frame
        pose, grasp = ten_d_action_to_pose(action, clip=False)
        gripper_pos, gripper_rotation = convert_action_from_fingertip_to_gripper(pose)

        gripper_quat = gripper_rotation.as_quat()  # [qx, qy, qz, qw]
        gripper_poses.append((gripper_pos, gripper_quat))

    return gripper_poses


def plot_trajectory_results(ts, ee_poses, target_poses):
    """Plot trajectory tracking results like figure_eight.py."""
    # Extract positions
    x_ee = [pose.position[0] for pose in ee_poses]
    y_ee = [pose.position[1] for pose in ee_poses]
    z_ee = [pose.position[2] for pose in ee_poses]

    x_t = [pose[0] for pose in target_poses]
    y_t = [pose[1] for pose in target_poses]
    z_t = [pose[2] for pose in target_poses]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # XY plot
    axes[0, 0].plot(x_ee, y_ee, label="actual", linewidth=2)
    axes[0, 0].plot(x_t, y_t, label="target", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("$x$ [m]")
    axes[0, 0].set_ylabel("$y$ [m]")
    axes[0, 0].set_title("XY Trajectory")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # XZ plot
    axes[0, 1].plot(x_ee, z_ee, label="actual", linewidth=2)
    axes[0, 1].plot(x_t, z_t, label="target", linestyle="--", linewidth=2)
    axes[0, 1].set_xlabel("$x$ [m]")
    axes[0, 1].set_ylabel("$z$ [m]")
    axes[0, 1].set_title("XZ Trajectory")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Position vs time
    axes[1, 0].plot(ts, x_ee, label="x actual", linewidth=2)
    axes[1, 0].plot(ts, x_t, label="x target", linestyle="--", linewidth=2)
    axes[1, 0].plot(ts, y_ee, label="y actual", linewidth=2)
    axes[1, 0].plot(ts, y_t, label="y target", linestyle="--", linewidth=2)
    axes[1, 0].plot(ts, z_ee, label="z actual", linewidth=2)
    axes[1, 0].plot(ts, z_t, label="z target", linestyle="--", linewidth=2)
    axes[1, 0].set_xlabel("$t$ [s]")
    axes[1, 0].set_ylabel("Position [m]")
    axes[1, 0].set_title("Position vs Time")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 3D trajectory
    ax3d = fig.add_subplot(2, 2, 4, projection='3d')
    ax3d.plot(x_ee, y_ee, z_ee, label="actual", linewidth=2)
    ax3d.plot(x_t, y_t, z_t, label="target", linestyle="--", linewidth=2)
    ax3d.set_xlabel("$x$ [m]")
    ax3d.set_ylabel("$y$ [m]")
    ax3d.set_zlabel("$z$ [m]")
    ax3d.set_title("3D Trajectory")
    ax3d.legend()

    # Remove the placeholder axes[1,1] since we replaced it with 3D
    axes[1, 1].remove()

    fig.tight_layout()
    plt.show()


def main():
    # Load config
    config = DPEvalConfig()
    CTRL_FREQ = config.joint_ctrl_freq
    NUM_INTERP_STEPS = config.n_interpolation

    # Initialize Pink IK client
    print("Connecting to Pink IK server...")
    ik_client = PinkIKClient(PINK_SERVER_URL)

    if not ik_client.health_check():
        print(f"ERROR: Pink IK server not available at {PINK_SERVER_URL}")
        print("Start it with: conda activate pink_solver && python pink_ik_server.py")
        return

    print("Pink IK server connected!")

    # Load trajectory from HDF5
    print(f"\nLoading trajectory from {HDF5_FILE}...")
    hand_poses, hand_grasp = load_trajectory_from_hdf5(HDF5_FILE)
    print(f"Loaded {len(hand_poses)} poses")

    # Convert to gripper frame
    print("Converting poses to gripper frame...")
    gripper_poses = convert_poses_to_gripper_frame(hand_poses)
    gripper_poses = gripper_poses[:]
    # Setup robot
    print("\nInitializing robot...")
    robot = setup_robot(config)

    # Get current joint state as initial guess for IK
    q_current = robot.joint_values.copy()

    # Create rate for timing
    rate = robot.node.create_rate(CTRL_FREQ)

    # Data collection for plotting
    ee_poses = []
    target_poses = []
    ts = []
    ik_timing_stats = []

    # Move to first pose
    print("\nSolving IK for first pose and moving there...")
    first_pos, first_quat = gripper_poses[0]
    q_first, success, timing = ik_client.solve_ik(first_pos, first_quat, q_current)
    ik_timing_stats.append(timing)
    print(f"First IK: success={success}, time={timing.get('total_time_ms', 0):.10f}ms, iters={timing.get('num_iterations', 0)}")

    if not success:
        print("WARNING: First IK did not converge!")

    # Interpolate from home to first pose
    print("Moving to initial pose...")
    q_start = q_current
    q_end = q_first
    num_init_steps = CTRL_FREQ*2  # More steps for initial movement

    for step in range(num_init_steps):
        alpha = (step + 1) / num_init_steps
        q_interp = (1 - alpha) * q_start + alpha * q_end
        robot.set_target_joint(q_interp)
        rate.sleep()

    q_current = q_first
    t = 0.0

    # Execute trajectory with per-step IK
    print(f"\nExecuting trajectory ({len(gripper_poses)} poses, {NUM_INTERP_STEPS} interp steps each)...")
    print("=" * 60)

    for i, (target_pos, target_quat) in enumerate(gripper_poses):
        # Solve IK for this pose
        print(f'target_pos: {target_pos}')
        # time.sleep(0.01)
        q_target, success, timing = ik_client.solve_ik(target_pos, target_quat, q_current)
        ik_timing_stats.append(timing)

        if i % 10 == 0:
            print(f"Step {i+1}/{len(gripper_poses)}: "
                  f"IK time={timing.get('total_time_ms', 0):.10f}ms, "
                  f"iters={timing.get('num_iterations', 0)}, "
                  f"success={success}")

        if not success:
            print(f"  WARNING: IK did not converge at step {i+1}")

        # Interpolate between current and target joint positions
        q_start = q_current
        q_end = q_target

        for step in range(NUM_INTERP_STEPS):
            alpha = (step + 1) / NUM_INTERP_STEPS
            q_interp = (1 - alpha) * q_start + alpha * q_end
            robot.set_target_joint(q_interp)
            rate.sleep()

            # Record data for plotting
            ee_poses.append(robot.end_effector_pose.copy())
            target_poses.append(target_pos.copy())
            ts.append(t)
            t += 1.0 / CTRL_FREQ

        q_current = q_target
        # q_current = robot.joint_values.copy()

    # Print timing statistics
    print("\n" + "=" * 60)
    print("IK Timing Statistics:")
    total_times = [s.get('total_time_ms', 0) for s in ik_timing_stats if s]
    solve_times = [s.get('solve_time_ms', 0) for s in ik_timing_stats if s]
    iterations = [s.get('num_iterations', 0) for s in ik_timing_stats if s]

    if total_times:
        print(f"  Total time: min={min(total_times):.4f}ms, max={max(total_times):.4f}ms, "
              f"avg={np.mean(total_times):.4f}ms")
        print(f"  Solve time: min={min(solve_times):.4f}ms, max={max(solve_times):.4f}ms, "
              f"avg={np.mean(solve_times):.4f}ms")
        print(f"  Iterations: min={min(iterations)}, max={max(iterations)}, "
              f"avg={np.mean(iterations):.1f}")

    # Let robot settle
    print("\nWaiting for robot to settle...")
    settle_time = 10.0
    settle_steps = int(settle_time * CTRL_FREQ)
    for _ in range(settle_steps):
        rate.sleep()
        ee_poses.append(robot.end_effector_pose.copy())
        target_poses.append(target_poses[-1] if target_poses else gripper_poses[-1][0])
        ts.append(t)
        t += 1.0 / CTRL_FREQ

    # Plot results
    print("\nPlotting trajectory results...")
    plot_trajectory_results(ts, ee_poses, target_poses)

    # Home and shutdown
    print("\nHoming robot...")
    robot.home()
    robot.shutdown()
    print("Done!")


if __name__ == "__main__":
    main()
