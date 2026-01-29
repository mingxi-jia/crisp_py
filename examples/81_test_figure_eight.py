"""Test figure eight trajectory with joint impedance control using Pink IK.

This script:
1. Generates figure eight target poses on the yz plane (like 01_figure_eight.py)
2. Sends poses to Pink IK server to get joint solutions
3. Executes joint trajectory using joint_impedance_controller
4. Plots trajectory tracking results

Usage:
    1. Start the Pink IK server (in pink_solver env):
       conda activate pink_solver
       python pink_ik_server.py --port 5002

    2. Run this script (in ROS2/crisp_py env):
       python 81_test_figure_eight.py
"""

import time
import requests
import numpy as np
import matplotlib.pyplot as plt

from crisp_py.robot import Robot
from crisp_py.robot_config import FrankaConfig

# Configuration
PINK_SERVER_URL = "http://localhost:5002"

# Figure eight parameters (from 01_figure_eight.py)
RADIUS = 0.2  # [m]
CENTER = np.array([0.4, 0.0, 0.4])
CTRL_FREQ = 50.0
SIN_FREQ_Y = 0.25  # rot / s
SIN_FREQ_Z = 0.125  # rot / s
MAX_TIME = 8.0

# Home joint configuration for Franka
HOME_JOINT_POSITION = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


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


def setup_robot():
    """Initialize robot with joint impedance controller."""
    # Setup robot
    config_franka = FrankaConfig()
    config_franka.home_config = HOME_JOINT_POSITION
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


def generate_figure_eight_poses(center, radius, sin_freq_y, sin_freq_z, ctrl_freq, max_time, orientation_quat):
    """Generate figure eight target poses.

    Args:
        center: [x, y, z] center position
        radius: radius of the figure eight
        sin_freq_y: frequency for y oscillation
        sin_freq_z: frequency for z oscillation
        ctrl_freq: control frequency
        max_time: maximum time for trajectory
        orientation_quat: fixed orientation [qx, qy, qz, qw]

    Returns:
        poses: list of (position, orientation_quat) tuples
        times: list of time values
    """
    poses = []
    times = []
    t = 0.0
    dt = 1.0 / ctrl_freq

    while t < max_time:
        x = center[0]
        y = radius * np.sin(2 * np.pi * sin_freq_y * t) + center[1]
        z = radius * np.sin(2 * np.pi * sin_freq_z * t) + center[2]

        position = np.array([x, y, z])
        poses.append((position, orientation_quat))
        times.append(t)

        t += dt

    return poses, times


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

    # YZ plot (the figure eight plane)
    axes[0, 0].plot(y_ee, z_ee, label="actual", linewidth=2)
    axes[0, 0].plot(y_t, z_t, label="target", linestyle="--", linewidth=2)
    axes[0, 0].set_xlabel("$y$ [m]")
    axes[0, 0].set_ylabel("$z$ [m]")
    axes[0, 0].set_title("YZ Trajectory (Figure Eight)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].axis('equal')

    # Position vs time - Y
    axes[0, 1].plot(ts, y_ee, label="y actual", linewidth=2)
    axes[0, 1].plot(ts, y_t, label="y target", linestyle="--", linewidth=2)
    axes[0, 1].set_xlabel("$t$ [s]")
    axes[0, 1].set_ylabel("$y$ [m]")
    axes[0, 1].set_title("Y Position vs Time")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Position vs time - Z
    axes[1, 0].plot(ts, z_ee, label="z actual", linewidth=2)
    axes[1, 0].plot(ts, z_t, label="z target", linestyle="--", linewidth=2)
    axes[1, 0].set_xlabel("$t$ [s]")
    axes[1, 0].set_ylabel("$z$ [m]")
    axes[1, 0].set_title("Z Position vs Time")
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
    # Initialize Pink IK client
    print("Connecting to Pink IK server...")
    ik_client = PinkIKClient(PINK_SERVER_URL)

    if not ik_client.health_check():
        print(f"ERROR: Pink IK server not available at {PINK_SERVER_URL}")
        print("Start it with: conda activate pink_solver && python pink_ik_server.py")
        return

    print("Pink IK server connected!")

    # Setup robot
    print("\nInitializing robot...")
    robot = setup_robot()

    # Get current pose orientation to maintain during figure eight
    current_pose = robot.end_effector_pose
    orientation_quat = current_pose.orientation.as_quat()  # [qx, qy, qz, qw]
    print(f"Using orientation: {orientation_quat}")

    # Generate figure eight poses
    print("\nGenerating figure eight trajectory...")
    target_poses_with_quat, _ = generate_figure_eight_poses(
        CENTER, RADIUS, SIN_FREQ_Y, SIN_FREQ_Z, CTRL_FREQ, MAX_TIME, orientation_quat
    )
    print(f"Generated {len(target_poses_with_quat)} poses")

    # Get current joint state as initial guess for IK
    q_current = robot.joint_values.copy()

    # Create rate for timing
    rate = robot.node.create_rate(CTRL_FREQ)

    # Data collection for plotting
    ee_poses = []
    target_poses = []
    ts = []
    ik_timing_stats = []

    # Move to center position first
    print("\nMoving to center position...")
    first_pos, first_quat = target_poses_with_quat[0]
    q_first, success, timing = ik_client.solve_ik(first_pos, first_quat, q_current)
    ik_timing_stats.append(timing)
    print(f"First IK: success={success}, time={timing.get('total_time_ms', 0):.4f}ms")

    if not success:
        print("WARNING: First IK did not converge!")

    # Interpolate from current to first pose
    q_start = q_current
    q_end = q_first
    num_init_steps = 100  # More steps for initial movement

    for step in range(num_init_steps):
        alpha = (step + 1) / num_init_steps
        q_interp = (1 - alpha) * q_start + alpha * q_end
        robot.set_target_joint(q_interp)
        rate.sleep()

    q_current = q_first
    time.sleep(0.5)  # Let robot settle at start position

    # Execute figure eight trajectory
    print(f"\nStarting to draw figure eight ({len(target_poses_with_quat)} poses)...")
    print("=" * 60)

    t = 0.0
    for i, (target_pos, target_quat) in enumerate(target_poses_with_quat):
        # Solve IK for this pose
        q_target, success, timing = ik_client.solve_ik(target_pos, target_quat, q_current)
        ik_timing_stats.append(timing)

        if i % 50 == 0:
            print(f"Step {i+1}/{len(target_poses_with_quat)}: "
                  f"IK time={timing.get('total_time_ms', 0):.4f}ms, "
                  f"success={success}")

        if not success:
            print(f"  WARNING: IK did not converge at step {i+1}")
            # Use previous joint config if IK fails
            q_target = q_current

        # Send joint target directly (no interpolation for real-time tracking)
        robot.set_target_joint(q_target)
        rate.sleep()

        # Record data for plotting
        ee_poses.append(robot.end_effector_pose.copy())
        target_poses.append(target_pos.copy())
        ts.append(t)
        t += 1.0 / CTRL_FREQ

        q_current = q_target

    # Let robot settle
    print("\nWaiting for robot to settle...")
    settle_time = 1.0
    settle_steps = int(settle_time * CTRL_FREQ)
    for _ in range(settle_steps):
        rate.sleep()
        ee_poses.append(robot.end_effector_pose.copy())
        target_poses.append(target_poses[-1] if target_poses else target_poses_with_quat[-1][0])
        ts.append(t)
        t += 1.0 / CTRL_FREQ

    print("Done drawing figure eight!")

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
