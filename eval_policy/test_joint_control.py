#!/usr/bin/env python3
"""Test script for joint control.

This script tests the joint control method used in the GelloController by:
1. Moving the robot to the home position defined in DPEvalConfig
2. Making minor changes to each joint sequentially

Usage:
    python eval_policy/test_joint_control.py
    python eval_policy/test_joint_control.py --delta 0.1  # Larger joint movements
    python eval_policy/test_joint_control.py --max-velocity 0.5  # Slower max velocity
"""

import argparse
import time
import numpy as np
import rclpy

from crisp_py.robot import Robot
from crisp_py.robot_config import FrankaConfig

from diff_eval_utils.diffusion_constants import DPEvalConfig

# Default maximum joint velocities (rad/s) - conservative values for safety
# Franka Emika FR3 has higher limits, but we use conservative values for testing
DEFAULT_MAX_JOINT_VELOCITY = 0.5  # rad/s


def parse_args():
    parser = argparse.ArgumentParser(description='Test joint control')
    parser.add_argument(
        '--delta',
        type=float,
        default=0.01,
        help='Joint position delta in radians (default: 0.05 rad ~= 2.9 degrees)'
    )
    parser.add_argument(
        '--max-velocity',
        type=float,
        default=DEFAULT_MAX_JOINT_VELOCITY,
        help=f'Maximum joint velocity in rad/s (default: {DEFAULT_MAX_JOINT_VELOCITY})'
    )
    parser.add_argument(
        '--min-time',
        type=float,
        default=0.5,
        help='Minimum time for any movement in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--pause',
        type=float,
        default=0.5,
        help='Pause between movements in seconds'
    )
    return parser.parse_args()


def compute_safe_time_to_goal(current_positions: np.ndarray,
                               target_positions: np.ndarray,
                               max_velocity: float,
                               min_time: float = 0.5) -> float:
    """Compute the minimum time required to reach target without exceeding max velocity.

    Args:
        current_positions: Current joint positions (rad)
        target_positions: Target joint positions (rad)
        max_velocity: Maximum allowed joint velocity (rad/s)
        min_time: Minimum time for any movement (s)

    Returns:
        Time in seconds required to complete the movement safely
    """
    # Calculate the maximum joint displacement
    displacements = np.abs(target_positions - current_positions)
    max_displacement = np.max(displacements)

    # Calculate minimum time required based on max velocity
    # time = distance / velocity
    required_time = max_displacement / max_velocity

    # Use the larger of required time or minimum time
    safe_time = max(required_time, min_time)

    return safe_time


def execute_joint_command(robot, target_positions: np.ndarray,
                          max_velocity: float, min_time: float = 0.5):
    """Execute a joint position command with velocity limiting.

    Args:
        robot: Robot instance
        target_positions: Array of 7 target joint positions
        max_velocity: Maximum allowed joint velocity (rad/s)
        min_time: Minimum time for any movement (s)
    """
    # Get current joint positions
    current_positions = np.array(robot.joint_values)

    # Compute safe time to goal based on max velocity
    time_to_goal = compute_safe_time_to_goal(
        current_positions, target_positions, max_velocity, min_time
    )

    # Calculate actual max velocity that will be used
    displacements = np.abs(target_positions - current_positions)
    max_displacement = np.max(displacements)
    actual_max_vel = max_displacement / time_to_goal if time_to_goal > 0 else 0

    print(f"    Max displacement: {max_displacement:.4f} rad, "
          f"Time: {time_to_goal:.2f}s, "
          f"Max velocity: {actual_max_vel:.3f} rad/s")

    robot.joint_trajectory_controller_client.send_joint_config(
        robot.config.joint_names,
        target_positions.tolist(),
        time_to_goal=time_to_goal,
        blocking=True
    )


def main():
    args = parse_args()

    # Get home position from DPEvalConfig
    config = DPEvalConfig()
    home_position = config.home_joint_position.copy()

    print("="*60)
    print("JOINT CONTROL TEST (VELOCITY LIMITED)")
    print("="*60)
    print(f"Home position: {home_position}")
    print(f"Joint delta: {args.delta} rad ({np.degrees(args.delta):.1f} degrees)")
    print(f"Max joint velocity: {args.max_velocity} rad/s ({np.degrees(args.max_velocity):.1f} deg/s)")
    print(f"Min movement time: {args.min_time}s")
    print(f"Pause between moves: {args.pause}s")
    print("="*60)

    # Initialize ROS2
    rclpy.init()

    try:
        # Setup robot
        print("\nInitializing robot...")
        config_franka = FrankaConfig()
        config_franka.home_config = home_position.tolist()
        robot = Robot(namespace="", robot_config=config_franka)
        robot.wait_until_ready()

        print(f"Robot ready. Current joint values: {robot.joint_values}")

        # Switch to joint trajectory controller
        print("\nSwitching to joint trajectory controller...")
        robot.controller_switcher_client.switch_controller("joint_trajectory_controller")
        time.sleep(0.5)

        # Step 1: Go to home position
        print("\n" + "-"*40)
        print("Step 1: Moving to home position...")
        print("-"*40)
        execute_joint_command(robot, home_position, args.max_velocity, min_time=2.0)
        print(f"Reached home position: {robot.joint_values}")
        time.sleep(args.pause)

        # Step 2: Test each joint with minor movements
        print("\n" + "-"*40)
        print("Step 2: Testing each joint individually...")
        print("-"*40)

        joint_names = robot.config.joint_names

        for i in range(len(home_position)):
            print(f"\n[Joint {i}] {joint_names[i]}")

            # Move joint positive
            target_pos = home_position.copy()
            target_pos[i] += args.delta
            print(f"  Moving +{args.delta:.3f} rad...")
            execute_joint_command(robot, target_pos, args.max_velocity, args.min_time)
            time.sleep(args.pause)

            # Move joint negative (back through home to negative)
            target_pos = home_position.copy()
            target_pos[i] -= args.delta
            print(f"  Moving -{args.delta:.3f} rad...")
            execute_joint_command(robot, target_pos, args.max_velocity, args.min_time)
            time.sleep(args.pause)

            # Return to home for this joint
            print(f"  Returning to home...")
            execute_joint_command(robot, home_position, args.max_velocity, args.min_time)
            time.sleep(args.pause)

        # Step 3: Combined movement (all joints slightly offset)
        print("\n" + "-"*40)
        print("Step 3: Combined movement (all joints)...")
        print("-"*40)

        combined_target = home_position.copy()
        combined_target += args.delta * 0.5  # Half delta for all joints
        print(f"Moving all joints +{args.delta*0.5:.3f} rad...")
        execute_joint_command(robot, combined_target, args.max_velocity, args.min_time)
        time.sleep(args.pause)

        # Return to home
        print("\n" + "-"*40)
        print("Returning to home position...")
        print("-"*40)
        execute_joint_command(robot, home_position, args.max_velocity, min_time=2.0)

        print("\n" + "="*60)
        print("TEST COMPLETE")
        print("="*60)
        print(f"Final joint values: {robot.joint_values}")

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nShutting down...")
        try:
            robot.shutdown()
        except:
            pass
        rclpy.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
