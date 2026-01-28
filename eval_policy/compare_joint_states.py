#!/usr/bin/env python3
"""Compare joint states between gello_control_signal and joint_states topics.

This script subscribes to both topics and prints a comparison table showing:
- Joint names
- Difference between gello and joint_states
- Joint states values
- Gello control signal values

Usage:
    python eval_policy/compare_joint_states.py
    python eval_policy/compare_joint_states.py --rate 10  # Update rate in Hz
"""

import argparse
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointComparisonNode(Node):
    """Node that compares joint states from two topics."""

    def __init__(self, update_rate: float = 5.0):
        super().__init__('joint_comparison_node')

        # Storage for latest messages
        self._joint_states_msg = None
        self._gello_msg = None
        self._lock = threading.Lock()

        # Create subscribers
        self._joint_states_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_states_callback,
            10
        )
        self._gello_sub = self.create_subscription(
            JointState,
            '/gello_control_signal',
            self._gello_callback,
            10
        )

        # Create timer for periodic printing
        self._timer = self.create_timer(1.0 / update_rate, self._print_comparison)

        self.get_logger().info('Joint comparison node started')
        self.get_logger().info('Subscribing to /joint_states and /gello_control_signal')

    def _joint_states_callback(self, msg: JointState):
        """Callback for joint_states topic."""
        with self._lock:
            self._joint_states_msg = msg

    def _gello_callback(self, msg: JointState):
        """Callback for gello_control_signal topic."""
        with self._lock:
            self._gello_msg = msg

    def _print_comparison(self):
        """Print comparison table of joint states."""
        with self._lock:
            joint_states_msg = self._joint_states_msg
            gello_msg = self._gello_msg

        # Check if we have data from both topics
        if joint_states_msg is None and gello_msg is None:
            print("\rWaiting for data from both topics...", end='', flush=True)
            return

        # Build dictionaries for easy lookup
        joint_states_dict = {}
        gello_dict = {}

        if joint_states_msg is not None:
            joint_states_dict = dict(zip(joint_states_msg.name, joint_states_msg.position))

        if gello_msg is not None:
            gello_dict = dict(zip(gello_msg.name, gello_msg.position))

        # Get all unique joint names (prioritize arm joints)
        all_names = set(joint_states_dict.keys()) | set(gello_dict.keys())

        # Filter to only arm joints (fr3_joint1 through fr3_joint7) and sort
        arm_joints = sorted([n for n in all_names if 'fr3_joint' in n])

        # If no arm joints found, use all joints
        if not arm_joints:
            arm_joints = sorted(all_names)

        # Clear screen and print header
        print("\033[2J\033[H", end='')  # Clear screen and move cursor to top
        print("=" * 75)
        print("JOINT STATE COMPARISON")
        print("=" * 75)
        print(f"{'Joint Name':<20} {'Difference':>12} {'Joint States':>15} {'Gello Signal':>15}")
        print("-" * 75)

        # Print comparison for each joint
        for name in arm_joints:
            js_val = joint_states_dict.get(name)
            gello_val = gello_dict.get(name)

            # Format values
            if js_val is not None:
                js_str = f"{js_val:>15.4f}"
            else:
                js_str = f"{'N/A':>15}"

            if gello_val is not None:
                gello_str = f"{gello_val:>15.4f}"
            else:
                gello_str = f"{'N/A':>15}"

            # Calculate difference
            if js_val is not None and gello_val is not None:
                diff = gello_val - js_val
                diff_str = f"{diff:>12.4f}"
            else:
                diff_str = f"{'N/A':>12}"

            print(f"{name:<20} {diff_str} {js_str} {gello_str}")

        print("-" * 75)

        # Print status
        js_status = "OK" if joint_states_msg is not None else "NO DATA"
        gello_status = "OK" if gello_msg is not None else "NO DATA"
        print(f"Status: joint_states={js_status}, gello_control_signal={gello_status}")
        print("Press Ctrl+C to exit")


def parse_args():
    parser = argparse.ArgumentParser(description='Compare joint states between topics')
    parser.add_argument(
        '--rate',
        type=float,
        default=5.0,
        help='Update rate in Hz (default: 5.0)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()

    try:
        node = JointComparisonNode(update_rate=args.rate)
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        rclpy.shutdown()
        print("Done.")


if __name__ == "__main__":
    main()
