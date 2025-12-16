#!/usr/bin/env python3
"""
Spacemouse teleoperation runner.

Usage:
    python spacemouse_runner.py  (from teleop directory)
    OR
    python -m teleop.spacemouse_runner  (from parent directory)
"""

import sys
import rclpy

from .spacemouse_teleop import SpacemouseTeleopController


def main():
    """Run spacemouse teleoperation."""
    controller = SpacemouseTeleopController(
        ctrl_freq=10.0,
        action_scale=0.01,  # 1 cm per action unit
        deadzone=0.1
    )
    controller.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
        try:
            rclpy.shutdown()
        except Exception:
            pass
        sys.exit(0)
