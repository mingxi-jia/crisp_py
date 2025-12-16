#!/usr/bin/env python3
"""
Keyboard teleoperation runner.

Usage:
    python keyboard_runner.py  (from teleop directory)
    OR
    python -m teleop.keyboard_runner  (from parent directory)
"""

import sys
import rclpy

from .keyboard_teleop import KeyboardTeleopController


def main():
    """Run keyboard teleoperation."""
    controller = KeyboardTeleopController(
        ctrl_freq=10.0,
        position_scale=0.01,  # 1 cm per key press
        rotation_scale=0.05   # radians per key press
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
