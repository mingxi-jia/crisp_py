#!/usr/bin/env python3
"""Spacemouse teleoperation against robot_server.py.

A policy class and a control loop. Nothing else.

Start the server first:
    python -m control.robot_server --camera-config config/camera_info.yaml --port 7000

Then, from the repository root:
    python -m deploy.deploy_spacemouse

Controls:
    spacemouse        move the fingertip
    spacemouse button toggle the gripper
    'r'               re-home and re-anchor
    Ctrl+C            quit

Swapping the human for a network means replacing SpacemousePolicy with any
object exposing ``reset(pose)`` and ``predict_action(obs) -> Action``; the loop
below does not change.
"""

import argparse
import time

from control.robot_client import RobotClient
from control.teleop import SpacemouseConfig, SpacemousePolicy


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:7000",
                   help="robot_server base URL")
    p.add_argument("--rate", type=float, default=50.0,
                   help="control loop frequency [Hz]")
    p.add_argument("--n-steps", type=int, default=None,
                   help="stop after this many steps (default: run until Ctrl+C)")
    p.add_argument("--action-size", type=float, default=0.003,
                   help="per-step translation gain [m]")
    p.add_argument("--deadzone", type=float, default=0.1,
                   help="spacemouse deadzone, 0..1")
    p.add_argument("--home", action="store_true",
                   help="home the robot before starting")
    return p.parse_args()


def main():
    args = parse_args()

    robot = RobotClient(args.url)

    # Teleop drives the arm in Cartesian space. If the server is in joint
    # space the pose targets are silently dropped -- the gripper button still
    # works, so it looks like a broken spacemouse rather than a mode mismatch.
    ctrl_space = robot.ctrl_space
    if ctrl_space is not None and ctrl_space != "cartesian":
        raise SystemExit(
            f"the robot server at {args.url} is in '{ctrl_space}' control "
            f"space, but spacemouse teleop sends Cartesian pose targets that "
            f"the joint controller ignores.\n"
            f"Restart it in Cartesian space:\n"
            f"    ./scripts/start_all.sh --stop\n"
            f"    ./scripts/start_all.sh")

    policy = SpacemousePolicy(SpacemouseConfig(
        action_size=args.action_size,
        deadzone=args.deadzone,
    ))

    if args.home:
        print("Homing...")
        robot.home()

    period = 1.0 / args.rate
    step = 0

    print(f"Teleoperating at {args.rate:.0f} Hz. 'r' to re-home, Ctrl+C to quit.")
    with policy:
        was_paused = False
        policy.reset(robot.get_state().fingertip_pose)

        try:
            while args.n_steps is None or step < args.n_steps:
                tick = time.perf_counter()

                state = robot.get_state()
                action = policy.predict_action(state)

                if action.reset:
                    print("[r] re-homing...")
                    robot.home()
                    policy.reset(robot.get_state().fingertip_pose)
                    continue

                result = robot.servo(action.position, action.quat_xyzw)
                if action.gripper is not None:
                    robot.set_gripper(action.gripper)

                # The policy integrates deltas onto a running target. While the
                # server holds them, that target keeps drifting away from where
                # the arm actually is, so re-anchor on the resume edge --
                # otherwise the arm jumps to the accumulated target.
                paused = bool(result.get("paused"))
                if paused and not was_paused:
                    print("  [server paused] holding; the target will re-anchor on resume")
                elif was_paused and not paused:
                    policy.reset(robot.get_state().fingertip_pose)
                    print("  [server resumed] target re-anchored to the current pose")
                was_paused = paused

                step += 1
                time.sleep(max(0.0, period - (time.perf_counter() - tick)))
        except KeyboardInterrupt:
            print("\nStopped.")

    print(f"Ran {step} steps.")


if __name__ == "__main__":
    main()
