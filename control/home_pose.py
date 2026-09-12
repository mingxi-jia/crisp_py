"""The single source of truth for the robot's home pose, and the tool that sets it.

Recording the arm's current pose as home:

    python -m control.home_pose              # show old vs new, then confirm
    python -m control.home_pose --show       # compare only, change nothing
    python -m control.home_pose --yes        # no prompt

The reader and the writer live together deliberately: they are two halves of
one concern, and splitting them meant two files that could only ever change
in lockstep.

Everything that needs a home configuration reads it from
``config/home_pose.yaml`` through this module: control/robot_server.py,
and the deploy scripts. Previously the pose
was duplicated as hardcoded literals in several places, which drift apart
silently -- and a wrong home pose means the arm moves somewhere unexpected.

Deliberately dependency-light: yaml and the standard library only. No ROS, no
torch. control/pink_ik_server.py reaches this (via diffusion_constants) from
the ROS-free `pink_solver` environment.
"""

import datetime
import shutil
from pathlib import Path

import yaml

HOME_POSE_FILE = Path(__file__).resolve().parent.parent / "config" / "home_pose.yaml"
N_JOINTS = 7

# Used only when the file is missing or unreadable. Matches the Franka "ready"
# pose that crisp_py's FrankaConfig ships.
FALLBACK_HOME = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]


class HomePoseError(RuntimeError):
    """config/home_pose.yaml exists but could not be used."""


def _read(path: Path | None = None) -> dict:
    path = path or HOME_POSE_FILE
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _validate(values, what: str) -> list[float]:
    try:
        out = [float(v) for v in values]
    except (TypeError, ValueError) as e:
        raise HomePoseError(f"{what}: not a list of numbers ({e})") from e
    if len(out) != N_JOINTS:
        raise HomePoseError(f"{what}: expected {N_JOINTS} joint values, got {len(out)}")
    return out


def load_home_config(name: str | int | None = None,
                     path: Path | None = None,
                     strict: bool = False) -> list[float]:
    """Home joint configuration in radians.

    Args:
        name: None for the primary pose, or a key under ``poses:`` for an
              alternate (e.g. 1, matching deploy/example.py's --home-pos).
        strict: raise on a bad file instead of falling back.
    """
    path = path or HOME_POSE_FILE
    try:
        data = _read(path)
        if not data:
            raise HomePoseError(f"{path} not found")
        if name is None:
            values = data.get("home_config")
            if values is None:
                raise HomePoseError("no 'home_config' key")
            return _validate(values, "home_config")
        poses = data.get("poses") or {}
        # YAML parses a bare `1:` key as an int, while callers pass either 1 or
        # "1". Normalise both sides rather than matching on one spelling.
        by_str = {str(k): v for k, v in poses.items()}
        key = str(name)
        if key not in by_str:
            raise HomePoseError(
                f"no pose named {key!r} under 'poses' "
                f"(have: {sorted(by_str) or 'none'})")
        return _validate(by_str[key], f"poses.{key}")
    except HomePoseError:
        if strict:
            raise
        print(f"[home_pose] {path.name}: falling back to the built-in pose")
        return list(FALLBACK_HOME)


def load_time_to_home(default: float = 5.0, path: Path | None = None) -> float:
    try:
        return float(_read(path).get("time_to_home", default))
    except Exception:
        return default


def available_poses(path: Path | None = None) -> list[str]:
    """Names of the alternate poses, if any."""
    try:
        return sorted(str(k) for k in (_read(path).get("poses") or {}))
    except Exception:
        return []


def save_home_config(home: list[float],
                     time_to_home: float = 5.0,
                     poses: dict | None = None,
                     path: Path | None = None,
                     backup: bool = True) -> Path:
    """Write the pose file, preserving any alternates already in it."""
    path = path or HOME_POSE_FILE
    home = _validate(home, "home_config")

    if poses is None:
        poses = (_read(path).get("poses") or {})

    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(".yaml.bak"))

    lines = [
        "# Home joint configuration for the Franka, in radians, joints 1-7.",
        "#",
        "# THE single source of truth for the home pose. Read through",
        "# control/home_pose.py by robot_server.py and DPEvalConfig, so there",
        "# is exactly one place to change it.",
        "#",
        "# Recapture the arm's current pose with:",
        "#     python -m control.home_pose",
        "home_config:",
    ]
    lines += [f"  - {v!r}" for v in home]
    lines += ["", "# Seconds allowed for the homing trajectory.",
              f"time_to_home: {float(time_to_home)}"]
    if poses:
        lines += ["", "# Alternate poses, selected by deploy/example.py --home-pos.", "poses:"]
        for key in sorted(poses):
            lines.append(f"  {key}:")
            lines += [f"    - {float(v)!r}" for v in poses[key]]
    lines += ["", "# Provenance, so a surprising pose can be traced.",
              f'captured_at: "{datetime.date.today().isoformat()}"',
              'captured_from: "/joint_states"', ""]
    path.write_text("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# CLI: record the arm's current pose as home
#
# Reads /joint_states directly, so it works whether or not the robot server is
# running. ROS imports stay inside the CLI so the loader above remains
# dependency-light -- pink_ik_server reaches it from the ROS-free pink_solver
# environment.
# ---------------------------------------------------------------------------
import time

FR3_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


def read_current_joints(timeout: float = 15.0) -> list[float] | None:
    """Current fr3 joint positions from /joint_states, or None on timeout."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = Node("set_home_capture")
    captured: dict = {}

    def cb(msg: JointState):
        by_name = dict(zip(msg.name, msg.position))
        if all(j in by_name for j in FR3_JOINTS):
            captured["q"] = [float(by_name[j]) for j in FR3_JOINTS]

    node.create_subscription(JointState, "/joint_states", cb, 10)
    deadline = time.time() + timeout
    while "q" not in captured and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()
    return captured.get("q")


def fmt(q) -> str:
    return "[" + ", ".join(f"{v:+.4f}" for v in q) + "]"


def main():
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--yes", "-y", action="store_true", help="skip the confirmation")
    p.add_argument("--show", action="store_true", help="print poses, change nothing")
    p.add_argument("--time-to-home", type=float, default=5.0,
                   help="seconds allowed for the homing trajectory")
    args = p.parse_args()

    stored = load_home_config()
    print(f"stored home : {fmt(stored) if stored else '(none)'}")

    current = read_current_joints()
    if current is None:
        print("\nCould not read /joint_states. Is the robot driver running?")
        sys.exit(2)
    print(f"current pose: {fmt(current)}")

    if args.show:
        return
    if stored is not None:
        delta = max(abs(a - b) for a, b in zip(current, stored))
        print(f"largest change: {delta:.4f} rad ({delta * 57.2958:.2f} deg)")

    if not args.yes:
        print(f"\nWrite the current pose to {HOME_POSE_FILE.name}?")
        if input("  [y/N] ").strip().lower() not in ("y", "yes"):
            print("  unchanged")
            return

    save_home_config(current, time_to_home=args.time_to_home)
    print(f"  wrote {HOME_POSE_FILE}")
    alts = available_poses()
    if alts:
        print(f"  alternates preserved: {', '.join(alts)}")
    print("  restart the robot server for it to take effect")


if __name__ == "__main__":
    main()
