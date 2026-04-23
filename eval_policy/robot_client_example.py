#!/usr/bin/env python3
"""Minimal client demo for robot_server.py.

Shows how to:
  - call /get_obs and decode the returned arrays (RGB/D + robot state)
  - call /goto to move the end-effector to a Cartesian pose

Run the server first, e.g.:
    python eval_policy/robot_server.py \
        --camera-config /path/to/camera_info.yaml --port 7000

Then:
    python eval_policy/robot_client_example.py --url http://localhost:7000
"""

import argparse
import base64
import io

import numpy as np
import requests


# ---------------------------------------------------------------------------
# Encoding helpers — must mirror robot_server.py
# ---------------------------------------------------------------------------
def decode_array(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["data"])
    return np.load(io.BytesIO(raw), allow_pickle=False)


def encode_array(arr: np.ndarray) -> dict:
    arr = np.ascontiguousarray(arr)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return {
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class RobotClient:
    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        # quick health check
        r = requests.get(f"{self.url}/health", timeout=5)
        r.raise_for_status()

    def get_obs(self) -> dict:
        """Fetch latest observation. Returns dict with decoded numpy arrays."""
        r = requests.post(f"{self.url}/get_obs", timeout=self.timeout)
        r.raise_for_status()
        raw = r.json()
        return {
            "timestamp":     raw["timestamp"],
            "rgb":           [decode_array(x) for x in raw["rgb"]],
            "depth":         [decode_array(x) for x in raw["depth"]],
            "inhand_rgb":    decode_array(raw["inhand_rgb"]),
            "inhand_depth":  decode_array(raw["inhand_depth"]),
            "joint_values":  decode_array(raw["joint_values"]),
            "ee_position":   decode_array(raw["ee_position"]),
            "ee_quat_xyzw":  decode_array(raw["ee_quat_xyzw"]),
            "gripper_value": raw["gripper_value"],
            "gripper_state": decode_array(raw["gripper_state"]) if raw.get("gripper_state") else None,
            "ft_wrench":     decode_array(raw["ft_wrench"]) if raw.get("ft_wrench") else None,
        }

    def goto(self,
             position,
             quat_xyzw=None,
             use_move_to: bool = False,
             speed: float = 0.05,
             gripper: float | None = None) -> dict:
        """Command the EE to a Cartesian pose.

        Args:
            position:  iterable of 3 floats [x, y, z] (base frame).
            quat_xyzw: optional [qx, qy, qz, qw]. None = hold current orient.
            use_move_to: if True, server uses robot.move_to (linear move).
            speed:    m/s, only used with use_move_to=True.
            gripper:  optional 0..1 (1=open, 0=close).
        """
        payload = {
            "position": list(map(float, position)),
            "use_move_to": bool(use_move_to),
            "speed": float(speed),
        }
        if quat_xyzw is not None:
            payload["quat_xyzw"] = list(map(float, quat_xyzw))
        if gripper is not None:
            payload["gripper"] = float(gripper)
        r = requests.post(f"{self.url}/goto", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:7000")
    p.add_argument("--dx", type=float, default=0.0,
                   help="x offset to apply via /goto (m)")
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--dz", type=float, default=0.05,
                   help="z offset to apply via /goto (m). Default lifts 5cm.")
    p.add_argument("--gripper", type=float, default=None,
                   help="0=close, 1=open. Omit to leave unchanged.")
    p.add_argument("--no-move", action="store_true",
                   help="Only print observation, do not call /goto.")
    return p.parse_args()


def main():
    args = parse_args()
    client = RobotClient(args.url)

    print(f"--- get_obs() from {args.url} ---")
    obs = client.get_obs()
    print(f"timestamp     : {obs['timestamp']:.3f}")
    print(f"joint_values  : {obs['joint_values']}")
    print(f"ee_position   : {obs['ee_position']}")
    print(f"ee_quat_xyzw  : {obs['ee_quat_xyzw']}")
    print(f"gripper_value : {obs['gripper_value']}")
    for i, (rgb, d) in enumerate(zip(obs["rgb"], obs["depth"]), 1):
        print(f"cam{i}: rgb={rgb.shape} {rgb.dtype}, depth={d.shape} {d.dtype}")
    print(f"inhand_rgb    : {obs['inhand_rgb'].shape} {obs['inhand_rgb'].dtype}")
    print(f"inhand_depth  : {obs['inhand_depth'].shape} {obs['inhand_depth'].dtype}")
    if obs["ft_wrench"] is not None:
        print(f"ft_wrench     : {obs['ft_wrench']}")

    if args.no_move:
        return

    target = obs["ee_position"] + np.array([args.dx, args.dy, args.dz],
                                           dtype=np.float32)
    print(f"\n--- goto({target.tolist()}) — holding current orientation ---")
    resp = client.goto(
        position=target,
        quat_xyzw=obs["ee_quat_xyzw"],
        use_move_to=False,
        gripper=args.gripper,
    )
    print(f"server response: {resp}")


if __name__ == "__main__":
    main()
