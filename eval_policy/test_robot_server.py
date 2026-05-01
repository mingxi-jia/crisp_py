#!/usr/bin/env python3
"""Tests for robot_server.py over HTTP.

Run the server in another terminal first:
    python eval_policy/robot_server.py \
        --camera-config /path/to/camera_info.yaml --port 7000 [--no-home]

Then run the test:
    python eval_policy/test_robot_server.py --url http://localhost:7000

By default this only exercises read-only endpoints (/health and /get_obs) so it
is safe to run on a live arm. Pass --test-goto to also issue a small relative
move via /goto (requires --confirm because it physically moves the robot).
"""

import argparse
import base64
import io
import sys
import time

import numpy as np
import requests


def decode_array(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["data"])
    return np.load(io.BytesIO(raw), allow_pickle=False)


# ---------------------------------------------------------------------------
# Individual tests — each prints PASS/FAIL and returns bool.
# ---------------------------------------------------------------------------
def test_health(url: str) -> bool:
    print("\n[test_health] GET /health")
    r = requests.get(f"{url}/health", timeout=5)
    r.raise_for_status()
    body = r.json()
    ok = body.get("ok") is True
    print(f"  response: {body}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def _check_array(field: str, payload, expected_ndim=None,
                 expected_dtype=None, allow_none=False) -> bool:
    if payload is None:
        if allow_none:
            print(f"  {field}: None (allowed)")
            return True
        print(f"  {field}: FAIL (was None, expected array)")
        return False
    try:
        arr = decode_array(payload)
    except Exception as e:
        print(f"  {field}: FAIL (decode error: {e})")
        return False
    if expected_ndim is not None and arr.ndim != expected_ndim:
        print(f"  {field}: FAIL (ndim={arr.ndim}, expected {expected_ndim})")
        return False
    if expected_dtype is not None and not np.issubdtype(arr.dtype, expected_dtype):
        print(f"  {field}: FAIL (dtype={arr.dtype}, expected {expected_dtype})")
        return False
    print(f"  {field}: shape={arr.shape} dtype={arr.dtype}")
    return True


def test_get_obs_schema(url: str) -> bool:
    print("\n[test_get_obs_schema] POST /get_obs (single call, schema check)")
    r = requests.post(f"{url}/get_obs", timeout=30)
    r.raise_for_status()
    raw = r.json()

    expected = {
        "timestamp", "rgb", "depth", "inhand_rgb", "inhand_depth",
        "joint_values", "ee_position", "ee_quat_xyzw",
        "gripper_value", "gripper_state", "ft_wrench",
    }
    missing = expected - set(raw.keys())
    if missing:
        print(f"  FAIL: missing keys {missing}")
        return False
    print(f"  keys present: ok ({sorted(raw.keys())})")

    if not isinstance(raw["timestamp"], (int, float)):
        print(f"  timestamp: FAIL (type={type(raw['timestamp'])})")
        return False
    print(f"  timestamp: {raw['timestamp']:.3f}")

    if not isinstance(raw["rgb"], list) or not raw["rgb"]:
        print(f"  rgb: FAIL (not a non-empty list)")
        return False
    if len(raw["depth"]) != len(raw["rgb"]):
        print(f"  FAIL: rgb has {len(raw['rgb'])} cams, depth has {len(raw['depth'])}")
        return False

    results = []
    for i, (rgb, depth) in enumerate(zip(raw["rgb"], raw["depth"])):
        results.append(_check_array(f"rgb[{i}]", rgb, expected_ndim=3, expected_dtype=np.integer))
        results.append(_check_array(f"depth[{i}]", depth, expected_ndim=2, expected_dtype=np.integer))

    results.append(_check_array("inhand_rgb", raw["inhand_rgb"],
                                expected_ndim=3, expected_dtype=np.integer))
    results.append(_check_array("inhand_depth", raw["inhand_depth"], expected_ndim=2))
    results.append(_check_array("joint_values", raw["joint_values"],
                                expected_ndim=1, expected_dtype=np.floating))
    results.append(_check_array("ee_position", raw["ee_position"],
                                expected_ndim=1, expected_dtype=np.floating))
    results.append(_check_array("ee_quat_xyzw", raw["ee_quat_xyzw"],
                                expected_ndim=1, expected_dtype=np.floating))

    gv = raw["gripper_value"]
    if gv is not None and not isinstance(gv, (int, float)):
        print(f"  gripper_value: FAIL (type={type(gv)})")
        results.append(False)
    else:
        print(f"  gripper_value: {gv}")
        results.append(True)

    results.append(_check_array("gripper_state", raw["gripper_state"], allow_none=True))
    results.append(_check_array("ft_wrench", raw["ft_wrench"],
                                expected_ndim=1, allow_none=True))

    # Sanity: joint_values length is 7 for a Franka, ee_position length 3, quat 4
    jv = decode_array(raw["joint_values"])
    ee = decode_array(raw["ee_position"])
    quat = decode_array(raw["ee_quat_xyzw"])
    if jv.shape != (7,):
        print(f"  joint_values: FAIL (shape={jv.shape}, expected (7,))")
        results.append(False)
    if ee.shape != (3,):
        print(f"  ee_position: FAIL (shape={ee.shape}, expected (3,))")
        results.append(False)
    if quat.shape != (4,):
        print(f"  ee_quat_xyzw: FAIL (shape={quat.shape}, expected (4,))")
        results.append(False)
    qnorm = float(np.linalg.norm(quat))
    if not (0.95 < qnorm < 1.05):
        print(f"  ee_quat_xyzw: FAIL (norm={qnorm:.3f}, expected ~1.0)")
        results.append(False)
    else:
        print(f"  ee_quat_xyzw norm: {qnorm:.3f}")

    ok = all(results)
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_get_obs_rate(url: str, n: int = 10) -> bool:
    print(f"\n[test_get_obs_rate] POST /get_obs x {n} (rate + freshness)")
    timestamps = []
    wallclock = []
    for _ in range(n):
        t0 = time.time()
        r = requests.post(f"{url}/get_obs", timeout=30)
        r.raise_for_status()
        raw = r.json()
        timestamps.append(raw["timestamp"])
        wallclock.append(time.time() - t0)
    intervals = np.diff(timestamps)
    if len(intervals) == 0 or np.all(intervals == 0):
        print(f"  FAIL: timestamps did not advance: {timestamps}")
        return False
    avg_rtt_ms = 1000 * np.mean(wallclock)
    avg_dt = float(np.mean(intervals))
    print(f"  avg server-stamp dt: {avg_dt*1000:.1f} ms ({1.0/avg_dt:.2f} Hz)")
    print(f"  avg request RTT:     {avg_rtt_ms:.1f} ms")
    print(f"  PASS")
    return True


def test_goto_relative(url: str, dz: float = 0.02,
                       use_move_to: bool = False) -> bool:
    """Move the EE up by `dz`, then back down. Requires the robot to be live."""
    print(f"\n[test_goto_relative] POST /goto (dz=+{dz:.3f} m, then return)")
    r = requests.post(f"{url}/get_obs", timeout=30)
    r.raise_for_status()
    raw = r.json()
    ee = decode_array(raw["ee_position"])
    quat = decode_array(raw["ee_quat_xyzw"])
    print(f"  start ee_position: {ee}")

    target_up = ee + np.array([0.0, 0.0, dz], dtype=np.float64)
    payload = {
        "position": target_up.tolist(),
        "quat_xyzw": quat.tolist(),
        "use_move_to": bool(use_move_to),
        "speed": 0.05,
    }
    r = requests.post(f"{url}/goto", json=payload, timeout=60)
    r.raise_for_status()
    resp_up = r.json()
    print(f"  up:   {resp_up}")
    if not resp_up.get("ok"):
        print("  FAIL: server reported not ok")
        return False
    time.sleep(0.5)

    payload["position"] = ee.tolist()
    r = requests.post(f"{url}/goto", json=payload, timeout=60)
    r.raise_for_status()
    resp_dn = r.json()
    print(f"  down: {resp_dn}")
    if not resp_dn.get("ok"):
        print("  FAIL: server reported not ok")
        return False

    # verify we actually moved (server-reported pose changed) then returned
    pos_up = np.asarray(resp_up["ee_position"])
    pos_dn = np.asarray(resp_dn["ee_position"])
    moved_up = bool(np.linalg.norm(pos_up - ee) > dz / 2)
    returned = bool(np.linalg.norm(pos_dn - ee) < dz)
    print(f"  moved up: {moved_up} (delta={np.linalg.norm(pos_up - ee):.4f} m)")
    print(f"  returned: {returned} (delta={np.linalg.norm(pos_dn - ee):.4f} m)")
    ok = moved_up and returned
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:7000")
    p.add_argument("--rate-samples", type=int, default=10,
                   help="Number of /get_obs calls for rate test")
    p.add_argument("--test-goto", action="store_true",
                   help="Also exercise /goto (MOVES the robot)")
    p.add_argument("--goto-dz", type=float, default=0.02,
                   help="Vertical step in meters for /goto test")
    p.add_argument("--use-move-to", action="store_true",
                   help="Use server.robot.move_to instead of interpolated set_target")
    p.add_argument("--confirm", action="store_true",
                   help="Required when --test-goto is set, to acknowledge the arm will move")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Target server: {args.url}")

    results = []
    results.append(("health", test_health(args.url)))
    results.append(("get_obs schema", test_get_obs_schema(args.url)))
    results.append(("get_obs rate", test_get_obs_rate(args.url, n=args.rate_samples)))

    if args.test_goto:
        if not args.confirm:
            print("\n[skip] --test-goto requires --confirm (the robot will MOVE)")
            results.append(("goto", False))
        else:
            results.append(("goto", test_goto_relative(args.url,
                                                      dz=args.goto_dz,
                                                      use_move_to=args.use_move_to)))

    print("\n=== Results ===")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    n_fail = sum(1 for _, ok in results if not ok)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
