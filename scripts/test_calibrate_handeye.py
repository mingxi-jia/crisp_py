"""Tests for the hand-eye solver. No robot, no camera, no network.

    python -m scripts.test_calibrate_handeye

The whole point of synthetic data here is that the answer is known exactly:
a transform is chosen, the images it would produce are generated from it, and
the solver has to give it back. A hand-eye result that is quietly 2 cm out
looks exactly like a good one on real data, so this is the only place the
maths can actually be checked.
"""

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.handeye import (
    HandEyeResult, invert, make_target, rigid, rotation_coverage, solve,
    yaml_block,
)

TRUE_K = np.array([[430.0, 0.0, 318.5],
                   [0.0, 429.0, 241.0],
                   [0.0, 0.0, 1.0]])
TRUE_DIST = np.array([-0.045, 0.052, 0.0008, -0.0004, -0.012])
IMAGE_SIZE = (640, 480)

# The answer the tests have to recover: the camera sits 6 cm to one side of
# the gripper frame, pulled back 4 cm, tilted 25 degrees to look ahead of the
# fingers. Nothing about it is special; it is just not near identity.
TRUE_X = rigid(Rotation.from_euler("xyz", [0.10, -0.44, 0.06]).as_matrix(),
               [0.058, -0.021, -0.043])
TRUE_TARGET = rigid(Rotation.from_euler("xyz", [0.0, 0.0, 0.3]).as_matrix(),
                    [0.55, 0.05, 0.02])


@dataclasses.dataclass
class FakeSample:
    object_points: np.ndarray
    image_points: np.ndarray
    ee_position: np.ndarray
    ee_quat_xyzw: np.ndarray


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    assert cond, name


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """A camera pose at `eye` pointing at `target`, in OpenCV convention."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, float), z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    return rigid(np.stack([x, np.cross(z, x), z], axis=1), eye)


def make_views(target, n=14, seed=0, noise_px=0.0, single_axis=False):
    """Gripper poses looking at the target, and the images they would make."""
    import cv2

    rng = np.random.default_rng(seed)
    samples, gripper_rots = [], []
    for i in range(n):
        # Camera poses spread over a patch of sphere around the target, each
        # with its own roll -- the variety hand-eye needs.
        angle = 2 * np.pi * i / n
        radius = 0.28 + 0.06 * rng.random()
        eye = TRUE_TARGET[:3, 3] + np.array([
            radius * 0.55 * np.cos(angle),
            radius * 0.55 * np.sin(angle),
            radius * (0.75 + 0.25 * rng.random())])
        T_base_cam = look_at(eye, TRUE_TARGET[:3, 3])
        if not single_axis:
            roll = Rotation.from_rotvec([0, 0, 0.5 * np.sin(2.3 * i)]).as_matrix()
            T_base_cam[:3, :3] = T_base_cam[:3, :3] @ roll

        # The gripper pose that would put the camera there.
        T_base_gripper = T_base_cam @ invert(TRUE_X)
        T_cam_target = invert(T_base_cam) @ TRUE_TARGET

        rvec, _ = cv2.Rodrigues(T_cam_target[:3, :3])
        img, _ = cv2.projectPoints(target.object_points.reshape(-1, 1, 3),
                                   rvec, T_cam_target[:3, 3], TRUE_K, TRUE_DIST)
        img = img.reshape(-1, 2)
        if noise_px:
            img = img + rng.normal(0, noise_px, img.shape)
        if (img[:, 0].min() < 0 or img[:, 1].min() < 0
                or img[:, 0].max() > IMAGE_SIZE[0] or img[:, 1].max() > IMAGE_SIZE[1]):
            continue                      # would not be visible; skip the view
        samples.append(FakeSample(
            object_points=target.object_points.copy(),
            image_points=img.astype(np.float32),
            ee_position=T_base_gripper[:3, 3].copy(),
            ee_quat_xyzw=Rotation.from_matrix(T_base_gripper[:3, :3]).as_quat()))
        gripper_rots.append(T_base_gripper[:3, :3])
    return samples, gripper_rots


def error_of(result: HandEyeResult) -> tuple[float, float]:
    """(position error [mm], rotation error [deg]) against the true transform."""
    pos = np.linalg.norm(result.T_gripper_cam[:3, 3] - TRUE_X[:3, 3]) * 1000.0
    rot = (Rotation.from_matrix(result.T_gripper_cam[:3, :3]).inv()
           * Rotation.from_matrix(TRUE_X[:3, :3])).magnitude()
    return float(pos), float(np.degrees(rot))


def main():
    board = make_target("checkerboard", rows=6, cols=9, square=0.025)

    print("=== exact data, intrinsics known ===")
    samples, _ = make_views(board, n=14)
    check("enough synthetic views survive the frame", len(samples) >= 8, f"{len(samples)}")
    r = solve(samples, IMAGE_SIZE, fixed_k=TRUE_K, fixed_dist=TRUE_DIST)
    pos_mm, rot_deg = error_of(r)
    check("recovers the transform", pos_mm < 0.5 and rot_deg < 0.05,
          f"{pos_mm:.3f} mm, {rot_deg:.4f} deg")
    check("reports the target as stationary", r.spread_mm < 0.5, f"{r.spread_mm:.3f} mm")
    check("finds where the target is",
          np.linalg.norm(r.T_base_target[:3, 3] - TRUE_TARGET[:3, 3]) < 1e-3)
    check("verdict is positive", r.verdict()[0], r.verdict()[1])

    print("\n=== exact data, intrinsics solved from the same views ===")
    r = solve(samples, IMAGE_SIZE)
    pos_mm, rot_deg = error_of(r)
    check("recovers the transform", pos_mm < 1.0 and rot_deg < 0.1,
          f"{pos_mm:.3f} mm, {rot_deg:.4f} deg")
    check("recovers the focal length",
          abs(r.k[0, 0] - TRUE_K[0, 0]) < 2.0, f"{r.k[0,0]:.2f} vs {TRUE_K[0,0]}")
    check("recovers the principal point",
          abs(r.k[0, 2] - TRUE_K[0, 2]) < 3.0, f"{r.k[0,2]:.2f} vs {TRUE_K[0,2]}")
    check("recovers the distortion",
          abs(r.dist[0] - TRUE_DIST[0]) < 0.02, f"{r.dist[0]:.4f} vs {TRUE_DIST[0]}")

    print("\n=== with detector noise ===")
    for noise in (0.1, 0.3):
        samples, _ = make_views(board, n=14, seed=3, noise_px=noise)
        r = solve(samples, IMAGE_SIZE, fixed_k=TRUE_K, fixed_dist=TRUE_DIST)
        pos_mm, rot_deg = error_of(r)
        # Sub-pixel detector noise should cost fractions of a millimetre, not
        # centimetres; if it does not, the problem is ill-conditioned.
        check(f"{noise} px noise stays sub-centimetre",
              pos_mm < 8.0, f"{pos_mm:.2f} mm, {rot_deg:.3f} deg")
        check(f"{noise} px noise shows up in the spread",
              r.spread_mm < 8.0, f"{r.spread_mm:.2f} mm")

    print("\n=== the degenerate case the meter exists to catch ===")
    samples, rots = make_views(board, n=12, seed=1, single_axis=True)
    cov = rotation_coverage(rots)
    check("rotation about one axis is reported as insufficient", not cov["ok"],
          f"axes {cov['axes']}")
    samples, rots = make_views(board, n=12, seed=1)
    cov = rotation_coverage(rots)
    check("varied rotation is reported as sufficient", cov["ok"], f"axes {cov['axes']}")
    check("a single pose cannot be judged", not rotation_coverage(rots[:1])["ok"])

    print("\n=== solvers agree ===")
    samples, _ = make_views(board, n=14, seed=5, noise_px=0.15)
    errors = {}
    for method in ("TSAI", "PARK", "HORAUD", "DANIILIDIS"):
        r = solve(samples, IMAGE_SIZE, fixed_k=TRUE_K, fixed_dist=TRUE_DIST,
                  method=method)
        errors[method] = round(error_of(r)[0], 2)
    check("every method lands in the same place", max(errors.values()) < 10.0,
          " ".join(f"{k}={v}mm" for k, v in errors.items()))

    print("\n=== a single ArUco marker ===")
    # Four points per view: the pose is determined, the lens is not, which is
    # why the script insists on intrinsics for this target.
    tag = make_target("aruco", marker_size=0.1, dictionary="DICT_ARUCO_ORIGINAL")
    samples, _ = make_views(tag, n=16, seed=2, noise_px=0.1)
    r = solve(samples, IMAGE_SIZE, fixed_k=TRUE_K, fixed_dist=TRUE_DIST)
    pos_mm, rot_deg = error_of(r)
    check("a 100 mm tag is enough with known intrinsics",
          pos_mm < 10.0, f"{pos_mm:.2f} mm, {rot_deg:.3f} deg")

    print("\n=== too few samples ===")
    try:
        solve(samples[:2], IMAGE_SIZE, fixed_k=TRUE_K)
        ok = False
    except ValueError:
        ok = True
    check("refuses to solve from two views", ok)

    print("\n=== the YAML entry the overlay will read ===")
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from deploy.pi05.overlay import CameraCalibration

    samples, _ = make_views(board, n=14)
    r = solve(samples, IMAGE_SIZE, fixed_k=TRUE_K, fixed_dist=TRUE_DIST)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml_block(r, "cam4", "fr3_hand_tcp"))
        path = f.name
    loaded = CameraCalibration.load("cam4", path)
    check("loads back as a moving camera", loaded is not None and loaded.moving)
    check("mounted on the frame it was written with", loaded.parent == "fr3_hand_tcp")
    check("round-trips the position",
          np.allclose(loaded.pos_base_cam, r.position, atol=1e-9))
    check("round-trips the rotation",
          np.allclose(loaded.rot_base_cam, r.T_gripper_cam[:3, :3], atol=1e-9))
    check("round-trips the intrinsics", np.allclose(loaded.k, r.k, atol=1e-9))
    check("records the image size", loaded.calib_size == IMAGE_SIZE)

    print("\n=== the panel can serialise all of it ===")
    # The panel hands these dicts to Flask's JSON encoder, which rejects
    # np.bool_ and np.float64. A single numpy scalar anywhere in the payload
    # turns the whole page into a 500, and nothing else in these tests would
    # notice, because numpy scalars compare and print exactly like the real
    # thing.
    import json
    from scripts.calibrate_handeye import Session, jsonable

    check("coverage survives json.dumps", json.dumps(r.coverage) is not None)
    check("its verdict flag is a real bool", type(r.coverage["ok"]) is bool,
          type(r.coverage["ok"]).__name__)

    class FakeArgs:
        camera, entry, frame, method = "inhand", "cam4", "fr3_hand_tcp", "TSAI"

    session = Session(make_target("aruco", marker_size=0.1), FakeArgs(),
                      camera_size=IMAGE_SIZE)
    session.result = r
    session.still = np.bool_(True)              # what the settle test produces
    session.live_detected = np.bool_(True)
    session.cross_check = {"camera": "agentview",
                           "distance_mm": np.float64(4.2),
                           "delta_mm": np.array([1.0, -2.0, 3.5])}
    payload = json.dumps(session.snapshot())
    check("a full snapshot serialises", len(payload) > 200, f"{len(payload)} bytes")
    back = json.loads(payload)
    check("numpy scalars came through as plain types",
          type(back["still"]) is bool and type(back["cross_check"]["distance_mm"]) is float)
    check("the yaml entry is carried along", "fr3_hand_tcp" in back["result"]["yaml"])

    check("jsonable leaves plain data alone",
          jsonable({"a": [1, "b", None, 2.5]}) == {"a": [1, "b", None, 2.5]})

    print("\nAll hand-eye tests passed.")


if __name__ == "__main__":
    main()
