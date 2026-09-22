"""Tests for deploy/pi05/overlay.py. No hardware, no checkpoint, no network.

    python -m deploy.pi05.test_overlay

The interesting test is `to_obs_pixels` against `fit_image`: the overlay is
only useful if a projected point lands on the same pixel the policy sees it
at, and the two functions are written separately (one moves images, the other
moves coordinates), so they are checked against each other on real arrays.
"""

import numpy as np

from control.kinematics import FINGERTIP_OFFSET_XYZ, fingertip_offset_matrix
from deploy.pi05.overlay import (
    TIP_LINK, Chain, CameraCalibration, Overlay, axis_tips, fk_residual,
    frame_offset, merge, predicted_path, rollout_joint_targets, to_obs_pixels,
)
from deploy.pi05.policy import IMAGE_SIZE, MAX_JOINT_DELTA, fit_image

# The Franka "ready" pose and the hand TCP position Franka documents for it.
READY_Q = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
READY_TCP = np.array([0.307, 0.0, 0.487])


def _with_samples(policy, samples, n_inferences):
    """The same fake policy with a different sample set (and cache key)."""
    policy.last_samples = samples
    policy.n_inferences = n_inferences
    return policy


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    assert cond, name


def marker_frame(width, height, u, v, radius=3):
    """A black frame with one bright disc, for tracking a pixel through a resize."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    img[((xx - u) ** 2 + (yy - v) ** 2) <= radius ** 2] = 255
    return img


def centroid(img):
    """Intensity-weighted centre of a resized marker, in (u, v)."""
    w = img[..., 0].astype(np.float64)
    if w.sum() < 1e-6:
        return None
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    return np.array([(w * xx).sum() / w.sum(), (w * yy).sum() / w.sum()])


def main():
    print("=== Chain / FK ===")
    chain = Chain.from_urdf()
    check("7 actuated joints", chain.n_joints == 7, f"{chain.n_joints}")
    tcp = chain.fk(READY_Q)[:3, 3]
    check("ready pose matches Franka's documented TCP",
          np.allclose(tcp, READY_TCP, atol=5e-4), f"{tcp.round(4)}")
    check("fk is a rigid transform",
          np.allclose(chain.fk(READY_Q)[:3, :3] @ chain.fk(READY_Q)[:3, :3].T, np.eye(3)))
    check("fk_batch stacks", chain.fk_batch(np.zeros((4, 7))).shape == (4, 4, 4))
    try:
        chain.fk(np.zeros(6)); ok = False
    except ValueError:
        ok = True
    check("rejects too few joint angles", ok)
    try:
        Chain.from_urdf(tip="not_a_link"); ok = False
    except ValueError:
        ok = True
    check("rejects an unreachable tip link", ok)

    print("\n=== fk_residual ===")
    from scipy.spatial.transform import Rotation
    T = chain.fk(READY_Q)
    pos_err, rot_err = fk_residual(chain, READY_Q, T[:3, 3],
                                   Rotation.from_matrix(T[:3, :3]).as_quat())
    check("zero against its own FK", pos_err < 1e-12 and rot_err < 1e-9,
          f"{pos_err:.2e} m, {rot_err:.2e} deg")
    pos_err, _ = fk_residual(chain, READY_Q, T[:3, 3] + [0.01, 0, 0])
    check("reports a 10 mm offset", abs(pos_err - 0.01) < 1e-9, f"{pos_err*1000:.2f} mm")

    print("\n=== frames ===")
    check("'ee' frame is identity", np.allclose(frame_offset("ee"), np.eye(4)))
    check("'fingertip' frame is the configured tool offset",
          np.allclose(frame_offset("fingertip"), fingertip_offset_matrix()))
    ftip = chain.fk(READY_Q) @ frame_offset("fingertip")
    # Derived from config/gripper_geometry.yaml, never restated: that file is
    # the only place the tool length lives, so hardcoding a distance here
    # would silently contradict it the next time the tool changes.
    expected = float(np.linalg.norm(FINGERTIP_OFFSET_XYZ))
    check(f"fingertip is {expected*1000:.1f} mm further along the gripper",
          abs(np.linalg.norm(ftip[:3, 3] - tcp) - expected) < 1e-9)
    try:
        frame_offset("elbow"); ok = False
    except ValueError:
        ok = True
    check("rejects an unknown frame", ok)

    print("\n=== rollout ===")
    chunk = np.zeros((6, 8))
    chunk[:, 0] = 1.0                                   # joint 1 at full speed
    q = rollout_joint_targets(chunk, np.zeros(7))
    check("full-scale step is MAX_JOINT_DELTA",
          abs(q[0, 0] - MAX_JOINT_DELTA) < 1e-12, f"{q[0, 0]}")
    check("integrates over the chunk",
          abs(q[-1, 0] - 6 * MAX_JOINT_DELTA) < 1e-12, f"{q[-1, 0]}")
    q = rollout_joint_targets(chunk, np.zeros(7), speed_scale=0.25)
    check("speed_scale scales each step", abs(q[0, 0] - 0.05) < 1e-12, f"{q[0, 0]}")
    q = rollout_joint_targets(chunk, np.zeros(7), max_step_rad=0.01)
    check("max_step_rad caps each step", abs(q[0, 0] - 0.01) < 1e-12, f"{q[0, 0]}")
    q = rollout_joint_targets(np.full((3, 8), 5.0), np.zeros(7))
    check("out-of-range velocities are clipped first",
          abs(q[0, 0] - MAX_JOINT_DELTA) < 1e-12, f"{q[0, 0]}")

    print("\n=== predicted_path ===")
    pos, poses = predicted_path(chain, chunk, READY_Q)
    check("one point per chunk step plus the start", pos.shape == (7, 3), f"{pos.shape}")
    check("starts at the current fingertip",
          np.allclose(pos[0], (chain.fk(READY_Q) @ frame_offset("fingertip"))[:3, 3]))
    check("a zero chunk does not move",
          np.allclose(predicted_path(chain, np.zeros((4, 8)), READY_Q)[0], pos[0]))
    check("axis tips are one length from the origin",
          np.allclose(np.linalg.norm(axis_tips(poses[0], 0.05) - pos[0], axis=1), 0.05))

    print("\n=== calibration ===")
    cal = CameraCalibration.load("cam1")
    check("cam1 loads from config/camera_info.yaml", cal is not None)
    check("an unknown camera has no calibration",
          CameraCalibration.load("no_such_camera") is None)
    check("missing file is not an error", CameraCalibration.load("cam1", "/nope.yaml") is None)

    # A point on the optical axis must land on the principal point, whatever
    # the resolution: this is the one projection answer knowable by hand.
    axis_point = cal.pos_base_cam + cal.rot_base_cam @ np.array([0.0, 0.0, 1.5])
    for size in [(480, 270), (848, 480)]:
        uv, front = cal.project(axis_point, size)
        k = cal.scaled_k(size)
        check(f"optical-axis point -> principal point at {size[0]}x{size[1]}",
              front[0] and np.allclose(uv[0], [k[0, 2], k[1, 2]], atol=1e-6),
              f"{uv[0].round(2)} vs {[round(k[0,2],2), round(k[1,2],2)]}")
    # Cross-check the distortion maths against OpenCV, which implements the
    # same plumb_bob model independently. Skipped rather than failed if cv2 is
    # not installed: it is a test dependency, not a deploy one.
    try:
        import cv2
    except ImportError:
        print("  SKIP  cv2 cross-check (opencv not installed)")
    else:
        rng = np.random.default_rng(1)
        cam_pts = np.stack([rng.uniform(-0.6, 0.6, 40), rng.uniform(-0.4, 0.4, 40),
                            rng.uniform(0.4, 2.0, 40)])
        pts = cal.pos_base_cam[:, None] + cal.rot_base_cam @ cam_pts
        size = (848, 480)
        mine, _ = cal.project(pts.T, size)
        rvec, _ = cv2.Rodrigues(cal.rot_base_cam.T)
        tvec = -cal.rot_base_cam.T @ cal.pos_base_cam
        theirs, _ = cv2.projectPoints(pts.T, rvec, tvec,
                                      cal.scaled_k(size), cal.dist)
        err = np.abs(mine - theirs.reshape(-1, 2)).max()
        check("matches cv2.projectPoints", err < 1e-6, f"max {err:.2e} px")

    behind = cal.pos_base_cam + cal.rot_base_cam @ np.array([0.0, 0.0, -1.0])
    check("a point behind the camera is flagged", not cal.project(behind, (848, 480))[1][0])
    check("intrinsics scale with resolution",
          np.isclose(cal.scaled_k((960, 540))[0, 0], cal.k[0, 0] * 2))

    print("\n=== to_obs_pixels agrees with fit_image ===")
    # Track a real marker through the real resize and compare centroids.
    W, H = 848, 480
    box = (254, 130, 574, 450)
    cases = [("box", box, [(300, 200), (500, 400), (280, 150)]),
             ("crop", None, [(424, 240), (300, 100), (560, 400)]),
             ("pad", None, [(424, 240), (100, 60), (800, 440)])]
    for mode, b, points in cases:
        worst = 0.0
        for (u, v) in points:
            out = fit_image(marker_frame(W, H, u, v), mode, b)
            got = centroid(out)
            want = to_obs_pixels((u, v), (W, H), mode, b)[0]
            worst = max(worst, np.abs(got - want).max())
        check(f"{mode}: predicted pixel matches the resized image",
              worst < 1.5, f"worst {worst:.2f} px")

    # Outside the crop the mapping must leave the image, not wrap into it.
    outside = to_obs_pixels((10, 10), (W, H), "box", box)[0]
    check("box: a point outside the crop maps outside 0..224",
          outside[0] < 0 and outside[1] < 0, f"{outside.round(1)}")

    print("\n=== Overlay ===")
    ov = Overlay("cam1", "box", box)
    check("exterior overlay is available", ov.available)
    check("an uncalibrated camera gets no overlay",
          not Overlay("no_such_camera", "pad").available)

    class FakePolicy:
        speed_scale, max_step_rad = 1.0, MAX_JOINT_DELTA
        binarize_gripper, open_loop_horizon = True, 8
        n_inferences = 1
        last_samples = None
        _chunk_index = 3
        _chunk = np.zeros((10, 8))
        last_frame_shapes = {"exterior": (W, H)}
        last_request = {"observation/joint_position": READY_Q}

    pol = FakePolicy()
    payload = ov.build(pol, None)
    raw, crop = payload["views"]["cam1:raw"], payload["views"]["cam1:obs"]
    check("path has a point per chunk step plus the start",
          len(raw["path"]) == 11, f"{len(raw['path'])}")
    check("gripper track is the same length", len(payload["gripper"]) == 11)
    check("three axis tips in each view",
          len(raw["axes"]) == 3 and len(crop["axes"]) == 3)
    check("raw view carries the frame aspect and the crop box",
          abs(raw["aspect"] - W/H) < 1e-3 and raw["crop"] is not None)
    # The same world point in two views: inside the crop the two must agree
    # under the box transform, which is what makes the toggle trustworthy.
    x0, y0, x1, y1 = box
    for (ur, vr), (uo, vo) in zip(raw["path"], crop["path"]):
        want = ((ur*W - x0)/(x1 - x0), (vr*H - y0)/(y1 - y0))
        agree = abs(want[0] - uo) < 1e-4 and abs(want[1] - vo) < 1e-4
        if not agree:
            break
    check("raw and policy-input views describe the same point", agree)

    class FakeState:
        joint_values = READY_Q
        ee_position = chain.fk(READY_Q)[:3, 3]
        ee_quat_xyzw = Rotation.from_matrix(chain.fk(READY_Q)[:3, :3]).as_quat()

    payload = ov.build(pol, FakeState())
    check("reports the FK residual", payload["fk_residual_mm"] < 0.01,
          f"{payload['fk_residual_mm']} mm")
    check("'now' matches the path start when the arm is where the plan began",
          all(np.allclose(v["now"], v["path"][0], atol=1e-4)
              for v in payload["views"].values()))
    check("static camera is reported as base-mounted",
          payload["mounted_on"] == "base")

    # A stream calibrated under another name -- the cam1/cam3 case on this rig.
    aliased = Overlay("cam1", "box", box, calibration_key="cam3")
    check("calibration entry can be named separately from the stream",
          aliased.available and aliased.calibration.name == "cam3")
    check("aliased calibration actually projects differently",
          not np.allclose(aliased.build(pol, None)["views"]["cam1:raw"]["path"],
                          payload["views"]["cam1:raw"]["path"]))
    check("config/cameras.yaml supplies the calibration entry",
          Overlay("agentview", "box", box).calibration_key == "cam3")

    print("\n=== several sampled plans ===")
    # pi0.5 draws its plan from noise, so one observation gives several. The
    # panel draws them all; their spread is the policy's own uncertainty.
    rng = np.random.default_rng(4)
    pol.last_samples = np.concatenate([
        np.zeros((1, 10, 8)),
        rng.normal(0, 0.3, (5, 10, 8))])
    pol.n_inferences = 7
    payload = ov.build(pol, FakeState())
    raw = payload["views"]["cam1:raw"]
    check("every sample becomes a path", len(raw["samples"]) == 6, f"{len(raw['samples'])}")
    check("each is as long as the drawn path",
          all(len(x) == len(raw["path"]) for x in raw["samples"]))
    check("counted in the payload", payload["n_samples"] == 6)
    check("the spread is reported in mm",
          payload["spread_mm"] > 0, f"{payload['spread_mm']} mm")
    check("a still chunk spreads less than a noisy one",
          ov.build(_with_samples(pol, np.zeros((4, 10, 8)), 8),
                   FakeState())["spread_mm"] == 0.0)

    # Projecting N plans is N times the forward kinematics; the panel asks
    # ~30 times per inference, so the answer has to be cached.
    pol.last_samples = np.concatenate([np.zeros((1, 10, 8)),
                                       rng.normal(0, 0.3, (5, 10, 8))])
    pol.n_inferences = 21
    first = ov.build(pol, FakeState())["views"]["cam1:raw"]["samples"]
    second = ov.build(pol, FakeState())["views"]["cam1:raw"]["samples"]
    check("recomputed only when the policy re-infers", first is second)
    pol.n_inferences = 22
    check("and recomputed when it does",
          ov.build(pol, FakeState())["views"]["cam1:raw"]["samples"] is not first)

    pol.last_samples = None
    payload = ov.build(pol, FakeState())
    check("no samples -> nothing drawn and no spread",
          payload["n_samples"] == 0 and payload["spread_mm"] is None)
    check("a single sample is not a fan",
          ov.build(_with_samples(pol, np.zeros((1, 10, 8)), 9),
                   FakeState())["n_samples"] == 0)

    print("\n=== why the fan is or is not there ===")
    # An empty fan has three causes that look identical on screen: the server
    # cannot sample, the sampled inference has not come round yet, or the
    # policy simply agreed with itself. The payload has to distinguish them.
    from deploy.pi05.panel import FeatureAnalyser, build_payload

    class Args:
        external_camera, wrist_camera = "cam1", "cam4"
        exterior_fit, wrist_fit = "box", "pad"
        max_steps, speed_scale, rate, dry_run = 600, 1.0, 15.0, True

    pol.last_samples = None
    pol.n_samples, pol.server_has_samples, pol.samples_every = 6, False, 1
    pol.samples_asked = 0
    pol.last_inference_ms, pol.n_inferences, pol.last_features = 80.0, 3, {}
    pol.last_raw_frames = {}
    status = build_payload(pol, None, Args(), 1, FeatureAnalyser())["samples"]
    check("asked for but unavailable is visible in the payload",
          status["requested"] == 6 and not status["available"]
          and status["returned"] == 0, str(status))

    pol.samples_asked = 6
    status = build_payload(pol, None, Args(), 1, FeatureAnalyser())["samples"]
    check("asked-and-refused is distinguishable from never-asked",
          status["asked"] == 6 and not status["available"])

    pol.server_has_samples, pol.samples_every = True, 3
    status = build_payload(pol, None, Args(), 1, FeatureAnalyser())["samples"]
    check("available but not this inference is distinguishable",
          status["available"] and status["every"] == 3 and status["returned"] == 0)

    pol.last_samples = np.zeros((6, 10, 8))
    status = build_payload(pol, None, Args(), 1, FeatureAnalyser())["samples"]
    check("and returned plans are counted", status["returned"] == 6)
    pol.last_samples = None

    print("\n=== a camera that rides the arm ===")
    # A camera bolted to the gripper sees the gripper in the same place no
    # matter where the arm is. That invariant is the whole of the hand-eye
    # composition, and it fails loudly if the transform is applied the wrong
    # way round or against the wrong link.
    hand_eye = CameraCalibration(
        name="wristcam", k=cal.k.copy(), dist=np.zeros(5),
        # Looking down the gripper's +z, offset to one side: roughly how a
        # D405 sits on a Robotiq, and pointed so the fingers are in view.
        rot_base_cam=Rotation.from_euler("xyz", [0.0, 0.25, 0.0]).as_matrix(),
        pos_base_cam=np.array([0.05, -0.02, -0.04]),
        calib_size=(480, 270), parent=TIP_LINK)
    check("a parent link other than base means a moving camera", hand_eye.moving)
    try:
        hand_eye.project(np.zeros(3), (480, 270)); ok = False
    except ValueError:
        ok = True
    check("moving camera refuses to project without joint positions", ok)

    mount = Chain.from_urdf(tip=TIP_LINK)
    rng = np.random.default_rng(7)
    seen = []
    for _ in range(12):
        q = READY_Q + rng.normal(0, 0.3, 7)
        tip = (chain.fk(q) @ frame_offset("fingertip"))[:3, 3]
        uv, front = hand_eye.project(tip, (480, 270), mount, q)
        if front[0]:
            seen.append(uv[0])
    spread = np.abs(np.asarray(seen) - seen[0]).max()
    check("the gripper stays at one pixel however the arm moves",
          len(seen) > 8 and spread < 1e-9, f"{len(seen)} poses, spread {spread:.1e} px")
    # ...and a fixed point in the room does not.
    room = np.array([0.4, 0.0, 0.25])
    moved = [hand_eye.project(room, (480, 270), mount, READY_Q + d)[0][0]
             for d in (np.zeros(7), np.r_[0.2, np.zeros(6)])]
    check("a point in the room moves in the wrist view",
          np.abs(moved[0] - moved[1]).max() > 5, f"{np.round(moved[0]-moved[1], 1)}")

    # `frame:` in the YAML is what turns a static entry into a moving one.
    import tempfile, textwrap
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(textwrap.dedent(f"""\
            wristcam:
              width: 480
              height: 270
              frame: {TIP_LINK}
              k: [250.0, 0.0, 240.0, 0.0, 250.0, 135.0, 0.0, 0.0, 1.0]
              d: [0.0, 0.0, 0.0, 0.0, 0.0]
              t: [0.04, -0.03, -0.10]
              q: [0.0, 0.0, 0.0, 1.0]
            """))
        path = f.name
    loaded = CameraCalibration.load("wristcam", path)
    check("`frame:` is read from the calibration file",
          loaded.moving and loaded.parent == TIP_LINK, loaded.parent)
    check("a static entry still defaults to the base frame",
          not CameraCalibration.load("cam1").moving)

    print("\n=== ToolProjector (what the robot panel draws) ===")
    from control.projection import ToolProjector

    check("an uncalibrated camera yields no projector",
          ToolProjector.for_camera("no_such_camera") is None)
    tp = ToolProjector.for_camera("agentview")
    check("agentview resolves through cameras.yaml to its entry",
          tp is not None and tp.calibration.name == "cam3" and not tp.calibration.moving)
    out = tp.project(READY_Q, (848, 480))
    check("projects a point and three axis tips",
          len(out["point"]) == 2 and len(out["axes"]) == 3)
    check("says whether it is in frame", isinstance(out["in_frame"], bool))
    before = out["point"]
    other = tp.project(READY_Q, (848, 480), frame="ee")
    check("the ee frame projects somewhere else", other["point"] != before)
    check("asking for one frame does not change the next caller's",
          tp.project(READY_Q, (848, 480))["point"] == before)

    wrist = ToolProjector.for_camera("inhand")
    if wrist is None:
        print("  SKIP  in-hand invariant (no calibration entry yet)")
    else:
        # The real hand-eye entry, checked the way it will fail if it is
        # wrong: a camera bolted to the gripper sees the gripper in the same
        # place whatever the arm is doing.
        rng = np.random.default_rng(11)
        seen = [wrist.project(READY_Q + rng.normal(0, 0.25, 7), (480, 270))["point"]
                for _ in range(10)]
        spread = np.abs(np.asarray(seen) - np.asarray(seen[0])).max()
        check("the fingertip is fixed in the in-hand view",
              spread < 1e-9, f"{spread:.1e} of a frame over 10 poses")

    print("\n=== merging cameras ===")
    merged = merge([payload, {"camera": "wristcam", "views": {"wristcam:raw": {}},
                              "order": ["wristcam:raw"]}])
    check("views from both cameras are present",
          set(merged["order"]) == {"cam1:raw", "cam1:obs", "wristcam:raw"})
    check("shared facts come from the first payload",
          merged["fk_residual_mm"] == payload["fk_residual_mm"])
    check("merge of nothing is None", merge([None, None]) is None)

    print("\n=== Overlay, continued ===")
    pol._chunk = None
    check("no chunk -> no overlay", ov.build(pol, FakeState()) is None)
    pol._chunk, pol.last_frame_shapes = np.zeros((4, 8)), {}
    check("no frame size -> no overlay", ov.build(pol, FakeState()) is None)

    # Consistency with the pixel mapping: with the policy's own fit settings,
    # a point at the crop centre must land at the centre of the observation.
    centre_uv = to_obs_pixels(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                              (W, H), "box", box)[0]
    check("crop centre maps to the image centre",
          np.allclose(centre_uv, [IMAGE_SIZE / 2, IMAGE_SIZE / 2], atol=1e-6))

    print("\nAll overlay tests passed.")


if __name__ == "__main__":
    main()
