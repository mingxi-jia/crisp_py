#!/usr/bin/env python3
"""Hand-eye calibration for the in-hand camera, flown by spacemouse.

The wrist camera rides the gripper, so it has no fixed pose in the robot base
frame -- only a fixed pose relative to the link it is bolted to. That transform
is what this measures, and it is the one thing config/camera_info.yaml cannot
be filled in without: everything else there (intrinsics) is factory data.

How it goes
-----------
1. Put a target where the wrist camera can see it and leave it alone. It must
   not move: the whole method rests on it being the one thing that doesn't.
2. Fly the arm with the spacemouse to a pose where the target is in view, let
   the arm settle, press the capture button. Repeat 10-20 times, and *rotate
   the wrist between captures* -- around two different axes, not just one.
   A set of poses that only differ by translation cannot determine the answer
   and the solver will not say so; the panel's coverage meter will.
3. Press solve. The panel shows the intrinsics fit, how still the target
   stayed once the solution is applied (the honest error measure), and draws
   the gripper's own fingertip back onto the live image -- if that lands on
   the fingers from every angle, the calibration is right.
4. Save. The YAML entry goes to the output directory, and with --write is
   appended to config/camera_info.yaml.

Running it
----------
The robot server must be in *cartesian* control space, since this flies the
arm the way deploy/deploy_spacemouse.py does:

    ./scripts/start_all.sh --stop && ./scripts/start_all.sh

    python -m scripts.calibrate_handeye --target aruco --marker-size 0.048

Intrinsics
----------
A single ArUco tag gives four points per view, which pins down the board pose
but not the lens. Take the camera's factory intrinsics instead -- stop the
robot server so the device is free, then:

    python -m scripts.calibrate_handeye --probe-intrinsics

and pass the printed entry back with --intrinsics. A charuco or checkerboard
target has enough points per view to solve the lens too, and then intrinsics
are estimated from the same captures with no extra step.

Controls
--------
    spacemouse            fly the fingertip
    button 0              capture a sample (only when the arm has settled)
    button 1              solve
    'r'                   re-home and re-anchor
    the web panel         same actions, plus delete/clear and save
    Ctrl+C                quit
"""

import argparse
import base64
import dataclasses
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):            # allow `python scripts/calibrate_handeye.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from control.robot_client import RobotClient
from control.teleop import SpacemouseConfig, SpacemousePolicy
from scripts.handeye import (
    MIN_SAMPLES, HandEyeResult, invert, make_target, rigid, solve, yaml_block,
)

PANEL_HTML = Path(__file__).with_name("calibrate_handeye.html")
CAMERA_INFO = Path(__file__).resolve().parent.parent / "config" / "camera_info.yaml"
DEFAULT_FRAME = "fr3_hand_tcp"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:7000",
                   help="robot_server base URL")
    p.add_argument("--camera", default="inhand",
                   help="which camera to calibrate (a name from config/cameras.yaml)")
    p.add_argument("--entry", default=None, metavar="NAME",
                   help="name to write the result under in camera_info.yaml "
                        "(default: the camera's `calibration:`, else its name)")
    p.add_argument("--frame", default=DEFAULT_FRAME,
                   help="the link the camera is bolted to. Must be the frame "
                        "the robot reports poses in, since those poses are "
                        "what the solve is built on")

    g = p.add_argument_group("target")
    g.add_argument("--target", choices=["aruco", "charuco", "checkerboard"],
                   default="aruco",
                   help="aruco: one printed marker, the least to set up. "
                        "charuco/checkerboard: many more points per view, so "
                        "they also determine the lens")
    g.add_argument("--marker-size", type=float, default=None, metavar="M",
                   help="side of the ArUco marker's black square [m]. Measure "
                        "it: every distance in the answer scales with it")
    g.add_argument("--marker-id", type=int, default=None,
                   help="which marker to use, if several are in view "
                        "(default: the largest)")
    g.add_argument("--dictionary", default="DICT_ARUCO_ORIGINAL",
                   help="ArUco dictionary of the marker or charuco board")
    g.add_argument("--rows", type=int, default=5,
                   help="charuco: squares down. checkerboard: inner corners down")
    g.add_argument("--cols", type=int, default=7,
                   help="charuco: squares across. checkerboard: inner corners across")
    g.add_argument("--square", type=float, default=0.03,
                   help="charuco/checkerboard square size [m]")

    g = p.add_argument_group("intrinsics")
    g.add_argument("--intrinsics", default=None, metavar="NAME",
                   help="take k/d from this entry of camera_info.yaml instead "
                        "of estimating them from the captures")
    g.add_argument("--estimate-intrinsics", action="store_true",
                   help="estimate k/d from the captures even for a "
                        "single-marker target, where it is ill-conditioned")
    g.add_argument("--probe-intrinsics", action="store_true",
                   help="read the camera's factory intrinsics off the device, "
                        "print them (--write puts them straight into "
                        "camera_info.yaml) and exit. Needs the device free: "
                        "stop the robot server first")

    g = p.add_argument_group("solve and output")
    g.add_argument("--method", default="TSAI",
                   choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"],
                   help="cv2.calibrateHandEye solver")
    g.add_argument("--out", default=None, metavar="DIR",
                   help="where to save samples and the result "
                        "(default: calibration/<camera>-<timestamp>)")
    g.add_argument("--replay", default=None, metavar="DIR",
                   help="re-solve a saved session and exit. No robot needed")
    g.add_argument("--write", action="store_true",
                   help="append the solved entry to config/camera_info.yaml "
                        "when saving")
    g.add_argument("--cross-check", default=None, metavar="CAMERA",
                   help="a second, already-calibrated camera that can also see "
                        "the target. Where it puts the target is compared with "
                        "where this calibration puts it -- an error measure "
                        "that shares nothing with the solve")

    g = p.add_argument_group("flying")
    g.add_argument("--rate", type=float, default=50.0, help="teleop rate [Hz]")
    g.add_argument("--action-size", type=float, default=0.003,
                   help="per-step translation gain [m]")
    g.add_argument("--deadzone", type=float, default=0.1)
    g.add_argument("--settle-mm", type=float, default=1.5,
                   help="the arm counts as still when it has moved less than "
                        "this over --settle-window [mm]. A blurred frame "
                        "captured mid-move is a bad sample that looks fine")
    g.add_argument("--settle-window", type=float, default=0.4,
                   help="how long the arm must have been still [s]")
    g.add_argument("--capture-button", type=int, default=0)
    g.add_argument("--solve-button", type=int, default=1)
    g.add_argument("--panel-port", type=int, default=7150)
    g.add_argument("--view-hz", type=float, default=5.0,
                   help="how often the panel's live view is refreshed")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Sample:
    """One capture: an image, and exactly where the arm was when it was taken.

    Image and pose come out of a single /get_obs response rather than two
    calls, so they cannot be a control period apart -- at 50 Hz that would be
    millimetres of lever arm error baked into the answer.
    """

    index: int
    rgb: np.ndarray
    joints: np.ndarray
    ee_position: np.ndarray
    ee_quat_xyzw: np.ndarray
    object_points: np.ndarray | None = None
    image_points: np.ndarray | None = None

    @property
    def detected(self) -> bool:
        return self.image_points is not None

    def save(self, directory: Path):
        np.savez_compressed(
            directory / f"sample_{self.index:03d}.npz",
            rgb=self.rgb, joints=self.joints, ee_position=self.ee_position,
            ee_quat_xyzw=self.ee_quat_xyzw)

    @classmethod
    def load(cls, path: Path, index: int) -> "Sample":
        d = np.load(path)
        return cls(index=index, rgb=d["rgb"], joints=d["joints"],
                   ee_position=d["ee_position"], ee_quat_xyzw=d["ee_quat_xyzw"])


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
def jsonable(value):
    """numpy scalars and arrays -> plain Python, recursively.

    Flask's encoder rejects np.bool_ and np.float64, and a single one
    anywhere in the payload turns the whole panel into a 500. Numbers pass
    through this module from several directions, so the sanitising happens
    once at the boundary rather than being remembered at every call site.
    """
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    return value


class Session:
    """What the control loop writes and the panel reads. One lock, held briefly."""

    def __init__(self, target, args, camera_size=None):
        self.target = target
        self.args = args
        self.camera_size = camera_size
        self.samples: list[Sample] = []
        self.result: HandEyeResult | None = None
        self.message = "fly the arm until the target is in view, then capture"
        self.live_jpeg = None
        self.live_detected = False
        self.live_points = 0
        self.still = False
        self.saved_to = None
        self.cross_check = None
        self._lock = threading.Lock()
        self._requests = []

    # -- from the control loop ------------------------------------------
    def set_live(self, jpeg, detected, points, still):
        with self._lock:
            self.live_jpeg = jpeg
            self.live_detected = detected
            self.live_points = points
            self.still = still

    def add(self, sample: Sample) -> str:
        from scripts.handeye import detect_into

        found = detect_into(self.target, sample)
        with self._lock:
            self.samples.append(sample)
            self.result = None          # a new sample invalidates the old solve
            self.message = (f"captured #{sample.index} with {found} target points"
                            if found else
                            f"captured #{sample.index} but the target was not "
                            f"found -- it will be ignored")
            return self.message

    # -- from the panel --------------------------------------------------
    def request(self, what: str):
        with self._lock:
            self._requests.append(what)

    def take_requests(self) -> list:
        with self._lock:
            out, self._requests = self._requests, []
            return out

    def delete_last(self):
        with self._lock:
            if self.samples:
                dropped = self.samples.pop()
                self.result = None
                self.message = f"deleted #{dropped.index}"

    def clear(self):
        with self._lock:
            self.samples.clear()
            self.result = None
            self.message = "cleared"

    # -- solving ---------------------------------------------------------
    def detected_samples(self) -> list:
        with self._lock:
            return [s for s in self.samples if s.detected]

    def run_solve(self, fixed_k=None, fixed_dist=None) -> str:
        usable = self.detected_samples()
        if len(usable) < MIN_SAMPLES:
            msg = (f"{len(usable)} usable sample(s); need {MIN_SAMPLES}. "
                   f"Capture more, from different wrist angles")
            with self._lock:
                self.message = msg
            return msg
        try:
            result = solve(usable, self.camera_size, fixed_k=fixed_k,
                           fixed_dist=fixed_dist, method=self.args.method)
        except Exception as e:
            with self._lock:
                self.message = f"solve failed: {e}"
            return self.message
        ok, why = result.verdict()
        with self._lock:
            self.result = result
            self.message = ("solved — looks good" if ok else f"solved, but: {why}")
        return self.message

    # -- reporting -------------------------------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            r = self.result
            out = {
                "message": self.message,
                "n_samples": len(self.samples),
                "n_usable": sum(1 for s in self.samples if s.detected),
                "min_samples": MIN_SAMPLES,
                "samples": [{"index": s.index, "detected": s.detected,
                             "points": 0 if s.image_points is None
                             else len(s.image_points)} for s in self.samples],
                "detected": self.live_detected,
                "points": self.live_points,
                "still": self.still,
                "target": str(self.target),
                "camera": self.args.camera,
                "frame": self.args.frame,
                "saved_to": self.saved_to,
                "cross_check": self.cross_check,
                "result": None,
            }
            if r is not None:
                ok, why = r.verdict()
                out["result"] = {
                    "ok": ok, "why": why,
                    "position": [round(v, 5) for v in r.position.tolist()],
                    "quat_xyzw": [round(v, 6) for v in r.quat_xyzw.tolist()],
                    "rms": round(r.reprojection_rms, 3),
                    "spread_mm": round(r.spread_mm, 2),
                    "spread_deg": round(r.spread_deg, 3),
                    "per_sample_mm": r.per_sample_mm,
                    "coverage": r.coverage,
                    "n": r.n_samples,
                    "method": r.method,
                    "fx": round(float(r.k[0, 0]), 2),
                    "fy": round(float(r.k[1, 1]), 2),
                    "cx": round(float(r.k[0, 2]), 2),
                    "cy": round(float(r.k[1, 2]), 2),
                    "yaml": yaml_block(r, self.args.entry, self.args.frame),
                }
            return jsonable(out)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def annotate(rgb, target, result=None, frame_chain=None, joints=None,
             tool_offset=None, quality: int = 75):
    """The live view, with whatever there is to show drawn on it -> JPEG bytes.

    Before a solve: the detected target points, so it is obvious whether the
    camera can actually see it from here. After one: the gripper's own
    fingertip, projected through the solved transform. That second one is the
    check that cannot be fooled -- the fingers are in the picture, and either
    the cross sits on them from every angle or the calibration is wrong.
    """
    import cv2

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    found = target.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if found is not None:
        obj, img = found
        for (x, y) in img:
            cv2.circle(bgr, (int(round(x)), int(round(y))), 4, (0, 0, 0), -1)
            cv2.circle(bgr, (int(round(x)), int(round(y))), 3, (80, 220, 120), -1)
        if len(img) >= 4:
            hull = cv2.convexHull(img.astype(np.int32))
            cv2.polylines(bgr, [hull], True, (80, 220, 120), 1, cv2.LINE_AA)

    if result is not None and frame_chain is not None and joints is not None:
        # The fingertip in camera coordinates: it is rigidly attached to the
        # same link the camera is, so this does not depend on the arm's pose.
        T_base_tip = frame_chain.fk(joints) @ tool_offset
        T_base_cam = frame_chain.fk(joints) @ result.T_gripper_cam
        tip_cam = invert(T_base_cam) @ T_base_tip
        if tip_cam[2, 3] > 1e-6:
            pts, _ = cv2.projectPoints(
                np.stack([tip_cam[:3, 3],
                          tip_cam[:3, 3] + tip_cam[:3, :3] @ [0.02, 0, 0],
                          tip_cam[:3, 3] + tip_cam[:3, :3] @ [0, 0.02, 0],
                          tip_cam[:3, 3] + tip_cam[:3, :3] @ [0, 0, 0.02]]),
                np.zeros(3), np.zeros(3), result.k, result.dist)
            pts = pts.reshape(-1, 2)
            o = tuple(int(round(v)) for v in pts[0])
            for i, colour in enumerate([(60, 60, 255), (60, 255, 60), (255, 190, 60)]):
                end = tuple(int(round(v)) for v in pts[i + 1])
                cv2.line(bgr, o, end, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.line(bgr, o, end, colour, 2, cv2.LINE_AA)
            cv2.drawMarker(bgr, o, (255, 255, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            cv2.putText(bgr, "fingertip", (o[0] + 10, o[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(bgr, "fingertip", (o[0] + 10, o[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else None


def thumbnail(rgb, width: int = 160, quality: int = 60):
    import cv2

    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (width, max(1, int(round(h * width / w)))))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
def serve(session: Session, port: int) -> threading.Thread:
    """Start the calibration panel in a daemon thread."""
    from flask import Flask, Response, jsonify, request

    app = Flask(__name__)
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.route("/")
    def index():
        return Response(PANEL_HTML.read_text(), mimetype="text/html")

    @app.route("/state")
    def state():
        return jsonify(session.snapshot())

    @app.route("/live.jpg")
    def live():
        jpeg = session.live_jpeg
        if not jpeg:
            return Response(status=503)
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.route("/sample/<int:index>.jpg")
    def sample(index):
        for s in session.samples:
            if s.index == index:
                return Response(thumbnail(s.rgb), mimetype="image/jpeg")
        return Response(status=404)

    @app.route("/do/<action>", methods=["POST"])
    def do(action):
        if action not in ("capture", "solve", "delete_last", "clear", "save"):
            return jsonify({"ok": False, "error": f"unknown action {action}"}), 400
        session.request(action)
        return jsonify({"ok": True})

    thread = threading.Thread(
        target=lambda: app.run(host="0.0.0.0", port=port, threaded=True,
                               debug=False, use_reloader=False),
        daemon=True)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# Intrinsics from the device
# ---------------------------------------------------------------------------
def probe_intrinsics(camera: str, entry: str = None, write: bool = False,
                     path=None) -> int:
    """Print the camera's factory intrinsics as a YAML fragment.

    Needs the device: only one process may hold a RealSense, so the robot
    server has to be stopped first. Nothing here is a measurement -- these
    numbers are burned into the camera at the factory.
    """
    from control.realsense_direct import load_camera_config

    specs, _ = load_camera_config()
    spec = next((s for s in specs if s.name == camera), None)
    if spec is None:
        print(f"no camera {camera!r} in config/cameras.yaml "
              f"(have: {', '.join(s.name for s in specs) or 'none'})")
        return 1

    import pyrealsense2 as rs

    cfg = rs.config()
    cfg.enable_device(spec.serial)
    cfg.enable_stream(rs.stream.color, spec.width, spec.height, rs.format.rgb8, spec.fps)
    pipe = rs.pipeline()
    try:
        profile = pipe.start(cfg)
    except RuntimeError as e:
        print(f"could not open {camera} ({spec.serial}): {e}\n"
              f"Only one process may own a RealSense. Stop the robot server:\n"
              f"    ./scripts/start_all.sh --stop")
        return 1
    try:
        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
    finally:
        pipe.stop()

    lines = [
        f"# {camera} ({spec.serial}) factory intrinsics at "
        f"{intr.width}x{intr.height}, model {intr.model}. Read off the device, "
        f"not measured.",
        "# Extrinsics (frame/t/q) come from scripts/calibrate_handeye.py; until "
        "they are here",
        "# the panel will not draw a trajectory on this camera.",
        f"{entry or camera}:",
        f"  height: {intr.height}",
        f"  width: {intr.width}",
        "  distortion_model: plumb_bob",
        "  d:",
    ]
    lines += [f"    - {float(v)!r}" for v in list(intr.coeffs)[:5]]
    lines.append("  k:")
    lines += [f"    - {float(v)!r}" for v in
              (intr.fx, 0.0, intr.ppx, 0.0, intr.fy, intr.ppy, 0.0, 0.0, 1.0)]
    block = "\n".join(lines) + "\n"
    print(block)
    if str(intr.model) not in ("distortion.brown_conrady",
                               "distortion.inverse_brown_conrady",
                               "distortion.none"):
        print(f"# NOTE: this stream's distortion model is {intr.model}, which is "
              f"not plumb_bob. The projection code assumes plumb_bob.")
    if write:
        print(upsert_entry(Path(path or CAMERA_INFO), entry or camera, block))
    else:
        print(f"# --write puts this into {path or CAMERA_INFO} for you.")
    return 0


# ---------------------------------------------------------------------------
# Saving and cross-checking
# ---------------------------------------------------------------------------
def save_session(session: Session, out_dir: Path, write_config: bool,
                 camera_info: Path) -> str:
    """Samples, the result and a paste-ready YAML entry -> `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in session.samples:
        s.save(out_dir)
    meta = {
        "camera": session.args.camera,
        "entry": session.args.entry,
        "frame": session.args.frame,
        "target": str(session.target),
        "target_kind": session.target.kind,
        "image_size": list(session.camera_size),
        "n_samples": len(session.samples),
        "argv": sys.argv[1:],
    }
    (out_dir / "session.json").write_text(json.dumps(meta, indent=2))

    result = session.result
    if result is None:
        return f"saved {len(session.samples)} samples to {out_dir} (not solved)"

    block = yaml_block(result, session.args.entry, session.args.frame)
    (out_dir / "camera_info_entry.yaml").write_text(block)
    np.savez(out_dir / "result.npz", k=result.k, dist=result.dist,
             T_gripper_cam=result.T_gripper_cam, T_base_target=result.T_base_target)
    message = f"saved to {out_dir}"

    if write_config:
        message += "; " + upsert_entry(camera_info, session.args.entry, block)
    return message


def cross_check(session: Session, robot, camera: str) -> dict | None:
    """Compare this calibration against an already-calibrated camera.

    Both cameras can see the same target. Its pose in the base frame, worked
    out through the wrist camera and through the static one, should be the
    same place. Nothing in this comparison was used by the solve, so unlike
    every other number on the panel it can fail independently.
    """
    import cv2

    from deploy.pi05.overlay import CameraCalibration

    result = session.result
    if result is None:
        return None
    calib = CameraCalibration.load(_calibration_entry(camera))
    if calib is None:
        return {"error": f"{camera} has no calibration entry either"}

    obs = robot.get_obs(include_depth=False)
    rgb = obs.rgb(camera)
    if rgb is None:
        return {"error": f"{camera} has no frame"}
    found = session.target.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if found is None:
        return {"error": f"the target is not visible from {camera}"}

    obj, img = found
    size = (rgb.shape[1], rgb.shape[0])
    ok, rvec, tvec = cv2.solvePnP(obj.reshape(-1, 1, 3), img.reshape(-1, 1, 2),
                                  calib.scaled_k(size), calib.dist)
    if not ok:
        return {"error": "solvePnP failed on the reference camera"}

    # Target in the base frame, the long way round: through the static camera.
    T_ref_target = rigid(cv2.Rodrigues(rvec)[0], tvec)
    T_base_ref = rigid(calib.rot_base_cam, calib.pos_base_cam)
    via_reference = T_base_ref @ T_ref_target
    delta = via_reference[:3, 3] - session.result.T_base_target[:3, 3]
    return {
        "camera": camera,
        "distance_mm": round(float(np.linalg.norm(delta) * 1000.0), 1),
        "delta_mm": [round(float(v) * 1000.0, 1) for v in delta],
        "note": f"where {camera} says the target is, minus where this "
                f"calibration says it is. {camera}'s own calibration error is "
                f"included, so this is an upper bound on both",
    }


def _calibration_entry(camera: str) -> str:
    from control.realsense_direct import calibration_key
    return calibration_key(camera)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def upsert_entry(path: Path, name: str, block: str) -> str:
    """Put `block` into `path` as the `name:` entry, replacing any that is there.

    Calibration comes in two passes -- intrinsics off the device, then the
    hand-eye solve -- and both write the same key. Appending twice would leave
    a duplicate that YAML resolves silently by taking the last one, which is a
    good way to spend an afternoon wondering why an edit did nothing.
    """
    text = path.read_text()
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.strip() and not line.startswith((" ", "\t", "#"))
                  and line.split(":")[0].strip() == name), None)
    if start is None:
        path.write_text(text.rstrip("\n") + "\n\n" + block)
        return f"added {name!r} to {path}"

    # Everything up to the next top-level key belongs to this entry.
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.startswith((" ", "\t", "#")):
            end = i
            break
    # Each write puts a "# Hand-eye calibration: ..." header above the entry.
    # Without taking the old ones with it, a third attempt leaves three
    # contradictory provenance lines and no way to tell which one is true.
    ours = ("# Hand-eye calibration:", "# Extrinsics (frame/t/q)")
    while start > 0 and (lines[start - 1].startswith(ours)
                         or " factory intrinsics at " in lines[start - 1]):
        start -= 1
    kept = lines[:start] + block.rstrip("\n").splitlines() + [""] + lines[end:]
    path.write_text("\n".join(kept).rstrip("\n") + "\n")
    return f"replaced {name!r} in {path}"


def read_intrinsics_entry(name: str, path: Path):
    """(k, dist, (width, height)) from a camera_info.yaml entry, or None.

    Deliberately not CameraCalibration.load: that needs `t`/`q` as well, and
    the whole point of the first pass is an entry that has intrinsics and no
    extrinsics yet.
    """
    import yaml

    if not path.exists():
        return None
    cfg = (yaml.safe_load(path.read_text()) or {}).get(name)
    if not cfg or "k" not in cfg:
        return None
    return (np.asarray(cfg["k"], dtype=np.float64).reshape(3, 3),
            np.asarray(cfg.get("d") or [0.0] * 5, dtype=np.float64).ravel()[:5],
            (int(cfg["width"]), int(cfg["height"])))


def load_fixed_intrinsics(entry: str, size, path=CAMERA_INFO):
    """k/d from an existing camera_info.yaml entry, scaled to this stream."""
    import yaml

    found = read_intrinsics_entry(entry, path)
    if found is None:
        have = sorted((yaml.safe_load(path.read_text()) or {}).keys()) \
            if path.exists() else []
        raise SystemExit(
            f"--intrinsics {entry!r}: {path} has no entry by that name with "
            f"intrinsics in it.\n"
            f"Entries present: {', '.join(have) or 'none'}\n"
            f"The in-hand camera's intrinsics are factory data, not something "
            f"to measure. Stop the robot server so the device is free, then "
            f"write them in with one command:\n"
            f"    ./scripts/start_all.sh --stop\n"
            f"    python -m scripts.calibrate_handeye --probe-intrinsics --write\n"
            f"    ./scripts/start_all.sh\n"
            f"and run this again.")
    k, dist, calib_size = found
    k = k.copy()
    k[0, :] *= size[0] / calib_size[0]
    k[1, :] *= size[1] / calib_size[1]
    return k, dist


def resolve_intrinsics(args, target, size):
    """(k, dist) to solve with, or (None, None) to estimate them.

    Raises SystemExit with the way out when neither is possible: a four-point
    target cannot determine a lens, and silently trying anyway produces a
    plausible-looking answer that is wrong.
    """
    from scripts.handeye import points_per_view_enough_for_intrinsics

    if args.intrinsics:
        k, dist = load_fixed_intrinsics(args.intrinsics, size)
        print(f"  intrinsics: fixed, from the {args.intrinsics!r} entry "
              f"(fx {k[0,0]:.1f}, fy {k[1,1]:.1f})")
        return k, dist
    if points_per_view_enough_for_intrinsics(target):
        print("  intrinsics: estimated from the captures")
        return None, None
    if args.estimate_intrinsics:
        print("  intrinsics: estimated from the captures\n"
              "              NOTE: four points per view barely constrain a "
              "lens. Expect focal length and distortion to trade off against "
              "each other; --intrinsics is better.")
        return None, None
    raise SystemExit(
        f"a single ArUco marker gives four points per view, which fixes the "
        f"marker's pose but not the lens.\n"
        f"Get the camera's factory intrinsics instead -- stop the robot server "
        f"so the device is free, then:\n"
        f"    python -m scripts.calibrate_handeye --probe-intrinsics "
        f"--camera {args.camera}\n"
        f"paste that into config/camera_info.yaml and pass --intrinsics <entry>.\n"
        f"Or use --target charuco / --target checkerboard, which have enough "
        f"points to solve the lens too, or --estimate-intrinsics to try anyway.")


def replay(args) -> int:
    """Re-solve a saved session without any hardware."""
    from scripts.handeye import detect_into

    directory = Path(args.replay)
    meta = json.loads((directory / "session.json").read_text())
    target = make_target(args.target, rows=args.rows, cols=args.cols,
                         square=args.square, dictionary=args.dictionary,
                         marker_size=args.marker_size, marker_id=args.marker_id)
    samples = []
    for i, path in enumerate(sorted(directory.glob("sample_*.npz"))):
        s = Sample.load(path, i)
        detect_into(target, s)
        samples.append(s)
    usable = [s for s in samples if s.detected]
    print(f"{directory}: {len(samples)} samples, {len(usable)} with the target "
          f"({target})")

    size = tuple(meta.get("image_size") or
                 (samples[0].rgb.shape[1], samples[0].rgb.shape[0]))
    fixed_k, fixed_dist = resolve_intrinsics(args, target, size)
    if len(usable) < MIN_SAMPLES:
        print(f"\nonly {len(usable)} usable sample(s); {MIN_SAMPLES} are "
              f"needed. Nothing to solve.")
        return 1
    try:
        result = solve(usable, size, fixed_k=fixed_k, fixed_dist=fixed_dist,
                       method=args.method)
    except Exception as e:
        print(f"\nsolve failed: {e}")
        return 1
    ok, why = result.verdict()
    print(f"\nreprojection {result.reprojection_rms:.3f} px | target held still "
          f"to {result.spread_mm:.1f} mm / {result.spread_deg:.2f} deg | "
          f"coverage {result.coverage['axes']}")
    print("GOOD" if ok else f"QUESTIONABLE: {why}")
    print()
    print(yaml_block(result, args.entry or meta.get("entry") or "cam4",
                     args.frame))
    return 0 if ok else 2


def main():
    args = parse_args()

    if args.entry is None:
        args.entry = _calibration_entry(args.camera)

    if args.probe_intrinsics:
        return probe_intrinsics(args.camera, args.entry, args.write)

    if args.replay:
        return replay(args)

    from scripts.handeye import points_per_view_enough_for_intrinsics
    from deploy.pi05.overlay import Chain, frame_offset

    target = make_target(args.target, rows=args.rows, cols=args.cols,
                         square=args.square, dictionary=args.dictionary,
                         marker_size=args.marker_size, marker_id=args.marker_id)

    robot = RobotClient(args.url)

    # Flying the arm needs the cartesian controller, exactly as teleop does.
    ctrl_space = robot.ctrl_space
    if ctrl_space is not None and ctrl_space != "cartesian":
        raise SystemExit(
            f"the robot server at {args.url} is in '{ctrl_space}' control "
            f"space, but this flies the arm with Cartesian pose targets that "
            f"the joint controller ignores.\n"
            f"Restart it in Cartesian space:\n"
            f"    ./scripts/start_all.sh --stop\n"
            f"    ./scripts/start_all.sh")

    obs = robot.get_obs(include_depth=False)
    rgb = obs.rgb(args.camera)
    if rgb is None:
        raise SystemExit(
            f"no frames from camera {args.camera!r}. "
            f"Cameras with frames: {obs.ready_cameras or 'none'}")
    size = (rgb.shape[1], rgb.shape[0])

    fixed_k, fixed_dist = resolve_intrinsics(args, target, size)

    chain = Chain.from_urdf(tip=args.frame)
    tool_offset = frame_offset("fingertip")

    out_dir = Path(args.out or
                   f"calibration/{args.camera}-{time.strftime('%Y%m%d-%H%M%S')}")

    session = Session(target, args, camera_size=size)
    serve(session, args.panel_port)
    print(f"\n  camera:  {args.camera} {size[0]}x{size[1]} -> entry "
          f"{args.entry!r}, mounted on {args.frame}")
    print(f"  target:  {target}")
    print(f"  panel:   http://localhost:{args.panel_port}/")
    print(f"  output:  {out_dir}")
    print("\n  fly with the spacemouse; button "
          f"{args.capture_button} captures, button {args.solve_button} solves.")
    print("  rotate the wrist between captures -- about two different axes, "
          "not just one.\n")

    policy = SpacemousePolicy(SpacemouseConfig(
        action_size=args.action_size, deadzone=args.deadzone,
        # The buttons are wanted for capture and solve, so the gripper toggle
        # is pointed at a button that does not exist.
        gripper_button=-1))

    stop = threading.Event()
    counter = {"n": 0}
    recent = []               # (t, position) history, for the settle test

    def is_still() -> bool:
        cutoff = time.time() - args.settle_window
        window = [p for t, p in recent if t >= cutoff]
        if len(window) < 3 or (recent and recent[0][0] > cutoff):
            return False
        spread = np.ptp(np.asarray(window), axis=0)
        return bool(np.linalg.norm(spread) * 1000.0 < args.settle_mm)

    def view_loop():
        """Live view, detection and captures. One /get_obs per tick, so an
        image and the pose that goes with it always come from one response."""
        period = 1.0 / max(0.5, args.view_hz)
        while not stop.is_set():
            tick = time.perf_counter()
            try:
                obs = robot.get_obs(include_depth=False)
            except Exception as e:
                session.message = f"robot server: {e}"
                time.sleep(0.5)
                continue
            rgb = obs.rgb(args.camera)
            if rgb is not None:
                still = is_still()
                for what in session.take_requests():
                    handle(what, obs, rgb, still)
                jpeg = annotate(rgb, target, session.result, chain,
                                np.asarray(obs.joint_values)[:7], tool_offset)
                found = target.detect(
                    __import__("cv2").cvtColor(rgb, __import__("cv2").COLOR_RGB2GRAY))
                session.set_live(jpeg, found is not None,
                                 0 if found is None else len(found[1]), still)
            time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    def handle(what, obs, rgb, still):
        if what == "capture":
            if not still:
                session.message = ("hold still -- the arm was still moving "
                                   "(a blurred frame is a bad sample that "
                                   "looks like a good one)")
                return
            counter["n"] += 1
            sample = Sample(index=counter["n"], rgb=np.array(rgb, copy=True),
                            joints=np.asarray(obs.joint_values)[:7].copy(),
                            ee_position=np.asarray(obs.ee_position).copy(),
                            ee_quat_xyzw=np.asarray(obs.ee_quat_xyzw).copy())
            print("  " + session.add(sample))
        elif what == "solve":
            print("  " + session.run_solve(fixed_k, fixed_dist))
            if session.result is not None and args.cross_check:
                session.cross_check = cross_check(session, robot, args.cross_check)
                if session.cross_check:
                    print(f"  cross-check vs {args.cross_check}: "
                          f"{session.cross_check.get('error') or str(session.cross_check['distance_mm']) + ' mm'}")
        elif what == "delete_last":
            session.delete_last()
            print("  " + session.message)
        elif what == "clear":
            session.clear()
        elif what == "save":
            message = save_session(session, out_dir, args.write,
                                   CAMERA_INFO)
            session.saved_to = str(out_dir)
            session.message = message
            print("  " + message)

    viewer = threading.Thread(target=view_loop, daemon=True)
    viewer.start()

    period = 1.0 / args.rate
    prev_buttons = {args.capture_button: False, args.solve_button: False}

    with policy:
        policy.reset(robot.get_state().fingertip_pose)
        try:
            while True:
                tick = time.perf_counter()
                state = robot.get_state()
                recent.append((time.time(), np.asarray(state.ee_position).copy()))
                del recent[:-200]

                action = policy.predict_action(state)
                if action.reset:
                    print("  [r] re-homing...")
                    robot.home()
                    policy.reset(robot.get_state().fingertip_pose)
                    continue
                robot.servo(action.position, action.quat_xyzw)

                # Buttons act on the rising edge; the view thread does the work
                # so that a capture always comes from a fresh, whole /get_obs.
                device = policy._spacemouse
                for button, what in ((args.capture_button, "capture"),
                                     (args.solve_button, "solve")):
                    pressed = bool(device.is_button_pressed(button))
                    if pressed and not prev_buttons[button]:
                        session.request(what)
                    prev_buttons[button] = pressed

                time.sleep(max(0.0, period - (time.perf_counter() - tick)))
        except KeyboardInterrupt:
            print("\nStopping.")
        finally:
            stop.set()
            viewer.join(timeout=2.0)

    if session.samples:
        print("  " + save_session(session, out_dir, args.write, CAMERA_INFO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
