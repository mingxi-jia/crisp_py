"""Where the policy is about to move, drawn on the image it is looking at.

pi0.5-DROID speaks joint velocities, so "where is the gripper going" is not in
the action at all -- it only appears after forward kinematics. This module
rolls the action chunk forward into joint targets, runs FK on each, and
projects the resulting Cartesian path into the camera frame and into the
224x224 crop the policy was shown, so the plan can be read against the scene
rather than against axes.

The geometry itself -- the URDF chain, the camera model, the fingertip frame
-- is the robot's, not pi0.5's, and lives in control/projection.py. What is
here is the part that is specific to this policy: turning an action chunk into
a path, and undoing the policy's own image crop so a projected point lands on
the pixel the model saw it at.

FK is checked against the robot's own reported end-effector pose
(:func:`fk_residual`) before anything is drawn, which catches a wrong tip
frame or a flipped joint.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from control.projection import (
    BASE_LINK, DEFAULT_CAMERA_INFO, DEFAULT_URDF, FRAMES, TIP_LINK,
    CameraCalibration, Chain, axis_tips, calibration_key_for, fk_residual,
    frame_offset, rigid,
)
from deploy.pi05.policy import IMAGE_SIZE, MAX_JOINT_DELTA, joint_velocity_to_delta

from pathlib import Path

def to_obs_pixels(uv, frame_size, mode: str, box=None, out: int = IMAGE_SIZE):
    """Raw-frame pixels -> pixels in the square image the policy was sent.

    Mirrors policy.fit_image exactly -- the same crop, the same letterbox --
    so a point drawn here lands on the same feature it lands on in the frame.
    """
    uv = np.atleast_2d(np.asarray(uv, dtype=np.float64))
    w, h = frame_size

    if mode == "box":
        if box is None:
            raise ValueError("mode 'box' needs a crop box")
        x0, y0, x1, y1 = (float(v) for v in box)
        return np.stack([(uv[:, 0] - x0) * out / (x1 - x0),
                         (uv[:, 1] - y0) * out / (y1 - y0)], axis=1)

    if mode == "crop":
        # resize_with_crop: centre-crop to square, then resize.
        if w > h:
            new_w = int(round(h))
            x0, y0, side = (w - new_w) // 2, 0, new_w
        else:
            new_h = int(round(w))
            x0, y0, side = 0, (h - new_h) // 2, new_h
        return np.stack([(uv[:, 0] - x0) * out / side,
                         (uv[:, 1] - y0) * out / side], axis=1)

    # resize_with_pad: scale to fit, then centre on a black square. int() and
    # the //2 truncation are PIL's, kept so the mapping is pixel-exact.
    ratio = max(w / out, h / out)
    rw, rh = int(w / ratio), int(h / ratio)
    pad_w = max(0, int((out - rw) / 2))
    pad_h = max(0, int((out - rh) / 2))
    return np.stack([uv[:, 0] / ratio + pad_w, uv[:, 1] / ratio + pad_h], axis=1)


# ---------------------------------------------------------------------------
# The predicted path
# ---------------------------------------------------------------------------
def rollout_joint_targets(chunk, q0, speed_scale: float = 1.0,
                          max_step_rad: float = MAX_JOINT_DELTA) -> np.ndarray:
    """Action chunk + a starting joint position -> the joint targets it implies.

    The control loop applies each delta to the *measured* position, not to the
    previous target, so this open-loop integration is a prediction: it is what
    the arm does if it tracks perfectly. Tracking error makes the real path
    fall short of this one rather than wander off it.
    """
    chunk = np.clip(np.atleast_2d(np.asarray(chunk, dtype=np.float64)), -1.0, 1.0)
    q = np.asarray(q0, dtype=np.float64).ravel()[:7].copy()
    out = np.empty((len(chunk), 7))
    for i, row in enumerate(chunk):
        delta = joint_velocity_to_delta(row[:7]) * speed_scale
        q = q + np.clip(delta, -max_step_rad, max_step_rad)
        out[i] = q
    return out


def predicted_path(chain: Chain, chunk, q0, speed_scale: float = 1.0,
                   max_step_rad: float = MAX_JOINT_DELTA,
                   frame: str = "fingertip") -> tuple[np.ndarray, np.ndarray]:
    """Chunk + starting joints -> ((N+1, 3) positions, (N+1, 4, 4) poses).

    Row 0 is the starting pose, so the drawn path begins at the gripper rather
    than one step ahead of it.
    """
    offset = frame_offset(frame)
    qs = np.vstack([np.asarray(q0, dtype=np.float64).ravel()[:7],
                    rollout_joint_targets(chunk, q0, speed_scale, max_step_rad)])
    poses = np.stack([chain.fk(q) @ offset for q in qs])
    return poses[:, :3, 3], poses


class Overlay:
    """Turns the current chunk into points on one camera's image.

    One instance per camera. Built by deploy_pi05 and asked for a payload every
    control step; returns None -- rather than raising -- for a camera it cannot
    place, so an uncalibrated wrist view simply gets no overlay while the
    agentview still gets one.
    """

    def __init__(self, camera: str, fit: str, box=None, frame: str = "fingertip",
                 urdf=None, camera_info=None, chain: Chain | None = None,
                 calibration_key: str | None = None, role: str = "exterior"):
        """`calibration_key` names the camera_info.yaml entry to use. Left
        unset it comes from the camera's `calibration:` in config/cameras.yaml
        (`agentview` -> `cam3`), and failing that from the camera's own name --
        so the mapping lives with the camera definition rather than in every
        caller.

        `role` is which of the policy's two images this camera provides,
        "exterior" or "wrist"; it selects the frame and its size out of the
        request the policy last sent.
        """
        self.camera = camera
        self.role = role
        self.calibration_key = calibration_key or calibration_key_for(camera)
        self.fit = fit
        self.box = box
        self.frame = frame
        self.chain = chain or Chain.from_urdf(urdf)
        self.calibration_path = Path(camera_info or DEFAULT_CAMERA_INFO)
        self.calibration = CameraCalibration.load(self.calibration_key,
                                                  self.calibration_path)
        self.offset = frame_offset(frame)
        # Projecting N sample plans is N times the forward kinematics, and
        # they only change when the policy re-infers -- so they are computed
        # once per inference and reused for the steps in between.
        self._sample_cache = {"key": None, "value": None}
        # A camera on a moving link is located by FK to the link it rides,
        # which is a different chain than the one ending at the gripper.
        self._mount_chain = None
        if self.calibration is not None and self.calibration.moving:
            try:
                self._mount_chain = Chain.from_urdf(urdf, tip=self.calibration.parent)
            except ValueError as e:
                raise ValueError(
                    f"camera {self.camera!r} is calibrated against "
                    f"{self.calibration.parent!r}, which is not a link in the "
                    f"robot model: {e}") from e

    @property
    def available(self) -> bool:
        return self.calibration is not None

    def unavailable_reason(self) -> str:
        return (f"no entry {self.calibration_key!r} in "
                f"{self.calibration_path.name} for camera {self.camera!r}")

    # -- checks ----------------------------------------------------------
    def residual(self, state) -> tuple[float, float]:
        """FK vs the robot's own end-effector pose, in mm and degrees."""
        pos_err, rot_err = fk_residual(
            self.chain, np.asarray(state.joint_values)[:7],
            state.ee_position, state.ee_quat_xyzw)
        return pos_err * 1000.0, rot_err

    def _measured_pose(self, state) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(state.ee_quat_xyzw).as_matrix()
        T[:3, 3] = state.ee_position
        return T @ self.offset

    # -- payload ---------------------------------------------------------
    def _project(self, points_base, frame_size, q):
        """Base-frame points -> two normalised views of the same projection.

        `raw` is the whole camera frame, `obs` the square crop the policy is
        actually given. Both are produced: the crop is what the model sees, but
        a path that leaves it -- which is easy, the gripper often sits on the
        crop edge -- only stays visible on the raw frame.
        """
        uv, front = self.calibration.project(
            points_base, frame_size, self._mount_chain, q)
        raw = uv / np.asarray(frame_size, dtype=np.float64)
        obs = to_obs_pixels(uv, frame_size, self.fit, self.box) / IMAGE_SIZE
        return raw, obs, front

    def _sample_paths(self, policy, q0, frame_size):
        """The other sampled plans, as paths -> (raw, obs, spread at horizon).

        pi0.5 draws its plan from noise, so several inferences on one
        observation give several plans. Drawing them together is the only
        place the policy's uncertainty is visible: where they agree it is
        confident, where they fan out it is not.
        """
        samples = getattr(policy, "last_samples", None)
        if samples is None or len(samples) < 2:
            return [], [], None

        key = (policy.n_inferences, len(samples), tuple(frame_size))
        if self._sample_cache["key"] == key:
            return self._sample_cache["value"]

        raws, obss, ends = [], [], []
        horizon = min(int(policy.open_loop_horizon), len(samples[0]))
        for chunk in samples:
            positions, _ = predicted_path(
                self.chain, chunk, q0, policy.speed_scale, policy.max_step_rad,
                self.frame)
            raw, obs, _ = self._project(positions, frame_size, q0)
            raws.append(np.round(raw, 5).tolist())
            obss.append(np.round(obs, 5).tolist())
            ends.append(positions[horizon])

        ends = np.asarray(ends)
        spread = round(float(np.linalg.norm(ends - ends.mean(axis=0), axis=1).max()
                             * 1000.0), 1)
        value = (raws, obss, spread)
        self._sample_cache = {"key": key, "value": value}
        return value

    def build(self, policy, state) -> dict | None:
        """This camera's contribution to the panel payload, or None.

        Returns `{"views": {...}, "order": [...], ...}`; deploy_pi05 merges one
        of these per camera, so adding a calibrated camera adds a tab and
        nothing else has to change.
        """
        if self.calibration is None:
            return None
        request = policy.last_request or {}
        q0 = request.get("observation/joint_position")
        chunk = getattr(policy, "_chunk", None)
        frame_size = (policy.last_frame_shapes or {}).get(self.role)
        if q0 is None or chunk is None or frame_size is None:
            return None
        q0 = np.asarray(q0, dtype=np.float64).ravel()[:7]

        # The path is rolled out from the joints the *request* carried, not
        # from the joints now: the image on screen is from that inference, and
        # a path anchored anywhere else would not line up with the gripper in
        # it. For a wrist camera this matters twice over -- the camera itself
        # was at that configuration too. Where the arm has since got to is sent
        # separately as `now`.
        positions, poses = predicted_path(
            self.chain, chunk, q0, policy.speed_scale, policy.max_step_rad,
            self.frame)
        path_raw, path_obs, front = self._project(positions, frame_size, q0)
        axes_raw, axes_obs, _ = self._project(axis_tips(poses[0]), frame_size, q0)

        samples_raw, samples_obs, spread = self._sample_paths(policy, q0, frame_size)

        now_raw = now_obs = None
        now_front = None
        if state is not None and getattr(state, "ee_position", None) is not None:
            # Placed with the joints of *now*, since that is where both the arm
            # and (for a wrist camera) the camera are at this instant.
            q_now = np.asarray(state.joint_values, dtype=np.float64).ravel()[:7]
            nr, no, nf = self._project(
                self._measured_pose(state)[:3, 3], frame_size, q_now)
            now_raw, now_obs, now_front = nr[0], no[0], bool(nf[0])

        w, h = frame_size
        x0, y0, x1, y1 = self.box if (self.fit == "box" and self.box) else (0, 0, w, h)
        views = {
            f"{self.camera}:raw": {
                "label": f"{self.camera} raw · {w}x{h}",
                "image": f"{self.role}_raw_jpeg",
                "path": np.round(path_raw, 5).tolist(),
                "axes": np.round(axes_raw, 5).tolist(),
                "samples": samples_raw,
                "now": None if now_raw is None else np.round(now_raw, 5).tolist(),
                "aspect": round(w / h, 4),
                "crop": [x0 / w, y0 / h, x1 / w, y1 / h],
            },
            f"{self.camera}:obs": {
                "label": f"{self.camera} policy input · {IMAGE_SIZE}px {self.fit}",
                "image": f"{self.role}_jpeg",
                "path": np.round(path_obs, 5).tolist(),
                "axes": np.round(axes_obs, 5).tolist(),
                "samples": samples_obs,
                "now": None if now_obs is None else np.round(now_obs, 5).tolist(),
                "aspect": 1.0,
                "crop": None,
            },
        }

        grip = np.clip(np.asarray(chunk, dtype=np.float64)[:, 7], 0.0, 1.0)
        if policy.binarize_gripper:
            grip = (grip > 0.5).astype(float)
        grip = np.concatenate([grip[:1], grip])          # row 0 is the start pose

        out = {
            "camera": self.camera,
            "calibration": self.calibration.name,
            "mounted_on": self.calibration.parent,
            "frame": self.frame,
            "views": views,
            "order": list(views),
            "front": front.astype(bool).tolist(),
            "gripper": grip.tolist(),
            "horizon": int(policy.open_loop_horizon),
            "cursor": int(max(0, min(policy._chunk_index, len(chunk)))),
            "aspect_mismatch": round(self.calibration.aspect_mismatch(frame_size), 4),
            "n_samples": 0 if not samples_raw else len(samples_raw),
            # How far apart the sampled plans have drifted by the end of the
            # part that will actually be executed. This is the policy's own
            # uncertainty in millimetres, which nothing else on the panel says.
            "spread_mm": spread,
        }
        if now_front is not None:
            out["now_front"] = now_front
            pos_mm, rot_deg = self.residual(state)
            out["fk_residual_mm"] = round(pos_mm, 2)
            out["fk_residual_deg"] = round(rot_deg, 2)
        return out


def merge(payloads) -> dict | None:
    """Several per-camera payloads -> the single one the panel consumes.

    The shared facts (the gripper track, the horizon, the FK residual) are the
    same in every camera's payload by construction, so the first one that has
    them wins and the rest contribute only their views.
    """
    payloads = [p for p in payloads if p]
    if not payloads:
        return None
    out = dict(payloads[0])
    for extra in payloads[1:]:
        out["views"] = {**out["views"], **extra["views"]}
        out["order"] = out["order"] + extra["order"]
    out["cameras"] = [p["camera"] for p in payloads]
    return out
