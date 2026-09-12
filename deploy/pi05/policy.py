"""π0.5-DROID policy client and the DROID<->crisp conversions.

Action space (third_party/droid/droid/franka/robot.py:82-84 and
robot_ik/robot_ik_solver.py:9-10, 88-100)
    The policy emits 8 numbers per step: 7 *normalised joint velocities* in
    [-1, 1] and 1 gripper *position*. DROID applies them as

        joint_delta = clip(v, -1, 1) * MAX_JOINT_DELTA        # 0.2 rad
        q_target    = q_measured_now + joint_delta

    at 15 Hz, i.e. a full-scale command is 0.2 rad per tick (3 rad/s). The
    delta is applied to the *measured* position each tick, not to a running
    target, so tracking error does not accumulate.

Gripper convention (third_party/droid/droid/franka/robot.py:123)
    DROID commands width = max_width * (1 - g), so g=0 is OPEN and g=1 is
    CLOSED. crisp_py is the opposite: 1.0 is open. Both directions are
    flipped here -- get it wrong and the gripper does exactly the inverse of
    what the policy intends.
"""

import dataclasses

import numpy as np

# action scale is per-tick, so running faster or slower rescales every motion.
DROID_CONTROL_HZ = 15.0

# droid/robot_ik/robot_ik_solver.py: max_joint_delta = relative_max_joint_delta.max()
MAX_JOINT_DELTA = 0.2      # rad per tick at 15 Hz, for |v| = 1

# Policy input image size (openpi droid_policy.make_droid_example)
IMAGE_SIZE = 224


def resize_with_crop(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Centre-crop to the target aspect ratio, then resize. No padding.

    The alternative, openpi's resize_with_pad, letterboxes: a 480x270 (16:9)
    frame becomes 224x224 with 98 black rows, so only 126 rows carry image and
    the horizontal field of view is squeezed into them. Cropping keeps the full
    vertical resolution and the native scale at the cost of the left and right
    edges.

    Note this is a departure from DROID: its ZED frames are also 16:9 and were
    padded during training, so a cropped frame is not quite the distribution
    the policy saw. Worth it when the workspace sits in the middle of the
    frame and the edges are wasted on background.
    """
    from PIL import Image

    image = np.asarray(image)
    h, w = image.shape[:2]
    target_ratio = width / height
    if w / h > target_ratio:                 # too wide: trim left/right
        new_w = int(round(h * target_ratio))
        x0 = (w - new_w) // 2
        cropped = image[:, x0:x0 + new_w]
    else:                                    # too tall: trim top/bottom
        new_h = int(round(w / target_ratio))
        y0 = (h - new_h) // 2
        cropped = image[y0:y0 + new_h, :]
    return np.asarray(
        Image.fromarray(cropped).resize((width, height), Image.BILINEAR))


def resize_with_box(image: np.ndarray, box, height: int, width: int) -> np.ndarray:
    """Crop an explicit (x0, y0, x1, y1) region, then resize to height x width.

    For when the useful part of the frame is not centred -- an 848x480 side
    view whose workspace sits left of centre, say. A centre crop would keep
    the wrong 480x480.

    The box is validated against the actual frame rather than trusted: a box
    left over from a different camera resolution would otherwise silently
    crop the wrong region, or throw a confusing numpy error.
    """
    from PIL import Image

    if image is None:
        raise ValueError("no image to crop (the camera has no fresh frame)")
    image = np.asarray(image)
    if image.ndim < 2:
        raise ValueError(f"expected an HxWxC image, got shape {image.shape}")
    h, w = image.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in box)
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(
            f"crop box (x0={x0}, y0={y0}, x1={x1}, y1={y1}) does not fit a "
            f"{w}x{h} frame. Give a box inside 0..{w} x 0..{h}.")
    cropped = image[y0:y1, x0:x1]
    return np.asarray(
        Image.fromarray(cropped).resize((width, height), Image.BILINEAR))


def fit_image(image: np.ndarray, mode: str, box=None) -> np.ndarray:
    """Bring a frame to IMAGE_SIZE square.

    mode 'box'  -- crop the explicit region in `box`, then resize
         'crop' -- centre-crop to square, then resize
         'pad'  -- letterbox, what DROID did during training
    """
    if mode == "box":
        if box is None:
            raise ValueError("mode 'box' needs a crop box")
        return resize_with_box(image, box, IMAGE_SIZE, IMAGE_SIZE)
    if mode == "crop":
        return resize_with_crop(image, IMAGE_SIZE, IMAGE_SIZE)
    from openpi_client import image_tools
    return image_tools.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)


class StaleCameraError(RuntimeError):
    """A camera the policy needs is not delivering fresh frames.

    Raised instead of quietly running on a frozen image: a wrist camera that
    dropped off the USB bus leaves its last frame in memory indefinitely, and
    the policy cannot tell that from a real observation.
    """


@dataclasses.dataclass
class JointAction:
    """One step of a π0.5-DROID chunk, already converted for this robot."""

    joint_positions: np.ndarray      # (7,) absolute target, rad
    gripper: float                   # crisp convention: 1.0 open, 0.0 closed
    raw_velocity: np.ndarray         # (7,) the normalised velocities, for logs
    raw_gripper: float               # DROID convention, before the flip


def droid_gripper_from_crisp(crisp_value: float) -> float:
    """crisp (1=open) -> DROID (0=open). See module docstring."""
    return float(np.clip(1.0 - crisp_value, 0.0, 1.0))


def crisp_gripper_from_droid(droid_value: float) -> float:
    """DROID (0=open) -> crisp (1=open)."""
    return float(np.clip(1.0 - droid_value, 0.0, 1.0))


def joint_velocity_to_delta(velocity: np.ndarray) -> np.ndarray:
    """Normalised joint velocity -> joint-position delta for one tick.

    Mirrors RobotIKSolver.joint_velocity_to_delta. Because
    relative_max_joint_delta is uniform (0.2), its norm-scaling step reduces to
    dividing by max|v| when that exceeds 1 -- i.e. a clip that preserves
    direction rather than clipping per-axis.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    max_norm = np.abs(velocity).max()
    if max_norm > 1.0:
        velocity = velocity / max_norm
    return velocity * MAX_JOINT_DELTA


class Pi05DroidPolicy:
    """Queries an openpi websocket server and returns robot-ready actions.

    Same interface as the other policies here -- ``reset(...)`` then
    ``predict_action(obs)`` -- so the control loop looks the same.

    Chunking: the server returns a whole action chunk per call. We execute
    ``open_loop_horizon`` steps of it before asking again, which is what the
    DROID reference client does (8 steps ~ 0.5 s at 15 Hz).
    """

    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8000,
                 prompt: str = "",
                 external_camera: str = "cam1",
                 wrist_camera: str = "cam4",
                 open_loop_horizon: int = 8,
                 exterior_fit: str = "box",
                 wrist_fit: str = "pad",
                 exterior_box=None,
                 wrist_box=None,
                 binarize_gripper: bool = True,
                 want_features: bool = False,
                 speed_scale: float = 1.0,
                 max_step_rad: float = MAX_JOINT_DELTA,
                 api_key: str | None = None):
        from openpi_client import websocket_client_policy

        self.prompt = prompt
        self.external_camera = external_camera
        self.wrist_camera = wrist_camera
        self.open_loop_horizon = open_loop_horizon
        # The side view is centre-cropped by default: padding a 16:9 frame
        # wastes 44% of the 224x224 on black bars. The wrist camera stays
        # padded -- it is close-range and the edges of its view matter.
        self.exterior_fit = exterior_fit
        self.wrist_fit = wrist_fit
        self.exterior_box = exterior_box
        # Ask the server for vision features only when something renders them.
        # They cost a second forward pass through the vision tower.
        self.want_features = bool(want_features)
        self.wrist_box = wrist_box
        self.binarize_gripper = binarize_gripper
        # Shrinks each step without changing the loop rate, so the policy keeps
        # observing at the 15 Hz it was trained at while the arm moves less per
        # tick. Preferred over lowering the rate for a cautious first rollout.
        self.speed_scale = float(speed_scale)
        # Hard ceiling per joint per step, applied after the scale. A backstop
        # against a bad chunk regardless of what the policy asks for.
        self.max_step_rad = float(max_step_rad)

        # WebsocketClientPolicy retries forever when nothing is listening, so
        # a missing policy server looks like a hang. Check the port first and
        # fail with something actionable.
        import socket
        probe = socket.socket()
        probe.settimeout(3.0)
        try:
            probe.connect((host if host != "0.0.0.0" else "127.0.0.1", int(port)))
        except OSError as e:
            raise ConnectionError(
                f"no policy server at {host}:{port} ({e}).\n"
                f"Start it first:\n"
                f"    cd third_party/openpi\n"
                f"    uv run python ../../deploy/pi05_policy_server.py "
                f"--config pi05_droid --dir gs://openpi-assets/checkpoints/pi05_droid"
            ) from e
        finally:
            probe.close()

        self.client = websocket_client_policy.WebsocketClientPolicy(
            host, port, api_key=api_key)
        self.server_metadata = self.client.get_server_metadata()

        self._chunk = None
        self._chunk_index = 0
        self.last_inference_ms = None
        self.n_inferences = 0
        # The exact payload last handed to the server, for --visualize. This is
        # post-resize_with_pad, i.e. what the model actually sees (letterbox
        # bars included), not the raw camera frame.
        self.last_request = None
        # Vision tokens from deploy/pi05_policy_server.py, {} against openpi's
        # stock server (which returns actions only). Never required for control.
        self.last_features = {}
        self.server_has_features = False
        self.last_feature_ms = None

    # -- lifecycle -------------------------------------------------------
    def reset(self, obs=None):
        """Drop the current chunk so the next call re-queries the server."""
        self._chunk = None
        self._chunk_index = 0

    def set_prompt(self, prompt: str):
        self.prompt = prompt
        self.reset()

    # -- observation -----------------------------------------------------
    def build_request(self, obs) -> dict:
        """Assemble the exact payload openpi's DroidInputs expects.

        Keys come from openpi/src/openpi/policies/droid_policy.py.
        """
        exterior = obs.rgb(self.external_camera)
        wrist = obs.rgb(self.wrist_camera)

        # The server nulls frames it judges stale, so `is None` covers both
        # "never arrived" and "camera dropped off the bus and its last image is
        # still sitting in memory". Name the difference in the message.
        problems = []
        for role, name, img in (("exterior", self.external_camera, exterior),
                                ("wrist", self.wrist_camera, wrist)):
            if img is not None:
                continue
            age = obs.frame_age(name) if hasattr(obs, "frame_age") else None
            if age is not None:
                problems.append(f"{name} ({role}) is STALE — last frame {age:.1f}s ago")
            else:
                problems.append(f"{name} ({role}) has no frame at all")
        if problems:
            raise StaleCameraError(
                "cannot query the policy: " + "; ".join(problems)
                + f". Fresh cameras: {obs.ready_cameras or 'none'}")

        if obs.gripper_value is None:
            raise RuntimeError("gripper value unavailable; cannot build state")

        return {
            "observation/exterior_image_1_left":
                fit_image(exterior, self.exterior_fit, self.exterior_box),
            "observation/wrist_image_left":
                fit_image(wrist, self.wrist_fit, self.wrist_box),
            "observation/joint_position":
                np.asarray(obs.joint_values, dtype=np.float64)[:7],
            "observation/gripper_position":
                np.array([droid_gripper_from_crisp(obs.gripper_value)]),
            "prompt": self.prompt,
            # Consumed by deploy/pi05_policy_server.py; openpi's own server
            # ignores unknown keys.
            "want_features": self.want_features,
        }

    # -- policy ----------------------------------------------------------
    def predict_action(self, obs) -> JointAction:
        """Next action, querying the server only when the chunk runs out."""
        import time

        if self._chunk is None or self._chunk_index >= self.open_loop_horizon \
                or self._chunk_index >= len(self._chunk):
            request = self.build_request(obs)
            self.last_request = request
            start = time.perf_counter()
            result = self.client.infer(request)
            self.last_inference_ms = (time.perf_counter() - start) * 1000.0
            self.n_inferences += 1

            self.last_feature_ms = result.get("feature_ms")
            feats = result.get("features") or {}
            self.last_features = {k: np.asarray(v, dtype=np.float32)
                                  for k, v in feats.items()}
            self.server_has_features = bool(self.last_features)

            chunk = np.asarray(result["actions"])
            if chunk.ndim != 2 or chunk.shape[1] != 8:
                raise RuntimeError(
                    f"expected an (N, 8) action chunk -- 7 joint velocities + "
                    f"1 gripper -- got {chunk.shape}")
            self._chunk = chunk
            self._chunk_index = 0

        action = self._chunk[self._chunk_index]
        self._chunk_index += 1

        # DROID clips the whole action to [-1, 1] before applying it.
        action = np.clip(action, -1.0, 1.0)
        velocity, droid_gripper = action[:7], float(action[7])

        if self.binarize_gripper:
            droid_gripper = 1.0 if droid_gripper > 0.5 else 0.0

        q_now = np.asarray(obs.joint_values, dtype=np.float64)[:7]
        delta = joint_velocity_to_delta(velocity) * self.speed_scale
        delta = np.clip(delta, -self.max_step_rad, self.max_step_rad)
        q_target = q_now + delta

        return JointAction(
            joint_positions=q_target,
            gripper=crisp_gripper_from_droid(droid_gripper),
            raw_velocity=velocity,
            raw_gripper=droid_gripper,
        )


# ---------------------------------------------------------------------------
