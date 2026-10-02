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
                 external_camera: str = "agentview",
                 wrist_camera: str = "inhand",
                 open_loop_horizon: int = 8,
                 exterior_fit: str = "box",
                 wrist_fit: str = "pad",
                 exterior_box=None,
                 wrist_box=None,
                 binarize_gripper: bool = True,
                 want_features: bool = False,
                 n_samples: int = 1,
                 samples_every: int = 1,
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
        # pi0.5 samples its plan from noise, so asking twice gives two
        # different plans. Drawing several shows where the policy is unsure --
        # at the price of a bigger batch on the GPU, so they are asked for
        # only every `samples_every` inferences and the control chunk is
        # always the first one returned.
        self.n_samples = max(1, int(n_samples))
        self.samples_every = max(1, int(samples_every))
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
        self.last_frame_shapes = {}
        self.last_raw_frames = {}
        # (N, T, 8) when the server returned several plans for one
        # observation, else None. Never used for control -- the executed chunk
        # is samples[0], which the server guarantees.
        self.last_samples = None
        self.samples_asked = 0
        self._pending_prompt = None
        # What the server said at connect time. Only advisory: it is replaced
        # by what actually comes back on the first sampled inference.
        self.server_has_samples = bool(self.server_metadata.get("samples"))
        if self.n_samples > 1 and not self.server_has_samples:
            print(f"\n  NOTE: --samples {self.n_samples} may do nothing: this "
                  f"policy server does not offer sampled plans "
                  f"(metadata: {self.server_metadata or '{}'}).\n"
                  f"  They are asked for anyway, in case the server was "
                  f"restarted since. They come from "
                  f"deploy/pi05_policy_server.py, run with --max-samples "
                  f"{self.n_samples} or more:\n"
                  f"      cd third_party/openpi\n"
                  f"      uv run python ../../deploy/pi05_policy_server.py "
                  f"--config pi05_droid \\\n"
                  f"          --dir gs://openpi-assets/checkpoints/pi05_droid\n")
        cap = int(self.server_metadata.get("max_samples") or 0)
        if cap and self.n_samples > cap:
            print(f"  NOTE: the server caps samples at {cap}; asking for that "
                  f"instead of {self.n_samples}.")
            self.n_samples = cap

    # -- lifecycle -------------------------------------------------------
    def reset(self, obs=None):
        """Drop the current chunk so the next call re-queries the server."""
        self._chunk = None
        self._chunk_index = 0

    def set_prompt(self, prompt: str):
        """Change the task now. Only safe from the control loop's own thread."""
        self.prompt = prompt
        self.reset()

    def request_prompt(self, prompt: str) -> str:
        """Ask for a new task from another thread (the panel).

        Queued rather than applied: reset() clears the action chunk, and doing
        that underneath predict_action -- which has already decided it does not
        need to re-infer and is about to index into the chunk -- would crash the
        rollout on a None. The control loop picks this up at the top of its
        next step, within one tick.
        """
        prompt = str(prompt)
        self._pending_prompt = prompt
        return prompt

    @property
    def pending_prompt(self) -> str | None:
        """A requested prompt that has not taken effect yet, else None."""
        pending = self._pending_prompt
        return None if pending is None or pending == self.prompt else pending

    def _apply_pending_prompt(self) -> bool:
        """Adopt a queued prompt. Called by the control loop, nobody else."""
        pending = self._pending_prompt
        self._pending_prompt = None
        if pending is None or pending == self.prompt:
            return False
        # The chunk in flight was planned for the old task, so drop it: the
        # next step re-infers against the new one rather than finishing a
        # motion nobody asked for any more.
        self.set_prompt(pending)
        return True

    def _samples_due(self) -> int:
        """How many plans to ask for on the next inference.

        Deliberately not gated on what the server's metadata said: that is read
        once at connect time, so a server restarted underneath a running
        rollout would leave this client believing sampling is unavailable
        forever, with no way to find out otherwise. Unknown request keys are
        ignored by every server here, so asking costs nothing and the answer
        is learned from what comes back.
        """
        if self.n_samples <= 1:
            return 1
        return self.n_samples if (self.n_inferences % self.samples_every) == 0 else 1

    @property
    def needs_inference(self) -> bool:
        """Whether the next predict_action will query the server.

        The control loop asks this to decide whether to fetch camera frames:
        /get_obs costs about ten times /get_state, so a step that is only
        replaying an existing chunk reads proprioception alone. That makes
        this the *contract* between loop and policy, not an internal detail --
        when the two disagree, predict_action gets a state-only reading and
        cannot build a request from it.

        A queued prompt counts. It drops the chunk when it is applied, which
        means an inference, which means the loop has to have fetched images.
        """
        return (self.pending_prompt is not None
                or self._chunk is None
                or self._chunk_index >= self.open_loop_horizon
                or self._chunk_index >= len(self._chunk))

    # -- observation -----------------------------------------------------
    def build_request(self, obs) -> dict:
        """Assemble the exact payload openpi's DroidInputs expects.

        Keys come from openpi/src/openpi/policies/droid_policy.py.
        """
        if not hasattr(obs, "rgb"):
            # A state-only reading, which means the caller decided no
            # inference was due and then one turned out to be. Say what the
            # contract is rather than dying on a missing attribute.
            raise RuntimeError(
                f"an inference is due but this is a {type(obs).__name__} with "
                f"no camera frames. The control loop must fetch get_obs() "
                f"whenever policy.needs_inference is True -- not re-derive "
                f"that condition, which is how the two come apart.")
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

        # Source resolution, kept because the panel's overlay has to undo
        # fit_image to put a projected point back on the right pixel.
        self.last_frame_shapes = {
            "exterior": (exterior.shape[1], exterior.shape[0]),
            "wrist": (wrist.shape[1], wrist.shape[0]),
        }
        # References, not copies: the panel draws the predicted path on the
        # uncropped frame, where a trajectory that leaves --exterior-crop is
        # still visible instead of silently clipped.
        self.last_raw_frames = {"exterior": exterior, "wrist": wrist}

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
            "n_samples": self._samples_due(),
        }

    # -- policy ----------------------------------------------------------
    def predict_action(self, obs) -> JointAction:
        """Next action, querying the server only when the chunk runs out."""
        import time

        # A prompt change from the panel lands here, where nothing else is
        # part-way through reading the chunk.
        self._apply_pending_prompt()

        if self.needs_inference:
            request = self.build_request(obs)
            self.last_request = request
            start = time.perf_counter()
            result = self.client.infer(request)
            self.last_inference_ms = (time.perf_counter() - start) * 1000.0
            self.n_inferences += 1

            samples = result.get("samples")
            self.last_samples = (None if samples is None
                                 else np.asarray(samples, dtype=np.float64))
            # Observed, not advertised: whatever the metadata claimed, this is
            # what the server actually does. `asked` is kept beside it so the
            # panel can say "asked for 6, got none" rather than just showing
            # an empty fan.
            if request["n_samples"] > 1:
                self.samples_asked = request["n_samples"]
                self.server_has_samples = (self.last_samples is not None
                                           and len(self.last_samples) > 1)
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
