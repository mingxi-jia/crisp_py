"""The π0.5 panel: a small web server showing what the policy sees and thinks.

Served on its own port by deploy_pi05 --panel, so nothing π0.5-specific has to
live in control/robot_server.py. Four panes:

  1. the observation actually sent to the policy (post fit_image, so you see
     the crop/letterbox the model sees, not the raw camera frame)
  2. a PCA map of the SigLIP patch tokens, per camera
  3. click a patch -> cosine similarity of that patch against every other
  4. the predicted action: the chunk run through FK and the camera
     calibration, drawn as the path the fingertip is about to take across the
     observation, plus the per-joint velocity behind it

The heavy tensors never reach the browser: features are projected onto a fixed
64-component PCA basis (~64 KB/camera) and the similarity map is computed in
the page on click. See features.reduce_for_client for how good that
approximation is.
"""

import base64
import io
import json
import threading
import time
from pathlib import Path

import numpy as np

from deploy.pi05.features import (
    DEFAULT_COMPONENTS, FeatureMap, PCABasis, pca_rgb,
    reduce_for_client,
)
from deploy.pi05.overlay import merge as overlay_merge
from deploy.pi05.policy import MAX_JOINT_DELTA, joint_velocity_to_delta

PANEL_HTML = Path(__file__).with_name("panel.html")


def _jpeg_b64(rgb, quality: int = 80) -> str | None:
    if rgb is None:
        return None
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# The raw exterior frame is ~40x the pixels of the 224x224 observation, and
# only changes when the policy re-infers. Encoding it every control step would
# cost more than everything else in the payload put together, so it is cached
# against the inference that produced it.
_raw_cache: dict = {}


def _raw_jpeg_b64(rgb, role: str, key) -> str | None:
    if rgb is None:
        return None
    hit = _raw_cache.get(role)
    if not hit or hit[0] != key:
        hit = (key, _jpeg_b64(rgb, quality=70))
        _raw_cache[role] = hit
    return hit[1]


def _png_b64(rgb) -> str | None:
    """PNG for the PCA map: it is 16x16, and JPEG artefacts would be lies."""
    if rgb is None:
        return None
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class FeatureAnalyser:
    """Holds the PCA basis and turns raw tokens into panel payloads.

    The basis is fitted once over the first `fit_frames` observations and then
    held: refitting per frame makes colours flicker, because SVD component
    order and sign are arbitrary.
    """

    def __init__(self, n_components: int = DEFAULT_COMPONENTS, fit_frames: int = 8):
        self.n_components = n_components
        self.fit_frames = fit_frames
        self._bases: dict[str, PCABasis] = {}
        self._pending: dict[str, list] = {}
        self._lock = threading.Lock()

    def refit(self):
        """Drop the bases so they are re-fitted on the next frames."""
        with self._lock:
            self._bases.clear()
            self._pending.clear()

    def basis_for(self, camera: str, tokens: np.ndarray) -> PCABasis | None:
        with self._lock:
            if camera in self._bases:
                return self._bases[camera]
            pending = self._pending.setdefault(camera, [])
            pending.append(np.asarray(tokens, dtype=np.float32))
            if len(pending) < self.fit_frames:
                # Still collecting: fit a provisional basis so the panel shows
                # something immediately rather than staying blank.
                return PCABasis.fit(np.stack(pending), self.n_components)
            basis = PCABasis.fit(np.stack(pending), self.n_components)
            self._bases[camera] = basis
            self._pending.pop(camera, None)
            return basis

    def analyse(self, features: dict) -> dict:
        """{role: tokens} -> per-camera payload for the panel."""
        out = {}
        for role, tokens in (features or {}).items():
            tokens = np.asarray(tokens, dtype=np.float32)
            try:
                fm = FeatureMap(tokens, camera=role)
            except ValueError as e:
                out[role] = {"error": str(e)}
                continue
            basis = self.basis_for(role, tokens)
            reduced = reduce_for_client(fm, basis, self.n_components)
            # As base64 float32, not a JSON list: JSON renders each float as
            # ~20 characters, which turned a 64 KB array into 329 KB on the
            # wire. base64 of the raw buffer is 87 KB and decodes in one step.
            reduced_b64 = base64.b64encode(
                np.ascontiguousarray(reduced, dtype=np.float32).tobytes()
            ).decode("ascii")
            with self._lock:
                settled = role in self._bases
            out[role] = {
                "grid": fm.grid,
                "dim": fm.dim,
                "pca_png": _png_b64(pca_rgb(fm, basis)),
                "reduced_b64": reduced_b64,
                "n_components": int(reduced.shape[1]),
                "basis_settled": settled,
                "explained_variance":
                    None if basis.explained_variance_ratio is None
                    else float(basis.explained_variance_ratio.sum()),
            }
        return out


class PromptHistory:
    """The tasks that have been asked for, most recent first.

    Kept on disk rather than in the browser: the useful history is "what have I
    tried on this robot", which should survive a reload, a different browser
    and a restarted rollout. Small enough that it is rewritten whole on every
    change -- there is no concurrent writer worth the complexity of anything
    else.
    """

    DEFAULT_PATH = Path.home() / ".cache" / "crisp" / "pi05_prompts.json"
    LIMIT = 50

    def __init__(self, path=None, limit: int = LIMIT):
        self.path = None if path is False else Path(path or self.DEFAULT_PATH)
        self.limit = int(limit)
        self._entries: list[dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if self.path is None or not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except Exception as e:
            print(f"  NOTE: could not read the prompt history at {self.path}: {e}")
            return
        if isinstance(data, list):
            self._entries = [e for e in data
                             if isinstance(e, dict) and e.get("prompt")][:self.limit]

    def _save(self):
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._entries, indent=1))
        except Exception as e:
            # Losing the history is not worth failing a rollout over.
            print(f"  NOTE: could not write the prompt history to {self.path}: {e}")

    def add(self, prompt: str) -> list:
        """Record a prompt, moving it to the front if it is already known."""
        prompt = str(prompt).strip()
        if not prompt:
            return self.entries()
        with self._lock:
            existing = next((e for e in self._entries if e["prompt"] == prompt), None)
            if existing is not None:
                self._entries.remove(existing)
            else:
                existing = {"prompt": prompt, "count": 0}
            existing["count"] = int(existing.get("count", 0)) + 1
            existing["last_used"] = time.time()
            self._entries.insert(0, existing)
            del self._entries[self.limit:]
            self._save()
            return [dict(e) for e in self._entries]

    def entries(self, limit: int | None = None) -> list:
        with self._lock:
            return [dict(e) for e in self._entries[:limit or self.limit]]

    def prompts(self, limit: int | None = None) -> list:
        return [e["prompt"] for e in self.entries(limit)]


class PanelState:
    """Latest payload, written by the control loop and read by the browser."""

    def __init__(self):
        self._payload: dict | None = None
        self._lock = threading.Lock()

    def set(self, payload: dict):
        payload = dict(payload)
        payload["received_at"] = time.time()
        with self._lock:
            self._payload = payload

    def get(self) -> dict:
        with self._lock:
            if self._payload is None:
                return {"present": False}
            out = dict(self._payload)
        out["present"] = True
        out["age_s"] = round(time.time() - out["received_at"], 3)
        return out


def _commanded_delta(policy, action) -> np.ndarray:
    """The joint delta actually sent this tick [rad], velocity -> motion."""
    delta = joint_velocity_to_delta(action.raw_velocity) * policy.speed_scale
    return np.clip(delta, -policy.max_step_rad, policy.max_step_rad)


def _prediction(policy) -> dict | None:
    """The current action chunk, plus where in it we are.

    `_chunk_index` has already been advanced past the step just executed, so
    the step the panel should highlight is one behind it. Only the first
    `open_loop_horizon` rows are ever executed; the rest are shown greyed, as
    the plan the policy had but will re-make before reaching.
    """
    chunk = getattr(policy, "_chunk", None)
    if chunk is None:
        return None
    chunk = np.asarray(chunk, dtype=np.float64)
    # Same clip DROID applies before acting, so the plot matches the motion.
    chunk = np.clip(chunk, -1.0, 1.0)
    return {
        "chunk": np.round(chunk, 4).tolist(),
        "cursor": int(max(0, min(policy._chunk_index - 1, len(chunk) - 1))),
        "index": int(policy._chunk_index),
        "horizon": int(policy.open_loop_horizon),
        "step_rad": float(policy.speed_scale * MAX_JOINT_DELTA),
        "max_step_rad": float(policy.max_step_rad),
        "binarized": bool(policy.binarize_gripper),
    }


def build_payload(policy, action, args, step: int, analyser: FeatureAnalyser,
                  state=None, overlays=(), history: PromptHistory | None = None,
                  listener=None) -> dict:
    """Everything the panel needs for one step.

    `state` is the robot reading this tick and `overlays` the per-camera
    projectors built by deploy_pi05; both are optional, so the panel still
    works with no camera calibration at all (it just loses the trajectory drawn
    on the image).
    """
    req = policy.last_request or {}
    joint = req.get("observation/joint_position")
    grip = req.get("observation/gripper_position")
    feats = getattr(policy, "last_features", None) or {}
    raw_frames = getattr(policy, "last_raw_frames", None) or {}
    return {
        # Three different prompts, and the difference matters while one is
        # taking effect: what the last inference actually ran with, what the
        # policy will use next, and what has been asked for but not picked up
        # yet.
        "prompt": req.get("prompt", ""),
        "prompt_active": getattr(policy, "prompt", ""),
        "prompt_pending": getattr(policy, "pending_prompt", None),
        "prompt_history": [] if history is None else history.entries(20),
        # What the microphone is doing, when there is one. The panel shows the
        # words as they accumulate, so a mis-heard task is obvious before the
        # terminator commits it.
        "speech": None if listener is None else listener.status(),
        "obs": {
            "exterior_jpeg": _jpeg_b64(req.get("observation/exterior_image_1_left")),
            "exterior_raw_jpeg": _raw_jpeg_b64(raw_frames.get("exterior"),
                                               "exterior", policy.n_inferences),
            "wrist_raw_jpeg": _raw_jpeg_b64(raw_frames.get("wrist"),
                                            "wrist", policy.n_inferences),
            "wrist_jpeg": _jpeg_b64(req.get("observation/wrist_image_left")),
            "external_camera": args.external_camera,
            "wrist_camera": args.wrist_camera,
            "exterior_fit": args.exterior_fit,
            "wrist_fit": args.wrist_fit,
            "joint_position": None if joint is None else np.asarray(joint).tolist(),
            "gripper_droid": None if grip is None else float(np.asarray(grip).ravel()[0]),
        },
        "features": analyser.analyse(feats),
        "action": None if action is None else {
            "velocity": np.asarray(action.raw_velocity).tolist(),
            "velocity_absmax": float(np.abs(action.raw_velocity).max()),
            "gripper_crisp": float(action.gripper),
            "gripper_droid": float(action.raw_gripper),
            # What the arm was actually told to do this tick, in radians: the
            # velocity after --speed-scale and the --max-step-rad clamp. The
            # gap between this and `velocity` is the throttling, made visible.
            "delta_rad": _commanded_delta(policy, action).tolist(),
            "target_joints": np.asarray(action.joint_positions).tolist(),
        },
        # The whole chunk the policy last returned, so the panel can show where
        # the current step sits in the plan and what is coming next.
        "prediction": _prediction(policy),
        # The same plan as a path through the image, via FK + the camera
        # calibration. None when the camera has no calibration.
        "overlay": overlay_merge(o.build(policy, state) for o in (overlays or ())),
        # Why there are or are not several plans to draw. Without this the
        # panel can only show nothing, and "nothing" looks the same whether
        # the policy is certain, the server cannot sample, or the sampled
        # inference simply has not come round yet.
        "samples": {
            "requested": int(getattr(policy, "n_samples", 1)),
            "asked": int(getattr(policy, "samples_asked", 0)),
            "available": bool(getattr(policy, "server_has_samples", False)),
            "every": int(getattr(policy, "samples_every", 1)),
            "returned": (0 if getattr(policy, "last_samples", None) is None
                         else int(len(policy.last_samples))),
        },
        "loop": {
            "step": step, "max_steps": args.max_steps,
            "inference_ms": policy.last_inference_ms,
            "n_inferences": policy.n_inferences,
            "chunk_index": policy._chunk_index,
            "open_loop_horizon": policy.open_loop_horizon,
            "speed_scale": args.speed_scale, "rate_hz": args.rate,
            "dry_run": bool(args.dry_run),
        },
    }


def build_app(state: PanelState, analyser: FeatureAnalyser,
              policy=None, history: PromptHistory | None = None):
    """The panel's Flask app, separate from serving it.

    Split out so the routes can be exercised with Flask's test client --
    retasking a policy from a web form is not something to find out is broken
    while a rollout is running.
    """
    from flask import Flask, Response, jsonify, request

    app = Flask(__name__)
    # Werkzeug logs every poll at INFO; at 2 Hz that buries the control output.
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.route("/")
    def index():
        return Response(PANEL_HTML.read_text(), mimetype="text/html")

    @app.route("/state")
    def get_state():
        return jsonify(state.get())

    @app.route("/prompt", methods=["POST"])
    def set_prompt():
        """Retask the policy. Queued, not applied here: see
        Pi05DroidPolicy.request_prompt -- this is a Flask thread and the
        control loop owns the action chunk."""
        if policy is None:
            return jsonify({"ok": False, "error": "no policy attached"}), 400
        prompt = str((request.get_json(silent=True) or {}).get("prompt", "")).strip()
        if not prompt:
            return jsonify({"ok": False, "error": "empty prompt"}), 400
        policy.request_prompt(prompt)
        print(f'  [panel] task -> "{prompt}"')
        return jsonify({
            "ok": True,
            "prompt": prompt,
            "history": [] if history is None else history.add(prompt),
        })

    @app.route("/refit", methods=["POST"])
    def refit():
        analyser.refit()
        return jsonify({"ok": True})

    return app


def serve(state: PanelState, analyser: FeatureAnalyser,
          host: str = "0.0.0.0", port: int = 7100,
          policy=None, history: PromptHistory | None = None) -> threading.Thread:
    """Start the panel in a daemon thread. Returns the thread.

    Checks the port first. Flask binds inside the thread, so a port already
    taken by an earlier rollout fails there, after this function has returned
    and after the URL has been printed -- and the browser then shows the older
    process's panel, which is a genuinely confusing way to debug nothing
    appearing.
    """
    import socket

    probe = socket.socket()
    try:
        probe.bind((host, port))
    except OSError as e:
        raise SystemExit(
            f"cannot serve the panel on port {port}: {e}\n"
            f"An earlier deploy_pi05 is probably still running — its panel is "
            f"what a browser on that port would show, which is not this "
            f"rollout. Stop it, or use --panel-port.") from e
    finally:
        probe.close()

    app = build_app(state, analyser, policy, history)

    thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True,
                               debug=False, use_reloader=False),
        daemon=True)
    thread.start()
    return thread
