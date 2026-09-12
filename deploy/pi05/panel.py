"""The π0.5 panel: a small web server showing what the policy sees and thinks.

Served on its own port by deploy_pi05 --panel, so nothing π0.5-specific has to
live in control/robot_server.py. Three panes:

  1. the observation actually sent to the policy (post fit_image, so you see
     the crop/letterbox the model sees, not the raw camera frame)
  2. a PCA map of the SigLIP patch tokens, per camera
  3. click a patch -> cosine similarity of that patch against every other

The heavy tensors never reach the browser: features are projected onto a fixed
64-component PCA basis (~64 KB/camera) and the similarity map is computed in
the page on click. See features.reduce_for_client for how good that
approximation is.
"""

import base64
import io
import threading
import time
from pathlib import Path

import numpy as np

from deploy.pi05.features import (
    DEFAULT_COMPONENTS, FeatureMap, PCABasis, pca_rgb,
    reduce_for_client,
)

PANEL_HTML = Path(__file__).with_name("panel.html")


def _jpeg_b64(rgb, quality: int = 80) -> str | None:
    if rgb is None:
        return None
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


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


def build_payload(policy, action, args, step: int, analyser: FeatureAnalyser) -> dict:
    """Everything the panel needs for one step."""
    req = policy.last_request or {}
    joint = req.get("observation/joint_position")
    grip = req.get("observation/gripper_position")
    feats = getattr(policy, "last_features", None) or {}
    return {
        "prompt": req.get("prompt", ""),
        "obs": {
            "exterior_jpeg": _jpeg_b64(req.get("observation/exterior_image_1_left")),
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


def serve(state: PanelState, analyser: FeatureAnalyser,
          host: str = "0.0.0.0", port: int = 7100) -> threading.Thread:
    """Start the panel in a daemon thread. Returns the thread."""
    from flask import Flask, Response, jsonify

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

    @app.route("/refit", methods=["POST"])
    def refit():
        analyser.refit()
        return jsonify({"ok": True})

    thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, threaded=True,
                               debug=False, use_reloader=False),
        daemon=True)
    thread.start()
    return thread
