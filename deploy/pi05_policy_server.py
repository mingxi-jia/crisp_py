#!/usr/bin/env python3
"""openpi policy server with the two extras the π0.5 panel needs.

openpi's own scripts/serve_policy.py returns only actions: Policy.infer gives
{state, actions, policy_timing} and DroidOutputs slices that to 8 dims. The
panel wants two more things -- the SigLIP vision tokens, and several sampled
plans for one observation -- so this wraps the real policy in a BasePolicy
decorator and hands it to openpi's own WebsocketPolicyServer, which accepts
any BasePolicy. third_party/openpi therefore stays a pristine clone; nothing
here is a fork.

The two extras are independent. --no-features turns off the vision tokens and
nothing else; --max-samples 1 turns off sampling and nothing else.

The features come from the same forward the action came from: the same
weights, the same DroidInputs transform, the same normalised images. See
third_party/openpi/src/openpi/models/pi0.py:114 --
    image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

Run it from the openpi checkout, which has the right environment:

    cd third_party/openpi
    uv run python ../../deploy/pi05_policy_server.py \
        --config pi05_droid --dir gs://openpi-assets/checkpoints/pi05_droid

The wire protocol is unchanged apart from an added "features" key, so the
stock openpi client still works against this server.
"""

import argparse
import logging
import sys

import numpy as np

# Camera role -> the image key DroidInputs produces (droid_policy.py).
# right_wrist_0_rgb is the zero-filled padding image pi0 expects; skip it.
IMAGE_KEY_BY_ROLE = {
    "exterior": "base_0_rgb",
    "wrist": "left_wrist_0_rgb",
}


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pi05_droid",
                   help="openpi train config name")
    p.add_argument("--dir", default="gs://openpi-assets/checkpoints/pi05_droid",
                   help="checkpoint directory")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--default-prompt", default=None)
    p.add_argument("--no-warmup", action="store_true",
                   help="skip the startup warm-up (JAX then compiles on your "
                        "first real inference, stalling the first chunk)")
    p.add_argument("--warmup-rounds", type=int, default=2,
                   help="dummy inferences at startup; the second confirms the "
                        "first actually compiled")
    p.add_argument("--feature-every", type=int, default=1,
                   help="compute vision features only every Nth inference. The "
                        "panel refreshes at ~2 Hz and does not need one per "
                        "inference; raising this cuts the cost proportionally")
    p.add_argument("--no-feature-jit", action="store_true",
                   help="run the vision tower eagerly (slow; for debugging)")
    p.add_argument("--no-features", action="store_true",
                   help="skip the vision tokens. Sampling is unaffected; use "
                        "--max-samples 1 to turn that off instead")
    p.add_argument("--feature-dtype", default="float16", choices=["float16", "float32"],
                   help="float16 halves the wire size; the panel reduces to "
                        "64 PCA components anyway")
    p.add_argument("--max-samples", type=int, default=8,
                   help="ceiling on the per-request n_samples. pi0.5 is a flow "
                        "model sampled from noise, so repeated inference on one "
                        "observation gives different plans; the panel draws "
                        "them to show where the policy is unsure. Batched, so "
                        "N plans cost well under N inferences -- but they are "
                        "still N times the work, hence a cap")
    return p


class PanelPolicy:
    """Wraps an openpi Policy, adding what the panel needs to each inference.

    Two independent extras: vision features, and several sampled plans per
    observation. Implements the BasePolicy surface WebsocketPolicyServer uses
    (infer, reset), delegating everything else to the wrapped policy.

    Always wrap, even with both extras off. The wrapper is also what
    advertises them in the server metadata, so a client can tell "this server
    cannot sample" from "this server chose not to this time"; bypassing it
    makes the server indistinguishable from openpi's stock one.
    """

    def __init__(self, policy, feature_dtype: str = "float16",
                 every: int = 1, jit: bool = True, max_samples: int = 8,
                 features: bool = True):
        self._policy = policy
        self._max_samples = max(1, int(max_samples))
        self._sample_warned = False
        self._features = bool(features)
        self._dtype = np.float16 if feature_dtype == "float16" else np.float32
        self._warned = False
        # The vision tower is a 400M-parameter SigLIP. Run eagerly it costs
        # ~1.5 s per call, dwarfing the ~80 ms jitted action path; jitted it is
        # a fraction of that. `every` additionally skips calls: the panel
        # refreshes at ~2 Hz and does not need a feature map per inference.
        self._every = max(1, int(every))
        self._use_jit = jit
        self._jitted = None
        self._calls = 0
        self._last_features = {}
        self.last_feature_ms = None

    # -- BasePolicy surface ----------------------------------------------
    @property
    def metadata(self) -> dict:
        meta = dict(getattr(self._policy, "metadata", {}) or {})
        meta["features"] = self._features
        meta["feature_dtype"] = np.dtype(self._dtype).name
        meta["samples"] = self._max_samples > 1
        meta["max_samples"] = self._max_samples
        return meta

    def reset(self):
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()

    def infer(self, obs: dict) -> dict:
        import time

        # A flow model samples its plan from noise, so the same observation
        # gives a different chunk every time. One request can ask for several
        # at once; the spread between them is the policy's own uncertainty,
        # which is otherwise invisible.
        n = max(1, min(int(obs.get("n_samples", 1) or 1), self._max_samples))
        if n > 1:
            try:
                result = self._infer_samples(obs, n)
            except Exception:
                # Never let a visualisation feature stop the robot: fall back
                # to the ordinary single-sample path.
                if not self._sample_warned:
                    logging.exception("batched sampling failed; serving one sample")
                    self._sample_warned = True
                result = self._policy.infer(obs)
        else:
            result = self._policy.infer(obs)
        self._calls += 1

        # Honour a per-request opt-out so a rollout without the panel pays
        # nothing, and skip all but every Nth call otherwise.
        want = self._features and bool(obs.get("want_features", True))
        due = (self._calls % self._every) == 0
        if not (want and due):
            result["features"] = self._last_features
            return result

        try:
            t0 = time.perf_counter()
            self._last_features = self._vision_tokens(obs)
            self.last_feature_ms = (time.perf_counter() - t0) * 1000.0
            result["features"] = self._last_features
            result["feature_ms"] = self.last_feature_ms
            if self._calls <= self._every * 2:
                logging.info("vision features: %.0f ms%s", self.last_feature_ms,
                             "" if self._jitted is not None else " (eager — jit unavailable)")
        except Exception as e:
            # Features are a debugging aid; never let them take down control.
            if not self._warned:
                logging.exception("vision features unavailable; serving actions only")
                self._warned = True
            result["features"] = {}
            result["feature_error"] = repr(e)
        return result

    # -- sampling ---------------------------------------------------------
    def _infer_samples(self, obs: dict, n: int) -> dict:
        """N plans for one observation, in a single batched forward pass.

        Goes through the policy's own transforms and its own jitted
        sample_actions, exactly as Policy.infer does -- the only difference is
        that the observation is repeated N times, so sample_actions draws N
        independent noise vectors (pi0.py:231) and integrates N trajectories
        together. Sequential calls would give the same answer at several times
        the latency.

        The first returned sample is handed back as `actions`, so the chunk
        being executed is one of the ones drawn rather than an N+1th that
        nothing shows.
        """
        import time

        import jax
        import jax.numpy as jnp
        import openpi.models.model as _model

        policy = self._policy
        inputs = policy._input_transform(jax.tree.map(lambda x: x, obs))  # noqa: SLF001
        inputs = jax.tree.map(
            lambda x: jnp.repeat(jnp.asarray(x)[np.newaxis, ...], n, axis=0), inputs)
        policy._rng, sample_rng = jax.random.split(policy._rng)           # noqa: SLF001

        observation = _model.Observation.from_dict(inputs)
        start = time.monotonic()
        actions = policy._sample_actions(                                 # noqa: SLF001
            sample_rng, observation, **policy._sample_kwargs)             # noqa: SLF001
        model_ms = (time.monotonic() - start) * 1000.0

        # The output transforms (unnormalise, then keep 8 dims) index the last
        # axis only, so a leading sample axis passes through untouched.
        out = policy._output_transform({                                  # noqa: SLF001
            "state": np.asarray(inputs["state"]),
            "actions": np.asarray(actions),
        })
        samples = np.asarray(out["actions"])
        return {
            "state": np.asarray(inputs["state"])[0],
            "actions": samples[0],
            # float16: 8 samples x 15 steps x 8 dims is 2 KB on the wire, and
            # these are only ever drawn.
            "samples": samples.astype(np.float16),
            "n_samples": int(samples.shape[0]),
            "policy_timing": {"infer_ms": model_ms},
        }

    # -- features ---------------------------------------------------------
    def _vision_tokens(self, obs: dict) -> dict:
        """{"exterior": (256, 2048), "wrist": (256, 2048)} for this observation.

        Replays the policy's own input transform so the tokens come from the
        exact tensors the action path saw, not a re-derived approximation.
        """
        import jax
        import jax.numpy as jnp
        import openpi.models.model as _model

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._policy._input_transform(inputs)          # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        model = self._policy._model                              # noqa: SLF001
        forward = self._vision_forward(model)
        out = {}
        for role, key in IMAGE_KEY_BY_ROLE.items():
            image = observation.images.get(key)
            if image is None:
                continue
            tokens = forward(image)
            out[role] = np.asarray(tokens[0]).astype(self._dtype)
        return out

    def _vision_forward(self, model):
        """The vision tower, jitted once if JAX allows it.

        Eager execution of a 400M ViT is ~1.5 s per call; the first jitted call
        pays compilation, every later one is cheap. Falls back to eager rather
        than failing, since features are only a debugging aid.
        """
        if self._jitted is not None:
            return self._jitted
        eager = lambda image: model.PaliGemma.img(image, train=False)[0]
        if not self._use_jit:
            return eager
        try:
            import jax
            jitted = jax.jit(eager)
            self._jitted = jitted
            logging.info("vision tower jitted; the first call compiles")
            return jitted
        except Exception as e:
            logging.warning("could not jit the vision tower (%s); running eager", e)
            self._use_jit = False
            return eager


def warm_up(served, rounds: int = 2, n_samples: int = 1) -> None:
    """Run dummy inferences so JAX compiles before the robot is live.

    Every jitted path -- openpi's sample_actions, the vision tower, and the
    batched sampler, which compiles again per batch size -- pays its
    compilation on first call. Without this the cost lands on the first action
    chunk of a rollout, with the arm already running.
    """
    import time

    from openpi.policies import droid_policy

    # Each distinct batch size is a separate compilation, so warm the batched
    # sampler at the size the panel will actually ask for.
    sizes = [1] * rounds + ([n_samples] * 2 if n_samples > 1 else [])
    for i, n in enumerate(sizes):
        example = droid_policy.make_droid_example()
        example["prompt"] = "warm up"
        example["want_features"] = True
        example["n_samples"] = n
        t = time.perf_counter()
        try:
            served.infer(example)
        except Exception as e:
            logging.warning("warm-up inference %d failed: %s", i + 1, e)
            return
        dt = (time.perf_counter() - t) * 1000
        logging.info("warm-up %d/%d: %.0f ms, %d sample(s)%s",
                     i + 1, len(sizes), dt, n,
                     "  (compiling)" if i in (0, rounds) else "  (compiled)")


def main():
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    try:
        from openpi.policies import policy_config as _policy_config
        from openpi.serving import websocket_policy_server
        from openpi.training import config as _config
    except ImportError as e:
        sys.exit(
            f"cannot import openpi ({e}).\n"
            "Run this from the openpi checkout so its environment is active:\n"
            "    cd third_party/openpi\n"
            "    uv run python ../../deploy/pi05_policy_server.py ...")

    train_config = _config.get_config(args.config)
    logging.info("loading %s from %s", args.config, args.dir)
    policy = _policy_config.create_trained_policy(
        train_config, args.dir, default_prompt=args.default_prompt)

    # Wrap unconditionally: --no-features switches the feature path off inside
    # the wrapper rather than removing it, so sampling -- and the metadata
    # that advertises both -- survives.
    served = PanelPolicy(
        policy, feature_dtype=args.feature_dtype,
        every=args.feature_every, jit=not args.no_feature_jit,
        max_samples=args.max_samples, features=not args.no_features)

    if not args.no_warmup:
        logging.info("warming up (JAX compiles now, not on your first chunk)...")
        warm_up(served, rounds=args.warmup_rounds,
                n_samples=args.max_samples)

    metadata = getattr(served, "metadata", {}) or {}
    logging.info("serving on %s:%d  features=%s  max_samples=%d",
                 args.host, args.port, not args.no_features, args.max_samples)
    websocket_policy_server.WebsocketPolicyServer(
        policy=served, host=args.host, port=args.port, metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
