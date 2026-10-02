"""Tests for deploy/pi05_policy_server.py. No checkpoint, no GPU, no network.

Run with openpi's interpreter, since this imports jax:

    cd third_party/openpi && .venv/bin/python ../../deploy/test_pi05_policy_server.py

The batched sampler reaches into openpi's Policy internals (_input_transform,
_sample_actions, _output_transform, _rng), which is fine -- the feature path
already does -- but it means the shapes it builds are only checked here. A
stub policy with the same attribute surface catches a wrong batch axis or a
missed rng split without loading 3 GB of weights.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deploy.pi05_policy_server import PanelPolicy

HORIZON, DIM = 15, 32


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    assert cond, name


class StubPolicy:
    """The attribute surface openpi's Policy exposes, and nothing else."""

    def __init__(self):
        self._rng = jax.random.key(0)
        self._sample_kwargs = {"num_steps": 10}
        self.metadata = {"stub": True}
        self.seen = []                      # (batch_size, rng) per sample_actions
        self.single_calls = 0

    def _input_transform(self, obs):
        return {"state": np.asarray(obs["observation/joint_position"], np.float32),
                "image": np.zeros((4, 4, 3), np.float32)}

    def _sample_actions(self, rng, observation, **kwargs):
        batch = observation["state"].shape[0]
        self.seen.append((batch, rng, kwargs))
        # Distinct per batch element, so a collapsed axis is visible.
        return jnp.stack([jnp.full((HORIZON, DIM), float(i)) for i in range(batch)])

    def _output_transform(self, data):
        return {"actions": np.asarray(data["actions"])[..., :8]}

    def infer(self, obs):
        self.single_calls += 1
        return {"state": np.zeros(7), "actions": np.zeros((HORIZON, 8)),
                "policy_timing": {"infer_ms": 1.0}}


class FakeObservation(dict):
    @classmethod
    def from_dict(cls, d):
        return cls(d)


def main():
    import openpi.models.model as _model
    _model.Observation.from_dict = FakeObservation.from_dict   # stub the model side

    obs = {"observation/joint_position": np.arange(7, dtype=np.float32),
           "prompt": "test"}

    print("=== one sample is the ordinary path ===")
    stub = StubPolicy()
    served = PanelPolicy(stub, every=10**9, jit=False, max_samples=8)
    out = served.infer({**obs, "n_samples": 1, "want_features": False})
    check("delegates to Policy.infer", stub.single_calls == 1 and not stub.seen)
    check("no samples key", "samples" not in out)

    print("\n=== N samples come from one batched call ===")
    stub = StubPolicy()
    served = PanelPolicy(stub, every=10**9, jit=False, max_samples=8)
    before = stub._rng
    out = served.infer({**obs, "n_samples": 5, "want_features": False})
    check("Policy.infer was not also called", stub.single_calls == 0)
    check("exactly one forward pass", len(stub.seen) == 1, f"{len(stub.seen)}")
    check("batched to N", stub.seen[0][0] == 5, f"batch {stub.seen[0][0]}")
    check("the rng was advanced", not jnp.array_equal(
        jax.random.key_data(before), jax.random.key_data(stub._rng)))
    check("sample_kwargs are passed through", stub.seen[0][2] == {"num_steps": 10})
    check("samples are (N, H, 8)", out["samples"].shape == (5, HORIZON, 8),
          str(out["samples"].shape))
    check("samples are float16 on the wire", out["samples"].dtype == np.float16)
    check("the executed chunk is sample 0",
          np.allclose(out["actions"], np.asarray(out["samples"][0], np.float64)))
    check("samples are distinct, not a broadcast copy",
          len({float(s[0, 0]) for s in out["samples"]}) == 5)
    check("state is unbatched", np.asarray(out["state"]).shape == (7,),
          str(np.asarray(out["state"]).shape))
    check("reports n_samples", out["n_samples"] == 5)
    check("timing is reported", out["policy_timing"]["infer_ms"] >= 0)

    print("\n=== the cap and the fallback ===")
    stub = StubPolicy()
    served = PanelPolicy(stub, every=10**9, jit=False, max_samples=3)
    out = served.infer({**obs, "n_samples": 99, "want_features": False})
    check("n_samples is capped", out["samples"].shape[0] == 3, str(out["samples"].shape))

    class Broken(StubPolicy):
        def _sample_actions(self, *a, **k):
            raise RuntimeError("out of memory")

    stub = Broken()
    served = PanelPolicy(stub, every=10**9, jit=False, max_samples=8)
    out = served.infer({**obs, "n_samples": 4, "want_features": False})
    check("a failed batch falls back to one sample",
          stub.single_calls == 1 and "samples" not in out)
    check("and the fallback still returns actions", out["actions"].shape == (HORIZON, 8))

    print("\n=== metadata advertises the feature ===")
    meta = PanelPolicy(StubPolicy(), max_samples=6).metadata
    check("samples announced", meta["samples"] is True)
    check("cap announced", meta["max_samples"] == 6)
    check("features announced", meta["features"] is True)

    print("\n=== the two extras are independent ===")
    # --no-features used to bypass the wrapper entirely, which silently took
    # sampling with it and left the server indistinguishable from the stock
    # one. Features off must mean features off and nothing else.
    stub = StubPolicy()
    served = PanelPolicy(stub, features=False, jit=False, max_samples=8)
    check("features off is advertised", served.metadata["features"] is False)
    check("sampling still advertised", served.metadata["samples"] is True)
    out = served.infer({**obs, "n_samples": 4, "want_features": True})
    check("and sampling still works", out["samples"].shape[0] == 4,
          str(out["samples"].shape))
    check("while no features are computed", out.get("features") in (None, {}),
          repr(out.get("features")))

    served = PanelPolicy(StubPolicy(), max_samples=1)
    check("max_samples 1 turns sampling off", served.metadata["samples"] is False)

    print("\nAll policy-server tests passed.")


if __name__ == "__main__":
    main()
