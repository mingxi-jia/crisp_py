"""Tests for the panel's own routes: retasking and the task history.

    python -m deploy.pi05.test_panel

No robot, no policy server, no browser. Retasking a policy from a web form is
not something to discover is broken while a rollout is running, and the
threading rule it depends on -- the panel queues, the control loop applies --
is invisible until it corrupts a chunk.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from deploy.pi05 import panel
from deploy.pi05.policy import Pi05DroidPolicy


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    assert cond, name


def fake_policy(prompt="pick up the green box"):
    """A policy with the attributes the panel touches, and no server."""
    p = Pi05DroidPolicy.__new__(Pi05DroidPolicy)
    p.prompt, p._pending_prompt = prompt, None
    p._chunk, p._chunk_index = np.zeros((5, 8)), 2
    p.last_request = {"prompt": prompt}
    p.last_features, p.last_raw_frames, p.last_frame_shapes = {}, {}, {}
    p.n_inferences, p.last_inference_ms, p.open_loop_horizon = 3, 80.0, 8
    p.speed_scale, p.max_step_rad, p.binarize_gripper = 1.0, 0.2, True
    p.n_samples, p.samples_every, p.samples_asked = 1, 1, 0
    p.server_has_samples, p.last_samples = False, None
    return p


class Args:
    external_camera, wrist_camera = "agentview", "inhand"
    exterior_fit, wrist_fit = "box", "pad"
    max_steps, speed_scale, rate, dry_run = 0, 1.0, 15.0, True


def main():
    policy = fake_policy()
    history = panel.PromptHistory(Path(tempfile.mkdtemp()) / "prompts.json")
    history.add(policy.prompt)
    state, analyser = panel.PanelState(), panel.FeatureAnalyser()
    client = panel.build_app(state, analyser, policy, history).test_client()

    print("=== retasking ===")
    answer = client.post("/prompt", json={"prompt": "put the tools on the plate"})
    check("accepted", answer.status_code == 200 and answer.get_json()["ok"])
    # The panel runs on a Flask thread; the control loop owns the chunk. If the
    # panel applied the change itself, reset() could land between
    # predict_action's "no need to re-infer" check and its index into the
    # chunk, and the rollout would die on a None.
    check("queued on the policy, not applied",
          policy.pending_prompt == "put the tools on the plate"
          and policy.prompt == "pick up the green box")
    check("the chunk in flight is untouched", policy._chunk is not None)

    check("applied by the control loop", policy._apply_pending_prompt())
    check("and the stale chunk is dropped", policy._chunk is None)
    check("the task is now the new one", policy.prompt == "put the tools on the plate")
    check("nothing left pending", policy.pending_prompt is None)
    check("re-applying is a no-op", not policy._apply_pending_prompt())

    print("\n=== retasking mid-chunk asks for images ===")
    # The crash this replaces: the loop decides whether to fetch camera frames
    # *before* calling predict_action. A queued prompt drops the chunk, which
    # forces an inference the loop had already decided against, and
    # build_request got a state-only reading.
    mid = fake_policy()                       # its own policy: the one above
    mid._chunk, mid._chunk_index = np.zeros((15, 8)), 2   # is still in use
    check("mid-chunk, no inference is due", not mid.needs_inference)
    mid.request_prompt("a different task")
    check("a queued prompt makes one due", mid.needs_inference)

    class StateOnly:
        """What robot.get_state() returns: proprioception, no cameras."""
        joint_values = np.zeros(7)
        gripper_value = 0.0

    try:
        mid.predict_action(StateOnly())
        failed = ""
    except RuntimeError as e:
        failed = str(e)
    except AttributeError as e:
        failed = f"AttributeError: {e}"
    check("a state-only reading is refused by name, not by AttributeError",
          "needs_inference" in failed, failed[:80])

    # And the condition the loop uses is the policy's, not a copy of it.
    loop = Path("deploy/deploy_pi05.py").read_text()
    check("the control loop does not re-derive it",
          "policy.needs_inference" in loop and "policy._chunk_index >=" not in loop)

    print("\n=== refusals ===")
    check("empty", client.post("/prompt", json={"prompt": "   "}).status_code == 400)
    check("absent", client.post("/prompt", json={}).status_code == 400)
    unattached = panel.build_app(state, analyser, None, history).test_client()
    check("no policy attached",
          unattached.post("/prompt", json={"prompt": "x"}).status_code == 400)

    print("\n=== history ===")
    for task in ("wipe the table", "put the tools on the plate"):
        client.post("/prompt", json={"prompt": task})
    prompts = history.prompts()
    check("most recently used first",
          prompts[0] == "put the tools on the plate", str(prompts))
    check("no duplicates", len(prompts) == len(set(prompts)))
    check("uses are counted",
          next(e for e in history.entries()
               if e["prompt"] == "put the tools on the plate")["count"] == 2)
    check("survives a restart", panel.PromptHistory(history.path).prompts() == prompts)
    check("a corrupt file is not fatal", _corrupt_is_survivable())
    check("history can be turned off entirely",
          panel.PromptHistory(False).add("x") and panel.PromptHistory(False).path is None)

    print("\n=== payload ===")
    policy.last_request = {"prompt": "wipe the table"}
    policy.request_prompt("stack the blocks")
    payload = panel.build_payload(policy, None, Args(), 1, analyser, None, (), history)
    # Three prompts that differ while a change is taking effect, and the
    # difference is the whole reason the panel can say "queued".
    check("what the last inference ran with", payload["prompt"] == "wipe the table")
    check("what the policy will run next",
          payload["prompt_active"] == "put the tools on the plate")
    check("what has been asked for", payload["prompt_pending"] == "stack the blocks")
    check("history rides along", len(payload["prompt_history"]) == 3)
    check("payload serialises", json.dumps(payload) is not None)

    print("\nAll panel tests passed.")


def _corrupt_is_survivable() -> bool:
    path = Path(tempfile.mkdtemp()) / "prompts.json"
    path.write_text("{not json at all")
    return panel.PromptHistory(path).prompts() == []


if __name__ == "__main__":
    main()
