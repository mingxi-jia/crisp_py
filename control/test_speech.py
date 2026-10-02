"""Tests for control/speech.py: the words, the endpointer and the hand-off.

    python -m control.test_speech

No microphone and no model. What is checked here is everything between the
audio and the policy -- which is where a spoken interface goes wrong quietly:
a terminator that fires inside a sentence, an endpointer that never closes an
utterance, or the robot reading its own confirmation back as the next task.
"""

import time

import numpy as np

from control.speech import (
    CANCEL, TERMINATOR, Endpointer, SpeechPrompter, is_filler, normalise,
    split_phrase, split_terminator,
)


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if detail else ""))
    assert cond, name


def frames(endpointer, level, count, seed=0):
    """`count` frames of noise at a given RMS, pushed through the endpointer."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        frame = rng.normal(0, level, endpointer.frame_len).astype(np.float32)
        got = endpointer.push(frame)
        if got is not None:
            out.append(got)
    return out


def main():
    print("=== hearing the terminator ===")
    for said, expect in [
        ("pick up the green box done", (True, "pick up the green box")),
        ("Pick up the green box. Done.", (True, "pick up the green box")),
        ("put the tools on the plate", (False, "put the tools on the plate")),
        ("Done.", (True, "")),
        # The one that would be maddening: a task with the word in the middle.
        ("put the done sign on the table", (False, "put the done sign on the table")),
    ]:
        check(f"{said!r}", split_terminator(said) == expect, str(split_terminator(said)))

    print("\n=== hearing the change of mind ===")
    for said, expect in [
        ("nevermind", True), ("Never mind.", True),
        ("pick up the box, never mind", True),
        ("never mind the mess", False), ("wipe the table", False),
    ]:
        got = split_phrase(said, CANCEL)[0]
        check(f"{said!r} -> {'cancels' if expect else 'does not cancel'}",
              got == expect)
    check("filler is not a task",
          all(is_filler(x) for x in ["Okay.", "Thank you.", "you", ""]))
    check("a task is not filler", not is_filler("pick up the green box"))

    print("\n=== endpointer ===")
    ep = Endpointer(silence_ms=200, min_speech_ms=100, threshold_scale=4.0,
                    floor_frames=20, floor_min=0.001)
    check("calibrates before it listens", not frames(ep, 0.002, 20) and ep.floor is not None,
          f"floor {ep.floor:.4f}")
    check("room noise is not speech", not frames(ep, 0.002, 50))
    utterances = frames(ep, 0.05, 30) + frames(ep, 0.002, 20)
    check("loud then quiet closes exactly one utterance",
          len(utterances) == 1, f"{len(utterances)}")
    check("and it is long enough to transcribe",
          utterances[0].size > ep.frame_len * 10, f"{utterances[0].size} samples")
    check("a click is discarded", not (frames(ep, 0.05, 2) + frames(ep, 0.002, 20)))

    long_ep = Endpointer(silence_ms=200, min_speech_ms=100, max_utterance_s=0.4,
                         floor_frames=10, floor_min=0.001)
    frames(long_ep, 0.002, 10)
    check("a monologue is cut at the ceiling rather than buffered forever",
          len(frames(long_ep, 0.05, 120)) >= 2)
    # A ceiling below the minimum length would throw every cut away as too
    # short, and the microphone would appear to work while never producing a
    # word.
    silly = Endpointer(silence_ms=800, min_speech_ms=250, max_utterance_s=0.1,
                       floor_frames=5, floor_min=0.001)
    frames(silly, 0.002, 5)
    check("an impossible ceiling is raised to something workable",
          len(frames(silly, 0.05, 200)) >= 1, f"max_frames {silly.max_frames}")

    print("\n=== the hand-off ===")
    got = []
    spoken = []

    class Fake(SpeechPrompter):
        """Everything but the model and the microphone."""
        def transcribe(self, audio):
            return self._script.pop(0)
        def say(self, text):
            spoken.append(text)

    def run(script, on_cancel=None):
        got.clear(); spoken.clear()
        p = Fake(got.append, on_cancel=on_cancel)
        p._script = list(script)
        for _ in script:
            p._handle(np.zeros(160, dtype=np.float32))
        return p

    p = run(["pick up the green box", "and put it in the bin", "done"])
    check("speech accumulates across breaths",
          got == ["pick up the green box and put it in the bin"], str(got))
    check("the task is read back", spoken == got, str(spoken))
    check("the buffer is empty afterwards", p.status()["buffer"] == "")

    p = run(["wipe the table done"])
    check("a task ended in one breath commits immediately",
          got == ["wipe the table"], str(got))

    p = run(["pick up the green box", "nevermind"])
    check("a change of mind mid-task sends nothing", got == [])
    check("and says so", spoken == ["cancelled"], str(spoken))
    check("leaving nothing buffered", p.status()["buffer"] == "")

    p = run(["nevermind"], on_cancel=lambda: "the previous task")
    check("a change of mind with nothing said restores the previous task",
          spoken == ["back to the previous task"], str(spoken))
    p = run(["nevermind"], on_cancel=lambda: None)
    check("and says when there is nothing to restore",
          spoken == ["nothing to go back to"], str(spoken))

    p = run(["Okay.", "pick up the box", "done"])
    check("filler between breaths is dropped", got == ["pick up the box"], str(got))

    print("\n=== the robot must not hear itself ===")
    # spd-say plays through the speaker the microphone is listening to. If the
    # capture loop did not mute, "pick up the green box" read back would be
    # transcribed and committed as the next task, over and over.
    said = []

    class Timed(SpeechPrompter):
        def transcribe(self, audio): return "x done"

    p = Timed(said.append, speak_command=["true"], mute_tail_s=0.15)
    p.say("pick up the green box")
    time.sleep(0.05)
    check("muted while speaking", p.status()["muted"])
    time.sleep(0.4)
    check("unmuted once the room is quiet", not p.status()["muted"])
    p.speak_enabled = False
    p.say("this should not mute")
    check("nothing to say, nothing muted", not p.status()["muted"])

    print("\n=== choosing a voice ===")
    from control.speech import (CommandSpeaker, PiperSpeaker, find_piper_voice,
                                make_speaker)

    check("none means silence", make_speaker("none") is None)
    check("spd-say is the fallback engine",
          isinstance(make_speaker("spd-say"), CommandSpeaker))
    if find_piper_voice() is None:
        print("  SKIP  piper (no voice downloaded)")
    else:
        check("auto prefers the neural voice",
              isinstance(make_speaker("auto"), PiperSpeaker))
        check("a voice can be named",
              find_piper_voice("en_US-lessac-medium") is not None)
    check("an unknown voice name finds nothing",
          find_piper_voice("no_such_voice") is None)

    print("\n=== the transcriber is told what to expect ===")
    # Whisper's initial_prompt biases decoding. The task history is exactly
    # the right vocabulary -- short commands, repeated, full of object names a
    # general model does not expect. Measured on this machine: without it,
    # "put the tools on the plate" came back "But the toolman plays".
    seen = {}

    class Hinted(SpeechPrompter):
        class _Model:
            def transcribe(self, audio, **kwargs):
                seen.update(kwargs)
                class Seg: text = "wipe the table"
                return [Seg()], None

    h = Hinted(lambda t: None, hint=lambda: "wipe the table. done.")
    h._model = Hinted._Model()
    h.transcribe(np.zeros(160, dtype=np.float32))
    check("the hint reaches the model",
          seen.get("initial_prompt") == "wipe the table. done.", str(seen.get("initial_prompt")))

    broken = Hinted(lambda t: None, hint=lambda: 1 / 0)
    broken._model = Hinted._Model()
    broken.transcribe(np.zeros(160, dtype=np.float32))
    check("a broken hint does not stop transcription",
          seen.get("initial_prompt") is None)

    print("\n=== status ===")
    status = SpeechPrompter(lambda t: None).status()
    for key in ("enabled", "active", "state", "error", "buffer", "heard",
                "last_prompt", "terminator", "cancel", "level", "threshold",
                "muted", "speaking"):
        check(f"status reports {key}", key in status)
    import json
    check("status serialises for the panel", json.dumps(status) is not None)

    print("\nAll speech tests passed.")


if __name__ == "__main__":
    main()
