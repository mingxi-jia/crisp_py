"""Speaking the task to the robot.

A microphone, an endpointer and Whisper. Say something, say "done", and the
text becomes the policy's prompt -- the same queued hand-off the panel's text
box uses, so nothing about the control loop changes.

Why "done" and not silence
--------------------------
Endpointing on silence alone would fire on every pause, and a task said with a
pause in the middle ("pick up the green box... and put it in the bin") would
arrive as two prompts, the first of which the robot would start doing. A word
that ends the utterance makes the boundary the speaker's decision. Speech
between terminators accumulates, so it can be said in as many breaths as it
takes.

Capture is `arecord`, which is already on any machine with ALSA, rather than a
Python audio binding: one less package in an environment where a numpy bump
would take ROS down with it.

Nothing here may raise into the control loop. A microphone that disappears, a
model that will not load, a transcription that fails -- all of them end up in
status() and leave the rollout running.
"""

import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 20
TERMINATOR = "done"
CANCEL = "nevermind"

# speech-dispatcher is already on any desktop Linux and drives espeak-ng; -w
# waits for the utterance to finish, which is what tells us when to unmute.
# It is the fallback, not the choice: espeak is robotic enough that Whisper
# itself mis-hears it ("put the tools on the plate" came back as "But the
# toolman plays"), which is a fair proxy for how a person finds it.
SPEAK_COMMAND = ["spd-say", "-w"]

# Piper is a small neural TTS that runs locally on CPU, faster than real time.
# Voices live wherever they were downloaded; this is where the setup note puts
# them.
PIPER_VOICE_DIR = Path.home() / ".cache" / "piper"
DEFAULT_PIPER_VOICE = "en_US-lessac-medium"

# Whisper writes "Okay." for silence and similar filler for noise; these are
# not tasks and must not accumulate into one.
FILLER = {"", "you", "thank you", "thanks", "bye", "okay", "ok", "uh", "um",
          "mm", "hmm", "ah", "oh", "so", "the", "a"}


def normalise(text: str) -> str:
    """Lowercase, unpunctuated, single-spaced -- for comparing words."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", str(text).lower())).strip()


def split_phrase(text: str, phrase: str):
    """(ended with it, text without it) -- did this utterance end with `phrase`?

    Only at the end: "put the done sign on the table" is a task, not a task
    called "put the" followed by a full stop. Multi-word phrases also match
    despaced, because Whisper writes "never mind" about as often as
    "nevermind" and the speaker cannot hear the difference.
    """
    words = normalise(text).split()
    wanted = normalise(phrase).split()
    if not wanted:
        return False, " ".join(words)
    if len(wanted) <= len(words) and words[-len(wanted):] == wanted:
        return True, " ".join(words[:-len(wanted)]).strip()
    joined = "".join(wanted)
    for take in (1, 2, 3):
        if take <= len(words) and "".join(words[-take:]) == joined:
            return True, " ".join(words[:-take]).strip()
    return False, " ".join(words).strip()


def split_terminator(text: str, terminator: str = TERMINATOR):
    """(ended, remaining text) -- does this utterance end the prompt?"""
    return split_phrase(text, terminator)


def is_filler(text: str) -> bool:
    return normalise(text) in FILLER


class CommandSpeaker:
    """Anything that takes the text as its last argument, e.g. spd-say -w."""

    name = "spd-say"

    def __init__(self, command=None):
        self.command = list(command or SPEAK_COMMAND)

    def say(self, text: str):
        subprocess.run(self.command + [str(text)], capture_output=True, timeout=60)


class PiperSpeaker:
    """Piper: text in, WAV out, straight into a player.

    Piped rather than written to a file so nothing has to be cleaned up, and
    the player starts as the audio arrives.
    """

    name = "piper"

    def __init__(self, voice, player=("aplay", "-q", "-")):
        self.voice = str(voice)
        self.player = list(player)

    def say(self, text: str):
        piper = subprocess.Popen(
            ["python3", "-m", "piper", "-m", self.voice, "-f", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        player = subprocess.Popen(self.player, stdin=piper.stdout,
                                  stderr=subprocess.DEVNULL)
        piper.stdout.close()          # the player owns the read end now
        try:
            piper.stdin.write(str(text).encode())
            piper.stdin.close()
            player.wait(timeout=120)
        finally:
            for proc in (piper, player):
                if proc.poll() is None:
                    proc.terminate()


def find_piper_voice(voice=None):
    """A voice file for Piper, or None. Accepts a path or a voice name."""
    if voice:
        candidate = Path(voice)
        if candidate.exists():
            return candidate
        candidate = PIPER_VOICE_DIR / f"{voice}.onnx"
        return candidate if candidate.exists() else None
    for name in (DEFAULT_PIPER_VOICE, "*"):
        found = sorted(PIPER_VOICE_DIR.glob(f"{name}.onnx"))
        if found:
            return found[0]
    return None


def make_speaker(engine: str = "auto", voice=None, command=None):
    """The best available way to talk, or None.

    'auto' prefers Piper because espeak is hard to listen to; it falls back
    rather than failing, since a missing voice should cost the confirmation,
    not the rollout.
    """
    if engine == "none":
        return None
    if engine in ("auto", "piper"):
        found = find_piper_voice(voice)
        if found is not None:
            return PiperSpeaker(found)
        if engine == "piper":
            raise SystemExit(
                f"no Piper voice found. Download one into {PIPER_VOICE_DIR}:\n"
                f"    python -c \"from piper.download_voices import "
                f"download_voice; download_voice('{DEFAULT_PIPER_VOICE}', "
                f"'{PIPER_VOICE_DIR}')\"\n"
                f"or pass --speak-engine spd-say for the robotic one.")
    if shutil.which(SPEAK_COMMAND[0]):
        return CommandSpeaker(command)
    return None


def _read_exact(stream, count: int) -> bytes | None:
    """Exactly `count` bytes, or None at end of stream.

    An unbuffered pipe returns whatever is ready, which for a 20 ms audio
    frame is regularly less than asked for. Treating a short read as the
    microphone dying made it look broken within a second of starting.
    """
    chunks, remaining = [], count
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class Endpointer:
    """Cuts a stream of audio frames into utterances, by energy.

    The noise floor is measured rather than assumed: a threshold that works in
    a quiet room fires continuously next to a robot's fans. Everything is in
    frames so the arithmetic stays integer and obvious.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, frame_ms: int = FRAME_MS,
                 silence_ms: int = 800, min_speech_ms: int = 250,
                 max_utterance_s: float = 20.0, threshold_scale: float = 4.0,
                 floor_frames: int = 40, floor_min: float = 0.004):
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        # A ceiling below the floor is a silent failure: every cut would be
        # thrown away as too short and nothing would ever be transcribed.
        self.max_frames = max(int(max_utterance_s * 1000 // frame_ms),
                              self.min_speech_frames + self.silence_frames + 1)
        self.threshold_scale = threshold_scale
        self.floor_frames = floor_frames
        # Even in a silent room the threshold should not chase zero, or the
        # first breath becomes an utterance.
        self.floor_min = floor_min

        self.floor = None
        self._floor_samples = []
        self._speech = []
        self._quiet = 0
        self.level = 0.0
        self.speaking = False

    def reset(self):
        """Forget any part-heard utterance. Used after the robot has spoken:
        whatever the microphone picked up during that was the speaker."""
        self._speech, self._quiet = [], 0
        self.speaking = False

    @property
    def threshold(self) -> float:
        floor = self.floor if self.floor is not None else self.floor_min
        return max(floor * self.threshold_scale, self.floor_min)

    def push(self, frame: np.ndarray):
        """One frame in; a finished utterance out, or None."""
        frame = np.asarray(frame, dtype=np.float32)
        self.level = float(np.sqrt(np.mean(frame ** 2)) if frame.size else 0.0)

        if self.floor is None:
            self._floor_samples.append(self.level)
            if len(self._floor_samples) >= self.floor_frames:
                # Median, not mean: a cough during calibration would otherwise
                # deafen the endpointer for the rest of the session.
                self.floor = float(np.median(self._floor_samples))
            return None

        if self.level > self.threshold:
            self._speech.append(frame)
            self._quiet = 0
            self.speaking = True
        elif self._speech:
            self._speech.append(frame)
            self._quiet += 1
            if self._quiet >= self.silence_frames:
                return self._finish()
        else:
            # Quiet and nothing buffered: keep tracking the room, slowly, so a
            # fan switching on does not leave the threshold stranded.
            self.floor = 0.98 * self.floor + 0.02 * self.level

        if len(self._speech) >= self.max_frames:
            return self._finish()
        return None

    def _finish(self):
        frames, self._speech, self._quiet = self._speech, [], 0
        self.speaking = False
        if len(frames) < self.min_speech_frames + self.silence_frames:
            return None                      # a click or a chair, not speech
        return np.concatenate(frames)


class SpeechPrompter:
    """Microphone -> Whisper -> `on_prompt(text)`, on its own thread."""

    def __init__(self, on_prompt, model: str = "base.en", device: str = "cpu",
                 compute_type: str = "int8", alsa_device: str | None = None,
                 language: str = "en", terminator: str = TERMINATOR,
                 cancel: str = CANCEL, on_cancel=None, speak: bool = True,
                 speak_command=None, speaker=None, speak_engine: str = "auto",
                 speak_voice=None, hint=None, mute_tail_s: float = 0.4,
                 silence_ms: int = 800, threshold_scale: float = 4.0,
                 max_utterance_s: float = 20.0):
        self.on_prompt = on_prompt
        # Called when the speaker changes their mind with nothing part-said;
        # returns the prompt that was restored, or None if there is none.
        self.on_cancel = on_cancel
        self.cancel = cancel
        self.speak_enabled = bool(speak)
        self.speaker = (speaker if speaker is not None
                        else (CommandSpeaker(speak_command) if speak_command
                              else make_speaker(speak_engine, speak_voice)))
        # Whisper takes an `initial_prompt` that biases decoding towards the
        # words in it. Fed the tasks this operator actually uses, it stops
        # turning "put the tools on the plate" into "But the toolman plays" --
        # measured, not hoped for.
        self.hint = hint
        # The microphone hears the speaker. Without a mute the robot reads its
        # own confirmation back as the next task -- and "pick up the green box"
        # spoken aloud is a perfectly good task, so nothing downstream would
        # catch it.
        self.mute_tail_s = float(mute_tail_s)
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.alsa_device = alsa_device
        self.language = language
        self.terminator = terminator
        self.endpointer_kwargs = dict(silence_ms=silence_ms,
                                      threshold_scale=threshold_scale,
                                      max_utterance_s=max_utterance_s)

        self._model = None
        self._thread = None
        self._stop = threading.Event()
        self._proc = None
        self._endpointer = Endpointer(**self.endpointer_kwargs)
        self._state = "stopped"
        self._error = None
        self._buffer = ""          # what has been said since the last "done"
        self._heard = ""           # the last transcript, prompt or not
        self._last_prompt = ""
        self._utterances = 0
        self._muted = False
        self._lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.active:
            return self.status()
        self._stop.clear()
        self._error = None
        self._state = "starting"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self.status()

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._state = "stopped"
        return self.status()

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": True,
                "active": self.active,
                "state": self._state,
                "error": self._error,
                "buffer": self._buffer,
                "heard": self._heard,
                "last_prompt": self._last_prompt,
                "utterances": self._utterances,
                "terminator": self.terminator,
                "model": self.model_name,
                "voice": None if self.speaker is None else self.speaker.name,
                "speaking": self._endpointer.speaking,
                "muted": self._muted,
                "cancel": self.cancel,
                # Level and threshold together are the only way to tell "the
                # mic is dead" from "you are too quiet" without guessing.
                "level": round(float(self._endpointer.level), 5),
                "threshold": round(float(self._endpointer.threshold), 5),
            }

    # -- the loop --------------------------------------------------------
    def _run(self):
        try:
            self._state = "loading the model"
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_name, device=self.device,
                                       compute_type=self.compute_type)
            self._state = "calibrating"
            self._proc = self._arecord()
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            self._state = "failed"
            return

        frame_bytes = self._endpointer.frame_len * 2
        try:
            while not self._stop.is_set():
                raw = _read_exact(self._proc.stdout, frame_bytes)
                if raw is None:
                    self._error = (
                        f"the microphone stopped delivering audio"
                        f"{f' ({self.alsa_device})' if self.alsa_device else ''}. "
                        f"`arecord -l` lists the capture devices; pass one with "
                        f"--listen-alsa, e.g. plughw:2,0")
                    self._state = "failed"
                    return
                if self._muted:
                    # Keep draining the pipe so ALSA does not overrun, but
                    # hear none of it: this is the robot talking.
                    self._state = "speaking"
                    continue
                frame = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
                utterance = self._endpointer.push(frame)
                if self._endpointer.floor is None:
                    continue
                self._state = "listening" if utterance is None else "transcribing"
                if utterance is not None:
                    self._handle(utterance)
                    self._state = "listening"
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            self._state = "failed"
        finally:
            if self._proc is not None:
                self._proc.terminate()

    def _arecord(self):
        command = ["arecord", "-f", "S16_LE", "-r", str(SAMPLE_RATE),
                   "-c", "1", "-t", "raw", "-q"]
        if self.alsa_device:
            command[1:1] = ["-D", self.alsa_device]
        return subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=0)

    def transcribe(self, audio: np.ndarray) -> str:
        hint = None
        if self.hint is not None:
            try:
                hint = self.hint()
            except Exception:
                hint = None
        segments, _ = self._model.transcribe(
            np.asarray(audio, dtype=np.float32), language=self.language,
            beam_size=1, condition_on_previous_text=False,
            initial_prompt=hint or None)
        return " ".join(s.text.strip() for s in segments).strip()

    def say(self, text: str):
        """Speak, with the microphone muted until the room is quiet again."""
        if not self.speak_enabled or not text or self.speaker is None:
            return

        def run():
            self._muted = True
            try:
                self.speaker.say(text)
            except Exception as e:
                with self._lock:
                    self._error = f"could not speak: {type(e).__name__}: {e}"
            finally:
                # spd-say -w returns when the utterance ends; the tail covers
                # the room's reverberation and the endpointer's own lag.
                time.sleep(self.mute_tail_s)
                self._endpointer.reset()
                self._muted = False

        threading.Thread(target=run, daemon=True).start()

    def _handle(self, utterance: np.ndarray):
        try:
            text = self.transcribe(utterance)
        except Exception as e:
            with self._lock:
                self._error = f"transcription failed: {type(e).__name__}: {e}"
            return

        # Cancel wins over the terminator: "...nevermind, done" is a change of
        # mind, not a task ending in the word nevermind.
        cancelled, _ = split_phrase(text, self.cancel)
        ended, remaining = split_terminator(text, self.terminator)

        with self._lock:
            self._utterances += 1
            self._heard = text
            if cancelled:
                had = bool(self._buffer)
                self._buffer = ""
            elif remaining and not is_filler(remaining):
                self._buffer = f"{self._buffer} {remaining}".strip()
            prompt = self._buffer if (ended and not cancelled) else ""
            if ended and not cancelled:
                self._buffer = ""
                if prompt:
                    self._last_prompt = prompt

        # Callbacks and speech outside the lock: they touch the policy, the
        # history file and a subprocess.
        if cancelled:
            self._cancelled(had)
        elif ended and prompt:
            self.on_prompt(prompt)
            self.say(prompt)

    def _cancelled(self, had_buffer: bool):
        """Change of mind: drop what was part-said, else undo the last task."""
        if had_buffer:
            self.say("cancelled")
            return
        restored = self.on_cancel() if self.on_cancel is not None else None
        if restored:
            with self._lock:
                self._last_prompt = restored
            self.say(f"back to {restored}")
        else:
            self.say("nothing to go back to")
