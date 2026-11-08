"""Regression tests for the repeated "I'm listening" voice loop.

What happened (2026-10-03, from the core's event history and log): one wake word;
speech recognition heard "."; the local model answered "I'm listening. What do
you need?". About 0.3 s after JARVIS started saying it, energy-based barge-in
fired: sound at the microphone (JARVIS's own voice, or the room) counted as the
user interrupting. Each barge-in opened a recording without a wake word, which
ran to the 20 s limit on room audio, became the next request, and that reply was
interrupted the same way, so JARVIS said "I'm listening" every ~21 s.

These tests drive the real pipeline, service and agent in a simulated room:
everything JARVIS plays reaches the microphone at full volume in real time,
followed by a reverberation tail after playback stops. Anything the pipeline
records that it shouldn't is transcribed as room chatter, so a loop would show
up as extra agent requests and extra speech.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from jarvis.voice.audio import Player
from jarvis.voice.pipeline import judge_transcript
from jarvis.voice.stt import Transcript
from jarvis.voice.tts import Speech
from tests.test_intelligence import make_runtime

FRAME = 480
RATE = 16000
REPLY = "I'm listening. What do you need?"


# ------------------------------------------------------------------- audio
def tone(seconds: float, amp: float = 0.3, hz: float = 220.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def quiet(seconds: float) -> np.ndarray:
    return np.random.default_rng(1).normal(0, 2e-4, int(RATE * seconds)).astype(np.float32)


WAKE_HZ = 5000  # the test detector's "hey jarvis": a 60 ms burst at 5 kHz, audible through any mix


def mark_wake(audio: np.ndarray, at: int) -> None:
    """Make the 60 ms starting at sample ``at`` sound like the wake phrase to the test detector."""
    n = min(len(audio) - at, int(RATE * 0.06))
    audio[at:at + n] += tone(n / RATE, 0.2, WAKE_HZ)[:n]


def wake_word() -> np.ndarray:
    a = np.concatenate([quiet(0.3), quiet(0.06), quiet(0.3)])
    mark_wake(a, int(RATE * 0.3))
    return a


def request(seconds: float = 1.0) -> np.ndarray:
    """A spoken request followed by enough quiet for the end of speech to be detected."""
    return np.concatenate([tone(seconds, 0.25, 160), quiet(1.2)])


# ----------------------------------------------------------------- the room
class Room:
    """A room with microphones, in real time: every 30 ms the microphones hear the
    sum of everything sounding at that moment (the person, JARVIS's speaker, its
    reverberation) on top of a quiet noise floor."""

    def __init__(self) -> None:
        self.mics: list[RoomMic] = []
        self.lock = threading.Lock()
        self._sounds: list[tuple[deque[np.ndarray], threading.Event]] = []
        self._rng = np.random.default_rng(7)
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._clock, name="test-room", daemon=True)
        self._thread.start()

    def mic(self) -> RoomMic:
        return RoomMic(self)

    @property
    def open_mics(self) -> list[RoomMic]:
        with self.lock:
            return [m for m in self.mics if m.cb is not None]

    def play(self, audio: np.ndarray) -> tuple[deque[np.ndarray], threading.Event]:
        """Start a sound; returns a handle whose event is set when it has finished."""
        pad = (-len(audio)) % FRAME
        audio = np.concatenate([audio.astype(np.float32), np.zeros(pad, np.float32)])
        sound = (deque(audio[i:i + FRAME] for i in range(0, len(audio), FRAME)), threading.Event())
        with self.lock:
            self._sounds.append(sound)
        return sound

    def silence(self, sound: tuple[deque[np.ndarray], threading.Event]) -> None:
        with self.lock:
            sound[0].clear()

    def say(self, audio: np.ndarray) -> None:
        self.play(audio)[1].wait()

    def _clock(self) -> None:
        tick = time.monotonic()
        while not self._closed.is_set():
            with self.lock:
                mix = self._rng.normal(0, 2e-4, FRAME).astype(np.float32)
                for frames, done in self._sounds:
                    if frames:
                        mix += frames.popleft()
                    if not frames:
                        done.set()
                self._sounds = [s for s in self._sounds if s[0]]
                mics = [m.cb for m in self.mics if m.cb is not None]
            for cb in mics:
                cb(mix.copy())
            tick += FRAME / RATE
            time.sleep(max(0.0, tick - time.monotonic()))

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=2)


class RoomMic:
    def __init__(self, room: Room) -> None:
        self.room = room
        self.cb: Any = None

    def start(self, on_frame: Any) -> None:
        self.cb = on_frame
        with self.room.lock:
            self.room.mics.append(self)

    def stop(self) -> None:
        self.cb = None


class RoomPlayer(Player):
    """JARVIS's speaker in the room: playback is heard by the microphone, then a reverberation tail."""

    def __init__(self, room: Room, gain: float = 1.0, tail_s: float = 0.3) -> None:
        super().__init__(dry_run=True)
        self.room, self.gain, self.tail_s = room, gain, tail_s
        self.items = 0

    def _play(self, pcm: np.ndarray, rate: int) -> None:
        self.items += 1
        if self.on_level:
            self.on_level(0.5)
        sound = self.room.play(pcm * self.gain)
        while not sound[1].wait(0.01):
            if self._stop.is_set():
                self.room.silence(sound)
                return
        if self.gain and self.tail_s:
            self.room.play(pcm[-int(RATE * self.tail_s):] * (self.gain * 0.5))


# ------------------------------------------------------------------ engines
class FakeWake:
    """Fires on audio marked by mark_wake (on every frame of it, like a real model's
    scores stay high across the phrase), whatever else is sounding at the time."""

    name = "test-wake"
    threshold = 0.5

    def __init__(self) -> None:
        self.resets = 0

    @staticmethod
    def dependency_ok() -> tuple[bool, str]:
        return True, ""

    def load(self) -> None:
        pass

    def reset(self) -> None:
        self.resets += 1

    def process(self, frame: np.ndarray) -> float:
        self.frames = getattr(self, "frames", 0) + 1
        bin_ = int(round(WAKE_HZ * len(frame) / RATE))
        return 0.99 if abs(np.fft.rfft(frame)[bin_]) > 5.0 else 0.0

    def detected(self, score: float) -> bool:
        return score >= self.threshold


class FakeSTT:
    """Returns scripted transcripts; anything unscripted it hears is room chatter."""

    name = "test-stt"
    model_name = "test"
    loaded = True
    load_error = None

    def __init__(self) -> None:
        self.replies: list[str] = []
        self.calls = 0
        self.delay = 0.0

    @staticmethod
    def dependency_ok() -> tuple[bool, str]:
        return True, ""

    def _ensure(self) -> FakeSTT:
        return self

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = "en") -> Transcript:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        text = self.replies.pop(0) if self.replies else "oh it can respond, that's the right approach"
        return Transcript(text, "en", round(len(audio) / RATE, 2), 1.0, 0.05)


class FakeTTS:
    name = "test-tts"
    voice = "test"

    def __init__(self) -> None:
        self.said: list[str] = []

    @staticmethod
    def dependency_ok() -> tuple[bool, str]:
        return True, ""

    def start(self) -> None:
        pass

    def synthesize(self, text: str, timeout: float = 60) -> Speech:
        self.said.append(text)
        pcm = tone(min(0.9, 0.3 + 0.015 * len(text)), 0.3, 180)
        if "jarvis" in text.lower():
            mark_wake(pcm, FRAME * 3)  # the wake model would fire on JARVIS saying its own name
        return Speech(pcm, RATE, text, 1.0)


# ------------------------------------------------------------------ harness
@pytest.fixture
async def room_env(tmp_path: Path):
    from jarvis.voice.service import VoiceService

    rt, provider = make_runtime(tmp_path)
    room = Room()
    svc = VoiceService(rt, source_factory=room.mic, dry_run_audio=True)
    svc.stt, svc.tts, svc.wake = FakeSTT(), FakeTTS(), FakeWake()
    svc.player = RoomPlayer(room)
    rt.extensions["voice"] = svc
    runs: list[str] = []
    original = rt.agent.run

    async def counted(**kw: Any) -> Any:
        runs.append(kw["text"])
        return await original(**kw)

    rt.agent.run = counted  # type: ignore[method-assign]
    await rt.start()
    events: list[Any] = []
    rt.events.on("*", events.append)
    # Plenty of replies: if the pipeline loops, every extra request gets answered and spoken.
    provider.push(*[{"text": REPLY} for _ in range(8)])
    yield SimpleNamespace(rt=rt, provider=provider, svc=svc, room=room, events=events, runs=runs)
    await svc.stop_pipeline()
    await rt.stop()
    room.close()


async def wait_for(cond: Any, timeout: float = 15.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.02)


def of(events: list[Any], type_: str) -> list[Any]:
    return [e for e in events if e.type == type_]


def states(events: list[Any]) -> list[str]:
    return [e.data["state"] for e in of(events, "MIC_STATE")]


def voice_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "jarvis-voice" and t.is_alive()]


async def say_to_jarvis(env: SimpleNamespace, *parts: np.ndarray) -> None:
    """The person speaks (in real time)."""
    await asyncio.to_thread(env.room.say, np.concatenate(parts))


async def settle(env: SimpleNamespace, seconds: float = 2.5) -> None:
    """Let the room keep running with nobody talking: a loop would show up here."""
    await asyncio.sleep(seconds)


async def one_exchange(env: SimpleNamespace, text: str = "what's the weather like") -> None:
    env.svc.stt.replies.append(text)
    await say_to_jarvis(env, wake_word(), request())
    await wait_for(lambda: of(env.events, "SPEECH_ENDED"))
    await wait_for(lambda: env.svc.pipeline.state == "standby")


# -------------------------------------------------------------------- tests
def test_transcripts_without_words_are_not_requests() -> None:
    assert judge_transcript(".", 0.2) == "nothing intelligible"
    assert judge_transcript("...", None) == "nothing intelligible"
    assert judge_transcript("Um, uh.", 0.1) == "only filler sounds"
    assert judge_transcript("Thanks for watching!", 0.1) is not None
    assert judge_transcript("you", 0.6) is not None
    assert judge_transcript("Thank you.", 0.1) is None  # a real thing to say to JARVIS
    assert judge_transcript("What time is it?", 0.1) is None
    assert judge_transcript("Oh, oh, it can respond yo", 0.1) is None


async def test_one_wake_word_is_exactly_one_interaction(room_env) -> None:
    env = room_env
    await env.svc.start_pipeline()
    await one_exchange(env, "what time is it")
    await settle(env)
    ev = env.events
    assert len(of(ev, "WAKE_WORD_DETECTED")) == 1
    assert states(ev).count("recording") == 1, states(ev)
    assert states(ev) == ["standby", "recording", "processing", "speaking", "standby"]
    assert env.runs == ["what time is it"], "one utterance must be exactly one agent request"
    assert len(env.provider.requests) == 1
    assert len(of(ev, "SPEECH_STARTED")) == 1 and len(of(ev, "SPEECH_ENDED")) == 1
    assert env.svc.tts.said == ["I'm listening.", "What do you need?"]  # the reply, spoken once
    assert env.svc.player.items == 2 and not of(ev, "SPEECH_INTERRUPTED")
    # Every event of the exchange carries the same interaction id.
    ids = {e.data.get("interaction") for e in ev if e.source == "voice" and e.type != "AUDIO_LEVEL"} - {None}
    assert len(ids) == 1 and next(iter(ids)).startswith("vi_")
    assert env.svc.pipeline.interaction is None


async def test_jarvis_cannot_hear_itself(room_env) -> None:
    """Its voice reaches the mic at full volume, plus reverb after playback: still one interaction."""
    env = room_env
    env.provider.script[:] = [{"text": "Sure. Here is a longer answer with several sentences. Each one is spoken out loud. "
                                       "The microphone hears all of it."}]
    await env.svc.start_pipeline()
    await one_exchange(env)
    await settle(env, 3.0)
    assert len(env.runs) == 1 and env.svc.stt.calls == 1
    assert not of(env.events, "SPEECH_INTERRUPTED"), "JARVIS's own voice must not count as an interruption"
    assert states(env.events).count("recording") == 1
    assert env.svc.player.items == 4  # every sentence was played to the end


async def test_jarvis_saying_its_own_name_does_not_wake_it(room_env) -> None:
    env = room_env
    env.provider.script[:] = [{"text": "I'm Jarvis, your assistant. Jarvis is listening."}]
    await env.svc.start_pipeline()
    await one_exchange(env, "who are you")
    await settle(env)
    assert len(of(env.events, "WAKE_WORD_DETECTED")) == 1
    assert not of(env.events, "SPEECH_INTERRUPTED")
    assert len(env.runs) == 1


async def test_wake_word_barge_in_still_interrupts(room_env) -> None:
    env = room_env
    env.provider.script[:] = [{"text": " ".join(f"This is sentence number {i} of a long answer." for i in range(1, 12))},
                              {"text": "Okay, stopping."}]
    await env.svc.start_pipeline()
    env.svc.stt.replies += ["tell me something long", "stop and tell me the time"]
    await say_to_jarvis(env, wake_word(), request())
    await wait_for(lambda: env.svc.pipeline.state == "speaking")
    first = env.svc.pipeline.interaction
    await asyncio.sleep(0.5)
    await say_to_jarvis(env, wake_word(), request())  # the person says "hey jarvis" over the reply
    await wait_for(lambda: len(env.runs) == 2)
    await wait_for(lambda: env.svc.pipeline.state == "standby", 20)
    await settle(env)
    interrupted = of(env.events, "SPEECH_INTERRUPTED")
    assert len(interrupted) == 1 and interrupted[0].data["interaction"] == first
    assert env.runs == ["tell me something long", "stop and tell me the time"]
    assert len(of(env.events, "WAKE_WORD_DETECTED")) == 2
    assert env.svc.player.items < 11 + 1  # the rest of the long answer was never played
    assert env.svc.tts.said[-1] == "Okay, stopping."


async def test_loud_speech_interrupts_only_in_headset_mode(room_env) -> None:
    env = room_env
    env.svc.config.barge_in_mode = "voice"
    env.svc.player.gain = 0.0  # headset: JARVIS's voice doesn't reach the microphone
    env.provider.script[:] = [{"text": " ".join(f"Sentence {i} of a long answer." for i in range(1, 12))},
                              {"text": "Okay."}]
    await env.svc.start_pipeline()
    env.svc.stt.replies += ["tell me something long", "never mind"]
    await say_to_jarvis(env, wake_word(), request())
    await wait_for(lambda: env.svc.pipeline.state == "speaking")
    await say_to_jarvis(env, tone(0.6, 0.3), quiet(1.2))  # the person talks over JARVIS
    await wait_for(lambda: of(env.events, "SPEECH_INTERRUPTED"))
    await wait_for(lambda: len(env.runs) == 2)
    assert env.svc.pipeline.stats["interruptions"] == 1


async def test_starting_voice_twice_keeps_one_listener(room_env) -> None:
    env = room_env
    await asyncio.gather(env.svc.start_pipeline(), env.svc.start_pipeline(), env.svc.start_pipeline())
    await env.svc.start_pipeline()
    assert len(voice_threads()) == 1 and len(env.room.open_mics) == 1
    await one_exchange(env)
    await settle(env, 1.0)
    assert len(env.runs) == 1 and len(of(env.events, "WAKE_WORD_DETECTED")) == 1


async def test_stop_and_restart_leave_no_old_listener(room_env) -> None:
    env = room_env
    for _ in range(3):
        await env.svc.start_pipeline()
        await env.svc.stop_pipeline()
    assert voice_threads() == [] and env.room.open_mics == []
    await env.svc.start_pipeline()
    assert len(voice_threads()) == 1 and len(env.room.open_mics) == 1
    await one_exchange(env)
    await settle(env, 1.0)
    assert len(env.runs) == 1 and len(of(env.events, "WAKE_WORD_DETECTED")) == 1


async def test_audio_queued_while_busy_cannot_start_an_interaction(room_env) -> None:
    env = room_env
    env.svc.stt.delay = 4.0
    await env.svc.start_pipeline()
    env.svc.stt.replies.append("what time is it")
    await say_to_jarvis(env, wake_word(), request())
    await wait_for(lambda: env.svc.pipeline.state == "processing")
    # While speech recognition is busy, the room hears the wake word and more talk.
    await say_to_jarvis(env, wake_word(), request(0.6))
    await wait_for(lambda: of(env.events, "SPEECH_ENDED"))
    await wait_for(lambda: env.svc.pipeline.state == "standby")
    await settle(env)
    assert env.runs == ["what time is it"] and env.svc.stt.calls == 1
    assert len(of(env.events, "WAKE_WORD_DETECTED")) == 1


async def test_stopping_voice_discards_a_pending_result(room_env) -> None:
    env = room_env
    env.svc.stt.delay = 1.5
    await env.svc.start_pipeline()
    env.svc.stt.replies.append("what time is it")
    await say_to_jarvis(env, wake_word(), request())
    await wait_for(lambda: env.svc.pipeline.state == "processing")
    await env.svc.stop_pipeline()  # stopped while the request is being recognised
    await asyncio.sleep(2.0)
    assert env.runs == [] and env.svc.tts.said == [] and env.svc.player.items == 0


async def test_unintelligible_transcript_is_not_answered(room_env) -> None:
    env = room_env
    await env.svc.start_pipeline()
    env.svc.stt.replies.append(".")  # what speech recognition heard in the real incident
    await say_to_jarvis(env, wake_word(), request())
    await wait_for(lambda: of(env.events, "TRANSCRIPT"))
    await wait_for(lambda: env.svc.pipeline.state == "standby")
    await settle(env, 1.0)
    t = of(env.events, "TRANSCRIPT")
    assert len(t) == 1 and t[0].data["ignored"] == "nothing intelligible"
    assert env.runs == [] and env.svc.tts.said == [] and not of(env.events, "SPEECH_STARTED")


async def test_stop_button_returns_to_waiting(room_env) -> None:
    env = room_env
    env.provider.script[:] = [{"text": " ".join(f"Sentence {i} of a long answer." for i in range(1, 12))}]
    await env.svc.start_pipeline()
    env.svc.stt.replies.append("tell me something long")
    await say_to_jarvis(env, wake_word(), request())
    await wait_for(lambda: env.svc.pipeline.state == "speaking")
    await asyncio.to_thread(env.svc.pipeline.interrupt, False, "stop button")
    await wait_for(lambda: env.svc.pipeline.state == "standby", 5)
    await settle(env)
    assert env.svc.pipeline.state == "standby" and len(env.runs) == 1
    assert len(of(env.events, "SPEECH_INTERRUPTED")) == 1


async def test_announcements_are_echo_protected(room_env) -> None:
    env = room_env
    await env.svc.start_pipeline()
    spoken = await env.svc.pipeline.announce("Hello, this is Jarvis. The kettle has boiled.")
    assert spoken > 0
    await settle(env)
    assert not of(env.events, "WAKE_WORD_DETECTED") and env.runs == []
    assert env.svc.pipeline.state == "standby"


async def test_follow_up_window_is_opt_in(room_env) -> None:
    env = room_env
    assert env.svc.config.conversation_window_s == 0  # default: one wake word, one request
    env.svc.config.conversation_window_s = 4.0
    await env.svc.start_pipeline()
    await one_exchange_followup(env)


async def one_exchange_followup(env: SimpleNamespace) -> None:
    env.svc.stt.replies += ["what's the weather", "and tomorrow"]
    await say_to_jarvis(env, wake_word(), request())
    await wait_for(lambda: env.svc.pipeline.state == "followup")
    # The reverb tail of JARVIS's reply is suppressed; a real answer is accepted without the wake word.
    await asyncio.sleep(0.8)
    await say_to_jarvis(env, request())
    await wait_for(lambda: len(env.runs) == 2)
    assert env.runs == ["what's the weather", "and tomorrow"]
    assert len(of(env.events, "WAKE_WORD_DETECTED")) == 1
