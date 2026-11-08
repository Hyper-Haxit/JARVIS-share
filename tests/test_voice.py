"""Phase 5: voice. Uses the real local models (Whisper, openWakeWord, the OS
voices); audio input is synthesized speech fed into the pipeline in place of a
microphone, and playback runs in dry-run mode so tests are silent."""

from __future__ import annotations

import asyncio
import io
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from jarvis.voice.tts import speakable, split_sentences
from jarvis.voice.vad import EnergyVAD
from tests.test_intelligence import make_runtime

voice_deps = pytest.importorskip("faster_whisper")
pytestmark = pytest.mark.voice


def tts_audio(text: str, voice: str = "Zira") -> np.ndarray:
    from jarvis.voice.audio import resample
    from jarvis.voice.tts import make_tts

    s = make_tts("pyttsx3", voice, 170).synthesize(text)
    return resample(s.pcm, s.sample_rate)


@pytest.fixture(scope="module")
def stt():
    from jarvis.voice.stt import FasterWhisperSTT

    return FasterWhisperSTT("base.en")


class PushSource:
    """Test microphone: frames are pushed by the test."""

    def __init__(self) -> None:
        self.cb = None

    def start(self, on_frame) -> None:
        self.cb = on_frame

    def stop(self) -> None:
        self.cb = None

    def push(self, audio: np.ndarray, realtime: float = 0.0) -> None:
        for i in range(0, len(audio) - 479, 480):
            if self.cb is None:
                return
            self.cb(audio[i:i + 480].astype(np.float32))
            if realtime:
                time.sleep(0.03 * realtime)


# --------------------------------------------------------------- pure helpers
def test_vad_detects_speech_and_silence() -> None:
    rng = np.random.default_rng(0)
    noise = lambda s: rng.normal(0, 0.002, int(16000 * s)).astype(np.float32)  # noqa: E731
    tone = (0.2 * np.sin(np.linspace(0, 2 * np.pi * 220 * 1.0, 16000))).astype(np.float32)
    audio = np.concatenate([noise(1.0), tone, noise(1.5)])
    vad = EnergyVAD(silence_ms=600)
    events = []
    for i in range(0, len(audio) - 479, 480):
        ev = vad.process(audio[i:i + 480])
        if ev:
            events.append((ev.kind, round(i / 16000, 2)))
    assert [k for k, _ in events] == ["start", "end"]
    assert 0.95 <= events[0][1] <= 1.25 and 2.4 <= events[1][1] <= 2.8


def test_vad_adapts_to_a_steady_loud_background() -> None:
    """A fan or a busy room louder than the starting floor estimate must not count as endless speech.

    (Measured on this machine: a steady -29 dBFS background made every request record until the 20 s limit.)
    """
    rng = np.random.default_rng(1)
    room = rng.normal(0, 0.05, 16000 * 8).astype(np.float32)  # about -26 dBFS, steady
    t = np.arange(int(16000 * 1.5)) / 16000
    # Someone speaking to the microphone: clearly (~18 dB) above the room.
    voice = (0.55 * np.sin(2 * np.pi * 180 * t) * (0.8 + 0.2 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)
    audio = room.copy()
    audio[16000 * 5:16000 * 5 + len(voice)] += voice
    vad = EnergyVAD(silence_ms=800)
    events = []
    for i in range(0, len(audio) - 479, 480):
        ev = vad.process(audio[i:i + 480])
        if ev:
            events.append((ev.kind, round(i / 16000, 2)))
    starts = [t for k, t in events if k == "start" and t >= 4.5]
    ends = [t for k, t in events if k == "end" and t >= 5.0]
    assert starts and 4.9 <= starts[0] <= 5.4, events
    assert ends and ends[0] <= 7.6, f"speech ends shortly after the voice stops, not at a time limit: {events}"
    assert -32 < vad.floor < -20


def test_wake_prefix_and_sentence_helpers() -> None:
    from jarvis.voice.pipeline import strip_wake

    assert strip_wake("Hey Jarvis, what time is it?") == "what time is it?"
    assert strip_wake("Jarvis turn on the lights") == "turn on the lights"
    assert strip_wake("tell Jarvis a joke") == "tell Jarvis a joke"
    sents, rest = split_sentences("First one. Second one! Third")
    assert sents == ["First one.", "Second one!"] and rest == "Third"
    assert speakable("**Bold** and `code` see https://x.y/z") == "Bold and code see a link"


# ----------------------------------------------------------------- models
def test_tts_to_stt_round_trip(stt) -> None:
    t = stt.transcribe(tts_audio("Please remember that my car is parked on level three."))
    assert "level" in t.text.lower() and ("three" in t.text.lower() or "3" in t.text)


def test_wake_word_separates_wake_from_other_speech() -> None:
    from jarvis.voice.wakeword import OpenWakeWordDetector

    det = OpenWakeWordDetector("hey_jarvis", 0.5)

    def best(audio: np.ndarray) -> float:
        det.reset()
        padded = np.concatenate([np.zeros(16000, np.float32), audio, np.zeros(16000, np.float32)])
        return max(det.process(padded[i:i + 480]) for i in range(0, len(padded) - 479, 480))

    assert best(tts_audio("Hey Jarvis")) > 0.9
    assert best(tts_audio("Hello there, how are you doing today")) < 0.1


# ---------------------------------------------------------------- pipeline
async def _wait(cond, timeout: float = 60.0) -> None:
    end = time.time() + timeout
    while not cond():
        if time.time() > end:
            raise AssertionError("timed out waiting for condition")
        await asyncio.sleep(0.05)


@pytest.fixture
async def voice_env(tmp_path: Path):
    from jarvis.voice.service import VoiceService

    rt, provider = make_runtime(tmp_path)
    source = PushSource()
    rt.extensions["voice"] = svc = VoiceService(rt, source_factory=lambda: source, dry_run_audio=True)
    spoken: list[np.ndarray] = []
    original = svc.player.enqueue
    svc.player.enqueue = lambda pcm, sr: (spoken.append((pcm, sr)), original(pcm, sr))[-1]  # type: ignore[method-assign]
    await rt.start()
    events: list[Any] = []
    rt.events.on("*", lambda e: events.append(e))
    yield rt, provider, svc, source, spoken, events
    await rt.stop()


async def test_voice_to_tool_to_voice_end_to_end(voice_env, stt) -> None:
    """voice -> wake word -> speech recognition -> agent -> tool -> reply -> speech"""
    rt, provider, svc, source, spoken, events = voice_env

    def respond(req):
        user = next(m.content for m in reversed(req.messages) if m.role == "user")
        if req.messages[-1].role == "tool":
            return {"text": "Twelve times twelve is 144. Anything else?"}
        assert "12" in user or "twelve" in user.lower(), user
        return {"tool_calls": [{"name": "calculate", "arguments": {"expression": "12*12"}}]}

    provider.push(respond, respond)
    await svc.start_pipeline()
    assert svc.pipeline.state == "standby"
    command = tts_audio("Hey Jarvis, what is twelve times twelve?")
    await asyncio.to_thread(source.push, np.concatenate([command, np.zeros(16000 * 2, np.float32)]))
    await _wait(lambda: any(e.type == "SPEECH_ENDED" for e in events), 90)
    types = [e.type for e in events]
    assert "WAKE_WORD_DETECTED" in types
    transcript = next(e for e in events if e.type == "TRANSCRIPT" and e.data.get("text"))
    assert "12" in transcript.data["text"] or "twelve" in transcript.data["text"].lower()
    tool_done = [e for e in events if e.type == "TOOL_COMPLETED"]
    assert tool_done and tool_done[0].data["tool"] == "calculate" and tool_done[0].data["ok"]
    assert types.index("SPEECH_STARTED") < types.index("SPEECH_ENDED")
    assert any(e.type == "AUDIO_LEVEL" and e.data["source"] == "speaker" and e.data["level"] > 0 for e in events)
    # Close the loop: what JARVIS said out loud really says the answer.
    said = np.concatenate([p for p, _ in spoken])
    heard = stt.transcribe(said).text.lower()
    assert "144" in heard or "hundred forty" in heard.replace("-", " "), heard
    await _wait(lambda: svc.pipeline.state in ("followup", "standby"), 20)
    states = [e.data["state"] for e in events if e.type == "MIC_STATE"]
    assert states[:4] == ["standby", "recording", "processing", "speaking"], states


async def test_barge_in_stops_speech(voice_env) -> None:
    """Headset mode: talking over JARVIS interrupts it (with speakers only the wake word does;
    see test_voice_loop.py)."""
    rt, provider, svc, source, spoken, events = voice_env
    long_reply = " ".join(f"This is sentence number {i} of a long answer." for i in range(1, 30))
    provider.push({"text": long_reply}, {"text": "Okay, stopping."})
    svc.config.barge_in_mode = "voice"
    await svc.start_pipeline()
    await asyncio.to_thread(source.push, np.concatenate([tts_audio("Hey Jarvis, tell me something long."), np.zeros(16000, np.float32)]))
    await _wait(lambda: svc.pipeline.state == "speaking", 60)
    # Simulate the user talking loudly over JARVIS.
    loud = (0.3 * np.sin(np.linspace(0, 2 * np.pi * 180 * 0.6, int(16000 * 0.6)))).astype(np.float32)
    await asyncio.to_thread(source.push, loud)
    await _wait(lambda: any(e.type == "SPEECH_INTERRUPTED" for e in events), 10)
    assert svc.pipeline.state in ("recording", "processing")
    assert svc.pipeline.stats["interruptions"] == 1


async def test_voice_api(tmp_path: Path, voice_env) -> None:
    from fastapi.testclient import TestClient

    from jarvis.api.server import create_app

    rt, provider, svc, source, spoken, events = voice_env
    app = create_app(rt, manage_lifecycle=False)
    auth = {"Authorization": f"Bearer {rt.local_token}"}
    with TestClient(app) as client:
        st = client.get("/api/v1/voice/status", headers=auth).json()
        assert st["stt"]["available"] and st["tts"]["available"] and st["wake"]["available"]
        assert client.get("/api/v1/voice/status").status_code == 401
        wav = client.post("/api/v1/voice/speak", json={"text": "The kettle has boiled."}, headers=auth)
        assert wav.status_code == 200 and wav.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(wav.content)) as w:
            assert w.getframerate() > 0 and w.getnframes() > 8000
        out = client.post("/api/v1/voice/transcribe", content=wav.content, headers={**auth, "Content-Type": "audio/wav"}).json()
        assert "kettle" in out["text"].lower()
        bad = client.post("/api/v1/voice/transcribe", content=b"not audio at all", headers={**auth, "Content-Type": "audio/webm"})
        assert bad.status_code == 422
