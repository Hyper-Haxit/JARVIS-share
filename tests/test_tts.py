"""JARVIS's voice: Kokoro as the primary voice, the system voice (SAPI) as fallback.

Deterministic tests use fake engines and a fake sound device. Tests marked
``kokoro`` use the real model and run only when it is installed
(``jarvis voice install``); they never download anything.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from jarvis.config.paths import JarvisPaths
from jarvis.config.settings import VoiceConfig
from jarvis.voice import kokoro as kokoro_mod
from jarvis.voice.audio import Player
from jarvis.voice.kokoro import KokoroTTS, missing_assets
from jarvis.voice.tts import (
    Speech,
    SpeechChunker,
    TTSCancelled,
    TTSUnavailable,
    VoiceChain,
    make_voice,
    prepare_text,
    split_sentences,
)

RATE = 16000
REAL_MODEL_DIR = JarvisPaths().models / "tts" / "kokoro"
HAVE_KOKORO = KokoroTTS.dependency_ok()[0] and not missing_assets(REAL_MODEL_DIR)
needs_kokoro = pytest.mark.skipif(not HAVE_KOKORO, reason=f"Kokoro model not installed in {REAL_MODEL_DIR} (jarvis voice install)")


# ---------------------------------------------------------------- fakes
class FakeEngine:
    """A voice engine that records what it was asked to say and can be told to fail."""

    def __init__(self, name: str, fail: int = 0, error: Exception | None = None, cancellable: bool = False,
                 block_until_stopped: bool = False) -> None:
        self.name = name
        self.voice = f"{name}-voice"
        self.fail = fail  # how many of the next calls fail
        self.error = None
        self._error = error or RuntimeError(f"{name} broke")
        self.cancellable = cancellable
        self.block_until_stopped = block_until_stopped
        self.said: list[str] = []
        self.started = 0
        self.cancelled = threading.Event()

    @staticmethod
    def dependency_ok() -> tuple[bool, str]:
        return True, ""

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        pass

    def cancel(self) -> None:
        self.cancelled.set()

    def synthesize(self, text: str, timeout: float = 60, should_stop: Any = None) -> Speech:
        if self.block_until_stopped:
            end = time.monotonic() + 10
            while time.monotonic() < end:
                if self.cancelled.is_set() or (should_stop is not None and should_stop()):
                    raise TTSCancelled("stopped")
                time.sleep(0.01)
        if self.fail:
            self.fail -= 1
            raise self._error
        self.said.append(text)
        return Speech(np.full(int(RATE * 0.2), 0.1, np.float32), RATE, text, 1.0, provider=self.name)


# ------------------------------------------------------------ text and chunks
def test_split_sentences_does_not_split_at_abbreviations() -> None:
    sents, rest = split_sentences("Ask Dr. Smith, e.g. on Oct. 6 at 9 a.m. tomorrow. It costs 2.5 dollars vs. 3. Done")
    assert sents == ["Ask Dr. Smith, e.g. on Oct. 6 at 9 a.m. tomorrow.", "It costs 2.5 dollars vs. 3."]
    assert rest == "Done"
    assert split_sentences("Version 2.") == ([], "Version 2."), "a number at the end may still grow (2.5)"


def stream(text: str, step: int = 2) -> list[str]:
    c = SpeechChunker()
    out = [ch for i in range(0, len(text), step) for ch in c.feed(text[i:i + step])]
    return out + c.flush()


def test_chunker_speaks_the_first_clause_early_without_fragments() -> None:
    assert stream("You have 3 meetings tomorrow, and the first one starts at 9:30. The update is done.") == [
        "You have 3 meetings tomorrow,", "and the first one starts at 9:30.", "The update is done."]
    # Short openings are whole sentences, never cut into fragments
    assert stream("Okay, stopping.") == ["Okay, stopping."]
    assert stream("I'm Jarvis, your assistant. Jarvis is listening.") == ["I'm Jarvis, your assistant.", "Jarvis is listening."]
    assert stream("I'm listening. What do you need?") == ["I'm listening.", "What do you need?"]


def test_chunker_keeps_a_short_opening_from_leaving_a_long_silence() -> None:
    chunks = stream("Sure. The write-ahead log records every change before it reaches the database file, which means "
                    "a crash can't corrupt the database, and readers keep reading while a writer works, so it's usually right.")
    assert chunks[0] == "Sure."
    assert chunks[1] == "The write-ahead log records every change before it reaches the database file,"
    assert all(len(c.split()) >= 4 for c in chunks[1:])
    assert " ".join(chunks).replace("  ", " ").startswith("Sure. The write-ahead")


def test_chunker_keeps_numbered_list_items_whole() -> None:
    reply = ("1. Set a clear goal for each session.  \n2. Work in 25-minute intervals.  \n3. Remove distractions.")
    assert stream(reply) == ["1. Set a clear goal for each session.", "2. Work in 25-minute intervals.", "3. Remove distractions."]
    assert stream("The answer is 42. Next question.") == ["The answer is 42.", "Next question."], "a number ending a sentence still ends it"


def test_chunker_cuts_a_long_opening_before_a_clause_word() -> None:
    """Measured on the real path: a 29-word opening with no early comma held first audio for 3.5 s."""
    chunks = stream("A pytest step would fail if a PostgreSQL container isn't ready yet because the test might "
                    "attempt to connect to the database before it's fully up and running, leading to a connection error.")
    assert chunks[0] == "A pytest step would fail if a PostgreSQL container isn't ready yet,"
    assert chunks[1].startswith("because the test might attempt")
    # Short heads are never cut off, and later sentences are left whole
    assert stream("If you want, I can help with that.") == ["If you want, I can help with that."]
    later = stream("Sure, it's done now. The service restarted cleanly because the config was valid and nothing else changed.")
    assert later == ["Sure, it's done now.", "The service restarted cleanly because the config was valid and nothing else changed."]


def test_chunker_feeds_whole_text_and_later_sentences_unsplit() -> None:
    c = SpeechChunker()
    text = "First sentence here. Second one, with a clause, stays whole. Third."
    assert c.feed(text) + c.flush() == ["First sentence here.", "Second one, with a clause, stays whole.", "Third."]


CHUNKER_CORPUS = [
    "You have 3 meetings tomorrow, and the first one starts at 9:30. The 2.4 GB update is 73% done.",
    "Okay, stopping.",
    "I'm listening. What do you need?",
    "Sure. The write-ahead log records every change before it reaches the database file, which means a crash can't corrupt "
    "the database, and readers keep reading while a writer works, so it's usually the right choice.",
    "A pytest step would fail if a PostgreSQL container isn't ready yet because the test might attempt to connect to the "
    "database before it's fully up and running, leading to a connection error.",
    "1. Set a clear goal for each session and keep it specific and measurable.  \n2. Work in 25-minute intervals with a "
    "5-minute break.  \n3. Remove distractions.",
    "Use the Pomodoro Technique: Work for 25 minutes, then take a 5-minute break.  Eliminate distractions: Turn off notifications.",
    "Here's the plan: first, I'll back up the database; then, only if that works, I'll run the migration. Sound good?",
    "Ask Dr. Smith, e.g. on Oct. 6 at 9 a.m. tomorrow. It costs 2.5 dollars vs. 3. Done",
    "The answer is 42. Next question.",
    "Backups matter because they protect against data loss from hardware failure, accidental deletion, or cyberattacks. "
    "Without backups, losing files means losing everything — documents, photos, projects.",
    "",
    "Hmm.",
]


def test_browser_and_core_cut_replies_the_same_way() -> None:
    """js/speech-chunks.js (the web client) must follow the same rules as the core's SpeechChunker."""
    import json

    from playwright import sync_api

    static = Path(__file__).resolve().parents[1] / "jarvis" / "clients" / "web" / "static"
    origin = "http://jarvis.test"
    page_html = "<!doctype html><meta charset=utf-8><script type=module>import * as m from '/static/js/speech-chunks.js'; window.chunks = m; window.ready = true;</script>"

    def serve(route: Any) -> None:
        path = route.request.url.split(origin, 1)[1].split("?")[0]
        if path == "/":
            route.fulfill(status=200, content_type="text/html", body=page_html)
        else:
            route.fulfill(status=200, content_type="text/javascript", body=(static / path.removeprefix("/static/")).read_text("utf-8"))

    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as e:  # noqa: BLE001
            pytest.skip(f"Chromium is not available: {e}")
        page = b.new_page()
        page.route(f"{origin}/**", serve)
        page.goto(f"{origin}/")
        page.wait_for_function("window.ready === true")
        for step in (1, 3, 7, 1000):  # the model streams a few characters at a time; whole text is step 1000
            js = page.evaluate("""([texts, step]) => texts.map((t) => {
                const c = new chunks.SpeechChunker(); const out = [];
                for (let i = 0; i < t.length; i += step) out.push(...c.feed(t.slice(i, i + step)));
                return out.concat(c.flush()); })""", [CHUNKER_CORPUS, step])
            py = [stream(t, step) for t in CHUNKER_CORPUS]
            assert js == py, json.dumps({"step": step, "js": js, "py": py}, indent=1)
        b.close()


def test_prepare_text_fixes_known_mispronunciations() -> None:
    assert prepare_text("Your pytest run failed; check the CLI and PostgreSQL.") == \
        "Your pie test run failed; check the C L I and Postgres Q L."
    assert prepare_text("It costs $12.50, or $1 a day, and $3.00 later.") == "It costs 12 dollars 50, or 1 dollar a day, and 3 dollars later."
    assert prepare_text("On Oct 6 and Sept 21, X vs. Y, e.g. this.") == "On October 6th and September 21st, X versus Y, for example this."
    assert prepare_text("It's 14°C, 10x faster, at 9:30.") == "It's 14 degrees Celsius, 10 times faster, at 9:30."
    assert prepare_text("**Bold** and `code`") == "Bold and code"


# --------------------------------------------------------------- the config
def test_default_voice_is_kokoro_with_the_system_voice_as_fallback(tmp_path: Path) -> None:
    cfg = VoiceConfig()
    assert (cfg.tts_engine, cfg.tts_voice, cfg.tts_fallback) == ("kokoro", "af_heart", "system")
    chain = make_voice(cfg, tmp_path)
    assert isinstance(chain, VoiceChain) and isinstance(chain.primary, KokoroTTS)
    assert chain.primary.voice == "af_heart" and chain.primary.model_dir == tmp_path
    assert chain.fallback.name == ("sapi" if sys.platform == "win32" else "pyttsx3")
    assert make_voice(VoiceConfig(tts_engine="none"), tmp_path) is None
    system = make_voice(VoiceConfig(tts_engine="system"), tmp_path)
    assert system.primary is None and system.fallback is not None
    assert make_voice(VoiceConfig(tts_fallback="none"), tmp_path).fallback is None
    assert make_voice(VoiceConfig(), tmp_path, voice_override="bm_george").primary.voice == "bm_george"
    with pytest.raises(ValueError):
        VoiceConfig(tts_engine="piper")  # never implemented; no longer silently accepted


# ---------------------------------------------------------------- the chain
def test_primary_speaks_and_says_so() -> None:
    primary, fallback = FakeEngine("kokoro"), FakeEngine("sapi")
    chain = VoiceChain(primary, fallback)
    s = chain.synthesize("Hello there.", reply="r1")
    assert s.provider == "kokoro" and primary.said == ["Hello there."] and fallback.said == []
    assert chain.stats["spoken_by"] == {"kokoro": 1} and chain.stats["last_provider"] == "kokoro"


def test_one_failed_utterance_falls_back_and_the_primary_recovers() -> None:
    notices: list[tuple[str, str]] = []
    primary, fallback = FakeEngine("kokoro", fail=1), FakeEngine("sapi")
    chain = VoiceChain(primary, fallback, on_fallback=lambda e, r: notices.append((e, r)))
    assert chain.synthesize("First.", reply="r1").provider == "sapi"
    assert chain.synthesize("Second, same reply.", reply="r1").provider == "sapi", "a reply never switches voices midway"
    assert chain.synthesize("A new reply.", reply="r2").provider == "kokoro", "one error must not disable the primary"
    assert chain.stats["fallbacks"] == 1 and chain.stats["spoken_by"] == {"sapi": 2, "kokoro": 1}
    assert notices == [("kokoro", "kokoro broke")]


def test_the_same_problem_is_announced_once() -> None:
    notices: list[str] = []
    chain = VoiceChain(FakeEngine("kokoro", fail=5), FakeEngine("sapi"), on_fallback=lambda e, r: notices.append(r))
    for i in range(4):
        assert chain.synthesize(f"Reply {i}.", reply=f"r{i}").provider == "sapi"
    assert notices == ["kokoro broke"] and chain.stats["fallbacks"] == 4


def test_both_voices_failing_raises_with_both_reasons() -> None:
    chain = VoiceChain(FakeEngine("kokoro", fail=1), FakeEngine("sapi", fail=1))
    with pytest.raises(TTSUnavailable) as e:
        chain.synthesize("Hello.")
    assert "kokoro: kokoro broke" in str(e.value) and "sapi: sapi broke" in str(e.value)
    assert chain.stats["failures"] == 1


def test_a_cancelled_utterance_is_not_spoken_by_the_fallback() -> None:
    fallback = FakeEngine("sapi")
    chain = VoiceChain(FakeEngine("kokoro", error=TTSCancelled("stop"), fail=1, cancellable=True), fallback)
    with pytest.raises(TTSCancelled):
        chain.synthesize("Hello.", should_stop=lambda: True)
    assert fallback.said == []


def test_fallback_starts_before_the_primary() -> None:
    order: list[str] = []
    a, b = FakeEngine("kokoro"), FakeEngine("sapi")
    a.start, b.start = (lambda: order.append("kokoro")), (lambda: order.append("sapi"))  # type: ignore[method-assign]
    VoiceChain(a, b).start()
    assert order == ["sapi", "kokoro"]


# ------------------------------------------------- the Kokoro engine itself
def tiny_assets(monkeypatch: pytest.MonkeyPatch, model_dir: Path, content: bytes) -> None:
    """Make the engine accept small stand-in files (so 'corrupt model' doesn't need 325 MB)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    assets = tuple(kokoro_mod.Asset(a.name, len(content), "0" * 64) for a in kokoro_mod.ASSETS)
    monkeypatch.setattr(kokoro_mod, "ASSETS", assets)
    for a in assets:
        (model_dir / a.name).write_bytes(content)


def test_missing_model_is_explicit_and_the_fallback_speaks(tmp_path: Path) -> None:
    k = KokoroTTS(tmp_path / "nowhere")
    k.start()
    assert k.state == "unavailable" and "jarvis voice install" in k.error and "kokoro-v1.0.onnx (missing)" in k.error
    t0 = time.perf_counter()
    with pytest.raises(TTSUnavailable):
        k.synthesize("Hello.")
    assert time.perf_counter() - t0 < 0.5, "a missing model must not make speech wait"
    fallback = FakeEngine("sapi")
    assert VoiceChain(k, fallback).synthesize("Hello.").provider == "sapi"


def test_a_model_that_fails_to_load_falls_back_and_is_retried_later(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tiny_assets(monkeypatch, tmp_path, b"this is not an onnx model")
    k = KokoroTTS(tmp_path)
    chain = VoiceChain(k, FakeEngine("sapi"))
    chain.start()
    k._loaded.wait(30)
    assert k.state == "failed" and k.error.startswith("could not load")
    assert chain.synthesize("Hello.").provider == "sapi"
    assert k.state == "failed", "not reloaded on every utterance (retried after RETRY_FAILED_LOAD_S)"
    monkeypatch.setattr(kokoro_mod, "RETRY_FAILED_LOAD_S", 0.0)
    k.start()  # the retry happens (and fails again: the file is still bad)
    k._loaded.wait(30)
    assert k.state == "failed"


def test_a_slow_first_load_falls_back_after_the_allowed_wait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tiny_assets(monkeypatch, tmp_path, b"x")
    release = threading.Event()
    monkeypatch.setattr(KokoroTTS, "_load", lambda self: release.wait(10))
    k = KokoroTTS(tmp_path, init_wait_s=0.3)
    chain = VoiceChain(k, FakeEngine("sapi"))
    t0 = time.perf_counter()
    assert chain.synthesize("Hello.").provider == "sapi"
    assert 0.25 < time.perf_counter() - t0 < 2.0 and k.state == "loading"
    with pytest.raises(TTSCancelled):  # cancelled while waiting for the load
        k.synthesize("Hello.", should_stop=lambda: True)
    release.set()


def test_reading_voice_status_never_starts_or_imports_an_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Status reads run on the server's event loop. Measured: an import racing Kokoro's background
    loader (and SAPI's start-up) held the first /voice/status answer for ~2 s after startup."""
    from jarvis.voice import tts as tts_mod

    for worker in (tts_mod._SapiWorker, tts_mod._Pyttsx3Worker):
        monkeypatch.setattr(worker, "_instance", None)
        monkeypatch.setattr(worker, "get", classmethod(lambda cls: pytest.fail("status started an engine")))
    monkeypatch.delitem(sys.modules, "kokoro_onnx", raising=False)
    chain = make_voice(VoiceConfig(), tmp_path)
    assert chain.dependency_ok() == (True, "")  # nothing started yet, nothing failed
    chain.status()
    assert chain.fallback.error is None and KokoroTTS.dependency_ok()[0]
    assert "kokoro_onnx" not in sys.modules, "checking availability must not import the package"


def test_starting_the_voice_never_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tiny_assets(monkeypatch, tmp_path, b"x")
    release = threading.Event()
    monkeypatch.setattr(KokoroTTS, "_load", lambda self: release.wait(10))
    k = KokoroTTS(tmp_path)
    t0 = time.perf_counter()
    k.start()
    k.start()  # idempotent: one loader
    assert time.perf_counter() - t0 < 0.2 and k.state == "loading"
    assert sum(1 for t in threading.enumerate() if t.name == "jarvis-kokoro-load") == 1
    release.set()


# --------------------------------------------------------------- playback
class FakeStream:
    opened: list[FakeStream] = []

    def __init__(self, samplerate: int, **kw: Any) -> None:
        self.rate, self.written, self.aborted, self.stopped, self.closed = samplerate, 0, False, False, False
        self.latency = 0.0
        FakeStream.opened.append(self)

    def start(self) -> None:
        pass

    def write(self, data: np.ndarray) -> None:
        time.sleep(len(data) / self.rate)  # real time, like a sound card
        self.written += len(data)

    def stop(self) -> None:
        self.stopped = True

    def abort(self) -> None:
        self.aborted = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_sound(monkeypatch: pytest.MonkeyPatch) -> type[FakeStream]:
    import sounddevice

    FakeStream.opened = []
    monkeypatch.setattr(sounddevice, "OutputStream", FakeStream)
    return FakeStream


def test_chunks_of_a_reply_play_through_one_stream(fake_sound: type[FakeStream]) -> None:
    p = Player()
    for _ in range(3):
        p.enqueue(np.zeros(2400, np.float32), 24000)  # 0.1 s each, queued back to back
    assert p.wait(5)
    time.sleep(Player.LINGER_S + 0.2)
    assert len(fake_sound.opened) == 1 and fake_sound.opened[0].written == 7200
    assert fake_sound.opened[0].stopped and fake_sound.opened[0].closed and not fake_sound.opened[0].aborted


def test_stopping_aborts_at_once_and_the_next_reply_gets_a_fresh_stream(fake_sound: type[FakeStream]) -> None:
    p = Player()
    p.enqueue(np.zeros(24000 * 3, np.float32), 24000)  # 3 s
    time.sleep(0.2)
    t0 = time.perf_counter()
    p.stop()
    assert p.wait(2) and time.perf_counter() - t0 < 0.3
    assert fake_sound.opened[0].aborted, "stop drops what the device still holds"
    p.enqueue(np.zeros(2400, np.float32), 24000)
    assert p.wait(2)
    assert len(fake_sound.opened) == 2 and fake_sound.opened[1].written == 2400


def test_a_playback_failure_is_counted_and_the_player_keeps_working(fake_sound: type[FakeStream], monkeypatch: pytest.MonkeyPatch) -> None:
    p = Player()
    monkeypatch.setattr(FakeStream, "write", lambda self, data: (_ for _ in ()).throw(OSError("device unplugged")))
    p.enqueue(np.zeros(2400, np.float32), 24000)
    assert p.wait(2)
    assert p.failures == 1 and "device unplugged" in p.last_error
    monkeypatch.undo()
    import sounddevice

    monkeypatch.setattr(sounddevice, "OutputStream", FakeStream)
    p.enqueue(np.zeros(2400, np.float32), 24000)
    assert p.wait(2) and p.failures == 1


# ------------------------------------------------- the pipeline with the chain
@pytest.fixture
async def chain_env(tmp_path: Path):
    from jarvis.voice.service import VoiceService
    from tests.test_intelligence import make_runtime
    from tests.test_voice_loop import FakeSTT, FakeWake, Room, RoomPlayer

    rt, provider = make_runtime(tmp_path)
    room = Room()
    svc = VoiceService(rt, source_factory=room.mic, dry_run_audio=True)
    primary, fallback = FakeEngine("kokoro", cancellable=True), FakeEngine("sapi")
    svc.stt, svc.wake = FakeSTT(), FakeWake()
    svc.tts = VoiceChain(primary, fallback, on_fallback=svc._fallback_notice)
    svc.player = RoomPlayer(room, gain=0.0)
    rt.extensions["voice"] = svc
    await rt.start()
    events: list[Any] = []
    rt.events.on("*", events.append)
    yield SimpleNamespace(rt=rt, provider=provider, svc=svc, room=room, events=events, primary=primary, fallback=fallback)
    await svc.stop_pipeline()
    await rt.stop()
    room.close()


def of(events: list[Any], type_: str) -> list[Any]:
    return [e for e in events if e.type == type_]


async def converse(env: SimpleNamespace, reply: str, request: str = "tell me") -> str:
    env.provider.push({"text": reply})
    await env.svc.start_pipeline()
    out = await env.svc.pipeline.converse(request)
    await asyncio.sleep(0.2)  # events reach subscribers asynchronously
    return out


async def test_a_reply_is_spoken_by_kokoro_chunk_by_chunk(chain_env: SimpleNamespace) -> None:
    env = chain_env
    reply = "You have three meetings tomorrow, and the first starts at nine. Nothing else is booked."
    await converse(env, reply)
    assert env.primary.said == ["You have three meetings tomorrow,", "and the first starts at nine.", "Nothing else is booked."]
    assert env.fallback.said == []
    started = of(env.events, "SPEECH_STARTED")
    assert len(started) == 1 and started[0].data["provider"] == "kokoro" and started[0].data["first_audio_ms"] >= 0
    assert len(of(env.events, "SPEECH_ENDED")) == 1 and env.svc.pipeline.last_provider == "kokoro"
    assert env.svc.player.items == 3
    assert env.svc.pipeline.state == "standby" and env.svc.pipeline.interaction is None


async def test_kokoro_failing_mid_reply_hands_the_rest_to_the_fallback(chain_env: SimpleNamespace) -> None:
    env = chain_env
    real = env.primary.synthesize
    calls = {"n": 0}

    def flaky(text: str, timeout: float = 60, should_stop: Any = None) -> Speech:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("synthesis failed: boom")
        return real(text, timeout, should_stop)

    env.primary.synthesize = flaky  # type: ignore[method-assign]
    await converse(env, "First sentence here. Second sentence here. Third sentence here.")
    assert env.primary.said == ["First sentence here."]
    assert env.fallback.said == ["Second sentence here.", "Third sentence here."]
    assert env.svc.player.items == 3 and len(of(env.events, "SPEECH_ENDED")) == 1
    notes = of(env.events, "NOTIFICATION")
    assert len(notes) == 1 and notes[0].data["title"] == "Using the backup voice"
    # the next reply is Kokoro again
    await converse(env, "All good now.")
    assert env.primary.said[-1] == "All good now."
    assert env.svc.pipeline.state == "standby"


async def test_when_no_voice_can_speak_the_reply_is_kept_and_nothing_hangs(chain_env: SimpleNamespace) -> None:
    env = chain_env
    env.primary.fail = env.fallback.fail = 99
    reply = await converse(env, "This cannot be spoken. Nor can this.")
    assert reply.startswith("This cannot be spoken."), "the reply text is still returned (and in the chat)"
    assert not of(env.events, "SPEECH_STARTED") and env.svc.player.items == 0
    notes = [e for e in of(env.events, "NOTIFICATION") if e.data["title"] == "JARVIS couldn't speak"]
    assert len(notes) == 1 and "no voice could speak" in notes[0].data["body"]
    assert env.svc.pipeline.state == "standby" and env.svc.pipeline.interaction is None
    msgs = env.rt.conversations.history(env.svc.pipeline.conversation_id)
    assert any(m.role == "assistant" and "cannot be spoken" in (m.content or "") for m in msgs)


async def test_interrupting_cancels_the_chunk_being_synthesized(chain_env: SimpleNamespace) -> None:
    env = chain_env
    env.primary.block_until_stopped = True  # synthesis "takes forever" until cancelled
    env.provider.push({"text": "A sentence that takes long to synthesize. Another one."})
    await env.svc.start_pipeline()
    task = asyncio.create_task(env.svc.pipeline.converse("tell me"))
    end = time.monotonic() + 5
    while env.svc.pipeline.interaction is None and time.monotonic() < end:
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.3)
    t0 = time.perf_counter()
    await asyncio.to_thread(env.svc.pipeline.interrupt, False, "stop button")
    await asyncio.wait_for(task, 5)
    assert time.perf_counter() - t0 < 2.0
    assert env.primary.cancelled.is_set(), "interrupt reached the synthesis in progress"
    assert env.primary.said == [] and env.fallback.said == [] and env.svc.player.items == 0
    assert env.svc.pipeline.state == "standby" and not of(env.events, "SPEECH_STARTED")


async def test_an_interrupted_announcement_says_so(chain_env: SimpleNamespace) -> None:
    """Measured on the live core: `jarvis voice say` reported "spoke 19.53 s" for speech cut off after ~2 s."""
    env = chain_env
    await env.svc.start_pipeline()
    assert await env.svc.pipeline.announce("Short and complete.") > 0
    assert env.svc.pipeline.last_announcement_interrupted is False
    text = " ".join(f"Sentence number {i} of a long announcement." for i in range(1, 9))
    task = asyncio.create_task(env.svc.pipeline.announce(text))
    end = time.monotonic() + 5
    while env.svc.pipeline.state != "speaking" and time.monotonic() < end:
        await asyncio.sleep(0.01)
    await asyncio.to_thread(env.svc.pipeline.interrupt, False, "stop requested")
    await asyncio.wait_for(task, 5)
    assert env.svc.pipeline.last_announcement_interrupted is True
    assert env.svc.pipeline.state == "standby" and env.fallback.said == []


async def test_saying_something_with_voice_off_is_chunked_and_interruptible(chain_env: SimpleNamespace) -> None:
    """`jarvis voice say` with the microphone off: chunk by chunk, and /voice/interrupt stops it."""
    from fastapi.testclient import TestClient

    from jarvis.api.server import create_app

    env = chain_env
    assert env.svc.pipeline is None or not env.svc.pipeline.running
    text = " ".join(f"Sentence number {i} of a long announcement." for i in range(1, 9))
    auth = {"Authorization": f"Bearer {env.rt.local_token}"}
    with TestClient(create_app(env.rt, manage_lifecycle=False)) as client:
        out = await asyncio.to_thread(client.post, "/api/v1/voice/say", json={"text": "Short and complete."}, headers=auth)
        assert out.json() == {"spoken_s": 0.2, "provider": "kokoro", "interrupted": False}
        task = asyncio.create_task(env.svc._say_directly(text))
        end = time.monotonic() + 5
        while env.svc.player.items < 2 and time.monotonic() < end:
            await asyncio.sleep(0.01)
        r = await asyncio.to_thread(client.post, "/api/v1/voice/interrupt", headers=auth)
        assert r.status_code == 200
        result = await asyncio.wait_for(task, 5)
    assert result["interrupted"] is True and result["provider"] == "kokoro"
    assert len(env.primary.said) - 1 >= 2, "spoken chunk by chunk"
    # Synthesis runs ahead of playback (Kokoro is ~5x faster than real time): what proves the stop
    # is that playback ended early. items counts chunks that started playing (1 for the short one).
    assert env.svc.player.items - 1 < 8 and not env.svc.player.busy
    assert env.fallback.said == [], "an interrupt is not a failure"


async def test_status_says_which_voice_is_speaking(chain_env: SimpleNamespace) -> None:
    env = chain_env
    await converse(env, "Hello there.")
    st = env.svc.status()["tts"]
    assert st["available"] and st["engine"] == "kokoro" and st["voice"] == "kokoro-voice"
    assert st["details"]["spoken_by"] == {"kokoro": 1}


async def test_voice_service_start_returns_before_kokoro_has_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jarvis.voice.service import VoiceService
    from tests.test_intelligence import make_runtime

    rt, _ = make_runtime(tmp_path)
    model_dir = rt.paths.models / "tts" / "kokoro"
    tiny_assets(monkeypatch, model_dir, b"x")
    release = threading.Event()
    monkeypatch.setattr(KokoroTTS, "_load", lambda self: release.wait(10))
    svc = VoiceService(rt, dry_run_audio=True)
    rt.extensions["voice"] = svc
    t0 = time.perf_counter()
    await rt.start()
    took = time.perf_counter() - t0
    end = time.monotonic() + 5
    while svc.tts.primary.state != "loading" and time.monotonic() < end:
        await asyncio.sleep(0.02)
    assert svc.tts.primary.state == "loading" and took < 2.0
    assert svc.status()["tts"]["speaking_with"] == "kokoro (loading)"
    release.set()
    await rt.stop()


# ------------------------------------------------------------ the real model
@pytest.fixture(scope="module")
def real_kokoro() -> KokoroTTS:
    k = KokoroTTS(REAL_MODEL_DIR)
    k.start()
    assert k._loaded.wait(120) and k.state == "ready", k.error
    return k


@pytest.mark.kokoro
@needs_kokoro
def test_real_kokoro_speaks_af_heart_at_24khz(real_kokoro: KokoroTTS) -> None:
    s = real_kokoro.synthesize("The build passed, and all forty-two tests ran.")
    assert s.provider == "kokoro" and s.sample_rate == 24000 and real_kokoro.voice == "af_heart"
    dur = len(s.pcm) / s.sample_rate
    assert 2.0 < dur < 6.0 and float(np.max(np.abs(s.pcm))) > 0.05
    assert np.all(s.pcm[-int(0.2 * 24000):] == 0), "a sentence ends with its pause"
    assert s.latency_ms < 10_000


@pytest.mark.kokoro
@needs_kokoro
def test_real_kokoro_is_intelligible(real_kokoro: KokoroTTS) -> None:
    pytest.importorskip("faster_whisper")
    from jarvis.voice.audio import resample
    from jarvis.voice.stt import FasterWhisperSTT

    s = real_kokoro.synthesize("Please remember that my car is parked on level three.")
    heard = FasterWhisperSTT("base.en").transcribe(resample(s.pcm, s.sample_rate), 16000).text.lower()
    assert "car" in heard and "parked" in heard and ("three" in heard or "3" in heard), heard


@pytest.mark.kokoro
@needs_kokoro
def test_real_kokoro_stops_mid_synthesis_and_recovers(real_kokoro: KokoroTTS) -> None:
    long = " ".join(["This sentence keeps going so that synthesizing it takes a good while"] * 6) + "."
    stop = threading.Event()
    threading.Timer(0.3, stop.set).start()
    t0 = time.perf_counter()
    with pytest.raises(TTSCancelled):
        real_kokoro.synthesize(long, should_stop=stop.is_set)
    assert time.perf_counter() - t0 < 0.3 + 0.5, "the run is terminated, not finished"
    threading.Timer(0.3, real_kokoro.cancel).start()
    with pytest.raises(TTSCancelled):
        real_kokoro.synthesize(long)
    assert real_kokoro.synthesize("Still here.").provider == "kokoro"


@pytest.mark.kokoro
@needs_kokoro
def test_real_kokoro_synthesis_error_falls_back_once(real_kokoro: KokoroTTS, monkeypatch: pytest.MonkeyPatch) -> None:
    chain = VoiceChain(real_kokoro, FakeEngine("sapi"))
    original = real_kokoro._kokoro.create
    calls = {"n": 0}

    def broken_once(*a: Any, **kw: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("onnxruntime exploded")
        return original(*a, **kw)

    monkeypatch.setattr(real_kokoro._kokoro, "create", broken_once)
    assert chain.synthesize("One.", reply="a").provider == "sapi"
    assert chain.synthesize("Two.", reply="b").provider == "kokoro"


@pytest.mark.kokoro
@needs_kokoro
def test_unknown_voice_uses_the_default_and_says_so(tmp_path: Path) -> None:
    for a in kokoro_mod.ASSETS:
        os.link(REAL_MODEL_DIR / a.name, tmp_path / a.name)  # no copy of 350 MB
    k = KokoroTTS(tmp_path, voice="zz_nobody")
    k.start()
    assert k._loaded.wait(120) and k.state == "ready"
    assert k.voice == "af_heart" and "zz_nobody" in k.notice
    assert k.synthesize("Hello.").provider == "kokoro"


@pytest.mark.kokoro
@needs_kokoro
async def test_api_reports_kokoro_and_reload_picks_up_installed_files(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    from jarvis.api.server import create_app
    from jarvis.voice.service import VoiceService
    from tests.test_intelligence import make_runtime

    rt, _ = make_runtime(tmp_path)
    svc = VoiceService(rt, dry_run_audio=True)
    rt.extensions["voice"] = svc
    await rt.start()
    try:
        auth = {"Authorization": f"Bearer {rt.local_token}"}
        with TestClient(create_app(rt, manage_lifecycle=False)) as client:
            st = client.get("/api/v1/voice/status", headers=auth).json()["tts"]
            assert st["engine"] == "kokoro" and st["speaking_with"] in ("sapi", "pyttsx3") and "jarvis voice install" in st["reason"]
            wav = client.post("/api/v1/voice/speak", json={"text": "Before."}, headers=auth)
            assert wav.status_code == 200 and wav.headers["x-jarvis-voice"] in ("sapi", "pyttsx3")
            kokoro_mod.install_assets(svc.tts_model_dir, REAL_MODEL_DIR, progress=lambda m: None)  # hard links
            client.post("/api/v1/voice/tts/reload", headers=auth)
            end = time.monotonic() + 60
            while client.get("/api/v1/voice/status", headers=auth).json()["tts"]["speaking_with"] != "kokoro" and time.monotonic() < end:
                time.sleep(0.2)
            wav = client.post("/api/v1/voice/speak", json={"text": "After."}, headers=auth)
            assert wav.status_code == 200 and wav.headers["x-jarvis-voice"] == "kokoro"
    finally:
        await rt.stop()
