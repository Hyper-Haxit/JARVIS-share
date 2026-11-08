"""Continuous voice pipeline.

One wake word starts exactly one interaction:

    standby     microphone open, wake-word detector armed; audio stays on this device
    recording   capturing one request; voice activity detection ends it
    processing  speech recognition, then the agent
    speaking    the reply is spoken sentence by sentence while it is still being generated
    standby     again, after a short echo guard

With ``conversation_window_s`` > 0 a ``followup`` window after a reply accepts an
answer without the wake word (off by default). Without a wake-word engine the
pipeline sits in ``listening`` and voice activity starts each request.

JARVIS must never hear itself:
  * While it speaks, the microphone can interrupt it only with the wake word
    (``barge_in_mode = "wake_word"``, the default and safe with speakers). Plain
    loud speech interrupts only with ``barge_in_mode = "voice"``, meant for
    headsets where JARVIS's voice can't reach the microphone.
  * Wake detections are ignored while JARVIS is saying its own name.
  * After it stops speaking the microphone is ignored for a short echo guard,
    then the detectors start fresh, so the tail of its own voice can't start a
    new interaction.

Every interaction has an id (``vi_...``) that its events and log lines carry, so
duplicate or stale activity is visible. Results that arrive for an interaction
that is no longer current (interrupted, stopped, replaced) are discarded.

Threads: audio frames arrive on the sound-device thread, the state machine runs
on its own thread, and agent runs execute on the core's asyncio loop.
"""

from __future__ import annotations

import asyncio
import functools
import queue
import re
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from jarvis.core.context import DeviceContext
from jarvis.core.events import EventType
from jarvis.core.observability import get_logger
from jarvis.voice.audio import AudioSource
from jarvis.voice.tts import SpeechChunker, TTSCancelled
from jarvis.voice.vad import EnergyVAD, level01

log = get_logger("voice.pipeline")

WAKE_PREFIX = re.compile(r"^\s*(?:(?:hey|hi|ok|okay|yo)\s*,?\s*)?jarvis\b[\s,.!?]*", re.I)
OWN_NAME = re.compile(r"\bjarvis\b", re.I)
NO_SPEECH_TIMEOUT_S = 6.0
BARGE_IN_MS = 300
CONVERSATION_IDLE_S = 600
ECHO_GUARD_S = 0.6  # output latency plus room reverberation after JARVIS stops speaking
SILENT_PEAK = 1e-4  # a frame whose peak stays within ~3 LSB of 16-bit audio carries no sound

_FILLER = {"uh", "um", "umm", "uhm", "hmm", "hm", "mm", "mhm", "mmm", "ah", "oh", "huh", "er", "erm"}
# Phrases speech recognition produces from noise or silence; never a request.
_NOISE_PHRASES = {"you", "thanks for watching", "thank you for watching", "thank you so much for watching",
                  "please subscribe", "subscribe", "bye bye"}


def strip_wake(text: str) -> str:
    return WAKE_PREFIX.sub("", text).strip()


def judge_transcript(text: str, no_speech_prob: float | None) -> str | None:
    """Why a transcript must not become a request, or None when it can."""
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
    if not words:
        return "nothing intelligible"
    if all(w in _FILLER for w in words):
        return "only filler sounds"
    if " ".join(words) in _NOISE_PHRASES:
        return "a phrase speech recognition produces from background noise"
    if no_speech_prob is not None and no_speech_prob > 0.85 and len(text) < 12:
        return "probably not speech"
    return None


def new_interaction_id() -> str:
    return "vi_" + secrets.token_hex(4)


class VoicePipeline:
    def __init__(self, service: Any, source_factory: Callable[[], AudioSource]) -> None:
        self.svc = service
        self.rt = service.runtime
        self.cfg = service.config
        self.source_factory = source_factory
        self.state = "off"
        self.error: str | None = None
        self.interaction: str | None = None
        self._lock = threading.RLock()
        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1500)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._source: AudioSource | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._run_cancel: asyncio.Event | None = None
        self._run_future: Any = None
        self._vad = EnergyVAD(threshold_db=self.cfg.vad_threshold_db, silence_ms=self.cfg.silence_ms)
        self._utterance: list[np.ndarray] = []
        self._speech_started = False
        self._rec_started = 0.0
        self._level_at = 0.0
        self._followup_until = 0.0
        self._suppress_until = 0.0
        self._suppressed = False
        self._says_own_name = False
        self._reset_detectors = False
        self.conversation_id: str | None = None
        self._conv_at = 0.0
        self.stats = {"wakes": 0, "utterances": 0, "interruptions": 0, "ignored": 0, "last_transcript": None}
        self.last_provider: str | None = None  # the voice engine that spoke last ("kokoro", or the fallback)
        self.last_announcement_interrupted = False
        # Microphone signal health: real frame count and level, so a muted or
        # privacy-blocked microphone (which delivers digital silence) is visible.
        self.frames = 0
        self.level_db = -120.0
        self.silent_frames = 0

    @property
    def barge_in_mode(self) -> str:
        return getattr(self.cfg, "barge_in_mode", "wake_word")

    def _idle_state(self) -> str:
        return "standby" if self.svc.wake is not None else "listening"

    # --------------------------------------------------------------- control
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._loop = loop
        self._stop.clear()
        self.error = None
        self._source = self.source_factory()
        self._source.start(self._on_frame)
        self._thread = threading.Thread(target=self._run, name="jarvis-voice", daemon=True)
        self._thread.start()
        log.info("voice listener started (thread %s; barge-in %s; follow-up window %.0f s)", self._thread.ident,
                 self.barge_in_mode if self.cfg.barge_in else "off", self.cfg.conversation_window_s)
        self._set_state(self._idle_state())

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            ended, self.interaction = self.interaction, None
        if ended:
            log.info("voice[%s] interaction ended (voice stopped)", ended)
        if self._source is not None:
            try:
                self._source.stop()
            except Exception:  # noqa: BLE001
                log.exception("stopping audio source failed")
        self.svc.player.stop()
        self.svc.cancel_speech()
        self._cancel_run()
        if self._thread is not None:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                log.warning("voice thread %s is still finishing speech recognition; its result will be discarded",
                            self._thread.ident)
            else:
                log.info("voice listener stopped (thread %s)", self._thread.ident)
        self._thread = None
        self._set_state("off")

    def signal(self) -> dict[str, Any]:
        if not self.running:
            return {"state": "off"}
        if self.frames == 0:
            state = "no audio arriving"
        elif self.silent_frames >= 100:  # 3 s with no sound at all: muted or blocked, not a quiet room
            state = "digital silence (microphone muted, or blocked by Windows privacy settings)"
        else:
            state = "ok"
        return {"state": state, "frames": self.frames, "level_db": self.level_db}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def interrupt(self, start_recording: bool = True, reason: str = "request", score: float | None = None) -> None:
        """Stop speaking now (barge-in or user request); ``score`` is the wake word's, when it caused this."""
        with self._lock:
            old = self.interaction
            was_speaking = self.svc.player.busy or self.state == "speaking"
            self.svc.player.stop()
            self.svc.cancel_speech()  # a chunk being synthesized right now is abandoned too
            self._cancel_run()
            if was_speaking:
                self.stats["interruptions"] += 1
                self._emit(EventType.SPEECH_INTERRUPTED, {"interaction": old, "reason": reason})
                log.info("voice[%s] speaking interrupted (%s)", old, reason)
            if not self.running:
                return
            if start_recording:
                self._begin_recording([], reason=f"interrupted by {reason}", score=score)
            else:
                self._end_interaction(old, "interrupted", echo_guard=was_speaking)

    # ------------------------------------------------------------- plumbing
    def _on_frame(self, frame: np.ndarray) -> None:
        self.frames += 1
        silent = frame.size == 0 or float(np.max(np.abs(frame))) < SILENT_PEAK
        self.silent_frames = self.silent_frames + 1 if silent else 0
        if self.frames % 10 == 0:
            self.level_db = round(_db(frame), 1)
        try:
            self._frames.put_nowait(frame)
        except queue.Full:
            try:
                self._frames.get_nowait()
                self._frames.put_nowait(frame)
            except queue.Empty:
                pass

    def _emit(self, type_: str, data: dict[str, Any]) -> None:
        self.rt.events.emit(type_, data, source="voice")

    def _set_state(self, state: str) -> None:
        with self._lock:
            if state == self.state:
                return
            previous, self.state = self.state, state
            self._emit(EventType.MIC_STATE, {"state": state, "interaction": self.interaction})
            log.info("voice[%s] state %s -> %s", self.interaction or "-", previous, state)

    def _level(self, frame: np.ndarray, source: str) -> None:
        now = time.monotonic()
        if now - self._level_at >= 1 / 15:
            self._level_at = now
            self._emit(EventType.AUDIO_LEVEL, {"level": round(level01(frame), 3), "source": source})

    def _drain(self) -> int:
        n = 0
        try:
            while True:
                self._frames.get_nowait()
                n += 1
        except queue.Empty:
            pass
        return n

    def _cancel_run(self) -> None:
        if self._run_cancel is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._run_cancel.set)

    def _begin_recording(self, preroll: list[np.ndarray], reason: str, score: float | None = None) -> str:
        iid = new_interaction_id()
        with self._lock:
            self.interaction = iid
            self._utterance = list(preroll)
            self._speech_started = False
            self._rec_started = time.monotonic()
            self._vad.reset()
            if score is not None:
                self.stats["wakes"] += 1
                self._emit(EventType.WAKE_WORD_DETECTED, {"score": round(score, 3), "interaction": iid})
                log.info("voice[%s] wake word detected (score %.2f)", iid, score)
            log.info("voice[%s] listening started (%s)", iid, reason)
            self._set_state("recording")
        return iid

    def _end_interaction(self, iid: str | None, outcome: str, echo_guard: bool = False) -> bool:
        """Close the current interaction and go back to waiting. No-op for a stale id."""
        with self._lock:
            if iid is None or self.interaction != iid:
                return False
            self.interaction = None
            self._says_own_name = False
            if echo_guard:
                self._suppress_until = time.monotonic() + ECHO_GUARD_S
                log.info("voice[%s] microphone suppressed for %.1f s (echo of JARVIS's own voice)", iid, ECHO_GUARD_S)
            self._reset_detectors = True
            follow = outcome == "answered" and self.cfg.conversation_window_s > 0 and self.running
            if follow:
                self._followup_until = time.monotonic() + ECHO_GUARD_S + self.cfg.conversation_window_s
            log.info("voice[%s] interaction ended (%s)", iid, outcome)
            if self.running:
                self._set_state("followup" if follow else self._idle_state())
        return True

    # ------------------------------------------------------------ main loop
    def _run(self) -> None:
        cfg = self.cfg
        wake = self.svc.wake
        preroll: deque[np.ndarray] = deque(maxlen=12)  # 360 ms
        barge = 0
        try:
            if wake is not None:
                wake.load()
                wake.reset()
        except Exception as e:  # noqa: BLE001
            self.error = str(e)
            log.error("wake word unavailable: %s", e)
            self._set_state("error")
            return
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.2)
            except queue.Empty:
                self._followup_expired()
                continue
            if self._stop.is_set():
                break
            # Echo guard: right after JARVIS spoke, the microphone still hears it.
            if time.monotonic() < self._suppress_until:
                self._suppressed = True
                continue
            if self._suppressed:
                self._suppressed = False
                log.info("microphone suppression lifted")
            if self._reset_detectors:
                # Start fresh after every interaction: nothing JARVIS said, and
                # nothing heard before, may linger in the detectors.
                self._reset_detectors = False
                preroll.clear()
                self._vad.reset()
                barge = 0
                if wake is not None:
                    wake.reset()
            st = self.state
            try:
                if st in ("standby", "followup", "listening"):
                    preroll.append(frame)
                    ev = self._vad.process(frame)
                    score = wake.process(frame) if wake is not None else 0.0
                    if wake is not None and wake.detected(score):
                        wake.reset()
                        self._begin_recording([], reason="wake word", score=score)
                    elif st in ("followup", "listening") and ev is not None and ev.kind == "start":
                        self._begin_recording(list(preroll), reason="speech in follow-up window" if st == "followup" else "speech")
                    else:
                        self._followup_expired()
                elif st == "recording":
                    self._utterance.append(frame)
                    self._level(frame, "mic")
                    ev = self._vad.process(frame)
                    elapsed = time.monotonic() - self._rec_started
                    if ev is not None and ev.kind == "start" and not self._speech_started:
                        self._speech_started = True
                        self._emit(EventType.USER_SPEECH_STARTED, {"interaction": self.interaction})
                        log.info("voice[%s] speech started", self.interaction)
                    if ev is not None and ev.kind == "end" and self._speech_started:
                        self._finish_utterance("end of speech")
                        preroll.clear()
                    elif elapsed > cfg.max_utterance_s:
                        self._finish_utterance(f"reached the {cfg.max_utterance_s:.0f} s limit")
                        preroll.clear()
                    elif not self._speech_started and elapsed > NO_SPEECH_TIMEOUT_S:
                        iid = self.interaction
                        self._emit(EventType.TRANSCRIPT, {"text": "", "reason": "no speech heard", "interaction": iid})
                        log.info("voice[%s] listening stopped: no speech within %.0f s", iid, NO_SPEECH_TIMEOUT_S)
                        self._end_interaction(iid, "no speech heard")
                elif st == "speaking":
                    # JARVIS's own voice is in this audio. Only the wake word (or, with a
                    # headset, clearly louder speech) may interrupt it.
                    score = wake.process(frame) if wake is not None else 0.0
                    if wake is not None and wake.detected(score):
                        wake.reset()
                        if self._says_own_name:
                            log.info("voice[%s] wake word ignored: JARVIS is saying its own name", self.interaction)
                        elif cfg.barge_in:
                            log.info("voice[%s] barge-in: wake word while speaking (score %.2f)", self.interaction, score)
                            self.interrupt(start_recording=True, reason="wake word", score=score)
                        continue
                    if cfg.barge_in and self.barge_in_mode == "voice":
                        db_voiced = self._vad.is_voiced(_db(frame), extra_margin_db=10)
                        barge = barge + 1 if db_voiced else 0
                        if barge * 30 >= BARGE_IN_MS:
                            barge = 0
                            log.info("voice[%s] barge-in: loud speech while speaking", self.interaction)
                            self.interrupt(start_recording=True, reason="voice")
                # processing: frames are discarded (speech recognition and the agent are working)
            except Exception as e:  # noqa: BLE001
                log.exception("voice loop error")
                self.error = f"{type(e).__name__}: {e}"

    def _followup_expired(self) -> None:
        with self._lock:
            if self.state == "followup" and self.interaction is None and time.monotonic() > self._followup_until:
                log.info("follow-up window closed")
                self._set_state(self._idle_state())

    def _finish_utterance(self, why: str) -> None:
        iid = self.interaction
        audio = np.concatenate(self._utterance) if self._utterance else np.zeros(0, np.float32)
        self._utterance = []
        duration = round(len(audio) / 16000, 2)
        self._set_state("processing")
        self._emit(EventType.USER_SPEECH_ENDED, {"duration_s": duration, "interaction": iid})
        log.info("voice[%s] listening stopped: %s (%.1f s of audio); speech recognition started", iid, why, duration)
        try:
            t = self.svc.stt.transcribe(audio)
        except Exception as e:  # noqa: BLE001
            self.error = f"speech recognition failed: {e}"
            log.error("voice[%s] speech recognition failed: %s", iid, e)
            self._emit(EventType.TRANSCRIPT, {"text": "", "error": self.error, "interaction": iid})
            self._drain()
            self._end_interaction(iid, "speech recognition failed")
            return
        dropped = self._drain()  # audio queued while recognising belongs to no request
        text = strip_wake(t.text)
        log.info("voice[%s] speech recognition finished in %.0f ms: %r (%d queued frames dropped)",
                 iid, t.latency_ms, text[:80], dropped)
        if self._stop.is_set() or self.interaction != iid:
            log.info("voice[%s] transcript discarded: voice was stopped or interrupted meanwhile", iid)
            return
        ignored = judge_transcript(text, t.no_speech_prob)
        self._emit(EventType.TRANSCRIPT, {"text": text, "latency_ms": t.latency_ms, "duration_s": t.duration_s,
                                          "interaction": iid, "ignored": ignored})
        if ignored:
            self.stats["ignored"] += 1
            log.info("voice[%s] transcript not sent to the agent: %s", iid, ignored)
            self._end_interaction(iid, f"ignored: {ignored}")
            return
        self.stats["utterances"] += 1
        self.stats["last_transcript"] = text
        assert self._loop is not None
        self._run_future = asyncio.run_coroutine_threadsafe(self.converse(text, iid), self._loop)

    # ---------------------------------------------------------- conversation
    def _is_current(self, iid: str, cancel: asyncio.Event | None = None) -> bool:
        return self.interaction == iid and not self._stop.is_set() and not (cancel is not None and cancel.is_set())

    async def converse(self, text: str, interaction: str | None = None) -> str:
        """Run the agent on a spoken request and speak the reply (also usable directly)."""
        rt = self.rt
        if interaction is None:
            with self._lock:
                interaction = self.interaction = new_interaction_id()
        iid = interaction
        if not self._is_current(iid):
            log.info("voice[%s] request dropped: the interaction is no longer current", iid)
            return ""
        cancel = asyncio.Event()
        self._run_cancel = cancel
        if self.conversation_id is None or time.time() - self._conv_at > CONVERSATION_IDLE_S or not rt.conversations.exists(self.conversation_id):
            self.conversation_id = rt.conversations.create(self.svc.principal.device_id, kind="chat")
        self._conv_at = time.time()
        loop = asyncio.get_running_loop()
        requested = time.monotonic()
        # The reply is spoken while it is still being written: the chunker hands over each
        # sentence (or an early first clause) as soon as it is complete, and chunks are
        # synthesized strictly one after another into the one player queue, so the next
        # chunk is prepared while the previous one plays and nothing ever overlaps.
        chunker = SpeechChunker()
        spoken_any = False
        voice_failed = False
        chunks_spoken = 0
        player_failures = self.svc.player.failures
        speech_chain: asyncio.Future[Any] | None = None

        def stopped() -> bool:
            return not self._is_current(iid, cancel)

        async def say(chunk: str) -> None:
            nonlocal spoken_any, voice_failed, chunks_spoken
            if voice_failed or stopped() or not chunk.strip() or self.svc.tts is None:
                return
            try:
                speech = await loop.run_in_executor(
                    None, functools.partial(self.svc.synthesize, chunk, reply=iid, should_stop=stopped))
            except TTSCancelled:
                return
            except Exception as e:  # noqa: BLE001 - no voice could speak; the reply itself is kept (it's in the chat)
                voice_failed = True
                self.error = f"speech failed: {e}"
                log.error("voice[%s] could not speak the reply: %s", iid, e)
                self._emit(EventType.NOTIFICATION, {"title": "JARVIS couldn't speak", "body": str(e)[:240], "level": "warn",
                                                    "interaction": iid})
                return
            if not speech.pcm.size:
                return
            with self._lock:
                if stopped():
                    return
                if OWN_NAME.search(chunk):
                    self._says_own_name = True
                chunks_spoken += 1
                self.last_provider = speech.provider
                if not spoken_any:
                    spoken_any = True
                    first_ms = round((time.monotonic() - requested) * 1000)
                    self._set_state("speaking")
                    self._emit(EventType.SPEECH_STARTED, {"interaction": iid, "provider": speech.provider, "first_audio_ms": first_ms})
                    log.info("voice[%s] speaking started via %s, %d ms after the request (microphone can interrupt only with %s)",
                             iid, speech.provider, first_ms,
                             "loud speech or the wake word" if self.barge_in_mode == "voice" else "the wake word")
                log.info("voice[%s] chunk %d via %s: synthesized in %.0f ms, %.1f s of audio: %r", iid, chunks_spoken,
                         speech.provider, speech.latency_ms, len(speech.pcm) / speech.sample_rate, chunk[:60])
                self.svc.player.enqueue(speech.pcm, speech.sample_rate)

        def queue_speech(chunks: list[str]) -> None:
            nonlocal speech_chain
            for c in chunks:
                prev = speech_chain

                async def chained(c: str = c, prev: Any = prev) -> None:
                    if prev is not None:
                        await prev
                    await say(c)

                speech_chain = asyncio.ensure_future(chained())

        async def emit(event: str, data: dict[str, Any]) -> None:
            if event == "delta":
                queue_speech(chunker.feed(data["text"]))
            elif event == "approval":
                queue_speech([f"I need your approval to use {data.get('tool', 'that tool')}. It's waiting on your screen."])

        device = DeviceContext(device_id=self.svc.principal.device_id, name="Voice on this computer", kind="core",
                               capabilities=["microphone", "speaker"], voice=True)
        started = time.monotonic()
        log.info("voice[%s] agent request created (conversation %s)", iid, self.conversation_id)
        outcome = "answered"
        try:
            result = await rt.agent.run(text=text, principal=self.svc.principal, conversation_id=self.conversation_id,
                                        device=device, emit=emit, cancel=cancel)
            log.info("voice[%s] agent request completed in %.0f ms (%d characters)", iid,
                     (time.monotonic() - started) * 1000, len(result.text or ""))
            if self._is_current(iid, cancel):
                rest = chunker.flush()
                if chunker.chunks == 0 and result.text:  # nothing was streamed: speak the final text, chunk by chunk
                    rest = chunker.feed(result.text) + chunker.flush()
                queue_speech(rest)
            if speech_chain is not None:
                await speech_chain
            if self._is_current(iid, cancel):
                await loop.run_in_executor(None, self.svc.player.wait, 120)
                if self.svc.player.failures > player_failures:
                    self.error = f"audio playback failed: {self.svc.player.last_error}"
                    log.error("voice[%s] %s", iid, self.error)
                    self._emit(EventType.NOTIFICATION, {"title": "JARVIS couldn't play its voice", "level": "warn",
                                                        "body": str(self.svc.player.last_error)[:240], "interaction": iid})
            reply = result.text
        except Exception as e:  # noqa: BLE001
            log.exception("voice[%s] conversation failed", iid)
            reply = ""
            self.error = str(e)
            outcome = "failed"
        finally:
            if self._run_cancel is cancel:
                self._run_cancel = None
        if spoken_any:
            self._emit(EventType.SPEECH_ENDED, {"interaction": iid})
            log.info("voice[%s] speaking finished", iid)
        if not self._end_interaction(iid, outcome, echo_guard=spoken_any):
            log.info("voice[%s] reply finished after the interaction was replaced or stopped", iid)
        return reply

    async def announce(self, text: str) -> float:
        """Speak on this computer's speakers outside a conversation, with the same echo protection.

        Spoken chunk by chunk like a reply: the first chunk starts playing while the rest is synthesized."""
        if self.svc.tts is None:
            raise RuntimeError("text-to-speech is disabled")
        loop = asyncio.get_running_loop()
        chunker = SpeechChunker()
        chunks = chunker.feed(text) + chunker.flush()
        if not chunks:
            return 0.0
        iid = new_interaction_id()
        first = await loop.run_in_executor(None, functools.partial(self.svc.synthesize, chunks[0], reply=iid))
        with self._lock:
            if self.interaction is not None:
                raise RuntimeError("JARVIS is in the middle of a voice interaction")
            self.interaction = iid
            self.last_provider = first.provider
            self._says_own_name = bool(OWN_NAME.search(text))
            self._set_state("speaking")
            self._emit(EventType.SPEECH_STARTED, {"interaction": iid, "provider": first.provider})
            log.info("voice[%s] announcement started via %s (%d chunk(s))", iid, first.provider, len(chunks))
            self.svc.player.enqueue(first.pcm, first.sample_rate)
        spoken = len(first.pcm) / first.sample_rate
        for chunk in chunks[1:]:
            try:
                speech = await loop.run_in_executor(
                    None, functools.partial(self.svc.synthesize, chunk, reply=iid, should_stop=lambda: self.interaction != iid))
            except TTSCancelled:
                break
            except Exception as e:  # noqa: BLE001
                log.error("voice[%s] announcement cut short: %s", iid, e)
                break
            with self._lock:
                if self.interaction != iid:
                    break
                self.last_provider = speech.provider
                self.svc.player.enqueue(speech.pcm, speech.sample_rate)
            spoken += len(speech.pcm) / speech.sample_rate
        await loop.run_in_executor(None, self.svc.player.wait, 120)
        self.last_announcement_interrupted = self.interaction != iid  # interrupt() ended it early
        self._emit(EventType.SPEECH_ENDED, {"interaction": iid})
        self._end_interaction(iid, "announced", echo_guard=True)
        return round(spoken, 2)


def _db(frame: np.ndarray) -> float:
    from jarvis.voice.vad import rms_dbfs  # noqa: PLC0415

    return rms_dbfs(frame)
