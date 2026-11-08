"""Voice service: owns speech recognition, synthesis, wake word, playback and the
continuous pipeline, and exposes them over the API.

Privacy: everything runs on this computer. Nothing is recorded while voice is
off; while it is on, audio stays on the device and only transcribed text reaches
the agent. Cloud speech providers are never used unless explicitly enabled.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from jarvis.core.events import EventType
from jarvis.core.observability import get_logger
from jarvis.core.permissions import Principal
from jarvis.voice.audio import MicSource, Player
from jarvis.voice.stt import FasterWhisperSTT
from jarvis.voice.tts import Speech, SpeechChunker, TTSCancelled, TTSUnavailable, VoiceChain, make_voice
from jarvis.voice.wakeword import OpenWakeWordDetector

log = get_logger("voice")
MAX_UPLOAD = 25 * 1024 * 1024


class SpeakBody(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class VoiceService:
    def __init__(self, runtime: Any, source_factory: Any = None, dry_run_audio: bool = False) -> None:
        self.runtime = runtime
        self.config = runtime.config.voice
        cfg = self.config
        self.stt = FasterWhisperSTT(cfg.stt_model)
        self.tts_model_dir = Path(cfg.tts_model_dir).expanduser() if cfg.tts_model_dir else runtime.paths.models / "tts" / "kokoro"
        # Building the voice is cheap: Kokoro loads later, on its own thread (start()).
        self.tts = make_voice(cfg, self.tts_model_dir, runtime.identity.get().voice, on_fallback=self._fallback_notice)
        self.wake = OpenWakeWordDetector(cfg.wake_word, cfg.wake_threshold) if cfg.wake_engine == "openwakeword" else None
        self._speaker_at = 0.0
        self.player = Player(cfg.output_device, on_level=self._speaker_level, dry_run=dry_run_audio)
        self.source_factory = source_factory or (lambda: MicSource(cfg.input_device))
        self.pipeline = None
        self._say_epoch = 0  # bumped by /voice/interrupt: a direct announcement in progress stops
        self.principal: Principal | None = None
        # Starting and stopping are serialized: two `voice start` requests arriving
        # together (CLI and browser, or a double click) must not open two listeners.
        self._lifecycle = asyncio.Lock()

    # ----------------------------------------------------------- lifecycle
    async def start(self) -> None:
        self.principal = self.runtime.auth.authenticate(self.runtime.local_token)
        if self.tts is not None and self.config.tts_preload:
            # In the background: Kokoro loads and warms up (~3-4 s) while JARVIS is already serving.
            threading.Thread(target=self.tts.start, name="jarvis-voice-preload", daemon=True).start()
        if self.config.enabled:
            try:
                await self.start_pipeline()
            except Exception as e:  # noqa: BLE001
                log.error("voice pipeline could not start: %s", e)

    async def stop(self) -> None:
        await self.stop_pipeline()

    async def start_pipeline(self) -> dict[str, Any]:
        from jarvis.voice.pipeline import VoicePipeline  # noqa: PLC0415

        ok, why = self.stt.dependency_ok()
        if not ok:
            raise RuntimeError(why)
        async with self._lifecycle:
            if self.pipeline is not None and self.pipeline.running:
                log.info("voice start requested while already listening; keeping the existing listener")
                return self.status()
            if self.principal is None:
                self.principal = self.runtime.auth.authenticate(self.runtime.local_token)
            # Warm the models before the microphone opens so the first command isn't slow.
            await asyncio.to_thread(self.stt._ensure)
            if self.wake is not None:
                await asyncio.to_thread(self.wake.load)
            if self.tts is not None:
                await asyncio.to_thread(self.tts.start)
            self.pipeline = VoicePipeline(self, self.source_factory)
            await asyncio.to_thread(self.pipeline.start, asyncio.get_running_loop())
        return self.status()

    async def stop_pipeline(self) -> dict[str, Any]:
        async with self._lifecycle:
            if self.pipeline is not None:
                await asyncio.to_thread(self.pipeline.stop)
        return self.status()

    # --------------------------------------------------------------- speech
    def synthesize(self, text: str, *, reply: str | None = None, should_stop: Callable[[], bool] | None = None) -> Speech:
        """Turn text into audio with JARVIS's voice (blocking; call from a worker thread).

        Raises TTSUnavailable when no voice can speak and TTSCancelled when ``should_stop``
        turned true. Plain engines (tests swap one in) are called as they are."""
        tts = self.tts
        if tts is None:
            raise TTSUnavailable("text-to-speech is disabled")
        if isinstance(tts, VoiceChain):
            return tts.synthesize(text, reply=reply, should_stop=should_stop)
        speech = tts.synthesize(text)
        speech.provider = speech.provider or getattr(tts, "name", "unknown")
        return speech

    async def _say_directly(self, text: str) -> dict[str, Any]:
        """Speak on the speakers while voice (the microphone) is off: chunk by chunk, so the
        first chunk plays while the rest is synthesized. /voice/interrupt stops it."""
        chunker = SpeechChunker()
        chunks = chunker.feed(text) + chunker.flush()
        epoch, rid = self._say_epoch, f"say_{time.monotonic_ns()}"
        spoken, provider, started = 0.0, None, False
        for chunk in chunks:
            try:
                speech = await asyncio.to_thread(self.synthesize, chunk, reply=rid, should_stop=lambda: self._say_epoch != epoch)
            except TTSCancelled:
                break
            except Exception as e:  # noqa: BLE001
                if not started:
                    raise  # nothing could be said: the caller reports it
                log.error("announcement cut short: %s", e)
                break
            if self._say_epoch != epoch:
                break
            if not started:
                started = True
                self.runtime.events.emit(EventType.SPEECH_STARTED, {"provider": speech.provider}, source="voice")
            log.info("speaking an announcement via %s (%.1f s of audio)", speech.provider, len(speech.pcm) / speech.sample_rate)
            provider = speech.provider
            self.player.enqueue(speech.pcm, speech.sample_rate)
            spoken += len(speech.pcm) / speech.sample_rate
        if started:
            await asyncio.to_thread(self.player.wait, 120)
            self.runtime.events.emit(EventType.SPEECH_ENDED, {}, source="voice")
        return {"spoken_s": round(spoken, 2), "provider": provider, "interrupted": self._say_epoch != epoch}

    def cancel_speech(self) -> None:
        """Stop a synthesis in progress (interrupt, stop); a no-op for engines that can't."""
        cancel = getattr(self.tts, "cancel", None)
        if cancel is not None:
            cancel()

    def _fallback_notice(self, engine: str, reason: str) -> None:
        """The primary voice couldn't speak and the fallback did: tell the person once per problem."""
        fallback = getattr(getattr(self.tts, "fallback", None), "name", "the backup voice")
        self.runtime.events.emit(EventType.NOTIFICATION, {
            "title": "Using the backup voice", "level": "warn",
            "body": f"The {engine} voice couldn't speak ({reason[:160]}); JARVIS is speaking with {fallback}."}, source="voice")

    def _tts_status(self) -> dict[str, Any]:
        tts = self.tts
        if tts is None:
            return {"available": False, "reason": "text-to-speech is disabled", "engine": None, "voice": None, "speaking_with": None}
        ok, why = tts.dependency_ok()
        if ok and getattr(tts, "error", None):
            ok, why = False, tts.error
        out: dict[str, Any] = {"available": ok, "reason": why or None, "engine": getattr(tts, "name", None),
                               "voice": getattr(tts, "voice", None), "speaking_with": getattr(tts, "name", None)}
        if isinstance(tts, VoiceChain):
            details = tts.status()
            out["details"] = details
            primary, fallback = details["primary"], details["fallback"]
            state = (primary or {}).get("state", "ready")  # an engine without a load state is ready when it exists
            if primary is None:
                out["speaking_with"] = fallback["engine"] if fallback else None
            elif state == "ready":
                out["speaking_with"] = primary["engine"]
                out["reason"] = primary.get("notice")
            elif state in ("idle", "loading"):
                out["speaking_with"] = f"{primary['engine']} (loading)"
                out["reason"] = f"{primary['engine']} is loading"
            else:
                out["speaking_with"] = fallback["engine"] if fallback and fallback["available"] else None
                out["reason"] = f"{primary['engine']} unavailable: {primary.get('error')}" + (
                    f"; speaking with {fallback['engine']}" if out["speaking_with"] else "")
        return out

    def _speaker_level(self, level: float) -> None:
        now = time.monotonic()
        if now - self._speaker_at >= 1 / 15 or level == 0.0:
            self._speaker_at = now
            self.runtime.events.emit(EventType.AUDIO_LEVEL, {"level": round(level, 3), "source": "speaker"}, source="voice")

    # -------------------------------------------------------------- status
    def _mic(self) -> dict[str, Any]:
        try:
            import sounddevice as sd  # noqa: PLC0415

            d = sd.query_devices(self.config.input_device, kind="input")
            return {"available": True, "name": d["name"]}
        except Exception as e:  # noqa: BLE001
            return {"available": False, "reason": str(e)}

    def status(self) -> dict[str, Any]:
        stt_ok, stt_why = self.stt.dependency_ok()
        if stt_ok and self.stt.load_error:
            stt_ok, stt_why = False, self.stt.load_error
        wake_ok, wake_why = (False, "wake word disabled") if self.wake is None else self.wake.dependency_ok()
        p = self.pipeline
        return {
            "stt": {"available": stt_ok, "reason": stt_why or None, "engine": self.stt.name, "model": self.stt.model_name, "loaded": self.stt.loaded},
            "tts": self._tts_status(),
            "wake": {"available": wake_ok, "reason": wake_why or None, "engine": getattr(self.wake, "name", None), "word": self.config.wake_word},
            "pipeline": {"running": bool(p and p.running), "state": p.state if p else "off", "error": p.error if p else None,
                         "stats": p.stats if p else None, "speaking": self.player.busy,
                         "interaction": p.interaction if p else None,
                         # Live listener threads: never more than one (a second would mean duplicate interactions).
                         "listeners": sum(1 for t in threading.enumerate() if t.name == "jarvis-voice" and t.is_alive()),
                         "barge_in": (self.config.barge_in_mode if self.config.barge_in else "off"),
                         "followup_window_s": self.config.conversation_window_s,
                         "signal": p.signal() if p else {"state": "off"}},
            "microphone": self._mic(),
            "cloud_audio": self.config.allow_cloud_audio,
        }

    def diagnostics(self) -> list[dict[str, Any]]:
        s = self.status()
        out = []
        for key, label in (("stt", "speech recognition"), ("wake", "wake word")):
            item = s[key]
            out.append({"name": f"voice:{label}", "status": "ok" if item["available"] else "warn",
                        "detail": f"{item.get('engine')} {item.get('model') or item.get('word') or ''}".strip()
                        if item["available"] else item["reason"], "fix": None if item["available"] else "pip install 'jarvis-core[voice]'",
                        "group": "voice"})
        t = s["tts"]
        primary = (t.get("details") or {}).get("primary") or {}
        degraded = bool(primary) and primary.get("state") in ("unavailable", "failed")
        out.append({"name": "voice:text-to-speech", "status": "warn" if degraded or not t["available"] else "ok",
                    "detail": f"{t['engine']} {t['voice'] or ''} - speaking with {t['speaking_with']}".strip()
                    + (f" ({t['reason']})" if t.get("reason") else ""),
                    "fix": ("jarvis voice install" if degraded and "not installed" in (primary.get("error") or "")
                            else None if t["available"] and not degraded else "pip install 'jarvis-core[voice]'"),
                    "group": "voice"})
        out.append({"name": "voice:pipeline", "status": "ok" if not s["pipeline"]["error"] else "fail",
                    "detail": f"{'listening for the wake word' if s['pipeline']['running'] else 'off'}" + (f"; {s['pipeline']['error']}" if s["pipeline"]["error"] else ""),
                    "fix": None, "group": "voice"})
        return out

    # ----------------------------------------------------------------- API
    @property
    def api_router(self) -> APIRouter:
        from jarvis.api.deps import interactive, owner  # noqa: PLC0415

        router = APIRouter(prefix="/api/v1/voice", tags=["voice"])

        @router.get("/status")
        async def status(p: Principal = Depends(interactive)) -> dict[str, Any]:
            return self.status()

        @router.post("/start")
        async def start(p: Principal = Depends(owner)) -> dict[str, Any]:
            try:
                return await self.start_pipeline()
            except Exception as e:  # noqa: BLE001
                raise HTTPException(503, f"voice could not start: {e}") from e

        @router.post("/stop")
        async def stop(p: Principal = Depends(owner)) -> dict[str, Any]:
            return await self.stop_pipeline()

        @router.post("/interrupt")
        async def interrupt(p: Principal = Depends(interactive)) -> dict[str, Any]:
            self._say_epoch += 1  # a direct announcement (voice off) stops queueing its chunks
            if self.pipeline is not None:
                await asyncio.to_thread(self.pipeline.interrupt, False, f"stop requested by {p.name}")
            else:
                self.player.stop()
                self.cancel_speech()
            return {"ok": True}

        @router.post("/transcribe")
        async def transcribe(request: Request, p: Principal = Depends(interactive)) -> dict[str, Any]:
            data = await request.body()
            if not data:
                raise HTTPException(422, "no audio received")
            if len(data) > MAX_UPLOAD:
                raise HTTPException(413, "audio too large")
            ok, why = self.stt.dependency_ok()
            if not ok:
                raise HTTPException(503, why)
            try:
                t = await asyncio.to_thread(self.stt.transcribe_bytes, data)
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            except Exception as e:  # noqa: BLE001
                raise HTTPException(503, f"speech recognition failed: {e}") from e
            self.runtime.events.emit(EventType.TRANSCRIPT, {"text": t.text, "latency_ms": t.latency_ms, "source": p.name}, source="voice")
            return {"text": t.text, "duration_s": t.duration_s, "latency_ms": t.latency_ms}

        @router.post("/speak")
        async def speak(body: SpeakBody, p: Principal = Depends(interactive)) -> Response:
            """Audio of ``text`` in JARVIS's voice, for a client to play (the browser speaks a reply chunk by chunk)."""
            if self.tts is None:
                raise HTTPException(503, "text-to-speech is disabled")
            try:
                speech = await asyncio.to_thread(self.synthesize, body.text)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(503, f"text-to-speech failed: {e}") from e
            return Response(speech.wav_bytes(), media_type="audio/wav",
                            headers={"Cache-Control": "no-store", "X-JARVIS-Voice": speech.provider})

        @router.post("/say")
        async def say(body: SpeakBody, p: Principal = Depends(owner)) -> dict[str, Any]:
            """Speak on this computer's speakers."""
            if self.tts is None:
                raise HTTPException(503, "text-to-speech is disabled")
            pipeline = self.pipeline
            if pipeline is not None and pipeline.running:
                # The microphone is open: speak through the pipeline so JARVIS can't hear itself.
                try:
                    spoken = await pipeline.announce(body.text)
                    return {"spoken_s": spoken, "provider": pipeline.last_provider,
                            "interrupted": pipeline.last_announcement_interrupted}
                except TTSUnavailable as e:
                    raise HTTPException(503, f"text-to-speech failed: {e}") from e
                except RuntimeError as e:
                    raise HTTPException(409, str(e)) from e
            try:
                return await self._say_directly(body.text)
            except Exception as e:  # noqa: BLE001 - nothing could be said
                raise HTTPException(503, f"text-to-speech failed: {e}") from e

        @router.post("/tts/reload")
        async def tts_reload(p: Principal = Depends(owner)) -> dict[str, Any]:
            """Load the primary voice again, e.g. after `jarvis voice install` put its model files in place."""
            reload = getattr(getattr(self.tts, "primary", None), "reload", None)
            if reload is None:
                raise HTTPException(409, "the configured voice has nothing to reload")
            await asyncio.to_thread(reload)
            return self._tts_status()

        return router


def register(runtime: Any) -> None:
    runtime.extensions["voice"] = VoiceService(runtime)
