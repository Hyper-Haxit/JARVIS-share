"""Kokoro: JARVIS's primary voice, a local neural text-to-speech model.

Kokoro-82M v1.0, full precision (fp32), run by ONNX Runtime on the CPU through the
kokoro-onnx package; 24 kHz mono. Nothing leaves the computer.

Model files are not part of JARVIS and are never downloaded at startup. They are
installed explicitly (``jarvis voice install``) into one place:

    <JARVIS home>/models/tts/kokoro/kokoro-v1.0.onnx   325,505,369 bytes
    <JARVIS home>/models/tts/kokoro/voices-v1.0.bin     28,214,398 bytes

(``[voice] tts_model_dir`` overrides the folder.) The files are pinned to the
``model-files-v1.1`` release of thewh1teagle/kokoro-onnx and checked against their
published SHA-256 digests. The int8 model of the same release is deliberately not
used: on this class of CPU it was 10-40x slower than fp32 and audibly different.

Runtime settings, each measured in the voice benchmark
(.jarvis-dev/kokoro-bench/results/SUMMARY.md, 6-core Ryzen, other programs busy):
  * ONNX Runtime intra-op spinning off ("session.intra_op.allow_spinning" = "0"):
    idle worker threads sleep instead of busy-waiting, ~30% faster synthesis when
    other programs compete for the CPU.
  * CPU memory arena left on (the ONNX Runtime default): turning it off halves
    memory (~1.06 GB -> ~0.55 GB private) but made synthesis 1.76x slower.
    Memory levels off after ~50 syntheses (200-synthesis soak: +1 MB over the last 100).
  * Above-normal process priority while (and only while) a chunk is synthesized
    (Windows): with 4 cores busy plus the local LLM generating, first speech took
    a median 0.62 s instead of 1.99 s, with no gaps instead of 14. The previous
    priority is restored right after; a higher priority set by the user is kept.

Lifecycle: start() loads the model on a background thread and warms it up (the
first synthesis pays ~1.2 s of one-time pronunciation start-up), so JARVIS's own
startup never waits. (Creating the ONNX Runtime session does hold Python's GIL for
~1.2 s, measured: other requests pause once, a few seconds after startup.
Synthesis itself releases it.) synthesize() waits for loading for at most ``init_wait_s``.
The loaded model is kept for the life of the process. A synthesis can be cancelled
mid-way: ONNX Runtime stops at the next operator when its run is terminated.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from jarvis.core.observability import get_logger
from jarvis.voice.tts import Speech, TTSCancelled, TTSUnavailable, prepare_text

log = get_logger("voice.kokoro")

SAMPLE_RATE = 24000
DEFAULT_VOICE = "af_heart"
RELEASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/"
# Silence after a chunk, by how it ends. These are kokoro-onnx's own pauses after a
# sentence and after a clause inside one call; it trims them off the end of a call.
SENTENCE_PAUSE_S = 0.25
CLAUSE_PAUSE_S = 0.1
RETRY_FAILED_LOAD_S = 120.0


@dataclass(frozen=True)
class Asset:
    name: str
    size: int
    sha256: str


MODEL = Asset("kokoro-v1.0.onnx", 325_505_369, "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a")
VOICES = Asset("voices-v1.0.bin", 28_214_398, "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d")
ASSETS = (MODEL, VOICES)


def lang_for(voice: str) -> str:
    """espeak-ng language for a Kokoro voice id ("af_heart": American, "bm_george": British)."""
    return "en-gb" if voice[:1] == "b" else "en-us"


def missing_assets(model_dir: Path) -> list[str]:
    """Files that are absent or the wrong size (a cheap check, done on every load)."""
    out = []
    for a in ASSETS:
        p = model_dir / a.name
        if not p.is_file():
            out.append(f"{a.name} (missing)")
        elif p.stat().st_size != a.size:
            out.append(f"{a.name} (wrong size: {p.stat().st_size} bytes, expected {a.size})")
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def install_assets(model_dir: Path, source_dir: Path | None = None,
                   progress: Callable[[str], None] = print) -> list[str]:
    """Put the model files in ``model_dir``, verified by SHA-256. Explicit user action only.

    With ``source_dir``, files are taken from there (hard-linked when on the same
    drive, so nothing is duplicated; copied otherwise); without it they are
    downloaded from the pinned release. Returns what was done, per file."""
    model_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for a in ASSETS:
        dest = model_dir / a.name
        if dest.is_file() and dest.stat().st_size == a.size and _sha256(dest) == a.sha256:
            done.append(f"{a.name}: already installed")
            progress(done[-1])
            continue
        part = dest.with_name(dest.name + ".part")
        part.unlink(missing_ok=True)
        if source_dir is not None:
            src = Path(source_dir) / a.name
            if not src.is_file():
                raise FileNotFoundError(f"{src} not found")
            try:
                os.link(src, part)
                how = "linked from"
            except OSError:
                shutil.copy2(src, part)
                how = "copied from"
            how = f"{how} {src}"
        else:
            progress(f"{a.name}: downloading {a.size / 1e6:.0f} MB from {RELEASE_URL}")
            urllib.request.urlretrieve(RELEASE_URL + a.name, part)
            how = "downloaded"
        got_size, got_sha = part.stat().st_size, _sha256(part)
        if got_size != a.size or got_sha != a.sha256:
            part.unlink(missing_ok=True)
            raise ValueError(f"{a.name}: checksum mismatch (size {got_size}, sha256 {got_sha}); nothing installed")
        dest.unlink(missing_ok=True)
        part.replace(dest)
        done.append(f"{a.name}: {how}, sha256 verified")
        progress(done[-1])
    return done


class _RunGuard:
    """The ONNX Runtime session as kokoro-onnx sees it, with a way to stop a run mid-way.

    kokoro-onnx calls ``session.run(None, inputs)``; each call here runs with its own
    RunOptions, whose ``terminate`` flag cancel() sets. ONNX Runtime checks the flag
    between operators, so an interrupted synthesis frees the CPU within milliseconds
    instead of finishing a sentence nobody will hear."""

    def __init__(self, session: Any) -> None:
        self._session = session
        self._lock = threading.Lock()
        self._current: Any = None
        self.cancelled = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)

    def run(self, output_names: Any, input_feed: Any, run_options: Any = None) -> Any:
        import onnxruntime as rt  # noqa: PLC0415

        opts = rt.RunOptions()
        with self._lock:
            if self.cancelled:
                raise TTSCancelled("synthesis cancelled")
            self._current = opts
        try:
            return self._session.run(output_names, input_feed, opts)
        except Exception:
            if self.cancelled:
                raise TTSCancelled("synthesis cancelled") from None
            raise
        finally:
            with self._lock:
                self._current = None

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True
            if self._current is not None:
                self._current.terminate = True

    def reset(self) -> None:
        with self._lock:
            self.cancelled = False


class _Watch:
    """While a chunk is synthesized: cancel it when ``should_stop`` turns true or the deadline passes."""

    def __init__(self, guard: _RunGuard, should_stop: Callable[[], bool] | None, deadline: float) -> None:
        self.guard, self.should_stop, self.deadline = guard, should_stop, deadline
        self.timed_out = False
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="jarvis-kokoro-watch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._done.wait(0.02):
            if self.should_stop is not None and self.should_stop():
                self.guard.cancel()
                return
            if time.monotonic() > self.deadline:
                self.timed_out = True
                self.guard.cancel()
                return

    def stop(self) -> None:
        self._done.set()
        self._thread.join(timeout=1)


class _PriorityBoost:
    """Above-normal process priority for the duration of one synthesis (Windows only).

    Only raised from normal or lower, and restored afterwards, so a priority the
    user chose (e.g. high) is left alone."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and sys.platform == "win32"
        self._previous: int | None = None

    def __enter__(self) -> _PriorityBoost:
        if self.enabled:
            try:
                import psutil  # noqa: PLC0415

                proc = psutil.Process()
                current = proc.nice()
                if current in (psutil.NORMAL_PRIORITY_CLASS, psutil.BELOW_NORMAL_PRIORITY_CLASS, psutil.IDLE_PRIORITY_CLASS):
                    proc.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
                    self._previous = current
            except Exception as e:  # noqa: BLE001 - speech works without the boost, just slower under load
                log.debug("could not raise priority for speech: %s", e)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._previous is not None:
            try:
                import psutil  # noqa: PLC0415

                psutil.Process().nice(self._previous)
            except Exception as e:  # noqa: BLE001
                log.warning("could not restore process priority after speech: %s", e)


class KokoroTTS:
    """The Kokoro engine. Thread-safe: syntheses run one at a time."""

    name = "kokoro"
    cancellable = True  # synthesize() takes should_stop and stops mid-way

    def __init__(self, model_dir: Path, voice: str = DEFAULT_VOICE, speed: float = 1.0,
                 boost_priority: bool = True, init_wait_s: float = 12.0) -> None:
        self.model_dir = Path(model_dir)
        self.voice = voice or DEFAULT_VOICE
        self.requested_voice = self.voice
        self.speed = speed
        self.boost_priority = boost_priority
        self.init_wait_s = init_wait_s
        # idle -> loading -> ready; or unavailable (dependency/files missing) / failed (error while loading)
        self.state = "idle"
        self.error: str | None = None
        self.notice: str | None = None  # e.g. the configured voice doesn't exist and the default is used
        self.voices: list[str] = []
        self.load_ms: float | None = None
        self.warm_ms: float | None = None
        self.stats: dict[str, Any] = {"utterances": 0, "cancelled": 0, "errors": 0, "last_ms": None, "last_rtf": None,
                                      "audio_s": 0.0, "synth_s": 0.0}
        self._kokoro: Any = None
        self._guard: _RunGuard | None = None
        self._loaded = threading.Event()  # loading finished, successfully or not
        self._lock = threading.Lock()     # state changes
        self._synth = threading.Lock()    # one synthesis at a time
        self._failed_at = 0.0

    # ----------------------------------------------------------- availability
    @staticmethod
    def dependency_ok() -> tuple[bool, str]:
        """Are the packages installed? Checked without importing them: status reads run on the
        server's event loop, and an import racing the background loader would block it."""
        import importlib.util  # noqa: PLC0415

        for module in ("kokoro_onnx", "onnxruntime"):
            if importlib.util.find_spec(module) is None:
                return False, f"kokoro-onnx is not installed ({module}); pip install 'jarvis-core[voice]'"
        return True, ""

    def start(self) -> None:
        """Begin loading in the background (no-op when loading or loaded). Never blocks for the load."""
        with self._lock:
            if self.state in ("loading", "ready"):
                return
            if self.state == "failed" and time.monotonic() - self._failed_at < RETRY_FAILED_LOAD_S:
                return
            ok, why = self.dependency_ok()
            missing = missing_assets(self.model_dir)
            if not ok or missing:
                reason = why or (f"model files not installed in {self.model_dir}: {', '.join(missing)}. "
                                 "Install them with `jarvis voice install`")
                if (self.state, self.error) != ("unavailable", reason):
                    log.warning("Kokoro voice unavailable: %s", reason)
                self.state, self.error = "unavailable", reason
                self._loaded.set()
                return
            self.state, self.error = "loading", None
            self._loaded.clear()
            threading.Thread(target=self._load, name="jarvis-kokoro-load", daemon=True).start()

    def reload(self) -> None:
        """Load again (after installing the files or changing the voice); the old model is dropped."""
        with self._lock:
            if self.state == "loading":
                return
            self.state, self.error, self._failed_at = "idle", None, 0.0
            self._kokoro = self._guard = None
        self.start()

    def _load(self) -> None:
        started = time.perf_counter()
        try:
            import onnxruntime as rt  # noqa: PLC0415
            from kokoro_onnx import Kokoro  # noqa: PLC0415

            options = rt.SessionOptions()
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")  # see the module docstring
            session = rt.InferenceSession(str(self.model_dir / MODEL.name), sess_options=options,
                                          providers=["CPUExecutionProvider"])
            guard = _RunGuard(session)
            kokoro = Kokoro.from_session(guard, str(self.model_dir / VOICES.name))
            voices = kokoro.get_voices()
            notice = None
            if self.requested_voice not in voices:
                notice = (f"voice {self.requested_voice!r} is not a Kokoro voice; using {DEFAULT_VOICE}. "
                          f"Available: {', '.join(voices)}")
                log.error("Kokoro: %s", notice)
                self.voice = DEFAULT_VOICE
            load_ms = (time.perf_counter() - started) * 1000
            warm = time.perf_counter()
            with self._synth, _PriorityBoost(self.boost_priority):
                kokoro.create("Okay.", voice=self.voice, speed=self.speed, lang=lang_for(self.voice))
            warm_ms = (time.perf_counter() - warm) * 1000
        except Exception as e:  # noqa: BLE001 - reported; the fallback voice takes over
            with self._lock:
                self.state, self.error = "failed", f"could not load: {type(e).__name__}: {e}"
                self._failed_at = time.monotonic()
            log.error("Kokoro voice failed to load (%s); will retry in %.0f s", e, RETRY_FAILED_LOAD_S)
            self._loaded.set()
            return
        with self._lock:
            self._kokoro, self._guard, self.voices, self.notice = kokoro, guard, voices, notice
            self.load_ms, self.warm_ms = round(load_ms), round(warm_ms)
            self.state, self.error = "ready", None
        log.info("Kokoro voice ready: %s (%s), model loaded in %.0f ms, warmed up in %.0f ms", self.voice,
                 lang_for(self.voice), load_ms, warm_ms)
        self._loaded.set()

    def _wait_until_loaded(self, should_stop: Callable[[], bool] | None) -> None:
        if self.state in ("idle", "unavailable", "failed"):
            self.start()  # files may have been installed since; a failed load is retried after a while
        if self.state == "loading":
            deadline = time.monotonic() + self.init_wait_s
            while not self._loaded.wait(0.05):
                if should_stop is not None and should_stop():
                    raise TTSCancelled("cancelled while the voice was loading")
                if time.monotonic() > deadline:
                    raise TTSUnavailable(f"still loading after {self.init_wait_s:.0f} s")
        if self.state != "ready":
            raise TTSUnavailable(self.error or f"not available ({self.state})")

    # -------------------------------------------------------------- speaking
    def synthesize(self, text: str, timeout: float = 60, should_stop: Callable[[], bool] | None = None) -> Speech:
        text = prepare_text(text)
        if not text:
            return Speech(np.zeros(0, np.float32), SAMPLE_RATE, "", 0.0, provider=self.name)
        self._wait_until_loaded(should_stop)
        with self._synth:
            kokoro, guard = self._kokoro, self._guard
            if kokoro is None or guard is None:
                raise TTSUnavailable("the model was unloaded")
            if should_stop is not None and should_stop():
                raise TTSCancelled("cancelled")
            guard.reset()
            started = time.perf_counter()
            watch = _Watch(guard, should_stop, time.monotonic() + timeout)
            try:
                with _PriorityBoost(self.boost_priority):
                    audio, rate = kokoro.create(text, voice=self.voice, speed=self.speed, lang=lang_for(self.voice))
            except TTSCancelled:
                self.stats["cancelled"] += 1
                if watch.timed_out:
                    raise TimeoutError(f"synthesis took longer than {timeout:.0f} s") from None
                raise
            except Exception as e:
                self.stats["errors"] += 1
                raise RuntimeError(f"synthesis failed: {type(e).__name__}: {e}") from e
            finally:
                watch.stop()
            elapsed = time.perf_counter() - started
        pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
        end = text.rstrip()[-1:]
        pause = SENTENCE_PAUSE_S if end in ".!?" else CLAUSE_PAUSE_S if end in ",;:—–-" else 0.0
        if pause:
            pcm = np.concatenate([pcm, np.zeros(int(rate * pause), np.float32)])
        dur = len(pcm) / rate
        s = self.stats
        s["utterances"] += 1
        s["last_ms"] = round(elapsed * 1000, 1)
        s["last_rtf"] = round(elapsed / dur, 3) if dur else None
        s["audio_s"] = round(s["audio_s"] + dur, 2)
        s["synth_s"] = round(s["synth_s"] + elapsed, 2)
        return Speech(pcm, rate, text, round(elapsed * 1000, 1), provider=self.name)

    def cancel(self) -> None:
        """Stop the synthesis in progress, if any. Every synthesis clears the flag when it starts."""
        guard = self._guard
        if guard is not None:
            guard.cancel()

    def stop(self) -> None:
        """The model stays loaded for the life of the process (loading it again costs seconds)."""

    def status(self) -> dict[str, Any]:
        return {"engine": self.name, "voice": self.voice, "state": self.state, "error": self.error, "notice": self.notice,
                "model_dir": str(self.model_dir), "model": MODEL.name, "precision": "fp32", "sample_rate": SAMPLE_RATE,
                "load_ms": self.load_ms, "warm_ms": self.warm_ms, "speed": self.speed,
                "priority_boost": self.boost_priority and sys.platform == "win32", **self.stats}
