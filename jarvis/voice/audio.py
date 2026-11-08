"""Audio input and output.

MicSource delivers 30 ms float32 frames at 16 kHz from a microphone (opening
the device at its native rate and resampling when it can't do 16 kHz).
ArraySource delivers frames from memory for tests and offline processing.
Player plays queued PCM and can be stopped instantly (barge-in); it reports the
real RMS level of what it plays.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from jarvis.core.observability import get_logger
from jarvis.voice.vad import level01

log = get_logger("voice.audio")

RATE = 16000
FRAME = 480  # 30 ms


def resample(audio: np.ndarray, src: int, dst: int = RATE) -> np.ndarray:
    if src == dst or audio.size == 0:
        return audio.astype(np.float32)
    try:
        from math import gcd  # noqa: PLC0415

        from scipy.signal import resample_poly  # noqa: PLC0415

        g = gcd(src, dst)
        return resample_poly(audio, dst // g, src // g).astype(np.float32)
    except ImportError:
        n = int(round(len(audio) * dst / src))
        return np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)


class AudioSource(Protocol):
    def start(self, on_frame: Callable[[np.ndarray], None]) -> None: ...
    def stop(self) -> None: ...


class MicSource:
    def __init__(self, device: str | int | None = None) -> None:
        self.device = device
        self._stream: Any = None
        self.native_rate: int | None = None
        self.device_name: str | None = None

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        import sounddevice as sd  # noqa: PLC0415

        info = sd.query_devices(self.device, kind="input")
        self.device_name = info["name"]
        pending = np.zeros(0, dtype=np.float32)

        def make_cb(rate: int) -> Callable[..., None]:
            def cb(indata: np.ndarray, frames: int, t: Any, status: Any) -> None:
                nonlocal pending
                mono = indata[:, 0] if indata.ndim > 1 else indata
                data = resample(mono, rate) if rate != RATE else mono.astype(np.float32)
                pending = np.concatenate([pending, data])
                while len(pending) >= FRAME:
                    on_frame(pending[:FRAME].copy())
                    pending = pending[FRAME:]
            return cb

        try:
            self._stream = sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=FRAME,
                                          device=self.device, callback=make_cb(RATE))
            self.native_rate = RATE
        except Exception:  # noqa: BLE001 - device can't do 16 kHz; use its own rate
            rate = int(info["default_samplerate"])
            self._stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32", blocksize=int(rate * 0.03),
                                          device=self.device, callback=make_cb(rate))
            self.native_rate = rate
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None


class ArraySource:
    """Feeds pre-recorded 16 kHz audio as frames (tests, offline processing)."""

    def __init__(self, audio: np.ndarray, realtime: bool = False, tail_silence_s: float = 2.0) -> None:
        self.audio = np.concatenate([audio.astype(np.float32), np.zeros(int(RATE * tail_silence_s), np.float32)])
        self.realtime = realtime
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.finished = threading.Event()

    def start(self, on_frame: Callable[[np.ndarray], None]) -> None:
        def run() -> None:
            for i in range(0, len(self.audio) - FRAME + 1, FRAME):
                if self._stop.is_set():
                    break
                on_frame(self.audio[i:i + FRAME])
                if self.realtime:
                    time.sleep(FRAME / RATE)
                else:
                    time.sleep(0.0005)
            self.finished.set()

        self._thread = threading.Thread(target=run, name="jarvis-array-source", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


class Player:
    """Sequential, interruptible playback with real output-level reporting.

    Consecutive chunks of one reply play through one open output stream (it stays
    open while chunks keep coming, LINGER_S after the last), so there is no device
    reopen between sentences; the pauses between them are in the audio itself.
    stop() aborts the stream: what the sound device still holds is dropped, so
    speech stops at once instead of draining its buffer."""

    LINGER_S = 0.3

    def __init__(self, device: str | int | None = None, on_level: Callable[[float], None] | None = None,
                 dry_run: bool = False) -> None:
        self.device = device
        self.on_level = on_level
        self.dry_run = dry_run  # tests: consume audio without a sound device
        self._q: queue.Queue[tuple[np.ndarray, int] | None] = queue.Queue()
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._out: Any = None  # the open output stream (player thread only)
        self._out_rate = 0
        self.failures = 0
        self.last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name="jarvis-player", daemon=True)
        self._thread.start()
        self.played_s = 0.0

    @property
    def busy(self) -> bool:
        return self._busy.is_set() or not self._q.empty()

    def enqueue(self, pcm: np.ndarray, sample_rate: int) -> None:
        if pcm.size:
            self._q.put((pcm.astype(np.float32), sample_rate))

    def stop(self) -> None:
        """Stop immediately and drop anything queued."""
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass

    def wait(self, timeout: float | None = None) -> bool:
        end = None if timeout is None else time.time() + timeout
        while self.busy:
            if end is not None and time.time() > end:
                return False
            time.sleep(0.02)
        return True

    def _run(self) -> None:
        item = None
        while True:
            if item is None:
                item = self._q.get()
            if item is None:
                self._close_stream()
                return
            pcm, rate = item
            self._stop.clear()
            self._busy.set()
            try:
                self._play(pcm, rate)
            except Exception as e:  # noqa: BLE001 - reported to whoever is speaking; the player keeps working
                self.failures += 1
                self.last_error = f"{type(e).__name__}: {e}"
                log.exception("playback failed")
                self._close_stream(abort=True)
            finally:
                self._busy.clear()
                if self.on_level:
                    self.on_level(0.0)
            if self._stop.is_set():
                self._close_stream(abort=True)
            try:  # the next chunk of the same reply usually follows: keep the stream open for it
                item = self._q.get(timeout=self.LINGER_S)
            except queue.Empty:
                self._close_stream()
                item = None

    def _close_stream(self, abort: bool = False) -> None:
        out, self._out = self._out, None
        if out is None:
            return
        try:
            out.abort() if abort else out.stop()  # stop() plays what is buffered; abort() drops it
            out.close()
        except Exception:  # noqa: BLE001
            log.exception("closing the output stream failed")

    def _play(self, pcm: np.ndarray, rate: int) -> None:
        block = int(rate * 0.04)
        if self.dry_run:
            for i in range(0, len(pcm), block):
                if self._stop.is_set():
                    return
                if self.on_level:
                    self.on_level(level01(pcm[i:i + block]))
                self.played_s += len(pcm[i:i + block]) / rate
                time.sleep(0.001)
            return
        import sounddevice as sd  # noqa: PLC0415

        if self._out is None or self._out_rate != rate:
            self._close_stream()
            self._out = sd.OutputStream(samplerate=rate, channels=1, dtype="float32", device=self.device, blocksize=block)
            self._out.start()
            self._out_rate = rate
        out = self._out
        for i in range(0, len(pcm), block):
            if self._stop.is_set():
                return
            chunk = pcm[i:i + block]
            if self.on_level:
                self.on_level(level01(chunk))
            out.write(chunk.reshape(-1, 1))
            self.played_s += len(chunk) / rate
        if self._q.empty():
            # The last blocks are still in the device's buffer: let them play before reporting
            # idle, so "speech ended" (and the echo guard after it) match what is heard.
            self._stop.wait(min(0.3, float(getattr(out, "latency", 0.0) or 0.0)))
