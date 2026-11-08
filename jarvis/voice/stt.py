"""Speech recognition.

Default: faster-whisper running locally (audio never leaves the device).
The model loads lazily on first use and is kept warm afterwards.
"""

from __future__ import annotations

import io
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from jarvis.core.observability import get_logger

log = get_logger("voice.stt")


@dataclass
class Transcript:
    text: str
    language: str | None
    duration_s: float
    latency_ms: float
    no_speech_prob: float | None = None


class STTUnavailable(RuntimeError):
    pass


class FasterWhisperSTT:
    name = "faster-whisper"

    def __init__(self, model: str = "base.en", download_root: str | None = None, device: str = "cpu",
                 compute_type: str = "int8") -> None:
        self.model_name = model
        self.download_root = download_root
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None
        self._lock = threading.Lock()
        self.load_error: str | None = None

    @staticmethod
    def dependency_ok() -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401, PLC0415
            return True, ""
        except ImportError:
            return False, "faster-whisper is not installed (pip install 'jarvis-core[voice]')"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _ensure(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel  # noqa: PLC0415

                    started = time.perf_counter()
                    import os  # noqa: PLC0415

                    self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute_type,
                                               download_root=self.download_root, cpu_threads=max(1, (os.cpu_count() or 4) - 1))
                    log.info("speech model %s loaded in %.1fs", self.model_name, time.perf_counter() - started)
                    self.load_error = None
                except Exception as e:  # noqa: BLE001
                    self.load_error = f"{type(e).__name__}: {e}"
                    raise STTUnavailable(f"could not load speech model {self.model_name}: {e}") from e
        return self._model

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000, language: str | None = "en") -> Transcript:
        """Transcribe mono float32 audio at 16 kHz."""
        if sample_rate != 16000:
            raise ValueError("audio must be 16 kHz")
        model = self._ensure()
        started = time.perf_counter()
        segments, info = model.transcribe(audio.astype(np.float32), language=language, beam_size=1, vad_filter=False,
                                          condition_on_previous_text=False)
        segs = list(segments)
        text = " ".join(s.text.strip() for s in segs).strip()
        nsp = min((s.no_speech_prob for s in segs), default=None)
        return Transcript(text=text, language=info.language, duration_s=round(len(audio) / 16000, 2),
                          latency_ms=round((time.perf_counter() - started) * 1000, 1), no_speech_prob=nsp)

    def transcribe_bytes(self, data: bytes) -> Transcript:
        """Transcribe an encoded audio file (wav, webm/opus, ogg, mp3, m4a...)."""
        return self.transcribe(decode_audio(data))


def decode_audio(data: bytes, rate: int = 16000) -> np.ndarray:
    """Decode any common audio container to mono float32 at ``rate``.

    WAV uses the standard library; everything else uses PyAV directly (the
    helper bundled with faster-whisper passes options newer PyAV rejects).
    """
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        import wave  # noqa: PLC0415

        from jarvis.voice.audio import resample  # noqa: PLC0415

        try:
            with wave.open(io.BytesIO(data)) as w:
                sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
                raw = w.readframes(w.getnframes())
        except wave.Error as e:
            raise ValueError(f"could not decode the audio: {e}") from e
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}.get(width)
        if dtype is None:
            raise ValueError(f"unsupported WAV sample width {width}")
        pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32)
        pcm = (pcm - 128) / 128 if width == 1 else pcm / float(np.iinfo(dtype).max)
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1)
        return resample(pcm, sr, rate)
    try:
        import av  # noqa: PLC0415

        chunks = []
        with av.open(io.BytesIO(data), mode="r") as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise ValueError("no audio stream found")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):
                chunks.append(out.to_ndarray().reshape(-1))
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"could not decode the audio: {e}") from e
    if not chunks:
        raise ValueError("the audio contained no samples")
    return (np.concatenate(chunks).astype(np.float32) / 32768.0)
