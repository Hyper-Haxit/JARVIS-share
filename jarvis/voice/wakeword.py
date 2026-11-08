"""Wake-word detection, run locally.

OpenWakeWordDetector uses the openWakeWord "hey_jarvis" ONNX model on 80 ms
chunks of 16 kHz audio; nothing is sent anywhere while it listens.
"""

from __future__ import annotations

import numpy as np

from jarvis.core.observability import get_logger

log = get_logger("voice.wake")

CHUNK = 1280  # 80 ms at 16 kHz, the model's native step


class WakeUnavailable(RuntimeError):
    pass


class OpenWakeWordDetector:
    name = "openwakeword"

    def __init__(self, model: str = "hey_jarvis", threshold: float = 0.5) -> None:
        self.model_name = model
        self.threshold = threshold
        self._model = None
        self._buf = np.zeros(0, dtype=np.int16)
        self.last_score = 0.0

    @staticmethod
    def dependency_ok() -> tuple[bool, str]:
        try:
            import openwakeword  # noqa: F401, PLC0415
            return True, ""
        except ImportError:
            return False, "openwakeword is not installed (pip install 'jarvis-core[voice]')"

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from openwakeword.model import Model  # noqa: PLC0415

            try:
                self._model = Model(wakeword_models=[self.model_name], inference_framework="onnx")
            except Exception:  # noqa: BLE001 - models not downloaded yet
                from openwakeword import utils  # noqa: PLC0415

                utils.download_models(model_names=[self.model_name])
                self._model = Model(wakeword_models=[self.model_name], inference_framework="onnx")
        except Exception as e:  # noqa: BLE001
            raise WakeUnavailable(f"wake-word model '{self.model_name}' unavailable: {e}") from e

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.int16)
        if self._model is not None:
            self._model.reset()
        self.last_score = 0.0

    def process(self, frame: np.ndarray) -> float:
        """Feed float32 16 kHz audio; returns the best score seen in this call."""
        self.load()
        pcm = (np.clip(frame, -1, 1) * 32767).astype(np.int16)
        self._buf = np.concatenate([self._buf, pcm])
        best = 0.0
        while len(self._buf) >= CHUNK:
            scores = self._model.predict(self._buf[:CHUNK])  # type: ignore[union-attr]
            self._buf = self._buf[CHUNK:]
            best = max(best, max(scores.values()) if scores else 0.0)
        self.last_score = best if best else self.last_score * 0.9
        return best

    def detected(self, score: float) -> bool:
        return score >= self.threshold
