"""Voice activity detection.

An energy detector with an adaptive noise floor: the floor follows the quiet
level of the room, and speech is declared when the frame level rises a
configurable number of decibels above it for long enough. Silence ends an
utterance only after a hangover period so natural pauses don't cut speech.

The floor learns from quiet frames, and also from the quietest moments of the
last few seconds: when even those are above the speech threshold, the room
itself has become louder (a fan, music, a busy room), so the floor rises to
meet it. Without that, a steady background louder than the starting estimate
would count as endless speech and every request would run to the time limit.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np


def rms_dbfs(frame: np.ndarray) -> float:
    if frame.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(frame.astype(np.float32)))))
    return 20 * math.log10(max(rms, 1e-6))


def level01(frame: np.ndarray) -> float:
    """Perceptual 0..1 level for UI (maps -60..-10 dBFS)."""
    return max(0.0, min(1.0, (rms_dbfs(frame) + 60) / 50))


@dataclass
class VadEvent:
    kind: str  # "start" | "end"
    at_frame: int


class EnergyVAD:
    def __init__(self, sample_rate: int = 16000, frame_ms: int = 30, threshold_db: float = 12.0,
                 min_speech_ms: int = 150, silence_ms: int = 800, floor_init_db: float = -55.0) -> None:
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.frame_ms = frame_ms
        self.threshold_db = threshold_db
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.floor = floor_init_db
        self._recent: deque[float] = deque(maxlen=max(1, 2000 // frame_ms))  # the last ~2 s of frame levels
        self.in_speech = False
        self._voiced = 0
        self._quiet = 0
        self._n = 0
        self.last_db = -120.0

    def reset(self) -> None:
        self.in_speech = False
        self._voiced = 0
        self._quiet = 0

    def is_voiced(self, db: float, extra_margin_db: float = 0.0) -> bool:
        return db > self.floor + self.threshold_db + extra_margin_db

    def process(self, frame: np.ndarray, extra_margin_db: float = 0.0) -> VadEvent | None:
        """Feed one frame; returns a start/end event when the state changes."""
        self._n += 1
        db = rms_dbfs(frame)
        self.last_db = db
        voiced = self.is_voiced(db, extra_margin_db)
        self._recent.append(db)
        if not voiced:
            # Track the floor quickly downwards, slowly upwards.
            alpha = 0.2 if db < self.floor else 0.02
            self.floor = (1 - alpha) * self.floor + alpha * max(db, -90.0)
        elif len(self._recent) == self._recent.maxlen:
            quietest = min(self._recent)
            if quietest > self.floor + self.threshold_db:
                # Not one quiet moment in 2 s: the background is louder than the floor estimate.
                self.floor += 0.08 * (min(quietest, -10.0) - self.floor)
        if not self.in_speech:
            self._voiced = self._voiced + 1 if voiced else 0
            if self._voiced >= self.min_speech_frames:
                self.in_speech = True
                self._quiet = 0
                return VadEvent("start", self._n - self._voiced)
        else:
            self._quiet = 0 if voiced else self._quiet + 1
            if self._quiet >= self.silence_frames:
                self.in_speech = False
                self._voiced = 0
                return VadEvent("end", self._n)
        return None
