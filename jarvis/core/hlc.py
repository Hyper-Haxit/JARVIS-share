"""Hybrid logical clock.

Timestamps are strings of the form ``<wall_ms:15>-<counter:05>-<node>`` that
sort lexicographically in causal order across nodes, which makes
last-writer-wins conflict resolution deterministic during synchronization.
"""

from __future__ import annotations

import threading
import time


class HybridClock:
    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        self._wall = 0
        self._counter = 0
        self._lock = threading.Lock()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def now(self) -> str:
        with self._lock:
            wall = self._now_ms()
            if wall > self._wall:
                self._wall, self._counter = wall, 0
            else:
                self._counter += 1
            return self.format(self._wall, self._counter, self.node_id)

    def observe(self, remote: str) -> None:
        """Advance the local clock past a timestamp received from a peer."""
        r_wall, r_counter, _ = self.parse(remote)
        with self._lock:
            wall = self._now_ms()
            if wall > self._wall and wall > r_wall:
                self._wall, self._counter = wall, 0
            elif r_wall > self._wall:
                self._wall, self._counter = r_wall, r_counter + 1
            elif r_wall == self._wall:
                self._counter = max(self._counter, r_counter) + 1
            else:
                self._counter += 1

    @staticmethod
    def format(wall: int, counter: int, node: str) -> str:
        return f"{wall:015d}-{counter:05d}-{node}"

    @staticmethod
    def parse(ts: str) -> tuple[int, int, str]:
        wall, counter, node = ts.split("-", 2)
        return int(wall), int(counter), node
