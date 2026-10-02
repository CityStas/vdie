"""Rolling history buffer.

The history is the memory of the engine.  Every temporal decision — gesture
confirmation, velocity peaks, micro-pauses, commit detection, trajectory
prediction — reads from here instead of keeping its own ad-hoc state.  That is
what makes record/replay produce *identical* results to live operation: the
same buffer is rebuilt from the recording, frame by frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..types import HandFeatures, MotionState


@dataclass(slots=True)
class FrameSample:
    """One entry of the history buffer."""

    timestamp: float
    frame_id: int
    point: np.ndarray  # control point (2,), image-normalized
    motion: MotionState
    features: HandFeatures | None = None
    hand_confidence: float = 0.0
    extra: dict = field(default_factory=dict)

    def position(self) -> np.ndarray:
        return self.point


class HistoryBuffer:
    """Fixed-capacity ring buffer with time-window queries."""

    def __init__(self, capacity: int = 90) -> None:
        self.capacity = int(capacity)
        self._buf: deque[FrameSample] = deque(maxlen=self.capacity)

    # -- basic ------------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._buf)

    @property
    def empty(self) -> bool:
        return not self._buf

    def append(self, sample: FrameSample) -> None:
        self._buf.append(sample)

    def clear(self) -> None:
        self._buf.clear()

    def last(self, n: int = 1) -> list[FrameSample]:
        if n <= 0:
            return []
        return list(self._buf)[-n:]

    def latest(self) -> FrameSample | None:
        return self._buf[-1] if self._buf else None

    def all(self) -> list[FrameSample]:
        return list(self._buf)

    # -- time queries ------------------------------------------------------ #
    def window(self, seconds: float, now: float | None = None) -> list[FrameSample]:
        """Samples inside ``[now - seconds, now]``.

        ``now`` defaults to the newest sample.  The upper bound matters: without
        it, querying a *past* instant still returns samples that had not happened
        yet, which silently corrupts any offline/replay analysis.
        """
        if not self._buf:
            return []
        t_now = self._buf[-1].timestamp if now is None else now
        cutoff = t_now - seconds
        return [s for s in self._buf if cutoff <= s.timestamp <= t_now]

    def positions(self, seconds: float, now: float | None = None) -> np.ndarray:
        w = self.window(seconds, now)
        if not w:
            return np.zeros((0, 2))
        return np.asarray([s.point for s in w], dtype=np.float64)

    def speeds(self, seconds: float, now: float | None = None) -> np.ndarray:
        return np.asarray([s.motion.speed for s in self.window(seconds, now)], dtype=np.float64)

    def timestamps(self, seconds: float, now: float | None = None) -> np.ndarray:
        return np.asarray([s.timestamp for s in self.window(seconds, now)], dtype=np.float64)

    def path_length(self, seconds: float, now: float | None = None) -> float:
        """Total distance travelled by the control point inside the window.

        Distinct from displacement on purpose: this is what separates a *whole
        hand* movement from a finger articulation.  Closing a pinch moves the
        index fingertip (the control point) by a couple of centimetres at high
        speed, which is indistinguishable from a flick by speed alone.
        """
        w = self.window(seconds, now)
        if len(w) < 2:
            return 0.0
        pts = np.asarray([s.point for s in w], dtype=np.float64)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())

    # -- derived statistics ------------------------------------------------ #
    def speed_peak(self, seconds: float, now: float | None = None) -> tuple[float, float]:
        """``(peak_speed, seconds_since_peak)`` inside the window.

        ``seconds_since_peak`` is measured against ``now`` (the newest sample by
        default), so it stays meaningful when the peak is older than the window
        edge instead of being clamped to zero.
        """
        w = self.window(seconds, now)
        if not w:
            return 0.0, 0.0
        speeds = np.asarray([s.motion.speed for s in w], dtype=np.float64)
        i = int(np.argmax(speeds))
        t_now = self._buf[-1].timestamp if now is None else now
        return float(speeds[i]), float(t_now - w[i].timestamp)

    def last_stable(self, max_speed: float, search_seconds: float = 1.0) -> FrameSample | None:
        """Most recent sample whose speed was below ``max_speed`` (re-anchor)."""
        for s in reversed(self.window(search_seconds)):
            if s.motion.speed <= max_speed:
                return s
        return None

    def sample_at_or_before(self, timestamp: float) -> FrameSample | None:
        best: FrameSample | None = None
        for s in self._buf:
            if s.timestamp <= timestamp:
                best = s
            else:
                break
        return best

    def direction_changes(self, seconds: float, threshold_rad: float) -> int:
        w = self.window(seconds)
        if len(w) < 3:
            return 0
        count = 0
        for i in range(2, len(w)):
            if abs(w[i].motion.direction_change) >= threshold_rad:
                count += 1
        return count
