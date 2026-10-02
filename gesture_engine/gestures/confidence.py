"""Confidence, stability, temporal voting and hysteresis.

Three separate mechanisms, deliberately not merged:

* **temporal voting** — a gesture is only accepted if it wins in M of the last N
  frames.  Kills single-frame false positives.
* **hysteresis** — entering and leaving a gesture use different thresholds, so
  the state cannot oscillate on a noisy boundary.
* **stability** — the fraction of the vote window that agreed with the emitted
  gesture; a second, independent quality signal that the action policy consumes.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass


@dataclass
class VoteBuffer:
    """Sliding window of per-frame gesture candidates."""

    window: int = 10
    required: int = 7

    def __post_init__(self) -> None:
        self._votes: deque[str] = deque(maxlen=max(self.window, 1))

    def reset(self) -> None:
        self._votes.clear()

    def push(self, gesture: str) -> None:
        self._votes.append(gesture)

    @property
    def size(self) -> int:
        return len(self._votes)

    def agreement(self, gesture: str) -> float:
        if not self._votes:
            return 0.0
        return sum(1 for v in self._votes if v == gesture) / len(self._votes)

    def winner(self) -> tuple[str, int]:
        if not self._votes:
            return "NONE", 0
        counts = Counter(self._votes)
        name, count = counts.most_common(1)[0]
        return name, count

    def confirmed(self) -> tuple[str, float] | None:
        """Return ``(gesture, agreement)`` once the vote threshold is reached."""
        name, count = self.winner()
        if name != "NONE" and count >= self.required:
            return name, count / max(len(self._votes), 1)
        return None

    def confirmed_tail(self, window: int, required: int) -> tuple[str, float] | None:
        """M-of-N confirmation over only the *last* ``window`` votes.

        Dynamic gestures need this.  A swipe is a ~12-frame pattern at 30 FPS and
        its score only saturates near the end of the stroke, so demanding 7 of the
        last 10 votes is arithmetically impossible — the pattern is already
        temporally confirmed by construction, and voting on top of it is
        double-counting temporal evidence.
        """
        tail = list(self._votes)[-max(window, 1) :]
        if not tail:
            return None
        counts = Counter(tail)
        name, count = counts.most_common(1)[0]
        if name != "NONE" and count >= required:
            return name, count / len(tail)
        return None


class Hysteresis:
    """Scalar Schmitt trigger with a minimum dwell time.

    Used for pinch, activation and target influence so that a value hovering on
    the threshold produces exactly one transition instead of a burst.
    """

    def __init__(self, enter: float, exit_value: float, min_dwell: float = 0.0) -> None:
        self.enter = float(enter)
        self.exit_value = float(exit_value)
        self.min_dwell = float(min_dwell)
        self.state = False
        self._since = 0.0
        self._last_change = -1e9

    def reset(self) -> None:
        self.state = False
        self._last_change = -1e9

    def update(self, value: float, timestamp: float) -> int:
        if not self.state:
            if value >= self.enter and (timestamp - self._last_change) >= self.min_dwell:
                self.state = True
                self._last_change = timestamp
                return 1
        else:
            if value <= self.exit_value:
                self.state = False
                self._last_change = timestamp
                return -1
        return 0


@dataclass
class Cooldown:
    """Simple rate limiter."""

    interval: float = 0.2
    _last: float = -1e9

    def reset(self) -> None:
        self._last = -1e9

    def ready(self, timestamp: float) -> bool:
        return (timestamp - self._last) >= self.interval

    def mark(self, timestamp: float) -> None:
        self._last = timestamp

    def try_consume(self, timestamp: float) -> bool:
        if self.ready(timestamp):
            self.mark(timestamp)
            return True
        return False


def stability_from_scores(history_scores: list[dict[str, float]], gesture: str, window: int = 10) -> float:
    """Mean score of ``gesture`` over the last ``window`` score dicts."""
    if not history_scores:
        return 0.0
    recent = history_scores[-window:]
    vals = [s.get(gesture, 0.0) for s in recent]
    return float(sum(vals) / len(vals))
