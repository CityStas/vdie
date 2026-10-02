"""Dynamic (trajectory) gesture scoring.

Static poses alone cannot express "swipe left" or "scroll down", and — more
importantly — the *same* pose means different things depending on how the user
arrived at it.  This module scores motion patterns over the history window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EngineConfig
from ..features.geometry import clamp01, smoothstep
from ..motion.history import HistoryBuffer


@dataclass(frozen=True, slots=True)
class StrokeStats:
    """Summary of one contiguous movement stroke."""

    displacement: np.ndarray
    duration: float
    mean_speed: float
    peak_speed: float
    path_length: float

    @property
    def magnitude(self) -> float:
        return float(np.hypot(self.displacement[0], self.displacement[1]))

    @property
    def axis_ratio(self) -> tuple[float, float]:
        """``(horizontal, vertical)`` dominance, each in ``[0, 1]``."""
        mag = max(self.magnitude, 1e-6)
        return abs(float(self.displacement[0])) / mag, abs(float(self.displacement[1])) / mag


def _movement_span(history: HistoryBuffer, seconds: float, cfg: EngineConfig):
    """The trailing *contiguous run of moving samples* inside the time window.

    Analysing the whole window is wrong for every temporal gesture: once the
    history is full the window is always ~1 s long, so a duration test measured
    the window instead of the movement and ``duration_conf`` was permanently 0.
    Restricting the analysis to the current stroke also stops a previous stroke
    from diluting the displacement of the next one.
    """
    samples = history.window(seconds)
    if len(samples) < 3:
        return samples
    threshold = cfg.motion.pause_speed
    end = None
    for i in range(len(samples) - 1, -1, -1):
        if samples[i].motion.speed > threshold:
            end = i
            break
    if end is None:
        return []
    start = end
    while start > 0 and samples[start - 1].motion.speed > threshold:
        start -= 1
    return samples[start : end + 1]


def _span_stats(span) -> StrokeStats:
    """Summarise one stroke.

    Peak speed is measured **over the stroke itself**, not over the whole history
    window.  The window version was actively harmful: a single fast frame (a
    tracker jump, a pose change) inside the last second suppressed every scroll
    score for a full second, because ``slow_enough`` was evaluating that spike
    instead of the slow stroke the user was actually making.
    """
    if len(span) < 2:
        return StrokeStats(np.zeros(2), 0.0, 0.0, 0.0, 0.0)
    pts = np.asarray([s.point for s in span], dtype=np.float64)
    ts = np.asarray([s.timestamp for s in span], dtype=np.float64)
    speeds = np.asarray([s.motion.speed for s in span], dtype=np.float64)
    path = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return StrokeStats(
        displacement=pts[-1] - pts[0],
        duration=float(max(ts[-1] - ts[0], 1e-6)),
        mean_speed=float(speeds.mean()),
        peak_speed=float(speeds.max()),
        path_length=path,
    )


def score_dynamic(history: HistoryBuffer, cfg: EngineConfig, extended_count: int | None = None) -> dict[str, float]:
    """Score swipe / scroll gestures from the recent trajectory."""
    g = cfg.gestures
    window = min(g.swipe_max_duration * 1.6, 1.2)
    span = _movement_span(history, window, cfg)
    scores: dict[str, float] = {}
    if len(span) < 3:
        return scores
    st = _span_stats(span)
    dx, dy = float(st.displacement[0]), float(st.displacement[1])
    horizontal, vertical = st.axis_ratio

    duration_conf = clamp01(1.0 - smoothstep(g.swipe_max_duration * 0.7, g.swipe_max_duration, st.duration))
    magnitude_conf = smoothstep(g.swipe_min_distance * 0.5, g.swipe_min_distance * 1.4, st.magnitude)
    speed_conf = smoothstep(g.swipe_min_speed * 0.6, g.swipe_min_speed * 1.3, st.mean_speed)
    # At least one finger must be extended: a closed hand sweeping across the
    # frame is not a gesture.  The knee has to sit between 0 and 1 extended
    # fingers — with the previous 0.5..1.5 band a single extended finger scored
    # 0.5, which halved every swipe and scroll and pushed them under
    # `min_confidence`, making both gestures unreachable once the cursor was
    # armed with the index pointing.
    openness = 1.0 if extended_count is None else smoothstep(0.25, 0.9, float(extended_count))

    base = magnitude_conf * duration_conf * speed_conf * openness

    if st.magnitude > 1e-6:
        ratio = g.swipe_axis_ratio
        if horizontal > vertical * ratio - 0.2:
            if dx < 0:
                scores["SWIPE_LEFT"] = float(base * clamp01(horizontal))
            else:
                scores["SWIPE_RIGHT"] = float(base * clamp01(horizontal))
        if vertical > horizontal * ratio - 0.2:
            if dy < 0:
                scores["SWIPE_UP"] = float(base * clamp01(vertical))
            else:
                scores["SWIPE_DOWN"] = float(base * clamp01(vertical))

    # Scroll: a *slow*, deliberate, vertically dominant stroke.
    #
    # The discriminator matters more than the score.  "Pointing while moving
    # vertically" is also what a travel to a target looks like, so scroll needs
    # three independent gates: the vertical axis must dominate the horizontal
    # one, the vertical displacement must be a real stroke (not the last 5 mm of
    # a deceleration), and the speed must stay under a ceiling.  The mean speed
    # alone is not enough — it stays low during a long deceleration — so the peak
    # is tested as well.
    #
    # Deliberately *no* duration gate here.  A scroll has no natural end, so a
    # duration term only fires once the user has already stopped moving: it
    # measured 0.4 s of dead latency on the scroll scenario.  A swipe is a finite
    # stroke and keeps its duration term; a scroll is bounded by speed instead.
    if st.path_length > 0.0:
        axis_ok = vertical > horizontal * g.swipe_axis_ratio
        stroke_conf = smoothstep(g.scroll_min_distance * 0.5, g.scroll_min_distance * 1.6, abs(dy))
        sustained = smoothstep(g.scroll_min_distance * 0.4, g.scroll_min_distance * 1.4, st.path_length)
        speed_ceiling = max(st.peak_speed, st.mean_speed)
        slow_enough = 1.0 - smoothstep(g.scroll_max_speed * 0.5, g.scroll_max_speed, speed_ceiling)
        scroll_conf = float(axis_ok * vertical * stroke_conf * sustained * slow_enough * openness)
        if scroll_conf > 0.05:
            if dy < 0:
                scores["SCROLL_UP"] = scroll_conf
            else:
                scores["SCROLL_DOWN"] = scroll_conf

    return scores


def best_dynamic(history: HistoryBuffer, cfg: EngineConfig, extended_count: int | None = None) -> tuple[str, float]:
    scores = score_dynamic(history, cfg, extended_count)
    if not scores:
        return "NONE", 0.0
    name = max(scores, key=lambda k: scores[k])
    return name, float(scores[name])
