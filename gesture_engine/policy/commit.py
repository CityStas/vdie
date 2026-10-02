"""Commit Point detection — the "Air Button".

High intent probability must not execute an action.  Something has to say *now*.
The conventional answer is a pinch.  This module implements the alternative that
the original concept sketches circled around but never specified: **commit by
kinematics**.

The observation is that when a human decides to hit something, the movement has a
characteristic shape regardless of the hand pose:

    accelerate -> travel -> decelerate -> micro-pause (or a tiny reversal) -> act

A pinch is a *pose* event; this is a *motion* event.  It is measurable, it needs
no extra gesture, and — crucially — it is the same signature a real mouse
produces.  Detecting it is what makes "click without pinching" possible.

Five signals are scored independently and fused:

===================  =====================================================
``MICRO_PAUSE``      speed collapsed after a peak, held for 45..450 ms
``VELOCITY_REVERSAL`` direction reversed > 120 deg with real speed
``THRUST``           short, fast burst that terminates (flick)
``TARGET_ENTRY``     the control point entered a confident target
``PINCH``            the reference mechanism, for comparison
===================  =====================================================

Nothing here fires an action; it emits a :class:`CommitSignal` that the action
policy may or may not honour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EngineConfig
from ..features.geometry import clamp01, smoothstep
from ..motion.history import HistoryBuffer
from ..state.state_machine import StateMachine
from ..types import CommitSignal, HandFeatures, MotionState, TargetBelief


@dataclass
class CommitDetector:
    cfg: EngineConfig
    _last_commit: float = -1e9
    _last_signal: CommitSignal = field(default_factory=CommitSignal)
    _thrust_start: float | None = None
    _thrust_from_rest: bool = False
    _prev_speed: float = 0.0
    _peak: float = 0.0
    _peak_time: float = 0.0
    _commits: int = 0

    def reset(self) -> None:
        self._last_commit = -1e9
        self._last_signal = CommitSignal()
        self._thrust_start = None
        self._thrust_from_rest = False
        self._prev_speed = 0.0
        self._peak = 0.0
        self._peak_time = 0.0
        self._commits = 0

    @property
    def commits(self) -> int:
        return self._commits

    # ------------------------------------------------------------------ #
    def update(
        self,
        motion: MotionState,
        history: HistoryBuffer,
        features: HandFeatures | None,
        beliefs: list[TargetBelief],
        fsm: StateMachine,
        timestamp: float,
        interaction_position: np.ndarray | None = None,
    ) -> CommitSignal:
        ccfg = self.cfg.commit
        if not ccfg.enabled or not fsm.cursor_enabled:
            self._last_signal = CommitSignal(committed=False, score=0.0, timestamp=timestamp)
            return self._last_signal

        scores: dict[str, float] = {}

        # ---- peak tracking ------------------------------------------- #
        if motion.speed > self._peak:
            self._peak = motion.speed
            self._peak_time = timestamp
        if motion.is_pausing and motion.pause_duration > ccfg.micro_pause_max:
            self._peak = motion.speed
            self._peak_time = timestamp

        # ---- 1. micro-pause ------------------------------------------ #
        decel_ok = self._peak > 1e-6 and (motion.speed / self._peak) <= ccfg.decel_ratio
        pause_ok = ccfg.micro_pause_min <= motion.pause_duration <= ccfg.micro_pause_max
        recent_peak = (timestamp - self._peak_time) <= (ccfg.micro_pause_max + 0.35)
        if decel_ok and pause_ok and recent_peak:
            # The stronger the peak, the more deliberate the stop.
            scores["MICRO_PAUSE"] = clamp01(smoothstep(0.35, 1.6, self._peak)) * 0.9 + 0.1

        # ---- 2. velocity reversal ------------------------------------ #
        # A reversal only commits if the speed is *dropping* at that instant.
        # Otherwise every zig-zag on the way to a target fires a false commit.
        if (
            abs(motion.direction_change) >= np.radians(ccfg.reversal_angle_deg)
            and motion.speed >= ccfg.reversal_min_speed
            and motion.speed_trend < 0.0
        ):
            scores["VELOCITY_REVERSAL"] = clamp01(
                smoothstep(np.radians(ccfg.reversal_angle_deg), np.radians(175.0), abs(motion.direction_change))
            )

        # ---- 3. thrust (flick) --------------------------------------- #
        # A thrust only counts if it *started from rest* and actually *travelled*.
        # Rest alone is not enough: closing a pinch articulates the index finger,
        # which moves the control point fast and from rest — a speed-and-duration
        # test reads that as a deliberate flick.  Requiring real displacement
        # separates "the hand moved" from "the fingers moved".
        if motion.speed >= ccfg.thrust_min_speed:
            if self._thrust_start is None:
                self._thrust_start = timestamp
                self._thrust_from_rest = self._prev_speed < ccfg.micro_pause_speed
        else:
            if self._thrust_start is not None:
                duration = timestamp - self._thrust_start
                if duration <= ccfg.thrust_max_duration and self._thrust_from_rest:
                    travelled = history.path_length(max(duration, 1e-3), now=timestamp)
                    if travelled >= ccfg.thrust_min_travel:
                        scores["THRUST"] = clamp01(1.0 - duration / ccfg.thrust_max_duration) * 0.9
                self._thrust_start = None
                self._thrust_from_rest = False
        self._prev_speed = motion.speed

        # ---- 4. target entry ----------------------------------------- #
        # Spatial confirmation: the point is inside a good target *and* the user
        # is settling into it.  Entering at speed is navigation, not selection.
        if beliefs:
            best = max(beliefs, key=lambda b: b.confidence)
            settling = motion.speed_trend <= 0.2 and motion.speed < ccfg.micro_pause_speed * 4.0
            point = motion.position if interaction_position is None else np.asarray(interaction_position, dtype=np.float64)
            if best.target.contains(point) and best.target.quality() >= self.cfg.targets.min_quality and settling:
                scores["TARGET_ENTRY"] = clamp01(
                    0.4 + 0.6 * best.target.quality() * max(best.target.approach_confidence, 0.3)
                )

        # ---- 5. pinch (reference) ------------------------------------ #
        if features is not None and features.pinch_distance < self.cfg.gestures.pinch_down:
            scores["PINCH"] = clamp01(smoothstep(self.cfg.gestures.pinch_down, self.cfg.gestures.pinch_release * 0.6, features.pinch_distance))

        if not scores:
            self._last_signal = CommitSignal(committed=False, score=0.0, timestamp=timestamp)
            return self._last_signal

        kind = max(scores, key=lambda k: scores[k])
        score = float(scores[kind])

        ready = (timestamp - self._last_commit) >= ccfg.cooldown
        committed = score >= ccfg.threshold and ready
        if committed:
            self._last_commit = timestamp
            self._commits += 1
            # A commit consumes the peak so the same stop cannot commit twice.
            self._peak = motion.speed
            self._peak_time = timestamp

        self._last_signal = CommitSignal(
            committed=committed,
            score=score,
            kind=kind if committed else "NONE",
            timestamp=timestamp,
        )
        return self._last_signal

    # ------------------------------------------------------------------ #
    def breakdown(self, motion: MotionState, history: HistoryBuffer, features: HandFeatures | None) -> dict[str, float]:
        """Diagnostics for the overlay / telemetry."""
        ccfg = self.cfg.commit
        peak = self._peak
        return {
            "peak": peak,
            "ratio": (motion.speed / peak) if peak > 1e-6 else 0.0,
            "pause": motion.pause_duration,
            "decel_ratio_target": ccfg.decel_ratio,
            "pause_window": ccfg.micro_pause_max,
        }
