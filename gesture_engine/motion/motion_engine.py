"""Motion Engine: landmarks in, continuous motion state out.

Produces :class:`~gesture_engine.types.MotionState` — a *continuous* description
of how the control point is moving.  No classification happens here on purpose:
the event engine, the gesture recogniser and the commit detector each interpret
this state differently, and all three need the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EngineConfig
from ..features.feature_vector import control_point
from ..types import HandFeatures, MotionState, Observation
from .history import FrameSample, HistoryBuffer
from .trajectory import analyse, speed_trend
from .velocity import Differentiator


@dataclass(slots=True)
class MotionEngineStats:
    frames: int = 0
    lost_frames: int = 0
    total_movement: float = 0.0


class MotionEngine:
    """Turns observations into motion state + history."""

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.history = HistoryBuffer(capacity=cfg.motion.history_size)
        self.diff = Differentiator(cfg.motion)
        self.stats = MotionEngineStats()

        self._pause_start: float | None = None
        self._pause_duration = 0.0
        self._prev_direction: float | None = None
        self._last_state = MotionState()
        self._lost_since: float | None = None

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.history.clear()
        self.diff.reset()
        self._pause_start = None
        self._pause_duration = 0.0
        self._prev_direction = None
        self._last_state = MotionState()
        self._lost_since = None

    @property
    def last_state(self) -> MotionState:
        return self._last_state

    # ------------------------------------------------------------------ #
    def update(self, obs: Observation, features: HandFeatures | None = None) -> MotionState:
        """Consume one observation, update history, return the motion state."""
        mcfg = self.cfg.motion
        feats = features if features is not None else obs.features

        if obs.hand is None:
            return self._handle_loss(obs)

        point = control_point(obs.hand.points, mcfg.control_point)
        velocity, acceleration, jerk = self.diff.update(point, obs.timestamp)
        speed = float(np.linalg.norm(velocity))
        direction = float(np.arctan2(velocity[1], velocity[0])) if speed > 1e-6 else (self._prev_direction or 0.0)
        direction_change = 0.0
        if self._prev_direction is not None and speed > 1e-6:
            d = (direction - self._prev_direction + np.pi) % (2 * np.pi) - np.pi
            direction_change = float(d)

        # Pause bookkeeping (time-based, not frame-based).
        if speed < mcfg.pause_speed:
            if self._pause_start is None:
                self._pause_start = obs.timestamp
            self._pause_duration = obs.timestamp - self._pause_start
        else:
            self._pause_start = None
            self._pause_duration = 0.0

        window = self.history.window(mcfg.stats_window)
        traj = analyse(window, pause_speed=mcfg.pause_speed) if len(window) >= 2 else None
        peak, since_peak = self.history.speed_peak(mcfg.stats_window)
        trend = 0.0
        if len(window) >= 3:
            speeds = np.asarray([s.motion.speed for s in window], dtype=np.float64)
            dt = max((window[-1].timestamp - window[0].timestamp) / max(len(window) - 1, 1), 1e-3)
            trend = speed_trend(speeds) / dt

        state = MotionState(
            timestamp=obs.timestamp,
            position=point,
            velocity=velocity,
            acceleration=acceleration,
            jerk=jerk,
            speed=speed,
            direction=direction,
            direction_change=direction_change,
            curvature=traj.curvature if traj else 0.0,
            amplitude=traj.amplitude if traj else 0.0,
            displacement=traj.displacement if traj else 0.0,
            path_efficiency=traj.path_efficiency if traj else 1.0,
            pause_duration=self._pause_duration,
            is_pausing=self._pause_duration > 0.0,
            confidence=float(np.clip(obs.hand_confidence, 0.0, 1.0)),
            speed_peak=max(peak, speed),
            time_since_peak=since_peak if peak >= speed else 0.0,
            speed_trend=trend,
        )

        self._prev_direction = direction if speed > 1e-6 else self._prev_direction
        self._last_state = state
        self._lost_since = None
        self.stats.frames += 1
        self.stats.total_movement += speed * (1.0 / max(self.cfg.capture.fps, 1))

        self.history.append(
            FrameSample(
                timestamp=obs.timestamp,
                frame_id=obs.frame_id,
                point=point,
                motion=state,
                features=feats,
                hand_confidence=obs.hand_confidence,
            )
        )
        obs.motion = state
        return state

    # ------------------------------------------------------------------ #
    def _handle_loss(self, obs: Observation) -> MotionState:
        """Tracking lost: freeze and coast.  Never emit a zero-velocity jump.

        Recovery is *not* decided here — the controller owns that.  The motion
        engine only keeps the last state alive and decays its confidence so
        downstream layers can degrade gracefully instead of hard-resetting.
        """
        self.stats.lost_frames += 1
        if self._lost_since is None:
            self._lost_since = obs.timestamp
        lost_for = obs.timestamp - self._lost_since

        last = self._last_state
        decay = float(np.clip(1.0 - lost_for / 0.5, 0.0, 1.0))
        state = MotionState(
            timestamp=obs.timestamp,
            position=last.position.copy(),
            velocity=last.velocity * decay,
            acceleration=last.acceleration * decay,
            jerk=0.0,
            speed=last.speed * decay,
            direction=last.direction,
            direction_change=0.0,
            curvature=0.0,
            amplitude=0.0,
            displacement=0.0,
            path_efficiency=1.0,
            pause_duration=lost_for,
            is_pausing=True,
            confidence=0.0,
            speed_peak=last.speed_peak,
            time_since_peak=last.time_since_peak + lost_for,
            speed_trend=-last.speed,
        )
        self._last_state = state
        obs.motion = state
        # Do NOT append to history: a lost hand must not pollute trajectory stats.
        return state

    # ------------------------------------------------------------------ #
    def trajectory_stats(self, seconds: float | None = None):
        window = self.history.window(seconds if seconds is not None else self.cfg.motion.stats_window)
        return analyse(window, pause_speed=self.cfg.motion.pause_speed)

    def recent_points(self, seconds: float) -> np.ndarray:
        return self.history.positions(seconds)
