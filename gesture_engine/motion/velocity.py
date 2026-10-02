"""Numerical differentiation with frame-rate independence.

Two things matter here and are easy to get wrong:

1. ``dt`` must be clamped.  A dropped frame produces a huge ``dt``; without a
   clamp the velocity spikes and the cursor teleports.
2. Derivatives of a noisy signal are noisy.  MediaPipe landmark jitter of
   ~0.002 normalized units at 30 FPS is ~0.06 units/s of raw velocity noise,
   which is already 10% of a typical slow movement.  Hence the EMA on velocity.
"""

from __future__ import annotations

import numpy as np

from ..config import MotionConfig


class Differentiator:
    """Velocity / acceleration / jerk from a stream of positions."""

    def __init__(self, cfg: MotionConfig) -> None:
        self.cfg = cfg
        self._prev_pos: np.ndarray | None = None
        self._prev_t: float | None = None
        self._velocity = np.zeros(2)
        self._accel = np.zeros(2)
        self._prev_velocity = np.zeros(2)
        self._jerk = 0.0
        self._smooth_pos: np.ndarray | None = None
        #: False until the first real velocity sample has been produced.
        self._primed = False

    def reset(self) -> None:
        self._prev_pos = None
        self._prev_t = None
        self._velocity = np.zeros(2)
        self._accel = np.zeros(2)
        self._prev_velocity = np.zeros(2)
        self._jerk = 0.0
        self._smooth_pos = None
        self._primed = False

    @property
    def velocity(self) -> np.ndarray:
        return self._velocity.copy()

    @property
    def acceleration(self) -> np.ndarray:
        return self._accel.copy()

    def dt(self, timestamp: float) -> float | None:
        if self._prev_t is None:
            return None
        raw = timestamp - self._prev_t
        if raw <= 0:
            return None
        return float(np.clip(raw, self.cfg.dt_min, self.cfg.dt_max))

    def update(self, position: np.ndarray, timestamp: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Returns ``(velocity, acceleration, jerk)`` in normalized units."""
        pos = np.asarray(position, dtype=np.float64).copy()

        if self.cfg.position_smoothing > 0.0:
            a = float(np.clip(self.cfg.position_smoothing, 0.0, 1.0))
            self._smooth_pos = pos if self._smooth_pos is None else self._smooth_pos + a * (pos - self._smooth_pos)
            pos = self._smooth_pos.copy()

        dt = self.dt(timestamp)
        if dt is None:
            self._prev_pos = pos
            self._prev_t = timestamp
            self._primed = False
            return self._velocity.copy(), self._accel.copy(), self._jerk

        raw_v = (pos - self._prev_pos) / dt  # type: ignore[operator]
        a_v = float(np.clip(self.cfg.velocity_alpha, 0.0, 1.0))
        if not self._primed:
            # Prime from the first real observation instead of ramping up from
            # zero.  Ramping makes a constant-velocity movement look like a
            # 1/dt-sized acceleration spike for the first few frames, which the
            # commit detector reads as "accelerating" and the intent field as
            # navigation evidence.
            self._velocity = raw_v.copy()
            self._accel = np.zeros(2)
            self._prev_velocity = self._velocity.copy()
            self._prev_pos = pos
            self._prev_t = timestamp
            self._primed = True
            self._jerk = 0.0
            return self._velocity.copy(), self._accel.copy(), self._jerk

        self._velocity = self._velocity + a_v * (raw_v - self._velocity) if self.cfg.velocity_filter == "ema" else raw_v

        raw_a = (self._velocity - self._prev_velocity) / dt
        a_a = float(np.clip(self.cfg.acceleration_alpha, 0.0, 1.0))
        self._accel = self._accel + a_a * (raw_a - self._accel)

        prev_speed = float(np.linalg.norm(self._prev_velocity))
        new_speed = float(np.linalg.norm(self._velocity))
        self._jerk = (new_speed - prev_speed) / dt

        self._prev_velocity = self._velocity.copy()
        self._prev_pos = pos
        self._prev_t = timestamp
        return self._velocity.copy(), self._accel.copy(), self._jerk


def finite_difference(points: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """Batch derivative, mainly for offline analysis and tests."""
    p = np.asarray(points, dtype=np.float64)
    t = np.asarray(timestamps, dtype=np.float64)
    if p.shape[0] < 2:
        return np.zeros_like(p)
    dt = np.diff(t).reshape(-1, 1)
    dt[dt <= 0] = 1e-6
    v = np.diff(p, axis=0) / dt
    return np.vstack([v[:1], v])


def ema_alpha_for_dt(alpha_at_reference: float, dt: float, reference_dt: float = 1.0 / 30.0) -> float:
    """Rescale an EMA coefficient so smoothing is time-based, not frame-based.

    Without this, the same config produces different smoothing at 30 and 60 FPS
    — a classic source of bogus A/B conclusions.
    """
    a = float(np.clip(alpha_at_reference, 1e-6, 1.0 - 1e-6))
    ratio = dt / max(reference_dt, 1e-6)
    return float(1.0 - (1.0 - a) ** ratio)
