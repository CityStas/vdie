"""Filters.

The V1 spec proposes a single adaptive EMA whose alpha depends on speed.  That is
a good start but it is a first-order low-pass: to kill tremor at rest you need a
small alpha, and a small alpha costs you lag the moment the user moves.  The
adaptive alpha only partially fixes this because the alpha responds to the *same*
noisy speed estimate it is trying to stabilise.

Two better primitives are provided here:

* :class:`OneEuroFilter` — the standard answer to this exact trade-off.  It makes
  the cutoff frequency an affine function of the *filtered* speed derivative, so
  jitter is removed at rest while fast movement passes through almost untouched.
  This is what the cursor controller uses by default.
* :class:`AdaptiveEMA` — the V1 behaviour, kept as the comparison baseline for
  the "fixed vs adaptive smoothing" experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..motion.velocity import ema_alpha_for_dt


@dataclass
class EMA:
    """Exponential moving average with a time-based coefficient."""

    alpha: float = 0.3
    reference_dt: float = 1.0 / 30.0
    value: np.ndarray | None = None

    def reset(self) -> None:
        self.value = None

    def update(self, x: np.ndarray, dt: float | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.value is None:
            self.value = x.copy()
            return self.value.copy()
        a = self.alpha if dt is None else ema_alpha_for_dt(self.alpha, dt, self.reference_dt)
        self.value = self.value + a * (x - self.value)
        return self.value.copy()


class AdaptiveEMA:
    """V1 adaptive smoothing: alpha interpolated between slow and fast by speed."""

    def __init__(
        self,
        alpha_slow: float = 0.12,
        alpha_medium: float = 0.30,
        alpha_fast: float = 0.65,
        speed_slow: float = 0.15,
        speed_fast: float = 1.6,
        reference_dt: float = 1.0 / 30.0,
    ) -> None:
        self.alpha_slow = alpha_slow
        self.alpha_medium = alpha_medium
        self.alpha_fast = alpha_fast
        self.speed_slow = speed_slow
        self.speed_fast = speed_fast
        self.reference_dt = reference_dt
        self.value: np.ndarray | None = None

    def reset(self) -> None:
        self.value = None

    def alpha_for(self, speed: float) -> float:
        if speed <= self.speed_slow:
            return self.alpha_slow
        if speed >= self.speed_fast:
            return self.alpha_fast
        u = (speed - self.speed_slow) / max(self.speed_fast - self.speed_slow, 1e-6)
        if u < 0.5:
            return self.alpha_slow + (self.alpha_medium - self.alpha_slow) * (u / 0.5)
        return self.alpha_medium + (self.alpha_fast - self.alpha_medium) * ((u - 0.5) / 0.5)

    def update(self, x: np.ndarray, speed: float, dt: float | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self.value is None:
            self.value = x.copy()
            return self.value.copy()
        a = self.alpha_for(speed)
        if dt is not None:
            a = ema_alpha_for_dt(a, dt, self.reference_dt)
        self.value = self.value + a * (x - self.value)
        return self.value.copy()


class OneEuroFilter:
    """1-Euro filter (Casiez et al. 2012) for a 2-D point.

    ``min_cutoff`` controls jitter at rest, ``beta`` controls how quickly the
    filter opens up as speed increases.  The two are independent, which is the
    whole point: you can be very smooth when still *and* very responsive when
    moving, which a fixed-alpha EMA cannot do.
    """

    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.045,
        d_cutoff: float = 1.0,
        dim: int = 2,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dim = dim
        self._x: np.ndarray | None = None
        self._dx: np.ndarray = np.zeros(dim)

    def reset(self) -> None:
        self._x = None
        self._dx = np.zeros(self.dim)

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def update(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()[: self.dim]
        dt = max(float(dt), 1e-6)
        if self._x is None:
            self._x = x.copy()
            return self._x.copy()

        raw_d = (x - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = self._dx + a_d * (raw_d - self._dx)

        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = self._alpha(cutoff, dt)
        self._x = self._x + a * (x - self._x)
        return self._x.copy()


class VelocityEstimator:
    """Filtered scalar speed with a time-based EMA (used for gain scheduling)."""

    def __init__(self, alpha: float = 0.4, reference_dt: float = 1.0 / 30.0) -> None:
        self.alpha = alpha
        self.reference_dt = reference_dt
        self.value = 0.0

    def reset(self) -> None:
        self.value = 0.0

    def update(self, speed: float, dt: float) -> float:
        a = ema_alpha_for_dt(self.alpha, dt, self.reference_dt)
        self.value += a * (float(speed) - self.value)
        return self.value
