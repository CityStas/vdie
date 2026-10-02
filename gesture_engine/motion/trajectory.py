"""Trajectory analysis over a history window.

Includes the closed-form *counterfactual futures* used by the intent layer:
given a partial trajectory, extrapolate a handful of plausible continuations and
see which intents they support.  Deliberately model-free — a K-hypothesis
kinematic extrapolation costs microseconds, while a learned predictor would cost
a training pipeline we do not need yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..features.geometry import EPS, curvature_radius, polyline_length, safe_normalize, straightness
from .history import FrameSample


@dataclass(slots=True)
class TrajectoryStats:
    amplitude: float = 0.0  # total path length
    displacement: float = 0.0  # straight-line start->end
    path_efficiency: float = 1.0  # displacement / amplitude
    straightness: float = 1.0
    curvature: float = 0.0  # 1/px, signed
    mean_speed: float = 0.0
    max_speed: float = 0.0
    speed_variance: float = 0.0
    direction_changes: int = 0
    duration: float = 0.0
    pause_fraction: float = 0.0
    jitter: float = 0.0  # high-frequency energy, proxy for tremor
    straightness_ratio: float = 1.0


def analyse(samples: list[FrameSample], pause_speed: float = 0.12) -> TrajectoryStats:
    if len(samples) < 2:
        return TrajectoryStats()

    pts = np.asarray([s.point for s in samples], dtype=np.float64)
    speeds = np.asarray([s.motion.speed for s in samples], dtype=np.float64)
    ts = np.asarray([s.timestamp for s in samples], dtype=np.float64)

    amplitude = polyline_length(pts)
    displacement = float(np.linalg.norm(pts[-1] - pts[0]))
    duration = float(max(ts[-1] - ts[0], EPS))

    # Jitter: residual after removing a 3-tap moving average.
    if pts.shape[0] >= 5:
        kernel = np.ones(3) / 3.0
        smooth = np.stack(
            [
                np.convolve(pts[:, 0], kernel, mode="valid"),
                np.convolve(pts[:, 1], kernel, mode="valid"),
            ],
            axis=1,
        )
        # `valid` convolution of length n with a 3-tap kernel yields n-2 samples,
        # aligned with pts[1:-1].
        jitter = float(np.linalg.norm(pts[1:-1] - smooth, axis=1).mean())
    else:
        jitter = 0.0

    return TrajectoryStats(
        amplitude=float(amplitude),
        displacement=displacement,
        path_efficiency=float(displacement / amplitude) if amplitude > EPS else 1.0,
        straightness=straightness(pts),
        curvature=float(1.0 / curvature_radius(pts)) if np.isfinite(curvature_radius(pts)) else 0.0,
        mean_speed=float(speeds.mean()),
        max_speed=float(speeds.max()),
        speed_variance=float(speeds.var()),
        direction_changes=int(np.sum(np.abs(np.diff(np.asarray([s.motion.direction_change for s in samples]))) > 0.6)),
        duration=duration,
        pause_fraction=float(np.mean(speeds < pause_speed)),
        jitter=jitter,
    )


# --------------------------------------------------------------------------- #
# Counterfactual futures
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Future:
    """One extrapolated continuation of the current trajectory."""

    name: str  # straight | decelerating | curved_left | curved_right | reversing
    points: np.ndarray  # (n, 2) predicted positions
    horizon: float
    probability: float = 0.0
    intents: dict[str, float] = field(default_factory=dict)

    @property
    def endpoint(self) -> np.ndarray:
        return self.points[-1]


def candidate_futures(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    horizons: tuple[float, ...] = (0.12, 0.24, 0.40),
    samples: int = 5,
) -> list[Future]:
    """Kinematic K-hypothesis extrapolation.

    Five cheap hypotheses:
      * ``straight``      constant velocity
      * ``decelerating``  exponential speed decay (classic target approach)
      * ``curved_left``   constant turn rate, +15 deg/s
      * ``curved_right``  constant turn rate, -15 deg/s
      * ``reversing``     velocity flips sign (abort / hesitation)
    """
    p = np.asarray(position, dtype=np.float64)[:2]
    v = np.asarray(velocity, dtype=np.float64)[:2]
    a = np.asarray(acceleration, dtype=np.float64)[:2]
    speed = float(np.linalg.norm(v))
    out: list[Future] = []
    if speed < EPS and float(np.linalg.norm(a)) < EPS:
        return out

    for horizon in horizons:
        ts = np.linspace(0.0, horizon, max(samples, 2))

        straight = p[None, :] + v[None, :] * ts[:, None] + 0.5 * a[None, :] * (ts[:, None] ** 2)
        out.append(Future("straight", straight, horizon))

        decay = np.exp(-ts / max(horizon * 0.45, 1e-3))
        decel = p[None, :] + (v[None, :] * decay[:, None]) * ts[:, None]
        out.append(Future("decelerating", decel, horizon))

        for name, sign in (("curved_left", 1.0), ("curved_right", -1.0)):
            omega = sign * np.radians(15.0)
            pts = []
            q = p.copy()
            cur_v = v.copy()
            dt = horizon / (len(ts) - 1)
            pts.append(q.copy())
            for _ in range(len(ts) - 1):
                th = omega * dt
                c, s = np.cos(th), np.sin(th)
                cur_v = np.array([c * cur_v[0] - s * cur_v[1], s * cur_v[0] + c * cur_v[1]])
                q = q + cur_v * dt
                pts.append(q.copy())
            out.append(Future(name, np.asarray(pts), horizon))

        rev = p[None, :] + v[None, :] * ts[:, None] * (1.0 - 2.0 * (ts[:, None] / max(horizon, EPS)))
        out.append(Future("reversing", rev, horizon))

    return out


def score_futures(futures: list[Future], intents_for: dict[str, dict[str, float]]) -> list[Future]:
    """Attach intent support to each future and normalise probabilities.

    ``intents_for`` maps a future name to ``{intent: weight}``.
    """
    if not futures:
        return futures
    weights = np.asarray([max(intents_for.get(f.name, {}).get("_prior", 1.0), 1e-6) for f in futures], dtype=np.float64)
    weights = weights / weights.sum()
    for f, w in zip(futures, weights):
        f.probability = float(w)
        f.intents = dict(intents_for.get(f.name, {}))
    return futures


def aggregate_intent_support(futures: list[Future]) -> dict[str, float]:
    """Probability mass per intent, across all hypotheses."""
    acc: dict[str, float] = {}
    for f in futures:
        for intent, weight in f.intents.items():
            if intent.startswith("_"):
                continue
            acc[intent] = acc.get(intent, 0.0) + f.probability * float(weight)
    total = sum(acc.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in acc.items()}


def predict_point(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    horizon: float,
    accel_weight: float = 0.35,
) -> np.ndarray:
    """Short-horizon lead used by the cursor controller (section 25)."""
    p = np.asarray(position, dtype=np.float64)[:2]
    v = np.asarray(velocity, dtype=np.float64)[:2]
    a = np.asarray(acceleration, dtype=np.float64)[:2]
    return p + v * horizon + 0.5 * accel_weight * a * (horizon**2)


def speed_trend(speeds: np.ndarray) -> float:
    """Least-squares slope of speed over the window (units/s)."""
    s = np.asarray(speeds, dtype=np.float64)
    if s.size < 2:
        return 0.0
    x = np.arange(s.size, dtype=np.float64)
    x -= x.mean()
    denom = float((x * x).sum())
    if denom < EPS:
        return 0.0
    return float((x * (s - s.mean())).sum() / denom)
