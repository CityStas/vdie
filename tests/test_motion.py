"""Motion engine: differentiation, frame-rate independence, pause tracking."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_engine.motion.history import FrameSample, HistoryBuffer
from gesture_engine.motion.motion_engine import MotionEngine
from gesture_engine.motion.trajectory import candidate_futures, predict_point, speed_trend
from gesture_engine.motion.velocity import Differentiator, ema_alpha_for_dt
from gesture_engine.types import MotionState, Observation


def _obs(t: float, point, fid: int = 0, conf: float = 1.0) -> Observation:
    from gesture_engine.types import Landmarks

    pts = np.zeros((21, 3))
    pts[:, 0] = point[0]
    pts[:, 1] = point[1]
    pts[8] = (point[0], point[1], 0.0)
    obs = Observation(timestamp=t, frame_id=fid)
    obs.hand = Landmarks(points=pts, name="hand")
    obs.hand_confidence = conf
    return obs


def test_differentiator_constant_velocity(cfg):
    d = Differentiator(cfg.motion)
    for i in range(10):
        v, a, _ = d.update(np.array([i * 0.1, 0.0]), i * (1 / 30))
    assert np.linalg.norm(v - np.array([3.0, 0.0])) < 0.2
    assert np.linalg.norm(a) < 1.0


def test_dt_is_clamped_on_frame_drops(cfg):
    d = Differentiator(cfg.motion)
    d.update(np.array([0.0, 0.0]), 0.0)
    v, _, _ = d.update(np.array([0.5, 0.0]), 10.0)  # absurd dt
    # Velocity must be bounded by the clamp, not 0.05 per second.
    assert np.linalg.norm(v) <= 0.5 / cfg.motion.dt_min


def test_velocity_is_frame_rate_independent(cfg):
    def run(fps: int) -> float:
        eng = MotionEngine(cfg)
        dt = 1.0 / fps
        for i in range(int(fps * 0.5)):
            eng.update(_obs(i * dt, (0.2 + 0.4 * i * dt, 0.5), i))
        return eng.last_state.speed

    v30 = run(30)
    v60 = run(60)
    assert v30 == pytest.approx(v60, rel=0.25)


def test_pause_detection(cfg):
    eng = MotionEngine(cfg)
    dt = 1 / 30
    for i in range(10):
        eng.update(_obs(i * dt, (0.3 + 0.02 * i, 0.5), i))
    for i in range(10, 40):
        eng.update(_obs(i * dt, (0.5, 0.5), i))
    assert eng.last_state.is_pausing
    assert eng.last_state.pause_duration > 0.5


def test_lost_hand_freezes_and_decays_confidence(cfg):
    eng = MotionEngine(cfg)
    dt = 1 / 30
    for i in range(10):
        eng.update(_obs(i * dt, (0.3 + 0.02 * i, 0.5), i))
    last_pos = eng.last_state.position.copy()
    for i in range(10, 20):
        eng.update(Observation(timestamp=i * dt, frame_id=i))  # no hand
    assert eng.last_state.confidence == 0.0
    assert np.allclose(eng.last_state.position, last_pos)
    # A lost hand must not pollute the trajectory history.
    assert len(eng.history) == 10


def test_history_window_and_peak(cfg):
    buf = HistoryBuffer(capacity=100)
    for i in range(30):
        m = MotionState(timestamp=i / 30, speed=float(i))
        buf.append(FrameSample(timestamp=i / 30, frame_id=i, point=np.zeros(2), motion=m))
    # The peak is the newest sample, so nothing has elapsed since it.
    peak, since = buf.speed_peak(0.5)
    assert peak == pytest.approx(29.0)
    assert since == pytest.approx(0.0, abs=1e-6)
    # Ask from 4 frames later: the same peak is now 4 frames old, and the window
    # must not reach past `now`.
    peak, since = buf.speed_peak(0.5, now=29 / 30 + 4 / 30)
    assert peak == pytest.approx(29.0)
    assert since == pytest.approx(4 / 30, abs=1e-6)
    # A window that starts after the peak excludes it entirely.
    peak, since = buf.speed_peak(0.05, now=1.0)
    assert peak == pytest.approx(29.0)
    assert len(buf.window(0.2)) <= 7


def test_ema_alpha_rescales_with_dt():
    a30 = ema_alpha_for_dt(0.3, 1 / 30)
    a60 = ema_alpha_for_dt(0.3, 1 / 60)
    assert a30 == pytest.approx(0.3)
    assert a60 < a30  # shorter step -> smaller effective alpha for the same time constant


def test_predict_point_extrapolates():
    p = np.array([0.0, 0.0])
    v = np.array([1.0, 0.0])
    a = np.zeros(2)
    assert np.allclose(predict_point(p, v, a, 0.1), [0.1, 0.0])


def test_candidate_futures_shapes_and_count():
    futures = candidate_futures(np.array([0.5, 0.5]), np.array([1.0, 0.0]), np.zeros(2), horizons=(0.1, 0.2), samples=5)
    assert len(futures) == 10
    names = {f.name for f in futures}
    assert names == {"straight", "decelerating", "curved_left", "curved_right", "reversing"}
    for f in futures:
        assert f.points.shape == (5, 2)


def test_candidate_futures_empty_when_stationary():
    assert candidate_futures(np.zeros(2), np.zeros(2), np.zeros(2)) == []


def test_decelerating_future_is_shorter_than_straight():
    v = np.array([1.0, 0.0])
    futures = {f.name: f for f in candidate_futures(np.zeros(2), v, np.zeros(2), horizons=(0.3,), samples=8)}
    straight_len = np.linalg.norm(futures["straight"].endpoint)
    decel_len = np.linalg.norm(futures["decelerating"].endpoint)
    assert decel_len < straight_len


def test_speed_trend_sign():
    assert speed_trend(np.array([0.1, 0.2, 0.3, 0.4])) > 0
    assert speed_trend(np.array([0.4, 0.3, 0.2, 0.1])) < 0
