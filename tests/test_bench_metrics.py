"""Benchmark metrics.

Every metric here has already been wrong once in a way that produced a
confident number instead of an error, so the tests are written against the
*physical* property each metric claims to measure, not against its output shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from gesture_engine.bench.metrics import (
    cursor_jitter,
    cursor_lag,
    _longest_run,
    _runs,
)
from gesture_engine.types import FrameReport, MotionState


def _report(
    timestamp: float,
    hand: tuple[float, float],
    cursor: tuple[float, float],
    speed: float,
    mode: str = "PRECISION",
) -> FrameReport:
    return FrameReport(
        timestamp=timestamp,
        motion=MotionState(
            timestamp=timestamp,
            position=np.array(hand, dtype=float),
            velocity=np.zeros(2),
            speed=speed,
            confidence=1.0,
        ),
        cursor=np.array(cursor, dtype=float),
        notes={"cursor_mode": mode},
    )


# --------------------------------------------------------------------------- #
# run helpers
# --------------------------------------------------------------------------- #


def test_runs_and_longest_run_agree():
    mask = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=bool)
    assert _runs(mask) == [(1, 3), (4, 8), (9, 10)]
    assert _longest_run(mask) == (4, 8)


# --------------------------------------------------------------------------- #
# jitter
# --------------------------------------------------------------------------- #


def _jitter_of(cursor_xs: list[float], hand_speed: float, settle_frames: int = 5) -> float:
    reports = [
        _report(i / 30.0, (0.5, 0.5), (x, 0.5), hand_speed)
        for i, x in enumerate(cursor_xs)
    ]
    return cursor_jitter(reports, screen=(1000, 1000), settle_frames=settle_frames)


def test_jitter_is_zero_for_a_perfectly_still_cursor():
    assert _jitter_of([0.5] * 60, hand_speed=0.0) == pytest.approx(0.0, abs=1e-9)


def test_jitter_ignores_movement_of_the_hand():
    """A moving hand is not jitter, however fast the cursor has to follow.

    This is the defect the metric had: the second difference of a real reach is
    dominated by the reach's own acceleration, so a responsive controller scored
    60 px of "shake" while the isolated probe measured 0.41 px.
    """
    moving = _jitter_of([0.5 + 0.004 * i for i in range(60)], hand_speed=1.5)
    assert moving == pytest.approx(0.0, abs=1e-9)


def test_jitter_measures_shake_when_the_hand_is_still():
    """An oscillating cursor with a still hand must register."""
    xs = [0.5 + 0.02 * (1 if i % 2 else -1) for i in range(60)]
    assert _jitter_of(xs, hand_speed=0.0) > 5.0


def test_jitter_skips_the_settle_window_of_each_still_period():
    """The cursor is still decelerating out of the previous movement there.

    Without the settle window the metric charges the controller for finishing a
    movement it was just asked to make.  The step here lands *inside* the
    window, so it must not be counted — and must be counted when the window is
    switched off, which is what makes this a test of the window rather than of
    the residual formula.
    """
    xs = [0.5, 0.8] + [0.8] * 58
    assert _jitter_of(xs, hand_speed=0.0, settle_frames=5) == pytest.approx(0.0, abs=1e-9)
    assert _jitter_of(xs, hand_speed=0.0, settle_frames=0) > 1.0


def test_jitter_ignores_session_boundaries():
    """A deliberate teleport at activation is not jitter."""
    reports = [_report(i / 30.0, (0.5, 0.5), (0.5, 0.5), 0.0, mode="PAUSED") for i in range(10)]
    reports += [_report(i / 30.0, (0.5, 0.5), (0.9, 0.5), 0.0) for i in range(10, 70)]
    assert cursor_jitter(reports, screen=(1000, 1000)) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# lag
# --------------------------------------------------------------------------- #


def _synthetic_session(
    n_strokes: int = 5,
    stroke_frames: int = 30,
    gap_frames: int = 30,
    delay_frames: int = 3,
    noise: float = 0.0,
    seed: int = 3,
) -> list[FrameReport]:
    """Hand strokes with the cursor following at a known integer delay.

    The cursor is a delayed copy of the hand, which is the one case where the
    right answer is known exactly, so the estimator can be checked rather than
    eyeballed.
    """
    rng = np.random.default_rng(seed)
    hand_x: list[float] = []
    speeds: list[float] = []
    for _ in range(n_strokes):
        for k in range(stroke_frames):
            u = k / max(stroke_frames - 1, 1)
            hand_x.append(0.30 + 0.40 * (0.5 - 0.5 * np.cos(np.pi * u)))
            speeds.append(0.9 * np.sin(np.pi * u))
        hand_x.extend([0.70] * gap_frames)
        speeds.extend([0.0] * gap_frames)

    reports = []
    for i, x in enumerate(hand_x):
        j = max(0, i - delay_frames)
        cursor_x = hand_x[j] + (rng.normal(0.0, noise) if noise else 0.0)
        reports.append(_report(i / 30.0, (x, 0.5), (cursor_x, 0.5), speeds[i]))
    return reports


def test_lag_recovers_a_known_delay():
    """Exact, not approximate: the estimator is checked against ground truth."""
    lag, quality = cursor_lag(_synthetic_session(delay_frames=3))
    assert lag == pytest.approx(3 / 30.0 * 1000.0, abs=1e-6)
    assert quality > 0.9


def test_lag_recovers_a_different_known_delay():
    """A metric that always returns the same number is not measuring anything."""
    lag, _ = cursor_lag(_synthetic_session(delay_frames=7))
    assert lag == pytest.approx(7 / 30.0 * 1000.0, abs=1e-6)


def test_lag_separates_two_delays():
    slow, _ = cursor_lag(_synthetic_session(delay_frames=8))
    fast, _ = cursor_lag(_synthetic_session(delay_frames=2))
    assert slow > fast


def test_lag_survives_short_strokes():
    """The lag must be sampled outside the hand's window.

    Defining the correlation window as "hand moving **or** cursor moving", or
    simply shifting the window with the lag, mixes in the frames where the hand
    has already stopped but the cursor is still arriving.  With 20-frame strokes
    and a known 4-frame delay those versions returned 47 ms and 160 ms instead
    of 133 ms.
    """
    lag, _ = cursor_lag(_synthetic_session(stroke_frames=20, delay_frames=4))
    assert lag == pytest.approx(4 / 30.0 * 1000.0, abs=1e-6)


def test_lag_is_unknown_when_the_session_has_no_long_strokes():
    """Better to report "unknown" than a number derived from one short stroke.

    This is what the previous implementation got wrong: it used only the longest
    moving run, so the answer depended on which stroke happened to be longest,
    and it returned a confident 133 ms for a controller whose isolated phase lag
    is 60-76 ms.
    """
    reports = _synthetic_session(n_strokes=4, stroke_frames=6, gap_frames=30)
    lag, quality = cursor_lag(reports)
    assert (lag, quality) == (0.0, 0.0)


def test_lag_needs_several_strokes_before_it_will_answer():
    reports = _synthetic_session(n_strokes=1, stroke_frames=60, gap_frames=20)
    assert cursor_lag(reports) == (0.0, 0.0)


def test_lag_uses_every_stroke_not_just_the_longest():
    """Several short strokes must give the same answer as a few long ones."""
    long_run = _synthetic_session(n_strokes=3, stroke_frames=60, gap_frames=20, delay_frames=4)
    many = _synthetic_session(n_strokes=6, stroke_frames=20, gap_frames=20, delay_frames=4)
    lag_long, _ = cursor_lag(long_run)
    lag_many, quality_many = cursor_lag(many)
    assert lag_many == pytest.approx(lag_long, abs=1e-6)
    assert quality_many > 0.8
