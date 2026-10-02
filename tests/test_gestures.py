"""Gesture recognition on known ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_engine.features.feature_vector import build_features
from gesture_engine.gestures.dynamic import score_dynamic
from gesture_engine.gestures.recognizer import GestureRecognizer
from gesture_engine.gestures.static import best_static, score_static
from gesture_engine.motion.history import FrameSample, HistoryBuffer
from gesture_engine.tracking.landmark_model import POSES, build_hand, make_pose
from gesture_engine.types import MotionState

#: Poses the static scorer is expected to identify.  POINT / INDEX_SIDE share a
#: finger state with INDEX_UP and are separated by orientation, so they are
#: tested separately.
EXPECTED = {
    "OPEN_PALM": "OPEN_PALM",
    "FIST": "FIST",
    "INDEX_UP": "INDEX_UP",
    "V_SIGN": "V_SIGN",
    "THREE": "THREE",
    "ROCK": "ROCK",
    "PINCH": "PINCH",
}


@pytest.mark.parametrize("pose_name,expected", EXPECTED.items())
def test_static_poses_are_recognised(cfg, pose_name, expected):
    f = build_features(build_hand(POSES[pose_name]), cfg)
    name, score, scores = best_static(f, cfg)
    assert name == expected, f"got {name} with scores {sorted(scores.items(), key=lambda kv: -kv[1])[:4]}"
    assert score > 0.5


def test_index_orientation_distinguishes_point_from_index_up(cfg):
    up = build_features(build_hand(make_pose("INDEX_UP", rotation_deg=0)), cfg)
    side = build_features(build_hand(make_pose("INDEX_SIDE", rotation_deg=90)), cfg)
    assert best_static(up, cfg)[0] == "INDEX_UP"
    assert best_static(side, cfg)[0] == "POINT"


def test_open_palm_beats_fist_and_vice_versa(cfg):
    palm = score_static(build_features(build_hand(POSES["OPEN_PALM"]), cfg), cfg)
    fist = score_static(build_features(build_hand(POSES["FIST"]), cfg), cfg)
    assert palm["OPEN_PALM"] > palm["FIST"]
    assert fist["FIST"] > fist["OPEN_PALM"]


def test_confidence_degrades_with_noise(cfg):
    clean = build_features(build_hand(POSES["OPEN_PALM"], noise=0.0), cfg)
    noisy = build_features(build_hand(POSES["OPEN_PALM"], noise=0.02), cfg)
    assert score_static(clean, cfg)["OPEN_PALM"] >= score_static(noisy, cfg)["OPEN_PALM"] - 0.05


# --------------------------------------------------------------------------- #


def _push_motion(history: HistoryBuffer, points, speeds, dt=1 / 30, start=0.0):
    for i, (p, s) in enumerate(zip(points, speeds)):
        m = MotionState(timestamp=start + i * dt, position=np.asarray(p, dtype=float), speed=float(s))
        history.append(FrameSample(timestamp=start + i * dt, frame_id=i, point=np.asarray(p, dtype=float), motion=m))


def test_swipe_left_detected(cfg):
    history = HistoryBuffer(capacity=200)
    pts = [(0.8 - 0.03 * i, 0.5) for i in range(15)]
    _push_motion(history, pts, [1.2] * 15)
    scores = score_dynamic(history, cfg, extended_count=4)
    assert scores.get("SWIPE_LEFT", 0.0) > 0.5
    assert "SWIPE_RIGHT" not in scores


def test_swipe_axis_ratio_rejects_diagonal(cfg):
    history = HistoryBuffer(capacity=200)
    pts = [(0.8 - 0.02 * i, 0.5 - 0.02 * i) for i in range(15)]
    _push_motion(history, pts, [1.2] * 15)
    scores = score_dynamic(history, cfg, extended_count=4)
    assert max(scores.values(), default=0.0) < 0.5


def test_scroll_requires_slow_vertical_motion(cfg):
    history = HistoryBuffer(capacity=200)
    pts = [(0.5, 0.6 - 0.012 * i) for i in range(20)]
    _push_motion(history, pts, [0.5] * 20)
    scores = score_dynamic(history, cfg, extended_count=1)
    assert scores.get("SCROLL_UP", 0.0) > scores.get("SWIPE_UP", 0.0)


def test_recognizer_needs_temporal_confirmation(cfg):
    rec = GestureRecognizer(cfg)
    history = HistoryBuffer(capacity=200)
    f = build_features(build_hand(POSES["PINCH"]), cfg)
    results = [rec.update(f, history, i / 30) for i in range(12)]
    # Nothing before the vote threshold is reached.
    assert all(r.gesture == "NONE" for r in results[:5])
    assert results[-1].gesture == "PINCH"
    assert results[-1].stability > 0.5


def test_recognizer_reports_no_gesture_without_features(cfg):
    rec = GestureRecognizer(cfg)
    history = HistoryBuffer(capacity=10)
    assert rec.update(None, history, 0.0).gesture == "NONE"


def test_recognizer_confidence_is_bounded(cfg):
    rec = GestureRecognizer(cfg)
    history = HistoryBuffer(capacity=10)
    f = build_features(build_hand(POSES["OPEN_PALM"]), cfg)
    for i in range(20):
        r = rec.update(f, history, i / 30)
        assert 0.0 <= r.confidence <= 1.0
        assert 0.0 <= r.stability <= 1.0


# --------------------------------------------------------------------------- #
# Regressions
# --------------------------------------------------------------------------- #


def test_confident_dynamic_gesture_suppresses_commit_selection_evidence(cfg):
    """A swipe ends exactly like a deliberate stop; it must not become a click.

    Measured on the swipe scenario before this rule existed: 2 swipes produced
    2 clicks, because the end of each stroke fired a MICRO_PAUSE commit at score
    1.0 and CommitEvidence turned it into SELECT.
    """
    from gesture_engine.intent.evidence import CommitEvidence
    from gesture_engine.types import CommitSignal, GestureResult, Intent

    provider = CommitEvidence(cfg=cfg)
    commit = CommitSignal(committed=True, score=1.0, kind="MICRO_PAUSE", timestamp=1.0)

    swipe = GestureResult(gesture="SWIPE_LEFT", confidence=1.0, stability=1.0, timestamp=1.0)
    assert provider.logits(commit=commit, gesture=swipe) == {}

    # ... but a weak dynamic candidate does not get to veto the commit.
    weak = GestureResult(gesture="SWIPE_LEFT", confidence=0.2, stability=0.5, timestamp=1.0)
    assert provider.logits(commit=commit, gesture=weak).get(Intent.SELECT, 0.0) > 0.0

    # ... and a static pose leaves the commit untouched (click-without-pinch).
    static = GestureResult(gesture="INDEX_UP", confidence=1.0, stability=1.0, timestamp=1.0)
    assert provider.logits(commit=commit, gesture=static).get(Intent.SELECT, 0.0) > 0.0


def test_scroll_scores_while_the_stroke_is_still_happening(cfg):
    """A scroll has no natural end; a duration gate delays it until the user stops.

    The swipe duration gate used to apply to scroll too, and the score only
    saturated ~0.4 s *after* the stroke finished.
    """
    history = HistoryBuffer(capacity=200)
    # A 0.9 s slow vertical stroke, sampled at 30 FPS.
    n = 27
    pts = [(0.5, 0.62 - 0.30 * (i / (n - 1))) for i in range(n)]
    _push_motion(history, pts, [0.35] * n)
    scores = score_dynamic(history, cfg, extended_count=1)
    assert scores.get("SCROLL_UP", 0.0) > 0.5, scores


def test_stroke_peak_speed_is_measured_over_the_stroke(cfg):
    """A single fast frame must not suppress scrolling for a whole second.

    ``slow_enough`` used to read the peak over the entire history window, so a
    tracker jump anywhere in the last second killed every scroll score.  Here a
    fast burst sits inside the window but *outside* the stroke, separated from it
    by stationary samples.
    """
    history = HistoryBuffer(capacity=200)
    jump_pts = [(0.5, 0.62)] * 3
    jump_speeds = [4.0] * 3
    gap_pts = [(0.5, 0.62)] * 3
    gap_speeds = [0.0] * 3
    stroke_pts = [(0.5, 0.62 - 0.01 * i) for i in range(20)]
    stroke_speeds = [0.3] * 20
    _push_motion(history, jump_pts + gap_pts + stroke_pts, jump_speeds + gap_speeds + stroke_speeds)
    scores = score_dynamic(history, cfg, extended_count=1)
    assert scores.get("SCROLL_UP", 0.0) > 0.5, scores
