"""Feature extraction: finger states, pinch, orientation, vector layout."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_engine.features.angles import finger_curl, finger_joint_angles, finger_states
from gesture_engine.features.feature_vector import (
    FEATURE_VECTOR_SIZE,
    FEATURE_VECTOR_VERSION,
    build_features,
    control_point,
    describe_vector,
    flatten,
    frame_sequence,
)
from gesture_engine.tracking.landmark_model import POSES, build_hand, make_pose

EXPECTED_FINGERS = {
    "OPEN_PALM": (1, 1, 1, 1, 1),
    "FIST": (0, 0, 0, 0, 0),
    "INDEX_UP": (0, 1, 0, 0, 0),
    "V_SIGN": (0, 1, 1, 0, 0),
    "THREE": (0, 1, 1, 1, 0),
    "ROCK": (0, 1, 0, 0, 1),
    "PINCH": (0, 1, 0, 0, 0),
}


@pytest.mark.parametrize("name,expected", EXPECTED_FINGERS.items())
def test_finger_states(cfg, name, expected):
    pts = build_hand(POSES[name])
    assert finger_states(pts, cfg.features) == expected


def test_fist_has_high_curl():
    pts = build_hand(POSES["FIST"])
    assert finger_curl(pts, 1) > 0.6


def test_open_palm_has_low_curl():
    pts = build_hand(POSES["OPEN_PALM"])
    assert finger_curl(pts, 1) < 0.15


def test_finger_angles_are_rotation_invariant(cfg):
    a = build_hand(make_pose("INDEX_UP", rotation_deg=0))
    b = build_hand(make_pose("INDEX_UP", rotation_deg=55))
    assert finger_joint_angles(a, 1) == pytest.approx(finger_joint_angles(b, 1), abs=1e-6)


def test_features_pinch_strength_monotone(cfg):
    closed = build_features(build_hand(POSES["PINCH"]), cfg)
    open_hand = build_features(build_hand(POSES["OPEN_PALM"]), cfg)
    assert closed.pinch_strength > open_hand.pinch_strength
    assert closed.pinch_strength == pytest.approx(1.0)
    assert open_hand.pinch_strength == pytest.approx(0.0, abs=1e-6)


def test_depth_tracks_scale(cfg):
    near = build_features(build_hand(make_pose("OPEN_PALM", scale=0.40)), cfg)
    far = build_features(build_hand(make_pose("OPEN_PALM", scale=0.10)), cfg)
    assert near.depth > far.depth


def test_control_point_variants(cfg):
    pts = build_hand(POSES["INDEX_UP"])
    assert np.allclose(control_point(pts, "index_tip"), pts[8, :2])
    assert np.allclose(control_point(pts, "index_mcp"), pts[5, :2])
    assert np.allclose(control_point(pts, "wrist"), pts[0, :2])
    with pytest.raises(ValueError):
        control_point(pts, "nose")


def test_feature_vector_size(cfg):
    f = build_features(build_hand(POSES["OPEN_PALM"]), cfg)
    assert flatten(f).shape == (FEATURE_VECTOR_SIZE,)
    assert flatten(f, motion=np.zeros(4)).shape == (FEATURE_VECTOR_SIZE + 4,)
    # The version must be reported, and it must track the constant (recordings
    # store the vector, so a silent layout change would break replay).
    assert f"v{FEATURE_VECTOR_VERSION}" in describe_vector()


def test_frame_sequence_pads_short_history(cfg):
    f = build_features(build_hand(POSES["OPEN_PALM"]), cfg)
    seq = frame_sequence([(f, np.zeros(4))], length=30)
    assert seq.shape == (30, FEATURE_VECTOR_SIZE + 4)
    assert np.allclose(seq[0], seq[-1])


def test_frame_sequence_takes_the_tail(cfg):
    a = build_features(build_hand(POSES["OPEN_PALM"]), cfg)
    b = build_features(build_hand(POSES["FIST"]), cfg)
    frames = [(a, np.zeros(4))] * 5 + [(b, np.zeros(4))] * 5
    seq = frame_sequence(frames, length=3)
    assert np.allclose(seq[-1][:5], np.asarray(b.fingers))
