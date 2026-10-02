"""Geometry and normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_engine.features import geometry as G
from gesture_engine.features import normalization as N
from gesture_engine.features.angles import index_orientation, pinch_distance
from gesture_engine.tracking.landmark_model import POSES, build_hand, make_pose


def test_distance_is_2d():
    assert G.distance(np.array([0, 0, 5.0]), np.array([3, 4, 0.0])) == pytest.approx(5.0)


def test_joint_angle_straight_is_pi():
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 1.0])
    c = np.array([0.0, 2.0])
    assert G.joint_angle(a, b, c) == pytest.approx(np.pi)


def test_joint_angle_right_angle():
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 1.0])
    c = np.array([1.0, 1.0])
    assert G.joint_angle(a, b, c) == pytest.approx(np.pi / 2)


def test_orientation_convention():
    assert G.orientation_from_vertical(np.array([0.0, -1.0])) == pytest.approx(0.0)
    assert G.orientation_from_vertical(np.array([1.0, 0.0])) == pytest.approx(90.0)
    assert G.orientation_from_vertical(np.array([-1.0, 0.0])) == pytest.approx(-90.0)
    assert abs(G.orientation_from_vertical(np.array([0.0, 1.0]))) == pytest.approx(180.0)


def test_angle_difference_wraps():
    assert G.angle_difference(0.1, 2 * np.pi - 0.1) == pytest.approx(0.2, abs=1e-6)


def test_polyline_and_straightness():
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert G.polyline_length(line) == pytest.approx(2.0)
    assert G.straightness(line) == pytest.approx(1.0)
    bent = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    assert G.straightness(bent) == pytest.approx(np.sqrt(2) / 2)


def test_curvature_radius_infinite_when_collinear():
    line = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert G.curvature_radius(line) == float("inf")


def test_smoothstep_bounds():
    assert G.smoothstep(0.0, 1.0, -1.0) == 0.0
    assert G.smoothstep(0.0, 1.0, 2.0) == 1.0
    assert G.smoothstep(0.0, 1.0, 0.5) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #


def test_palm_scale_is_wrist_to_middle_mcp():
    pts = build_hand(make_pose("OPEN_PALM"))
    s = N.palm_scale(pts)
    assert s == pytest.approx(np.linalg.norm(pts[9, :2] - pts[0, :2]))


def test_normalization_is_scale_and_translation_invariant():
    a = build_hand(make_pose("INDEX_UP", position=(0.4, 0.6), scale=0.15))
    b = build_hand(make_pose("INDEX_UP", position=(0.6, 0.3), scale=0.30))
    na, _, _ = N.normalize(a)
    nb, _, _ = N.normalize(b)
    assert np.allclose(na, nb, atol=1e-9)


def test_normalization_is_not_rotation_invariant_by_default():
    a = build_hand(make_pose("INDEX_UP", rotation_deg=0))
    b = build_hand(make_pose("INDEX_UP", rotation_deg=45))
    na, _, _ = N.normalize(a)
    nb, _, _ = N.normalize(b)
    assert not np.allclose(na, nb, atol=1e-3)


def test_rotation_invariant_normalization_removes_roll():
    a = build_hand(make_pose("INDEX_UP", rotation_deg=0))
    b = build_hand(make_pose("INDEX_UP", rotation_deg=50))
    na, _ = N.normalize_rotation_invariant(a)
    nb, _ = N.normalize_rotation_invariant(b)
    assert np.allclose(na, nb, atol=1e-6)


def test_palm_rotation_matches_roll():
    for roll in (0.0, 30.0, -70.0, 120.0):
        pts = build_hand(make_pose("OPEN_PALM", rotation_deg=roll))
        assert np.degrees(N.palm_rotation(pts)) == pytest.approx(roll, abs=1e-6)


def test_index_orientation_tracks_rotation():
    # The rest pose has its own offset (the index chain is 14 deg off the palm
    # axis, plus the small cumulative bend of a nearly straight finger), so the
    # invariant under test is the *delta*, not an absolute angle.
    base = index_orientation(build_hand(POSES["INDEX_UP"]))
    for roll in (0.0, 40.0, -40.0):
        pts = build_hand(make_pose("INDEX_UP", rotation_deg=roll))
        assert index_orientation(pts) == pytest.approx(base + roll, abs=1.0)


def test_pinch_distance_zero_for_forced_pinch():
    pts = build_hand(make_pose("PINCH"))
    assert pinch_distance(pts) == pytest.approx(0.0, abs=1e-9)
