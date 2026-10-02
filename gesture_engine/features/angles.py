"""Angular features: finger extension, orientation, curl.

Finger state is derived from *joint angles*, not from raw image coordinates.
That is the single most important robustness decision in the feature layer:
``tip.y < pip.y`` breaks the moment the hand rolls 40 degrees, whereas
``angle(MCP, PIP, DIP) > 100 deg`` does not.
"""

from __future__ import annotations

import numpy as np

from ..config import FeaturesConfig
from ..types import (
    FINGER_CHAINS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_TIP,
    PALM_CENTROID_INDICES,
    PINKY_TIP,
    RING_TIP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
)
from .geometry import EPS, distance, joint_angle, orientation_from_vertical, safe_normalize
from .normalization import palm_axis, palm_scale

#: Chain layout inside ``FINGER_CHAINS``: (root, joint1, joint2, tip).
_J0, _J1, _J2, _TIP = 0, 1, 2, 3


def finger_joint_angles(points: np.ndarray, finger: int) -> tuple[float, float]:
    """``(angle at joint1, angle at joint2)`` for one finger chain."""
    p = np.asarray(points, dtype=np.float64)
    c0, c1, c2, c3 = FINGER_CHAINS[finger]
    return (
        joint_angle(p[c0], p[c1], p[c2]),
        joint_angle(p[c1], p[c2], p[c3]),
    )


def all_finger_angles(points: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple(finger_joint_angles(points, f) for f in range(5))


def finger_extended(points: np.ndarray, finger: int, cfg: FeaturesConfig) -> bool:
    """Joint-angle based extension test.

    The thumb gets its own threshold because its CMC joint is far more mobile
    than the MCP joints of the other fingers.
    """
    a1, a2 = finger_joint_angles(points, finger)
    threshold = cfg.thumb_extension_angle if finger == 0 else cfg.extension_angle
    if cfg.require_both_joints:
        primary = a1 > threshold and a2 > cfg.dip_extension_angle
    else:
        primary = a1 > threshold
    if finger != 0:
        return bool(primary)
    # Thumb: joint angle alone over-triggers when the hand is closed but the
    # thumb lies flat across the fingers.  Require reach as well.
    return bool(primary and thumb_reach(points) > 0.75)


def thumb_reach(points: np.ndarray) -> float:
    """Distance from the thumb tip to the palm centre, in palm-scale units."""
    p = np.asarray(points, dtype=np.float64)
    center = p[list(PALM_CENTROID_INDICES)].mean(axis=0)
    return distance(p[THUMB_TIP], center) / max(palm_scale(p), EPS)


def finger_states(points: np.ndarray, cfg: FeaturesConfig) -> tuple[int, int, int, int, int]:
    """Compact state vector ``[thumb, index, middle, ring, pinky]``."""
    return tuple(1 if finger_extended(points, f, cfg) else 0 for f in range(5))  # type: ignore[return-value]


def index_orientation(points: np.ndarray) -> float:
    """Tilt of the index finger from vertical, in degrees.

    ``0`` = up, ``+90`` = right, ``-90`` = left, ``+-180`` = down.
    """
    p = np.asarray(points, dtype=np.float64)
    v = p[INDEX_TIP, :2] - p[INDEX_MCP, :2]
    return orientation_from_vertical(v)


#: Interior joint angle (rad) at which a finger counts as *fully folded*.
#:
#: Normalising against ``pi`` (the mathematically "straightest" angle) looks
#: natural but is wrong in practice: a closed finger's PIP joint sits at ~70 deg
#: and its DIP at ~110 deg, so the naive ``1 - angle/pi`` yields 0.5 for a fist
#: and every consumer that assumes "1.0 = fist" silently mis-calibrates.  The
#: reference below is the anatomical fold angle, so ``curl`` really does span
#: ``0 = flat .. 1 = fist``.
FOLDED_ANGLE = 1.75  # rad, ~100 deg


def finger_curl(points: np.ndarray, finger: int, folded_angle: float = FOLDED_ANGLE) -> float:
    """0 = fully straight, 1 = fully folded, from the two joint angles.

    ``folded_angle`` is the interior joint angle that counts as a full fold; pass
    ``features.extension_angle`` to keep it consistent with the extension test.
    """
    a1, a2 = finger_joint_angles(points, finger)
    span = max(np.pi - float(folded_angle), EPS)
    c1 = (np.pi - a1) / span
    c2 = (np.pi - a2) / span
    return float(np.clip(0.5 * (c1 + c2), 0.0, 1.0))


def mean_curl(points: np.ndarray, folded_angle: float = FOLDED_ANGLE) -> float:
    """Mean curl of index..pinky; the thumb is excluded (too mobile)."""
    return float(np.mean([finger_curl(points, f, folded_angle) for f in range(1, 5)]))


def finger_spread(points: np.ndarray) -> tuple[float, ...]:
    """Perpendicular distance of each tip from the palm axis, palm-scale units."""
    p = np.asarray(points, dtype=np.float64)
    axis = palm_axis(p)
    side = np.array([-axis[1], axis[0]])
    origin = p[WRIST, :2]
    out: list[float] = []
    for f in range(5):
        tip = p[FINGER_CHAINS[f][_TIP], :2]
        out.append(float(np.dot(tip - origin, side) / max(palm_scale(p), EPS)))
    return tuple(out)


def hand_span(points: np.ndarray) -> float:
    """Largest tip-to-tip distance, in palm-scale units.  Proxy for openness."""
    p = np.asarray(points, dtype=np.float64)
    tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    best = 0.0
    for i, ti in enumerate(tips):
        for tj in tips[i + 1 :]:
            best = max(best, distance(p[ti], p[tj]))
    return float(best / max(palm_scale(p), EPS))


def pinch_distance(points: np.ndarray) -> float:
    """Thumb-tip to index-tip distance in palm-scale units (V1 pinch feature)."""
    p = np.asarray(points, dtype=np.float64)
    return float(distance(p[THUMB_TIP], p[INDEX_TIP]) / max(palm_scale(p), EPS))


def middle_pinch_distance(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)
    return float(distance(p[THUMB_TIP], p[MIDDLE_TIP]) / max(palm_scale(p), EPS))


def pinch_strength(distance_norm: float, cfg: FeaturesConfig) -> float:
    """Map the pinch distance onto a continuous 0..1 "how hard is the pinch".

    This is the flagship *continuous* channel (master prompt section 53): the
    hand is not a button array, so the analog value is preserved rather than
    thresholded away.
    """
    lo, hi = cfg.pinch_strength_closed, cfg.pinch_strength_open
    if hi - lo < EPS:
        return 0.0
    return float(np.clip((hi - distance_norm) / (hi - lo), 0.0, 1.0))


def palm_rotation_deg(points: np.ndarray) -> float:
    from .normalization import palm_rotation

    return float(np.degrees(palm_rotation(points)))


def direction_unit(points: np.ndarray, finger: int) -> np.ndarray:
    """Unit vector along a finger chain (root -> tip)."""
    p = np.asarray(points, dtype=np.float64)
    c0, _, _, c3 = FINGER_CHAINS[finger]
    return safe_normalize(p[c3, :2] - p[c0, :2])
