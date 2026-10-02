"""Hand normalisation.

The V1 spec normalises by translation + scale only::

    P = palm centre
    S = |wrist - middle_MCP|
    p' = (p - P) / S

That removes hand size and camera distance but *not* hand roll.  A rotated hand
produces a different landmark cloud, which is why the V1 activation rule needs
an explicit ``index_angle_from_vertical < 25 deg`` guard.

We keep the V1 form as the default (so the baseline stays comparable) and add a
palm frame as an extra, optional representation.  Finger *joint angles* are
already rotation-invariant, so the palm frame mainly helps distance features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import (
    INDEX_MCP,
    MIDDLE_MCP,
    PALM_CENTROID_INDICES,
    WRIST,
)
from .geometry import EPS, safe_normalize


@dataclass(frozen=True, slots=True)
class PalmFrame:
    """Orthonormal 2-D basis attached to the palm."""

    center: np.ndarray  # (3,)
    scale: float
    rotation: float  # radians; angle of the wrist->middle-MCP axis from "up"
    basis: np.ndarray  # (2, 2) rows = (forward, side), in image axes

    def to_palm(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64).copy()
        p[:, :2] = (p[:, :2] - self.center[:2]) @ self.basis.T
        return p


def palm_center(points: np.ndarray) -> np.ndarray:
    """Centroid of wrist + the four MCPs (V1 definition)."""
    idx = list(PALM_CENTROID_INDICES)
    return np.asarray(points, dtype=np.float64)[idx].mean(axis=0)


def palm_scale(points: np.ndarray) -> float:
    """Wrist -> middle-MCP distance; the V1 scale reference."""
    p = np.asarray(points, dtype=np.float64)
    d = p[MIDDLE_MCP, :2] - p[WRIST, :2]
    return float(max(np.linalg.norm(d), EPS))


def normalize(points: np.ndarray, scale: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Translate to the palm centre and divide by palm scale.

    Returns ``(normalized_points, center, scale)``.
    """
    p = np.asarray(points, dtype=np.float64)
    center = palm_center(p)
    s = palm_scale(p) if scale is None else float(scale)
    out = (p - center) / max(s, EPS)
    return out, center, s


def palm_rotation(points: np.ndarray) -> float:
    """Angle of the wrist->middle-MCP axis relative to *up*, in radians.

    ``0`` = fingers pointing up, ``+pi/2`` = pointing right, ``pi`` = down.
    """
    p = np.asarray(points, dtype=np.float64)
    v = p[MIDDLE_MCP, :2] - p[WRIST, :2]
    return float(np.arctan2(v[0], -v[1]))


def palm_basis(points: np.ndarray) -> np.ndarray:
    """Rotation matrix mapping image axes -> palm axes (rows: forward, side)."""
    theta = palm_rotation(points)
    c, s = np.cos(theta), np.sin(theta)
    # Rotate by -theta so that the palm axis lands on -y (screen up).
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def build_palm_frame(points: np.ndarray) -> PalmFrame:
    p = np.asarray(points, dtype=np.float64)
    return PalmFrame(
        center=palm_center(p),
        scale=palm_scale(p),
        rotation=palm_rotation(p),
        basis=palm_basis(p),
    )


def normalize_rotation_invariant(points: np.ndarray) -> tuple[np.ndarray, PalmFrame]:
    """Full similarity normalisation: translation + scale + rotation."""
    p = np.asarray(points, dtype=np.float64)
    frame = build_palm_frame(p)
    out = (p - frame.center) / max(frame.scale, EPS)
    out[:, :2] = out[:, :2] @ frame.basis.T
    return out, frame


def palm_axis(points: np.ndarray) -> np.ndarray:
    """Unit vector from the wrist towards the middle MCP."""
    p = np.asarray(points, dtype=np.float64)
    return safe_normalize(p[MIDDLE_MCP, :2] - p[WRIST, :2])


def palm_side_axis(points: np.ndarray) -> np.ndarray:
    """Unit vector from the index MCP towards the pinky MCP."""
    from ..types import PINKY_MCP

    p = np.asarray(points, dtype=np.float64)
    return safe_normalize(p[PINKY_MCP, :2] - p[INDEX_MCP, :2])
