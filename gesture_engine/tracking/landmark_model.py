"""Analytic hand model + synthetic pose generator.

Why this exists: an engine that can only be exercised with a camera attached is
an engine you cannot regression-test.  This module builds a plausible 21-point
hand from a compact parametric description (per-finger curl, palm roll, position,
scale), which gives us:

* **unit tests** with known ground truth ("this is INDEX_UP, the recogniser must
  say INDEX_UP");
* **synthetic recordings** so the replay/benchmark harness has data on day one;
* **a smoke-test source** for CI where no camera exists.

It is a *kinematic approximation*, not a biomechanical model.  That is fine —
its only job is to be self-consistent with :mod:`gesture_engine.features`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..features.geometry import rotate_2d

# --------------------------------------------------------------------------- #
# Skeleton definition, in units of palm scale (wrist -> middle MCP == 1.0)
# --------------------------------------------------------------------------- #

#: (x, y) offsets relative to the wrist, y negative = "up" in image coordinates.
MCP_POSITIONS: dict[str, tuple[float, float]] = {
    "index": (-0.42, -0.85),
    "middle": (0.00, -1.00),
    "ring": (0.36, -0.93),
    "pinky": (0.68, -0.78),
}
THUMB_CMC_POSITION = (-0.45, -0.15)

#: (proximal, middle, distal) bone lengths.
BONE_LENGTHS: dict[str, tuple[float, float, float]] = {
    "thumb": (0.32, 0.30, 0.22),
    "index": (0.42, 0.25, 0.18),
    "middle": (0.46, 0.28, 0.19),
    "ring": (0.42, 0.26, 0.18),
    "pinky": (0.33, 0.20, 0.16),
}

#: Rest direction of each finger, in degrees from "up" (positive = toward pinky).
REST_ANGLES_DEG: dict[str, float] = {
    "thumb": -58.0,
    "index": -14.0,
    "middle": 0.0,
    "ring": 15.0,
    "pinky": 30.0,
}

#: Maximum bend at (MCP, PIP, DIP) in degrees when curl == 1.
MAX_BEND_DEG: dict[str, tuple[float, float, float]] = {
    "thumb": (40.0, 75.0, 70.0),
    "index": (75.0, 110.0, 70.0),
    "middle": (78.0, 112.0, 72.0),
    "ring": (76.0, 110.0, 70.0),
    "pinky": (72.0, 105.0, 68.0),
}

FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")


@dataclass(slots=True)
class HandPose:
    """Compact parametric description of a hand configuration."""

    #: Per-finger curl 0 = straight, 1 = fully folded.  Order: thumb..pinky.
    curl: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    #: Palm roll in degrees (0 = fingers up).
    rotation_deg: float = 0.0
    #: Wrist position in image-normalized coordinates.
    position: tuple[float, float] = (0.5, 0.7)
    #: Palm scale in image-normalized units.
    scale: float = 0.22
    #: Force the thumb tip onto the index tip (used to synthesise PINCH poses,
    #: because pure forward kinematics will not make two independent chains meet).
    force_pinch: bool = False
    #: Extra per-finger spread offset in degrees.
    spread_deg: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    def with_curl(self, **kwargs: float) -> "HandPose":
        values = list(self.curl)
        for name, value in kwargs.items():
            values[FINGER_ORDER.index(name)] = float(value)
        return HandPose(
            curl=tuple(values),  # type: ignore[arg-type]
            rotation_deg=self.rotation_deg,
            position=self.position,
            scale=self.scale,
            force_pinch=self.force_pinch,
            spread_deg=self.spread_deg,
        )


def _finger_points(name: str, curl: float, spread_deg: float) -> np.ndarray:
    """Forward kinematics for one finger chain, in palm-local units.

    Returns 4 points: root (MCP/CMC), j1, j2, tip.
    """
    if name == "thumb":
        root = np.array(THUMB_CMC_POSITION, dtype=np.float64)
    else:
        root = np.array(MCP_POSITIONS[name], dtype=np.float64)

    proximal, middle, distal = BONE_LENGTHS[name]
    bend_mcp, bend_pip, bend_dip = MAX_BEND_DEG[name]

    # Rest direction points "up" (-y); the angle grows towards the pinky side.
    angle = np.radians(REST_ANGLES_DEG[name] + spread_deg)
    direction = np.array([np.sin(angle), -np.cos(angle)], dtype=np.float64)

    pts = [root.copy()]
    seg_bends = (bend_mcp * curl, bend_pip * curl, bend_dip * curl)
    lengths = (proximal, middle, distal)
    for length, bend in zip(lengths, seg_bends):
        # Rotating by +bend curls the finger towards the palm (clockwise in
        # image coordinates when the hand points up).
        theta = np.radians(bend)
        c, s = np.cos(theta), np.sin(theta)
        direction = np.array([c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]])
        pts.append(pts[-1] + direction * length)
    return np.asarray(pts, dtype=np.float64)


def build_hand(pose: HandPose, noise: float = 0.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return a ``(21, 3)`` landmark array in image-normalized coordinates."""
    chains: list[np.ndarray] = []
    for i, name in enumerate(FINGER_ORDER):
        chains.append(_finger_points(name, float(np.clip(pose.curl[i], 0.0, 1.0)), pose.spread_deg[i]))

    # Assemble in MediaPipe index order: wrist, thumb(1-4), index(5-8), ...
    pts = [np.array([0.0, 0.0], dtype=np.float64)]
    pts.extend(chains[0][0:])  # thumb: CMC, MCP, IP, TIP
    for chain in chains[1:]:
        pts.extend(chain[0:])
    local = np.asarray(pts, dtype=np.float64)

    if pose.force_pinch:
        # thumb tip = index 4, index tip = index 8
        local[4] = local[8]

    local = rotate_2d(local, np.radians(pose.rotation_deg))
    local = local * pose.scale + np.asarray(pose.position, dtype=np.float64)

    if noise > 0.0:
        rng = rng if rng is not None else np.random.default_rng(0)
        local = local + rng.normal(0.0, noise * pose.scale, size=local.shape)

    z = np.zeros((local.shape[0], 1), dtype=np.float64)
    return np.hstack([local, z])


# --------------------------------------------------------------------------- #
# Pose presets — the vocabulary the tests and the synthetic source use
# --------------------------------------------------------------------------- #

POSES: dict[str, HandPose] = {
    "OPEN_PALM": HandPose(curl=(0.0, 0.0, 0.0, 0.0, 0.0), spread_deg=(-6, 0, 0, 0, 0)),
    "FIST": HandPose(curl=(0.55, 1.0, 1.0, 1.0, 1.0)),
    "INDEX_UP": HandPose(curl=(0.72, 0.03, 1.0, 1.0, 1.0)),
    "INDEX_SIDE": HandPose(curl=(0.72, 0.03, 1.0, 1.0, 1.0), rotation_deg=90.0),
    "V_SIGN": HandPose(curl=(0.88, 0.03, 0.03, 1.0, 1.0), spread_deg=(0, -12, 12, 0, 0)),
    "THREE": HandPose(curl=(0.85, 0.03, 0.03, 0.03, 1.0)),
    "PINCH": HandPose(curl=(0.42, 0.42, 1.0, 1.0, 1.0), force_pinch=True),
    "MIDDLE_PINCH": HandPose(curl=(0.40, 1.0, 0.40, 1.0, 1.0), force_pinch=False),
    "ROCK": HandPose(curl=(0.70, 0.03, 1.0, 1.0, 0.03)),
}


def pose(name: str) -> HandPose:
    try:
        return POSES[name]
    except KeyError as exc:
        raise KeyError(f"unknown pose {name!r}; known: {sorted(POSES)}") from exc


def wrist_for_tip(
    name: str,
    tip: tuple[float, float],
    *,
    landmark: int = 8,
    rotation_deg: float | None = None,
    scale: float | None = None,
) -> tuple[float, float]:
    """Wrist position that puts ``landmark`` (default 8 = index tip) at ``tip``.

    Scenarios are written in terms of *where the cursor is*, but
    :attr:`HandPose.position` is the **wrist**.  With ``INDEX_UP`` the index tip
    sits about 0.37 image units above the wrist, so the two are not
    interchangeable: the click study placed the wrist on each target, which left
    the control point 0.39 units away from it.  The whole target model —
    ``contains``, approach detection, the gravity field, ``TARGET_ENTRY`` — never
    engaged in any benchmark run, and the flagship "predictive target
    acquisition" feature was measuring nothing.

    The hand model is affine in ``position`` (``points = local * scale +
    position``), so the inverse is exact rather than an approximation.
    """
    base = make_pose(name, rotation_deg=rotation_deg, position=(0.0, 0.0), scale=scale)
    offset = build_hand(base)[landmark, :2]
    return (float(tip[0] - offset[0]), float(tip[1] - offset[1]))


def make_pose(
    name: str,
    *,
    rotation_deg: float | None = None,
    position: tuple[float, float] | None = None,
    scale: float | None = None,
) -> HandPose:
    base = pose(name)
    return HandPose(
        curl=base.curl,
        rotation_deg=base.rotation_deg if rotation_deg is None else rotation_deg,
        position=base.position if position is None else position,
        scale=base.scale if scale is None else scale,
        force_pinch=base.force_pinch,
        spread_deg=base.spread_deg,
    )


@dataclass(slots=True)
class Keyframe:
    """A pose held over a time interval, with optional linear interpolation."""

    t: float
    pose: HandPose
    #: If set, the pose is interpolated towards the *next* keyframe over this
    #: many seconds.  0 = hold, then jump.
    blend: float = 0.0
    tags: dict = field(default_factory=dict)
