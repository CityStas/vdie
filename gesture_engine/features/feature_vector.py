"""Assembles :class:`HandFeatures` from raw landmarks and flattens them into a
fixed-length vector for the (optional) temporal model.

The vector layout is frozen and versioned: ``FEATURE_VECTOR_VERSION``.  Any
change to the layout must bump the version, because recordings store feature
vectors and an old recording must remain replayable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EngineConfig, FeaturesConfig
from ..types import (
    FINGER_CHAINS,
    INDEX_MCP,
    INDEX_TIP,
    PALM_CENTROID_INDICES,
    WRIST,
    HandFeatures,
)
from .angles import (
    all_finger_angles,
    finger_spread,
    finger_states,
    hand_span,
    index_orientation,
    mean_curl,
    middle_pinch_distance,
    palm_rotation_deg,
    pinch_distance,
    pinch_strength,
    thumb_reach,
)
from .geometry import EPS
from .normalization import build_palm_frame, normalize, normalize_rotation_invariant

FEATURE_VECTOR_VERSION = 2

#: Scalar (non-block) part of the vector, in the order :func:`flatten` emits it.
_SCALAR_FEATURES: tuple[str, ...] = (
    "index_orientation_norm",
    "pinch_distance",
    "middle_pinch_distance",
    "pinch_strength",
    "hand_span",
    "palm_rotation_norm",
    "depth_norm",
    "extended_count_norm",
    "curl",
    "thumb_reach",
    "palm_center_x",
    "palm_center_y",
)

#: Human-readable layout of :meth:`flatten`.
FEATURE_LAYOUT: tuple[str, ...] = (
    "fingers[5]",
    "pip_angles[5]",
    "dip_angles[5]",
    "finger_spread[5]",
    *_SCALAR_FEATURES,
)

#: Derived from the layout so the two can never drift apart.  The old literal
#: (``5 * 4 + 8 + 4``) claimed 32 while ``flatten`` emitted 31 — a silently
#: truncated vector for every downstream consumer.
FEATURE_VECTOR_SIZE = 5 * 4 + len(_SCALAR_FEATURES)


def control_point(points: np.ndarray, which: str = "index_tip") -> np.ndarray:
    """The 2-D point that drives cursor / motion analysis."""
    p = np.asarray(points, dtype=np.float64)
    if which == "index_tip":
        return p[INDEX_TIP, :2].copy()
    if which == "index_mcp":
        return p[INDEX_MCP, :2].copy()
    if which == "wrist":
        return p[WRIST, :2].copy()
    if which == "palm":
        return p[list(PALM_CENTROID_INDICES)].mean(axis=0)[:2]
    raise ValueError(f"unknown control_point: {which!r}")


def build_features(points: np.ndarray, cfg: EngineConfig) -> HandFeatures:
    fcfg: FeaturesConfig = cfg.features
    p = np.asarray(points, dtype=np.float64)

    if fcfg.rotation_invariant:
        normalized, frame = normalize_rotation_invariant(p)
    else:
        normalized, center, scale = normalize(p)
        frame = build_palm_frame(p)

    angles = all_finger_angles(p)
    pip_angles = tuple(a[0] for a in angles)
    dip_angles = tuple(a[1] for a in angles)

    pd = pinch_distance(p)
    depth = frame.scale / max(fcfg.reference_palm_scale, EPS)

    return HandFeatures(
        normalized=normalized,
        palm_center=frame.center,
        palm_scale=frame.scale,
        palm_frame=frame.basis,
        fingers=finger_states(p, fcfg),
        finger_angles=pip_angles,
        finger_tip_angles=dip_angles,
        finger_spread=finger_spread(p),
        index_orientation=index_orientation(p),
        pinch_distance=pd,
        pinch_strength=pinch_strength(pd, fcfg),
        middle_pinch_distance=middle_pinch_distance(p),
        hand_span=hand_span(p),
        palm_rotation=palm_rotation_deg(p),
        depth=float(depth),
        extended_count=int(sum(finger_states(p, fcfg))),
        curl=mean_curl(p, fcfg.extension_angle),
        thumb_reach=thumb_reach(p),
    )


def flatten(features: HandFeatures, motion: np.ndarray | None = None) -> np.ndarray:
    """Fixed-length vector: geometric features + optional motion block.

    ``motion`` is expected to be ``[vx, vy, ax, ay]``.  With motion the vector is
    :data:`FEATURE_VECTOR_SIZE` + 4 long.
    """
    base = np.concatenate(
        [
            np.asarray(features.fingers, dtype=np.float64),
            np.asarray(features.finger_angles, dtype=np.float64),
            np.asarray(features.finger_tip_angles, dtype=np.float64),
            np.asarray(features.finger_spread, dtype=np.float64),
            [
                features.index_orientation / 180.0,
                features.pinch_distance,
                features.middle_pinch_distance,
                features.pinch_strength,
                features.hand_span,
                features.palm_rotation / 180.0,
                features.depth,
                features.extended_count / 5.0,
                features.curl,
                features.thumb_reach,
                features.palm_center[0],
                features.palm_center[1],
            ],
        ]
    )
    if motion is not None:
        base = np.concatenate([base, np.asarray(motion, dtype=np.float64).ravel()[:4]])
    return base


def frame_sequence(
    frames: list[tuple[HandFeatures, np.ndarray]],
    length: int,
    motion_dim: int = 4,
) -> np.ndarray:
    """Stack the last ``length`` frames into a ``(length, D)`` array.

    Short histories are left-padded by repeating the first available frame so
    the temporal model always sees a fixed-size input.
    """
    dim = FEATURE_VECTOR_SIZE + motion_dim
    if not frames:
        return np.zeros((length, dim), dtype=np.float64)
    vecs = [flatten(f, m) for f, m in frames[-length:]]
    out = np.zeros((length, dim), dtype=np.float64)
    if len(vecs) < length:
        pad = [vecs[0]] * (length - len(vecs))
        vecs = pad + vecs
    arr = np.asarray(vecs, dtype=np.float64)
    out[:, : arr.shape[1]] = arr[:, :dim]
    return out


def describe_vector() -> str:
    return (
        f"feature vector v{FEATURE_VECTOR_VERSION} ({FEATURE_VECTOR_SIZE} dims): "
        + ", ".join(FEATURE_LAYOUT)
    )
