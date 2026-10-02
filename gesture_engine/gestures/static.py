"""Static (pose) gesture scoring.

Every recogniser returns a *score in [0, 1]* per gesture, never a boolean.  The
score is a weighted geometric mean of soft conditions, so a partially-formed
gesture degrades smoothly instead of flipping.  Thresholding happens once, at the
very end, in :mod:`gesture_engine.gestures.recognizer`.

Note on the V1 spec: it defines ``[1,0,0,0,0] == index up``, i.e. the thumb must
also be folded.  In practice the thumb is the least reliable landmark and the
most common source of false negatives for ``INDEX_UP``, so thumb state is
*excluded* from the ``INDEX_UP`` / ``POINT`` conditions by default and the
thumb-dependent gestures get their own rules.  The stricter V1 behaviour is
available via ``features.require_both_joints``-style config, but is not the
default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EngineConfig
from ..features.geometry import clamp01, smoothstep
from ..types import FINGER_NAMES, HandFeatures


@dataclass(frozen=True, slots=True)
class Condition:
    """One soft condition contributing to a gesture score."""

    name: str
    weight: float = 1.0


def _ext_conf(features: HandFeatures, finger: int, cfg: EngineConfig) -> float:
    """Soft extension confidence for one finger, in [0, 1].

    Two joint tests, blended 0.65 / 0.35.  The distal one uses its own threshold
    (``features.dip_extension_angle``): the DIP's usable range is much higher than
    the PIP's, so a folded finger (DIP ~110 deg) has to fall *below* it.  Reusing
    a fraction of the PIP threshold made a folded finger score 0.35 here, which
    leaked into ``fold = 1 - ext`` and let a fist score as a pinch.
    """
    thr = cfg.features.thumb_extension_angle if finger == 0 else cfg.features.extension_angle
    pip = features.finger_angles[finger]
    dip = features.finger_tip_angles[finger]
    band = 0.30
    primary = smoothstep(thr - band, thr + band, pip)
    secondary = smoothstep(cfg.features.dip_extension_angle - band, cfg.features.dip_extension_angle + band, dip)
    conf = 0.65 * primary + 0.35 * secondary
    if finger == 0:
        # Joint angles alone cannot separate "thumb out" from "thumb tucked across
        # the palm": in both cases the joints are open.  ``finger_extended`` has
        # always required ``thumb_reach > 0.75``; the soft path needs the same
        # term, otherwise a fist scores as a thumbs-up.
        conf *= smoothstep(0.55, 0.80, features.thumb_reach)
    return clamp01(conf)


def _fold_conf(features: HandFeatures, finger: int, cfg: EngineConfig) -> float:
    return clamp01(1.0 - _ext_conf(features, finger, cfg))


def _pinch_conf(features: HandFeatures, cfg: EngineConfig, which: str = "index") -> float:
    d = features.pinch_distance if which == "index" else features.middle_pinch_distance
    # 1.0 when closed (<= pinch_down), 0.0 when open (>= pinch_release).
    return clamp01(1.0 - smoothstep(cfg.gestures.pinch_down, cfg.gestures.pinch_release, d))


def _orientation_conf(features: HandFeatures, target_deg: float, tolerance_deg: float) -> float:
    diff = abs((features.index_orientation - target_deg + 180.0) % 360.0 - 180.0)
    return clamp01(1.0 - smoothstep(tolerance_deg * 0.5, tolerance_deg, diff))


def _geometric_mean(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    v = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0)
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        return 0.0
    return float(np.exp((w * np.log(v)).sum() / w.sum()))


# --------------------------------------------------------------------------- #
# Gesture definitions
# --------------------------------------------------------------------------- #


def score_static(features: HandFeatures, cfg: EngineConfig) -> dict[str, float]:
    """Score every static gesture against the current frame."""
    ext = [_ext_conf(features, f, cfg) for f in range(5)]
    fold = [1.0 - e for e in ext]
    thumb, index, middle, ring, pinky = ext
    f_thumb, f_index, f_middle, f_ring, f_pinky = fold

    span_open = smoothstep(1.2, 2.2, features.hand_span)
    curl_closed = smoothstep(0.45, 0.80, features.curl)
    pinch_i = _pinch_conf(features, cfg, "index")
    pinch_m = _pinch_conf(features, cfg, "middle")
    strength = features.pinch_strength

    scores: dict[str, float] = {}

    scores["OPEN_PALM"] = _geometric_mean(
        [thumb, index, middle, ring, pinky, span_open],
        [1.0, 1.2, 1.2, 1.1, 1.0, 0.8],
    )
    scores["FIST"] = _geometric_mean(
        [f_index, f_middle, f_ring, f_pinky, curl_closed, 0.5 + 0.5 * f_thumb],
        [1.2, 1.2, 1.1, 1.0, 1.0, 0.5],
    )
    scores["INDEX_UP"] = _geometric_mean(
        [index, f_middle, f_ring, f_pinky, _orientation_conf(features, 0.0, 40.0)],
        [1.4, 1.1, 1.0, 0.9, 0.8],
    )
    scores["POINT"] = _geometric_mean(
        [index, f_middle, f_ring, f_pinky, max(_orientation_conf(features, 90.0, 45.0), _orientation_conf(features, -90.0, 45.0))],
        [1.3, 1.0, 0.9, 0.8, 0.6],
    )
    scores["V_SIGN"] = _geometric_mean(
        [index, middle, f_ring, f_pinky, _orientation_conf(features, 0.0, 60.0)],
        [1.2, 1.2, 1.1, 1.0, 0.4],
    )
    scores["THREE"] = _geometric_mean(
        [index, middle, ring, f_pinky],
        [1.1, 1.1, 1.1, 1.0],
    )
    scores["ROCK"] = _geometric_mean(
        [index, f_middle, f_ring, pinky],
        [1.2, 1.1, 1.0, 1.1],
    )
    scores["THUMB_UP"] = _geometric_mean(
        [thumb, f_index, f_middle, f_ring, f_pinky],
        [1.2, 1.0, 1.0, 0.9, 0.8],
    )
    # PINCH must beat MIDDLE_PINCH on the index side and vice versa.
    #
    # The index (respectively middle) *extension* factor is not cosmetic: a fist
    # also puts the thumb tip next to the folded index tip, so proximity alone
    # makes PINCH fire on a closed hand.  A real pinch is a *straight-ish* finger
    # meeting the thumb; the extension term is what separates the two.
    scores["PINCH"] = _geometric_mean(
        [pinch_i, 0.30 + 0.70 * index, 0.35 + 0.65 * f_middle, 0.5 + 0.5 * f_ring, 0.5 + 0.5 * f_pinky, 0.4 + 0.6 * strength],
        [1.6, 1.0, 1.0, 0.7, 0.6, 0.7],
    )
    scores["MIDDLE_PINCH"] = _geometric_mean(
        [pinch_m, 0.30 + 0.70 * middle, 0.35 + 0.65 * f_index, 0.5 + 0.5 * f_ring, 0.5 + 0.5 * f_pinky, 0.4 + 0.6 * strength],
        [1.6, 1.0, 1.0, 0.7, 0.6, 0.7],
    )
    return scores


def best_static(features: HandFeatures, cfg: EngineConfig) -> tuple[str, float, dict[str, float]]:
    scores = score_static(features, cfg)
    if not scores:
        return "NONE", 0.0, scores
    name = max(scores, key=lambda k: scores[k])
    return name, float(scores[name]), scores


def describe(features: HandFeatures) -> str:
    """Compact human-readable finger-state string for the overlay."""
    return "".join(str(x) for x in features.fingers) + " | " + " ".join(
        f"{n[0].upper()}{features.finger_angles[i]:.2f}" for i, n in enumerate(FINGER_NAMES)
    )
