"""Core data types shared by every layer of the engine.

Design rule: the *only* thing that flows between layers is one of these
immutable-ish dataclasses.  No layer reaches into another layer's internals,
and no layer except ``tracking`` knows anything about MediaPipe, and no layer
except ``capture`` knows anything about cameras.

This is what makes deterministic replay possible: a recorded session is just a
list of ``Observation`` + the events/features derived from them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Landmarks
# --------------------------------------------------------------------------- #

# MediaPipe hand topology, kept here so no other module hard-codes magic numbers.
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_CHAINS: tuple[tuple[int, int, int, int], ...] = (
    (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)

FINGER_NAMES: tuple[str, ...] = ("thumb", "index", "middle", "ring", "pinky")

MCP_INDICES: tuple[int, ...] = (THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
PIP_INDICES: tuple[int, ...] = (THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
TIP_INDICES: tuple[int, ...] = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

#: Landmarks used to compute the palm centroid (V1 definition, kept verbatim).
PALM_CENTROID_INDICES: tuple[int, ...] = (WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)


@dataclass(frozen=True, slots=True)
class Landmarks:
    """A set of N 3-D points in image-normalized coordinates.

    ``points[:, 0]`` is x in ``[0, 1]`` (left -> right),
    ``points[:, 1]`` is y in ``[0, 1]`` (top -> bottom),
    ``points[:, 2]`` is a *relative* depth in roughly the same scale as x.
    """

    points: np.ndarray
    visibility: np.ndarray | None = None
    name: str = "hand"

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.points[idx]

    def xy(self) -> np.ndarray:
        return self.points[:, :2]

    def centroid(self) -> np.ndarray:
        return self.points.mean(axis=0)

    @staticmethod
    def from_sequence(seq: Sequence[Sequence[float]], name: str = "hand") -> "Landmarks":
        return Landmarks(points=np.asarray(seq, dtype=np.float64), name=name)


# --------------------------------------------------------------------------- #
# Observation / motion
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class MotionState:
    """Continuous motion state of the *control point*.

    This is deliberately continuous (no classification) — classification lives
    in the event/gesture layers.  Everything is expressed in normalized units
    per second so that it is frame-rate independent.
    """

    timestamp: float = 0.0
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2))
    jerk: float = 0.0
    speed: float = 0.0
    direction: float = 0.0  # radians, atan2(vy, vx)
    direction_change: float = 0.0  # radians since previous frame, signed
    curvature: float = 0.0  # 1/px, signed
    amplitude: float = 0.0  # path length over the recent window
    displacement: float = 0.0  # straight-line distance over the recent window
    path_efficiency: float = 1.0  # displacement / amplitude
    pause_duration: float = 0.0  # seconds of continuous near-zero speed
    is_pausing: bool = False
    confidence: float = 1.0
    #: Peak speed inside the statistics window, and how long ago it happened.
    #: These two are what make the kinematic commit point possible.
    speed_peak: float = 0.0
    time_since_peak: float = 0.0
    #: Signed speed trend: > 0 accelerating, < 0 decelerating.
    speed_trend: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy(),
            "acceleration": self.acceleration.copy(),
            "jerk": self.jerk,
            "speed": self.speed,
            "direction": self.direction,
            "curvature": self.curvature,
            "amplitude": self.amplitude,
            "pause_duration": self.pause_duration,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class Observation:
    """One sensor sample.  Generic on purpose: hands today, anything later."""

    timestamp: float
    hand: Landmarks | None = None
    hand_confidence: float = 0.0
    # Multi-hand representation. ``hand`` remains the primary-hand compatibility
    # field used by the V1 pipeline; ``hands`` contains every detected hand.
    hands: dict[str, Landmarks] = field(default_factory=dict)
    hand_confidences: dict[str, float] = field(default_factory=dict)
    primary_hand: str | None = None
    pose: Landmarks | None = None
    face: Landmarks | None = None
    camera: Mapping[str, Any] = field(default_factory=dict)
    frame_id: int = 0
    #: Filled in by the Motion Engine, not by the sensor adapter.
    motion: MotionState | None = None
    #: Filled in by the Feature Engine.
    features: "HandFeatures | None" = None

    @property
    def has_hand(self) -> bool:
        return self.hand is not None

    @property
    def hand_count(self) -> int:
        return len(self.hands) if self.hands else (1 if self.hand is not None else 0)

    def with_motion(self, motion: MotionState) -> "Observation":
        self.motion = motion
        return self

    def with_features(self, features: "HandFeatures") -> "Observation":
        self.features = features
        return self

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "hand_confidence": self.hand_confidence,
        }
        if self.hand is not None:
            d["hand"] = self.hand.points.tolist()
        if self.hands:
            d["hands"] = {k: v.points.tolist() for k, v in self.hands.items()}
            d["hand_confidences"] = dict(self.hand_confidences)
            d["primary_hand"] = self.primary_hand
        if self.motion is not None:
            m = self.motion
            d["motion"] = {
                "position": m.position.tolist(),
                "velocity": m.velocity.tolist(),
                "acceleration": m.acceleration.tolist(),
                "speed": m.speed,
                "direction": m.direction,
                "curvature": m.curvature,
                "amplitude": m.amplitude,
                "pause_duration": m.pause_duration,
                "confidence": m.confidence,
            }
        if self.features is not None:
            d["features"] = self.features.as_dict()
        return d


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class HandFeatures:
    """Deterministic geometric description of one hand in one frame."""

    normalized: np.ndarray  # (21, 3) hand-centered / palm-scaled
    palm_center: np.ndarray  # (3,)
    palm_scale: float
    palm_frame: np.ndarray  # (2, 2) rotation from image axes to palm axes
    fingers: tuple[int, int, int, int, int]
    finger_angles: tuple[float, ...]  # PIP angle per finger, radians
    finger_tip_angles: tuple[float, ...]  # DIP angle per finger, radians
    finger_spread: tuple[float, ...]  # tip distance from palm axis, normalized
    index_orientation: float  # degrees, 0 = up, +90 = right
    pinch_distance: float  # thumb_tip..index_tip, normalized by palm scale
    pinch_strength: float  # continuous 0..1 (1 = fully closed)
    middle_pinch_distance: float
    hand_span: float  # max pairwise distance among tips, normalized
    palm_rotation: float  # degrees, 0 = fingers pointing up
    depth: float  # palm scale proxy -> larger = closer to camera
    extended_count: int
    curl: float  # mean finger curl, 0 = flat, 1 = fully folded
    #: Thumb tip distance from the palm centre, in palm-scale units.  Exposed as
    #: a feature because "thumb out" vs "thumb tucked across the palm" is the only
    #: thing that separates THUMB_UP from FIST on the soft (non-boolean) path.
    thumb_reach: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "fingers": list(self.fingers),
            "finger_angles": list(self.finger_angles),
            "index_orientation": self.index_orientation,
            "pinch_distance": self.pinch_distance,
            "pinch_strength": self.pinch_strength,
            "middle_pinch_distance": self.middle_pinch_distance,
            "hand_span": self.hand_span,
            "palm_rotation": self.palm_rotation,
            "palm_scale": self.palm_scale,
            "depth": self.depth,
            "extended_count": self.extended_count,
            "curl": self.curl,
            "thumb_reach": self.thumb_reach,
        }


# --------------------------------------------------------------------------- #
# Gesture / event / intent
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GestureResult:
    """A gesture is *evidence*, never a command."""

    gesture: str
    confidence: float
    stability: float = 0.0
    duration: float = 0.0
    timestamp: float = 0.0
    source: str = "deterministic"

    @property
    def is_none(self) -> bool:
        return self.gesture == "NONE"


GESTURE_NONE = GestureResult(gesture="NONE", confidence=0.0)

#: Gestures that only exist as a *trajectory*; a single static frame cannot
#: produce them.  Lives here rather than in the recogniser because the intent
#: layer needs the same definition: a dynamic gesture is a complete command in
#: itself and must not be re-confirmed by a kinematic commit point.
DYNAMIC_GESTURES = frozenset(
    {"SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN", "SCROLL_UP", "SCROLL_DOWN"}
)


class EventType(str, Enum):
    POINT_START = "POINT_START"
    POINT_STOP = "POINT_STOP"
    ACCELERATE = "ACCELERATE"
    DECELERATE = "DECELERATE"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    DIRECTION_CHANGE = "DIRECTION_CHANGE"
    PINCH_START = "PINCH_START"
    PINCH_RELEASE = "PINCH_RELEASE"
    TARGET_APPROACH = "TARGET_APPROACH"
    TARGET_EXIT = "TARGET_EXIT"
    HAND_LOST = "HAND_LOST"
    HAND_REACQUIRED = "HAND_REACQUIRED"
    COMMIT = "COMMIT"
    CANCEL = "CANCEL"
    MODE_CHANGE = "MODE_CHANGE"
    GESTURE_ENTER = "GESTURE_ENTER"
    GESTURE_EXIT = "GESTURE_EXIT"


@dataclass(frozen=True, slots=True)
class Event:
    type: EventType
    timestamp: float
    confidence: float = 1.0
    source: str = "motion"
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CommitSignal:
    """Result of the commit-point detector."""

    committed: bool = False
    score: float = 0.0
    kind: str = "NONE"  # MICRO_PAUSE | VELOCITY_REVERSAL | THRUST | TARGET_ENTRY | PINCH | NONE
    timestamp: float = 0.0


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #


class TargetState(str, Enum):
    NEUTRAL = "NEUTRAL"
    APPROACH = "APPROACH"
    AVOID = "AVOID"


@dataclass(slots=True)
class Target:
    id: str
    bounds: tuple[float, float, float, float]  # x0, y0, x1, y1 in screen coords
    type: str = "unknown"
    semantic_label: str = ""
    visual_confidence: float = 0.0
    semantic_confidence: float = 0.0
    capabilities: tuple[str, ...] = ("hover", "click")
    state: TargetState = TargetState.NEUTRAL
    influence: float = 0.0
    approach_confidence: float = 0.0

    @property
    def center(self) -> np.ndarray:
        x0, y0, x1, y1 = self.bounds
        return np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5])

    @property
    def size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return (max(x1 - x0, 1.0), max(y1 - y0, 1.0))

    def contains(self, p: Sequence[float], margin: float = 0.0) -> bool:
        x0, y0, x1, y1 = self.bounds
        return (x0 - margin) <= p[0] <= (x1 + margin) and (y0 - margin) <= p[1] <= (y1 + margin)

    def quality(self) -> float:
        """How trustworthy this target is as an interaction candidate."""
        return float(np.clip(0.5 * self.semantic_confidence + 0.5 * self.visual_confidence, 0.0, 1.0))


@dataclass(slots=True)
class TargetBelief:
    target: Target
    confidence: float
    distance: float
    closing_speed: float = 0.0


# --------------------------------------------------------------------------- #
# Intent
# --------------------------------------------------------------------------- #


class Intent(str, Enum):
    UNKNOWN = "UNKNOWN"
    MOVE_CURSOR = "MOVE_CURSOR"
    SELECT = "SELECT"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    DRAG = "DRAG"
    RIGHT_CLICK = "RIGHT_CLICK"
    SCROLL = "SCROLL"
    WINDOW_SWITCH = "WINDOW_SWITCH"
    WINDOW_CONTROL = "WINDOW_CONTROL"
    PAUSE = "PAUSE"
    CANCEL = "CANCEL"
    CUSTOM = "CUSTOM"


@dataclass(slots=True)
class IntentField:
    """A distribution over intents plus the evidence that produced it.

    ``logits`` are kept so that downstream code can fuse or decay evidence
    without re-deriving it.
    """

    probabilities: dict[Intent, float] = field(default_factory=dict)
    logits: dict[Intent, float] = field(default_factory=dict)
    entropy: float = 0.0
    timestamp: float = 0.0
    dominant: Intent = Intent.UNKNOWN
    dominant_probability: float = 0.0
    committed: Intent | None = None

    def get(self, intent: Intent) -> float:
        return float(self.probabilities.get(intent, 0.0))

    def top_k(self, k: int = 3) -> list[tuple[Intent, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def as_dict(self) -> dict[str, float]:
        return {i.value: round(p, 4) for i, p in self.top_k(len(Intent))}


@dataclass(frozen=True, slots=True)
class ActionRequest:
    """What the engine asks the OS to do.  Policies produce these."""

    intent: Intent
    kind: str  # move | click | down | up | scroll | key | window | mode | none
    payload: Mapping[str, Any] = field(default_factory=dict)
    risk: float = 0.0
    reversibility: float = 1.0
    timestamp: float = 0.0

    @property
    def is_noop(self) -> bool:
        return self.kind == "none"


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class FrameReport:
    """Everything the telemetry/overlay layer needs for one frame."""

    frame_id: int = 0
    timestamp: float = 0.0
    fps: float = 0.0
    stage_ms: dict[str, float] = field(default_factory=dict)
    state: str = "IDLE"
    gesture: GestureResult = GESTURE_NONE
    intents: IntentField = field(default_factory=IntentField)
    motion: MotionState | None = None
    features: HandFeatures | None = None
    target_beliefs: list[TargetBelief] = field(default_factory=list)
    actions: list[ActionRequest] = field(default_factory=list)
    cursor: np.ndarray | None = None
    events: list[Event] = field(default_factory=list)
    commit: CommitSignal | None = None
    latency_ms: float = 0.0
    notes: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> "FrameReport":
        return replace(self)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(value, lo, hi))


def softmax(logits: Mapping[Any, float], temperature: float = 1.0) -> dict[Any, float]:
    if not logits:
        return {}
    temp = max(temperature, 1e-6)
    keys = list(logits.keys())
    vals = np.array([float(logits[k]) for k in keys], dtype=np.float64) / temp
    vals -= vals.max()
    exp = np.exp(vals)
    total = exp.sum()
    if total <= 0 or not np.isfinite(total):
        return {k: 1.0 / len(keys) for k in keys}
    probs = exp / total
    return {k: float(p) for k, p in zip(keys, probs)}
