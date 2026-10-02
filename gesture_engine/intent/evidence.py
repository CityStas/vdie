"""Evidence providers for the Intent Field.

Each provider looks at one aspect of the world and returns **logits** (not
probabilities) per intent.  Working in logit space matters: summing logits is
Bayesian evidence accumulation, whereas averaging probabilities makes the field
collapse towards whatever subset of providers happens to be active this frame.

A provider may return negative logits to *suppress* an intent.  That is how, for
example, a fist suppresses everything except CANCEL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..config import EngineConfig
from ..features.geometry import clamp01, smoothstep
from ..motion.history import HistoryBuffer
from ..state.state_machine import State, StateMachine
from ..types import (
    DYNAMIC_GESTURES,
    CommitSignal,
    Event,
    EventType,
    GestureResult,
    HandFeatures,
    Intent,
    IntentField,
    MotionState,
    TargetBelief,
)


class EvidenceProvider(Protocol):
    name: str
    #: "rate" = logits per second (multiplied by dt);
    #: "impulse" = a one-off logit injection for an instantaneous event.
    kind: str

    def logits(
        self,
        *,
        features: HandFeatures | None,
        motion: MotionState,
        gesture: GestureResult | None,
        history: HistoryBuffer,
        events: list[Event],
        beliefs: list[TargetBelief],
        fsm: StateMachine,
        commit: CommitSignal | None,
        timestamp: float,
    ) -> dict[Intent, float]: ...


def _l(intent: Intent, value: float, out: dict[Intent, float]) -> None:
    out[intent] = out.get(intent, 0.0) + float(value)


# --------------------------------------------------------------------------- #
# Gesture evidence
# --------------------------------------------------------------------------- #

GESTURE_INTENT_MAP: dict[str, tuple[Intent, float]] = {
    "INDEX_UP": (Intent.MOVE_CURSOR, 1.6),
    "POINT": (Intent.MOVE_CURSOR, 1.4),
    "OPEN_PALM": (Intent.PAUSE, 2.0),
    "FIST": (Intent.CANCEL, 2.6),
    "PINCH": (Intent.SELECT, 1.8),
    "MIDDLE_PINCH": (Intent.RIGHT_CLICK, 2.0),
    "V_SIGN": (Intent.WINDOW_SWITCH, 1.8),
    "THREE": (Intent.CUSTOM, 1.5),
    "ROCK": (Intent.CUSTOM, 1.4),
    "SWIPE_LEFT": (Intent.WINDOW_SWITCH, 2.0),
    "SWIPE_RIGHT": (Intent.WINDOW_SWITCH, 2.0),
    "SWIPE_UP": (Intent.WINDOW_CONTROL, 2.0),
    "SWIPE_DOWN": (Intent.WINDOW_CONTROL, 2.0),
    "SCROLL_UP": (Intent.SCROLL, 1.8),
    "SCROLL_DOWN": (Intent.SCROLL, 1.8),
}


@dataclass
class GestureEvidence:
    name: str = "gesture"
    kind: str = "rate"
    cfg: EngineConfig | None = None

    def logits(self, *, gesture: GestureResult | None, motion: MotionState | None = None, **_: object) -> dict[Intent, float]:
        out: dict[Intent, float] = {}
        if gesture is None or gesture.is_none:
            return out
        entry = GESTURE_INTENT_MAP.get(gesture.gesture)
        if entry is None:
            return out
        intent, weight = entry
        # Confidence and stability both scale the evidence strength.
        strength = weight * clamp01(gesture.confidence) * (0.5 + 0.5 * clamp01(gesture.stability))
        # A *steering* pose held perfectly still is weak evidence for steering:
        # the user is parked, not navigating.  Without this modulation the
        # continuously-present INDEX_UP pose pins the field at p(MOVE)=1 and no
        # discrete action can ever out-vote it.
        if intent is Intent.MOVE_CURSOR and motion is not None:
            strength *= 0.30 + 0.70 * smoothstep(0.05, 1.10, motion.speed)
        _l(intent, strength, out)
        return out


# --------------------------------------------------------------------------- #
# Motion evidence
# --------------------------------------------------------------------------- #


@dataclass
class MotionEvidence:
    kind: str = "rate"
    name: str = "motion"

    def logits(
        self,
        *,
        motion: MotionState,
        features: HandFeatures | None,
        **_: object,
    ) -> dict[Intent, float]:
        out: dict[Intent, float] = {}
        speed = motion.speed
        fast = smoothstep(0.35, 1.5, speed)
        slow = 1.0 - fast

        # Sustained movement is the baseline evidence for "the user is steering".
        _l(Intent.MOVE_CURSOR, 0.9 * fast, out)

        # Deceleration + pause while something is under the cursor: selection.
        decel = clamp01(-motion.speed_trend / 2.0)
        pause = smoothstep(0.03, 0.25, motion.pause_duration)
        _l(Intent.SELECT, 0.85 * decel * pause, out)

        # Fast, straight, long travel is navigation, not selection.
        straight = smoothstep(0.85, 0.98, motion.path_efficiency)
        _l(Intent.MOVE_CURSOR, 0.5 * fast * straight, out)
        _l(Intent.SELECT, -0.6 * fast * straight, out)

        # Pinch closed + movement = drag, not click.
        if features is not None:
            closed = smoothstep(0.45, 0.85, features.pinch_strength)
            moving = smoothstep(0.20, 0.9, speed)
            _l(Intent.DRAG, 1.1 * closed * moving, out)
            _l(Intent.SELECT, 0.8 * closed * (1.0 - moving), out)
            _l(Intent.MOVE_CURSOR, -0.5 * closed, out)

        # Vertical oscillation = scrolling.
        osc = smoothstep(0.45, 1.10, abs(motion.direction_change))
        _l(Intent.SCROLL, 0.4 * osc * slow, out)
        return out


# --------------------------------------------------------------------------- #
# Temporal evidence (event sequences)
# --------------------------------------------------------------------------- #


@dataclass
class TemporalEvidence:
    """Pattern matching over the recent event stream.

    This is the seed of the interaction grammar: a gesture is a *sequence* of
    events, and different sequences over the same pose mean different things.
    """

    name: str = "temporal"
    kind: str = "rate"
    window: float = 0.9
    _recent: list[Event] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self._recent is None:
            self._recent = []

    def observe(self, events: list[Event], timestamp: float) -> None:
        self._recent.extend(events)
        cutoff = timestamp - self.window
        self._recent = [e for e in self._recent if e.timestamp >= cutoff]

    def logits(self, *, timestamp: float, **_: object) -> dict[Intent, float]:
        out: dict[Intent, float] = {}
        types = [e.type for e in self._recent]
        if not types:
            return out

        def contains(*seq: EventType) -> bool:
            """True if ``seq`` occurs as an ordered subsequence of ``types``."""
            i = 0
            for t in types:
                if t is seq[i]:
                    i += 1
                    if i == len(seq):
                        return True
            return False

        if contains(EventType.POINT_START, EventType.DECELERATE, EventType.PAUSE):
            _l(Intent.SELECT, 1.4, out)
        if contains(EventType.POINT_START, EventType.PINCH_START) and EventType.POINT_STOP in types:
            _l(Intent.DRAG, 1.2, out)
        if contains(EventType.PINCH_START, EventType.PINCH_RELEASE):
            n = sum(1 for t in types if t is EventType.PINCH_START)
            if n == 1:
                _l(Intent.SELECT, 0.9, out)
            elif n >= 2:
                _l(Intent.DOUBLE_CLICK, 1.6, out)
        if contains(EventType.DIRECTION_CHANGE, EventType.DIRECTION_CHANGE):
            _l(Intent.SCROLL, 0.7, out)
        if contains(EventType.HAND_LOST):
            _l(Intent.PAUSE, 1.0, out)
        return out


# --------------------------------------------------------------------------- #
# Target evidence
# --------------------------------------------------------------------------- #


@dataclass
class TargetEvidence:
    kind: str = "rate"
    name: str = "target"

    def logits(
        self,
        *,
        beliefs: list[TargetBelief],
        motion: MotionState,
        **_: object,
    ) -> dict[Intent, float]:
        out: dict[Intent, float] = {}
        if not beliefs:
            return out
        best = max(beliefs, key=lambda b: b.confidence)
        # Approaching a good target raises SELECT; travelling past lowers it.
        approach = best.target.approach_confidence
        quality = best.target.quality()
        _l(Intent.SELECT, 1.5 * best.confidence * quality * (0.4 + 0.6 * approach), out)
        if best.target.state.value == "AVOID":
            _l(Intent.SELECT, -1.0, out)
        return out


# --------------------------------------------------------------------------- #
# Context evidence (FSM)
# --------------------------------------------------------------------------- #


@dataclass
class ContextEvidence:
    kind: str = "rate"
    name: str = "context"

    def logits(self, *, fsm: StateMachine, **_: object) -> dict[Intent, float]:
        out: dict[Intent, float] = {}
        if fsm.state is State.IDLE:
            _l(Intent.PAUSE, 1.5, out)
            for i in (Intent.SELECT, Intent.DRAG, Intent.SCROLL):
                _l(i, -2.0, out)
        elif fsm.state is State.ACTIVATING:
            _l(Intent.MOVE_CURSOR, 1.2, out)
        elif fsm.state is State.PAUSED:
            _l(Intent.PAUSE, 2.2, out)
        elif fsm.state is State.PINCH_DOWN:
            _l(Intent.SELECT, 0.8, out)
        elif fsm.state is State.DRAGGING:
            _l(Intent.DRAG, 2.0, out)
            _l(Intent.SELECT, -1.0, out)
        elif fsm.state is State.SCROLL:
            _l(Intent.SCROLL, 2.0, out)
        elif fsm.state is State.COOLDOWN:
            for i in (Intent.SELECT, Intent.DRAG):
                _l(i, -1.5, out)
        return out


# --------------------------------------------------------------------------- #
# Commit evidence
# --------------------------------------------------------------------------- #


@dataclass
class CommitEvidence:
    """A commit is decisive evidence, including *against* continuing to steer.

    A commit means the movement has terminated deliberately.  Treating it as
    positive evidence for SELECT alone is not enough: MOVE_CURSOR has been
    accumulating for the whole approach and will still dominate the softmax.
    Suppressing the navigation hypotheses at the commit instant is what makes
    click-without-pinch actually work.

    It does **not** apply while a confident dynamic gesture is active.  A swipe
    ends with exactly the same kinematics as a deliberate stop (accelerate ->
    travel -> decelerate -> stop), so the end of every swipe fired a MICRO_PAUSE
    commit at score 1.0 and produced a click instead of a window switch —
    measured on the swipe scenario: 2 swipes, 2 false clicks.  A dynamic gesture
    is already a complete command; there is nothing left for a commit point to
    confirm.  The commit is still *detected* (it is a fact about the trajectory
    and the benchmark reports it); it is simply not read as a selection.
    """

    name: str = "commit"
    #: Impulse evidence: a commit is an instantaneous event, not a rate.
    kind: str = "impulse"
    cfg: EngineConfig | None = None

    def logits(
        self,
        *,
        commit: CommitSignal | None,
        gesture: GestureResult | None = None,
        **_: object,
    ) -> dict[Intent, float]:
        out: dict[Intent, float] = {}
        if commit is None or not commit.committed:
            return out
        if gesture is not None and gesture.gesture in DYNAMIC_GESTURES:
            threshold = self.cfg.gestures.min_confidence if self.cfg is not None else 0.55
            if gesture.confidence >= threshold:
                return out
        score = clamp01(commit.score)
        if commit.kind in ("MICRO_PAUSE", "TARGET_ENTRY", "VELOCITY_REVERSAL"):
            _l(Intent.SELECT, 3.2 * score, out)
        elif commit.kind == "THRUST":
            _l(Intent.SELECT, 2.6 * score, out)
        elif commit.kind == "PINCH":
            _l(Intent.SELECT, 2.2 * score, out)
        else:
            return out
        # The movement stopped on purpose: steering is no longer the hypothesis.
        # A commit is a rare, deliberate event, so it is allowed to dominate the
        # field for the ~0.3 s its impulse survives.  Suppression of the
        # navigation hypotheses is what makes click-without-pinch possible.
        _l(Intent.MOVE_CURSOR, -2.6 * score, out)
        _l(Intent.SCROLL, -1.8 * score, out)
        _l(Intent.DRAG, -1.2 * score, out)
        return out


# --------------------------------------------------------------------------- #
# Counterfactual future evidence
# --------------------------------------------------------------------------- #


#: Which intents each extrapolated future supports.
FUTURE_INTENT_AFFINITY: dict[str, dict[str, float]] = {
    "straight": {"_prior": 1.0, "MOVE_CURSOR": 1.0, "DRAG": 0.4},
    "decelerating": {"_prior": 0.8, "SELECT": 1.0, "SCROLL": 0.2},
    "curved_left": {"_prior": 0.4, "MOVE_CURSOR": 0.6, "SCROLL": 0.3},
    "curved_right": {"_prior": 0.4, "MOVE_CURSOR": 0.6, "SCROLL": 0.3},
    "reversing": {"_prior": 0.5, "CANCEL": 0.8, "PAUSE": 0.5, "SELECT": -0.3},
}


@dataclass
class FutureEvidence:
    """Turns counterfactual trajectory extrapolations into intent support."""

    name: str = "future"
    kind: str = "rate"

    def logits(self, *, futures: list | None = None, **_: object) -> dict[Intent, float]:
        out: dict[Intent, float] = {}
        if not futures:
            return out
        from ..motion.trajectory import aggregate_intent_support, score_futures

        scored = score_futures(list(futures), FUTURE_INTENT_AFFINITY)
        support = aggregate_intent_support(scored)
        for name, value in support.items():
            try:
                intent = Intent(name)
            except ValueError:
                continue
            _l(intent, 1.2 * (value - 0.25), out)
        return out


# --------------------------------------------------------------------------- #
# User-model evidence
# --------------------------------------------------------------------------- #


@dataclass
class UserEvidence:
    """Adapts intent priors to the individual user's motion style."""

    name: str = "user"
    kind: str = "rate"
    signature: object | None = None

    def logits(self, *, motion: MotionState, **_: object) -> dict[Intent, float]:
        out: dict[Intent, float] = {}
        sig = self.signature
        if sig is None:
            return out
        # A user who consistently commits with a micro-pause gets stronger
        # selection evidence from pausing; a "thrusty" user gets it from thrusts.
        pause_bias = float(getattr(sig, "pause_bias", 0.0))
        thrust_bias = float(getattr(sig, "thrust_bias", 0.0))
        pause = smoothstep(0.02, 0.2, motion.pause_duration)
        thrust = smoothstep(0.8, 1.6, motion.speed)
        _l(Intent.SELECT, 0.5 * pause_bias * pause + 0.4 * thrust_bias * thrust, out)
        return out
