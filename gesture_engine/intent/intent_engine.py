"""V1 baseline: discrete intent engine.

Kept deliberately dumb and deterministic.  It exists to be the control condition
in every A/B experiment ("V1 gesture engine vs V2 intent field").  Do not add
heuristics here — that is what :mod:`gesture_engine.intent.intent_field` is for.
"""

from __future__ import annotations

from ..config import EngineConfig
from ..state.state_machine import State, StateMachine
from ..types import GestureResult, HandFeatures, Intent, IntentField, MotionState
from .evidence import GESTURE_INTENT_MAP

#: Direct gesture -> intent mapping, with the FSM state as the only context.
V1_PRIORITY: tuple[tuple[str, Intent], ...] = (
    ("FIST", Intent.CANCEL),
    ("OPEN_PALM", Intent.PAUSE),
    ("MIDDLE_PINCH", Intent.RIGHT_CLICK),
    ("PINCH", Intent.SELECT),
    ("V_SIGN", Intent.WINDOW_SWITCH),
    ("SWIPE_LEFT", Intent.WINDOW_SWITCH),
    ("SWIPE_RIGHT", Intent.WINDOW_SWITCH),
    ("SWIPE_UP", Intent.WINDOW_CONTROL),
    ("SWIPE_DOWN", Intent.WINDOW_CONTROL),
    ("SCROLL_UP", Intent.SCROLL),
    ("SCROLL_DOWN", Intent.SCROLL),
    ("THREE", Intent.CUSTOM),
    ("ROCK", Intent.CUSTOM),
    ("INDEX_UP", Intent.MOVE_CURSOR),
    ("POINT", Intent.MOVE_CURSOR),
)


class V1IntentEngine:
    """Maps a confirmed gesture to exactly one intent."""

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._last = Intent.UNKNOWN

    def reset(self) -> None:
        self._last = Intent.UNKNOWN

    def update(
        self,
        gesture: GestureResult | None,
        fsm: StateMachine,
        motion: MotionState | None = None,
        features: HandFeatures | None = None,
        timestamp: float = 0.0,
    ) -> IntentField:
        intent = Intent.UNKNOWN
        if fsm.state is State.IDLE:
            intent = Intent.PAUSE
        elif fsm.state is State.DRAGGING:
            intent = Intent.DRAG
        elif gesture is not None and not gesture.is_none:
            for name, mapped in V1_PRIORITY:
                if gesture.gesture == name:
                    intent = mapped
                    break
            if gesture.gesture == "PINCH" and motion is not None and motion.speed > self.cfg.motion.pause_speed:
                intent = Intent.DRAG
        elif fsm.cursor_enabled:
            intent = Intent.MOVE_CURSOR

        confidence = float(gesture.confidence) if (gesture and not gesture.is_none) else (0.6 if fsm.cursor_enabled else 0.0)
        probs = {intent: confidence} if confidence > 0 else {Intent.UNKNOWN: 1.0}
        self._last = intent
        return IntentField(
            probabilities=probs,
            logits={intent: confidence},
            entropy=0.0,
            timestamp=timestamp,
            dominant=intent,
            dominant_probability=confidence,
            committed=intent if confidence > 0 else None,
        )
