"""Deterministic state machine (V1 core).

The state machine answers one question: *is the engine allowed to touch the
cursor right now, and in which mode?*  It is intentionally boring and explicit —
every transition is a named guard, so a recorded session can be replayed and the
transition log diffed between algorithm versions.

States
------
``IDLE``               observation only, no OS input at all
``ACTIVATING``         activation pose detected, waiting for the dwell time
``ARMED``              activation confirmed, cursor control switching on
``CURSOR``             cursor follows the hand
``PINCH_DOWN``         pinch closed, deciding between click and drag
``DRAGGING``           mouse button held down
``SCROLL``             scroll mode
``PAUSED``             user asked for a temporary stop
``COOLDOWN``           short refractory period after an action
``EMERGENCY_CANCEL``   fist: release everything, drop to IDLE
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..config import EngineConfig
from ..types import Event, EventType, GestureResult, HandFeatures, MotionState


class State(str, Enum):
    IDLE = "IDLE"
    ACTIVATING = "ACTIVATING"
    ARMED = "ARMED"
    CURSOR = "CURSOR"
    PINCH_DOWN = "PINCH_DOWN"
    DRAGGING = "DRAGGING"
    SCROLL = "SCROLL"
    PAUSED = "PAUSED"
    COOLDOWN = "COOLDOWN"
    EMERGENCY_CANCEL = "EMERGENCY_CANCEL"


@dataclass
class Transition:
    from_state: State
    to_state: State
    timestamp: float
    reason: str = ""


@dataclass
class FSMContext:
    timestamp: float
    features: HandFeatures | None = None
    motion: MotionState | None = None
    gesture: GestureResult | None = None
    hand_confidence: float = 1.0


@dataclass
class StateMachine:
    cfg: EngineConfig
    state: State = State.IDLE
    previous: State = State.IDLE
    since: float = 0.0
    transitions: list[Transition] = field(default_factory=list)
    _activation_dwell: float = 0.0
    _last_pinch_time: float = -1e9
    _drag_started: float = 0.0
    _cooldown_until: float = 0.0
    _pinch_anchor: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    @property
    def cursor_enabled(self) -> bool:
        return self.state in (State.CURSOR, State.PINCH_DOWN, State.DRAGGING, State.SCROLL)

    @property
    def dragging(self) -> bool:
        return self.state is State.DRAGGING

    @property
    def accepts_actions(self) -> bool:
        return self.state not in (State.IDLE, State.ACTIVATING, State.PAUSED, State.EMERGENCY_CANCEL)

    def dwell(self, timestamp: float) -> float:
        return timestamp - self.since

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.state = State.IDLE
        self.previous = State.IDLE
        self.since = 0.0
        self.transitions.clear()
        self._activation_dwell = 0.0
        self._last_pinch_time = -1e9
        self._drag_started = 0.0
        self._cooldown_until = 0.0
        self._pinch_anchor = None

    def force(self, state: State, timestamp: float, reason: str = "force") -> None:
        self._transition(state, timestamp, reason)

    def _transition(self, to: State, timestamp: float, reason: str) -> None:
        if to is self.state:
            return
        self.transitions.append(Transition(self.state, to, timestamp, reason))
        self.previous = self.state
        self.state = to
        self.since = timestamp
        self._activation_dwell = 0.0

    # ------------------------------------------------------------------ #
    def update(self, ctx: FSMContext) -> State:
        handler = {
            State.IDLE: self._idle,
            State.ACTIVATING: self._activating,
            State.ARMED: self._armed,
            State.CURSOR: self._cursor,
            State.PINCH_DOWN: self._pinch_down,
            State.DRAGGING: self._dragging,
            State.SCROLL: self._scroll,
            State.PAUSED: self._paused,
            State.COOLDOWN: self._cooldown,
            State.EMERGENCY_CANCEL: self._emergency,
        }[self.state]
        handler(ctx)
        return self.state

    # ------------------------------------------------------------------ #
    # Guards
    # ------------------------------------------------------------------ #
    def _activation_ready(self, ctx: FSMContext) -> bool:
        f = ctx.features
        if f is None or ctx.motion is None:
            return False
        scfg = self.cfg.state

        # The old gate required the raw finger-state vector to be perfect on
        # every frame.  That is too brittle on a real camera: MediaPipe can
        # briefly classify a folded finger as extended while the recognizer
        # already has stable INDEX_UP evidence.  Activation remains explicit
        # and temporal, but accepts either the geometric pose OR a confirmed
        # INDEX_UP recognition.
        index_only = (
            f.fingers[1] == 1
            and f.fingers[2] == 0
            and f.fingers[3] == 0
            and f.fingers[4] == 0
        )
        vertical = abs(f.index_orientation) <= scfg.activation_max_tilt_deg
        still = ctx.motion.speed < scfg.activation_velocity
        confident = ctx.hand_confidence >= scfg.activation_confidence
        geometric = index_only and vertical and still and confident

        recognized = (
            ctx.gesture is not None
            and ctx.gesture.gesture == "INDEX_UP"
            and ctx.gesture.confidence >= self.cfg.gestures.min_confidence
            and ctx.gesture.stability >= 0.45
            and still
            and ctx.hand_confidence >= min(scfg.activation_confidence, 0.70)
        )
        return bool(geometric or recognized)

    def _is_pinching(self, ctx: FSMContext) -> bool:
        f = ctx.features
        if f is None:
            return False
        return f.pinch_distance < self.cfg.gestures.pinch_down

    def _pinch_released(self, ctx: FSMContext) -> bool:
        f = ctx.features
        if f is None:
            return True
        return f.pinch_distance > self.cfg.gestures.pinch_release

    def _is_palm(self, ctx: FSMContext) -> bool:
        f = ctx.features
        return bool(f is not None and f.extended_count >= 4 and f.hand_span > 1.2)

    def _is_fist(self, ctx: FSMContext) -> bool:
        f = ctx.features
        return bool(f is not None and f.extended_count == 0 and f.curl > 0.5)

    def _is_v(self, ctx: FSMContext) -> bool:
        f = ctx.features
        return bool(f is not None and f.fingers[1] == 1 and f.fingers[2] == 1 and f.fingers[3] == 0 and f.fingers[4] == 0)

    # ------------------------------------------------------------------ #
    # State handlers
    # ------------------------------------------------------------------ #
    def _idle(self, ctx: FSMContext) -> None:
        if self._is_fist(ctx):
            return
        if self._activation_ready(ctx):
            self._transition(State.ACTIVATING, ctx.timestamp, "activation pose")

    def _activating(self, ctx: FSMContext) -> None:
        if not self._activation_ready(ctx):
            self._activation_dwell = 0.0
            self._transition(State.IDLE, ctx.timestamp, "activation pose lost")
            return
        self._activation_dwell = ctx.timestamp - self.since
        if self._activation_dwell >= self.cfg.state.activation_stable_ms / 1000.0:
            self._transition(State.ARMED, ctx.timestamp, "activation dwell reached")

    def _armed(self, ctx: FSMContext) -> None:
        # ARMED is a one-frame hand-off: cursor control comes on immediately.
        self._transition(State.CURSOR, ctx.timestamp, "armed -> cursor")

    def _cursor(self, ctx: FSMContext) -> None:
        if self._is_fist(ctx):
            self._transition(State.EMERGENCY_CANCEL, ctx.timestamp, "emergency gesture")
            return
        if self._is_palm(ctx):
            self._transition(State.PAUSED, ctx.timestamp, "open palm")
            return
        if self._is_pinching(ctx):
            self._last_pinch_time = ctx.timestamp
            self._pinch_anchor = ctx.motion.position.copy() if ctx.motion is not None else None
            self._transition(State.PINCH_DOWN, ctx.timestamp, "pinch closed")
            return
        if ctx.hand_confidence < self.cfg.safety.freeze_below_confidence:
            self._transition(State.COOLDOWN, ctx.timestamp, "low tracking confidence")
            return
        if ctx.gesture is not None and ctx.gesture.gesture in ("SCROLL_UP", "SCROLL_DOWN"):
            self._transition(State.SCROLL, ctx.timestamp, "scroll gesture")
            return
        idle_for = ctx.timestamp - self.since
        if idle_for * 1000.0 > self.cfg.state.cursor_idle_timeout_ms:
            self._transition(State.IDLE, ctx.timestamp, "cursor timeout")

    def _pinch_down(self, ctx: FSMContext) -> None:
        if self._pinch_released(ctx):
            self._transition(State.COOLDOWN, ctx.timestamp, "pinch released -> click")
            self._cooldown_until = ctx.timestamp + self.cfg.state.cooldown_ms / 1000.0
            return
        if self._drag_started_at(ctx):
            self._drag_started = ctx.timestamp
            self._transition(State.DRAGGING, ctx.timestamp, "pinch + displacement -> drag")

    def _drag_started_at(self, ctx: FSMContext) -> bool:
        """True once the hand has genuinely travelled since the pinch closed.

        Two independent conditions, both required:

        * displacement from the pinch anchor exceeds ``drag_min_displacement``;
        * the pinch has been held at least ``drag_min_duration``.

        Instantaneous speed alone is not usable here — curling the fingers moves
        the control point, which would turn every click into a 3-pixel drag.
        """
        if ctx.motion is None or self._pinch_anchor is None:
            return False
        held = ctx.timestamp - self.since
        travelled = float(np.linalg.norm(ctx.motion.position - self._pinch_anchor))
        # In the touch-surface profile, a single-hand pinch should remain a
        # click unless the user explicitly enables one-hand drag. The second
        # hand is the safer modifier for drag.
        if (
            self.cfg.cursor.controller == "touch_surface"
            and self.cfg.bimanual.enabled
            and not self.cfg.bimanual.one_hand_drag_enabled
        ):
            return False
        return (
            travelled >= self.cfg.state.drag_min_displacement
            and held >= self.cfg.state.drag_min_duration
        )

    def _dragging(self, ctx: FSMContext) -> None:
        if self._pinch_released(ctx):
            self._transition(State.COOLDOWN, ctx.timestamp, "drag released")
            self._cooldown_until = ctx.timestamp + self.cfg.state.cooldown_ms / 1000.0
            return
        if self._is_fist(ctx) or ctx.hand_confidence < self.cfg.safety.stop_below_confidence:
            self._transition(State.EMERGENCY_CANCEL, ctx.timestamp, "drag aborted")
            return
        held_ms = (ctx.timestamp - self._drag_started) * 1000.0
        if held_ms > self.cfg.state.drag_max_hold_ms:
            self._transition(State.COOLDOWN, ctx.timestamp, "drag hold limit")
            self._cooldown_until = ctx.timestamp + self.cfg.state.cooldown_ms / 1000.0

    def _scroll(self, ctx: FSMContext) -> None:
        if self._is_fist(ctx):
            self._transition(State.EMERGENCY_CANCEL, ctx.timestamp, "emergency gesture")
            return
        if ctx.gesture is None or ctx.gesture.gesture not in ("SCROLL_UP", "SCROLL_DOWN"):
            self._transition(State.CURSOR, ctx.timestamp, "scroll ended")

    def _paused(self, ctx: FSMContext) -> None:
        if self._is_fist(ctx):
            self._transition(State.EMERGENCY_CANCEL, ctx.timestamp, "emergency gesture")
            return
        if not self._is_palm(ctx) and (ctx.timestamp - self.since) * 1000.0 > self.cfg.state.pause_release_ms:
            self._transition(State.CURSOR, ctx.timestamp, "pause released")

    def _cooldown(self, ctx: FSMContext) -> None:
        # Do not bounce CURSOR<->COOLDOWN while MediaPipe has no hand. The old
        # behaviour re-enabled the cursor every 250 ms and immediately lost it
        # again, creating thousands of recovery/cooldown frames in long logs.
        if ctx.hand_confidence < self.cfg.safety.freeze_below_confidence or ctx.features is None:
            return
        if ctx.timestamp >= self._cooldown_until:
            self._transition(State.CURSOR, ctx.timestamp, "cooldown finished")

    def _emergency(self, ctx: FSMContext) -> None:
        self._transition(State.IDLE, ctx.timestamp, "emergency reset")

    # ------------------------------------------------------------------ #
    def events(self, ctx: FSMContext) -> list[Event]:
        """Events implied by the transitions that happened this frame."""
        out: list[Event] = []
        for tr in self.transitions:
            if tr.timestamp != ctx.timestamp:
                continue
            out.append(Event(type=EventType.MODE_CHANGE, timestamp=tr.timestamp, source="fsm", data={"from": tr.from_state.value, "to": tr.to_state.value, "reason": tr.reason}))
            if tr.to_state is State.EMERGENCY_CANCEL:
                out.append(Event(type=EventType.CANCEL, timestamp=tr.timestamp, source="fsm"))
            if tr.to_state is State.PINCH_DOWN:
                out.append(Event(type=EventType.PINCH_START, timestamp=tr.timestamp, source="fsm"))
        return out
