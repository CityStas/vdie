"""Action Policy.

Sits between the Intent Field and the OS.  Its whole job is to answer *"is this
intent committed enough to be executed?"* — because the intent field is
deliberately allowed to be ambiguous and the cursor controller is deliberately
allowed to be twitchy.

Decision inputs: intent probability, intent stability, commit signal, target
confidence, FSM state, action risk, cooldowns.

Actions are not booleans; each carries ``risk`` and ``reversibility`` so that the
policy can demand more evidence for irreversible things.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EngineConfig
from ..state.state_machine import State, StateMachine
from ..types import (
    ActionRequest,
    CommitSignal,
    GestureResult,
    HandFeatures,
    Intent,
    IntentField,
    MotionState,
    TargetBelief,
)


@dataclass
class PolicyDecision:
    executed: Intent | None
    reason: str
    score: float
    blocked_by: str = ""


@dataclass
class ActionPolicy:
    cfg: EngineConfig
    _cooldowns: dict[str, float] = field(default_factory=dict)
    _click_times: list[float] = field(default_factory=list)
    _scroll_times: list[float] = field(default_factory=list)
    _last_left_click_gesture: float = -1e9
    _last_right_click_gesture: float = -1e9
    _dragging: bool = False
    _fast_pinch_latched: bool = False
    _last_decision: PolicyDecision | None = None
    _log: list[tuple[float, str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._cooldowns.clear()
        self._click_times.clear()
        self._scroll_times.clear()
        self._last_left_click_gesture = -1e9
        self._last_right_click_gesture = -1e9
        self._dragging = False
        self._fast_pinch_latched = False
        self._last_decision = None
        self._log.clear()

    def decision_log(self) -> list[tuple[float, str, str]]:
        return list(self._log)

    # ------------------------------------------------------------------ #
    def _limits(self, intent: Intent) -> dict[str, float]:
        pcfg = self.cfg.policy
        base = {
            "commit": pcfg.default_commit_threshold,
            "stability": pcfg.default_stability,
            "cooldown": pcfg.default_cooldown,
            "risk": 0.5,
        }
        base.update(pcfg.per_intent.get(intent.value, {}))
        return base

    def _ready(self, key: str, timestamp: float, cooldown: float) -> bool:
        last = self._cooldowns.get(key, -1e9)
        if timestamp - last < cooldown:
            return False
        self._cooldowns[key] = timestamp
        return True

    def _rate_ok(self, times: list[float], timestamp: float, max_hz: float) -> bool:
        cutoff = timestamp - 1.0
        times[:] = [t for t in times if t >= cutoff]
        return len(times) < max(1, int(max_hz))

    # ------------------------------------------------------------------ #
    def update(
        self,
        *,
        intents: IntentField,
        commit: CommitSignal | None,
        gesture: GestureResult | None,
        motion: MotionState,
        features: HandFeatures | None,
        beliefs: list[TargetBelief],
        fsm: StateMachine,
        timestamp: float,
        bimanual_active: bool = False,
    ) -> list[ActionRequest]:
        actions: list[ActionRequest] = []

        # Drag release must always be honoured, even outside the cursor states.
        if self._dragging and not fsm.dragging:
            actions.append(ActionRequest(intent=Intent.DRAG, kind="up", payload={"button": "left"}, risk=0.0, timestamp=timestamp))
            self._dragging = False
            self._log.append((timestamp, "DRAG", "release"))
            return actions

        if not fsm.accepts_actions:
            self._last_decision = PolicyDecision(None, f"state {fsm.state.value} blocks actions", 0.0)
            self._fast_pinch_latched = False
            return actions

        # ---------------- FAST ONE-HAND PINCH CLICK -------------------- #
        # A click is a contact edge, not a dwell. Use the geometric pinch
        # threshold directly when a second hand is not acting as the modifier.
        # This keeps the existing Intent Field as the research/diagnostic layer
        # while giving the live desktop a low-latency path.
        if (
            self.cfg.state.fast_pinch_click
            and not bimanual_active
            and features is not None
            and fsm.state in (State.CURSOR, State.PINCH_DOWN)
        ):
            down = float(self.cfg.gestures.pinch_down)
            release = float(self.cfg.gestures.pinch_release)
            if features.pinch_distance <= down and not self._fast_pinch_latched:
                if self._rate_ok(self._click_times, timestamp, self.cfg.safety.max_click_rate_hz):
                    self._fast_pinch_latched = True
                    self._click_times.append(timestamp)
                    action = ActionRequest(
                        intent=Intent.SELECT,
                        kind="click",
                        payload={"button": "left", "source": "fast_pinch"},
                        risk=0.20,
                        timestamp=timestamp,
                    )
                    self._last_decision = PolicyDecision(Intent.SELECT, "fast pinch edge", 1.0)
                    self._log.append((timestamp, "SELECT", "fast_pinch"))
                    return [action]
            elif features.pinch_distance >= release:
                self._fast_pinch_latched = False

        # ---------------- DRAG (button held) --------------------------- #
        if fsm.dragging:
            if not self._dragging:
                actions.append(ActionRequest(intent=Intent.DRAG, kind="down", payload={"button": "left"}, risk=0.3, timestamp=timestamp))
                self._dragging = True
                self._log.append((timestamp, "DRAG", "press"))
            return actions

        # ---------------- discrete actions ----------------------------- #
        candidate = intents.committed or intents.dominant
        prob = intents.get(candidate)
        limits = self._limits(candidate)
        stability = float(gesture.stability) if gesture is not None else 0.0

        if candidate is Intent.MOVE_CURSOR:
            self._last_decision = PolicyDecision(None, "cursor motion only", prob)
            return actions

        # Live desktop safety: target/trajectory inference must never turn
        # ordinary pointing into a physical click.  The explicit pinch may be
        # one or a few frames before the commit because the FSM moves through
        # PINCH_DOWN/COOLDOWN; remember only a very short evidence window.
        observed = gesture.gesture if gesture is not None else "NONE"
        if observed == "PINCH":
            self._last_left_click_gesture = timestamp
        elif observed == "MIDDLE_PINCH":
            self._last_right_click_gesture = timestamp

        recent_left = (timestamp - self._last_left_click_gesture) <= 0.45
        recent_right = (timestamp - self._last_right_click_gesture) <= 0.45
        # A normal left click is committed while the FSM is in PINCH_DOWN or
        # its immediate COOLDOWN after release.  This is stronger evidence than
        # a target-entry/trajectory commit and keeps the live desktop safe even
        # when the recognizer's PINCH label arrives a frame late.
        left_state_evidence = fsm.state in (State.PINCH_DOWN, State.COOLDOWN)
        click_evidence_ok = {
            Intent.SELECT: recent_left or left_state_evidence,
            Intent.RIGHT_CLICK: recent_right,
            Intent.DOUBLE_CLICK: recent_left or left_state_evidence,
        }
        if candidate in click_evidence_ok and not click_evidence_ok[candidate]:
            self._last_decision = PolicyDecision(
                candidate,
                "explicit recent click gesture required",
                prob,
                blocked_by="gesture",
            )
            return actions

        # Evidence gate.
        if prob < limits["commit"]:
            self._last_decision = PolicyDecision(candidate, "below commit threshold", prob, blocked_by="confidence")
            return actions

        # Commit gate: irreversible actions need an explicit commit signal.
        irreversible = candidate in (Intent.SELECT, Intent.DOUBLE_CLICK, Intent.RIGHT_CLICK, Intent.CUSTOM)
        if irreversible and self.cfg.commit.enabled:
            if commit is None or not commit.committed:
                self._last_decision = PolicyDecision(candidate, "no commit signal", prob, blocked_by="commit")
                return actions
            # More risk -> more commit evidence required.
            required = limits["commit"] * (1.0 - 0.25 * limits["risk"])
            if prob < required:
                self._last_decision = PolicyDecision(candidate, "commit but weak intent", prob, blocked_by="confidence")
                return actions

        if candidate in (Intent.SELECT, Intent.DOUBLE_CLICK, Intent.RIGHT_CLICK) and stability and stability < limits["stability"]:
            self._last_decision = PolicyDecision(candidate, "unstable gesture", prob, blocked_by="stability")
            return actions

        if not self._ready(candidate.value, timestamp, limits["cooldown"]):
            self._last_decision = PolicyDecision(candidate, "cooldown", prob, blocked_by="cooldown")
            return actions

        action = self._build(candidate, intents, motion, features, beliefs, gesture, timestamp)
        if action is None:
            self._last_decision = PolicyDecision(candidate, "no action mapping", prob)
            return actions

        if action.kind in ("click", "double_click", "right_click"):
            if not self._rate_ok(self._click_times, timestamp, self.cfg.safety.max_click_rate_hz):
                self._last_decision = PolicyDecision(candidate, "click rate limit", prob, blocked_by="rate")
                return actions
            self._click_times.append(timestamp)

        actions.append(action)
        self._last_decision = PolicyDecision(candidate, "executed", prob)
        self._log.append((timestamp, candidate.value, action.kind))
        return actions

    # ------------------------------------------------------------------ #
    def _build(
        self,
        intent: Intent,
        intents: IntentField,
        motion: MotionState,
        features: HandFeatures | None,
        beliefs: list[TargetBelief],
        gesture: GestureResult | None,
        timestamp: float,
    ) -> ActionRequest | None:
        target = max(beliefs, key=lambda b: b.confidence).target.id if beliefs else None

        if intent is Intent.SELECT:
            return ActionRequest(intent=intent, kind="click", payload={"button": "left", "target": target}, risk=0.35, timestamp=timestamp)
        if intent is Intent.DOUBLE_CLICK:
            return ActionRequest(intent=intent, kind="double_click", payload={"button": "left", "target": target}, risk=0.45, timestamp=timestamp)
        if intent is Intent.RIGHT_CLICK:
            return ActionRequest(intent=intent, kind="right_click", payload={"button": "right", "target": target}, risk=0.55, timestamp=timestamp)
        if intent is Intent.SCROLL:
            # Continuous channel: the scroll amount is a function of the hand's
            # vertical velocity, not a fixed notch.  This is the "the hand is an
            # analog device" idea applied to a real interaction.
            dy = float(motion.velocity[1])
            if abs(dy) < self.cfg.motion.pause_speed:
                return None
            amount = float(np.clip(dy * 900.0 * self.cfg.control.scroll_multiplier, -1200.0, 1200.0))
            if not self._rate_ok(self._scroll_times, timestamp, self.cfg.safety.max_scroll_rate_hz):
                return None
            self._scroll_times.append(timestamp)
            return ActionRequest(intent=intent, kind="scroll", payload={"dy": amount}, risk=0.1, timestamp=timestamp)
        if intent is Intent.WINDOW_SWITCH:
            direction = -1 if (gesture and gesture.gesture == "SWIPE_LEFT") else 1
            return ActionRequest(intent=intent, kind="key", payload={"combo": "alt+tab", "direction": direction}, risk=0.4, timestamp=timestamp)
        if intent is Intent.WINDOW_CONTROL:
            combo = "win+up" if (gesture and gesture.gesture == "SWIPE_UP") else "win+down"
            return ActionRequest(intent=intent, kind="window", payload={"combo": combo}, risk=0.45, timestamp=timestamp)
        if intent is Intent.CUSTOM:
            name = gesture.gesture if gesture else "CUSTOM"
            return ActionRequest(intent=intent, kind="custom", payload={"name": name}, risk=0.6, timestamp=timestamp)
        return None

    # ------------------------------------------------------------------ #
    @property
    def last_decision(self) -> PolicyDecision | None:
        return self._last_decision

    @property
    def dragging(self) -> bool:
        return self._dragging
