"""Event Engine.

Converts the continuous motion state into a sparse stream of temporal events.
This is the layer that lets a gesture be described as an *event sequence* rather
than as a pose:

    POINT_START -> DECELERATE -> PAUSE -> COMMIT      => SELECT
    POINT_START -> PINCH_START -> MOVE -> PINCH_RELEASE => DRAG

Everything here is a Schmitt trigger (distinct on/off thresholds) plus a minimum
inter-event interval, which is what stops the event stream from chattering on
sensor noise.  Events carry timestamps and confidence, so they can be recorded
and replayed exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EngineConfig
from ..types import Event, EventType, GestureResult, HandFeatures, MotionState, Observation, TargetBelief


@dataclass
class _Edge:
    """Schmitt trigger with a minimum interval between rising edges.

    ``invert=True`` flips the polarity: the trigger fires when the value falls
    *below* ``on`` and releases when it rises above ``off``.  Pinch needs this —
    it is a *distance*, so "closed" is the low end of the scale.  Without the
    flag the pinch edge fired on the hand opening.
    """

    name: str
    on: float
    off: float
    min_interval: float = 0.0
    invert: bool = False
    state: bool = False
    last_on: float = -1e9
    last_off: float = -1e9

    def update(self, value: float, timestamp: float) -> int:
        """Returns ``+1`` on the active edge, ``-1`` on the release, else ``0``."""
        if self.invert:
            if not self.state:
                if value <= self.on and (timestamp - self.last_on) >= self.min_interval:
                    self.state = True
                    self.last_on = timestamp
                    return 1
                return 0
            if value >= self.off:
                self.state = False
                self.last_off = timestamp
                return -1
            return 0

        if not self.state:
            if value >= self.on and (timestamp - self.last_on) >= self.min_interval:
                self.state = True
                self.last_on = timestamp
                return 1
            return 0
        if value <= self.off:
            self.state = False
            self.last_off = timestamp
            return -1
        return 0

    def reset(self) -> None:
        self.state = False
        self.last_on = -1e9
        self.last_off = -1e9


class EventEngine:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        ecfg = cfg.events
        mi = ecfg.min_event_interval

        self.moving = _Edge("moving", on=ecfg.pause_speed * 1.6, off=ecfg.pause_speed, min_interval=mi)
        self.accel = _Edge("accel", on=ecfg.accel_threshold, off=ecfg.accel_threshold * 0.4, min_interval=mi)
        self.decel = _Edge("decel", on=ecfg.decel_threshold, off=ecfg.decel_threshold * 0.4, min_interval=mi)
        self.pinching = _Edge(
            "pinch",
            on=cfg.gestures.pinch_down,
            off=cfg.gestures.pinch_release,
            min_interval=mi,
            # pinch_distance is a distance: closed == below pinch_down.
            invert=True,
        )
        self.hand = _Edge("hand", on=cfg.tracking.min_hand_confidence, off=cfg.tracking.min_hand_confidence * 0.6)
        self.target = _Edge("target", on=ecfg.target_approach_radius, off=ecfg.target_approach_radius * 2.2)

        self._last_direction: float | None = None
        self._last_gesture = "NONE"
        self._pending: list[Event] = []
        self._counts: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        for e in (self.moving, self.accel, self.decel, self.pinching, self.hand, self.target):
            e.reset()
        self._last_direction = None
        self._last_gesture = "NONE"
        self._pending.clear()

    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def _emit(self, etype: EventType, ts: float, confidence: float = 1.0, source: str = "motion", **data) -> None:
        self._pending.append(Event(type=etype, timestamp=ts, confidence=confidence, source=source, data=data))
        self._counts[etype.value] = self._counts.get(etype.value, 0) + 1

    # ------------------------------------------------------------------ #
    def update(
        self,
        obs: Observation,
        motion: MotionState,
        features: HandFeatures | None = None,
        gesture: GestureResult | None = None,
        target_beliefs: list[TargetBelief] | None = None,
        pinch_distance: float | None = None,
    ) -> list[Event]:
        """Feed one frame; returns the events produced by this frame."""
        self._pending = []
        ts = obs.timestamp
        conf = motion.confidence

        # -- hand presence -------------------------------------------------- #
        if self.hand.update(conf, ts) == 1:
            self._emit(EventType.HAND_REACQUIRED, ts, conf, source="tracking")
        elif self.hand.state is False and conf < self.cfg.tracking.min_hand_confidence * 0.6:
            if self._counts.get("_hand_lost_flag", 0) == 0:
                self._emit(EventType.HAND_LOST, ts, 1.0 - conf, source="tracking")
                self._counts["_hand_lost_flag"] = 1

        if conf < self.cfg.safety.stop_below_confidence:
            return list(self._pending)
        if self._counts.get("_hand_lost_flag", 0) == 1 and self.hand.state:
            self._counts["_hand_lost_flag"] = 0

        # -- movement ------------------------------------------------------- #
        if self.moving.update(motion.speed, ts) == 1:
            self._emit(EventType.POINT_START, ts, conf, speed=motion.speed)
        elif self.moving.state is False and self.moving.last_off == ts:
            self._emit(EventType.POINT_STOP, ts, conf, pause=motion.pause_duration)

        # -- acceleration profile ------------------------------------------- #
        if self.accel.update(motion.speed_trend, ts) == 1:
            self._emit(EventType.ACCELERATE, ts, conf, trend=motion.speed_trend)
        if self.decel.update(motion.speed_trend, ts) == 1:
            self._emit(EventType.DECELERATE, ts, conf, trend=motion.speed_trend)

        # -- pause / resume -------------------------------------------------- #
        if motion.is_pausing and motion.pause_duration >= self.cfg.events.pause_min_duration:
            if not self.moving.state and self.moving.last_off != ts:
                if not getattr(self, "_pause_emitted", False):
                    self._emit(EventType.PAUSE, ts, conf, duration=motion.pause_duration)
                    self._pause_emitted = True
        else:
            if getattr(self, "_pause_emitted", False):
                self._emit(EventType.RESUME, ts, conf)
                self._pause_emitted = False

        # -- direction change ------------------------------------------------ #
        if motion.speed > self.cfg.motion.pause_speed and self._last_direction is not None:
            delta = abs(float((motion.direction - self._last_direction + np.pi) % (2 * np.pi) - np.pi))
            if delta >= np.radians(self.cfg.events.direction_change_deg):
                self._emit(EventType.DIRECTION_CHANGE, ts, conf, delta_deg=float(np.degrees(delta)))
        if motion.speed > self.cfg.motion.pause_speed:
            self._last_direction = motion.direction

        # -- pinch (continuous signal -> edge) ------------------------------- #
        pd = pinch_distance if pinch_distance is not None else (features.pinch_distance if features else None)
        if pd is not None:
            edge = self.pinching.update(pd, ts)
            if edge == 1:
                self._emit(EventType.PINCH_START, ts, conf, distance=pd)
            elif edge == -1:
                self._emit(EventType.PINCH_RELEASE, ts, conf, distance=pd)

        # -- target approach -------------------------------------------------- #
        if target_beliefs:
            best = min(target_beliefs, key=lambda b: b.distance)
            edge = self.target.update(best.distance, ts)
            if edge == 1:
                self._emit(EventType.TARGET_APPROACH, ts, best.confidence, source="target", target=best.target.id)
            elif edge == -1:
                self._emit(EventType.TARGET_EXIT, ts, best.confidence, source="target", target=best.target.id)

        # -- gesture transitions ---------------------------------------------- #
        if gesture is not None and gesture.gesture != self._last_gesture:
            if gesture.gesture != "NONE":
                self._emit(EventType.GESTURE_ENTER, ts, gesture.confidence, source="gesture", gesture=gesture.gesture)
            if self._last_gesture != "NONE":
                self._emit(EventType.GESTURE_EXIT, ts, 1.0, source="gesture", gesture=self._last_gesture)
            self._last_gesture = gesture.gesture

        return list(self._pending)

    # ------------------------------------------------------------------ #
    def emit_custom(self, etype: EventType, timestamp: float, confidence: float = 1.0, **data) -> Event:
        """Let other layers (commit detector, state machine) inject events."""
        ev = Event(type=etype, timestamp=timestamp, confidence=confidence, source="engine", data=data)
        self._counts[etype.value] = self._counts.get(etype.value, 0) + 1
        return ev
