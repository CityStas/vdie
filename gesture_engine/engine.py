"""The engine: wires every layer together and runs one frame end to end.

Pipeline (V1 baseline and V2 research layers coexist here):

    source -> tracker -> features -> motion -> history -> events
           -> gesture (evidence)  ->  targets -> commit
           -> intent field (V2) or V1 intent engine (baseline)
           -> action policy -> cursor controller -> dispatcher -> OS

Everything is driven by ``process_frame()``, which is a pure function of
``(frame, timestamp)`` plus internal state.  That is what makes replay exact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .config import EngineConfig
from .control.cursor import CursorController, CursorState
from .control.bimanual import BimanualInteraction
from .control.dispatcher import ActionDispatcher
from .debug.recorder import Recorder
from .debug.diagnostic import DiagnosticTrace
from .debug.telemetry import StageTimer, Telemetry
from .events.event_engine import EventEngine
from .features.feature_vector import build_features
from .gestures.recognizer import GestureRecognizer
from .grammar.sequence_parser import GestureGrammar
from .intent.intent_engine import V1IntentEngine
from .intent.intent_field import IntentFieldBuilder
from .motion.history import HistoryBuffer
from .motion.motion_engine import MotionEngine
from .motion.trajectory import candidate_futures
from .policy.action_policy import ActionPolicy
from .policy.commit import CommitDetector
from .state.state_machine import FSMContext, State, StateMachine
from .targets.target_model import TargetModel, TargetProvider
from .types import (
    ActionRequest,
    CommitSignal,
    EventType,
    FrameReport,
    GestureResult,
    Intent,
    IntentField,
    MotionState,
    Observation,
)
from .user.user_model import UserModel

log = logging.getLogger(__name__)


class TrackerLike(Protocol):
    def process(self, frame: np.ndarray | None, timestamp: float, frame_id: int = 0, camera: dict | None = None) -> Observation | None: ...

    def close(self) -> None: ...


@dataclass
class EngineStats:
    frames: int = 0
    hands_seen: int = 0
    commits: int = 0
    actions: int = 0
    activations: int = 0


class GestureEngine:
    """Top-level orchestrator."""

    def __init__(
        self,
        cfg: EngineConfig,
        tracker: TrackerLike,
        dispatcher: ActionDispatcher | None = None,
        recorder: Recorder | None = None,
        target_provider: TargetProvider | None = None,
        screen: tuple[int, int] | None = None,
        diagnostic: DiagnosticTrace | None = None,
    ) -> None:
        self.cfg = cfg
        self.tracker = tracker
        self.dispatcher = dispatcher or ActionDispatcher(cfg=cfg, dry_run=True)
        self.recorder = recorder
        self.diagnostic = diagnostic

        self.motion = MotionEngine(cfg)
        self.recognizer = GestureRecognizer(cfg)
        self.events = EventEngine(cfg)
        self.fsm = StateMachine(cfg=cfg)
        self.targets = TargetModel(cfg, provider=target_provider)
        self.commit = CommitDetector(cfg=cfg)
        self.intent_field = IntentFieldBuilder(cfg)
        self.v1_intent = V1IntentEngine(cfg)
        self.policy = ActionPolicy(cfg)
        self.cursor = CursorController(cfg, screen=screen or self.dispatcher.mouse.screen_size())
        self.bimanual = BimanualInteraction(cfg, self.cursor.touch_surface)
        self.grammar = GestureGrammar(cfg)
        self.user_model = UserModel(cfg)
        self.telemetry = Telemetry()
        self.stats = EngineStats()

        self._last_timestamp: float | None = None
        self._last_report: FrameReport | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        self.bimanual.reset()
        if self.recorder is not None:
            self.recorder.open()
        if self.diagnostic is not None:
            self.diagnostic.open()

    def stop(self) -> None:
        """Stop the engine and release every owned external resource."""
        try:
            self.dispatcher.close()
        finally:
            try:
                self.targets.close()
            finally:
                try:
                    self.tracker.close()
                finally:
                    if self.recorder is not None:
                        self.recorder.close()
                    if self.diagnostic is not None:
                        self.diagnostic.close()

    def __enter__(self) -> "GestureEngine":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    @property
    def last_report(self) -> FrameReport | None:
        return self._last_report

    # ------------------------------------------------------------------ #
    # One frame
    # ------------------------------------------------------------------ #
    def process_frame(
        self,
        frame: np.ndarray | None,
        timestamp: float,
        frame_id: int = 0,
        camera_meta: dict | None = None,
    ) -> FrameReport | None:
        timer = StageTimer()
        cfg = self.cfg

        dt = 1.0 / max(cfg.capture.fps, 1) if self._last_timestamp is None else max(timestamp - self._last_timestamp, 1e-4)
        self._last_timestamp = timestamp

        # ---- tracking -------------------------------------------------- #
        with timer.stage("tracking"):
            obs = self.tracker.process(frame, timestamp, frame_id, camera_meta)
        if obs is None:
            return None
        obs.timestamp = timestamp if not obs.camera.get("replay") else obs.timestamp
        self.stats.frames += 1
        if obs.has_hand:
            self.stats.hands_seen += 1

        # ---- features -------------------------------------------------- #
        features = None
        secondary_features: dict[str, Any] = {}
        with timer.stage("features"):
            if obs.hand is not None:
                features = build_features(obs.hand.points, cfg)
                obs.with_features(features)
            for label, hand in (obs.hands or ({obs.primary_hand or "hand": obs.hand} if obs.hand is not None else {})).items():
                if obs.primary_hand is not None and label == obs.primary_hand:
                    continue
                if hand is not None:
                    secondary_features[label] = build_features(hand.points, cfg)

        # ---- motion ---------------------------------------------------- #
        with timer.stage("motion"):
            motion = self.motion.update(obs, features)

        # ---- gesture (evidence) ---------------------------------------- #
        with timer.stage("gesture"):
            gesture: GestureResult = self.recognizer.update(features, self.motion.history, obs.timestamp, obs.hand_confidence)

        # ---- state machine --------------------------------------------- #
        ctx = FSMContext(timestamp=obs.timestamp, features=features, motion=motion, gesture=gesture, hand_confidence=obs.hand_confidence)
        prev_state = self.fsm.state
        self.fsm.update(ctx)
        if prev_state is State.ARMED and self.fsm.state is State.CURSOR:
            self.stats.activations += 1
            # A fresh activation puts the cursor *where the hand is*.  Without
            # this the controller starts from its initial (0, 0) and slides
            # across the screen, which is both bad UX and was being measured as
            # 233-467 ms of "cursor lag" in the benchmark.
            try:
                os_cursor = self.dispatcher.mouse.position()
            except Exception:
                os_cursor = self.cursor.position.copy()
            self.cursor.anchor_to(motion.position, cursor_position=os_cursor)
        # Explicit activation diagnostics: this makes a failed activation explainable
        # without having to infer it from the final FSM state.
        activation_debug = None
        if features is not None:
            index_only = (
                features.fingers[1] == 1
                and features.fingers[2] == 0
                and features.fingers[3] == 0
                and features.fingers[4] == 0
            )
            vertical = abs(features.index_orientation) <= cfg.state.activation_max_tilt_deg
            still = motion.speed < cfg.state.activation_velocity
            confident = obs.hand_confidence >= cfg.state.activation_confidence
            recognized = (
                gesture.gesture == "INDEX_UP"
                and gesture.confidence >= cfg.gestures.min_confidence
                and gesture.stability >= 0.45
                and still
                and obs.hand_confidence >= min(cfg.state.activation_confidence, 0.70)
            )
            activation_debug = {
                "index_only": index_only,
                "vertical": vertical,
                "still": still,
                "confident": confident,
                "geometric": index_only and vertical and still and confident,
                "recognized_index_up": recognized,
                "activation_ready": (index_only and vertical and still and confident) or recognized,
            }
        fsm_events = self.fsm.events(ctx)

        # ---- targets --------------------------------------------------- #
        # Target geometry lives in desktop/screen space.  In cooperative cursor
        # mode the hand is a *control signal*, not the screen coordinate, so target
        # inference must use the actual cursor and its velocity.
        interaction_position = self.cursor.normalized_position()
        interaction_velocity = self.cursor.normalized_velocity()
        with timer.stage("targets"):
            if self.fsm.cursor_enabled:
                provider = self.targets.provider
                set_probe = getattr(provider, "set_probe_path", None)
                if callable(set_probe):
                    cursor_px = np.asarray(self.cursor.position, dtype=np.float64)
                    cursor_v = np.asarray(self.cursor.velocity, dtype=np.float64)
                    probe_points: list[tuple[float, float]] = []
                    for horizon in (0.0, 0.05, 0.10, 0.16, 0.24):
                        q = cursor_px + cursor_v * horizon
                        probe_points.append((float(q[0]), float(q[1])))
                    set_probe(probe_points)
            self.targets.refresh(obs.timestamp)
            if self.fsm.cursor_enabled:
                beliefs = self.targets.beliefs(
                    interaction_position, interaction_velocity, self.intent_field.last_field, obs.timestamp
                )
                gravity = self.targets.gravity(
                    interaction_position, interaction_velocity, beliefs, self.intent_field.last_field, dt
                )
            else:
                beliefs = []
                gravity = np.zeros(2)

        # ---- commit point ---------------------------------------------- #
        with timer.stage("commit"):
            commit: CommitSignal = self.commit.update(
                motion, self.motion.history, features, beliefs, self.fsm, obs.timestamp,
                interaction_position=interaction_position,
            )
        if commit.committed:
            self.stats.commits += 1

        # ---- events ----------------------------------------------------- #
        with timer.stage("events"):
            events = self.events.update(
                obs,
                motion,
                features,
                gesture,
                beliefs,
                pinch_distance=features.pinch_distance if features else None,
            )
            events.extend(fsm_events)
            if commit.committed:
                events.append(
                    self.events.emit_custom(EventType.COMMIT, obs.timestamp, commit.score, kind=commit.kind)
                )

        # ---- counterfactual futures ------------------------------------- #
        futures = None
        if cfg.intent.futures_enabled:
            future_position = interaction_position if cfg.cursor.controller in ("cooperative", "touch_surface") else motion.position
            future_velocity = interaction_velocity if cfg.cursor.controller in ("cooperative", "touch_surface") else motion.velocity
            future_acceleration = np.zeros(2) if cfg.cursor.controller == "cooperative" else motion.acceleration
            futures = candidate_futures(
                future_position,
                future_velocity,
                future_acceleration,
                horizons=tuple(cfg.intent.futures_horizons),
                samples=cfg.intent.futures_samples,
            )

        # ---- intent ------------------------------------------------------ #
        with timer.stage("intent"):
            intents = self.intent_field.update(
                timestamp=obs.timestamp,
                features=features,
                motion=motion,
                gesture=gesture,
                history=self.motion.history,
                events=events,
                beliefs=beliefs,
                fsm=self.fsm,
                commit=commit,
                futures=futures,
                user_signature=self.user_model.signature,
            )
            v1 = self.v1_intent.update(gesture, self.fsm, motion, features, obs.timestamp)

        # ---- grammar ----------------------------------------------------- #
        grammar_matches = self.grammar.push(gesture.gesture, obs.timestamp) if gesture else []

        # ---- policy ------------------------------------------------------ #
        with timer.stage("policy"):
            actions = self.policy.update(
                intents=intents,
                commit=commit,
                gesture=gesture,
                motion=motion,
                features=features,
                beliefs=beliefs,
                fsm=self.fsm,
                timestamp=obs.timestamp,
                bimanual_active=bool(
                    self.bimanual.snapshot.enabled
                    and (
                        self.bimanual.snapshot.secondary_pinch
                        or self.bimanual.snapshot.zoom_active
                        or self.bimanual.snapshot.selection_active
                        or self.bimanual.snapshot.pan_active
                    )
                ),
            )
            for m in grammar_matches:
                actions.append(
                    ActionRequest(intent=Intent.CUSTOM, kind="custom", payload={"name": m.name, "sequence": m.pattern}, risk=0.6, timestamp=obs.timestamp)
                )

        # ---- cursor ------------------------------------------------------ #
        with timer.stage("cursor"):
            cursor_state: CursorState = self.cursor.update(
                motion, features, intents, beliefs, self.fsm, dt, obs.timestamp, gravity_force=gravity
            )

        # ---- bimanual interaction --------------------------------------- #
        with timer.stage("bimanual"):
            # A second-hand modifier is an action channel, not an activation
            # gesture. It must never click/drag while the primary pointer is
            # still IDLE/ACTIVATING/PAUSED.
            if self.fsm.accepts_actions:
                bimanual_actions = self.bimanual.update(
                    obs.hands or ({obs.primary_hand or "hand": obs.hand} if obs.hand is not None else {}),
                    obs.hand_confidences or ({obs.primary_hand or "hand": obs.hand_confidence} if obs.hand is not None else {}),
                    obs.primary_hand,
                    obs.timestamp,
                    cursor_state.position.copy(),
                    dt,
                )
            else:
                bimanual_actions = self.bimanual.update({}, {}, None, obs.timestamp, cursor_state.position.copy(), dt)
                self.bimanual.snapshot.gate_reason = f"fsm_{self.fsm.state.value.lower()}"
            if bimanual_actions:
                actions.extend(bimanual_actions)

        # ---- user model --------------------------------------------------- #
        self.user_model.observe(motion, commit)

        # ---- dispatch ------------------------------------------------------ #
        with timer.stage("dispatch"):
            suppress_pointer_move = any(
                str(a.payload.get("source", "")) in {"bimanual_selection", "bimanual_pan"}
                and a.kind in {"move", "down", "up"}
                for a in bimanual_actions
            )
            if self.fsm.cursor_enabled and cfg.control.move_mouse and not suppress_pointer_move:
                # CursorController already works in physical screen pixels.
                # Do not normalize a second time here: that used to collapse
                # every live move into roughly (0..1, 0..1) pixels.
                self.dispatcher.move_cursor(cursor_state.position)
            self.dispatcher.dispatch(actions, cursor=cursor_state.position, timestamp=obs.timestamp)
        if actions:
            self.stats.actions += len(actions)

        # ---- diagnostics ----------------------------------------------------
        raw_index_tip = None
        touch_error_px = None
        if obs.hand is not None:
            raw_index_tip = np.asarray(obs.hand.points, dtype=np.float64)[8, :2]
            try:
                mapped_raw = self.cursor.touch_surface.map_point(raw_index_tip)
                touch_error_px = float(np.linalg.norm(mapped_raw - cursor_state.position))
            except Exception:
                pass

        previous_cursor_px = None
        cursor_delta_px = None
        if self._last_report is not None and self._last_report.cursor is not None:
            prev_norm = np.asarray(self._last_report.cursor, dtype=np.float64)
            previous_cursor_px = prev_norm * np.asarray(self.cursor.screen, dtype=np.float64)
            cursor_delta_px = float(np.linalg.norm(cursor_state.position - previous_cursor_px))

        # ---- report -------------------------------------------------------- #
        fps = self.telemetry.tick(obs.timestamp, timer)
        report = FrameReport(
            frame_id=obs.frame_id,
            timestamp=obs.timestamp,
            fps=fps,
            stage_ms=timer.as_dict(),
            state=self.fsm.state.value,
            gesture=gesture,
            intents=intents,
            motion=motion,
            features=features,
            target_beliefs=beliefs,
            actions=actions,
            cursor=self._screen_norm(cursor_state.position),
            events=events,
            commit=commit,
            latency_ms=timer.total_ms,
            notes={
                "dt": dt,
                "v1_intent": v1.dominant.value,
                "cursor_mode": cursor_state.mode.value,
                "gravity": gravity.tolist(),
                "grammar_partial": self.grammar.partial(),
                "futures": len(futures) if futures else 0,
                "user_commit_pref": self.user_model.commit_preference(),
                "policy": self.policy.last_decision.reason if self.policy.last_decision else "",
                "label": obs.camera.get("label", ""),
                "intent_label": obs.camera.get("intent_label", ""),
                "hand_points": obs.hand.points if obs.hand is not None else None,
                "hands": {k: v.points for k, v in (obs.hands or {}).items()},
                "hand_confidences": dict(obs.hand_confidences),
                "primary_hand": obs.primary_hand,
                "hand_count": obs.hand_count,
                "index_tip_raw": raw_index_tip,
                "hand_confidence": obs.hand_confidence,
                "event_types": [e.type.value for e in events],
                "screen": list(self.cursor.screen),
                "cursor_pixel": cursor_state.position.copy(),
                "cursor_target": cursor_state.target_id,
                "cursor_target_score": cursor_state.target_score,
                "spatial_intent": cursor_state.spatial_intent,
                "index_tip_prediction": cursor_state.predicted_hand.copy(),
                "cursor_enabled": self.fsm.cursor_enabled,
                "fsm_previous": prev_state.value,
                "fsm_since": self.fsm.since,
                "activation_dwell": self.fsm._activation_dwell,
                "activation_debug": activation_debug,
                "bimanual": self.bimanual.snapshot.as_dict(),
                "touch_surface": self.cursor.touch_surface.state(),
                "touch_error_px": touch_error_px,
                "cursor_delta_px": cursor_delta_px,
                "finger_states": list(features.fingers) if features is not None else None,
                "index_orientation_deg": float(features.index_orientation) if features is not None else None,
                "extended_count": int(features.extended_count) if features is not None else None,
                "pinch_distance": float(features.pinch_distance) if features is not None else None,
                "hand_confidence_raw": float(obs.hand_confidence),
                "dispatcher_last": self.dispatcher.audit()[-1] if self.dispatcher.records else None,
                "dispatcher_records": len(self.dispatcher.records),
                "actions_dispatched_this_frame": len(actions) + (1 if self.fsm.cursor_enabled and cfg.control.move_mouse else 0),
            },
        )
        self._last_report = report

        if self.recorder is not None:
            self.recorder.record(report)

        if self.diagnostic is not None:
            self.diagnostic.write({
                "schema": "diagnostic_v2",
                "frame": report.frame_id,
                "timestamp": report.timestamp,
                "fps": report.fps,
                "state": report.state,
                "gesture": report.gesture.gesture,
                "gesture_confidence": report.gesture.confidence,
                "gesture_stability": report.gesture.stability,
                "hand_confidence": obs.hand_confidence,
                "has_hand": obs.has_hand,
                "finger_states": list(features.fingers) if features is not None else None,
                "index_orientation_deg": float(features.index_orientation) if features is not None else None,
                "extended_count": int(features.extended_count) if features is not None else None,
                "pinch_distance": float(features.pinch_distance) if features is not None else None,
                "motion_position": motion.position,
                "motion_speed": motion.speed,
                "motion_confidence": motion.confidence,
                "intent": {k.value: float(v) for k, v in report.intents.probabilities.items()},
                "cursor_enabled": self.fsm.cursor_enabled,
                "cursor_mode": cursor_state.mode.value,
                "cursor_pixel": cursor_state.position,
                "screen": self.cursor.screen,
                "cursor_gain": cursor_state.gain,
                "cursor_frozen": cursor_state.frozen,
                "cursor_velocity": cursor_state.velocity,
                "cursor_filtered_hand": cursor_state.filtered_hand,
                "cursor_predicted_hand": cursor_state.predicted_hand,
                "cursor_target": cursor_state.target_id,
                "cursor_target_score": cursor_state.target_score,
                "spatial_intent": cursor_state.spatial_intent,
                "cursor_target_scores": {b.target.id: float(b.confidence) for b in beliefs[:12]},
                "cursor_target_states": {b.target.id: b.target.state.value for b in beliefs[:12]},
                "index_tip_raw": raw_index_tip,
                "index_tip_filtered": cursor_state.filtered_hand,
                "index_tip_prediction": cursor_state.predicted_hand,
                "cursor_target_bounds": (
                    next((b.target.bounds for b in beliefs if b.target.id == cursor_state.target_id), None)
                ),
                "actions": [{"kind": a.kind, "intent": a.intent.value, "payload": a.payload} for a in actions],
                "dispatcher_records": len(self.dispatcher.records),
                "dispatcher_last": self.dispatcher.audit()[-1] if self.dispatcher.records else None,
                "fsm_transition": (f"{prev_state.value}->{self.fsm.state.value}" if prev_state is not self.fsm.state else None),
                "fsm_reason": (self.fsm.transitions[-1].reason if self.fsm.transitions and self.fsm.transitions[-1].timestamp == obs.timestamp else None),
                "activation_dwell": self.fsm._activation_dwell,
                "activation_debug": activation_debug,
                "hands": {k: {"conf": float(obs.hand_confidences.get(k, 0.0)), "index_tip": [float(v.points[8,0]), float(v.points[8,1])]} for k, v in (obs.hands or {}).items()},
                "primary_hand": obs.primary_hand,
                "hand_count": obs.hand_count,
                "bimanual": self.bimanual.snapshot.as_dict(),
                "touch_surface": self.cursor.touch_surface.state(),
                "stage_ms": report.stage_ms,
            })
        self._count_telemetry(report)
        return report

    # ------------------------------------------------------------------ #
    def _screen_norm(self, position_px: np.ndarray) -> np.ndarray:
        w, h = self.cursor.screen
        return np.array([float(position_px[0]) / max(w, 1), float(position_px[1]) / max(h, 1)], dtype=np.float64)

    def _count_telemetry(self, report: FrameReport) -> None:
        t = self.telemetry
        t.count("frames")
        if report.gesture.gesture != "NONE":
            t.count(f"gesture:{report.gesture.gesture}")
        if report.commit is not None and report.commit.committed:
            t.count(f"commit:{report.commit.kind}")
        for a in report.actions:
            t.count(f"action:{a.kind}")
        if report.state in ("IDLE", "PAUSED") and report.intents.get(Intent.SELECT) > 0.5:
            t.count("possible_false_activation")

    # ------------------------------------------------------------------ #
    # Convenience: run a source to exhaustion (used by replay and synthetic)
    # ------------------------------------------------------------------ #
    def run_source(self, source: Any, max_frames: int | None = None, callback=None) -> list[FrameReport]:
        reports: list[FrameReport] = []
        i = 0
        self.start()
        try:
            while True:
                if max_frames is not None and i >= max_frames:
                    break
                frame_obj = source.read() if hasattr(source, "read") else None
                image = frame_obj.image if frame_obj is not None else None
                timestamp = frame_obj.timestamp if frame_obj is not None else 0.0
                meta = frame_obj.meta if frame_obj is not None else {}
                report = self.process_frame(image, timestamp, i, meta)
                if report is None:
                    break
                reports.append(report)
                if callback is not None:
                    callback(report, image)
                i += 1
        finally:
            self.stop()
        return reports
