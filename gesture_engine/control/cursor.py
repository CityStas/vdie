"""Experimental cursor controller.

The cursor is a *controller*, not a coordinate mapper.  This module is the
single place where all of the "cooperative cursor" ideas meet:

* velocity-dependent gain (V1 sensitivity curve),
* intent-adaptive gain — when the intent field is ambiguous the cursor
  deliberately hesitates instead of racing (see :meth:`_intent_gain`),
* uncertainty as a control signal — low tracking confidence reduces gain, very
  low confidence freezes (section 56),
* soft target gravity, disabled during fast travel,
* precision mode near a target when the user is selecting,
* short-horizon prediction (lead) to hide pipeline latency,
* recovery: coast on a constant-velocity model while tracking is lost, then
  re-anchor without a jump (section 57).

Three controller back-ends are implemented so they can be benchmarked against
each other: ``gain`` (plain alpha EMA), ``velocity_curve`` (V1 adaptive EMA) and
``spring`` (default; a second-order critically-damped tracking controller).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..config import EngineConfig
from ..features.geometry import clamp01, smoothstep
from ..filters.ema import AdaptiveEMA, OneEuroFilter, VelocityEstimator
from ..motion.trajectory import predict_point
from ..state.state_machine import StateMachine
from .touch_surface import TouchSurfaceMapper
from ..types import HandFeatures, Intent, IntentField, MotionState, TargetBelief


class CursorMode(str, Enum):
    FAST_TRAVEL = "FAST_TRAVEL"
    PRECISION = "PRECISION"
    TARGET_APPROACH = "TARGET_APPROACH"
    DRAG = "DRAG"
    RECOVERY = "RECOVERY"
    PAUSED = "PAUSED"


class _AlphaBetaPointer:
    """Low-latency predictor for the index fingertip.

    This is deliberately not an EMA.  It estimates position *and* velocity,
    predicts through the next sample, and uses the measurement residual to
    correct both states.  That gives the cooperative cursor a stable control
    signal without turning the pointer into a slow, rubbery follower.
    """

    def __init__(self, alpha_slow: float, alpha_fast: float, beta: float, max_innovation: float, stationary_speed: float, prediction_horizon: float) -> None:
        self.alpha_slow = float(alpha_slow)
        self.alpha_fast = float(alpha_fast)
        self.beta = float(beta)
        self.max_innovation = float(max_innovation)
        self.stationary_speed = float(stationary_speed)
        self.prediction_horizon = float(np.clip(prediction_horizon, 0.0, 0.20))
        self.position: np.ndarray | None = None
        self.velocity = np.zeros(2, dtype=np.float64)
        self._last_raw: np.ndarray | None = None

    def reset(self, position: np.ndarray | None = None) -> None:
        self.position = None if position is None else np.asarray(position, dtype=np.float64)[:2].copy()
        self.velocity = np.zeros(2, dtype=np.float64)
        self._last_raw = None if position is None else np.asarray(position, dtype=np.float64)[:2].copy()

    def update(self, measurement: np.ndarray, dt: float, confidence: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = np.asarray(measurement, dtype=np.float64)[:2]
        dt = float(np.clip(dt, 1e-3, 0.12))
        if self.position is None:
            self.position = m.copy()
            self._last_raw = m.copy()
            return self.position.copy(), self.velocity.copy(), self.position.copy()

        raw_velocity = np.zeros(2, dtype=np.float64)
        if self._last_raw is not None:
            raw_velocity = (m - self._last_raw) / dt
        raw_speed = float(np.linalg.norm(raw_velocity))
        u = float(np.clip((raw_speed - 0.03) / 0.9, 0.0, 1.0))
        alpha = self.alpha_slow + (self.alpha_fast - self.alpha_slow) * u
        alpha *= float(np.clip(0.78 + 0.22 * confidence, 0.70, 1.0))

        predicted = self.position + self.velocity * dt
        residual = m - predicted
        innovation = float(np.linalg.norm(residual))
        if innovation > self.max_innovation:
            residual *= self.max_innovation / max(innovation, 1e-9)

        self.position = predicted + alpha * residual
        self.velocity = self.velocity + (self.beta / dt) * residual
        speed = float(np.linalg.norm(self.velocity))
        if speed > 4.0:
            self.velocity *= 4.0 / max(speed, 1e-9)

        # When the actual fingertip is stationary, kill only the estimated
        # velocity; do not pull the position toward an artificial origin.
        if raw_speed < self.stationary_speed and innovation < 0.02:
            self.velocity *= math.exp(-10.0 * dt)

        self._last_raw = m.copy()
        predicted_next = self.position + self.velocity * self.prediction_horizon
        return self.position.copy(), self.velocity.copy(), predicted_next


@dataclass
class CursorState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    mode: CursorMode = CursorMode.PAUSED
    gain: float = 0.0
    filtered_hand: np.ndarray = field(default_factory=lambda: np.zeros(2))
    predicted_hand: np.ndarray = field(default_factory=lambda: np.zeros(2))
    gravity_force: np.ndarray = field(default_factory=lambda: np.zeros(2))
    intent_entropy: float = 0.0
    confidence: float = 1.0
    target_id: str | None = None
    target_score: float = 0.0
    spatial_intent: float = 0.0
    frozen: bool = False

    def as_dict(self) -> dict:
        return {
            "x": float(self.position[0]),
            "y": float(self.position[1]),
            "mode": self.mode.value,
            "gain": self.gain,
            "target": self.target_id,
            "frozen": self.frozen,
        }


class CursorController:
    def __init__(self, cfg: EngineConfig, screen: tuple[int, int] = (1920, 1080)) -> None:
        self.cfg = cfg
        self.screen = (int(screen[0]), int(screen[1]))
        ccfg = cfg.cursor

        self.ema = AdaptiveEMA(
            alpha_slow=ccfg.alpha_slow,
            alpha_medium=ccfg.alpha_medium,
            alpha_fast=ccfg.alpha_fast,
            speed_slow=cfg.motion.pause_speed * 1.2,
            speed_fast=cfg.motion.fast_speed,
        )
        self.euro = OneEuroFilter(
            min_cutoff=ccfg.euro_min_cutoff,
            beta=ccfg.euro_beta,
            d_cutoff=ccfg.euro_d_cutoff,
            dim=2,
        )
        self.speed_filter = VelocityEstimator(alpha=0.45)

        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.mode = CursorMode.PAUSED
        self._anchor_hand: np.ndarray | None = None
        self._anchor_cursor: np.ndarray | None = None
        self._last_hand: np.ndarray | None = None
        self._prev_desired: np.ndarray | None = None
        self._last_dt = 1.0 / 30.0
        self._frozen = False
        self._pause_since: float | None = None
        # Cooperative cursor state.  This mode is deliberately independent of
        # the legacy spring/absolute controller so the research baseline remains
        # reproducible while live desktop control gets a purpose-built path.
        self._hand_velocity = np.zeros(2)
        self._locked_target_id: str | None = None
        self._locked_target_since: float | None = None
        self._locked_target_score = 0.0
        self._coop_last_hand: np.ndarray | None = None
        self._coop_prev_control: np.ndarray | None = None
        self._coop_prev_raw: np.ndarray | None = None
        self._coop_raw_deltas: deque[np.ndarray] = deque(maxlen=3)
        self._coop_delta_ema = np.zeros(2, dtype=np.float64)
        self._coop_noise_speed = 0.0025
        self._coop_rearm = False
        self._coop_predictor = _AlphaBetaPointer(
            alpha_slow=ccfg.cooperative_predict_alpha_slow,
            alpha_fast=ccfg.cooperative_predict_alpha_fast,
            beta=ccfg.cooperative_predict_beta,
            max_innovation=ccfg.cooperative_max_innovation,
            stationary_speed=ccfg.cooperative_stationary_speed,
            prediction_horizon=ccfg.cooperative_predict_horizon,
        )
        self.touch_surface = TouchSurfaceMapper(
            screen=self.screen,
            calibration_path=ccfg.touch_surface_calibration_path,
            use_calibration=ccfg.touch_surface_use_calibration,
            hysteresis_px=ccfg.touch_surface_hysteresis_px,
        )
        self._touch_predictor = _AlphaBetaPointer(
            alpha_slow=ccfg.touch_surface_alpha_slow,
            alpha_fast=ccfg.touch_surface_alpha_fast,
            beta=ccfg.touch_surface_beta,
            max_innovation=ccfg.touch_surface_max_innovation,
            stationary_speed=ccfg.touch_surface_stationary_speed,
            prediction_horizon=ccfg.touch_surface_prediction_horizon,
        )
        self._touch_reanchor = False
        self._touch_last_screen_target: np.ndarray | None = None
        self._state = CursorState()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.ema.reset()
        self.euro.reset()
        self.speed_filter.reset()
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.mode = CursorMode.PAUSED
        self._anchor_hand = None
        self._anchor_cursor = None
        self._last_hand = None
        self._prev_desired = None
        self._frozen = False
        self._pause_since = None
        self._hand_velocity = np.zeros(2)
        self._locked_target_id = None
        self._locked_target_since = None
        self._locked_target_score = 0.0
        self._coop_last_hand = None
        self._coop_prev_control = None
        self._coop_prev_raw = None
        self._coop_raw_deltas.clear()
        self._coop_delta_ema[:] = 0.0
        self._coop_noise_speed = 0.0025
        self._coop_rearm = False
        self._coop_predictor.reset()
        self.touch_surface.reset()
        self._touch_predictor.reset()
        self._touch_reanchor = False
        self._touch_last_screen_target = None
        self._state = CursorState()

    def set_screen(self, size: tuple[int, int]) -> None:
        self.screen = (int(size[0]), int(size[1]))
        self.touch_surface.set_screen(self.screen)

    # ------------------------------------------------------------------ #
    # Mapping
    # ------------------------------------------------------------------ #
    def _map_to_screen(self, normalized: np.ndarray) -> np.ndarray:
        """Normalized hand position -> screen pixels through the active region."""
        rx0, ry0, rx1, ry1 = self.cfg.control.active_region
        u = (normalized[0] - rx0) / max(rx1 - rx0, 1e-6)
        v = (normalized[1] - ry0) / max(ry1 - ry0, 1e-6)
        return np.array([clamp01(u) * self.screen[0], clamp01(v) * self.screen[1]], dtype=np.float64)

    def _sensitivity_curve(self, value: float) -> float:
        """``sign(d) * |d|^gamma``; range-preserving (1.0 -> 1.0)."""
        gamma = self.cfg.cursor.gamma
        return float(np.sign(value) * (abs(value) ** gamma))

    # ------------------------------------------------------------------ #
    # Gain scheduling
    # ------------------------------------------------------------------ #
    def _speed_gain(self, speed: float) -> float:
        mcfg = self.cfg.motion
        u = clamp01((speed - mcfg.pause_speed) / max(mcfg.fast_speed - mcfg.pause_speed, 1e-6))
        return 0.55 + 0.45 * smoothstep(0.0, 1.0, u)

    def _intent_gain(self, intents: IntentField) -> float:
        """Hesitation under ambiguity — applied to *responsiveness*, not position.

        Counter-intuitive but deliberate: when the intent field is flat the
        cursor becomes *less* responsive.  Racing towards a region while the
        user has not committed is exactly how accidental clicks happen.

        The gain is applied to the velocity feed-forward, not to the spring
        stiffness.  Both suppress the racing, but only one of them is free:
        scaling the stiffness scales the *tracking* too, so the cursor lags the
        hand by an extra 49 ms whenever the field is flat — measured, and paid
        on every frame, while the commit precision it was supposed to protect
        stayed at 1.00 either way.  Scaling the feed-forward leaves positional
        tracking untouched and only refuses to *amplify* fast movement.

        The ambiguity measure is the **decision margin**, not Shannon entropy
        over all intents.  With twelve intents the normalised entropy of a field
        whose dominant intent already holds 0.80 is 0.39 — entropy treats the
        eleven near-zero rivals as live alternatives and the cursor hesitates
        for no reason.  The margin asks the question that actually matters: how
        far ahead is the leader?
        """
        if not self.cfg.cursor.intent_adaptive:
            return 1.0
        ranked = sorted(intents.probabilities.values(), reverse=True)
        if len(ranked) < 2:
            return 1.0
        margin = float(ranked[0] - ranked[1])
        return float(1.0 - self.cfg.cursor.intent_entropy_gain * (1.0 - margin))

    def _uncertainty_gain(self, confidence: float) -> float:
        c = self.cfg.cursor
        if confidence <= c.uncertainty_freeze_below:
            return 0.0
        floor = c.uncertainty_gain_floor
        return float(floor + (1.0 - floor) * smoothstep(c.uncertainty_freeze_below, 0.75, confidence))

    def _target_gain(self, beliefs: list[TargetBelief], intents: IntentField) -> tuple[float, str | None]:
        """Reduce gain near a target while selecting, for fine positioning."""
        if not self.cfg.cursor.precision_near_target or not beliefs:
            return 1.0, None
        selecting = max(intents.get(Intent.SELECT), intents.get(Intent.DRAG))
        if selecting < 0.35:
            return 1.0, None
        near = min(beliefs, key=lambda b: b.distance)
        if near.distance > self.cfg.targets.gravity_radius * 3.0:
            return 1.0, None
        u = clamp01(1.0 - near.distance / (self.cfg.targets.gravity_radius * 3.0))
        gain = 1.0 - (1.0 - self.cfg.cursor.precision_gain) * u * selecting
        return float(gain), near.target.id

    # ------------------------------------------------------------------ #
    def update(
        self,
        motion: MotionState,
        features: HandFeatures | None,
        intents: IntentField,
        beliefs: list[TargetBelief],
        fsm: StateMachine,
        dt: float,
        timestamp: float,
        gravity_force: np.ndarray | None = None,
    ) -> CursorState:
        ccfg = self.cfg.cursor
        self._last_dt = max(dt, 1e-4)

        if not fsm.cursor_enabled:
            self._pause_since = timestamp
            self.mode = CursorMode.PAUSED
            self.velocity *= 0.0
            self._coop_rearm = True
            self._state = CursorState(
                position=self.position.copy(),
                velocity=self.velocity.copy(),
                mode=self.mode,
                gain=0.0,
                intent_entropy=intents.entropy,
                confidence=motion.confidence,
                frozen=True,
            )
            return self._state

        # ---------------- tracking loss / recovery --------------------- #
        if motion.confidence < self.cfg.safety.freeze_below_confidence:
            if not ccfg.recovery_enabled:
                self._frozen = True
                self.mode = CursorMode.RECOVERY
                self._state = CursorState(
                    position=self.position.copy(),
                    velocity=self.velocity.copy(),
                    mode=self.mode,
                    confidence=motion.confidence,
                    frozen=True,
                )
                return self._state
            # Coast on a constant-velocity model, decaying to a stop.
            horizon = ccfg.recovery_prediction_horizon
            coast = self.position + self.velocity * dt
            self.velocity *= float(np.clip(1.0 - dt / max(horizon, 1e-3), 0.0, 1.0))
            self.position = coast
            self._frozen = True
            self._touch_reanchor = self.cfg.cursor.controller == "touch_surface"
            self.mode = CursorMode.RECOVERY
            self._state = CursorState(
                position=self.position.copy(),
                velocity=self.velocity.copy(),
                mode=self.mode,
                confidence=motion.confidence,
                frozen=True,
            )
            return self._state

        # ---------------- normal operation ----------------------------- #
        if ccfg.controller == "touch_surface" and not ccfg.relative_mode:
            return self._touch_surface_update(motion, intents, fsm, dt)

        if ccfg.controller == "cooperative" and ccfg.relative_mode:
            return self._cooperative_update(
                motion=motion,
                features=features,
                intents=intents,
                beliefs=beliefs,
                fsm=fsm,
                dt=dt,
                timestamp=timestamp,
                gravity_force=gravity_force,
            )

        raw = np.asarray(motion.position, dtype=np.float64)
        speed = self.speed_filter.update(motion.speed, dt)

        if ccfg.controller == "spring":
            filtered = self.euro.update(raw, dt)
        else:
            filtered = self.ema.update(raw, speed, dt)

        # Re-anchor after a freeze: never let reacquisition teleport the cursor.
        if self._frozen:
            if self._last_hand is not None:
                jump = float(np.linalg.norm(filtered - self._last_hand))
                if jump > ccfg.recovery_max_jump:
                    self.euro.reset()
                    self.ema.reset()
                    # Re-anchoring is a discontinuity: a feed-forward derived
                    # across it would be a velocity spike.
                    self._prev_desired = None
                    filtered = self.euro.update(raw, dt) if ccfg.controller == "spring" else self.ema.update(raw, speed, dt)
            self._frozen = False

        predicted = predict_point(filtered, motion.velocity, motion.acceleration, ccfg.prediction_horizon, ccfg.prediction_accel_weight)

        # ---------------- gain composition ----------------------------- #
        # Stiffness gains.  Each of these trades positional tracking for
        # something worth having: tremor rejection at rest, a hard stop when
        # tracking confidence collapses, fine positioning near a target.
        gain = self._speed_gain(speed)
        gain *= self._uncertainty_gain(motion.confidence)
        target_gain, target_id = self._target_gain(beliefs, intents)
        gain *= target_gain
        gain *= ccfg.sensitivity

        # Responsiveness gain.  Intent ambiguity is deliberately kept out of the
        # stiffness composition above: scaling stiffness scales tracking, and
        # that costs ~49 ms of lag for no measured benefit.
        intent_gain = self._intent_gain(intents)

        if gain <= 1e-4:
            self._frozen = True
            self.mode = CursorMode.PRECISION
            self._state = CursorState(
                position=self.position.copy(),
                velocity=self.velocity.copy(),
                mode=self.mode,
                gain=0.0,
                filtered_hand=filtered,
                predicted_hand=predicted,
                intent_entropy=intents.entropy,
                confidence=motion.confidence,
                target_id=target_id,
                frozen=True,
            )
            return self._state

        # ---------------- desired position ----------------------------- #
        if ccfg.relative_mode:
            desired = self._relative_target(predicted, speed, timestamp)
        else:
            curve = np.array([self._sensitivity_curve(2.0 * (predicted[0] - 0.5)) * 0.5 + 0.5,
                              self._sensitivity_curve(2.0 * (predicted[1] - 0.5)) * 0.5 + 0.5])
            desired = self._map_to_screen(curve)

        # Feed-forward is derived from the *mapping* only, before gravity: the
        # gravity term is an acceleration nudge, and differentiating it into a
        # velocity would double-count it.
        v_ff: np.ndarray | None = None
        if ccfg.feedforward and ccfg.controller == "spring":
            if self._prev_desired is None:
                v_ff = np.zeros(2)
            else:
                v_ff = (desired - self._prev_desired) / self._last_dt
                v_ff = np.clip(v_ff, -ccfg.feedforward_max_px_s, ccfg.feedforward_max_px_s)
            v_ff = v_ff * ccfg.feedforward_gain * intent_gain
        self._prev_desired = desired.copy()

        if gravity_force is not None and np.any(gravity_force):
            # The force is an acceleration; convert it to an equivalent
            # positional offset so it composes with the position controller.
            desired = desired + gravity_force * (dt**2) * 1e4

        # ---------------- control law ---------------------------------- #
        if ccfg.controller == "spring":
            self._spring_step(desired, speed, gain, dt, v_ff)
        else:
            a = self.ema.alpha_for(speed) if ccfg.controller == "velocity_curve" else 0.35
            a = float(np.clip(a * gain, 0.02, 1.0))
            new_pos = self.position + a * (desired - self.position)
            self.velocity = (new_pos - self.position) / dt
            self.position = new_pos

        self._clamp_step(dt)

        # The OS desktop is a hard physical boundary.  A spring/feed-forward
        # controller may overshoot its desired point for one or two frames;
        # never let that numerical overshoot become an out-of-screen command.
        # Without this guard the cursor could oscillate against the Windows
        # edge (and, on some backends, appear to "swim" around the target).
        if self.screen[0] > 0 and self.screen[1] > 0:
            clipped = np.clip(self.position, [0.0, 0.0], [float(self.screen[0] - 1), float(self.screen[1] - 1)])
            if not np.allclose(clipped, self.position):
                self.position = clipped
                # Remove the component of velocity that points outside the
                # boundary, while preserving the component parallel to it.
                if self.position[0] <= 0.0 and self.velocity[0] < 0:
                    self.velocity[0] = 0.0
                elif self.position[0] >= self.screen[0] - 1 and self.velocity[0] > 0:
                    self.velocity[0] = 0.0
                if self.position[1] <= 0.0 and self.velocity[1] < 0:
                    self.velocity[1] = 0.0
                elif self.position[1] >= self.screen[1] - 1 and self.velocity[1] > 0:
                    self.velocity[1] = 0.0

        self._last_hand = filtered.copy()
        self.mode = self._mode_for(fsm, speed, target_id, intents, beliefs)

        self._state = CursorState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mode=self.mode,
            gain=float(gain),
            filtered_hand=filtered,
            predicted_hand=predicted,
            gravity_force=np.zeros(2) if gravity_force is None else np.asarray(gravity_force, dtype=np.float64).copy(),
            intent_entropy=intents.entropy,
            confidence=motion.confidence,
            target_id=target_id,
            frozen=False,
        )
        return self._state

    # ------------------------------------------------------------------ #
    # Absolute virtual touch surface
    # ------------------------------------------------------------------ #
    def _touch_surface_update(
        self, motion: MotionState, intents: IntentField, fsm: StateMachine, dt: float
    ) -> CursorState:
        """Map the filtered index tip directly to the physical desktop.

        This is intentionally position-based, not velocity-integrated: on the
        same frame the fingertip reaches a location, the desktop cursor reaches
        its corresponding location. The old cooperative controller remains
        available as a separate mode for users who prefer a relative joystick.
        """
        raw = np.asarray(motion.position, dtype=np.float64)[:2]
        filtered, hand_v, predicted = self._touch_predictor.update(raw, max(dt, 1e-3), 1.0)
        # A recovery re-anchor preserves the OS cursor for the reacquisition
        # frame, but that offset is only a temporary guard. Once the user moves
        # deliberately, return to true absolute fingertip mapping.
        if (
            self.touch_surface.offset_enabled
            and not self._touch_reanchor
        ):
            reanchor_point = self.touch_surface.state().get("reanchor_point")
            if reanchor_point is not None:
                if float(np.linalg.norm(filtered - np.asarray(reanchor_point, dtype=np.float64))) >= self.cfg.cursor.touch_surface_recovery_release_motion:
                    self.touch_surface.clear_offset()

        target = self.touch_surface.map_filtered(filtered)

        if self._touch_reanchor and self.cfg.cursor.touch_surface_recovery_reanchor:
            target = self.touch_surface.reanchor(filtered, self.position)
            self._touch_reanchor = False

        # Avoid sub-pixel cursor chatter while retaining immediate large movement.
        if self._touch_last_screen_target is not None:
            delta = target - self._touch_last_screen_target
            if float(np.linalg.norm(delta)) <= self.cfg.cursor.touch_surface_hysteresis_px:
                target = self._touch_last_screen_target.copy()
        self._touch_last_screen_target = target.copy()

        self.velocity = (target - self.position) / max(dt, 1e-3)
        # Position is the authoritative setpoint; velocity is diagnostic only.
        self.position = np.clip(target, [0.0, 0.0], [float(self.screen[0] - 1), float(self.screen[1] - 1)])
        hand_speed = float(np.linalg.norm(hand_v))
        if hand_speed >= self.cfg.motion.fast_speed:
            self.mode = CursorMode.FAST_TRAVEL
        elif self._locked_target_id:
            self.mode = CursorMode.TARGET_APPROACH
        else:
            self.mode = CursorMode.PRECISION
        self._state = CursorState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mode=self.mode,
            gain=1.0,
            filtered_hand=filtered.copy(),
            predicted_hand=predicted.copy(),
            gravity_force=np.zeros(2, dtype=np.float64),
            intent_entropy=intents.entropy,
            confidence=motion.confidence,
            target_id=None,
            target_score=0.0,
            spatial_intent=0.0,
            frozen=False,
        )
        return self._state

    # ------------------------------------------------------------------ #
    # Cooperative cursor (live desktop path)
    # ------------------------------------------------------------------ #
    def _screen_norm(self, position: np.ndarray) -> np.ndarray:
        w, h = self.screen
        return np.array(
            [float(position[0]) / max(w, 1), float(position[1]) / max(h, 1)],
            dtype=np.float64,
        )

    def normalized_position(self) -> np.ndarray:
        """Current screen cursor position in normalized coordinates."""
        return self._screen_norm(self.position)

    def normalized_velocity(self) -> np.ndarray:
        """Current cursor velocity in normalized screen units / second."""
        w, h = self.screen
        return np.array(
            [float(self.velocity[0]) / max(w, 1), float(self.velocity[1]) / max(h, 1)],
            dtype=np.float64,
        )

    @staticmethod
    def _distance_to_rect_px(point: np.ndarray, bounds: tuple[float, float, float, float], screen: tuple[int, int]) -> tuple[float, np.ndarray]:
        """Return distance to the rectangle and the closest screen-space point."""
        w, h = screen
        x0, y0, x1, y1 = bounds
        rect = np.array([x0 * w, y0 * h, x1 * w, y1 * h], dtype=np.float64)
        closest = np.array(
            [np.clip(point[0], rect[0], rect[2]), np.clip(point[1], rect[1], rect[3])],
            dtype=np.float64,
        )
        return float(np.linalg.norm(closest - point)), closest

    @staticmethod
    def _segment_distance_to_rect(point_a: np.ndarray, point_b: np.ndarray, rect_px: np.ndarray) -> float:
        """Cheap segment/rect distance used as a predictive path feature."""
        x0, y0, x1, y1 = rect_px
        a = np.asarray(point_a, dtype=np.float64)
        b = np.asarray(point_b, dtype=np.float64)
        # Inside / crossing test first.
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            q = a + (b - a) * t
            if x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
                return 0.0

        def seg_point_dist(p: np.ndarray, q0: np.ndarray, q1: np.ndarray) -> float:
            d = q1 - q0
            dd = float(np.dot(d, d))
            if dd < 1e-9:
                return float(np.linalg.norm(p - q0))
            u = float(np.clip(np.dot(p - q0, d) / dd, 0.0, 1.0))
            return float(np.linalg.norm(p - (q0 + u * d)))

        edges = (
            (np.array([x0, y0]), np.array([x1, y0])),
            (np.array([x1, y0]), np.array([x1, y1])),
            (np.array([x1, y1]), np.array([x0, y1])),
            (np.array([x0, y1]), np.array([x0, y0])),
        )
        # Sample the path at a few points as a robust fallback for crossings.
        best = 1e9
        for t in np.linspace(0.0, 1.0, 9):
            q = a + (b - a) * t
            dx = max(x0 - q[0], 0.0, q[0] - x1)
            dy = max(y0 - q[1], 0.0, q[1] - y1)
            best = min(best, float(math.hypot(dx, dy)))
        for e0, e1 in edges:
            best = min(best, seg_point_dist(e0, a, b), seg_point_dist(e1, a, b))
        return best

    def _cooperative_target(
        self,
        beliefs: list[TargetBelief],
        desired_velocity: np.ndarray,
    ) -> tuple[TargetBelief | None, float, np.ndarray | None, float, float]:
        """Build a spatial-intent field from the *predicted cursor path*.

        Targets are not selected because they are merely nearby.  A candidate
        must be in front of the pointer, close to the predicted path, and have
        adequate semantic quality.  Once engaged, hysteresis prevents a group
        of neighbouring title-bar buttons from fighting over the cursor.
        """
        ccfg = self.cfg.cursor
        if not beliefs:
            self._locked_target_id = None
            self._locked_target_since = None
            self._locked_target_score = 0.0
            return None, 0.0, None, 1e9, 0.0

        p = self.position.copy()
        v = np.asarray(desired_velocity, dtype=np.float64)
        speed = float(np.linalg.norm(v))
        # A target cannot become an active attractor while the pointer is
        # stationary. This prevents title-bar buttons/folders near the cursor
        # from changing the control mode during an otherwise neutral hover.
        if speed < ccfg.target_min_motion_px_s:
            self._locked_target_id = None
            self._locked_target_since = None
            self._locked_target_score = 0.0
            return None, 0.0, None, 1e9, 0.0
        horizon = float(np.clip(self.cfg.cursor.target_prediction_horizon, 0.05, 0.16))
        predicted = p + v * horizon
        lock_radius = float(self.cfg.cursor.target_lock_radius_px)
        exit_radius = float(self.cfg.cursor.target_lock_exit_px)
        screen = np.array(self.screen, dtype=np.float64)

        def score(b: TargetBelief) -> tuple[float, float, np.ndarray, float]:
            rect = np.asarray(b.target.bounds, dtype=np.float64) * np.array([*screen, *screen])
            dist, closest = self._distance_to_rect_px(p, b.target.bounds, self.screen)
            pred_dist, pred_closest = self._distance_to_rect_px(predicted, b.target.bounds, self.screen)
            path_dist = self._segment_distance_to_rect(p, predicted, rect)
            capture = lock_radius + 0.45 * min(float(rect[2] - rect[0]), float(rect[3] - rect[1]))
            proximity = float(np.clip(1.0 - min(dist, pred_dist, path_dist) / max(capture, 1.0), 0.0, 1.0))
            path = float(np.clip(1.0 - path_dist / max(capture, 1.0), 0.0, 1.0))

            center_px = b.target.center * screen
            to_target = center_px - p
            center_dist = float(np.linalg.norm(to_target))
            alignment = 0.5
            if speed > 30.0 and center_dist > 1e-6:
                alignment = float(np.clip((np.dot(v / speed, to_target / center_dist) + 1.0) * 0.5, 0.0, 1.0))
            predicted_inside = 1.0 if pred_dist <= 3.0 else 0.0
            approach = float(np.clip(b.target.approach_confidence, 0.0, 1.0))
            quality = float(np.clip(b.target.quality(), 0.0, 1.0))
            avoid_factor = 0.55 if str(getattr(b.target.state, "value", b.target.state)) == "AVOID" else 1.0
            size_bonus = float(np.clip(math.sqrt(max((rect[2] - rect[0]) * (rect[3] - rect[1]), 1.0)) / 180.0, 0.15, 1.0))
            spatial = (
                0.34 * path
                + 0.22 * proximity
                + 0.18 * alignment
                + 0.13 * approach
                + 0.08 * quality
                + 0.05 * size_bonus
            ) * avoid_factor
            point = pred_closest if pred_dist <= dist else closest
            # Do not attract toward an edge.  Move the magnetic point part-way
            # into the target so title-bar buttons/folders feel like a soft basin
            # rather than a wall that the pointer bounces against.
            point = point + (center_px - point) * 0.45
            return float(spatial), float(min(dist, pred_dist, path_dist)), point, alignment

        by_id = {b.target.id: b for b in beliefs}
        locked = by_id.get(self._locked_target_id) if self._locked_target_id else None
        if locked is not None:
            locked_score, locked_dist, locked_point, locked_alignment = score(locked)
            if (
                locked_dist <= exit_radius
                and locked_score >= self.cfg.cursor.target_lock_exit_score
                and locked_alignment >= 0.20
            ):
                self._locked_target_score = locked_score
                return locked, locked_score, locked_point, locked_dist, locked_alignment
            self._locked_target_id = None
            self._locked_target_since = None
            self._locked_target_score = 0.0

        # Fast travel is target-blind.  This is intentional: magnetism is a
        # precision aid, not a steering wheel.
        if speed > self.cfg.cursor.target_lock_speed_px_s:
            return None, 0.0, None, 1e9, 0.0

        ranked: list[tuple[float, float, TargetBelief, np.ndarray, float]] = []
        for b in beliefs:
            s, dist, point, alignment = score(b)
            if dist <= lock_radius * 1.15 and alignment >= 0.35:
                ranked.append((s, dist, b, point, alignment))
        if not ranked:
            return None, 0.0, None, 1e9, 0.0
        ranked.sort(key=lambda x: x[0], reverse=True)
        best_score, best_dist, best, best_point, best_alignment = ranked[0]
        if best_score < self.cfg.cursor.target_lock_enter_score:
            return None, 0.0, None, 1e9, 0.0

        self._locked_target_id = best.target.id
        if self._locked_target_since is None:
            self._locked_target_since = 0.0
        self._locked_target_score = best_score
        return best, best_score, best_point, best_dist, best_alignment

    def _cooperative_update(
        self,
        *,
        motion: MotionState,
        features: HandFeatures | None,
        intents: IntentField,
        beliefs: list[TargetBelief],
        fsm: StateMachine,
        dt: float,
        timestamp: float,
        gravity_force: np.ndarray | None,
    ) -> CursorState:
        """Cooperative pointer for real Windows interaction.

        The important distinction from a demo gesture mouse is that the camera
        never dictates an absolute desktop coordinate.  The index tip is a
        *continuous displacement sensor* and the cursor is a separate stateful
        object.  We integrate real tip deltas, learn the short-term camera noise
        floor, predict the next tip position, and then let the Windows UIA target
        field gently bend the resulting trajectory when there is evidence of a
        target approach.

        This is intentionally not a spring around the finger.  When the user
        stops moving the finger, the cursor stops; there is no hidden set-point
        chasing and therefore no target-induced swimming.
        """
        ccfg = self.cfg.cursor
        raw = np.asarray(motion.position, dtype=np.float64)[:2]
        raw_dt = max(float(dt), 1e-3)

        # The index tip is the actual actuator sensor. Prediction is deliberately
        # NOT allowed to move the cursor: a predictor can be a few pixels wrong
        # even when the 21-point skeleton is perfect. Prediction belongs to the
        # *intent* layer (target acquisition), not to the hand-to-cursor signal.
        raw_delta = np.zeros(2, dtype=np.float64)
        if self._coop_prev_raw is not None:
            raw_delta = raw - self._coop_prev_raw
            # A very large jump is more likely to be a frame drop / re-acquisition
            # than a genuine finger movement. Treating it as a delta creates the
            # exact screen-spanning cursor jumps seen in the earlier logs.
            if float(np.linalg.norm(raw_delta)) > 0.35:
                raw_delta[:] = 0.0
        self._coop_prev_raw = raw.copy()

        if self._coop_rearm or self._coop_prev_control is None:
            self._coop_predictor.reset(raw)
            filtered, _, predicted_hand = self._coop_predictor.update(raw, max(dt, 1e-3), motion.confidence)
            self._coop_prev_control = raw.copy()
            self._coop_prev_raw = raw.copy()
            self._coop_raw_deltas.clear()
            self._coop_raw_deltas.append(np.zeros(2, dtype=np.float64))
            self._coop_delta_ema[:] = 0.0
            self._coop_noise_speed = max(0.0003, float(ccfg.cooperative_motion_deadband))
            self._coop_last_hand = filtered.copy()
            self._hand_velocity[:] = 0.0
            self._coop_rearm = False
        else:
            filtered, hand_v, predicted_hand = self._coop_predictor.update(raw, max(dt, 1e-3), motion.confidence)
            self._hand_velocity[:] = hand_v
            self._coop_raw_deltas.append(raw_delta.copy())

        # Median-of-3 is an outlier suppressor, not a positional smoother: it
        # keeps the measured fingertip direction while rejecting one bad frame.
        # That is important here because the user explicitly sees the endpoint
        # moving correctly in the skeleton overlay.
        delta = np.median(np.stack(tuple(self._coop_raw_deltas), axis=0), axis=0)
        self._coop_prev_control = raw.copy()
        self._coop_last_hand = filtered.copy()

        dt_safe = max(float(dt), 1e-3)
        raw_speed = float(np.linalg.norm(raw_delta)) / raw_dt
        measured_speed = float(np.linalg.norm(delta)) / dt_safe

        # Learn the camera's local noise floor only while the finger is nearly
        # still.  This is the key difference from a fixed deadband: a quiet camera
        # keeps millimetre movements usable, a noisy camera raises the threshold
        # just enough to stop cursor tremor.
        # Estimate the noise floor from the RAW fingertip displacement.  The
        # filtered/predicted signal is intentionally allowed to ramp during the
        # first frames of a real movement; using it here would mistake that ramp
        # for camera noise and raise the deadband until the pointer stopped moving.
        if raw_speed < ccfg.cooperative_noise_sample_speed:
            self._coop_noise_speed = (
                ccfg.cooperative_noise_ema * self._coop_noise_speed
                + (1.0 - ccfg.cooperative_noise_ema) * raw_speed
            )
        dead_speed = max(
            ccfg.cooperative_motion_deadband,
            self._coop_noise_speed * ccfg.cooperative_noise_multiplier,
        )

        # Soft deadband: only suppress low-speed motion when it also has poor
        # directional agreement. A deliberate tiny movement from the fingertip
        # must remain usable even if it is slower than the learned camera floor.
        recent = [d for d in tuple(self._coop_raw_deltas)[-3:] if float(np.linalg.norm(d)) > 1e-7]
        direction_consistency = 0.0
        if recent:
            units = [d / max(float(np.linalg.norm(d)), 1e-9) for d in recent]
            direction_consistency = float(np.linalg.norm(np.sum(units, axis=0)) / len(units))
        if measured_speed <= dead_speed and direction_consistency < 0.70:
            delta[:] = 0.0
            clean_speed = 0.0
        elif measured_speed <= dead_speed:
            clean_speed = measured_speed
        else:
            scale = float(np.clip((measured_speed - 0.45 * dead_speed) / max(measured_speed, 1e-9), 0.0, 1.0))
            delta *= scale
            clean_speed = measured_speed - 0.45 * dead_speed

        # Smooth the *increment*, not the cursor position.  This preserves the
        # relative-motion semantics while suppressing one-frame detector noise.
        u = float(np.clip(clean_speed / max(ccfg.cooperative_fast_speed, 1e-6), 0.0, 1.0))
        delta_alpha = ccfg.cooperative_delta_alpha_slow + (
            ccfg.cooperative_delta_alpha_fast - ccfg.cooperative_delta_alpha_slow
        ) * u
        self._coop_delta_ema = delta_alpha * delta + (1.0 - delta_alpha) * self._coop_delta_ema

        # Hand speed itself controls reach.  Slow movement is deliberately close
        # to 1:1 so that the fingertip feels connected; fast motion earns extra
        # gain so the user can cross a monitor without huge arm travel.
        gain_u = float(smoothstep(ccfg.cooperative_slow_speed, ccfg.cooperative_fast_speed, max(clean_speed, 0.0)))
        hand_gain = ccfg.cooperative_gain_slow + (ccfg.cooperative_gain_fast - ccfg.cooperative_gain_slow) * gain_u

        desired_move = self._coop_delta_ema * np.array(self.screen, dtype=np.float64) * hand_gain * ccfg.sensitivity
        move_mag = float(np.linalg.norm(desired_move))
        max_step = float(ccfg.cooperative_max_delta_px_per_frame)
        if move_mag > max_step:
            desired_move *= max_step / max(move_mag, 1e-9)

        desired_velocity = desired_move / dt_safe
        target, target_score, target_point, target_dist, alignment = self._cooperative_target(beliefs, desired_velocity)

        spatial_intent = float(np.clip(target_score * (0.35 + 0.65 * alignment), 0.0, 1.0))
        mode = CursorMode.FAST_TRAVEL if clean_speed >= ccfg.cooperative_fast_speed else CursorMode.PRECISION

        # Precision is a *gain reduction*, not a positional spring.  It means the
        # user can make very small motions around a button without overshooting it.
        if target is not None and target_dist < ccfg.target_precision_radius_px * 1.8:
            near = float(np.clip(1.0 - target_dist / max(ccfg.target_precision_radius_px * 1.8, 1.0), 0.0, 1.0))
            precision_factor = 1.0 - (1.0 - ccfg.cooperative_precision_gain) * near * spatial_intent
            desired_move *= precision_factor

        # Predictive magnetic field.  The target can only bend an already-present
        # user trajectory.  It cannot start moving a stationary cursor.  Attraction
        # gets stronger during a slow approach and weaker during fast travel.
        if target is not None and target_point is not None and target_score > 0.0 and move_mag > 0.0:
            to_target = np.asarray(target_point, dtype=np.float64) - self.position
            distance = float(np.linalg.norm(to_target))
            if distance > 2.0:
                move_dir = desired_move / max(move_mag, 1e-9)
                # Spatial intent bends the trajectory; it does not pull the
                # pointer forward.  This is the key "cooperation" behaviour:
                # user speed/direction remain the dominant term, the UI only
                # corrects lateral error toward the predicted target corridor.
                lateral = to_target - move_dir * float(np.dot(to_target, move_dir))
                lateral_mag = float(np.linalg.norm(lateral))
                if lateral_mag > 1e-6:
                    lateral_dir = lateral / lateral_mag
                    current_speed = float(np.linalg.norm(self.velocity))
                    approach_gate = float(np.clip(1.0 - current_speed / max(ccfg.target_lock_speed_px_s, 1.0), 0.0, 1.0))
                    slow_gate = float(np.clip(1.0 - clean_speed / max(ccfg.cooperative_fast_speed, 1e-6), 0.0, 1.0))
                    gate = float(np.clip(0.45 * approach_gate + 0.55 * slow_gate, 0.0, 1.0))
                    pull = float(np.clip(ccfg.target_attraction_strength * target_score * alignment * gate, 0.0, ccfg.cooperative_max_target_pull))
                    inside = bool(target.target.contains(self._screen_norm(self.position)))
                    if inside:
                        pull = 0.0
                    pull_px = min(ccfg.cooperative_target_pull_max_px_per_frame, lateral_mag * ccfg.cooperative_target_pull_fraction)
                    desired_move += lateral_dir * pull_px * pull
                    if distance < ccfg.target_precision_radius_px:
                        mode = CursorMode.TARGET_APPROACH

        # No legacy gravity force here.  The cooperative field above is the only
        # live magnetic field, which prevents two independent attractors from
        # fighting over the same cursor.
        del gravity_force

        # Apply the user's desired movement directly with a very small dynamic
        # actuator filter.  The cursor never chases an arbitrary position, so when
        # delta becomes zero it naturally settles instead of oscillating.
        desired_velocity = desired_move / dt_safe
        actuator_alpha = float(
            ccfg.cooperative_actuator_alpha_slow
            + (ccfg.cooperative_actuator_alpha_fast - ccfg.cooperative_actuator_alpha_slow) * gain_u
        )
        self.velocity = actuator_alpha * desired_velocity + (1.0 - actuator_alpha) * self.velocity

        vmax = float(ccfg.cooperative_max_speed_px_s)
        vmag = float(np.linalg.norm(self.velocity))
        if vmag > vmax:
            self.velocity *= vmax / max(vmag, 1e-9)

        if np.linalg.norm(desired_move) <= 0.001:
            self.velocity *= math.exp(-ccfg.cooperative_idle_damping * dt_safe)

        self.position += self.velocity * dt_safe

        # Hard desktop boundary.  Unlike the old spring controller, there is no
        # out-of-screen position to recover from, and edge contact kills only the
        # outward velocity component.
        w, h = self.screen
        before = self.position.copy()
        self.position = np.clip(self.position, [0.0, 0.0], [max(0.0, w - 1.0), max(0.0, h - 1.0)])
        if not np.allclose(before, self.position):
            if self.position[0] <= 0.0 and self.velocity[0] < 0.0:
                self.velocity[0] = 0.0
            elif self.position[0] >= w - 1 and self.velocity[0] > 0.0:
                self.velocity[0] = 0.0
            if self.position[1] <= 0.0 and self.velocity[1] < 0.0:
                self.velocity[1] = 0.0
            elif self.position[1] >= h - 1 and self.velocity[1] > 0.0:
                self.velocity[1] = 0.0

        self._state = CursorState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            mode=mode,
            gain=float(hand_gain),
            filtered_hand=filtered,
            predicted_hand=predicted_hand,
            gravity_force=np.zeros(2, dtype=np.float64),
            intent_entropy=intents.entropy,
            confidence=motion.confidence,
            target_id=target.target.id if target is not None else None,
            target_score=float(target_score),
            spatial_intent=spatial_intent,
            frozen=False,
        )
        return self._state

    # ------------------------------------------------------------------ #
    def _spring_step(
        self,
        desired: np.ndarray,
        speed: float,
        gain: float,
        dt: float,
        feedforward_velocity: np.ndarray | None = None,
    ) -> None:
        """Critically-damped second-order tracking controller.

        ``omega`` and ``zeta`` are scheduled by speed: heavily over-damped at
        rest (kills tremor with zero lag penalty because there is no motion to
        lag behind) and near-critical when moving (snappy, no overshoot ring).
        This is why a spring beats an EMA: one primitive gives you both regimes.

        ``feedforward_velocity`` enters the damping term, so the controller
        converges on the *setpoint velocity* instead of on the setpoint
        position.  Tracking a moving target is a different problem from
        arriving at a static one: measured on a 0.5 Hz hand sweep this is worth
        76 ms of phase lag against 342 ms with no compensation and 230 ms for a
        positional lead of the same nominal horizon.

        Integration sub-steps keep the scheme stable.  Both terms need bounding:
        the stiffness term requires ``omega * h < 2`` and the damping term
        ``2 * zeta * omega * h < 2``, so the sub-step is limited by
        ``zeta * omega * h``.  Bounding only ``omega * h`` leaves a latent
        blow-up: raising ``spring_damping_slow`` alone makes the cursor diverge
        instead of settling.
        """
        c = self.cfg.cursor
        u = clamp01((speed - self.cfg.motion.pause_speed) / max(self.cfg.motion.fast_speed - self.cfg.motion.pause_speed, 1e-6))
        omega = c.spring_omega_slow + (c.spring_omega_fast - c.spring_omega_slow) * u
        zeta = c.spring_damping_slow + (c.spring_damping_fast - c.spring_damping_slow) * u
        omega *= float(np.clip(gain, 0.2, 1.6))

        max_dt = c.spring_max_omega_dt / (omega * max(zeta, 1e-6))
        steps = max(1, min(int(math.ceil(dt / max_dt)), 16)) if max_dt > 0.0 else 1
        h = dt / steps

        ff = np.zeros(2) if feedforward_velocity is None else np.asarray(feedforward_velocity, dtype=np.float64)[:2]

        for _ in range(steps):
            accel = omega * omega * (desired - self.position) + 2.0 * zeta * omega * (ff - self.velocity)
            self.velocity = self.velocity + accel * h
            self.position = self.position + self.velocity * h

    def _clamp_step(self, dt: float) -> None:
        limit = self.cfg.cursor.max_step_px
        step = self.velocity * dt
        mag = float(np.linalg.norm(step))
        if mag > limit and mag > 0:
            self.velocity = self.velocity * (limit / mag)

    def _relative_target(self, predicted: np.ndarray, speed: float, timestamp: float) -> np.ndarray:
        """Joystick mode: the hand steers a velocity, the cursor re-anchors at rest."""
        if self._anchor_hand is None:
            self._anchor_hand = predicted.copy()
            self._anchor_cursor = self.position.copy()
        # In relative desktop mode the anchor is deliberately stable.  Do not
        # recenter it merely because the hand is moving slowly: that makes a
        # careful, slow hand movement disappear (the anchor follows the hand).
        # The user can re-center safely by re-activating with INDEX_UP.
        self._pause_since = None
        delta = predicted - self._anchor_hand  # type: ignore[operator]
        # Convert hand displacement to screen displacement.  Keep a small
        # dead-zone around the anchor, then use a mild nonlinear curve.  The
        # previous x2.2 mapping amplified ordinary MediaPipe jitter into visible
        # cursor swimming.
        dz = float(self.cfg.cursor.dead_zone)
        delta = np.sign(delta) * np.maximum(np.abs(delta) - dz, 0.0)
        delta = np.array([
            self._sensitivity_curve(float(delta[0]) * 2.2) / 2.2,
            self._sensitivity_curve(float(delta[1]) * 2.2) / 2.2,
        ])
        px = delta * np.array(self.screen, dtype=np.float64) * 1.15
        base = self._anchor_cursor if self._anchor_cursor is not None else self.position
        return base + px

    def _mode_for(
        self,
        fsm: StateMachine,
        speed: float,
        target_id: str | None,
        intents: IntentField,
        beliefs: list[TargetBelief],
    ) -> CursorMode:
        if fsm.dragging:
            return CursorMode.DRAG
        if target_id is not None and max(intents.get(Intent.SELECT), intents.get(Intent.DRAG)) > 0.35:
            return CursorMode.TARGET_APPROACH
        if speed >= self.cfg.motion.fast_speed:
            return CursorMode.FAST_TRAVEL
        if speed <= self.cfg.motion.pause_speed * 1.5:
            return CursorMode.PRECISION
        return CursorMode.TARGET_APPROACH if beliefs else CursorMode.PRECISION

    # ------------------------------------------------------------------ #
    def jump_to(self, position: np.ndarray) -> None:
        """Used by the dispatcher for the first move after activation."""
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity = np.zeros(2)

    def anchor_to(self, normalized: np.ndarray, cursor_position: np.ndarray | None = None) -> None:
        """Place the cursor at the mapped hand position and clear the filters.

        Called on a **fresh activation**, and deliberately not on recovery from a
        tracking loss: during recovery the cursor is already on screen and
        teleporting it is exactly what the safety design forbids, but on the
        first activation there is no cursor yet — there is nothing to teleport.

        Without this the controller starts at ``(0, 0)`` and an EMA with an
        effective alpha of ~0.19 needs about 15 frames to crawl to the hand.
        That is half a second of the cursor sliding in from the top-left corner,
        and it is what the benchmark was measuring as "cursor lag 233-467 ms" —
        a transient, not a tracking property.
        """
        point = np.asarray(normalized, dtype=np.float64)
        self.velocity = np.zeros(2)
        self.ema.reset()
        self.euro.reset()
        self.speed_filter.reset()
        self._last_hand = point.copy()
        self._prev_desired = None
        self._coop_raw_deltas.clear()

        # Live desktop control is deliberately relative: activation must not
        # teleport the real Windows cursor to the finger.  The hand position
        # becomes the origin and the current OS cursor position becomes the
        # destination origin.  This makes small MediaPipe jitter local instead
        # of turning it into a full-screen cursor jump.
        if self.cfg.cursor.controller == "touch_surface" and not self.cfg.cursor.relative_mode:
            self._touch_predictor.reset(point)
            self.touch_surface.clear_offset()
            self._touch_last_screen_target = self.touch_surface.map_point(point)
            self.position = self._touch_last_screen_target.copy()
            self._touch_reanchor = False
            self._anchor_hand = point.copy()
            self._anchor_cursor = self.position.copy()
            return

        if self.cfg.cursor.relative_mode:
            self._anchor_hand = point.copy()
            if cursor_position is None:
                cursor_position = self.position
            self._anchor_cursor = np.asarray(cursor_position, dtype=np.float64).copy()
            self.position = self._anchor_cursor.copy()
            return

        curve = np.array(
            [
                self._sensitivity_curve(2.0 * (point[0] - 0.5)) * 0.5 + 0.5,
                self._sensitivity_curve(2.0 * (point[1] - 0.5)) * 0.5 + 0.5,
            ]
        )
        self.position = self._map_to_screen(curve)
        self._anchor_hand = None
        self._anchor_cursor = None
