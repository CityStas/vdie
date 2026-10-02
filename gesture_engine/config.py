"""Typed configuration.

Everything that a researcher may want to tune lives here and is loaded from
``config/config.yaml``.  Nothing downstream hard-codes thresholds: modules
receive a config object.

The loader does a *deep merge* of the YAML file onto the dataclass defaults, so
a config file may specify only the keys it wants to override.  That keeps
"camera configuration" and "algorithm version" cleanly separable — a benchmark
can hold the algorithm config fixed and swap only ``capture.profile``.
"""

from __future__ import annotations

import dataclasses
import types as _pytypes
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar, get_args, get_origin, get_type_hints

T = TypeVar("T")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


@dataclass
class CaptureConfig:
    device: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    profile: str = "g85_12_60"
    backend: str = "msmf"  # msmf | dshow | any
    fourcc: str = "MJPG"
    buffer_size: int = 1
    flip_horizontal: bool = True
    max_read_failures: int = 30


@dataclass
class CameraProfile:
    lens: str = "unknown"
    distortion_correction: str = "false"  # false | optional | required
    sensor_width_mm: float = 0.0
    fisheye_fov_deg: float = 0.0
    notes: str = ""


@dataclass
class CalibrationConfig:
    enabled: bool = False
    path: str = "config/calibration/g85_fisheye.npz"
    balance: float = 0.0  # cv2.fisheye.estimateNewCameraMatrixForUndistortRectify balance
    fov_scale: float = 1.0


@dataclass
class TrackingConfig:
    backend: str = "mediapipe"  # mediapipe | synthetic | replay | null
    model_path: str = "models/hand_landmarker.task"
    model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )
    auto_download: bool = True
    running_mode: str = "video"  # video | image
    max_hands: int = 2
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    min_hand_confidence: float = 0.7
    use_gpu: bool = False
    smooth_landmarks: bool = False
    #: Frames of missing detection tolerated before the hand is considered lost.
    loss_grace_frames: int = 3


@dataclass
class FeaturesConfig:
    #: Joint angle above which a finger counts as extended (radians).
    extension_angle: float = 1.75  # ~100 deg
    #: Thumb is anatomically different, so it gets its own threshold.
    thumb_extension_angle: float = 1.40  # ~80 deg
    #: Extension must hold for MCP->PIP *and* PIP->DIP.
    require_both_joints: bool = True
    #: Pinch distances are expressed in units of palm scale (wrist->middle MCP).
    pinch_span: float = 1.0
    #: Normalize landmark cloud by rotation into the palm frame.
    rotation_invariant: bool = False
    #: Depth proxy: palm_scale relative to this value maps to 1.0.
    reference_palm_scale: float = 0.22
    #: Interior DIP angle (rad) above which the distal joint counts as extended.
    #:
    #: This must be a *separate* threshold from ``extension_angle``: the DIP has a
    #: different anatomical range (180 deg straight -> ~110 deg fully folded) than
    #: the PIP, so reusing a fraction of the PIP threshold made a folded finger's
    #: DIP read as "extended" and floored the soft extension confidence at 0.35.
    dip_extension_angle: float = 2.36  # rad, ~135 deg
    #: Continuous pinch_strength: 1.0 when distance <= closed, 0.0 when >= open.
    pinch_strength_closed: float = 0.35
    pinch_strength_open: float = 1.20


@dataclass
class MotionConfig:
    history_size: int = 90  # ~3 s at 30 FPS
    #: Ignore dt outside this range (frame drops / duplicate frames).
    dt_min: float = 0.002
    dt_max: float = 0.25
    #: Low-pass on the raw control point before differentiation (0 = off).
    position_smoothing: float = 0.0
    #: Speed below which the control point is considered stationary (normalized/s).
    pause_speed: float = 0.12
    #: Speed above which movement is "fast" (used for adaptive gain).
    fast_speed: float = 1.60
    #: Window (seconds) used for amplitude / path-efficiency statistics.
    stats_window: float = 0.75
    #: Velocity filter: none | ema | savgol
    velocity_filter: str = "ema"
    velocity_alpha: float = 0.45
    acceleration_alpha: float = 0.30
    #: Which landmark drives the control point: index_tip | palm | index_mcp
    control_point: str = "index_tip"


@dataclass
class GestureConfig:
    # -- pinch (hysteresis: distinct down/release thresholds) --
    pinch_down: float = 0.42
    pinch_release: float = 0.62
    middle_pinch_down: float = 0.42
    middle_pinch_release: float = 0.62
    # -- temporal confirmation --
    vote_window: int = 10
    vote_required: int = 7
    #: Dynamic gestures (swipe/scroll) are ~12-frame patterns at 30 FPS and their
    #: score only saturates near the end of the stroke, so the 7-of-10 rule is
    #: arithmetically unreachable for them.  They carry their own temporal
    #: evidence (the trajectory itself), so a short tail window is enough and
    #: avoids double-counting time.
    vote_window_dynamic: int = 4
    vote_required_dynamic: int = 3
    min_gesture_duration: float = 0.10
    gesture_cooldown: float = 0.18
    double_click_window: float = 0.35
    # -- dynamic gestures --
    swipe_min_distance: float = 0.22
    swipe_max_duration: float = 0.60
    swipe_min_speed: float = 0.55
    swipe_axis_ratio: float = 1.6
    #: Scroll is a *continuous* gesture — it has no natural end, so applying the
    #: swipe duration gate to it only delayed detection until the user stopped
    #: moving (measured: the SCROLL_* score saturated 0.4 s *after* the stroke).
    #: Its real discriminators are the axis, the accumulated stroke and a speed
    #: *ceiling* — cursor travel to a target is the fast vertical movement that
    #: must not be read as scrolling.
    scroll_min_distance: float = 0.10
    scroll_max_speed: float = 1.10
    # -- confidence shaping --
    stability_window: float = 0.30
    min_confidence: float = 0.55


@dataclass
class StateConfig:
    activation_stable_ms: float = 250.0
    activation_velocity: float = 0.35
    activation_max_tilt_deg: float = 25.0
    activation_confidence: float = 0.85
    cursor_idle_timeout_ms: float = 8000.0
    pause_release_ms: float = 350.0
    cooldown_ms: float = 250.0
    drag_max_hold_ms: float = 15000.0
    #: A drag starts only when the hand has *actually travelled* since the pinch
    #: closed.  Using instantaneous speed instead is a classic bug: closing a
    #: pinch moves the fingertip, so every click would be misread as a drag.
    drag_min_displacement: float = 0.035
    drag_min_duration: float = 0.14
    #: Time allowed between pinch close and drag start before it is treated as a
    #: click instead.
    pinch_click_max_ms: float = 220.0
    #: Fast one-hand click edge. Geometry is checked directly, so the user does not
    #: have to wait for the full intent-field/commit pipeline.
    fast_pinch_click: bool = True


@dataclass
class EventConfig:
    accel_threshold: float = 1.2  # normalized/s^2
    decel_threshold: float = -1.2
    direction_change_deg: float = 35.0
    pause_speed: float = 0.12
    pause_min_duration: float = 0.12
    thrust_speed: float = 1.1
    thrust_max_duration: float = 0.30
    target_approach_radius: float = 0.10
    min_event_interval: float = 0.08


@dataclass
class IntentConfig:
    temperature: float = 1.0
    #: Exponential decay (per second) applied to stale evidence logits.  With a
    #: constant evidence rate ``e`` the steady-state logit is ``e / decay``, so
    #: this is the field's memory length (1/decay seconds).
    decay_per_second: float = 2.0
    #: Global multiplier on all *rate* evidence.  Kept as one knob so that the
    #: relative weights below keep their meaning when the field is retuned.
    evidence_gain: float = 3.5
    #: Evidence source weights: gesture, motion, temporal, target, context, user.
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "gesture": 1.0,
            "motion": 0.85,
            "temporal": 0.7,
            "target": 0.9,
            "context": 0.5,
            "user": 0.3,
            "commit": 1.2,
            "future": 0.5,
        }
    )
    hysteresis: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "MOVE_CURSOR": {"enter": 0.45, "stay": 0.30, "exit": 0.20},
            "SELECT": {"enter": 0.72, "stay": 0.50, "exit": 0.32},
            "DRAG": {"enter": 0.70, "stay": 0.48, "exit": 0.30},
            "SCROLL": {"enter": 0.60, "stay": 0.40, "exit": 0.25},
            "RIGHT_CLICK": {"enter": 0.78, "stay": 0.55, "exit": 0.35},
            "WINDOW_SWITCH": {"enter": 0.72, "stay": 0.50, "exit": 0.32},
            "WINDOW_CONTROL": {"enter": 0.72, "stay": 0.50, "exit": 0.32},
            "PAUSE": {"enter": 0.60, "stay": 0.40, "exit": 0.25},
            "CANCEL": {"enter": 0.65, "stay": 0.45, "exit": 0.28},
            "DOUBLE_CLICK": {"enter": 0.78, "stay": 0.55, "exit": 0.35},
            "CUSTOM": {"enter": 0.75, "stay": 0.52, "exit": 0.34},
            "UNKNOWN": {"enter": 0.0, "stay": 0.0, "exit": 0.0},
        }
    )
    #: Counterfactual futures (section 60 of the master prompt).
    futures_enabled: bool = True
    futures_horizons: tuple[float, ...] = (0.12, 0.24, 0.40)
    futures_samples: int = 5
    #: Static logit offsets applied every frame — the *base rate* of each intent.
    #: MOVE_CURSOR is positive because, once the cursor is active, "the user is
    #: steering" is genuinely the most likely state; without a base rate the
    #: continuously-present INDEX_UP gesture evidence saturates the field at
    #: p(MOVE_CURSOR)=1 and no discrete action can ever win.
    priors: dict[str, float] = field(
        default_factory=lambda: {
            "MOVE_CURSOR": 0.50,
            "SELECT": -0.20,
            "DRAG": -0.40,
            "SCROLL": -0.90,
            "PAUSE": -0.80,
            "CANCEL": -1.00,
            "DOUBLE_CLICK": -1.60,
            "RIGHT_CLICK": -1.60,
            "WINDOW_SWITCH": -1.60,
            "WINDOW_CONTROL": -1.60,
            "CUSTOM": -1.60,
            "UNKNOWN": -2.00,
        }
    )


@dataclass
class TargetsConfig:
    provider: str = "uia"  # uia | cv | static | none
    refresh_interval_ms: float = 350.0
    min_quality: float = 0.25
    #: Soft gravity field, expressed as normalized screen acceleration / s^2.
    gravity_k: float = 0.22
    gravity_radius: float = 0.045
    gravity_max_force: float = 0.22
    #: Gravity is suppressed above this speed (fast travel must stay free).
    gravity_suppress_speed: float = 1.20
    #: Approach/avoid detection.
    approach_min_closing_speed: float = 0.10
    avoid_decay: float = 0.85
    avoid_penalty: float = 0.55
    #: Never let the field snap harder than this fraction of remaining distance.
    max_pull_fraction: float = 0.25


@dataclass
class CommitConfig:
    enabled: bool = True
    #: Kinematic commit: deceleration followed by a micro-pause near a target.
    decel_ratio: float = 0.55  # speed_now / speed_peak must drop below this
    micro_pause_speed: float = 0.22
    micro_pause_min: float = 0.045
    micro_pause_max: float = 0.45
    reversal_angle_deg: float = 120.0
    reversal_min_speed: float = 0.35
    thrust_min_speed: float = 1.05
    thrust_max_duration: float = 0.28
    #: A flick moves the *hand*.  Closing a pinch also moves the control point
    #: (the index tip) fast and from rest, so speed and duration alone are not
    #: enough — the burst must also cover real distance.  Calibrated against the
    #: hand model: an INDEX_UP -> PINCH transition articulates the index tip by
    #: ~0.07 normalized units, a deliberate hand flick covers 0.2+.
    thrust_min_travel: float = 0.10
    #: Total commit score needed to emit COMMIT.
    threshold: float = 0.55
    cooldown: float = 0.35


@dataclass
class PolicyConfig:
    default_commit_threshold: float = 0.62
    default_stability: float = 0.55
    default_cooldown: float = 0.20
    #: Per-intent overrides: commit threshold, required stability, cooldown, risk.
    per_intent: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "MOVE_CURSOR": {"risk": 0.0, "commit": 0.0, "stability": 0.0, "cooldown": 0.0},
            "SELECT": {"risk": 0.35, "commit": 0.62, "stability": 0.55, "cooldown": 0.20},
            "DOUBLE_CLICK": {"risk": 0.45, "commit": 0.68, "stability": 0.60, "cooldown": 0.30},
            "RIGHT_CLICK": {"risk": 0.55, "commit": 0.72, "stability": 0.62, "cooldown": 0.35},
            "DRAG": {"risk": 0.50, "commit": 0.60, "stability": 0.50, "cooldown": 0.25},
            "SCROLL": {"risk": 0.10, "commit": 0.45, "stability": 0.35, "cooldown": 0.05},
            "WINDOW_SWITCH": {"risk": 0.40, "commit": 0.66, "stability": 0.58, "cooldown": 0.40},
            "WINDOW_CONTROL": {"risk": 0.45, "commit": 0.66, "stability": 0.58, "cooldown": 0.40},
            "PAUSE": {"risk": 0.05, "commit": 0.50, "stability": 0.40, "cooldown": 0.30},
            "CANCEL": {"risk": 0.0, "commit": 0.35, "stability": 0.25, "cooldown": 0.50},
            "CUSTOM": {"risk": 0.60, "commit": 0.75, "stability": 0.65, "cooldown": 0.50},
        }
    )


@dataclass
class CursorConfig:
    #: touch_surface | cooperative | gain | velocity_curve | spring
    # The shipped live profile uses touch_surface: absolute index-tip -> desktop.
    controller: str = "touch_surface"
    sensitivity: float = 1.0
    gamma: float = 1.35
    #: Adaptive EMA alphas (used by gain/velocity_curve controllers).
    alpha_slow: float = 0.12
    alpha_medium: float = 0.30
    alpha_fast: float = 0.65
    #: One-Euro pre-filter.  ``beta`` is what makes the filter *adaptive*: the
    #: cutoff is ``min_cutoff + beta * speed``.  The speed here is in
    #: **normalized units per second** (a fast hand move is ~2-3), so beta has to
    #: be on the order of 1 for the cutoff to open at all — at beta=0.05 the
    #: cutoff barely moves off ``min_cutoff`` and a 2 Hz movement is attenuated
    #: to a third of its amplitude.
    euro_min_cutoff: float = 3.0
    euro_beta: float = 1.2
    euro_d_cutoff: float = 1.0
    #: Predictive lead time (seconds) applied to the filtered point.
    #:
    #: Off by default.  Measured on a 0.5 Hz hand sweep, the lead leaves the
    #: cursor tracking 60 % of the hand's displacement with a 230 ms phase lag,
    #: against 100 % and 76 ms for :attr:`feedforward` — the same latency hidden
    #: two different ways, and only one of them works.  Kept so the
    #: "prediction on/off" experiment in the plan can still be run.
    #:
    #: Note: the lead is *not* an overshoot generator in general.  An earlier
    #: note here claimed it made the cursor overshoot every fast movement by
    #: ``v * horizon`` px; that could not be reproduced.  The 190 px seen in the
    #: probe was the ``active_region`` clamp turning the lead into a fixed
    #: positional error at the region edge, not a property of the lead.
    prediction_horizon: float = 0.0
    prediction_accel_weight: float = 0.35
    #: Velocity feed-forward: inject the setpoint velocity into the spring's
    #: damping term so the controller *tracks* a moving target rather than
    #: chasing it.  This is where the latency actually gets hidden: 76 ms of
    #: phase lag at 0.5 Hz against 230 ms for the positional lead and 342 ms for
    #: no compensation at all.
    feedforward: bool = True
    #: 0.8, not 1.0.  Full feed-forward makes the cursor *lead* the hand by
    #: 20-30 ms and overshoot a 0.5 Hz movement by 13 %; 0.8 gives a flat
    #: response (1.00 / 0.99 / 0.83 at 0.5 / 1 / 2 Hz) with a few ms of lead,
    #: which is exactly what the downstream camera-and-inference latency needs.
    feedforward_gain: float = 0.8
    #: Clamp on the feed-forward velocity (px/s).  A discontinuous setpoint jump
    #: (gravity waking up, re-anchor after recovery) must not become a velocity
    #: spike.
    feedforward_max_px_s: float = 6000.0
    #: Critically-damped spring controller.
    spring_omega_slow: float = 9.0  # rad/s at rest -> heavily damped
    spring_omega_fast: float = 60.0  # rad/s at speed -> snappy
    spring_damping_slow: float = 1.35  # over-damped at rest
    spring_damping_fast: float = 0.95  # slightly under-damped when moving fast
    #: Upper bound on ``omega * h`` per integration sub-step.  Semi-implicit
    #: Euler (the scheme in :meth:`CursorController._spring_step`) is unstable
    #: for ``omega * dt >= 2``: at 30 fps that is ``omega_fast >= 60``, which is
    #: why raising the stiffness alone blows the controller up instead of making
    #: it faster.  Sub-stepping removes the ceiling.
    spring_max_omega_dt: float = 0.5
    max_step_px: float = 900.0  # per-frame clamp, kills jump artefacts
    #: Dead zone around the anchor, in normalized units.
    dead_zone: float = 0.0015
    #: Relative (joystick) mode re-anchors the hand to the cursor when idle.
    relative_mode: bool = False
    anchor_recenter_ms: float = 600.0
    #: Intent-adaptive gain: uncertainty makes the cursor hesitate.
    #:
    #: **Off by default, and that is a measurement, not an omission.**  Against
    #: an identical variant with commit enabled, turning it on costs 30 ms of
    #: cursor lag (84.4 -> 54.4 ms) and changes *nothing else*: intent recall,
    #: commit precision and the corrective-movement count are bit-identical.
    #: Any mechanism that makes the cursor respond less to a moving input costs
    #: lag on that input — there is no free hesitation.  The hesitation that
    #: works lives in the commit threshold, where the intent field already
    #: declines to act, and it is measured there (commit precision 1.00).
    intent_adaptive: bool = False
    intent_entropy_gain: float = 0.55
    precision_near_target: bool = True
    precision_gain: float = 0.55
    #: Uncertainty as a control signal (section 56).
    uncertainty_gain_floor: float = 0.15
    uncertainty_freeze_below: float = 0.35
    #: Recovery mode (section 57).
    recovery_enabled: bool = True
    recovery_prediction_horizon: float = 0.35
    recovery_max_jump: float = 0.12
    # Live cooperative cursor.  Hand motion becomes a velocity command so the
    # cursor is not trapped inside a finite camera displacement rectangle.
    cooperative_velocity_alpha: float = 0.32
    cooperative_deadband: float = 0.022
    cooperative_slow_speed: float = 0.055
    cooperative_fast_speed: float = 0.80
    cooperative_gain_slow: float = 1.15
    cooperative_gain_fast: float = 2.10
    cooperative_max_speed_px_s: float = 2200.0
    cooperative_accel_px_s2: float = 10000.0
    cooperative_brake_px_s2: float = 14000.0
    cooperative_idle_damping: float = 11.0
    cooperative_predict_alpha_slow: float = 0.62
    cooperative_predict_alpha_fast: float = 0.84
    cooperative_predict_beta: float = 0.085
    cooperative_predict_horizon: float = 0.10
    cooperative_predict_blend: float = 0.26
    cooperative_max_innovation: float = 0.14
    cooperative_stationary_speed: float = 0.020
    cooperative_motion_deadband: float = 0.00025
    cooperative_noise_sample_speed: float = 0.008
    cooperative_noise_ema: float = 0.94
    cooperative_noise_multiplier: float = 2.2
    cooperative_delta_alpha_slow: float = 0.72
    cooperative_delta_alpha_fast: float = 0.90
    cooperative_max_delta_px_per_frame: float = 64.0
    cooperative_actuator_alpha_slow: float = 0.92
    cooperative_actuator_alpha_fast: float = 0.98
    cooperative_precision_gain: float = 0.58
    cooperative_max_target_pull: float = 0.35
    cooperative_target_pull_max_px_per_frame: float = 7.0
    cooperative_target_pull_fraction: float = 0.10
    # Predictive target field.  These values are intentionally modest: the UI
    # element should help the user finish an aimed movement, never take over it.
    target_prediction_horizon: float = 0.14
    target_min_motion_px_s: float = 32.0
    target_lock_radius_px: float = 170.0
    target_lock_exit_px: float = 270.0
    target_lock_speed_px_s: float = 1100.0
    target_lock_enter_score: float = 0.46
    target_lock_exit_score: float = 0.14
    target_attraction_strength: float = 0.42
    target_arrival_speed_px_s: float = 420.0
    target_arrival_gain: float = 4.5
    target_precision_radius_px: float = 72.0
    target_precision_speed_px_s: float = 240.0
    target_gravity_accel_scale: float = 0.85

    # Absolute virtual touch-surface controller.
    touch_surface_use_calibration: bool = False
    touch_surface_calibration_path: str = "config/calibration/touch_surface.json"
    touch_surface_hysteresis_px: float = 1.0
    touch_surface_alpha_slow: float = 0.94
    touch_surface_alpha_fast: float = 0.995
    touch_surface_beta: float = 0.060
    touch_surface_max_innovation: float = 0.30
    touch_surface_stationary_speed: float = 0.018
    touch_surface_min_update_px: float = 1.0
    touch_surface_prediction_horizon: float = 0.0
    touch_surface_recovery_reanchor: bool = False
    touch_surface_recovery_release_motion: float = 0.010


@dataclass
class BimanualConfig:
    enabled: bool = True
    # Secondary hand action modifier.
    left_click_max_ms: float = 260.0
    right_click_hold_ms: float = 380.0
    # One-hand drag is intentionally disabled in the shipped touch profile.
    # Dragging a folder from an accidental pinch is much worse than requiring
    # the second-hand modifier for a drag.
    one_hand_drag_enabled: bool = False
    drag_start_ms: float = 280.0
    drag_start_px: float = 26.0
    double_click_window: float = 0.38
    # Two-hand pinch zoom. It requires a deliberate separation change; a
    # held pinch by itself must not emit wheel events.
    zoom_enabled: bool = True
    zoom_start_ms: float = 180.0
    zoom_deadband_ratio: float = 0.08
    zoom_emit_interval: float = 0.12
    zoom_baseline_alpha: float = 0.04
    zoom_steps_gain: float = 2.0
    zoom_max_steps: float = 1.0
    # Two-index selection is opt-in. Plain two-hand pointing is too common to
    # treat as "hold LMB and select desktop items" by default.
    selection_enabled: bool = False
    selection_arm_ms: float = 450.0
    selection_min_separation_px: float = 120.0
    # Two close index fingers can pan a canvas with a middle-button drag.
    pan_enabled: bool = False
    pan_start_px: float = 24.0
    # Prevent a continuous modifier from firing too often.
    action_cooldown: float = 0.18
    action_min_confidence: float = 0.70


@dataclass
class UserModelConfig:
    enabled: bool = True
    half_life: float = 45.0  # seconds
    warmup_samples: int = 120
    sensitivity_range: tuple[float, float] = (0.6, 1.8)
    adaptation_rate: float = 0.05


@dataclass
class GrammarConfig:
    enabled: bool = True
    #: Pause longer than this separates two tokens of a sequence.
    separator_ms: float = 320.0
    #: Whole sequence must complete within this window.
    sequence_timeout_ms: float = 1600.0
    #: Rules: sequence of gesture names -> named command.
    rules: dict[str, list[str]] = field(
        default_factory=lambda: {
            "CLOSE_TAB": ["INDEX_UP", "pause", "SWIPE_LEFT"],
            "REOPEN_TAB": ["INDEX_UP", "pause", "SWIPE_RIGHT"],
            "NEW_WINDOW": ["INDEX_UP", "pause", "V_SIGN"],
            "MISSION_CONTROL": ["OPEN_PALM", "pause", "V_SIGN"],
            "SCREENSHOT": ["V_SIGN", "pause", "PINCH"],
            "LOCK": ["FIST", "pause", "FIST"],
        }
    )


@dataclass
class RecordConfig:
    directory: str = "recordings"
    format: str = "jsonl"
    include_frames: bool = True
    include_landmarks: bool = True


@dataclass
class DebugConfig:
    overlay: bool = True
    telemetry: bool = True
    show_landmarks: bool = True
    show_trajectory: bool = True
    show_prediction: bool = True
    show_targets: bool = True
    show_activation_zone: bool = True
    print_every_ms: float = 1000.0
    window_name: str = "intent-gesture-engine"
    scale: float = 1.0
    window_width: int = 960
    window_height: int = 540


@dataclass
class ControlConfig:
    enabled: bool = True  # False = dry-run, no OS input at all
    move_mouse: bool = True
    clicks: bool = True
    keyboard: bool = True
    windows: bool = True
    scroll_multiplier: float = 1.0
    #: Absolute | relative | hybrid
    mapping: str = "absolute"
    #: Screen region (normalized) mapped onto the full screen.
    active_region: tuple[float, float, float, float] = (0.10, 0.10, 0.90, 0.90)
    mouse_interval_ms: float = 0.0


@dataclass
class SafetyConfig:
    emergency_gesture: str = "FIST"
    require_activation: bool = True
    release_buttons_on_exit: bool = True
    max_click_rate_hz: float = 8.0
    max_scroll_rate_hz: float = 30.0
    freeze_below_confidence: float = 0.35
    stop_below_confidence: float = 0.20


@dataclass
class EngineConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    profiles: dict[str, CameraProfile] = field(
        default_factory=lambda: {
            "g85_12_60": CameraProfile(
                lens="Panasonic 12-60mm",
                distortion_correction="false",
                notes="Primary experimental lens. Framing at ~1.2 m, hand fills ~35% of frame.",
            ),
            "g85_fisheye": CameraProfile(
                lens="Samyang 3.5 fisheye",
                distortion_correction="optional",
                fisheye_fov_deg=180.0,
                notes="Room-scale interaction. Undistortion optional and must be benchmarked.",
            ),
            "webcam": CameraProfile(lens="generic UVC", notes="Fallback / regression baseline."),
        }
    )
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    state: StateConfig = field(default_factory=StateConfig)
    events: EventConfig = field(default_factory=EventConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    targets: TargetsConfig = field(default_factory=TargetsConfig)
    commit: CommitConfig = field(default_factory=CommitConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    cursor: CursorConfig = field(default_factory=CursorConfig)
    bimanual: BimanualConfig = field(default_factory=BimanualConfig)
    user_model: UserModelConfig = field(default_factory=UserModelConfig)
    grammar: GrammarConfig = field(default_factory=GrammarConfig)
    record: RecordConfig = field(default_factory=RecordConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)

    def active_profile(self) -> CameraProfile:
        return self.profiles.get(self.capture.profile, CameraProfile())


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _coerce(value: Any, hint: Any) -> Any:
    """Best-effort conversion of a YAML scalar into the annotated type."""
    if hint is None or hint is Any:
        return value
    origin = get_origin(hint)

    if origin in (list, Sequence, tuple):
        args = get_args(hint)
        if origin is tuple and args and len(args) == 2 and args[1] is Ellipsis:
            inner = args[0]
            return tuple(_coerce(v, inner) for v in value)
        if origin is tuple and args:
            return tuple(_coerce(v, a) for v, a in zip(value, args))
        inner = args[0] if args else Any
        return [_coerce(v, inner) for v in value]

    if origin is dict:
        args = get_args(hint)
        kt, vt = (args + (Any, Any))[:2] if args else (Any, Any)
        return {_coerce(k, kt): _coerce(v, vt) for k, v in value.items()}

    if origin in (dict,):
        return dict(value)

    if isinstance(hint, _pytypes.UnionType) or str(origin).endswith("Union"):
        for arg in get_args(hint):
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg)
            except Exception:  # noqa: BLE001 - try the next union member
                continue
        return value

    if isinstance(hint, type):
        if is_dataclass(hint):
            return _build(hint, value)
        if hint is float:
            return float(value)
        if hint is int:
            return int(value)
        if hint is bool:
            return bool(value)
        if hint is str:
            return str(value)
    return value


def _build(cls: type[T], data: Mapping[str, Any]) -> T:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):  # type: ignore[arg-type]
        if f.name not in data:
            continue
        kwargs[f.name] = _coerce(data[f.name], hints.get(f.name, f.type))
    return cls(**kwargs)  # type: ignore[arg-type]


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> EngineConfig:
    """Load a config, deep-merging YAML onto defaults."""
    cfg = EngineConfig()
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if target.exists():
        data = _read_yaml(target)
        if isinstance(data, Mapping):
            cfg = _build(EngineConfig, data)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    return cfg


def _read_yaml(path: Path) -> Mapping[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - yaml is a hard dependency
        raise RuntimeError(
            "PyYAML is required to read config files. Install it with `pip install pyyaml` "
            "or pass a pre-built EngineConfig."
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def apply_overrides(cfg: EngineConfig, overrides: Mapping[str, Any]) -> EngineConfig:
    """Apply ``{"cursor.sensitivity": 1.4}`` style overrides in place."""
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        obj: Any = cfg
        for part in parts[:-1]:
            if not hasattr(obj, part):
                raise KeyError(f"unknown config path: {dotted}")
            obj = getattr(obj, part)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise KeyError(f"unknown config path: {dotted}")
        current = getattr(obj, leaf)
        setattr(obj, leaf, _coerce(value, type(current)))
    return cfg


def to_dict(cfg: EngineConfig) -> dict[str, Any]:
    """Serializable snapshot (used by the recorder and the benchmark runner)."""

    def enc(v: Any) -> Any:
        if is_dataclass(v) and not isinstance(v, type):
            return {f.name: enc(getattr(v, f.name)) for f in fields(v)}
        if isinstance(v, dict):
            return {str(k): enc(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [enc(x) for x in v]
        return v

    return enc(cfg)  # type: ignore[return-value]
