"""Motion Signature / User Model.

Passive, online, no training.  The engine watches how *this* user moves and
derives a handful of scalars that modulate sensitivity, smoothing and the commit
detector's preference.  Personal calibration therefore costs nothing: the first
few interactions are the calibration.

Deliberately **not** an identity mechanism — it is a control-style model.  Nothing
here is stored in a form that identifies a person, and the model is reset with the
session unless the caller persists it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EngineConfig
from ..features.geometry import clamp01
from ..types import CommitSignal, MotionState


@dataclass
class MotionSignature:
    samples: int = 0
    typical_speed: float = 0.0
    peak_speed: float = 0.0
    acceleration_style: float = 0.0  # >0 = accelerates hard, <0 = eases in
    micro_pause_rate: float = 0.0  # pauses per second of movement
    trajectory_noise: float = 0.0
    preferred_amplitude: float = 0.0
    pinch_timing: float = 0.0  # mean time from deceleration to pinch
    direction_bias: float = 0.0  # fraction of movement that is horizontal
    pause_bias: float = 0.5  # how often this user commits with a micro-pause
    thrust_bias: float = 0.5  # ... versus a short thrust

    @property
    def ready(self) -> bool:
        return self.samples > 0

    def as_dict(self) -> dict[str, float]:
        return {
            "samples": float(self.samples),
            "typical_speed": self.typical_speed,
            "peak_speed": self.peak_speed,
            "acceleration_style": self.acceleration_style,
            "micro_pause_rate": self.micro_pause_rate,
            "trajectory_noise": self.trajectory_noise,
            "preferred_amplitude": self.preferred_amplitude,
            "direction_bias": self.direction_bias,
            "pause_bias": self.pause_bias,
            "thrust_bias": self.thrust_bias,
        }


@dataclass
class UserModel:
    cfg: EngineConfig
    signature: MotionSignature = field(default_factory=MotionSignature)
    _last_t: float | None = None
    _movement_time: float = 0.0
    _pauses: int = 0
    _pausing: bool = False
    _prev_commit_t: float | None = None
    _commit_kinds: list[str] = field(default_factory=list)
    _last_speed_at_commit: float = 0.0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.signature = MotionSignature()
        self._last_t = None
        self._movement_time = 0.0
        self._pauses = 0
        self._pausing = False
        self._prev_commit_t = None
        self._commit_kinds.clear()

    @property
    def enabled(self) -> bool:
        return self.cfg.user_model.enabled

    # ------------------------------------------------------------------ #
    def observe(self, motion: MotionState, commit: CommitSignal | None = None) -> None:
        if not self.enabled:
            return
        ucfg = self.cfg.user_model
        sig = self.signature
        rate = ucfg.adaptation_rate
        dt = 0.0 if self._last_t is None else max(motion.timestamp - self._last_t, 0.0)
        self._last_t = motion.timestamp
        if dt <= 0:
            return

        sig.samples += 1
        warmup = sig.samples < ucfg.warmup_samples
        a = max(rate, 0.5 / max(sig.samples, 1)) if warmup else rate

        sig.typical_speed += a * (motion.speed - sig.typical_speed)
        sig.peak_speed = max(sig.peak_speed * 0.999, motion.speed)
        sig.acceleration_style += a * (motion.speed_trend / max(motion.speed + 0.3, 0.3) - sig.acceleration_style)
        sig.preferred_amplitude += a * (motion.amplitude - sig.preferred_amplitude)

        if motion.speed > self.cfg.motion.pause_speed:
            self._movement_time += dt
            total = float(np.hypot(*motion.velocity)) + 1e-6
            horizontal = abs(float(motion.velocity[0])) / total
            sig.direction_bias += a * (horizontal - sig.direction_bias)

        if motion.is_pausing:
            if not self._pausing:
                self._pausing = True
                self._pauses += 1
        else:
            self._pausing = False

        if self._movement_time > 1.0:
            sig.micro_pause_rate += a * (self._pauses / self._movement_time - sig.micro_pause_rate)

        if commit is not None and commit.committed:
            self._commit_kinds.append(commit.kind)
            self._commit_kinds = self._commit_kinds[-50:]
            pause_like = sum(1 for k in self._commit_kinds if k in ("MICRO_PAUSE", "TARGET_ENTRY", "VELOCITY_REVERSAL"))
            thrust_like = sum(1 for k in self._commit_kinds if k == "THRUST")
            total = max(pause_like + thrust_like, 1)
            sig.pause_bias += a * (pause_like / total - sig.pause_bias)
            sig.thrust_bias += a * (thrust_like / total - sig.thrust_bias)

    # ------------------------------------------------------------------ #
    def sensitivity_scale(self) -> float:
        """How much to scale cursor sensitivity for this user.

        A user with a low typical speed and short amplitude movements needs a
        *higher* gain to cover the same screen distance; a fast, wide mover needs
        less.  Clamped, because this must never make the cursor unusable.
        """
        if not self.enabled or not self.signature.ready:
            return 1.0
        sig = self.signature
        lo, hi = self.cfg.user_model.sensitivity_range
        reference_speed = 0.8
        reference_amp = 0.35
        speed_ratio = reference_speed / max(sig.typical_speed, 0.15)
        amp_ratio = reference_amp / max(sig.preferred_amplitude, 0.05)
        raw = 0.5 * (speed_ratio + amp_ratio)
        return float(np.clip(raw, lo, hi))

    def smoothing_bias(self) -> float:
        """Extra smoothing for jittery users (0 = none, 1 = maximum)."""
        if not self.enabled or not self.signature.ready:
            return 0.0
        return float(clamp01(self.signature.trajectory_noise / 0.01))

    def commit_preference(self) -> str:
        sig = self.signature
        if not sig.ready:
            return "balanced"
        if sig.pause_bias - sig.thrust_bias > 0.2:
            return "pause"
        if sig.thrust_bias - sig.pause_bias > 0.2:
            return "thrust"
        return "balanced"

    def to_dict(self) -> dict:
        return self.signature.as_dict()

    @staticmethod
    def from_dict(cfg: EngineConfig, data: dict) -> "UserModel":
        model = UserModel(cfg=cfg)
        for k, v in data.items():
            if hasattr(model.signature, k):
                setattr(model.signature, k, v)
        model.signature.samples = int(model.signature.samples)
        return model
