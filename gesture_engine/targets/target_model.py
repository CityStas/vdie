"""Target Model + soft gravity field.

The model keeps a belief distribution over *what the user is trying to reach*.
Two rules from the master prompt are enforced here:

* a high target probability must never trigger an action by itself — it only
  modulates the control policy;
* the field must be soft.  Hard snapping is explicitly forbidden.

Additionally this module implements **negative intent**: a target the user
repeatedly skims past without entering loses influence.  Naive magnetic cursors
get *worse* with use because every nearby widget keeps pulling; decaying the
influence of avoided targets is what makes the field survivable in a dense UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np

from ..config import EngineConfig
from ..features.geometry import clamp01, smoothstep
from ..types import Intent, IntentField, Target, TargetBelief, TargetState


class TargetProvider(Protocol):
    """Anything that can supply candidate targets."""

    def refresh(self, timestamp: float) -> list[Target]: ...

    def close(self) -> None: ...


@dataclass
class StaticTargetProvider:
    """Fixed list of targets in normalized screen coordinates.

    This is the default: it makes the target-aware behaviour testable and
    benchmarkable without depending on UI Automation being present or sane.
    """

    targets: list[Target] = field(default_factory=list)

    def refresh(self, timestamp: float) -> list[Target]:
        return self.targets

    def close(self) -> None:
        pass

    @staticmethod
    def from_rects(rects: Iterable[tuple[str, tuple[float, float, float, float], float]]) -> "StaticTargetProvider":
        return StaticTargetProvider(
            targets=[
                Target(
                    id=name,
                    bounds=bounds,
                    type="ui",
                    semantic_label=name,
                    visual_confidence=conf,
                    semantic_confidence=conf,
                )
                for name, bounds, conf in rects
            ]
        )


@dataclass
class TargetFieldState:
    influence: float = 0.0
    approach: float = 0.0
    avoid_score: float = 0.0
    distance: float = 0.0
    closing_speed: float = 0.0
    inside: bool = False


class TargetModel:
    """Holds targets, their beliefs, and their memory of being avoided."""

    def __init__(self, cfg: EngineConfig, provider: TargetProvider | None = None) -> None:
        self.cfg = cfg
        self.provider = provider or StaticTargetProvider()
        self.targets: list[Target] = []
        self._state: dict[str, TargetFieldState] = {}
        self._last_refresh = -1e9
        self._prev_distance: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.targets = []
        self._state.clear()
        self._prev_distance.clear()

    def close(self) -> None:
        """Release resources owned by the target provider."""
        self.provider.close()

    def refresh(self, timestamp: float) -> None:
        tcfg = self.cfg.targets
        if (timestamp - self._last_refresh) * 1000.0 < tcfg.refresh_interval_ms:
            return
        self._last_refresh = timestamp
        self.targets = [t for t in self.provider.refresh(timestamp) if t.quality() >= tcfg.min_quality]
        for t in self.targets:
            self._state.setdefault(t.id, TargetFieldState())

    # ------------------------------------------------------------------ #
    def beliefs(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        intents: IntentField,
        timestamp: float,
    ) -> list[TargetBelief]:
        """Distance-weighted belief distribution over targets."""
        if not self.targets:
            return []
        pos = np.asarray(position, dtype=np.float64)
        vel = np.asarray(velocity, dtype=np.float64)
        out: list[TargetBelief] = []
        weights: list[float] = []

        for t in self.targets:
            st = self._state.setdefault(t.id, TargetFieldState())
            center = t.center
            delta = center - pos
            dist = float(np.linalg.norm(delta))
            st.distance = dist
            st.inside = t.contains(pos)

            self._prev_distance[t.id] = dist
            # Closing speed is the projection of the *actual* velocity onto the
            # direction of the target.  The previous frame-difference estimate
            # divided by a hard-coded 1/30 and therefore (a) was frame-rate
            # dependent and (b) disagreed with the velocity the rest of the
            # engine uses.
            if dist > 1e-9:
                st.closing_speed = float(np.dot(vel, (center - pos) / dist))
            else:
                st.closing_speed = 0.0

            size = float(np.hypot(*t.size))
            proximity = clamp01(1.0 - smoothstep(0.0, self.cfg.targets.gravity_radius * 4.0 + size * 0.5, dist))
            approach = clamp01(
                smoothstep(
                    self.cfg.targets.approach_min_closing_speed * 0.5,
                    self.cfg.targets.approach_min_closing_speed * 2.0,
                    st.closing_speed,
                )
            )
            st.approach = approach
            t.approach_confidence = approach

            if st.inside:
                st.avoid_score *= 0.7
                t.state = TargetState.APPROACH
            elif st.closing_speed > self.cfg.targets.approach_min_closing_speed:
                t.state = TargetState.APPROACH
            elif st.closing_speed < -self.cfg.targets.approach_min_closing_speed and dist < 0.25:
                t.state = TargetState.AVOID
                st.avoid_score = min(1.0, st.avoid_score + 0.25)
            else:
                t.state = TargetState.NEUTRAL
                st.avoid_score *= self.cfg.targets.avoid_decay

            quality = t.quality()
            penalty = 1.0 - self.cfg.targets.avoid_penalty * clamp01(st.avoid_score)
            w = proximity * quality * penalty * (0.5 + 0.5 * approach)
            weights.append(max(w, 0.0))
            out.append(TargetBelief(target=t, confidence=w, distance=dist, closing_speed=st.closing_speed))

        total = sum(weights)
        if total > 0:
            for b, w in zip(out, weights):
                b.confidence = float(w / total)
        return out

    # ------------------------------------------------------------------ #
    def gravity(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        beliefs: list[TargetBelief],
        intents: IntentField,
        dt: float,
    ) -> np.ndarray:
        """Net soft attraction force, in normalized screen units per second^2."""
        if not beliefs:
            return np.zeros(2)
        tcfg = self.cfg.targets
        pos = np.asarray(position, dtype=np.float64)
        speed = float(np.linalg.norm(velocity))

        # Fast travel must stay free: no gravity above the suppress speed.
        travel_scale = 1.0 - smoothstep(tcfg.gravity_suppress_speed * 0.6, tcfg.gravity_suppress_speed, speed)
        if travel_scale <= 0.0:
            for b in beliefs:
                b.target.influence = 0.0
            return np.zeros(2)

        intent_conf = max(
            intents.get(Intent.SELECT),
            intents.get(Intent.DRAG),
            intents.get(Intent.DOUBLE_CLICK),
        )
        # Gravity only wakes up when the user is actually selecting something.
        intent_scale = smoothstep(0.25, 0.75, intent_conf)
        if intent_scale <= 0.0:
            for b in beliefs:
                b.target.influence = 0.0
            return np.zeros(2)

        force = np.zeros(2)
        for b in beliefs:
            t = b.target
            if t.state is TargetState.AVOID:
                t.influence = 0.0
                continue
            delta = t.center - pos
            dist = float(np.linalg.norm(delta))
            if dist < 1e-6:
                continue
            direction = delta / dist
            proximity = clamp01(1.0 - smoothstep(0.0, tcfg.gravity_radius * 3.0, dist))
            magnitude = (
                tcfg.gravity_k
                * proximity
                * t.quality()
                * intent_conf
                * max(t.approach_confidence, 0.25)
                * travel_scale
                * intent_scale
            )
            magnitude = min(magnitude, tcfg.gravity_max_force)
            # Never pull harder than a fraction of the remaining distance: this
            # is what keeps the field soft and prevents snapping.
            magnitude = min(magnitude, dist * tcfg.max_pull_fraction / max(dt, 1e-3))
            t.influence = float(magnitude)
            force += direction * magnitude
        return force

    # ------------------------------------------------------------------ #
    def predicted_target(self, position: np.ndarray, predicted: np.ndarray) -> Target | None:
        """Target whose bounds the predicted point lands in, if any."""
        for t in self.targets:
            if t.contains(predicted):
                return t
        best, best_d = None, 1e9
        for t in self.targets:
            d = float(np.linalg.norm(t.center - np.asarray(predicted, dtype=np.float64)))
            if d < best_d:
                best, best_d = t, d
        return best

    def snapshot(self) -> list[dict]:
        return [
            {
                "id": t.id,
                "bounds": t.bounds,
                "state": t.state.value,
                "influence": t.influence,
                "approach": t.approach_confidence,
                "quality": t.quality(),
                "avoid": self._state.get(t.id, TargetFieldState()).avoid_score,
            }
            for t in self.targets
        ]
