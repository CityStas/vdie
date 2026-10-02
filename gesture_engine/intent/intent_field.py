"""Intent Field.

The Intent Field is a *continuously changing distribution* over intents, not a
classification result.  It accumulates weighted evidence in logit space with
exponential decay, converts to a distribution, and applies per-intent hysteresis
so that the dominant intent does not flicker on a noisy boundary.

Two design notes:

* Evidence is summed as logits with a decay term.  With decay rate ``lambda``,
  the steady-state logit for a constant evidence stream ``e`` is ``e / lambda``,
  so weights and decay are *not* independent knobs — the benchmark harness treats
  ``decay_per_second`` as part of the algorithm version for exactly this reason.
* The field is allowed to be ambiguous.  It reports ``entropy`` and refuses to
  pick a committed intent when nothing clears its ``enter`` threshold.  The cursor
  controller consumes that ambiguity as a control signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EngineConfig
from ..types import (
    CommitSignal,
    Event,
    GestureResult,
    HandFeatures,
    Intent,
    IntentField,
    MotionState,
    softmax,
)
from ..motion.history import HistoryBuffer
from ..state.state_machine import StateMachine
from ..types import TargetBelief
from .evidence import (
    CommitEvidence,
    ContextEvidence,
    EvidenceProvider,
    FutureEvidence,
    GestureEvidence,
    MotionEvidence,
    TargetEvidence,
    TemporalEvidence,
    UserEvidence,
)


@dataclass
class IntentFieldBuilder:
    cfg: EngineConfig

    def __post_init__(self) -> None:
        self.providers: dict[str, EvidenceProvider] = {
            "gesture": GestureEvidence(cfg=self.cfg),
            "motion": MotionEvidence(),
            "temporal": TemporalEvidence(window=0.9),
            "target": TargetEvidence(),
            "context": ContextEvidence(),
            "commit": CommitEvidence(cfg=self.cfg),
            "future": FutureEvidence(),
            "user": UserEvidence(),
        }
        self._accum: dict[Intent, float] = {}
        self._committed: Intent | None = None
        self._last_timestamp: float | None = None
        self._last_field = IntentField()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._accum.clear()
        self._committed = None
        self._last_timestamp = None
        self._last_field = IntentField()

    @property
    def committed(self) -> Intent | None:
        return self._committed

    @property
    def last_field(self) -> IntentField:
        return self._last_field

    # ------------------------------------------------------------------ #
    def update(
        self,
        *,
        timestamp: float,
        features: HandFeatures | None,
        motion: MotionState,
        gesture: GestureResult | None,
        history: HistoryBuffer,
        events: list[Event],
        beliefs: list[TargetBelief],
        fsm: StateMachine,
        commit: CommitSignal | None,
        futures: list | None = None,
        user_signature: object | None = None,
    ) -> IntentField:
        icfg = self.cfg.intent

        # ---- decay ---------------------------------------------------- #
        dt = 0.0 if self._last_timestamp is None else max(timestamp - self._last_timestamp, 0.0)
        self._last_timestamp = timestamp
        # ---- decay toward base rates ----------------------------------- #
        # Each intent's accumulated logit decays exponentially *towards its
        # prior* rather than towards zero.  With a constant evidence stream `e`
        # the steady state becomes `prior + e / decay`, which is what makes the
        # priors meaningful base rates instead of a slow drift to zero.
        priors: dict[Intent, float] = {}
        for name, value in icfg.priors.items():
            try:
                priors[Intent(name)] = float(value)
            except ValueError:
                continue
        if dt > 0:
            factor = float(np.exp(-icfg.decay_per_second * dt))
            for key in set(self._accum) | set(priors):
                prev = self._accum.get(key, 0.0)
                base = priors.get(key, 0.0)
                self._accum[key] = base + (prev - base) * factor

        # ---- temporal provider needs to observe the event stream ------ #
        temporal = self.providers["temporal"]
        if isinstance(temporal, TemporalEvidence):
            temporal.observe(events, timestamp)
        user = self.providers["user"]
        if isinstance(user, UserEvidence):
            user.signature = user_signature

        # ---- gather evidence ------------------------------------------ #
        kwargs = dict(
            features=features,
            motion=motion,
            gesture=gesture,
            history=history,
            events=events,
            beliefs=beliefs,
            fsm=fsm,
            commit=commit,
            timestamp=timestamp,
            futures=futures,
        )
        per_source: dict[str, dict[Intent, float]] = {}
        for name, provider in self.providers.items():
            weight = icfg.weights.get(name, 0.0)
            if weight == 0.0:
                continue
            try:
                logits = provider.logits(**kwargs)  # type: ignore[arg-type]
            except TypeError:
                logits = provider.logits(**{k: v for k, v in kwargs.items() if k != "futures"})  # type: ignore[arg-type]
            if not logits:
                continue
            # Evidence is a *rate* in logits per second for continuous sources and
            # an *impulse* for instantaneous events (a commit).  Without this
            # distinction the field becomes frame-rate dependent: a 60 FPS run
            # would accumulate twice the evidence of a 30 FPS run.
            if getattr(provider, "kind", "rate") == "impulse":
                # An impulse is an *absolute* statement about the world, not a
                # relative decrement: "the movement stopped deliberately" and
                # "there is a selection now" must not depend on how long the
                # competing hypotheses have been accumulating.  Positive logits
                # therefore raise a floor, negative logits lower a ceiling.
                # Adding them instead let MOVE_CURSOR's steady state (~prior +
                # rate/decay) out-vote a commit that the design says must
                # dominate the field for the ~0.3 s its impulse survives.
                for intent, value in logits.items():
                    base = priors.get(intent, 0.0)
                    target = base + value * weight
                    current = self._accum.get(intent, base)
                    self._accum[intent] = max(current, target) if value >= 0 else min(current, target)
                per_source[name] = {k: v * weight for k, v in logits.items()}
                continue
            scale = weight * dt * icfg.evidence_gain
            if scale <= 0.0:
                continue
            per_source[name] = {k: v * scale for k, v in logits.items()}
            for intent, value in logits.items():
                self._accum[intent] = self._accum.get(intent, 0.0) + value * scale

        # ---- distribution --------------------------------------------- #
        probs = softmax(self._accum, temperature=icfg.temperature)
        probs = {i: p for i, p in probs.items() if p > 1e-4}
        if not probs:
            probs = {Intent.UNKNOWN: 1.0}
        p = np.asarray(list(probs.values()), dtype=np.float64)
        entropy = float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())

        dominant, dominant_p = max(probs.items(), key=lambda kv: kv[1])

        # ---- hysteresis ------------------------------------------------ #
        committed = self._committed
        hyst = icfg.hysteresis.get(committed.value if committed else "", {})
        stay = hyst.get("stay", 0.5)
        exit_t = hyst.get("exit", 0.3)
        if committed is not None:
            current_p = probs.get(committed, 0.0)
            if current_p < exit_t:
                committed = None
        if committed is None:
            enter = icfg.hysteresis.get(dominant.value, {}).get("enter", 0.5)
            if dominant_p >= enter and dominant is not Intent.UNKNOWN:
                committed = dominant
        elif committed is not dominant:
            enter = icfg.hysteresis.get(dominant.value, {}).get("enter", 0.5)
            if dominant_p >= enter and probs.get(committed, 0.0) < stay:
                committed = dominant
        self._committed = committed

        field_obj = IntentField(
            probabilities=probs,
            logits=dict(self._accum),
            entropy=entropy,
            timestamp=timestamp,
            dominant=dominant,
            dominant_probability=float(dominant_p),
            committed=committed,
        )
        self._last_field = field_obj
        return field_obj

    # ------------------------------------------------------------------ #
    def debug_snapshot(self) -> dict[str, float]:
        return {k.value: round(v, 4) for k, v in sorted(self._accum.items(), key=lambda kv: -kv[1])}
