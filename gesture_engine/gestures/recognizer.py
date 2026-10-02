"""Gesture Recognizer — an *evidence provider*, not a decision maker.

The recogniser merges static pose scores and dynamic trajectory scores, applies
temporal voting and hysteresis, and emits a :class:`GestureResult` with
confidence, stability and duration.  It does **not** decide what should happen;
that is the intent field's job.
"""

from __future__ import annotations

from collections import deque

from ..config import EngineConfig
from ..motion.history import HistoryBuffer
from ..types import DYNAMIC_GESTURES, GESTURE_NONE, GestureResult, HandFeatures
from .confidence import Cooldown, VoteBuffer
from .dynamic import score_dynamic
from .static import score_static


class GestureRecognizer:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.votes = VoteBuffer(window=cfg.gestures.vote_window, required=cfg.gestures.vote_required)
        self._scores: deque[dict[str, float]] = deque(maxlen=max(cfg.gestures.vote_window * 3, 30))
        self._current = "NONE"
        self._current_since = 0.0
        self._cooldown = Cooldown(interval=cfg.gestures.gesture_cooldown)
        self._last_result = GESTURE_NONE

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.votes.reset()
        self._scores.clear()
        self._current = "NONE"
        self._current_since = 0.0
        self._cooldown.reset()
        self._last_result = GESTURE_NONE

    @property
    def current(self) -> str:
        return self._current

    def score_history(self) -> list[dict[str, float]]:
        return list(self._scores)

    # ------------------------------------------------------------------ #
    def update(
        self,
        features: HandFeatures | None,
        history: HistoryBuffer,
        timestamp: float,
        hand_confidence: float = 1.0,
    ) -> GestureResult:
        g = self.cfg.gestures
        if features is None or hand_confidence < self.cfg.safety.stop_below_confidence:
            self.votes.push("NONE")
            self._scores.append({})
            self._current = "NONE"
            self._last_result = GestureResult(gesture="NONE", confidence=0.0, timestamp=timestamp)
            return self._last_result

        static_scores = score_static(features, self.cfg)
        dyn = score_dynamic(history, self.cfg, features.extended_count)
        scores = dict(static_scores)
        for name, value in dyn.items():
            scores[name] = max(scores.get(name, 0.0), value)
        self._scores.append(scores)

        # A dynamic gesture is not a *competitor* of the static pose that carries
        # it — the pose is a necessary condition of the trajectory.  Taking the
        # global argmax therefore made every dynamic gesture unreachable: a held
        # INDEX_UP scores ~1.0, so SWIPE_*/SCROLL_* could never win.  A confident
        # trajectory wins outright; otherwise the best static pose is the answer.
        best_dyn = max(dyn, key=lambda k: dyn[k]) if dyn else None
        if best_dyn is not None and dyn[best_dyn] >= g.min_confidence:
            best = best_dyn
        else:
            best = max(static_scores, key=lambda k: static_scores[k]) if static_scores else "NONE"
        best_score = float(scores.get(best, 0.0))
        if best_score < g.min_confidence:
            best = "NONE"

        self.votes.push(best)

        confirmed = self.votes.confirmed()
        if confirmed is None:
            # Dynamic patterns get their own, shorter confirmation window (see
            # VoteBuffer.confirmed_tail).
            tail = self.votes.confirmed_tail(g.vote_window_dynamic, g.vote_required_dynamic)
            if tail is not None and tail[0] in DYNAMIC_GESTURES:
                confirmed = tail
        emitted = self._current

        if confirmed is not None:
            name, agreement = confirmed
            if name != self._current:
                # Switch only if the candidate is decisively better than the
                # incumbent, or the incumbent is no longer supported at all.
                # A dynamic candidate needs no margin over a static incumbent:
                # "SWIPE_LEFT while pointing" is strictly more informative than
                # "pointing".
                incumbent_score = scores.get(self._current, 0.0)
                candidate_score = scores.get(name, 0.0)
                dynamic_override = name in DYNAMIC_GESTURES and self._current not in DYNAMIC_GESTURES
                decisive = dynamic_override or candidate_score >= incumbent_score + 0.12 or incumbent_score < g.min_confidence
                if decisive and self._cooldown.try_consume(timestamp):
                    emitted = name
                    self._current = name
                    self._current_since = timestamp
            agreement_ratio = agreement
        else:
            agreement_ratio = self.votes.agreement(self._current) if self._current != "NONE" else 0.0
            # Drop out when the incumbent loses support.
            if self._current != "NONE":
                incumbent_score = scores.get(self._current, 0.0)
                if incumbent_score < g.min_confidence * 0.75:
                    self._current = "NONE"
                    emitted = "NONE"

        duration = timestamp - self._current_since if emitted != "NONE" else 0.0
        if emitted != "NONE" and duration < g.min_gesture_duration:
            emitted = "NONE"
            duration = 0.0

        stability = 0.5 * agreement_ratio + 0.5 * self._mean_score(self._current)

        self._last_result = GestureResult(
            gesture=emitted if emitted != "NONE" else "NONE",
            confidence=float(scores.get(emitted, 0.0)) if emitted != "NONE" else 0.0,
            stability=float(min(stability, 1.0)),
            duration=float(duration),
            timestamp=timestamp,
            source="deterministic",
        )
        return self._last_result

    # ------------------------------------------------------------------ #
    def _mean_score(self, gesture: str) -> float:
        if gesture == "NONE" or not self._scores:
            return 0.0
        window = list(self._scores)[-self.cfg.gestures.vote_window :]
        vals = [s.get(gesture, 0.0) for s in window]
        return float(sum(vals) / len(vals)) if vals else 0.0

    # ------------------------------------------------------------------ #
    def all_scores(self) -> dict[str, float]:
        return dict(self._scores[-1]) if self._scores else {}
