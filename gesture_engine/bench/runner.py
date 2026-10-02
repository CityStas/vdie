"""Benchmark runner.

Ties together the synthetic/replay sources, the engine and the metrics.  The
point is to make experiments *reproducible*: one call defines a source, one call
defines an algorithm variant, and the harness guarantees the same input stream.

Experiment axes implemented here (master prompt section 53):

============  ==========================================================
``algorithm`` v1 (discrete intent) vs v2 (intent field)
``smoothing`` fixed vs adaptive vs one-euro (via ``cursor.controller``)
``prediction`` on vs off (``cursor.prediction_horizon``)
``commit``    pinch-only vs kinematic commit (``commit.enabled``)
``targets``   off vs static vs gravity (``targets.provider`` / ``gravity_k``)
``fps``       30 vs 60 (source resampling)
============  ==========================================================
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..config import EngineConfig, load_config
from ..control.dispatcher import ActionDispatcher
from ..debug.recorder import Recorder, ReplayTracker, load_recording
from ..engine import GestureEngine
from ..targets.target_model import StaticTargetProvider
from ..targets.providers import default_static_targets
from ..tracking.synthetic import Segment, SyntheticTracker, get_scenario
from .metrics import Evaluation, SegmentTruth, compare, evaluate, format_table, timing_table


def truth_from_segments(segments: list[Segment], t0: float = 0.0) -> list[SegmentTruth]:
    out: list[SegmentTruth] = []
    t = t0
    for seg in segments:
        out.append(SegmentTruth(t0=t, t1=t + seg.duration, label=seg.label, intent=seg.intent, tags=dict(seg.tags)))
        t += seg.duration
    return out


@dataclass
class Variant:
    """One algorithm configuration under test."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def apply(self, cfg: EngineConfig) -> EngineConfig:
        from ..config import apply_overrides

        clone = copy.deepcopy(cfg)
        apply_overrides(clone, self.overrides)
        return clone


#: The variant set that answers the master prompt's "5+ substantially different
#: interaction concepts" question with measurements instead of opinions.
DEFAULT_VARIANTS: tuple[Variant, ...] = (
    Variant("baseline_pinch", {"commit.enabled": False, "cursor.controller": "gain", "cursor.intent_adaptive": False}),
    Variant("adaptive_ema", {"commit.enabled": False, "cursor.controller": "velocity_curve", "cursor.intent_adaptive": False}),
    Variant("spring_oneeuro", {"commit.enabled": False, "cursor.controller": "spring", "cursor.intent_adaptive": False}),
    Variant("intent_adaptive_gain", {"commit.enabled": False, "cursor.controller": "spring", "cursor.intent_adaptive": True}),
    Variant("kinematic_commit", {"commit.enabled": True, "cursor.controller": "spring", "cursor.intent_adaptive": True}),
    # Same as ``kinematic_commit`` with the intent-adaptive cursor gain off.
    # This pair is what decides whether the hesitation is worth its lag.
    Variant("kinematic_commit_no_intent_gain", {"commit.enabled": True, "cursor.controller": "spring", "cursor.intent_adaptive": False}),
    Variant("kinematic_commit_no_gravity", {"commit.enabled": True, "targets.gravity_k": 0.0}),
    # Ablation of the cursor's latency hiding.  ``no_feedforward`` restores the
    # old positional lead, which is what "prediction" used to mean here.
    Variant("no_feedforward", {"cursor.feedforward": False, "cursor.prediction_horizon": 0.045}),
    Variant("no_adaptive_user_model", {"user_model.enabled": False}),
)


class BenchmarkRunner:
    def __init__(self, cfg: EngineConfig | None = None, fps: int = 30, duration: float | None = None) -> None:
        self.cfg = cfg or load_config()
        self.fps = fps
        self.duration = duration
        self.results: list[Evaluation] = []

    # ------------------------------------------------------------------ #
    def _truth(self, segments: list[Segment], start_time: float) -> list[SegmentTruth]:
        return truth_from_segments(segments, t0=start_time)

    def run_variant(
        self,
        variant: Variant,
        segments: list[Segment],
        duration: float | None = None,
        screen: tuple[int, int] = (1920, 1080),
    ) -> Evaluation:
        cfg = variant.apply(self.cfg)
        cfg.capture.fps = self.fps
        cfg.tracking.backend = "synthetic"
        cfg.control.enabled = False  # benchmark never touches the OS

        tracker = SyntheticTracker(cfg, segments=segments, loop=False)
        provider = StaticTargetProvider(default_static_targets())
        engine = GestureEngine(
            cfg,
            tracker,
            dispatcher=ActionDispatcher(cfg=cfg, dry_run=True),
            target_provider=provider,
            screen=screen,
        )

        total = duration if duration is not None else sum(s.duration for s in segments)
        frames = int(total * self.fps)
        dt = 1.0 / self.fps
        reports = []
        engine.start()
        try:
            for i in range(frames):
                t = i * dt
                report = engine.process_frame(None, t, i, {"benchmark": variant.name})
                if report is None:
                    break
                reports.append(report)
        finally:
            engine.stop()

        truth = self._truth(segments, start_time=0.0)
        ev = evaluate(reports, truth, name=variant.name, screen=screen)
        self.results.append(ev)
        return ev

    # ------------------------------------------------------------------ #
    def run_matrix(
        self,
        variants: tuple[Variant, ...] | list[Variant] | None = None,
        scenario: str = "click",
        duration: float | None = None,
    ) -> dict[str, dict]:
        variants = variants or DEFAULT_VARIANTS
        segments = get_scenario(scenario)
        for v in variants:
            self.run_variant(v, segments, duration=duration)
        return compare(self.results)

    # ------------------------------------------------------------------ #
    def replay_matrix(
        self,
        recording: str,
        variants: tuple[Variant, ...] | list[Variant] | None = None,
    ) -> dict[str, dict]:
        """Replay one recording through several algorithm versions."""
        variants = variants or DEFAULT_VARIANTS
        rec = load_recording(recording)
        base_cfg = rec.config() or self.cfg
        for v in variants:
            cfg = v.apply(base_cfg)
            cfg.control.enabled = False
            engine = GestureEngine(
                cfg,
                ReplayTracker(rec),
                dispatcher=ActionDispatcher(cfg=cfg, dry_run=True),
                target_provider=StaticTargetProvider(default_static_targets()),
            )
            reports = []
            engine.start()
            try:
                while True:
                    report = engine.process_frame(None, 0.0, len(reports), {"benchmark": v.name})
                    if report is None:
                        break
                    reports.append(report)
            finally:
                engine.stop()
            truth = [
                SegmentTruth(
                    t0=f["t"],
                    t1=f["t"],
                    label=f.get("notes", {}).get("label", ""),
                    intent=f.get("notes", {}).get("intent_label", ""),
                )
                for f in rec.frames
            ]
            self.results.append(evaluate(reports, None, name=v.name))
        return compare(self.results)

    # ------------------------------------------------------------------ #
    def report(self) -> str:
        return format_table(compare(self.results))

    def timing_report(self) -> dict[str, dict]:
        """Wall-clock cost per stage.  Kept out of :meth:`report` on purpose."""
        return timing_table(self.results)

    def as_dict(self) -> dict:
        return {ev.name: ev.as_dict() for ev in self.results}


# --------------------------------------------------------------------------- #
# Timing harness (measure the pipeline cost of each stage in isolation)
# --------------------------------------------------------------------------- #


def time_pipeline(cfg: EngineConfig, frames: int = 300, fps: int = 30) -> dict[str, float]:
    """Wall-clock cost per stage, measured on synthetic input.

    Reports the *pipeline* cost only.  Camera capture and MediaPipe inference are
    excluded by construction (synthetic source), so this number is a lower bound
    on the real system — which is exactly what you want when comparing algorithm
    versions.
    """
    cfg = copy.deepcopy(cfg)
    cfg.control.enabled = False
    tracker = SyntheticTracker(cfg, segments=get_scenario("click"), loop=True)
    engine = GestureEngine(cfg, tracker, dispatcher=ActionDispatcher(cfg=cfg, dry_run=True))
    dt = 1.0 / fps
    engine.start()
    try:
        for i in range(frames):
            engine.process_frame(None, i * dt, i)
    finally:
        engine.stop()
    return engine.telemetry.summary()
