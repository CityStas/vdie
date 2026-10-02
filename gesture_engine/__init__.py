"""Intent-Based Gesture Control Engine.

A real-time hand-interaction engine that treats gesture recognition as *one
evidence source* among several, and makes the intent field, the target model and
the action policy the centre of the system.

Quick start (headless, no camera needed)::

    from gesture_engine import load_config, GestureEngine, build_tracker

    cfg = load_config()
    tracker = build_tracker(cfg, source="synthetic", scenario="demo")
    engine = GestureEngine(cfg, tracker)
    engine.run_source(NullSource(cfg), max_frames=600)
"""

from __future__ import annotations

__version__ = "0.3.1"

from .config import EngineConfig, load_config, to_dict
from .engine import GestureEngine
from .types import (
    ActionRequest,
    CommitSignal,
    Event,
    EventType,
    FrameReport,
    GestureResult,
    HandFeatures,
    Intent,
    IntentField,
    Landmarks,
    MotionState,
    Observation,
    Target,
    TargetBelief,
)

__all__ = [
    "__version__",
    "EngineConfig",
    "load_config",
    "to_dict",
    "GestureEngine",
    "ActionRequest",
    "CommitSignal",
    "Event",
    "EventType",
    "FrameReport",
    "GestureResult",
    "HandFeatures",
    "Intent",
    "IntentField",
    "Landmarks",
    "MotionState",
    "Observation",
    "Target",
    "TargetBelief",
    "build_tracker",
    "build_dispatcher",
    "build_target_provider",
]


def build_tracker(cfg: EngineConfig, source: str | None = None, scenario: str = "demo", recording: str | None = None):
    """Factory for the three tracking backends: mediapipe / synthetic / replay."""
    source = source or cfg.tracking.backend
    if source == "mediapipe":
        from .tracking.hand_tracker import HandTracker

        return HandTracker(cfg)
    if source == "synthetic":
        from .tracking.synthetic import SyntheticTracker, get_scenario

        return SyntheticTracker(cfg, segments=get_scenario(scenario))
    if source == "replay":
        if not recording:
            raise ValueError("replay source requires `recording=<path>`")
        from .debug.recorder import ReplayTracker, load_recording

        return ReplayTracker(load_recording(recording))
    if source == "null":
        from .tracking.hand_tracker import NullTracker

        return NullTracker(cfg)
    raise ValueError(f"unknown tracking source {source!r}")


def build_dispatcher(cfg: EngineConfig, dry_run: bool | None = None):
    from .control.dispatcher import ActionDispatcher

    if dry_run is None:
        dry_run = not cfg.control.enabled
    return ActionDispatcher(cfg=cfg, dry_run=dry_run)


def build_target_provider(cfg: EngineConfig, screen: tuple[int, int] = (1920, 1080)):
    """Target provider selection: static / uia / cv / none."""
    from .targets.providers import cv_provider, default_static_targets, uia_provider
    from .targets.target_model import StaticTargetProvider

    name = cfg.targets.provider
    if name == "uia":
        provider = uia_provider(screen=screen)
        # A live UIA failure must degrade to *no targets*, never to the synthetic
        # T0..T5 benchmark layout.  Synthetic targets are only legitimate in
        # explicit benchmark/replay configurations.
        return provider if provider is not None else StaticTargetProvider([])
    if name == "cv":
        provider = cv_provider(cfg, screen=screen)
        return provider if provider is not None else StaticTargetProvider([])
    if name == "none":
        return StaticTargetProvider([])
    # Static targets are intentionally only for benchmarks/replay.
    return StaticTargetProvider(default_static_targets())
