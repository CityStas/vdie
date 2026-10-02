"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_engine import build_dispatcher, build_target_provider, load_config
from gesture_engine.engine import GestureEngine
from gesture_engine.tracking.landmark_model import POSES, build_hand
from gesture_engine.tracking.synthetic import SyntheticTracker, get_scenario


@pytest.fixture()
def cfg():
    c = load_config()
    c.control.enabled = False
    c.tracking.backend = "synthetic"
    return c


@pytest.fixture()
def engine_factory(cfg):
    """Build an engine around a synthetic scenario, with OS control disabled."""

    def _make(scenario: str = "demo", overrides: dict | None = None, screen=(1920, 1080)) -> GestureEngine:
        c = load_config()
        c.control.enabled = False
        c.tracking.backend = "synthetic"
        if overrides:
            from gesture_engine.config import apply_overrides

            apply_overrides(c, overrides)
        tracker = SyntheticTracker(c, segments=get_scenario(scenario), loop=False)
        eng = GestureEngine(
            c,
            tracker,
            dispatcher=build_dispatcher(c, dry_run=True),
            target_provider=build_target_provider(c),
            screen=screen,
        )
        return eng

    return _make


def run_engine(engine: GestureEngine, duration: float, fps: int = 30):
    reports = []
    engine.start()
    try:
        for i in range(int(duration * fps)):
            report = engine.process_frame(None, i / fps, i)
            if report is None:
                break
            reports.append(report)
    finally:
        engine.stop()
    return reports


@pytest.fixture()
def hand_poses() -> dict[str, np.ndarray]:
    return {name: build_hand(pose) for name, pose in POSES.items()}
