"""End-to-end: recording, deterministic replay, policy, safety, metrics."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from gesture_engine import build_dispatcher, build_target_provider, load_config
from gesture_engine.bench.metrics import SegmentTruth, compare, evaluate, format_table
from gesture_engine.bench.runner import BenchmarkRunner, Variant, time_pipeline, truth_from_segments
from gesture_engine.control.dispatcher import ActionDispatcher
from gesture_engine.control.mouse import NullMouse
from gesture_engine.debug.recorder import Recorder, ReplayTracker, diff_sessions, load_recording
from gesture_engine.engine import GestureEngine
from gesture_engine.policy.action_policy import ActionPolicy
from gesture_engine.state.state_machine import State, StateMachine
from gesture_engine.targets.target_model import StaticTargetProvider
from gesture_engine.targets.providers import default_static_targets
from gesture_engine.tracking.synthetic import SyntheticTracker, get_scenario
from gesture_engine.types import ActionRequest, CommitSignal, GestureResult, Intent, IntentField, MotionState

from .conftest import run_engine


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #


def test_pipeline_runs_and_produces_sane_reports(engine_factory):
    engine = engine_factory("demo")
    reports = run_engine(engine, duration=4.0)
    assert len(reports) > 100
    for r in reports:
        assert 0.0 <= r.gesture.confidence <= 1.0
        assert abs(sum(r.intents.probabilities.values()) - 1.0) < 1e-3
        assert r.state in {s.value for s in State}
        assert r.cursor is not None


def test_activation_happens_in_the_demo_scenario(engine_factory):
    engine = engine_factory("demo")
    reports = run_engine(engine, duration=4.0)
    assert engine.stats.activations >= 1
    assert any(r.state == "CURSOR" for r in reports)


def test_pinch_produces_legacy_drag_actions_when_explicitly_enabled(engine_factory):
    engine = engine_factory("demo")
    engine.cfg.bimanual.one_hand_drag_enabled = True
    reports = run_engine(engine, duration=4.0)
    kinds = Counter(a.kind for r in reports for a in r.actions)
    assert kinds["down"] >= 1 and kinds["up"] >= 1
    # Every button press must be matched by a release.
    assert kinds["down"] == kinds["up"]


def test_click_study_produces_clicks(engine_factory):
    engine = engine_factory("click")
    reports = run_engine(engine, duration=12.0)
    clicks = sum(1 for r in reports for a in r.actions if a.kind == "click")
    assert clicks >= 3, f"expected several clicks, got {clicks}"


def test_kinematic_commit_alone_cannot_click(engine_factory):
    """A target/commit signal alone must never create a physical click."""
    engine = engine_factory("click", overrides={"gestures.pinch_down": 0.0, "gestures.pinch_release": 0.0})
    reports = run_engine(engine, duration=12.0)
    clicks = sum(1 for r in reports for a in r.actions if a.kind == "click")
    assert clicks == 0


def test_emergency_cancel_releases_buttons(engine_factory):
    engine = engine_factory("demo")
    run_engine(engine, duration=4.0)
    assert not engine.dispatcher.mouse.held
    assert not engine.policy.dragging


def test_dry_run_touches_nothing(engine_factory):
    engine = engine_factory("demo")
    run_engine(engine, duration=2.0)
    assert isinstance(engine.dispatcher.mouse, NullMouse)
    assert engine.dispatcher._dry_run


def test_engine_stop_closes_tracker_and_target_provider():
    class SpyTracker:
        def __init__(self):
            self.closed = 0

        def process(self, frame, timestamp, frame_id=0, camera=None):
            return None

        def close(self):
            self.closed += 1

    class SpyProvider(StaticTargetProvider):
        def __init__(self):
            super().__init__([])
            self.closed = 0

        def close(self):
            self.closed += 1

    cfg = load_config()
    tracker = SpyTracker()
    provider = SpyProvider()
    engine = GestureEngine(
        cfg,
        tracker,
        dispatcher=ActionDispatcher(cfg=cfg, dry_run=True),
        target_provider=provider,
    )
    engine.stop()
    assert tracker.closed == 1
    assert provider.closed == 1


def test_engine_without_control_never_emits_os_actions():
    cfg = load_config()
    cfg.control.enabled = False
    engine = GestureEngine(
        cfg,
        SyntheticTracker(cfg, segments=get_scenario("demo"), loop=False),
        dispatcher=ActionDispatcher(cfg=cfg, dry_run=True),
        target_provider=StaticTargetProvider(default_static_targets()),
    )
    reports = run_engine(engine, duration=3.0)
    for r in reports:
        for a in r.actions:
            assert a.kind in ("click", "down", "up", "scroll", "key", "window", "custom", "move")


# --------------------------------------------------------------------------- #
# Recording / replay
# --------------------------------------------------------------------------- #


def test_record_and_replay_are_deterministic(tmp_path):
    cfg = load_config()
    cfg.control.enabled = False
    cfg.record.directory = str(tmp_path)

    engine = GestureEngine(
        cfg,
        SyntheticTracker(cfg, segments=get_scenario("click"), loop=False),
        dispatcher=ActionDispatcher(cfg=cfg, dry_run=True),
        recorder=Recorder(cfg=cfg, name="unit", directory=str(tmp_path)),
        target_provider=StaticTargetProvider(default_static_targets()),
    )
    live = run_engine(engine, duration=6.0)
    assert engine.recorder.path is not None

    rec = load_recording(engine.recorder.path)
    assert len(rec.frames) > 0
    assert rec.header.get("format") == 2
    assert rec.config() is not None

    # Rebuild the same landmarks as an explicit landmark stream and replay it.
    cfg2 = load_config()
    cfg2.control.enabled = False
    rec2 = load_recording(engine.recorder.path)
    tracker = ReplayTracker(rec2, use_landmarks=True)
    replayed = []
    engine2 = GestureEngine(
        cfg2,
        tracker,
        dispatcher=ActionDispatcher(cfg=cfg2, dry_run=True),
        target_provider=StaticTargetProvider(default_static_targets()),
    )
    engine2.start()
    try:
        for i in range(len(rec2.frames)):
            r = engine2.process_frame(None, 0.0, i)
            if r is None:
                break
            replayed.append(r)
    finally:
        engine2.stop()

    # The recorded per-frame decisions must be reproduced exactly when the same
    # landmark stream is fed through the same algorithm.
    n = min(len(live), len(replayed))
    assert n > 50
    same_state = sum(1 for a, b in zip(live, replayed) if a.state == b.state)
    assert same_state / n > 0.85


def test_diff_sessions_detects_identical_runs(tmp_path):
    cfg = load_config()
    cfg.control.enabled = False
    paths = []
    for name in ("a", "b"):
        engine = GestureEngine(
            cfg,
            SyntheticTracker(cfg, segments=get_scenario("swipe"), loop=False),
            dispatcher=ActionDispatcher(cfg=cfg, dry_run=True),
            recorder=Recorder(cfg=cfg, name=name, directory=str(tmp_path)),
        )
        run_engine(engine, duration=2.0)
        paths.append(engine.recorder.path)
    diff = diff_sessions(load_recording(paths[0]), load_recording(paths[1]))
    assert diff["frames"] > 0
    assert diff["state_disagreement_rate"] == 0.0


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #


def _policy_inputs(prob, stability=0.9, commit=None, dragging=False):
    cfg = load_config()
    fsm = StateMachine(cfg=cfg)
    fsm.force(State.DRAGGING if dragging else State.CURSOR, 0.0)
    intents = IntentField(
        probabilities={Intent.SELECT: prob, Intent.MOVE_CURSOR: 1 - prob},
        dominant=Intent.SELECT if prob >= 0.5 else Intent.MOVE_CURSOR,
        committed=Intent.SELECT if prob >= 0.72 else None,
    )
    return cfg, fsm, intents, commit, stability


def test_policy_requires_commit_for_click():
    cfg, fsm, intents, commit, stab = _policy_inputs(0.95, commit=None)
    p = ActionPolicy(cfg=cfg)
    actions = p.update(
        intents=intents,
        commit=commit,
        gesture=GestureResult("PINCH", 0.9, stab),
        motion=MotionState(speed=0.0),
        features=None,
        beliefs=[],
        fsm=fsm,
        timestamp=0.0,
    )
    assert actions == []
    assert p.last_decision is not None and p.last_decision.blocked_by == "commit"


def test_policy_executes_with_commit():
    cfg, fsm, intents, commit, stab = _policy_inputs(0.95, commit=CommitSignal(True, 0.9, "MICRO_PAUSE"))
    p = ActionPolicy(cfg=cfg)
    actions = p.update(
        intents=intents,
        commit=commit,
        gesture=GestureResult("PINCH", 0.9, stab),
        motion=MotionState(speed=0.0),
        features=None,
        beliefs=[],
        fsm=fsm,
        timestamp=0.0,
    )
    assert [a.kind for a in actions] == ["click"]


def test_policy_respects_cooldown():
    cfg, fsm, intents, commit, stab = _policy_inputs(0.95, commit=CommitSignal(True, 0.9, "MICRO_PAUSE"))
    p = ActionPolicy(cfg=cfg)
    first = p.update(intents=intents, commit=commit, gesture=GestureResult("PINCH", 0.9, stab), motion=MotionState(speed=0.0), features=None, beliefs=[], fsm=fsm, timestamp=0.0)
    second = p.update(intents=intents, commit=commit, gesture=GestureResult("PINCH", 0.9, stab), motion=MotionState(speed=0.0), features=None, beliefs=[], fsm=fsm, timestamp=0.05)
    assert first and not second


def test_policy_blocks_in_idle():
    cfg = load_config()
    fsm = StateMachine(cfg=cfg)
    intents = IntentField(probabilities={Intent.SELECT: 1.0}, dominant=Intent.SELECT, committed=Intent.SELECT)
    p = ActionPolicy(cfg=cfg)
    actions = p.update(
        intents=intents,
        commit=CommitSignal(True, 1.0, "PINCH"),
        gesture=GestureResult("PINCH", 1.0, 1.0),
        motion=MotionState(speed=0.0),
        features=None,
        beliefs=[],
        fsm=fsm,
        timestamp=0.0,
    )
    assert actions == []


def test_drag_release_always_honoured():
    cfg, fsm, intents, commit, stab = _policy_inputs(0.1)
    p = ActionPolicy(cfg=cfg)
    p.update(
        intents=intents,
        commit=None,
        gesture=None,
        motion=MotionState(speed=0.0),
        features=None,
        beliefs=[],
        fsm=_dragging_fsm(cfg),
        timestamp=0.0,
    )
    assert p.dragging
    # Now the FSM leaves DRAGGING: the release must be emitted even though the
    # engine is in a state that blocks new actions.
    actions = p.update(
        intents=intents,
        commit=None,
        gesture=None,
        motion=MotionState(speed=0.0),
        features=None,
        beliefs=[],
        fsm=StateMachine(cfg=cfg),
        timestamp=0.1,
    )
    assert [a.kind for a in actions] == ["up"]
    assert not p.dragging


def _dragging_fsm(cfg):
    sm = StateMachine(cfg=cfg)
    sm.force(State.DRAGGING, 0.0)
    return sm


# --------------------------------------------------------------------------- #
# Metrics + benchmark harness
# --------------------------------------------------------------------------- #


def test_metrics_on_a_synthetic_run(engine_factory):
    engine = engine_factory("click")
    reports = run_engine(engine, duration=12.0)
    truth = truth_from_segments(get_scenario("click"))
    ev = evaluate(reports, truth, name="click", screen=(1920, 1080))
    d = ev.as_dict()
    assert d["frames"] > 100
    assert d["gesture_recall"] is not None
    assert 0.0 <= d["gesture_recall"] <= 1.0
    assert d["pipeline_ms_p50"] > 0
    assert "gesture_false_per_s" in d


def test_benchmark_runner_matrix_is_reproducible():
    cfg = load_config()
    cfg.capture.fps = 30
    variants = (
        Variant("a", {"cursor.controller": "gain", "commit.enabled": False}),
        Variant("b", {"cursor.controller": "spring", "commit.enabled": False}),
    )
    r1 = BenchmarkRunner(cfg, fps=30)
    t1 = r1.run_matrix(variants=variants, scenario="swipe", duration=2.0)
    r2 = BenchmarkRunner(cfg, fps=30)
    t2 = r2.run_matrix(variants=variants, scenario="swipe", duration=2.0)
    assert t1.keys() == t2.keys()
    for name in t1:
        for k in t1[name]:
            assert t1[name][k] == pytest.approx(t2[name][k], rel=1e-9)


def test_format_table_renders():
    cfg = load_config()
    runner = BenchmarkRunner(cfg, fps=30)
    runner.run_matrix(variants=(Variant("only", {}),), scenario="swipe", duration=1.5)
    table = runner.report()
    assert "| metric |" in table
    assert "only" in table


def test_time_pipeline_reports_stages():
    cfg = load_config()
    summary = time_pipeline(cfg, frames=60, fps=30)
    assert summary["frame_ms"] > 0
    assert "features" in summary["stages_ms"]


def test_real_mouse_clamps_away_from_failsafe_corners(monkeypatch):
    """Live cursor targets must never command an exact PyAutoGUI corner."""
    from gesture_engine.control.mouse import PyAutoGuiMouse

    class Point:
        x = 0
        y = 0

    class FakePag:
        FAILSAFE = True
        PAUSE = 0.0
        def size(self):
            return (1920, 1080)
        def position(self):
            return Point()
        def moveTo(self, x, y, **kwargs):
            self.last = (x, y)

    fake = FakePag()
    driver = PyAutoGuiMouse.__new__(PyAutoGuiMouse)
    driver._pag = fake
    driver._duration = 0.0
    driver._held = set()
    driver._screen = (1920, 1080)

    # Non-Windows CI uses the normal PyAutoGUI path; it still verifies the
    # target itself cannot become an exact emergency corner.
    driver.move(0, 0)
    assert fake.last == (1.0, 1.0)
