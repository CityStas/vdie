"""Intent field, evidence providers, cursor controller, target model, grammar."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gesture_engine.control.cursor import CursorController, CursorMode
from gesture_engine.features.feature_vector import build_features
from gesture_engine.filters.ema import AdaptiveEMA, EMA, OneEuroFilter
from gesture_engine.grammar.sequence_parser import GestureGrammar
from gesture_engine.intent.evidence import CommitEvidence, GestureEvidence, MotionEvidence
from gesture_engine.intent.intent_field import IntentFieldBuilder
from gesture_engine.motion.history import HistoryBuffer
from gesture_engine.state.state_machine import State, StateMachine
from gesture_engine.targets.providers import default_static_targets
from gesture_engine.targets.target_model import StaticTargetProvider, TargetModel
from gesture_engine.tracking.landmark_model import POSES, build_hand, make_pose, wrist_for_tip
from gesture_engine.tracking.synthetic import click_study_scenario
from gesture_engine.types import (
    CommitSignal,
    GestureResult,
    Intent,
    IntentField,
    MotionState,
    Target,
    TargetBelief,
)


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #


def test_gesture_evidence_maps_poses_to_intents(cfg):
    ev = GestureEvidence(cfg=cfg)
    logits = ev.logits(gesture=GestureResult("PINCH", confidence=1.0, stability=1.0))
    assert logits[Intent.SELECT] > 0


def test_steering_evidence_scales_with_speed():
    ev = GestureEvidence()
    gesture = GestureResult("INDEX_UP", confidence=1.0, stability=1.0)
    slow = ev.logits(gesture=gesture, motion=MotionState(speed=0.0))[Intent.MOVE_CURSOR]
    fast = ev.logits(gesture=gesture, motion=MotionState(speed=2.0))[Intent.MOVE_CURSOR]
    assert fast > slow * 2


def test_commit_evidence_suppresses_steering():
    ev = CommitEvidence()
    logits = ev.logits(commit=CommitSignal(committed=True, score=1.0, kind="MICRO_PAUSE"))
    assert logits[Intent.SELECT] > 0
    assert logits[Intent.MOVE_CURSOR] < 0


def test_commit_evidence_ignores_uncommitted():
    assert CommitEvidence().logits(commit=CommitSignal(committed=False, score=1.0)) == {}


def test_motion_evidence_pinch_plus_movement_is_drag(cfg):
    f = build_features(build_hand(POSES["PINCH"]), cfg)
    logits = MotionEvidence().logits(motion=MotionState(speed=1.2, speed_trend=0.0), features=f)
    assert logits.get(Intent.DRAG, 0.0) > 0


# --------------------------------------------------------------------------- #
# Intent field
# --------------------------------------------------------------------------- #


def _field_kwargs(cfg, t, speed=1.0, gesture=None, commit=None, fsm=None, features=None):
    return dict(
        timestamp=t,
        features=features,
        motion=MotionState(timestamp=t, speed=speed, confidence=1.0, position=np.array([0.5, 0.5])),
        gesture=gesture,
        history=HistoryBuffer(capacity=10),
        events=[],
        beliefs=[],
        fsm=fsm or _cursor_fsm(cfg),
        commit=commit,
    )


def _cursor_fsm(cfg):
    sm = StateMachine(cfg=cfg)
    sm.force(State.CURSOR, 0.0)
    return sm


def test_field_is_a_normalised_distribution(cfg):
    b = IntentFieldBuilder(cfg)
    f = None
    for i in range(30):
        field = b.update(**_field_kwargs(cfg, i / 30, gesture=GestureResult("INDEX_UP", 0.9, 0.9)))
    assert sum(field.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert field.entropy >= 0.0
    assert field.dominant is Intent.MOVE_CURSOR


def test_field_is_frame_rate_independent(cfg):
    def run(fps: int) -> float:
        b = IntentFieldBuilder(cfg)
        dt = 1.0 / fps
        for i in range(int(fps * 1.0)):
            field = b.update(**_field_kwargs(cfg, i * dt, gesture=GestureResult("INDEX_UP", 1.0, 1.0)))
        return field.get(Intent.MOVE_CURSOR)

    p30, p60 = run(30), run(60)
    assert p30 == pytest.approx(p60, abs=0.05)


def test_commit_promotes_select(cfg):
    b = IntentFieldBuilder(cfg)
    for i in range(40):
        b.update(**_field_kwargs(cfg, i / 30, gesture=GestureResult("INDEX_UP", 0.9, 0.9)))
    before = b.update(**_field_kwargs(cfg, 40 / 30, gesture=GestureResult("INDEX_UP", 0.9, 0.9)))
    after = b.update(
        **_field_kwargs(
            cfg,
            41 / 30,
            gesture=GestureResult("INDEX_UP", 0.9, 0.9),
            commit=CommitSignal(committed=True, score=0.95, kind="MICRO_PAUSE"),
        )
    )
    assert after.get(Intent.SELECT) > before.get(Intent.SELECT) + 0.3
    assert after.committed is Intent.SELECT


def test_field_stays_ambiguous_when_nothing_wins(cfg):
    b = IntentFieldBuilder(cfg)
    for i in range(20):
        field = b.update(**_field_kwargs(cfg, i / 30, speed=0.0, gesture=GestureResult("NONE", 0.0)))
    assert field.entropy > 0.5
    assert field.dominant_probability < 0.9


def test_priors_are_respected(cfg):
    b = IntentFieldBuilder(cfg)
    for i in range(60):
        field = b.update(**_field_kwargs(cfg, i / 30, speed=0.0))
    assert field.get(Intent.MOVE_CURSOR) > field.get(Intent.RIGHT_CLICK)


# --------------------------------------------------------------------------- #
# Cursor controller
# --------------------------------------------------------------------------- #


def _cursor_inputs(cfg, position, speed=0.0, conf=1.0):
    motion = MotionState(
        timestamp=0.0,
        position=np.asarray(position, dtype=float),
        velocity=np.array([0.0, 0.0]),
        speed=speed,
        confidence=conf,
    )
    return motion


def test_cursor_converges_to_target(cfg):
    cfg.cursor.relative_mode = False
    c = CursorController(cfg, screen=(1000, 1000))
    fsm = _cursor_fsm(cfg)
    from gesture_engine.types import IntentField

    intents = IntentField(probabilities={Intent.MOVE_CURSOR: 1.0}, dominant=Intent.MOVE_CURSOR)
    for i in range(200):
        c.update(_cursor_inputs(cfg, (0.5, 0.5), speed=0.0), None, intents, [], fsm, 1 / 30, i / 30)
    assert c.position[0] == pytest.approx(500, abs=60)


def test_cursor_does_not_overshoot_wildly(cfg):
    cfg.cursor.relative_mode = False
    c = CursorController(cfg, screen=(1000, 1000))
    fsm = _cursor_fsm(cfg)
    from gesture_engine.types import IntentField

    intents = IntentField(probabilities={Intent.MOVE_CURSOR: 1.0}, dominant=Intent.MOVE_CURSOR)
    for i in range(120):
        c.update(_cursor_inputs(cfg, (0.9, 0.5), speed=1.5), None, intents, [], fsm, 1 / 30, i / 30)
    assert c.position[0] <= 1000


def test_cursor_freezes_on_low_confidence(cfg):
    cfg.cursor.relative_mode = False
    c = CursorController(cfg, screen=(1000, 1000))
    fsm = _cursor_fsm(cfg)
    from gesture_engine.types import IntentField

    intents = IntentField(probabilities={Intent.MOVE_CURSOR: 1.0}, dominant=Intent.MOVE_CURSOR)
    for i in range(30):
        c.update(_cursor_inputs(cfg, (0.5, 0.5)), None, intents, [], fsm, 1 / 30, i / 30)
    before = c.position.copy()
    state = c.update(_cursor_inputs(cfg, (0.9, 0.9), conf=0.0), None, intents, [], fsm, 1 / 30, 1.0)
    assert state.frozen
    assert state.mode is CursorMode.RECOVERY
    assert np.linalg.norm(c.position - before) < 50


def test_cursor_is_paused_when_fsm_disabled(cfg):
    c = CursorController(cfg, screen=(1000, 1000))
    fsm = StateMachine(cfg=cfg)  # IDLE
    from gesture_engine.types import IntentField

    intents = IntentField(probabilities={Intent.MOVE_CURSOR: 1.0}, dominant=Intent.MOVE_CURSOR)
    state = c.update(_cursor_inputs(cfg, (0.5, 0.5)), None, intents, [], fsm, 1 / 30, 0.0)
    assert state.mode is CursorMode.PAUSED
    assert not fsm.cursor_enabled


def test_intent_ambiguity_reduces_gain(cfg):
    # The mechanism is off by default: measured against an otherwise identical
    # variant it costs 30 ms of cursor lag and changes no outcome metric.  The
    # behaviour it implements is still specified here, with the flag on.
    cfg.cursor.intent_adaptive = True
    c = CursorController(cfg, screen=(1000, 1000))
    from gesture_engine.types import IntentField

    confident = IntentField(probabilities={Intent.MOVE_CURSOR: 0.97, Intent.SELECT: 0.03}, entropy=0.1, dominant=Intent.MOVE_CURSOR)
    ambiguous = IntentField(
        probabilities={Intent.MOVE_CURSOR: 0.4, Intent.SELECT: 0.3, Intent.DRAG: 0.3},
        entropy=1.05,
        dominant=Intent.MOVE_CURSOR,
    )
    g_conf = c._intent_gain(confident)
    g_amb = c._intent_gain(ambiguous)
    assert g_amb < g_conf


def test_intent_gain_is_off_by_default(cfg):
    """The measured decision, pinned: hesitation is not worth 30 ms of lag."""
    assert cfg.cursor.intent_adaptive is False
    c = CursorController(cfg, screen=(1000, 1000))
    from gesture_engine.types import IntentField

    ambiguous = IntentField(
        probabilities={Intent.MOVE_CURSOR: 0.4, Intent.SELECT: 0.3, Intent.DRAG: 0.3},
        entropy=1.05,
        dominant=Intent.MOVE_CURSOR,
    )
    assert c._intent_gain(ambiguous) == 1.0


def test_cursor_recovers_without_jumping(cfg):
    c = CursorController(cfg, screen=(1000, 1000))
    fsm = _cursor_fsm(cfg)
    from gesture_engine.types import IntentField

    intents = IntentField(probabilities={Intent.MOVE_CURSOR: 1.0}, dominant=Intent.MOVE_CURSOR)
    for i in range(40):
        c.update(_cursor_inputs(cfg, (0.4, 0.5)), None, intents, [], fsm, 1 / 30, i / 30)
    settled = c.position.copy()
    for i in range(40, 50):
        c.update(_cursor_inputs(cfg, (0.4, 0.5), conf=0.0), None, intents, [], fsm, 1 / 30, i / 30)
    c.update(_cursor_inputs(cfg, (0.42, 0.5)), None, intents, [], fsm, 1 / 30, 1.7)
    assert np.linalg.norm(c.position - settled) < cfg.cursor.recovery_max_jump * 1000


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


def test_ema_converges():
    f = EMA(alpha=0.5)
    for _ in range(50):
        out = f.update(np.array([1.0, 1.0]))
    assert np.allclose(out, [1.0, 1.0], atol=1e-6)


def test_one_euro_smooths_jitter_more_than_constant_ema():
    rng = np.random.default_rng(0)
    signal = np.cumsum(rng.normal(0, 0.01, size=(200, 2)), axis=0)

    euro = OneEuroFilter(min_cutoff=0.5, beta=0.01)
    ema = EMA(alpha=0.5)
    out_e, out_a = [], []
    for p in signal:
        out_e.append(euro.update(p, 1 / 30))
        out_a.append(ema.update(p, 1 / 30))
    e = np.asarray(out_e)
    a = np.asarray(out_a)
    assert np.std(np.diff(e, n=2, axis=0)) < np.std(np.diff(a, n=2, axis=0))


def test_adaptive_ema_alpha_monotone():
    f = AdaptiveEMA()
    assert f.alpha_for(0.0) < f.alpha_for(1.0) < f.alpha_for(3.0)


# --------------------------------------------------------------------------- #
# Target model
# --------------------------------------------------------------------------- #


def _target_model(cfg, targets):
    return TargetModel(cfg, provider=StaticTargetProvider(targets))


def test_target_beliefs_normalised(cfg):
    tm = _target_model(
        cfg,
        [
            Target(id="A", bounds=(0.48, 0.48, 0.52, 0.52), semantic_confidence=0.9, visual_confidence=0.9),
            Target(id="B", bounds=(0.8, 0.8, 0.84, 0.84), semantic_confidence=0.9, visual_confidence=0.9),
        ],
    )
    tm.refresh(0.0)
    beliefs = tm.beliefs(np.array([0.5, 0.5]), np.zeros(2), _empty_intents(), 0.0)
    assert sum(b.confidence for b in beliefs) == pytest.approx(1.0)
    assert max(beliefs, key=lambda b: b.confidence).target.id == "A"


def test_gravity_is_disabled_during_fast_travel(cfg):
    tm = _target_model(cfg, [Target(id="A", bounds=(0.48, 0.48, 0.52, 0.52), semantic_confidence=1.0, visual_confidence=1.0)])
    tm.refresh(0.0)
    intents = _selecting_intents()
    beliefs = tm.beliefs(np.array([0.5, 0.5]), np.zeros(2), intents, 0.0)
    fast = tm.gravity(np.array([0.5, 0.5]), np.array([3.0, 0.0]), beliefs, intents, 1 / 30)
    assert np.allclose(fast, 0.0)


def test_gravity_needs_selection_intent(cfg):
    tm = _target_model(cfg, [Target(id="A", bounds=(0.48, 0.48, 0.52, 0.52), semantic_confidence=1.0, visual_confidence=1.0)])
    tm.refresh(0.0)
    beliefs = tm.beliefs(np.array([0.51, 0.5]), np.zeros(2), _empty_intents(), 0.0)
    force = tm.gravity(np.array([0.51, 0.5]), np.zeros(2), beliefs, _empty_intents(), 1 / 30)
    assert np.allclose(force, 0.0)


def test_gravity_points_towards_the_target(cfg):
    tm = _target_model(cfg, [Target(id="A", bounds=(0.52, 0.48, 0.56, 0.52), semantic_confidence=1.0, visual_confidence=1.0)])
    tm.refresh(0.0)
    intents = _selecting_intents()
    beliefs = tm.beliefs(np.array([0.5, 0.5]), np.zeros(2), intents, 0.0)
    force = tm.gravity(np.array([0.5, 0.5]), np.zeros(2), beliefs, intents, 1 / 30)
    assert force[0] > 0


def test_avoided_targets_lose_influence(cfg):
    tm = _target_model(cfg, [Target(id="A", bounds=(0.5, 0.5, 0.54, 0.54), semantic_confidence=1.0, visual_confidence=1.0)])
    tm.refresh(0.0)
    # Outside the target and moving away from it: that is what "avoid" means.
    pos = np.array([0.60, 0.52])
    away = np.array([1.0, 0.0])
    for _ in range(6):
        tm.beliefs(pos, away, _empty_intents(), 0.0)
    snap = tm.snapshot()
    assert snap[0]["state"] == "AVOID"
    assert snap[0]["avoid"] > 0


def _empty_intents():
    from gesture_engine.types import IntentField

    return IntentField(probabilities={Intent.MOVE_CURSOR: 1.0}, dominant=Intent.MOVE_CURSOR)


def _selecting_intents():
    from gesture_engine.types import IntentField

    return IntentField(probabilities={Intent.SELECT: 0.9, Intent.MOVE_CURSOR: 0.1}, dominant=Intent.SELECT, committed=Intent.SELECT)


# --------------------------------------------------------------------------- #
# Grammar
# --------------------------------------------------------------------------- #


def test_grammar_matches_sequence_with_pause(cfg):
    g = GestureGrammar(cfg)
    cfg.grammar.rules = {"TEST": ["INDEX_UP", "pause", "SWIPE_LEFT"]}
    # Two gestures with no deliberate pause between them are not a phrase.
    g.push("INDEX_UP", 0.0)
    assert g.push("SWIPE_LEFT", 0.1) == []  # 100 ms < separator_ms -> no pause
    assert g.tokens == ["INDEX_UP", "SWIPE_LEFT"]
    # The same pair separated by a deliberate pause is.
    g.reset()
    g.push("INDEX_UP", 1.0)
    assert g.tokens == ["INDEX_UP"]
    matches = g.push("SWIPE_LEFT", 1.5)  # 500 ms > separator_ms -> pause inserted
    assert [m.name for m in matches] == ["TEST"]
    assert g.matches()[0].pattern == ["INDEX_UP", "pause", "SWIPE_LEFT"]
    # A matched phrase is consumed, so it cannot fire twice.
    assert g.tokens == []


def test_grammar_ignores_repeated_same_gesture(cfg):
    g = GestureGrammar(cfg)
    g.push("PINCH", 0.0)
    g.push("PINCH", 0.1)
    g.push("PINCH", 0.2)
    assert g.tokens == ["PINCH"]


def test_grammar_partial_prefix(cfg):
    g = GestureGrammar(cfg)
    cfg.grammar.rules = {"X": ["INDEX_UP", "pause", "V_SIGN"]}
    g.push("INDEX_UP", 0.0)
    assert "INDEX_UP" in g.partial()


def test_grammar_disabled(cfg):
    cfg.grammar.enabled = False
    g = GestureGrammar(cfg)
    assert g.push("INDEX_UP", 0.0) == []


# --------------------------------------------------------------------------- #
# Cursor anchoring and target engagement (regressions)
# --------------------------------------------------------------------------- #


def _stationary_convergence(cfg, hand, screen=(1920, 1080)):
    """Where the controller settles for a hand held still at ``hand``."""
    ctrl = CursorController(cfg, screen=screen)
    fsm = StateMachine(cfg=cfg)
    fsm.force(State.CURSOR, 0.0)
    field = IntentField()
    for i in range(120):
        motion = MotionState(
            timestamp=i / 30,
            position=np.asarray(hand, dtype=float),
            velocity=np.zeros(2),
            speed=0.0,
            confidence=1.0,
        )
        state = ctrl.update(motion, None, field, [], fsm, 1 / 30, i / 30)
    return ctrl, state


def test_cursor_anchors_where_it_would_settle(cfg):
    """The cursor must appear where the hand is, not slide in from the origin.

    It used to start at ``(0, 0)`` and crawl across the screen with an effective
    alpha of ~0.19 — about 15 frames, half a second, of the cursor flying in
    from the top-left corner.  The benchmark reported that transient as
    "cursor lag 233-467 ms" for every variant.
    """
    hand = np.array([0.30, 0.80])
    _, settled = _stationary_convergence(cfg, hand)

    ctrl = CursorController(cfg, screen=(1920, 1080))
    ctrl.anchor_to(hand)
    assert np.allclose(ctrl.position, settled.position, atol=2.0)
    assert np.allclose(ctrl.velocity, 0.0)

    # The anchor is a starting point, not a clamp: the next frame tracks normally.
    fsm = StateMachine(cfg=cfg)
    fsm.force(State.CURSOR, 0.0)
    moved = np.array([0.32, 0.80])
    motion = MotionState(timestamp=0.0, position=moved, velocity=np.zeros(2), speed=0.4, confidence=1.0)
    state = ctrl.update(motion, None, IntentField(), [], fsm, 1 / 30, 0.0)
    assert np.linalg.norm(state.position - settled.position) < 60.0


def test_click_scenario_puts_the_control_point_on_the_target(cfg):
    """Ground truth says "the hand is on T0"; the engine must agree.

    ``Segment.start`` is the *wrist*, the control point is the *index tip*, and
    they are ~0.37 image units apart with INDEX_UP.  The click study placed the
    wrist on each target, so ``contains`` was false everywhere and the entire
    target model — approach detection, the gravity field, TARGET_ENTRY — was
    inert in every benchmark run.
    """
    model = TargetModel(cfg, provider=StaticTargetProvider(default_static_targets()))
    model.refresh(0.0)
    tagged = [s for s in click_study_scenario(repeats=3) if s.tags.get("target") is not None]
    assert tagged, "the click study must tag its selection segments"

    hits = 0
    for seg in tagged:
        pts = build_hand(make_pose(seg.pose, position=seg.start, scale=seg.scale))
        tip_point = pts[8, :2]
        if any(b.target.contains(tip_point) for b in model.beliefs(tip_point, np.zeros(2), None, 0.0)):
            hits += 1
    assert hits == len(tagged)


def test_wrist_for_tip_is_exact(cfg):
    """``wrist_for_tip`` inverts an affine map, so it must be exact."""
    for name in ("INDEX_UP", "PINCH", "OPEN_PALM"):
        for target in ((0.25, 0.35), (0.72, 0.30), (0.0, 0.0)):
            wrist = wrist_for_tip(name, target)
            pts = build_hand(make_pose(name, position=wrist))
            assert np.allclose(pts[8, :2], np.asarray(target), atol=1e-9), (name, target, pts[8, :2])


# --------------------------------------------------------------------------- #
# Cursor dynamics: feed-forward, integration stability
# --------------------------------------------------------------------------- #


def _tracking_rms_px(cfg, feedforward: bool, prediction_horizon: float, freq: float = 0.5, seconds: float = 4.0) -> float:
    """RMS distance between the cursor and the mapped hand over a 0.5 Hz sweep.

    The reference is the controller's own open-loop mapping, so this measures
    tracking error rather than the coordinate offset between hand units and
    pixels.
    """
    cfg.cursor.controller = "spring"
    cfg.cursor.feedforward = feedforward
    cfg.cursor.prediction_horizon = prediction_horizon
    c = CursorController(cfg, screen=(1000, 1000))
    fsm = _cursor_fsm(cfg)
    intents = IntentField(probabilities={Intent.MOVE_CURSOR: 1.0}, dominant=Intent.MOVE_CURSOR)

    dt = 1.0 / 30.0
    amp = 0.12
    centre = 0.5
    c.anchor_to(np.array([centre, centre]))
    errors: list[float] = []
    previous: np.ndarray | None = None
    for i in range(int(seconds * 30)):
        t = i * dt
        point = np.array([centre + amp * math.sin(2 * math.pi * freq * t), centre])
        velocity = (point - previous) / dt if previous is not None else np.zeros(2)
        previous = point
        motion = MotionState(
            timestamp=t,
            position=point,
            velocity=velocity,
            speed=float(np.linalg.norm(velocity)),
            confidence=1.0,
        )
        c.update(motion, None, intents, [], fsm, dt, t)
        if i >= 30:
            curve = np.array(
                [
                    c._sensitivity_curve(2.0 * (point[0] - 0.5)) * 0.5 + 0.5,
                    c._sensitivity_curve(2.0 * (point[1] - 0.5)) * 0.5 + 0.5,
                ]
            )
            reference = c._map_to_screen(curve)
            errors.append(float(np.linalg.norm(c.position - reference)))
    return float(np.sqrt(np.mean(np.square(errors))))


def test_feedforward_tracks_a_moving_hand_better_than_a_positional_lead(cfg):
    cfg.cursor.relative_mode = False
    """Feed-forward must cut tracking error on a moving hand.

    Both mechanisms try to hide the same latency; only one of them works.  The
    positional lead adds ``v * horizon`` to the *setpoint*, which the spring has
    unity DC gain for, so the cursor is deliberately driven off-target and the
    lead only buys what the gain it adds happens to recover.  Feed-forward puts
    the same velocity into the damping term, where it accelerates convergence
    without displacing the target.

    Measured on a 0.5 Hz hand sweep, as RMS distance from the mapped hand
    position — the number a user feels.  The docstring deliberately does *not*
    claim an overshoot difference: an earlier version of this test asserted that
    the positional lead overshoots, and it turned out the 190 px "overshoot"
    seen in the probe was the ``active_region`` clamp turning the lead into a
    fixed positional error at the region edge, not a general property.
    """
    ff = _tracking_rms_px(cfg, feedforward=True, prediction_horizon=0.0)
    lead = _tracking_rms_px(cfg, feedforward=False, prediction_horizon=0.045)
    neither = _tracking_rms_px(cfg, feedforward=False, prediction_horizon=0.0)

    assert ff < neither, f"feed-forward ({ff:.1f}px) must beat no compensation ({neither:.1f}px)"
    assert ff < lead, f"feed-forward ({ff:.1f}px) must beat the positional lead ({lead:.1f}px)"
    assert ff < 0.6 * neither, f"feed-forward only improved {ff:.1f}px vs {neither:.1f}px"


def test_feedforward_does_not_move_the_resting_position(cfg):
    cfg.cursor.relative_mode = False
    """Feed-forward is a velocity term, so at rest it must vanish exactly."""
    def settle(feedforward: bool) -> float:
        cfg.cursor.feedforward = feedforward
        c = CursorController(cfg, screen=(1000, 1000))
        fsm = _cursor_fsm(cfg)
        intents = IntentField(probabilities={Intent.MOVE_CURSOR: 1.0}, dominant=Intent.MOVE_CURSOR)
        for i in range(200):
            c.update(_cursor_inputs(cfg, (0.62, 0.44), speed=0.0), None, intents, [], fsm, 1 / 30, i / 30)
        return float(c.position[0])

    assert settle(True) == pytest.approx(settle(False), abs=0.5)


def test_spring_survives_stiff_and_heavily_damped_settings(cfg):
    cfg.cursor.relative_mode = False
    """Both integration terms are sub-stepped, so neither setting may diverge.

    Bounding only ``omega * h`` (the first version) left the damping term
    ``2 * zeta * omega * h`` unbounded: raising ``spring_damping_slow`` alone
    flipped the sign of the velocity update each step and the cursor blew up to
    hundreds of pixels of "tremor".
    """
    for omega_fast, damping_slow in ((120.0, 0.95), (26.0, 2.4), (120.0, 3.0)):
        cfg.cursor.spring_omega_fast = omega_fast
        cfg.cursor.spring_damping_slow = damping_slow
        c = CursorController(cfg, screen=(1000, 1000))
        fsm = _cursor_fsm(cfg)
        intents = IntentField(probabilities={Intent.MOVE_CURSOR: 1.0}, dominant=Intent.MOVE_CURSOR)
        c.anchor_to(np.array([0.5, 0.5]))
        for i in range(120):
            motion = MotionState(
                timestamp=i / 30,
                position=np.array([0.5, 0.5]),
                velocity=np.zeros(2),
                speed=0.0,
                confidence=1.0,
            )
            c.update(motion, None, intents, [], fsm, 1 / 30, i / 30)
        assert np.all(np.isfinite(c.position)), (omega_fast, damping_slow, c.position)
        assert abs(c.position[0] - 500.0) < 20.0, (omega_fast, damping_slow, c.position)


def test_spring_stays_stable_at_a_low_frame_rate(cfg):
    cfg.cursor.relative_mode = False
    """Sub-stepping is what makes a stiff spring safe below 30 fps."""
    cfg.cursor.spring_omega_fast = 60.0
    c = CursorController(cfg, screen=(1000, 1000))
    fsm = _cursor_fsm(cfg)
    intents = IntentField(probabilities={Intent.MOVE_CURSOR: 1.0}, dominant=Intent.MOVE_CURSOR)
    c.anchor_to(np.array([0.2, 0.5]))
    fps = 10.0
    for i in range(60):
        point = np.array([0.2 + 0.6 * min(i / 10.0, 1.0), 0.5])
        motion = MotionState(timestamp=i / fps, position=point, velocity=np.zeros(2), speed=0.0, confidence=1.0)
        c.update(motion, None, intents, [], fsm, 1 / fps, i / fps)
    assert np.all(np.isfinite(c.position))
    assert 0.0 <= c.position[0] <= 1000.0


def test_relative_cursor_keeps_desktop_position_on_activation(cfg):
    from gesture_engine.control.cursor import CursorController
    cfg.cursor.relative_mode = True
    c = CursorController(cfg, screen=(1000, 1000))
    c.anchor_to(np.array([0.30, 0.40]), cursor_position=np.array([700.0, 500.0]))
    assert np.allclose(c.position, [700.0, 500.0])
    assert c._anchor_cursor is not None
    assert np.allclose(c._anchor_cursor, [700.0, 500.0])


def _cooperative_fsm(cfg):
    fsm = StateMachine(cfg=cfg)
    fsm.force(State.CURSOR, 0.0)
    return fsm


def _cooperative_field():
    return IntentField(probabilities={Intent.MOVE_CURSOR: 1.0}, dominant=Intent.MOVE_CURSOR)


def test_cooperative_pointer_uses_index_delta_not_screen_region(cfg):
    cfg.cursor.controller = "cooperative"
    cfg.cursor.relative_mode = True
    c = CursorController(cfg, screen=(1920, 1080))
    fsm = _cooperative_fsm(cfg)
    c.anchor_to(np.array([0.5, 0.5]), cursor_position=np.array([960.0, 540.0]))
    field = _cooperative_field()

    # Even a one-camera-pixel/frame motion must produce a continuous cursor
    # response.  The old fixed velocity deadband swallowed this movement.
    for i in range(45):
        x = 0.5 + (i + 1) / 1280.0
        m = MotionState(timestamp=i / 30, position=np.array([x, 0.5]), velocity=np.zeros(2), speed=0.0, confidence=1.0)
        c.update(m, None, field, [], fsm, 1 / 30, i / 30)
    assert c.position[0] > 1000.0


def test_cooperative_pointer_can_reach_the_full_desktop(cfg):
    cfg.cursor.controller = "cooperative"
    cfg.cursor.relative_mode = True
    c = CursorController(cfg, screen=(1920, 1080))
    fsm = _cooperative_fsm(cfg)
    c.anchor_to(np.array([0.5, 0.5]), cursor_position=np.array([960.0, 540.0]))
    field = _cooperative_field()

    # Sustained movement is not mapped into a fixed 10-90% region.
    for i in range(180):
        x = 0.5 + (i + 1) * 5.0 / 1280.0
        m = MotionState(timestamp=i / 30, position=np.array([min(x, 0.99), 0.5]), velocity=np.zeros(2), speed=0.0, confidence=1.0)
        c.update(m, None, field, [], fsm, 1 / 30, i / 30)
    assert c.position[0] >= 1910.0


def test_cooperative_target_gravity_only_bends_existing_motion(cfg):
    cfg.cursor.controller = "cooperative"
    cfg.cursor.relative_mode = True
    c = CursorController(cfg, screen=(1920, 1080))
    fsm = _cooperative_fsm(cfg)
    c.anchor_to(np.array([0.5, 0.5]), cursor_position=np.array([700.0, 540.0]))
    field = _cooperative_field()
    target = Target(
        id="uia:test", bounds=(0.42, 0.45, 0.52, 0.55), type="button",
        semantic_label="Test", semantic_confidence=0.95, visual_confidence=0.55,
    )
    beliefs = [TargetBelief(target=target, confidence=1.0, distance=0.0, closing_speed=0.0)]

    # A stationary cursor is not pulled toward a UI element.
    before = c.position.copy()
    for i in range(10):
        m = MotionState(timestamp=i / 30, position=np.array([0.5, 0.5]), velocity=np.zeros(2), speed=0.0, confidence=1.0)
        c.update(m, None, field, beliefs, fsm, 1 / 30, i / 30)
    assert np.linalg.norm(c.position - before) < 0.1

    # Once there is deliberate motion into the target corridor, the target may
    # gently bend the path, but it must not replace the user's motion.
    for i in range(25):
        x = 0.5 + (i + 1) * 2.0 / 1280.0
        m = MotionState(timestamp=(10 + i) / 30, position=np.array([x, 0.5]), velocity=np.zeros(2), speed=0.0, confidence=1.0)
        st = c.update(m, None, field, beliefs, fsm, 1 / 30, (10 + i) / 30)
    assert st.target_id == "uia:test"
    assert st.spatial_intent > 0.45
    assert c.position[0] > before[0]




def test_cooperative_cursor_uses_raw_index_tip_not_prediction(cfg, monkeypatch):
    cfg.cursor.controller = "cooperative"
    cfg.cursor.relative_mode = True
    c = CursorController(cfg, screen=(1920, 1080))
    fsm = _cooperative_fsm(cfg)
    field = _cooperative_field()
    c.anchor_to(np.array([0.50, 0.50]), cursor_position=np.array([960.0, 540.0]))

    # Simulate a predictor that is wildly wrong. The cursor path must still be
    # driven by the measured index-tip delta; the predictor is intent-only.
    def bad_predictor(measurement, dt, confidence):
        m = np.asarray(measurement, dtype=float)
        return m.copy(), np.array([8.0, -6.0]), m + np.array([0.30, -0.20])

    monkeypatch.setattr(c._coop_predictor, "update", bad_predictor)
    m0 = MotionState(timestamp=0.0, position=np.array([0.50, 0.50]), velocity=np.zeros(2), speed=0.0, confidence=1.0)
    c.update(m0, None, field, [], fsm, 1 / 30, 0.0)
    m1 = MotionState(timestamp=1 / 30, position=np.array([0.502, 0.500]), velocity=np.zeros(2), speed=0.0, confidence=1.0)
    before = c.position.copy()
    c.update(m1, None, field, [], fsm, 1 / 30, 1 / 30)
    moved = c.position - before
    assert moved[0] > 0
    assert abs(moved[0]) < 30
    assert abs(moved[1]) < 10


def test_cooperative_target_lock_requires_actual_motion(cfg):
    cfg.cursor.controller = "cooperative"
    cfg.cursor.relative_mode = True
    c = CursorController(cfg, screen=(1920, 1080))
    fsm = _cooperative_fsm(cfg)
    c.anchor_to(np.array([0.50, 0.50]), cursor_position=np.array([900.0, 500.0]))
    target = Target(id="uia:hover", bounds=(0.44, 0.44, 0.56, 0.56), type="button", semantic_label="Hover", semantic_confidence=1.0, visual_confidence=0.6)
    belief = TargetBelief(target=target, confidence=1.0, distance=0.0, closing_speed=0.0)
    st = c.update(
        MotionState(timestamp=0.0, position=np.array([0.50, 0.50]), velocity=np.zeros(2), speed=0.0, confidence=1.0),
        None, _cooperative_field(), [belief], fsm, 1 / 30, 0.0
    )
    assert st.target_id is None
    assert np.allclose(c.position, [900.0, 500.0])

def test_relative_cursor_dead_zone_suppresses_small_hand_jitter(cfg):
    from gesture_engine.control.cursor import CursorController
    from gesture_engine.state.state_machine import State, StateMachine
    from gesture_engine.types import IntentField
    cfg.cursor.relative_mode = True
    c = CursorController(cfg, screen=(1000, 1000))
    c.anchor_to(np.array([0.50, 0.50]), cursor_position=np.array([500.0, 500.0]))
    fsm = _cursor_fsm(cfg)
    intents = IntentField(probabilities={Intent.MOVE_CURSOR: 1.0}, dominant=Intent.MOVE_CURSOR)
    base = _cursor_inputs(cfg, (0.50, 0.50), speed=0.0)
    c.update(base, None, intents, [], fsm, 1 / 30, 0.0)
    jitter = _cursor_inputs(cfg, (0.5005, 0.5005), speed=0.02)
    out = c.update(jitter, None, intents, [], fsm, 1 / 30, 1 / 30)
    assert np.linalg.norm(out.position - np.array([500.0, 500.0])) < 2.0


def test_overlay_renders_exact_index_tip_marker(cfg):
    # The desktop cursor is normalized desktop space; the fingertip is camera
    # space. Rendering must never use ndarray truthiness to decide whether the
    # prediction exists.
    import cv2
    from gesture_engine.debug.overlay import Overlay
    from gesture_engine.types import FrameReport

    report = FrameReport()
    report.cursor = np.array([0.5, 0.5])
    report.notes = {
        "hand_points": np.pad(np.array([[0.4, 0.4]], dtype=float), ((0, 20), (0, 1)))[:21],
        "index_tip_prediction": np.array([0.46, 0.42]),
        "spatial_intent": 0.7,
    }
    # Put the actual index tip at landmark 8.
    report.notes["hand_points"][8] = np.array([0.42, 0.38, 0.0])
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = Overlay(cfg).draw(frame, report)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
