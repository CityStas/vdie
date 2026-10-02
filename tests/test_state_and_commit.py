"""State machine, events, commit point and safety."""

from __future__ import annotations

import numpy as np
import pytest

from gesture_engine.events.event_engine import EventEngine
from gesture_engine.features.feature_vector import build_features
from gesture_engine.motion.history import FrameSample, HistoryBuffer
from gesture_engine.policy.commit import CommitDetector
from gesture_engine.state.state_machine import FSMContext, State, StateMachine
from gesture_engine.tracking.landmark_model import POSES, build_hand, make_pose
from gesture_engine.types import EventType, MotionState, Observation


def _ctx(cfg, pose: str, t: float, speed: float = 0.0, conf: float = 1.0, position=(0.5, 0.5), rotation: float | None = None):
    pose_obj = make_pose(pose, rotation_deg=rotation, position=position)
    pts = build_hand(pose_obj)
    features = build_features(pts, cfg)
    motion = MotionState(
        timestamp=t,
        position=np.asarray(position, dtype=float),
        velocity=np.array([speed, 0.0]),
        speed=speed,
        confidence=conf,
    )
    return FSMContext(timestamp=t, features=features, motion=motion, hand_confidence=conf)


def _obs(t: float) -> Observation:
    """The event engine only needs a timestamp; it reads confidence from motion."""
    return Observation(timestamp=t)


# --------------------------------------------------------------------------- #
# State machine
# --------------------------------------------------------------------------- #


def test_activation_requires_dwell(cfg):
    sm = StateMachine(cfg=cfg)
    sm.update(_ctx(cfg, "INDEX_UP", 0.0))
    assert sm.state is State.ACTIVATING
    sm.update(_ctx(cfg, "INDEX_UP", 0.1))
    assert sm.state is State.ACTIVATING
    # Dwell reached -> ARMED, which is a one-frame hand-off to CURSOR.
    sm.update(_ctx(cfg, "INDEX_UP", 0.3))
    assert sm.state is State.ARMED
    sm.update(_ctx(cfg, "INDEX_UP", 0.31))
    assert sm.state is State.CURSOR
    assert sm.cursor_enabled


def test_activation_accepts_confirmed_index_up_gesture_when_raw_finger_state_is_noisy(cfg):
    sm = StateMachine(cfg=cfg)
    ctx = _ctx(cfg, "THREE", 0.0, speed=0.05, conf=0.90)
    ctx.gesture = __import__("gesture_engine.types", fromlist=["GestureResult"]).GestureResult(
        gesture="INDEX_UP", confidence=0.72, stability=0.70, timestamp=0.0
    )
    sm.update(ctx)
    assert sm.state is State.ACTIVATING
    from dataclasses import replace
    ctx.timestamp = 0.30
    ctx.gesture = replace(ctx.gesture, timestamp=0.30)
    sm.update(ctx)
    assert sm.state is State.ARMED
    ctx.timestamp = 0.31
    ctx.gesture = replace(ctx.gesture, timestamp=0.31)
    sm.update(ctx)
    assert sm.state is State.CURSOR


def test_activation_aborted_when_pose_lost(cfg):
    sm = StateMachine(cfg=cfg)
    sm.update(_ctx(cfg, "INDEX_UP", 0.0))
    sm.update(_ctx(cfg, "OPEN_PALM", 0.1))
    assert sm.state is State.IDLE
    assert not sm.cursor_enabled


def test_activation_needs_low_velocity(cfg):
    sm = StateMachine(cfg=cfg)
    for i in range(10):
        sm.update(_ctx(cfg, "INDEX_UP", i * 0.1, speed=2.0))
    assert sm.state is State.IDLE


def test_emergency_cancel_from_cursor(cfg):
    sm = StateMachine(cfg=cfg)
    sm.force(State.CURSOR, 0.0)
    sm.update(_ctx(cfg, "FIST", 0.1))
    assert sm.state is State.EMERGENCY_CANCEL
    sm.update(_ctx(cfg, "FIST", 0.2))
    assert sm.state is State.IDLE


def test_pinch_click_does_not_become_drag(cfg):
    """Closing a pinch moves the fingertip; that must not count as a drag."""
    sm = StateMachine(cfg=cfg)
    sm.force(State.CURSOR, 0.0)
    sm.update(_ctx(cfg, "PINCH", 0.1, speed=0.0, position=(0.5, 0.5)))
    assert sm.state is State.PINCH_DOWN
    sm.update(_ctx(cfg, "PINCH", 0.2, speed=0.0, position=(0.5, 0.5)))
    assert sm.state is State.PINCH_DOWN, "stationary pinch must stay a click candidate"


def test_pinch_plus_displacement_becomes_drag_when_explicitly_enabled(cfg):
    cfg.bimanual.one_hand_drag_enabled = True
    sm = StateMachine(cfg=cfg)
    sm.force(State.CURSOR, 0.0)
    sm.update(_ctx(cfg, "PINCH", 0.1, speed=0.1, position=(0.5, 0.5)))
    for i in range(6):
        sm.update(_ctx(cfg, "PINCH", 0.2 + i * 0.05, speed=0.8, position=(0.5 + 0.02 * (i + 1), 0.5)))
    assert sm.state is State.DRAGGING


def test_drag_release_returns_to_cooldown(cfg):
    sm = StateMachine(cfg=cfg)
    sm.force(State.DRAGGING, 0.0)
    sm.update(_ctx(cfg, "INDEX_UP", 0.1))
    assert sm.state is State.COOLDOWN
    sm.update(_ctx(cfg, "INDEX_UP", 0.5))
    assert sm.state is State.CURSOR


def test_low_confidence_goes_to_cooldown(cfg):
    sm = StateMachine(cfg=cfg)
    sm.force(State.CURSOR, 0.0)
    sm.update(_ctx(cfg, "INDEX_UP", 0.1, conf=0.1))
    assert sm.state is State.COOLDOWN


def test_open_palm_pauses_and_releases(cfg):
    sm = StateMachine(cfg=cfg)
    sm.force(State.CURSOR, 0.0)
    sm.update(_ctx(cfg, "OPEN_PALM", 0.1))
    assert sm.state is State.PAUSED
    sm.update(_ctx(cfg, "INDEX_UP", 0.2))
    assert sm.state is State.PAUSED  # release needs pause_release_ms
    sm.update(_ctx(cfg, "INDEX_UP", 0.8))
    assert sm.state is State.CURSOR


def test_transitions_are_recorded(cfg):
    sm = StateMachine(cfg=cfg)
    sm.force(State.CURSOR, 0.0, "test")
    assert sm.transitions[-1].from_state is State.IDLE
    assert sm.transitions[-1].to_state is State.CURSOR


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


def test_event_engine_emits_pinch_edges(cfg):
    ee = EventEngine(cfg)
    ee.update(_obs(0.0), MotionState(timestamp=0.0, speed=0.0, confidence=1.0))
    closed = ee.update(_obs(0.1), MotionState(timestamp=0.1, speed=0.0, confidence=1.0), pinch_distance=0.1)
    assert any(e.type is EventType.PINCH_START for e in closed)
    opened = ee.update(_obs(0.2), MotionState(timestamp=0.2, speed=0.0, confidence=1.0), pinch_distance=0.9)
    assert any(e.type is EventType.PINCH_RELEASE for e in opened)


def test_event_engine_emits_point_start_stop(cfg):
    ee = EventEngine(cfg)
    ee.update(_obs(0.0), MotionState(timestamp=0.0, speed=0.0, confidence=1.0))
    moving = ee.update(_obs(0.1), MotionState(timestamp=0.1, speed=1.0, confidence=1.0))
    assert any(e.type is EventType.POINT_START for e in moving)


def test_event_engine_hand_lost(cfg):
    ee = EventEngine(cfg)
    ee.update(_obs(0.0), MotionState(timestamp=0.0, speed=0.0, confidence=1.0))
    lost = ee.update(_obs(0.1), MotionState(timestamp=0.1, speed=0.0, confidence=0.0))
    assert any(e.type is EventType.HAND_LOST for e in lost)


def test_event_engine_direction_change(cfg):
    ee = EventEngine(cfg)
    ee.update(_obs(0.0), MotionState(timestamp=0.0, speed=1.0, direction=0.0, confidence=1.0))
    turn = ee.update(_obs(0.1), MotionState(timestamp=0.1, speed=1.0, direction=2.5, direction_change=2.5, confidence=1.0))
    assert any(e.type is EventType.DIRECTION_CHANGE for e in turn)


# --------------------------------------------------------------------------- #
# Commit point
# --------------------------------------------------------------------------- #


def _commit_sequence(det: CommitDetector, cfg, speeds, dt=1 / 30):
    """Feed a speed profile; pause_duration accumulates while below pause_speed."""
    history = HistoryBuffer(capacity=200)
    signals = []
    t = 0.0
    pause = 0.0
    for speed in speeds:
        if speed < cfg.motion.pause_speed:
            pause += dt
        else:
            pause = 0.0
        motion = MotionState(
            timestamp=t,
            position=np.zeros(2),
            velocity=np.array([speed, 0.0]),
            speed=speed,
            speed_trend=0.0,
            pause_duration=pause,
            is_pausing=pause > 0,
        )
        signals.append(det.update(motion, history, None, [], _fsm_cursor(cfg), t))
        t += dt
    return signals


def _fsm_cursor(cfg):
    sm = StateMachine(cfg=cfg)
    sm.force(State.CURSOR, 0.0)
    return sm


def test_kinematic_commit_fires_on_decelerate_then_pause(cfg):
    det = CommitDetector(cfg=cfg)
    # accelerate, travel, decelerate, stop
    speeds = [0.1, 0.5, 1.0, 1.6, 1.8, 1.6, 1.0, 0.5, 0.2, 0.05] + [0.02] * 10
    signals = _commit_sequence(det, cfg, speeds)
    assert any(s.committed for s in signals), "deceleration followed by a micro-pause must commit"
    kinds = {s.kind for s in signals if s.committed}
    assert kinds & {"MICRO_PAUSE", "THRUST", "TARGET_ENTRY"}


def test_no_commit_during_steady_travel(cfg):
    det = CommitDetector(cfg=cfg)
    signals = _commit_sequence(det, cfg, [1.5] * 40)
    assert not any(s.committed for s in signals)


def test_commit_respects_cooldown(cfg):
    det = CommitDetector(cfg=cfg)
    speeds = [0.1, 1.2, 0.2, 0.05, 0.02, 0.02] * 4
    signals = _commit_sequence(det, cfg, speeds)
    times = [s.timestamp for s in signals if s.committed]
    assert len(times) <= len(speeds) * (1 / 30) / cfg.commit.cooldown + 1


def test_commit_disabled_config(cfg):
    cfg.commit.enabled = False
    det = CommitDetector(cfg=cfg)
    signals = _commit_sequence(det, cfg, [0.1, 1.5, 0.5, 0.05] + [0.02] * 10)
    assert not any(s.committed for s in signals)


def _thrust_probe(cfg, step: float) -> float:
    """Feed a rest -> fast burst -> one slow frame and return the peak THRUST score.

    Asserts on the *score*, not on a commit: at the current tuning
    ``thrust_max_duration`` (0.28 s) plus the linear ``1 - d / max`` score means a
    THRUST needs a burst of at most ~3 frames to clear the 0.55 threshold, so a
    realistic hand flick scores but does not commit.  The score is still carried
    on the signal, so the travel gate can be tested directly.
    """
    det = CommitDetector(cfg=cfg)
    history = HistoryBuffer(capacity=200)
    dt = 1 / 30
    t = 0.0

    def push(point, speed, pause):
        nonlocal t
        history.append(
            FrameSample(
                timestamp=t,
                frame_id=len(history),
                point=np.asarray(point, dtype=float),
                motion=MotionState(
                    timestamp=t,
                    position=np.asarray(point, dtype=float),
                    speed=speed,
                    pause_duration=pause,
                    is_pausing=pause > 0,
                ),
            )
        )

    scores = []
    for _ in range(4):  # at rest
        push((0.5, 0.5), 0.0, 0.1)
        scores.append(det.update(history.latest().motion, history, None, [], _fsm_cursor(cfg), t).score)
        t += dt
    for i in range(2):  # a short, fast burst
        push((0.5 + step * (i + 1), 0.5), 1.6, 0.0)
        scores.append(det.update(history.latest().motion, history, None, [], _fsm_cursor(cfg), t).score)
        t += dt
    # One frame below the thrust threshold ends the burst.  pause_duration is 0
    # so the micro-pause detector cannot fire and mask the result.
    push((0.5 + step * 2, 0.5), 0.5, 0.0)
    scores.append(det.update(history.latest().motion, history, None, [], _fsm_cursor(cfg), t).score)
    return max(scores)


def test_finger_articulation_is_not_a_thrust(cfg):
    """Closing a pinch moves the index tip fast and from rest — not a flick.

    The control point *is* the index tip, so an INDEX_UP -> PINCH transition
    articulates it by ~0.07 normalized units.  Without a travel gate that burst
    passed the speed-and-duration test and committed a click on every pinch:
    measured on the click scenario, 3 of 12 commits came from finger
    articulation rather than hand movement.
    """
    assert _thrust_probe(cfg, step=0.035) == 0.0


def test_hand_flick_is_a_thrust(cfg):
    """The same speed profile over a real distance is a deliberate flick."""
    assert _thrust_probe(cfg, step=0.12) > 0.4
