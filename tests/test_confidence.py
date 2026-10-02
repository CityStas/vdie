"""Temporal confirmation primitives: voting, hysteresis, cooldown."""

from __future__ import annotations

import pytest

from gesture_engine.gestures.confidence import Cooldown, Hysteresis, VoteBuffer


def test_vote_buffer_requires_m_of_n():
    v = VoteBuffer(window=10, required=7)
    for _ in range(6):
        v.push("PINCH")
    assert v.confirmed() is None
    v.push("NONE")
    # 6 PINCH out of 7 -- still one short of the required count.
    assert v.confirmed() is None
    v.push("PINCH")
    confirmed = v.confirmed()
    assert confirmed is not None
    assert confirmed[0] == "PINCH"
    # agreement is measured against the *filled* window, not the requirement
    assert confirmed[1] == pytest.approx(7 / 8)
    # fill the window completely: 7 PINCH + 3 NONE -> agreement 0.7
    v.push("NONE")
    v.push("NONE")
    confirmed = v.confirmed()
    assert confirmed is not None
    assert confirmed[1] == pytest.approx(0.7)


def test_vote_buffer_ignores_none():
    v = VoteBuffer(window=4, required=3)
    for _ in range(4):
        v.push("NONE")
    assert v.confirmed() is None


def test_vote_buffer_agreement():
    v = VoteBuffer(window=4, required=2)
    v.push("A")
    v.push("A")
    v.push("B")
    v.push("B")
    assert v.agreement("A") == pytest.approx(0.5)


def test_hysteresis_distinct_thresholds():
    # Schmitt trigger: a high threshold to enter, a lower one to leave.
    h = Hysteresis(enter=0.35, exit_value=0.25)
    assert h.update(0.30, 0.0) == 0  # between the two thresholds: no change
    assert h.update(0.36, 0.1) == 1  # above enter
    assert h.update(0.30, 0.2) == 0  # still above exit
    assert h.update(0.24, 0.3) == -1  # below exit -> release


def test_hysteresis_does_not_chatter():
    h = Hysteresis(enter=0.35, exit_value=0.25)
    transitions = []
    for i in range(40):
        value = 0.30 if i % 2 == 0 else 0.29
        edge = h.update(value, i * 0.01)
        if edge:
            transitions.append(edge)
    assert transitions == []


def test_hysteresis_min_dwell():
    h = Hysteresis(enter=0.5, exit_value=0.4, min_dwell=0.5)
    assert h.update(0.9, 0.0) == 1
    h2 = Hysteresis(enter=0.5, exit_value=0.4, min_dwell=0.5)
    h2.state = True
    h2._last_change = 0.0
    assert h2.update(0.9, 0.1) == 0


def test_cooldown():
    c = Cooldown(interval=0.2)
    assert c.try_consume(0.0)
    assert not c.try_consume(0.1)
    assert c.try_consume(0.25)
