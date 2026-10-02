from __future__ import annotations

import numpy as np

from gesture_engine import load_config
from gesture_engine.control.bimanual import BimanualInteraction
from gesture_engine.control.mouse import NullMouse
from gesture_engine.control.touch_surface import TouchSurfaceMapper
from gesture_engine.control.keyboard import NullKeyboard
from gesture_engine.types import HandFeatures, Landmarks
from gesture_engine.features.feature_vector import build_features


def _pose(cfg, pose: str, pos=(0.5,0.5)):
    from gesture_engine.tracking.landmark_model import make_pose, build_hand
    return Landmarks(build_hand(make_pose(pose, position=pos, scale=0.22)))


def test_touch_surface_maps_full_desktop() -> None:
    cfg=load_config(); m=TouchSurfaceMapper((1920,1080), hysteresis_px=0.0)
    assert np.allclose(m.map_point(np.array([0.0,0.0])), [0,0])
    assert np.allclose(m.map_point(np.array([1.0,1.0])), [1919,1079])
    assert np.allclose(m.map_point(np.array([0.5,0.5])), [959.5,539.5])


def test_bimanual_short_pinch_clicks() -> None:
    cfg=load_config(); cfg.bimanual.enabled=True
    mapper=TouchSurfaceMapper((1920,1080), hysteresis_px=0)
    bi=BimanualInteraction(cfg, mapper)
    # Finger poses place the two hands apart; secondary PINCH is the modifier.
    a=_pose(cfg,'INDEX_UP',(0.35,0.5)); b=_pose(cfg,'PINCH',(0.65,0.5))
    hands={'left':a,'right':b}
    conf={'left':0.95,'right':0.95}
    assert bi.update(hands,conf,'left',0.0,np.array([650.,540.]),1/30)==[]
    acts=bi.update(hands,conf,'left',0.08,np.array([650.,540.]),1/30)
    assert acts==[]
    open_b=_pose(cfg,'INDEX_UP',(0.65,0.5))
    acts=bi.update({'left':a,'right':open_b},conf,'left',0.16,np.array([650.,540.]),1/30)
    assert any(a.kind=='click' for a in acts)


def test_two_hand_zoom_requires_deliberate_separation_and_delay() -> None:
    cfg=load_config(); cfg.bimanual.enabled=True
    mapper=TouchSurfaceMapper((1920,1080), hysteresis_px=0)
    bi=BimanualInteraction(cfg, mapper)
    a=_pose(cfg,'PINCH',(0.35,0.5)); b=_pose(cfg,'PINCH',(0.65,0.5))
    hands={'left':a,'right':b}; conf={'left':.95,'right':.95}
    bi.update(hands,conf,'left',0.0,np.array([650.,540.]),1/30)
    acts=bi.update(hands,conf,'left',0.10,np.array([650.,540.]),1/30)
    assert not any(x.kind=='zoom' for x in acts)
    b2=_pose(cfg,'PINCH',(0.76,0.5))
    acts=bi.update({'left':a,'right':b2},conf,'left',0.22,np.array([650.,540.]),1/30)
    zooms=[x for x in acts if x.kind=='zoom']
    assert zooms and abs(float(zooms[0].payload['steps'])) <= cfg.bimanual.zoom_max_steps


def test_two_index_selection_is_disabled_by_default() -> None:
    cfg=load_config(); cfg.bimanual.enabled=True
    mapper=TouchSurfaceMapper((1920,1080), hysteresis_px=0)
    bi=BimanualInteraction(cfg, mapper)
    a=_pose(cfg,'INDEX_UP',(0.20,0.5)); b=_pose(cfg,'INDEX_UP',(0.80,0.5))
    acts=bi.update({'left':a,'right':b},{'left':.95,'right':.95},'left',0.0,np.array([384.,540.]),1/30)
    assert not any(x.payload.get('source')=='bimanual_selection' for x in acts)

def test_bimanual_close_index_pan():
    cfg=load_config(); cfg.bimanual.enabled=True; cfg.bimanual.pan_enabled=True
    mapper=TouchSurfaceMapper((1920,1080), hysteresis_px=0)
    bi=BimanualInteraction(cfg, mapper)
    a=_pose(cfg,'INDEX_UP',(0.50,0.5)); b=_pose(cfg,'INDEX_UP',(0.51,0.5))
    bi.update({'left':a,'right':b},{'left':.95,'right':.95},'left',0.0,np.array([960.,540.]),1/30)
    a2=_pose(cfg,'INDEX_UP',(0.52,0.52)); b2=_pose(cfg,'INDEX_UP',(0.53,0.52))
    acts=bi.update({'left':a2,'right':b2},{'left':.95,'right':.95},'left',0.10,np.array([999.,560.]),1/30)
    assert any(a.kind=='down' and a.payload.get('button')=='middle' for a in acts) or bi.snapshot.pan_active


def test_bimanual_action_requires_confidence_threshold():
    cfg = load_config(); cfg.bimanual.enabled = True
    mapper = TouchSurfaceMapper((1920, 1080), hysteresis_px=0)
    bi = BimanualInteraction(cfg, mapper)
    a = _pose(cfg, 'INDEX_UP', (0.35, 0.5)); b = _pose(cfg, 'PINCH', (0.65, 0.5))
    # Secondary hand is visible but below the action confidence threshold.
    acts = bi.update({'left': a, 'right': b}, {'left': .95, 'right': .60}, 'left', 0.0, np.array([650., 540.]), 1/30)
    assert acts == []
    assert bi.snapshot.gate_reason == 'action_confidence_below_threshold'


def test_fast_one_hand_pinch_click_does_not_need_commit(cfg):
    from gesture_engine.policy.action_policy import ActionPolicy
    from gesture_engine.state.state_machine import StateMachine
    from gesture_engine.types import MotionState, IntentField
    from gesture_engine.features.feature_vector import build_features
    from gesture_engine.tracking.landmark_model import make_pose, build_hand
    policy=ActionPolicy(cfg)
    cfg.state.fast_pinch_click=True
    sm=StateMachine(cfg)
    sm.force(sm.state.CURSOR if hasattr(sm.state,'CURSOR') else type(sm.state).CURSOR, 0.0)
    pts=build_hand(make_pose('PINCH', position=(0.5,0.5), scale=0.22))
    f=build_features(pts,cfg)
    motion=MotionState(position=np.array([0.5,0.5]),velocity=np.zeros(2),acceleration=np.zeros(2),speed=0,direction=0,curvature=0,amplitude=0,pause_duration=.2,confidence=.95)
    acts=policy.update(intents=IntentField(),commit=None,gesture=None,motion=motion,features=f,beliefs=[],fsm=sm,timestamp=.1,bimanual_active=False)
    assert any(a.kind=='click' and a.payload.get('source')=='fast_pinch' for a in acts)


def test_bimanual_releases_held_buttons_on_tracking_loss():
    cfg = load_config(); cfg.bimanual.enabled = True; cfg.bimanual.one_hand_drag_enabled = True
    mapper = TouchSurfaceMapper((1920, 1080), hysteresis_px=0)
    bi = BimanualInteraction(cfg, mapper)
    a = _pose(cfg, 'INDEX_UP', (0.30, 0.5)); b = _pose(cfg, 'PINCH', (0.70, 0.5))
    conf = {'left': .95, 'right': .95}
    bi.update({'left': a, 'right': b}, conf, 'left', 0.0, np.array([576., 540.]), 1/30)
    acts = bi.update({'left': _pose(cfg, 'INDEX_UP', (0.40, 0.55)), 'right': b}, conf, 'left', 0.30, np.array([768., 594.]), 1/30)
    assert any(x.kind == 'down' and x.payload.get('button') == 'left' for x in acts)
    assert bi._dragging
    # Low confidence/partial tracking loss must always release a held drag/pan/selection.
    acts = bi.update({'left': _pose(cfg, 'INDEX_UP', (0.40, 0.55)), 'right': b}, {'left': .95, 'right': .05}, 'left', 0.25, np.array([768., 594.]), 1/30)
    assert any(x.kind == 'up' and x.payload.get('button') == 'left' for x in acts)
    assert not bi._dragging


def test_touch_surface_low_confidence_does_not_add_positional_lag():
    from gesture_engine.control.cursor import CursorController
    from gesture_engine.state.state_machine import State, StateMachine
    from gesture_engine.types import IntentField, MotionState

    cfg = load_config()
    cursor = CursorController(cfg, screen=(1920, 1080))
    fsm = StateMachine(cfg)
    fsm.force(State.CURSOR, 0.0)
    m1 = MotionState(position=np.array([0.50, 0.50]), velocity=np.zeros(2), confidence=0.95)
    cursor.update(m1, None, IntentField(), [], fsm, 1/30, 0.0)
    m2 = MotionState(position=np.array([0.70, 0.50]), velocity=np.zeros(2), confidence=0.40)
    state = cursor.update(m2, None, IntentField(), [], fsm, 1/30, 1/30)
    # 0.40 is above the safety freeze threshold. It must not be turned into
    # extra positional damping in the absolute touch controller.
    expected = cursor.touch_surface.map_point(np.array([0.70, 0.50]))
    assert np.linalg.norm(state.position - expected) < 35.0
