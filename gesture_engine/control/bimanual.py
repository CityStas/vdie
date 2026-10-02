"""Bimanual desktop interaction layer.

One hand answers *where* (the primary index tip). The other answers *what*:
short pinch = left click, held pinch = right click/context menu, pinch+move =
drag. Two-hand pinches become a zoom channel. Two index fingers can define a
selection rectangle. All decisions are explicit and rate-limited; hover itself
never causes an action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from ..config import EngineConfig
from ..features.feature_vector import build_features
from ..types import ActionRequest, HandFeatures, Intent, Landmarks
from .touch_surface import TouchSurfaceMapper


@dataclass
class BimanualSnapshot:
    enabled: bool = False
    primary: str | None = None
    secondary: str | None = None
    secondary_pinch: bool = False
    secondary_pinch_age: float = 0.0
    secondary_action: str = "NONE"
    zoom_active: bool = False
    zoom_ratio: float = 1.0
    selection_active: bool = False
    selection_start: list[float] | None = None
    selection_end: list[float] | None = None
    pan_active: bool = False
    midpoint: list[float] | None = None
    gate_reason: str = "inactive"
    zoom_delta_ratio: float = 0.0

    def as_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "primary": self.primary,
            "secondary": self.secondary,
            "secondary_pinch": self.secondary_pinch,
            "secondary_pinch_age": self.secondary_pinch_age,
            "secondary_action": self.secondary_action,
            "zoom_active": self.zoom_active,
            "zoom_ratio": self.zoom_ratio,
            "selection_active": self.selection_active,
            "selection_start": self.selection_start,
            "selection_end": self.selection_end,
            "pan_active": self.pan_active,
            "midpoint": self.midpoint,
            "gate_reason": self.gate_reason,
            "zoom_delta_ratio": self.zoom_delta_ratio,
        }


class BimanualInteraction:
    def __init__(self, cfg: EngineConfig, mapper: TouchSurfaceMapper) -> None:
        self.cfg = cfg
        self.mapper = mapper
        self._pinch_down_at: float | None = None
        self._pinch_start_primary: np.ndarray | None = None
        self._pinch_hand: str | None = None
        self._pinch_consumed = False
        self._dragging = False
        self._right_clicked = False
        self._last_left_click = -1e9
        self._zoom_active = False
        self._zoom_consumed = False
        self._zoom_baseline = 0.0
        self._zoom_started_at = -1e9
        self._zoom_last_emit = -1e9
        self._selection_active = False
        self._selection_start: np.ndarray | None = None
        self._selection_last_secondary: np.ndarray | None = None
        self._pan_active = False
        self._pan_start_midpoint: np.ndarray | None = None
        self._pan_last_midpoint: np.ndarray | None = None
        self._snapshot = BimanualSnapshot()

    @property
    def snapshot(self) -> BimanualSnapshot:
        return self._snapshot

    def reset(self) -> None:
        self._pinch_down_at = None
        self._pinch_start_primary = None
        self._pinch_hand = None
        self._pinch_consumed = False
        self._dragging = False
        self._right_clicked = False
        self._zoom_active = False
        self._zoom_consumed = False
        self._zoom_baseline = 0.0
        self._zoom_started_at = -1e9
        self._zoom_last_emit = -1e9
        self._selection_active = False
        self._selection_start = None
        self._selection_last_secondary = None
        self._pan_active = False
        self._pan_start_midpoint = None
        self._pan_last_midpoint = None
        self._snapshot = BimanualSnapshot()

    @staticmethod
    def _pinching(f: HandFeatures | None) -> bool:
        return bool(f is not None and f.pinch_strength >= 0.75)

    @staticmethod
    def _index_only(f: HandFeatures | None) -> bool:
        if f is None:
            return False
        return f.fingers[1] == 1 and f.fingers[2] == 0 and f.fingers[3] == 0 and f.fingers[4] == 0

    def _names(self, hands: Mapping[str, Landmarks], primary: str | None) -> tuple[str | None, str | None]:
        names = list(hands.keys())
        if primary not in names:
            primary = names[0] if names else None
        secondary = next((n for n in names if n != primary), None)
        return primary, secondary

    def _release_active_buttons(self, timestamp: float) -> list[ActionRequest]:
        actions: list[ActionRequest] = []
        if self._dragging:
            actions.append(ActionRequest(Intent.DRAG, "up", {"button": "left", "source": "bimanual"}, risk=0.0, timestamp=timestamp))
        if self._selection_active:
            actions.append(ActionRequest(Intent.DRAG, "up", {"button": "left", "source": "bimanual_selection"}, risk=0.0, timestamp=timestamp))
        if self._pan_active:
            actions.append(ActionRequest(Intent.DRAG, "up", {"button": "middle", "source": "bimanual_pan"}, risk=0.0, timestamp=timestamp))
        if not actions and self._snapshot.gate_reason == "inactive":
            self._snapshot.gate_reason = "idle"
        return actions

    def update(
        self,
        hands: Mapping[str, Landmarks],
        confidences: Mapping[str, float],
        primary: str | None,
        timestamp: float,
        primary_screen: np.ndarray,
        dt: float,
    ) -> list[ActionRequest]:
        bcfg = self.cfg.bimanual
        if not bcfg.enabled or len(hands) < 2 or primary is None:
            actions = self._release_active_buttons(timestamp)
            self.reset()
            return actions

        p_name, s_name = self._names(hands, primary)
        if p_name is None or s_name is None:
            actions = self._release_active_buttons(timestamp)
            self.reset()
            return actions

        freeze_conf = float(self.cfg.safety.freeze_below_confidence)
        if any(float(confidences.get(name, 0.0)) < freeze_conf for name in (p_name, s_name)):
            actions = self._release_active_buttons(timestamp)
            self.reset()
            self._snapshot.gate_reason = "tracking_confidence_below_freeze"
            return actions

        action_conf = float(bcfg.action_min_confidence)
        if any(float(confidences.get(name, 0.0)) < action_conf for name in (p_name, s_name)):
            actions = self._release_active_buttons(timestamp)
            self.reset()
            self._snapshot.gate_reason = "action_confidence_below_threshold"
            return actions

        pf = build_features(hands[p_name].points, self.cfg)
        sf = build_features(hands[s_name].points, self.cfg)
        p_tip = self.mapper.map_point(hands[p_name].points[8, :2])
        s_tip = self.mapper.map_point(hands[s_name].points[8, :2])
        p_pin = self._pinching(pf)
        s_pin = self._pinching(sf)
        actions: list[ActionRequest] = []

        midpoint = (p_tip + s_tip) * 0.5
        separation = float(np.linalg.norm(p_tip - s_tip))
        self._snapshot = BimanualSnapshot(
            enabled=True,
            primary=p_name,
            secondary=s_name,
            secondary_pinch=s_pin,
            midpoint=midpoint.tolist(),
        )

        # Two-hand pinch = zoom. It owns the secondary channel while active,
        # but *holding* a pinch must be inert: only a deliberate separation
        # change after a short arming delay generates wheel steps.
        if p_pin and s_pin and bcfg.zoom_enabled:
            if not self._zoom_active:
                self._zoom_active = True
                self._zoom_consumed = True
                self._zoom_baseline = max(separation, 1.0)
                self._zoom_started_at = timestamp
                self._zoom_last_emit = timestamp
            else:
                age = timestamp - self._zoom_started_at
                baseline = max(self._zoom_baseline, 1.0)
                ratio = separation / baseline
                delta_ratio = ratio - 1.0
                self._snapshot.zoom_active = True
                self._snapshot.zoom_ratio = ratio
                self._snapshot.zoom_delta_ratio = delta_ratio
                if (
                    age >= bcfg.zoom_start_ms / 1000.0
                    and abs(delta_ratio) >= bcfg.zoom_deadband_ratio
                    and timestamp - self._zoom_last_emit >= bcfg.zoom_emit_interval
                ):
                    steps = float(np.clip(delta_ratio * bcfg.zoom_steps_gain, -bcfg.zoom_max_steps, bcfg.zoom_max_steps))
                    if abs(steps) >= 0.35:
                        actions.append(ActionRequest(
                            Intent.CUSTOM, "zoom",
                            {"steps": steps, "source": "bimanual", "ratio": ratio, "delta_ratio": delta_ratio},
                            risk=0.15, timestamp=timestamp
                        ))
                        self._zoom_last_emit = timestamp
                        # Rebase only after a real emit; measurement noise does
                        # not continuously move the baseline.
                        a = float(np.clip(bcfg.zoom_baseline_alpha, 0.0, 1.0))
                        self._zoom_baseline = (1.0 - a) * self._zoom_baseline + a * separation
            # Close an in-progress drag if both hands enter zoom.
            if self._dragging:
                actions.append(ActionRequest(Intent.DRAG, "up", {"button": "left", "source": "bimanual"}, risk=0.0, timestamp=timestamp))
                self._dragging = False
            self._snapshot.gate_reason = "two_hand_pinch_zoom"
            return actions
        if self._zoom_active:
            self._zoom_active = False
            self._zoom_baseline = 0.0
            self._zoom_started_at = -1e9
            self._snapshot.gate_reason = "zoom_released"
        # If zoom consumed the secondary pinch, do not reinterpret the same
        # physical pinch as a click until that secondary hand fully opens.
        if self._zoom_consumed:
            if not s_pin:
                self._zoom_consumed = False
            else:
                return actions

        # Secondary pinch = action modifier.  Short release = left click; held
        # stationary pinch = right click/context menu; pinch + primary movement = drag.
        if s_pin and self._pinch_down_at is None:
            self._pinch_down_at = timestamp
            self._snapshot.gate_reason = "secondary_pinch_armed"
            self._pinch_start_primary = primary_screen.copy()
            self._pinch_hand = s_name
            self._pinch_consumed = False
            self._right_clicked = False
        if s_pin and self._pinch_down_at is not None:
            age = timestamp - self._pinch_down_at
            travel = 0.0 if self._pinch_start_primary is None else float(np.linalg.norm(primary_screen - self._pinch_start_primary))
            self._snapshot.secondary_pinch_age = age
            if not self._dragging and travel >= bcfg.drag_start_px and age >= bcfg.drag_start_ms / 1000.0:
                actions.append(ActionRequest(Intent.DRAG, "down", {"button": "left", "source": "bimanual", "modifier_hand": s_name}, risk=0.25, timestamp=timestamp))
                self._dragging = True
                self._pinch_consumed = True
                self._snapshot.secondary_action = "DRAG"
            elif not self._right_clicked and age >= bcfg.right_click_hold_ms / 1000.0 and travel < bcfg.drag_start_px:
                actions.append(ActionRequest(Intent.RIGHT_CLICK, "right_click", {"button": "right", "source": "bimanual", "modifier_hand": s_name, "held_ms": age * 1000.0}, risk=0.25, timestamp=timestamp))
                self._right_clicked = True
                self._pinch_consumed = True
                self._snapshot.secondary_action = "RIGHT_CLICK"
            if self._dragging:
                self._snapshot.secondary_action = "DRAG"
            return actions

        if not s_pin and self._pinch_down_at is not None:
            age = timestamp - self._pinch_down_at
            was_drag = self._dragging
            if self._dragging:
                actions.append(ActionRequest(Intent.DRAG, "up", {"button": "left", "source": "bimanual"}, risk=0.0, timestamp=timestamp))
                self._dragging = False
            elif not self._pinch_consumed and age <= bcfg.left_click_max_ms / 1000.0:
                # Two short modifier pinches in a narrow window become double-click.
                clicks = 2 if timestamp - self._last_left_click <= bcfg.double_click_window and self._last_left_click > 0 else 1
                kind = "double_click" if clicks == 2 else "click"
                actions.append(ActionRequest(Intent.DOUBLE_CLICK if clicks == 2 else Intent.SELECT, kind, {"button": "left", "source": "bimanual", "modifier_hand": s_name}, risk=0.20, timestamp=timestamp))
                self._last_left_click = timestamp
                self._snapshot.secondary_action = kind.upper()
            elif self._right_clicked:
                self._snapshot.secondary_action = "RIGHT_CLICK"
            else:
                self._snapshot.secondary_action = "NONE"
            self._pinch_down_at = None
            self._pinch_start_primary = None
            self._pinch_hand = None
            self._pinch_consumed = False
            self._right_clicked = False
            return actions

        # Two index fingers = selection rectangle.  It is deliberately a separate
        # pose from pinch so the modifier channel remains deterministic.
        both_index = self._index_only(pf) and self._index_only(sf)
        if bcfg.selection_enabled and both_index and separation >= bcfg.selection_min_separation_px:
            if self._pan_active:
                actions.append(ActionRequest(Intent.DRAG, "up", {"button": "middle", "source": "bimanual_pan"}, risk=0.0, timestamp=timestamp))
                self._pan_active = False
                self._pan_start_midpoint = None
                self._pan_last_midpoint = None
            if not self._selection_active:
                self._selection_active = True
                self._selection_start = p_tip.copy()
                self._selection_last_secondary = s_tip.copy()
                self._snapshot.selection_active = True
                self._snapshot.selection_start = p_tip.tolist()
                self._snapshot.selection_end = s_tip.tolist()
                actions.append(ActionRequest(Intent.DRAG, "move", {"x": float(p_tip[0]), "y": float(p_tip[1]), "source": "bimanual_selection"}, risk=0.0, timestamp=timestamp))
                actions.append(ActionRequest(Intent.DRAG, "down", {"button": "left", "source": "bimanual_selection"}, risk=0.15, timestamp=timestamp))
            else:
                self._selection_last_secondary = s_tip.copy()
                self._snapshot.selection_active = True
                self._snapshot.selection_start = self._selection_start.tolist() if self._selection_start is not None else None
                self._snapshot.selection_end = s_tip.tolist()
            actions.append(ActionRequest(Intent.DRAG, "move", {"x": float(s_tip[0]), "y": float(s_tip[1]), "source": "bimanual_selection"}, risk=0.0, timestamp=timestamp))
            return actions

        if self._selection_active:
            actions.append(ActionRequest(Intent.DRAG, "up", {"button": "left", "source": "bimanual_selection"}, risk=0.0, timestamp=timestamp))
            self._selection_active = False
            self._selection_start = None
            self._selection_last_secondary = None

        # Two close index fingers = bimanual canvas pan. Middle-button drag is
        # broadly understood by Windows applications that expose canvas panning.
        # The cursor dispatcher suppresses the normal single-finger move while this
        # source is active, so there is only one OS movement stream.
        if bcfg.pan_enabled and both_index and separation < bcfg.selection_min_separation_px:
            if self._pan_start_midpoint is None:
                self._pan_start_midpoint = midpoint.copy()
            travel = float(np.linalg.norm(midpoint - self._pan_start_midpoint))
            if not self._pan_active and travel >= bcfg.pan_start_px:
                self._pan_active = True
                actions.append(ActionRequest(Intent.DRAG, "move", {"x": float(midpoint[0]), "y": float(midpoint[1]), "source": "bimanual_pan"}, risk=0.0, timestamp=timestamp))
                actions.append(ActionRequest(Intent.DRAG, "down", {"button": "middle", "source": "bimanual_pan"}, risk=0.15, timestamp=timestamp))
            if self._pan_active:
                self._snapshot.pan_active = True
                self._snapshot.secondary_action = "PAN"
                actions.append(ActionRequest(Intent.DRAG, "move", {"x": float(midpoint[0]), "y": float(midpoint[1]), "source": "bimanual_pan"}, risk=0.0, timestamp=timestamp))
                self._pan_last_midpoint = midpoint.copy()
                return actions
        else:
            self._pan_start_midpoint = None

        if self._pan_active:
            actions.append(ActionRequest(Intent.DRAG, "up", {"button": "middle", "source": "bimanual_pan"}, risk=0.0, timestamp=timestamp))
            self._pan_active = False
            self._pan_last_midpoint = None

        return actions
