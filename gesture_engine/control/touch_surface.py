"""Absolute virtual touch-surface mapping for the desktop.

The camera supplies an index-fingertip position in normalized image coordinates.
This module turns that position into a physical desktop coordinate.  A calibrated
3x3 homography can be used when the camera perspective is non-linear; otherwise
mapping is full-frame and strictly constrained to the desktop bounds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class TouchSurfaceError(RuntimeError):
    pass


class TouchSurfaceMapper:
    """Low-latency absolute mapper with optional homography calibration."""

    def __init__(
        self,
        screen: tuple[int, int] = (1920, 1080),
        calibration_path: str | None = None,
        use_calibration: bool = False,
        hysteresis_px: float = 1.0,
        edge_margin_px: float = 0.0,
    ) -> None:
        self.screen = (max(1, int(screen[0])), max(1, int(screen[1])))
        self.calibration_path = Path(calibration_path) if calibration_path else None
        self.use_calibration = bool(use_calibration)
        self.hysteresis_px = max(0.0, float(hysteresis_px))
        self.edge_margin_px = max(0.0, float(edge_margin_px))
        self.H: np.ndarray | None = None
        self.version = "full_frame"
        self._last_output: np.ndarray | None = None
        self._offset_px = np.zeros(2, dtype=np.float64)
        self._offset_enabled = False
        self._reanchor_point: np.ndarray | None = None
        self.load()

    def set_screen(self, size: tuple[int, int]) -> None:
        self.screen = (max(1, int(size[0])), max(1, int(size[1])))

    @property
    def calibrated(self) -> bool:
        return self.H is not None

    @property
    def offset_enabled(self) -> bool:
        return self._offset_enabled

    def load(self) -> None:
        self.H = None
        self.version = "full_frame"
        if not self.use_calibration or self.calibration_path is None:
            return
        if not self.calibration_path.exists():
            log.info("touch calibration not found at %s; using full-frame mapping", self.calibration_path)
            return
        try:
            data = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            matrix = np.asarray(data.get("homography"), dtype=np.float64)
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                raise TouchSurfaceError("homography must be a finite 3x3 matrix")
            self.H = matrix / max(abs(matrix[2, 2]), 1e-12)
            self.version = str(data.get("version", "homography_v1"))
            log.info("touch calibration loaded: %s", self.calibration_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("invalid touch calibration %s: %s; using full-frame mapping", self.calibration_path, exc)
            self.H = None

    def reset(self) -> None:
        self._last_output = None
        self._offset_px[:] = 0.0
        self._offset_enabled = False
        self._reanchor_point = None

    def _full_frame(self, point: np.ndarray) -> np.ndarray:
        w, h = self.screen
        u = float(np.clip(point[0], 0.0, 1.0))
        v = float(np.clip(point[1], 0.0, 1.0))
        return np.array([u * (w - 1), v * (h - 1)], dtype=np.float64)

    def map_point(self, point: np.ndarray) -> np.ndarray:
        """Map normalized image coordinates to physical screen pixels."""
        p = np.asarray(point, dtype=np.float64)[:2]
        if self.H is None:
            out = self._full_frame(p)
        else:
            hp = self.H @ np.array([p[0], p[1], 1.0], dtype=np.float64)
            denom = hp[2]
            if abs(denom) < 1e-9:
                out = self._full_frame(p)
            else:
                # Calibration stores desktop pixels directly.
                out = hp[:2] / denom
        w, h = self.screen
        out = np.clip(out, [self.edge_margin_px, self.edge_margin_px], [max(self.edge_margin_px, w - 1 - self.edge_margin_px), max(self.edge_margin_px, h - 1 - self.edge_margin_px)])
        return out

    def map_filtered(self, point: np.ndarray) -> np.ndarray:
        target = self.map_point(point)
        if self._offset_enabled:
            target = target + self._offset_px
        if self._last_output is not None and self.hysteresis_px > 0.0:
            delta = target - self._last_output
            if float(np.linalg.norm(delta)) <= self.hysteresis_px:
                return self._last_output.copy()
        self._last_output = target.copy()
        return target

    def reanchor(self, point: np.ndarray, cursor_position: np.ndarray) -> np.ndarray:
        """Keep the current desktop cursor while beginning from a new finger point."""
        mapped = self.map_point(point)
        self._offset_px = np.asarray(cursor_position, dtype=np.float64)[:2] - mapped
        self._offset_enabled = True
        self._reanchor_point = np.asarray(point, dtype=np.float64)[:2].copy()
        self._last_output = np.asarray(cursor_position, dtype=np.float64)[:2].copy()
        return self._last_output.copy()

    def clear_offset(self) -> None:
        self._offset_px[:] = 0.0
        self._offset_enabled = False
        self._reanchor_point = None

    def state(self) -> dict:
        return {
            "version": self.version,
            "calibrated": self.calibrated,
            "path": str(self.calibration_path) if self.calibration_path else None,
            "hysteresis_px": self.hysteresis_px,
            "screen": [self.screen[0], self.screen[1]],
            "offset_enabled": self._offset_enabled,
            "offset_px": self._offset_px.copy(),
            "reanchor_point": self._reanchor_point.copy() if self._reanchor_point is not None else None,
        }
