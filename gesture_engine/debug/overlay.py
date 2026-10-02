"""Debug overlay.

Without this, tuning a gesture engine is guesswork.  The overlay shows the
*decision chain*, not just the landmarks: state, gesture, intent distribution,
commit signal, target influence and the cursor.  When something misbehaves you can
see which layer was wrong.

Rendering is optional and fully guarded — the engine runs headless with no OpenCV
window at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EngineConfig
from ..state.state_machine import StateMachine
from ..types import FrameReport, Intent

#: MediaPipe hand skeleton (21 landmarks).
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "IDLE": (140, 140, 140),
    "ACTIVATING": (0, 200, 255),
    "ARMED": (0, 255, 255),
    "CURSOR": (0, 220, 0),
    "PINCH_DOWN": (255, 160, 0),
    "DRAGGING": (255, 80, 0),
    "SCROLL": (255, 0, 200),
    "PAUSED": (0, 165, 255),
    "COOLDOWN": (180, 180, 0),
    "EMERGENCY_CANCEL": (0, 0, 255),
}


@dataclass
class Overlay:
    cfg: EngineConfig
    scale: float = 1.0

    # ------------------------------------------------------------------ #
    def draw(self, frame: np.ndarray, report: FrameReport, telemetry_lines: list[str] | None = None, history_points: np.ndarray | None = None, grammar_partial: str = "") -> np.ndarray:
        if frame is None:
            return frame
        try:
            import cv2
        except ImportError:  # pragma: no cover
            return frame

        img = frame.copy()
        h, w = img.shape[:2]
        d = self.cfg.debug

        # In cooperative mode the OS cursor is in *desktop space*, not camera
        # space. Drawing it over the camera feed made the cross appear detached
        # from the fingertip and was a major source of confusion while tuning.
        # Show the actual INDEX TIP on the camera image and the desktop cursor in
        # a separate mini-map.
        if d.show_activation_zone and self.cfg.cursor.controller != "cooperative":
            self._draw_active_region(cv2, img, w, h)
        if d.show_landmarks and report.features is not None:
            self._draw_landmarks(cv2, img, report, w, h)
        if d.show_trajectory and history_points is not None and len(history_points) > 1:
            pts = np.asarray(history_points, dtype=np.float64)
            xy = np.stack([pts[:, 0] * w, pts[:, 1] * h], axis=1).astype(np.int32)
            cv2.polylines(img, [xy.reshape(-1, 1, 2)], False, (255, 200, 0), 2, cv2.LINE_AA)
        if d.show_prediction and report.motion is not None:
            self._draw_prediction(cv2, img, report, w, h)
        if d.show_targets or report.cursor is not None:
            self._draw_desktop_map(cv2, img, report, w, h)

        self._draw_panel(cv2, img, report, telemetry_lines or [], grammar_partial, w, h)
        return img

    # ------------------------------------------------------------------ #
    def _draw_landmarks(self, cv2, img, report: FrameReport, w: int, h: int) -> None:
        # Landmarks are already normalized in the report's features; for drawing we
        # need the raw points, which the engine passes via notes.
        raw = report.notes.get("hand_points")
        if raw is None:
            return
        pts = np.asarray(raw, dtype=np.float64)
        px = np.stack([pts[:, 0] * w, pts[:, 1] * h], axis=1).astype(np.int32)
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, tuple(px[a]), tuple(px[b]), (0, 255, 0), 1, cv2.LINE_AA)
        for i, p in enumerate(px):
            color = (0, 0, 255) if i in (4, 12, 16, 20) else (255, 255, 255)
            radius = 3
            if i == 8:
                color = (0, 230, 255)
                radius = 7
            cv2.circle(img, tuple(p), radius, color, -1, cv2.LINE_AA)
        tip = px[8]
        cv2.circle(img, tuple(tip), 11, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, "INDEX TIP", (int(tip[0]) + 12, int(tip[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 230, 255), 1, cv2.LINE_AA)
        pred = report.notes.get("index_tip_prediction")
        if pred is not None:
            q = np.asarray(pred, dtype=np.float64)
            qp = (int(q[0] * w), int(q[1] * h))
            cv2.arrowedLine(img, tuple(tip), qp, (255, 220, 0), 2, cv2.LINE_AA, tipLength=0.25)

    def _draw_desktop_map(self, cv2, img, report: FrameReport, w: int, h: int) -> None:
        """Render the desktop-space cursor/targets as a compact HUD map."""
        map_w = min(360, max(250, int(w * 0.30)))
        map_h = int(map_w * 9 / 16)
        pad = 14
        x0 = max(8, w - map_w - pad)
        y0 = max(8, h - map_h - pad)
        x1, y1 = x0 + map_w, y0 + map_h

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
        cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(img, "DESKTOP / SPATIAL INTENT", (x0 + 9, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

        def map_point(p: np.ndarray | list[float] | tuple[float, float]) -> tuple[int, int]:
            a = np.asarray(p, dtype=np.float64)
            return (x0 + 5 + int(np.clip(a[0], 0.0, 1.0) * max(map_w - 10, 1)),
                    y0 + 5 + int(np.clip(a[1], 0.0, 1.0) * max(map_h - 10, 1)))

        # UIA targets in desktop coordinates.
        for b in report.target_beliefs[:18]:
            tx0, ty0, tx1, ty1 = b.target.bounds
            q0 = map_point((tx0, ty0))
            q1 = map_point((tx1, ty1))
            active = report.cursor is not None and b.target.id == report.notes.get("cursor_target")
            col = (0, 230, 255) if active else (0, 140, 255)
            cv2.rectangle(img, q0, q1, col, 1 if not active else 2, cv2.LINE_AA)

        if report.cursor is not None:
            c = map_point(report.cursor)
            cv2.circle(img, c, 7, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(img, (c[0] - 11, c[1]), (c[0] + 11, c[1]), (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(img, (c[0], c[1] - 11), (c[0], c[1] + 11), (255, 255, 255), 1, cv2.LINE_AA)
            target_name = str(report.notes.get("cursor_target", ""))
            score = float(report.notes.get("cursor_target_score", 0.0) or 0.0)
            if target_name:
                cv2.putText(img, f"LOCK {target_name} {score:.2f}", (x0 + 9, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 255), 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "FREE TRAVEL", (x0 + 9, y1 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 210, 255), 1, cv2.LINE_AA)

    def _draw_active_region(self, cv2, img, w: int, h: int) -> None:
        rx0, ry0, rx1, ry1 = self.cfg.control.active_region
        cv2.rectangle(
            img,
            (int(rx0 * w), int(ry0 * h)),
            (int(rx1 * w), int(ry1 * h)),
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

    def _draw_prediction(self, cv2, img, report: FrameReport, w: int, h: int) -> None:
        m = report.motion
        assert m is not None
        p = (int(m.position[0] * w), int(m.position[1] * h))
        v = m.velocity
        q = (int((m.position[0] + v[0] * 0.12) * w), int((m.position[1] + v[1] * 0.12) * h))
        cv2.arrowedLine(img, p, q, (0, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)

    # ------------------------------------------------------------------ #
    def _draw_panel(self, cv2, img, report: FrameReport, telemetry_lines: list[str], grammar_partial: str, w: int, h: int) -> None:
        color = STATE_COLORS.get(report.state, (255, 255, 255))
        lines: list[tuple[str, tuple[int, int, int]]] = [
            (f"STATE {report.state}", color),
            (f"GESTURE {report.gesture.gesture}  conf {report.gesture.confidence:.2f}  stab {report.gesture.stability:.2f}", (255, 255, 255)),
        ]
        for intent, prob in report.intents.top_k(3):
            mark = "*" if report.intents.committed is intent else " "
            lines.append((f"{mark}{intent.value:<14} {prob:.2f}", (200, 255, 200)))
        lines.append((f"entropy {report.intents.entropy:.2f}  commit {report.commit.kind if report.commit else '-'}", (200, 200, 200)))
        if report.motion is not None:
            m = report.motion
            lines.append((f"speed {m.speed:.2f}  peak {m.speed_peak:.2f}  pause {m.pause_duration:.2f}s", (200, 200, 200)))
        if report.features is not None:
            f = report.features
            lines.append((f"fingers {''.join(str(x) for x in f.fingers)}  pinch {f.pinch_distance:.2f} str {f.pinch_strength:.2f}", (200, 200, 200)))
            lines.append((f"tilt {f.index_orientation:+.0f} deg  span {f.hand_span:.2f}  depth {f.depth:.2f}", (200, 200, 200)))
        raw = report.notes.get("hand_points")
        if raw is not None:
            tip = np.asarray(raw, dtype=np.float64)[8]
            lines.append((f"INDEX TIP  {tip[0]:.3f},{tip[1]:.3f}", (0, 230, 255)))
        if report.cursor is not None:
            lines.append((f"cursor {report.cursor[0]:.0f},{report.cursor[1]:.0f}  spatial {float(report.notes.get('spatial_intent', 0.0) or 0.0):.2f}", (200, 200, 200)))
        if report.actions:
            lines.append(("actions " + ",".join(a.kind for a in report.actions), (0, 255, 255)))
        if grammar_partial:
            lines.append((f"grammar: {grammar_partial}", (255, 200, 255)))
        lines.extend((ln, (180, 180, 180)) for ln in telemetry_lines)

        y = 18
        for text, col in lines:
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            y += 17

    # ------------------------------------------------------------------ #
    @staticmethod
    def render_text(report: FrameReport, telemetry_lines: list[str] | None = None) -> str:
        """Headless one-frame summary for the console."""
        intents = " ".join(f"{i.value}:{p:.2f}" for i, p in report.intents.top_k(3))
        bits = [
            f"[{report.timestamp:7.3f}] {report.state:<17}",
            f"gesture={report.gesture.gesture:<14} c={report.gesture.confidence:.2f}",
            f"intent={intents}",
        ]
        if report.motion is not None:
            bits.append(f"speed={report.motion.speed:.2f} pause={report.motion.pause_duration:.2f}")
        if report.commit is not None and report.commit.kind != "NONE":
            bits.append(f"commit={report.commit.kind}({report.commit.score:.2f})")
        if report.actions:
            bits.append("actions=" + ",".join(a.kind for a in report.actions))
        if telemetry_lines:
            bits.append(telemetry_lines[0])
        return " | ".join(bits)
