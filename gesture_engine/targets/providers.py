"""Target providers.

Three sources, in order of preference:

1. **UIA** (``targets.provider: uia``) — Windows UI Automation.  Preferred when
   available: semantic labels, real bounds, real capabilities.  Optional import;
   silently falls back if the package or the accessibility tree is unusable.
2. **CV** (``targets.provider: cv``) — screenshot + contour detection.  Works
   everywhere, knows nothing about semantics.  Deliberately conservative: it
   returns *visual* targets only, and their confidence is capped so that a
   detected blob can never outrank a semantic target.
3. **static** (default) — a fixed list.  This exists so the target-aware
   behaviour is testable and benchmarkable in isolation, which is the only way to
   attribute an improvement to the target model rather than to the UI.

The engine never assumes semantic UI information is available.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field

import numpy as np

from ..config import EngineConfig
from ..features.geometry import bbox_iou
from ..types import Target

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 1. UI Automation
# --------------------------------------------------------------------------- #


@dataclass
class UIAProvider:
    """Windows UI Automation target provider for the foreground application.

    The provider is intentionally conservative.  It exposes only controls that
    are useful as pointer targets (buttons, list items, tabs, edits, etc.) and
    ignores our own camera preview process so the debug window can never attract
    the live cursor.
    """

    screen: tuple[int, int] = (1920, 1080)
    max_depth: int = 8
    max_elements: int = 140
    control_types: tuple[str, ...] = (
        "ButtonControl", "MenuItemControl", "TabItemControl", "ListItemControl",
        "TreeItemControl", "DataItemControl", "EditControl", "HyperlinkControl",
        "CheckBoxControl", "RadioButtonControl", "SplitButtonControl",
    )
    _impl: object | None = None
    _last: list[Target] = field(default_factory=list)
    _window_token: tuple[int, str] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _wake: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _worker: threading.Thread | None = field(default=None, init=False, repr=False)
    _refresh_requested: bool = field(default=False, init=False, repr=False)
    _probe_points: list[tuple[float, float]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import uiautomation  # type: ignore

            self._impl = uiautomation
        except Exception:  # noqa: BLE001
            self._impl = None
            return
        # UI Automation is a COM-based API. Microsoft explicitly recommends
        # keeping UIA calls off the UI/input thread; the worker also prevents a
        # deep control-tree traversal from stalling camera inference.
        self._worker = threading.Thread(target=self._worker_loop, name="uia-targets", daemon=True)
        self._worker.start()

    @staticmethod
    def _rect_tuple(rect) -> tuple[float, float, float, float] | None:
        try:
            left, top, right, bottom = float(rect.left), float(rect.top), float(rect.right), float(rect.bottom)
        except Exception:  # noqa: BLE001
            try:
                left, top, right, bottom = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
            except Exception:  # noqa: BLE001
                return None
        if right <= left or bottom <= top:
            return None
        return left, top, right, bottom

    def set_probe_path(self, points: list[tuple[float, float]] | tuple[tuple[float, float], ...]) -> None:
        """Hint the worker with predicted desktop points along the cursor path.

        Point probing complements foreground-tree enumeration: title-bar buttons,
        desktop icons and other controls are often easier to identify exactly at
        a physical point than by walking a deep accessibility subtree.
        """
        if self._impl is None:
            return
        clean: list[tuple[float, float]] = []
        w, h = self.screen
        for x, y in points[:8]:
            px = float(np.clip(x, 0.0, max(w - 1, 0)))
            py = float(np.clip(y, 0.0, max(h - 1, 0)))
            clean.append((px, py))
        with self._lock:
            self._probe_points = clean
            self._refresh_requested = True
        self._wake.set()

    def refresh(self, timestamp: float) -> list[Target]:
        if self._impl is None:
            return []
        with self._lock:
            self._refresh_requested = True
            cached = list(self._last)
        self._wake.set()
        return cached

    def _worker_loop(self) -> None:
        # UI Automation is COM-based.  Give the worker its own apartment so a
        # slow accessibility tree walk cannot block the camera/input thread.
        co_initialized = False
        ole32 = None
        try:
            if os.name == "nt":
                import ctypes

                ole32 = ctypes.windll.ole32
                hr = int(ole32.CoInitializeEx(None, 0x2))  # COINIT_APARTMENTTHREADED
                co_initialized = hr in (0, 1, -2147417850)  # S_OK/S_FALSE/RPC_E_CHANGED_MODE fallback
            while not self._stop.is_set():
                self._wake.wait(0.20)
                self._wake.clear()
                if self._stop.is_set():
                    break
                with self._lock:
                    if not self._refresh_requested:
                        continue
                    self._refresh_requested = False
                try:
                    latest = self._refresh_sync()
                except Exception as exc:  # noqa: BLE001
                    log.debug("UIA background refresh failed: %s", exc)
                    continue
                with self._lock:
                    self._last = latest
        finally:
            if co_initialized and ole32 is not None:
                try:
                    ole32.CoUninitialize()
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _capabilities_for(control_type: str) -> tuple[str, ...]:
        base = ["hover", "click", "right_click", "double_click"]
        if control_type in {"ButtonControl", "MenuItemControl", "ListItemControl", "TreeItemControl", "TabItemControl"}:
            base.append("invoke")
        if control_type in {"EditControl", "ListItemControl", "TreeItemControl", "DataItemControl"}:
            base.append("drag")
        return tuple(dict.fromkeys(base))

    def _refresh_sync(self) -> list[Target]:
        if self._impl is None:
            return []
        try:
            uia = self._impl
            root = uia.GetForegroundControl()
            if root is None:
                with self._lock:
                    return list(self._last)

            # The camera preview is owned by this process.  Never expose it as
            # a target while the user is testing desktop control.
            pid = int(getattr(root, "ProcessId", -1) or -1)
            name = str(getattr(root, "Name", "") or "")
            token = (pid, name)
            own_foreground = pid == os.getpid()
            if own_foreground:
                # The preview may remain the foreground window while the real
                # pointer is already over another monitor/desktop region.  Do not
                # abort the refresh: point probing below is global to the desktop.
                self._window_token = token
            elif self._window_token is not None and token != self._window_token:
                # Foreground application changed: stale targets are more harmful
                # than a brief target-free interval.
                with self._lock:
                    self._last = []
            self._window_token = token

            w, h = self.screen
            with self._lock:
                probe_points = list(self._probe_points)
            out: list[Target] = []
            stack: list[tuple[object, int]] = [(root, 0)]
            seen: set[tuple[str, str, int, int, int, int]] = set()
            allow = set(self.control_types)

            # First ask UIA for the controls physically under the predicted path.
            # Microsoft exposes ElementFromPoint specifically in physical desktop
            # coordinates; this is the most direct spatial-intent primitive for a
            # real cursor, and it also reaches shell/title-bar elements that a
            # foreground subtree walk can miss.
            try:
                control_from_point = getattr(uia, "ControlFromPoint")
            except Exception:  # noqa: BLE001
                control_from_point = None

            def append_control(control: object | None) -> None:
                if control is None:
                    return
                current = control
                for _ in range(6):
                    if current is None:
                        break
                    try:
                        ctype = str(getattr(current, "ControlTypeName", "") or "")
                        cname = str(getattr(current, "Name", "") or "")
                        rect = self._rect_tuple(getattr(current, "BoundingRectangle", None))
                    except Exception:  # noqa: BLE001
                        rect = None
                        ctype = ""
                        cname = ""
                    if ctype in allow and rect is not None:
                        left, top, right, bottom = rect
                        rw, rh = right - left, bottom - top
                        if 8 <= rw <= w * 0.85 and 8 <= rh <= h * 0.55:
                            try:
                                pid2 = int(getattr(current, "ProcessId", -1) or -1)
                            except Exception:  # noqa: BLE001
                                pid2 = -1
                            if pid2 != os.getpid():
                                try:
                                    offscreen = bool(current.IsOffscreen)
                                except Exception:  # noqa: BLE001
                                    offscreen = False
                                try:
                                    enabled = bool(current.IsEnabled)
                                except Exception:  # noqa: BLE001
                                    enabled = True
                                if not offscreen and enabled:
                                    key = (ctype, cname, round(left / 4), round(top / 4), round(right / 4), round(bottom / 4))
                                    if key not in seen:
                                        seen.add(key)
                                        conf = 0.96 if ctype in {"ButtonControl", "MenuItemControl", "ListItemControl", "TreeItemControl", "TabItemControl"} else 0.86
                                        out.append(Target(
                                            id=f"uia:{abs(hash(key)) % 10_000_000}",
                                            bounds=(left / w, top / h, right / w, bottom / h),
                                            type=ctype.removesuffix("Control").lower(),
                                            semantic_label=cname,
                                            semantic_confidence=conf,
                                            visual_confidence=0.60,
                                            capabilities=self._capabilities_for(ctype),
                                        ))
                        # Once an allowed semantic target is found, do not walk into
                        # generic ancestors; the child at the point is the useful one.
                        if ctype in allow:
                            break
                    try:
                        current = current.GetParentControl()
                    except Exception:  # noqa: BLE001
                        break

            if control_from_point is not None:
                for px, py in probe_points:
                    try:
                        append_control(control_from_point(int(round(px)), int(round(py))))
                    except Exception:  # noqa: BLE001
                        continue
            if own_foreground:
                # Never enumerate our own preview tree, but keep the global point
                # probes collected above.
                with self._lock:
                    self._last = out[: self.max_elements]
                return out[: self.max_elements]

            while stack and len(out) < self.max_elements:
                control, depth = stack.pop()
                if depth >= self.max_depth:
                    continue
                try:
                    rect_obj = control.BoundingRectangle
                    rect = self._rect_tuple(rect_obj)
                    ctype = str(getattr(control, "ControlTypeName", "") or "")
                    name = str(getattr(control, "Name", "") or "")
                except Exception:  # noqa: BLE001
                    continue

                if rect is not None and ctype in allow:
                    left, top, right, bottom = rect
                    # Skip offscreen/invalid objects.  Keep a small tolerance for
                    # desktop and multi-monitor negative origins.
                    if -w * 0.05 <= left < w * 1.05 and -h * 0.05 <= top < h * 1.05:
                        rw, rh = right - left, bottom - top
                        if rw >= 8 and rh >= 8 and rw <= w * 0.85 and rh <= h * 0.55:
                            try:
                                offscreen = bool(control.IsOffscreen)
                            except Exception:  # noqa: BLE001
                                offscreen = False
                            try:
                                enabled = bool(control.IsEnabled)
                            except Exception:  # noqa: BLE001
                                enabled = True
                            if not offscreen and enabled:
                                key = (ctype, name, round(left / 4), round(top / 4), round(right / 4), round(bottom / 4))
                                if key not in seen:
                                    seen.add(key)
                                    conf = 0.92 if ctype in {"ButtonControl", "MenuItemControl", "ListItemControl", "TabItemControl"} else 0.82
                                    out.append(
                                        Target(
                                            id=f"uia:{abs(hash(key)) % 10_000_000}",
                                            bounds=(left / w, top / h, right / w, bottom / h),
                                            type=ctype.removesuffix("Control").lower(),
                                            semantic_label=name,
                                            semantic_confidence=conf,
                                            visual_confidence=0.55,
                                            capabilities=self._capabilities_for(ctype),
                                        )
                                    )
                try:
                    children = control.GetChildren()
                except Exception:  # noqa: BLE001
                    children = []
                for child in reversed(children):
                    stack.append((child, depth + 1))

            # Small controls are more useful than large containers.  Keep the
            # most interaction-relevant set when the tree is bigger than the cap.
            out.sort(key=lambda t: (t.quality(), -float(np.prod(t.size))), reverse=True)
            return out[: self.max_elements]
        except Exception as exc:  # noqa: BLE001
            log.debug("UIA refresh failed: %s", exc)
            with self._lock:
                return list(self._last)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=0.4)
        with self._lock:
            self._last = []



def uia_provider(screen: tuple[int, int] = (1920, 1080)) -> UIAProvider | None:
    provider = UIAProvider(screen=screen)
    if provider._impl is None:
        log.info("uiautomation not available; running without semantic targets")
        return None
    return provider


# --------------------------------------------------------------------------- #
# 2. Computer-vision fallback
# --------------------------------------------------------------------------- #


@dataclass
class CVProvider:
    """Screenshot + contour detection.

    Semantically blind by construction: it finds rectangles, nothing more.  Its
    confidence is capped at :attr:`max_confidence` so that semantic targets always
    win a fusion conflict.
    """

    cfg: EngineConfig
    screen: tuple[int, int] = (1920, 1080)
    max_confidence: float = 0.45
    min_area_frac: float = 0.0004
    max_targets: int = 24
    _last: list[Target] = field(default_factory=list)

    def refresh(self, timestamp: float) -> list[Target]:
        try:
            import cv2
            import mss  # type: ignore
        except Exception:  # noqa: BLE001
            return self._last

        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                shot = np.asarray(sct.grab(monitor))[:, :, :3]
        except Exception as exc:  # noqa: BLE001
            log.debug("screenshot failed: %s", exc)
            return self._last

        gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        h, w = shot.shape[:2]
        area_total = float(h * w)
        found: list[Target] = []
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area / area_total < self.min_area_frac or area / area_total > 0.4:
                continue
            aspect = bw / max(bh, 1)
            if aspect < 0.15 or aspect > 8.0:
                continue
            rect = (x / w, y / h, (x + bw) / w, (y + bh) / h)
            if any(bbox_iou(rect, t.bounds) > 0.5 for t in found):
                continue
            found.append(
                Target(
                    id=f"cv:{x}_{y}_{bw}_{bh}",
                    bounds=rect,
                    type="visual",
                    semantic_label="",
                    visual_confidence=float(np.clip(0.25 + area / area_total * 4.0, 0.2, self.max_confidence)),
                    semantic_confidence=0.0,
                )
            )
            if len(found) >= self.max_targets:
                break
        if found:
            self._last = found
        return self._last

    def close(self) -> None:
        pass


def cv_provider(cfg: EngineConfig, screen: tuple[int, int] = (1920, 1080)) -> CVProvider | None:
    try:
        import cv2  # noqa: F401
        import mss  # noqa: F401
    except Exception:  # noqa: BLE001
        log.info("cv target provider needs opencv-python and mss; falling back")
        return None
    return CVProvider(cfg=cfg, screen=screen)


# --------------------------------------------------------------------------- #
# 3. Static (default)
# --------------------------------------------------------------------------- #

#: Target layout used by the synthetic click study.  The coordinates match
#: ``gesture_engine.tracking.synthetic.click_study_scenario`` so that the
#: ground-truth "target" tag and the target model agree.
CLICK_STUDY_TARGETS: tuple[tuple[str, tuple[float, float, float, float]], ...] = (
    ("T0", (0.21, 0.31, 0.29, 0.39)),
    ("T1", (0.68, 0.26, 0.76, 0.34)),
    ("T2", (0.26, 0.64, 0.34, 0.72)),
    ("T3", (0.64, 0.58, 0.72, 0.66)),
    ("T4", (0.46, 0.24, 0.54, 0.32)),
    ("T5", (0.16, 0.51, 0.24, 0.59)),
)


def default_static_targets() -> list[Target]:
    return [
        Target(
            id=name,
            bounds=bounds,
            type="button",
            semantic_label=name,
            semantic_confidence=0.85,
            visual_confidence=0.7,
        )
        for name, bounds in CLICK_STUDY_TARGETS
    ]
