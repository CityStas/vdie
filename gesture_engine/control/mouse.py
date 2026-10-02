"""Mouse driver abstraction.

Two implementations: a real one on top of ``pyautogui`` and a null one used for
dry runs and benchmarks.  The engine never imports ``pyautogui`` directly, so a
benchmark can run the full pipeline headlessly with zero OS side effects.
"""

from __future__ import annotations

import logging
import sys
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class MouseDriver(Protocol):
    def screen_size(self) -> tuple[int, int]: ...
    def position(self) -> np.ndarray: ...
    def move(self, x: float, y: float) -> None: ...
    def click(self, button: str = "left", clicks: int = 1) -> None: ...
    def down(self, button: str = "left") -> None: ...
    def up(self, button: str = "left") -> None: ...
    def scroll(self, dy: float) -> None: ...
    def release_all(self) -> None: ...


class NullMouse:
    """Records calls instead of performing them."""

    def __init__(self, screen: tuple[int, int] = (1920, 1080)) -> None:
        self._screen = screen
        self.calls: list[tuple[str, tuple]] = []
        self.position = np.zeros(2)
        self.held: set[str] = set()

    def screen_size(self) -> tuple[int, int]:
        return self._screen

    def position(self) -> np.ndarray:
        return self.position.copy()

    def move(self, x: float, y: float) -> None:
        self.position = np.array([x, y], dtype=np.float64)
        self.calls.append(("move", (round(x, 2), round(y, 2))))

    def click(self, button: str = "left", clicks: int = 1) -> None:
        self.calls.append(("click", (button, clicks)))

    def down(self, button: str = "left") -> None:
        self.held.add(button)
        self.calls.append(("down", (button,)))

    def up(self, button: str = "left") -> None:
        self.held.discard(button)
        self.calls.append(("up", (button,)))

    def scroll(self, dy: float) -> None:
        self.calls.append(("scroll", (round(dy, 2),)))

    def release_all(self) -> None:
        for b in list(self.held):
            self.up(b)


class PyAutoGuiMouse:
    """Real driver.  Imported lazily so the engine works without it."""

    def __init__(self, move_duration: float = 0.0, interval: float = 0.0) -> None:
        try:
            import pyautogui  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyautogui is required for real OS control (`pip install pyautogui`)") from exc
        self._pag = pyautogui
        self._pag.FAILSAFE = True
        self._pag.PAUSE = interval
        self._duration = move_duration
        self._held: set[str] = set()
        self._screen = tuple(self._pag.size())  # type: ignore[arg-type]
        if sys.platform == "win32":
            try:
                from ..win32 import make_process_dpi_aware
                make_process_dpi_aware()
                user32 = __import__("ctypes").windll.user32
                self._screen = (int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1)))
            except Exception:
                pass

    def screen_size(self) -> tuple[int, int]:
        return (int(self._screen[0]), int(self._screen[1]))

    def position(self) -> np.ndarray:
        if sys.platform == "win32":
            try:
                import ctypes
                class POINT(ctypes.Structure):
                    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
                pt = POINT()
                if ctypes.windll.user32.GetPhysicalCursorPos(ctypes.byref(pt)):
                    return np.array([float(pt.x), float(pt.y)], dtype=np.float64)
            except Exception:
                pass
        p = self._pag.position()
        return np.array([float(p.x), float(p.y)], dtype=np.float64)

    def move(self, x: float, y: float) -> None:
        w, h = self.screen_size()
        # Never command an exact screen corner.  PyAutoGUI intentionally uses
        # the four corners as its emergency-stop points.  The gesture engine
        # may legitimately map a hand close to an edge, so leave a 1 px safety
        # margin while preserving PyAutoGUI's fail-safe for clicks/other input.
        cx = float(np.clip(x, 1, max(1, w - 2)))
        cy = float(np.clip(y, 1, max(1, h - 2)))

        # PyAutoGUI checks the *current* cursor position before every move.
        # If the user starts the live mode with Windows' cursor already in a
        # fail-safe corner, even a perfectly safe first target is rejected.
        # Recover only that one condition through the native Windows API; do
        # not globally disable FAILSAFE.  Subsequent moves go through
        # PyAutoGUI normally, so the emergency corner remains meaningful.
        try:
            current = self._pag.position()
            corners = {(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)}
            if (int(current.x), int(current.y)) in corners and sys.platform == "win32":
                import ctypes
                if ctypes.windll.user32.SetCursorPos(int(cx), int(cy)):
                    return
        except Exception:  # noqa: BLE001
            log.debug("native corner recovery unavailable", exc_info=True)

        if self._duration > 0:
            self._pag.moveTo(cx, cy, duration=self._duration)
        else:
            self._pag.moveTo(cx, cy, _pause=False)

    def click(self, button: str = "left", clicks: int = 1) -> None:
        self._pag.click(button=button, clicks=clicks, _pause=False)

    def down(self, button: str = "left") -> None:
        self._pag.mouseDown(button=button, _pause=False)
        self._held.add(button)

    def up(self, button: str = "left") -> None:
        self._pag.mouseUp(button=button, _pause=False)
        self._held.discard(button)

    def scroll(self, dy: float) -> None:
        # pyautogui takes integer "clicks"; we forward the fractional part
        # accumulated across frames so slow scrolling is not quantised away.
        self._acc = getattr(self, "_acc", 0.0) + float(dy)
        whole = int(self._acc)
        if whole != 0:
            self._pag.scroll(whole, _pause=False)
            self._acc -= whole

    def release_all(self) -> None:
        for b in list(self._held):
            try:
                self.up(b)
            except Exception:  # noqa: BLE001
                log.warning("failed to release mouse button %s", b)
