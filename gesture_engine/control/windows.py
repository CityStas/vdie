"""Window / desktop control.

Kept as a separate concern from the keyboard so that a future implementation can
use the Win32 API (``SetForegroundWindow``, ``ShowWindow``) instead of synthetic
key combos without touching the policy layer.
"""

from __future__ import annotations

import logging
from typing import Protocol

log = logging.getLogger(__name__)

#: Semantic window commands -> key combos.  Replaceable per platform.
DEFAULT_BINDINGS: dict[str, str] = {
    "next_window": "alt+tab",
    "prev_window": "alt+shift+tab",
    "maximize": "win+up",
    "minimize": "win+down",
    "snap_left": "win+left",
    "snap_right": "win+right",
    "show_desktop": "win+d",
    "task_view": "win+tab",
    "close": "alt+f4",
}


class WindowController(Protocol):
    def execute(self, command: str) -> bool: ...
    def active_window(self) -> str: ...
    def release_all(self) -> None: ...


class ComboWindowController:
    """Window commands implemented as key combos via a keyboard driver."""

    def __init__(self, keyboard, bindings: dict[str, str] | None = None) -> None:
        self.keyboard = keyboard
        self.bindings = dict(DEFAULT_BINDINGS)
        if bindings:
            self.bindings.update(bindings)
        self.log: list[str] = []

    def execute(self, command: str) -> bool:
        combo = self.bindings.get(command)
        if combo is None:
            log.warning("unknown window command %r", command)
            return False
        self.keyboard.combo(combo)
        self.log.append(command)
        return True

    def active_window(self) -> str:
        try:
            import pygetwindow  # type: ignore

            win = pygetwindow.getActiveWindow()
            return getattr(win, "title", "") or ""
        except Exception:  # noqa: BLE001
            return ""

    def release_all(self) -> None:
        self.keyboard.release_all()


class NullWindowController:
    def __init__(self) -> None:
        self.log: list[str] = []

    def execute(self, command: str) -> bool:
        self.log.append(command)
        return True

    def active_window(self) -> str:
        return "synthetic-window"

    def release_all(self) -> None:
        pass
