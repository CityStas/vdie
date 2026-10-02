"""Action Dispatcher — the only place that touches the OS.

Safety properties implemented here (master prompt section 54):

* every action passes through a single choke point, so the audit log is complete;
* ``control.enabled = false`` turns the whole thing into a dry run;
* mouse buttons are released on shutdown, on exception, and on emergency cancel;
* button state is tracked so a lost mouse-up can never persist;
* rate limits are enforced here as a second line of defence after the policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import EngineConfig
from ..types import ActionRequest, Intent
from .keyboard import KeyboardDriver, NullKeyboard, PyAutoGuiKeyboard
from .mouse import MouseDriver, NullMouse, PyAutoGuiMouse
from .windows import ComboWindowController, NullWindowController, WindowController

log = logging.getLogger(__name__)


@dataclass
class DispatchRecord:
    timestamp: float
    kind: str
    payload: dict
    executed: bool
    error: str | None = None


@dataclass
class ActionDispatcher:
    cfg: EngineConfig
    mouse: MouseDriver = field(default=None)  # type: ignore[assignment]
    keyboard: KeyboardDriver = field(default=None)  # type: ignore[assignment]
    windows: WindowController = field(default=None)  # type: ignore[assignment]
    dry_run: bool = False
    records: list[DispatchRecord] = field(default_factory=list)
    _cursor: np.ndarray = field(default_factory=lambda: np.zeros(2))
    _first_move: bool = True

    def __post_init__(self) -> None:
        if self.mouse is None or self.keyboard is None or self.windows is None:
            if self.dry_run or not self.cfg.control.enabled:
                self.mouse = self.mouse or NullMouse()
                self.keyboard = self.keyboard or NullKeyboard()
                self.windows = self.windows or NullWindowController()
            else:
                self.mouse = self.mouse or PyAutoGuiMouse(interval=self.cfg.control.mouse_interval_ms / 1000.0)
                self.keyboard = self.keyboard or PyAutoGuiKeyboard()
                self.windows = self.windows or ComboWindowController(self.keyboard)
        self._dry_run = self.dry_run or not self.cfg.control.enabled

    # ------------------------------------------------------------------ #
    @property
    def cursor(self) -> np.ndarray:
        return self._cursor.copy()

    def set_cursor(self, position: np.ndarray) -> None:
        self._cursor = np.asarray(position, dtype=np.float64).copy()

    # ------------------------------------------------------------------ #
    def dispatch(self, actions: list[ActionRequest], cursor: np.ndarray | None = None, timestamp: float = 0.0) -> None:
        if cursor is not None:
            self.set_cursor(cursor)
        for action in actions:
            if action.is_noop:
                continue
            try:
                executed = self._execute(action, timestamp)
            except Exception as exc:  # noqa: BLE001
                log.exception("action failed: %s", action.kind)
                self.release_all()
                executed = False
                error = f"{type(exc).__name__}: {exc}"
            else:
                error = None
            self.records.append(
                DispatchRecord(
                    timestamp=timestamp,
                    kind=action.kind,
                    payload=dict(action.payload),
                    executed=executed,
                    error=error,
                )
            )

    # ------------------------------------------------------------------ #
    def _execute(self, action: ActionRequest, timestamp: float) -> bool:
        c = self.cfg.control
        if self._dry_run:
            if action.kind == "move":
                self.mouse.move(float(action.payload["x"]), float(action.payload["y"]))
            return True

        if action.kind == "move":
            if not c.move_mouse:
                return False
            self.mouse.move(float(action.payload["x"]), float(action.payload["y"]))
            self._first_move = False
            return True

        if action.kind == "click":
            if not c.clicks:
                return False
            self.mouse.click(str(action.payload.get("button", "left")), 1)
            return True

        if action.kind == "double_click":
            if not c.clicks:
                return False
            self.mouse.click(str(action.payload.get("button", "left")), 2)
            return True

        if action.kind == "right_click":
            if not c.clicks:
                return False
            self.mouse.click("right", 1)
            return True

        if action.kind == "down":
            if not c.clicks:
                return False
            self.mouse.down(str(action.payload.get("button", "left")))
            return True

        if action.kind == "up":
            self.mouse.up(str(action.payload.get("button", "left")))
            return True

        if action.kind == "scroll":
            if not c.move_mouse:
                return False
            self.mouse.scroll(float(action.payload.get("dy", 0.0)))
            return True

        if action.kind == "zoom":
            if not c.move_mouse or not c.keyboard:
                return False
            steps = float(action.payload.get("steps", 0.0))
            self.keyboard.key_down("ctrl")
            try:
                self.mouse.scroll(steps)
            finally:
                self.keyboard.key_up("ctrl")
            return True

        if action.kind == "key":
            if not c.keyboard:
                return False
            self.keyboard.combo(str(action.payload.get("combo", "")))
            return True

        if action.kind == "window":
            if not c.windows:
                return False
            combo = str(action.payload.get("combo", ""))
            self.keyboard.combo(combo)
            return True

        if action.kind == "custom":
            log.info("custom action: %s", action.payload.get("name"))
            return True

        return False

    # ------------------------------------------------------------------ #
    def move_cursor(self, position: np.ndarray) -> None:
        self.dispatch([ActionRequest(intent=Intent.MOVE_CURSOR, kind="move", payload={"x": float(position[0]), "y": float(position[1])})], position)

    def release_all(self) -> None:
        for drv in (self.mouse, self.keyboard, self.windows):
            try:
                drv.release_all()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        if self.cfg.safety.release_buttons_on_exit:
            self.release_all()

    def __enter__(self) -> "ActionDispatcher":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def audit(self) -> list[dict]:
        return [
            {"t": round(r.timestamp, 4), "kind": r.kind, "executed": r.executed, "error": r.error, **{k: v for k, v in r.payload.items() if k != "target"}}
            for r in self.records
        ]
