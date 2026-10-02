"""Keyboard driver.

Combos are described as strings ("alt+tab", "win+up") so that the action policy
never encodes a platform detail.  ``win`` is translated to the Windows key, which
``pyautogui`` calls ``win``; ``cmd``/``super`` are accepted aliases.
"""

from __future__ import annotations

from typing import Protocol

_KEY_ALIASES = {
    "win": "win",
    "super": "win",
    "meta": "win",
    "cmd": "win",
    "ctrl": "ctrl",
    "control": "ctrl",
    "alt": "alt",
    "shift": "shift",
    "esc": "esc",
    "enter": "enter",
    "tab": "tab",
    "space": "space",
}


class KeyboardDriver(Protocol):
    def combo(self, combo: str) -> None: ...
    def tap(self, key: str) -> None: ...
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def release_all(self) -> None: ...


class NullKeyboard:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def combo(self, combo: str) -> None:
        self.calls.append(combo)

    def tap(self, key: str) -> None:
        self.calls.append(key)

    def key_down(self, key: str) -> None:
        self.calls.append(f"down:{key}")

    def key_up(self, key: str) -> None:
        self.calls.append(f"up:{key}")

    def release_all(self) -> None:
        pass


class PyAutoGuiKeyboard:
    def __init__(self, interval: float = 0.0) -> None:
        try:
            import pyautogui  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyautogui is required for real OS control") from exc
        self._pag = pyautogui
        self._pag.PAUSE = interval
        self._held: list[str] = []

    @staticmethod
    def parse(combo: str) -> list[str]:
        parts = [p.strip().lower() for p in combo.replace(" ", "").split("+") if p.strip()]
        return [_KEY_ALIASES.get(p, p) for p in parts]

    def combo(self, combo: str) -> None:
        keys = self.parse(combo)
        if not keys:
            return
        for k in keys[:-1]:
            self._pag.keyDown(k)
            self._held.append(k)
        try:
            self._pag.press(keys[-1])
        finally:
            for k in reversed(keys[:-1]):
                self._pag.keyUp(k)
                if k in self._held:
                    self._held.remove(k)

    def tap(self, key: str) -> None:
        self._pag.press(_KEY_ALIASES.get(key.lower(), key))

    def key_down(self, key: str) -> None:
        mapped = _KEY_ALIASES.get(key.lower(), key)
        self._pag.keyDown(mapped)
        self._held.append(mapped)

    def key_up(self, key: str) -> None:
        mapped = _KEY_ALIASES.get(key.lower(), key)
        self._pag.keyUp(mapped)
        if mapped in self._held:
            self._held.remove(mapped)

    def release_all(self) -> None:
        for k in reversed(self._held):
            try:
                self._pag.keyUp(k)
            except Exception:  # noqa: BLE001
                pass
        self._held.clear()
