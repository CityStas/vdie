"""High-detail live diagnostic trace for camera/gesture/control debugging."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _clean(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


class DiagnosticTrace:
    """Append-only JSONL trace. One line per processed camera frame."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def open(self) -> None:
        self._fh = self.path.open("w", encoding="utf-8", buffering=1)

    def write(self, payload: dict[str, Any]) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(_clean(payload), ensure_ascii=False, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
