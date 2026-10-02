"""Telemetry: latency breakdown, FPS, resource usage, interaction counters.

The master prompt requires separating capture / tracking / feature / recognition /
intent / filter / OS-input latency.  That split is only useful if it is measured
the same way everywhere, so all timing goes through :class:`StageTimer`.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

STAGES = (
    "capture",
    "tracking",
    "features",
    "motion",
    "events",
    "gesture",
    "targets",
    "commit",
    "intent",
    "policy",
    "cursor",
    "dispatch",
    "overlay",
)


class StageTimer:
    """Collects per-stage wall-clock durations for one frame."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] = self.times.get(name, 0.0) + (time.perf_counter() - start) * 1000.0

    def mark(self, name: str, ms: float) -> None:
        self.times[name] = self.times.get(name, 0.0) + ms

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 3) for k, v in self.times.items()}


@dataclass
class Telemetry:
    window: int = 120
    fps_history: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    source_fps_history: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    frame_times: deque[float] = field(default_factory=lambda: deque(maxlen=120))
    stage_times: dict[str, deque[float]] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    _last_ts: float | None = None
    _last_wall: float | None = None
    _last_report: float = 0.0
    _psutil: object | None = None
    _proc: object | None = None

    def __post_init__(self) -> None:
        for s in STAGES:
            self.stage_times[s] = deque(maxlen=self.window)
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._proc = psutil.Process()
        except Exception:  # noqa: BLE001
            self._psutil = None

    # ------------------------------------------------------------------ #
    def tick(self, timestamp: float, timer: StageTimer) -> float:
        """Record one frame.

        Two different "FPS" numbers matter and they are not the same thing:

        * ``fps`` — real throughput (wall clock).  This is the performance number.
        * ``source_fps`` — the rate of the *observation stream*.  For a synthetic
          or replayed source this is the nominal rate, not the execution speed.
        """
        wall = time.perf_counter()
        dt_wall = 0.0 if self._last_wall is None else max(wall - self._last_wall, 1e-9)
        self._last_wall = wall
        if dt_wall > 0:
            self.fps_history.append(1.0 / dt_wall)

        dt_src = 0.0 if self._last_ts is None else max(timestamp - self._last_ts, 1e-6)
        self._last_ts = timestamp
        if dt_src > 0:
            self.source_fps_history.append(1.0 / dt_src)

        self.frame_times.append(timer.total_ms)
        for name, value in timer.times.items():
            self.stage_times.setdefault(name, deque(maxlen=self.window)).append(value)
        return 1.0 / dt_wall if dt_wall > 0 else 0.0

    def count(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    # ------------------------------------------------------------------ #
    @property
    def fps(self) -> float:
        return float(statistics.fmean(self.fps_history)) if self.fps_history else 0.0

    @property
    def source_fps(self) -> float:
        return float(statistics.fmean(self.source_fps_history)) if self.source_fps_history else 0.0

    @property
    def frame_ms(self) -> float:
        return float(statistics.fmean(self.frame_times)) if self.frame_times else 0.0

    def stage_mean(self, name: str) -> float:
        d = self.stage_times.get(name)
        return float(statistics.fmean(d)) if d else 0.0

    def stage_p95(self, name: str) -> float:
        d = self.stage_times.get(name)
        return float(np.percentile(np.asarray(d), 95)) if d else 0.0

    def resources(self) -> dict[str, float]:
        if self._proc is None:
            return {}
        try:
            import os

            with self._proc.oneshot():  # type: ignore[attr-defined]
                mem = self._proc.memory_info().rss / (1024 * 1024)
                cpu = self._proc.cpu_percent(interval=None) / max(os.cpu_count() or 1, 1)
            return {"cpu_pct": float(cpu), "rss_mb": float(mem)}
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------ #
    def summary(self) -> dict:
        stage_means = {s: round(self.stage_mean(s), 3) for s in self.stage_times if self.stage_times[s]}
        stage_p95 = {s: round(self.stage_p95(s), 3) for s in self.stage_times if self.stage_times[s]}
        pipeline_ms = sum(stage_means.get(s, 0.0) for s in ("capture", "tracking", "features", "motion", "gesture", "intent", "policy", "cursor"))
        return {
            "fps": round(self.fps, 2),
            "source_fps": round(self.source_fps, 2),
            "frame_ms": round(self.frame_ms, 3),
            "pipeline_ms": round(pipeline_ms, 3),
            "stages_ms": stage_means,
            "stages_p95_ms": stage_p95,
            "counters": dict(self.counters),
            "resources": self.resources(),
        }

    def lines(self) -> list[str]:
        out = [
            f"FPS {self.fps:5.1f}   frame {self.frame_ms:5.1f} ms",
            "stages: " + "  ".join(f"{s[:5]} {self.stage_mean(s):.1f}" for s in STAGES if self.stage_mean(s) > 0),
        ]
        res = self.resources()
        if res:
            out.append(f"CPU {res.get('cpu_pct', 0):.0%}  RAM {res.get('rss_mb', 0):.0f} MB")
        return out
