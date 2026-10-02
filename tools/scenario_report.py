"""Scenario inspector.

Runs every synthetic scenario through the full pipeline and prints what the
engine actually decided.  This is the tuning instrument: the benchmark table
tells you *how well* a variant scores, this tells you *what it did*.

Usage::

    python -m tools.scenario_report                     # all scenarios
    python -m tools.scenario_report swipe scroll        # a subset
    python -m tools.scenario_report swipe --trace       # per-frame trace
    python -m tools.scenario_report swipe --set commit.enabled=false
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter

from gesture_engine.bench.runner import BenchmarkRunner
from gesture_engine.config import load_config
from gesture_engine.control.dispatcher import ActionDispatcher
from gesture_engine.engine import GestureEngine
from gesture_engine.targets.providers import default_static_targets
from gesture_engine.targets.target_model import StaticTargetProvider
from gesture_engine.tracking.synthetic import SCENARIOS, SyntheticTracker, get_scenario


def run_scenario(name: str, cfg, screen=(1920, 1080), trace: bool = False):
    segments = get_scenario(name)
    fps = cfg.capture.fps
    tracker = SyntheticTracker(cfg, segments=segments, loop=False)
    engine = GestureEngine(
        cfg,
        tracker,
        dispatcher=ActionDispatcher(cfg=cfg, dry_run=True),
        target_provider=StaticTargetProvider(default_static_targets()),
        screen=screen,
    )

    total = sum(s.duration for s in segments)
    dt = 1.0 / fps
    reports = []
    engine.start()
    try:
        for i in range(int(total * fps)):
            r = engine.process_frame(None, i * dt, i, {"scenario": name})
            if r is None:
                break
            reports.append(r)
    finally:
        engine.stop()
    return reports


def summarise(name: str, reports) -> None:
    gestures = Counter(r.gesture.gesture for r in reports)
    states = Counter(r.state for r in reports)
    actions = Counter(a.kind for r in reports for a in r.actions)
    commits = sum(1 for r in reports if r.commit is not None and r.commit.committed)
    print(f"\n=== {name}  ({len(reports)} frames) ===")
    print(f"  actions : {dict(actions)}")
    print(f"  commits : {commits}")
    print(f"  gestures: {dict(gestures)}")
    print(f"  states  : {dict(states)}")


def trace(name: str, reports, every: int = 3) -> None:
    print(f"\n--- trace {name} (every {every} frames) ---")
    print(f"{'t':>6} {'state':<10} {'gesture':<14} {'conf':>5} {'stab':>5} {'spd':>6} {'pos':>18} {'intent':<14} {'p':>5}")
    for i, r in enumerate(reports):
        if i % every:
            continue
        intent = r.intents.dominant.value if r.intents else "-"
        prob = r.intents.dominant_probability if r.intents else 0.0
        pos = r.motion.position if r.motion else (0.0, 0.0)
        spd = r.motion.speed if r.motion else 0.0
        print(
            f"{r.timestamp:6.2f} {r.state:<10} {r.gesture.gesture:<14} "
            f"{r.gesture.confidence:5.2f} {r.gesture.stability:5.2f} {spd:6.2f} "
            f"({pos[0]:+.3f},{pos[1]:+.3f}) {intent:<14} {prob:5.2f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="*", default=None, help="scenario names (default: all)")
    ap.add_argument("--trace", action="store_true", help="print a per-frame trace")
    ap.add_argument("--every", type=int, default=3, help="trace stride")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    overrides = {"control.enabled": False, "capture.fps": args.fps}
    for item in args.overrides:
        key, _, value = item.partition("=")
        overrides[key.strip()] = value.strip()
    cfg = load_config(overrides=overrides)

    names = args.scenarios or list(SCENARIOS)
    for name in names:
        if name not in SCENARIOS:
            print(f"unknown scenario: {name} (known: {', '.join(SCENARIOS)})")
            continue
        reports = run_scenario(name, cfg)
        summarise(name, reports)
        if args.trace:
            trace(name, reports, every=args.every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
