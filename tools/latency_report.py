"""End-to-end latency budget.

The master prompt requires the latency of the *whole* chain — physical hand to
OS action — measured and split per stage.  Part of that chain is measurable in
software and part of it is not, and conflating the two is how "we have 3 ms
latency" claims get made.

Measured here
-------------
* **per-stage wall clock** — tracking / features / motion / gesture / intent /
  policy / cursor / dispatch, mean and p95, on a real frame stream.
* **event-to-action latency** — how long after a ground-truth event (a labelled
  segment in a recording) the engine actually does something.  This is the
  number a user feels, and it is measurable without any hardware.

Not measured here
-----------------
* **camera capture latency** — the delay between the photons and ``frame.read()``
  returning.  Depends on the capture card, the exposure time and the driver
  buffering; it is typically 1-3 frames and it cannot be inferred from software.
* **physical reaction time** — the time between the user *deciding* and the hand
  actually moving.

Protocol for the hardware half
------------------------------
1. Put a bright LED in the frame next to the hand; drive it from the same script
   that generates the ground-truth event.
2. Record at the highest shutter speed the lighting allows (short exposure is
   what makes the LED edge sharp).
3. Run ``--source mediapipe --record`` while flashing the LED on the labelled
   segments.
4. The LED edge is visible in the recording at a known frame; the first frame in
   which the engine reports the corresponding action is in the same recording,
   so ``event_to_action_latency`` below *is* the physical number, up to one
   frame of camera exposure.
5. Capture latency alone is then ``event_to_action - pipeline``.

Usage::

    python -m tools.latency_report                     # synthetic, all scenarios
    python -m tools.latency_report --scenario click
    python -m tools.latency_report --recording recordings/session.jsonl
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from gesture_engine.bench.metrics import _pipeline_ms  # noqa: PLC2701 - shared definition
from gesture_engine.config import load_config
from gesture_engine.debug.recorder import load_recording
from gesture_engine.tracking.synthetic import SCENARIOS, get_scenario

STAGES = ("tracking", "features", "motion", "gesture", "intent", "policy", "cursor", "dispatch")


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def stage_table(reports) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for stage in STAGES:
        out[stage] = _percentiles([float(r.stage_ms.get(stage, 0.0)) for r in reports])
    out["pipeline_total"] = _percentiles([_pipeline_ms(r) for r in reports])
    return out


def event_to_action(reports) -> list[dict]:
    """Latency from a ground-truth segment start to the first action it produces.

    A segment counts as answered by the first action emitted between its start
    and its end plus a tolerance.  Segments that produce no action are reported
    with ``latency_ms = None`` — "nothing happened" and "it happened late" are
    different failures.
    """
    events: list[dict] = []
    if not reports:
        return events
    t0 = reports[0].timestamp
    spans: list[dict] = []
    for r in reports:
        label = str(r.notes.get("label", "") or "")
        if not label or label == "NONE":
            continue
        if spans and spans[-1]["label"] == label and abs(r.timestamp - spans[-1]["end"]) < 0.2:
            spans[-1]["end"] = r.timestamp
        else:
            spans.append({"label": label, "start": r.timestamp, "end": r.timestamp})

    for span in spans:
        found = None
        for r in reports:
            if r.timestamp < span["start"]:
                continue
            if r.timestamp > span["end"] + 0.5:
                break
            if r.actions:
                found = r.timestamp
                break
        events.append(
            {
                "label": span["label"],
                "start": round(span["start"] - t0, 3),
                "duration": round(span["end"] - span["start"], 3),
                "action_latency_ms": None if found is None else round((found - span["start"]) * 1000.0, 1),
            }
        )
    return events


def _run_synthetic(scenario: str, cfg, fps: int):
    from tools.scenario_report import run_scenario

    cfg.capture.fps = fps
    return run_scenario(scenario, cfg)


def _run_recording(path: str, cfg):
    from gesture_engine.control.dispatcher import ActionDispatcher
    from gesture_engine.debug.recorder import ReplayTracker
    from gesture_engine.engine import GestureEngine
    from gesture_engine.targets.providers import default_static_targets
    from gesture_engine.targets.target_model import StaticTargetProvider

    rec = load_recording(path)
    cfg = rec.config() or cfg
    cfg.control.enabled = False
    engine = GestureEngine(
        cfg,
        ReplayTracker(rec, use_landmarks=True),
        dispatcher=ActionDispatcher(cfg=cfg, dry_run=True),
        target_provider=StaticTargetProvider(default_static_targets()),
    )
    reports = []
    engine.start()
    try:
        while True:
            r = engine.process_frame(None, 0.0, len(reports))
            if r is None:
                break
            reports.append(r)
    finally:
        engine.stop()
    return reports


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="demo", choices=sorted(SCENARIOS))
    ap.add_argument("--recording", default=None, help="replay a recorded session instead of synthetic input")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    cfg = load_config(overrides={"control.enabled": False})
    reports = _run_recording(args.recording, cfg) if args.recording else _run_synthetic(args.scenario, cfg, args.fps)
    if not reports:
        print("no frames")
        return 1

    stages = stage_table(reports)
    events = event_to_action(reports)

    if args.json:
        print(json.dumps({"stages": stages, "events": events}, indent=2))
        return 0

    source = args.recording or f"synthetic:{args.scenario}"
    span = reports[-1].timestamp - reports[0].timestamp
    print(f"# latency budget — {source}  ({len(reports)} frames, {span:.2f} s @ {args.fps} FPS)")
    print()
    print(f"| stage | mean ms | p50 ms | p95 ms | max ms | share |")
    print("|---|---|---|---|---|---|")
    total = stages["pipeline_total"]["mean"] or 1e-9
    for stage in STAGES:
        s = stages[stage]
        print(f"| {stage} | {s['mean']:.3f} | {s['p50']:.3f} | {s['p95']:.3f} | {s['max']:.3f} | {100 * s['mean'] / total:.1f}% |")
    t = stages["pipeline_total"]
    print(f"| **pipeline total** | **{t['mean']:.3f}** | **{t['p50']:.3f}** | **{t['p95']:.3f}** | **{t['max']:.3f}** | 100% |")
    print()
    print(f"  budget at {args.fps} FPS: {1000.0 / args.fps:.2f} ms per frame "
          f"({100 * t['p95'] / (1000.0 / args.fps):.1f}% of it used at p95)")
    print()
    print("## event -> action latency (ground-truth segments)")
    print()
    print("| segment | starts at s | duration s | action latency ms |")
    print("|---|---|---|---|")
    for e in events:
        lat = "-" if e["action_latency_ms"] is None else f"{e['action_latency_ms']:.0f}"
        print(f"| {e['label']} | {e['start']:.2f} | {e['duration']:.2f} | {lat} |")
    answered = [e["action_latency_ms"] for e in events if e["action_latency_ms"] is not None]
    if answered:
        print()
        print(f"  median {np.median(answered):.0f} ms, p95 {np.percentile(answered, 95):.0f} ms, "
              f"unanswered {len(events) - len(answered)}/{len(events)}")
    print()
    print("## not measurable in software")
    print("  camera capture (photons -> frame.read)   : depends on exposure + driver buffering, typically 1-3 frames")
    print("  physical reaction (decision -> movement) : not an engineering property")
    print("  protocol: flash an LED in-frame on the labelled segments, record with the shortest")
    print("            shutter the light allows, then re-run this tool on the recording.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
