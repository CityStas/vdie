"""Command-line entry point.

On Windows, ``run.cmd`` wraps all of this and picks the venv itself, so the
shortest form is ``run --source mediapipe``.  No ``PYTHONPATH`` is needed from
the project root: running ``app.py`` here already puts the root on ``sys.path``.
It is required only when executing a script that lives in a subdirectory
*directly* (``python tools\\scenario_report.py``); use ``python -m
tools.scenario_report`` instead and the problem does not arise.

    python app.py --source synthetic --scenario demo --headless
    python app.py --source synthetic --scenario click --record out
    python app.py --source replay --recording recordings/click-*.jsonl --no-control
    python app.py --bench --scenario click           # algorithm matrix
    python app.py --time-pipeline                    # per-stage cost, no camera

Real camera, in the order worth running it:

    python app.py --list-cameras --backend msmf      # 1. which index, what fps
    python app.py --source mediapipe --no-control --headless --duration 10
                                                     # 2. dry run: engine + tracking, no OS input
    python app.py --source mediapipe                 # 3. live, with overlay and OS control

Step 1 matters because a backend can open a device and still deliver a third of
the frames (measured: dshow 10.5 fps vs msmf 32.6 fps at 1280x720) — `CAP_PROP_FPS`
reports the requested value in both cases, so only a timed read exposes it.
Step 2 matters because it is the only cheap way to tell "the engine is broken"
apart from "the hand was not in frame"; it needs no OS permissions.
Step 3 moves the real mouse: a closed fist is the emergency cancel.

Everything is overridable with ``--set a.b=c`` so no threshold has to be edited in
a file to run an experiment.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gesture_engine import __version__, build_dispatcher, build_target_provider, build_tracker, load_config
from gesture_engine.bench.metrics import format_table
from gesture_engine.bench.runner import BenchmarkRunner, time_pipeline
from gesture_engine.capture.camera import CameraSource, list_cameras
from gesture_engine.config import apply_overrides
from gesture_engine.win32 import make_process_dpi_aware
from gesture_engine.control.dispatcher import ActionDispatcher
from gesture_engine.debug.overlay import Overlay
from gesture_engine.debug.recorder import Recorder, ReplayTracker, load_recording
from gesture_engine.debug.diagnostic import DiagnosticTrace
from gesture_engine.engine import GestureEngine
from gesture_engine.tracking.synthetic import get_scenario


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser("intent-gesture-engine", description="Intent-Based Gesture Control Engine")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--config", type=str, default=None, help="path to config.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config value, e.g. --set cursor.sensitivity=1.4")
    p.add_argument("--source", choices=["mediapipe", "synthetic", "replay", "null"], default="synthetic")
    p.add_argument("--scenario", default="demo", choices=["demo", "click", "swipe", "scroll", "loss"])
    p.add_argument("--recording", type=str, default=None, help="recording path for --source replay")
    p.add_argument("--device", type=int, default=None, help="camera device index")
    p.add_argument("--backend", type=str, default=None, choices=["any", "msmf", "dshow"], help="camera backend")
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    p.add_argument("--headless", action="store_true", help="no OpenCV window")
    p.add_argument("--no-control", action="store_true", help="disable all OS input (dry run)")
    p.add_argument("--mode", choices=["touch_surface", "cooperative", "gain", "velocity_curve", "spring"], default=None, help="cursor controller mode")
    p.add_argument("--no-bimanual", action="store_true", help="disable two-hand interactions")
    p.add_argument("--record", type=str, nargs="?", const="recordings", default=None, help="record the session to this directory")
    p.add_argument("--bench", action="store_true", help="run the algorithm benchmark matrix")
    p.add_argument("--bench-replay", type=str, default=None, help="benchmark several algorithm versions on one recording")
    p.add_argument("--time-pipeline", action="store_true", help="measure per-stage cost and exit")
    p.add_argument("--list-cameras", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--diagnostic", type=str, default=None, help="write one JSON line per frame with full tracking/FSM/cursor/dispatcher diagnostics")
    return p.parse_args(argv)


def _apply_cli_overrides(cfg, args) -> None:
    overrides: dict[str, object] = {}
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = _coerce_scalar(v.strip())
    if args.device is not None:
        overrides["capture.device"] = args.device
    if args.backend is not None:
        overrides["capture.backend"] = args.backend
    if args.fps is not None:
        overrides["capture.fps"] = args.fps
    if args.no_control:
        overrides["control.enabled"] = False
    if args.mode:
        overrides["cursor.controller"] = args.mode
        if args.mode == "touch_surface":
            overrides["cursor.relative_mode"] = False
        elif args.mode == "cooperative":
            overrides["cursor.relative_mode"] = True
    if args.no_bimanual:
        overrides["bimanual.enabled"] = False
    if args.headless:
        overrides["debug.overlay"] = False
    if args.record is not None:
        overrides["record.directory"] = args.record
    if overrides:
        apply_overrides(cfg, overrides)


def _coerce_scalar(text: str):
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        if "." in text or "e" in low:
            return float(text)
        return int(text)
    except ValueError:
        if text.startswith("[") and text.endswith("]"):
            return [float(x) for x in text[1:-1].split(",") if x.strip()]
        return text


def _build_engine(cfg, args, source: str | None = None):
    source = source or args.source
    if source == "replay":
        if not args.recording:
            raise SystemExit("--source replay requires --recording PATH")
        tracker = ReplayTracker(load_recording(args.recording))
    else:
        tracker = build_tracker(cfg, source=source, scenario=args.scenario)

    dispatcher = build_dispatcher(cfg, dry_run=args.no_control or source in ("synthetic", "replay", "null"))
    recorder = None
    if args.record is not None:
        recorder = Recorder(cfg=cfg, name=args.scenario if source != "replay" else "replay", label=args.scenario)
    diagnostic = DiagnosticTrace(args.diagnostic) if args.diagnostic else None
    engine = GestureEngine(
        cfg,
        tracker,
        dispatcher=dispatcher,
        recorder=recorder,
        target_provider=build_target_provider(cfg, screen=dispatcher.mouse.screen_size()),
        diagnostic=diagnostic,
    )
    return engine, dispatcher


def run_loop(cfg, args) -> int:
    engine, dispatcher = _build_engine(cfg, args)
    overlay = Overlay(cfg) if cfg.debug.overlay and not args.headless else None
    camera = CameraSource(cfg) if args.source == "mediapipe" else None
    window = cfg.debug.window_name

    frames = 0
    engine.start()
    try:
        if camera is not None:
            camera.open()
            if overlay is not None:
                import cv2
                flags = cv2.WINDOW_NORMAL
                if hasattr(cv2, "WINDOW_KEEPRATIO"):
                    flags |= cv2.WINDOW_KEEPRATIO
                cv2.namedWindow(window, flags)
                cv2.resizeWindow(
                    window,
                    max(320, int(cfg.debug.window_width)),
                    max(240, int(cfg.debug.window_height)),
                )
        while True:
            if camera is not None:
                frame_obj = camera.read()
                image, timestamp, meta = frame_obj.image, frame_obj.timestamp, frame_obj.meta
            else:
                # Synthetic / null sources are driven by the *nominal* frame clock,
                # not wall time, so a run is deterministic and independent of how
                # fast the machine happens to execute it.
                timestamp = frames / max(cfg.capture.fps, 1)
                image, meta = None, {"profile": "synthetic"}

            report = engine.process_frame(image, timestamp, frames, meta)
            if report is None:
                break
            frames += 1

            if not args.quiet and frames % max(int(cfg.capture.fps * 0.5), 1) == 0:
                print(Overlay.render_text(report, engine.telemetry.lines()), flush=True)

            if overlay is not None and image is not None:
                import cv2

                vis = overlay.draw(image, report, engine.telemetry.lines(), engine.motion.recent_points(0.75), report.notes.get("grammar_partial", ""))
                cv2.imshow(window, vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

            if args.frames is not None and frames >= args.frames:
                break
            if args.duration is not None:
                # Synthetic / replay sources run on the nominal frame clock (see
                # above), so the limit must be evaluated against the same clock.
                # Using wall time here made a 6 s run execute 129 s worth of
                # frames whenever the machine ran slower than real time.
                elapsed = timestamp if camera is not None else frames / max(cfg.capture.fps, 1)
                if elapsed >= args.duration:
                    break
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
    finally:
        engine.stop()
        if camera is not None:
            camera.close()
        if overlay is not None:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass

    if args.diagnostic:
        print(f"diagnostic log: {Path(args.diagnostic).resolve()}")
    print("\n=== telemetry ===")
    summary = engine.telemetry.summary()
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\n=== engine ===")
    print(
        f"frames={engine.stats.frames} hands={engine.stats.hands_seen} "
        f"activations={engine.stats.activations} commits={engine.stats.commits} actions={engine.stats.actions}"
    )
    if engine.policy.decision_log():
        print(f"policy decisions: {len(engine.policy.decision_log())} (last 5: {engine.policy.decision_log()[-5:]})")
    if engine.dispatcher.records:
        print(f"dispatched: {len(engine.dispatcher.records)} (sample: {engine.dispatcher.audit()[:5]})")
    return 0


def main(argv: list[str] | None = None) -> int:
    # UI Automation bounding rectangles and physical cursor coordinates are
    # reported in physical desktop pixels. Make the process DPI-aware before
    # constructing the live dispatcher/provider so both layers share a space.
    make_process_dpi_aware()
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)

    if args.list_cameras:
        # Probed with the *configured* backend, because that is what the engine
        # will actually use.  The probe reads and times real frames, so a backend
        # that opens but delivers a third of the frames shows up as low fps
        # instead of looking healthy.
        print(f"# probing indices 0..5 with backend={cfg.capture.backend!r} "
              f"(requesting {cfg.capture.width}x{cfg.capture.height} @{cfg.capture.fps})")
        any_found = False
        for cam in list_cameras(
            backend=cfg.capture.backend,
            width=cfg.capture.width,
            height=cfg.capture.height,
            fps=cfg.capture.fps,
        ):
            if cam["opened"] and cam.get("frames"):
                any_found = True
                print(
                    f"  index {cam['index']}: {cam['width']}x{cam['height']} "
                    f"@ {cam['fps']:.1f} fps (measured over {cam['frames']} frames, "
                    f"backend={cam['backend']})"
                )
            elif cam["opened"]:
                print(f"  index {cam['index']}: opened but delivered no frame (backend={cam['backend']})")
        if not any_found:
            print("  no camera delivered frames — try --backend msmf, or another --device index")

        return 0

    if args.time_pipeline:
        print(time_pipeline(cfg, frames=args.frames or 300, fps=cfg.capture.fps))
        return 0

    if args.bench:
        runner = BenchmarkRunner(cfg, fps=cfg.capture.fps)
        table = runner.run_matrix(scenario=args.scenario, duration=args.duration)
        print(f"# scenario: {args.scenario}  fps: {cfg.capture.fps}\n")
        print(format_table(table))
        print("\n# wall-clock latency (not part of the reproducible table)")
        for name, t in runner.timing_report().items():
            stages = " ".join(f"{k}={v:.2f}" for k, v in sorted(t["stage_ms"].items()))
            print(f"  {name:28s} p50={t['pipeline_ms_p50']:.2f}ms p95={t['pipeline_ms_p95']:.2f}ms  {stages}")
        return 0

    if args.bench_replay:
        runner = BenchmarkRunner(cfg, fps=cfg.capture.fps)
        table = runner.replay_matrix(args.bench_replay)
        print(format_table(table))
        return 0

    return run_loop(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
