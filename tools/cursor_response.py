"""Cursor-controller frequency response and transient probe.

This is the **instrument of record for cursor latency**.  It drives
``CursorController`` with a known stimulus and measures phase, amplitude and
settling directly, with the whole pipeline out of the picture.

Why this and not an in-session metric: the cursor is a *filter*, not a delay.
The One-Euro pre-filter, the speed-scheduled spring and the feed-forward each
contribute frequency-dependent phase, so a session cannot be reduced to one lag
number.  ``gesture_engine.bench.metrics.cursor_lag`` measures a session and
correctly refuses to answer when the strokes disagree — which, for a
gain-scheduled controller, they do.  A transfer function has to be measured
against a known input.

Stimuli
-------
* ramp  — hand travels 0.30 -> 0.70 in 0.30 s then holds; time for the cursor to
          reach 90 % of that travel, measured from the moment the hand arrives.
* sine  — hand oscillates at f Hz; phase lag and amplitude ratio of the cursor.
* flick — hand sweeps fast then holds; arrival, overshoot and settling time.
* tremor— hand stationary but shaky (9.5 Hz physiological + white); cursor RMS.

The sine is the load-bearing measurement and the others are context.  All of
them run in the *moving* regime (speed scheduler saturated, intent field peaked)
because that is the regime where lag is felt; ``--flat-field`` shows the
hesitation penalty instead.

Usage::

    python -m tools.cursor_response                       # controller comparison
    python -m tools.cursor_response --sweep-euro --tremor # filter trade-off
    python -m tools.cursor_response --freqs 0.5,1,2,4     # custom sweep
"""

from __future__ import annotations

import argparse
import copy
import math

import numpy as np

from gesture_engine.config import EngineConfig, load_config
from gesture_engine.control.cursor import CursorController
from gesture_engine.types import Intent, IntentField, MotionState


class _FakeFSM:
    cursor_enabled = True
    dragging = False


def _motion(point: np.ndarray, velocity: np.ndarray) -> MotionState:
    return MotionState(
        position=point,
        velocity=velocity,
        acceleration=np.zeros(2),
        speed=float(np.linalg.norm(velocity)),
        confidence=1.0,
    )


def _field(flat: bool) -> IntentField:
    """Build an IntentField with a *correctly computed* entropy.

    ``IntentField`` is a plain dataclass — entropy is not derived from the
    probabilities, so a hand-built field silently carries ``entropy == 0`` and
    the cursor's hesitation gain disappears.  Compute it here.
    """
    if flat:
        probs = {i: 1.0 / len(Intent) for i in Intent}
    else:
        probs = {i: 0.02 for i in Intent}
        probs[Intent.MOVE_CURSOR] = 0.9
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0.0)
    dom = max(probs.items(), key=lambda kv: kv[1])
    return IntentField(
        probabilities=probs,
        entropy=entropy,
        dominant=dom[0],
        dominant_probability=dom[1],
    )


def _drive(ctl: CursorController, point: np.ndarray, velocity: np.ndarray, dt: float, t: float, field: IntentField) -> None:
    ctl.update(_motion(point, velocity), None, field, [], _FakeFSM(), dt, t)  # type: ignore[arg-type]


def _screen(ctl: CursorController, normalized: np.ndarray) -> np.ndarray:
    """Where the controller would settle for this hand point (open loop)."""
    curve = np.array(
        [
            ctl._sensitivity_curve(2.0 * (normalized[0] - 0.5)) * 0.5 + 0.5,
            ctl._sensitivity_curve(2.0 * (normalized[1] - 0.5)) * 0.5 + 0.5,
        ]
    )
    return ctl._map_to_screen(curve)


def ramp_lag(cfg: EngineConfig, fps: float, field: IntentField, travel: float = 0.30) -> tuple[float, float]:
    """(t90_ms, t99_ms) after the hand stops, in ms."""
    ctl = CursorController(cfg, screen=(1920, 1080))
    dt = 1.0 / fps
    start = np.array([0.5 - travel / 2, 0.5])
    end = np.array([0.5 + travel / 2, 0.5])
    move_s = 0.30
    hold_s = 0.80
    speed = travel / move_s

    ctl.anchor_to(start)
    p0 = ctl.position.copy()
    goal_px = _screen(ctl, end)
    span = float(np.linalg.norm(goal_px - p0))

    t90 = t99 = float("nan")
    frames_move = int(move_s * fps)
    frames_hold = int(hold_s * fps)
    for i in range(frames_move + frames_hold):
        t = i * dt
        if i < frames_move:
            u = (i + 1) / frames_move
            point = start + (end - start) * u
            vel = (end - start) / dt
        else:
            point = end
            vel = np.zeros(2)
        _drive(ctl, point, vel, dt, t, field)
        if i >= frames_move:
            elapsed = (i - frames_move + 1) * dt * 1000.0
            frac = float(np.linalg.norm(ctl.position - p0)) / max(span, 1e-9)
            if math.isnan(t90) and frac >= 0.90:
                t90 = elapsed
            if math.isnan(t99) and frac >= 0.99:
                t99 = elapsed
    return t90, t99


def sine_lag(cfg: EngineConfig, fps: float, freq: float, field: IntentField, seconds: float = 4.0) -> tuple[float, float]:
    """(lag_ms, amplitude_ratio) at ``freq`` Hz.  Positive lag = cursor behind hand."""
    ctl = CursorController(cfg, screen=(1920, 1080))
    dt = 1.0 / fps
    amp = 0.18
    center = 0.5
    warm = int(1.0 * fps)
    n = int(seconds * fps)

    xs: list[float] = []
    ys: list[float] = []
    for i in range(warm + n):
        t = i * dt
        w = 2.0 * math.pi * freq
        point = np.array([center + amp * math.sin(w * t), center])
        vel = np.array([w * amp * math.cos(w * t), 0.0])
        _drive(ctl, point, vel, dt, t, field)
        if i >= warm:
            # Compare in the *same* units: the open-loop screen position the
            # controller is aiming at, not the normalised hand coordinate.
            xs.append(_screen(ctl, point)[0])
            ys.append(ctl.position[0])

    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    tt = np.arange(len(x)) * dt
    w = 2.0 * math.pi * freq
    basis = np.vstack([np.sin(w * tt), np.cos(w * tt)]).T
    cx = np.linalg.lstsq(basis, x, rcond=None)[0]
    cy = np.linalg.lstsq(basis, y, rcond=None)[0]
    ph_x = math.atan2(cx[1], cx[0])
    ph_y = math.atan2(cy[1], cy[0])
    dphi = (ph_y - ph_x + math.pi) % (2.0 * math.pi) - math.pi
    lag_ms = -dphi / w * 1000.0
    ratio = float(np.hypot(*cy) / max(np.hypot(*cx), 1e-9))
    return lag_ms, ratio


def tremor(cfg: EngineConfig, fps: float, field: IntentField, seconds: float = 4.0) -> tuple[float, float]:
    """Cursor RMS deviation (px) while the hand is *stationary* but shaky.

    Two components, because they stress different things:

    * **physiological tremor** — an 8-11 Hz oscillation (here 9.5 Hz at 0.006
      normalized units ~ 12 px).  This is what the filter's cutoff has to kill,
      and it is the component that a white-noise model completely fails to
      represent: white noise at 30 fps has no meaningful spectral structure.
    * **sensor noise** — small white jitter at 0.003.

    Returns ``(rms_px, peak_px)``.  This is the other half of the filter
    trade-off: any change that improves tracking lag must be paid for here or it
    is not a real improvement.
    """
    rng = np.random.default_rng(12345)
    ctl = CursorController(cfg, screen=(1920, 1080))
    dt = 1.0 / fps
    center = np.array([0.5, 0.5])
    ctl.anchor_to(center)
    samples: list[np.ndarray] = []
    for i in range(int(seconds * fps)):
        t = i * dt
        tr = 0.006 * math.sin(2.0 * math.pi * 9.5 * t)
        point = center + np.array([tr, tr * 0.6]) + rng.normal(0.0, 0.003, 2)
        _drive(ctl, point, np.zeros(2), dt, t, field)
        if i >= fps:
            samples.append(ctl.position.copy())
    arr = np.asarray(samples)
    dev = arr - arr.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum(dev**2, axis=1))))
    peak = float(np.max(np.linalg.norm(dev, axis=1)))
    return rms, peak


def flick(
    cfg: EngineConfig, fps: float, field: IntentField, travel: float = 0.60, move_s: float = 0.25
) -> tuple[float, float, float]:
    """Fast sweep then hold — the movement that actually stresses the cursor.

    Returns ``(arrival_pct, overshoot_px, settle_ms)``:

    * ``arrival_pct`` — how much of the mapped travel is covered on the frame
      the hand stops.  Informative, but *not* a quality score: a cursor that is
      legitimately still catching up scores low here and that is not a defect.
    * ``overshoot_px`` — how far past its own final resting position the cursor
      goes during the hold.  This is the number that matters: it is the
      difference between "landed on the button" and "flew past it and came
      back".
    * ``settle_ms`` — time from the hand stopping to the cursor being within 2 %
      of the mapped travel *and staying there*.
    """
    ctl = CursorController(cfg, screen=(1920, 1080))
    dt = 1.0 / fps
    start = np.array([0.5 - travel / 2, 0.5])
    end = np.array([0.5 + travel / 2, 0.5])

    ctl.anchor_to(start)
    p0 = ctl.position.copy()
    span = float(np.linalg.norm(_screen(ctl, end) - p0))

    frames_move = max(int(move_s * fps), 1)
    hold_frames = int(1.0 * fps)

    arrival = 0.0
    positions: list[float] = []
    for i in range(frames_move + hold_frames):
        t = i * dt
        if i < frames_move:
            u = (i + 1) / frames_move
            point = start + (end - start) * u
            vel = (end - start) / dt
        else:
            point = end
            vel = np.zeros(2)
        _drive(ctl, point, vel, dt, t, field)
        if i == frames_move - 1:
            arrival = float(np.linalg.norm(ctl.position - p0)) / max(span, 1e-9)
        if i >= frames_move:
            positions.append(float(ctl.position[0]))

    final = positions[-1]
    overshoot = max(0.0, max(positions) - final) if positions else 0.0

    settle = float("nan")
    tol = 0.02 * max(span, 1e-9)
    for k in range(len(positions)):
        if all(abs(v - final) <= tol for v in positions[k:]):
            settle = k * dt * 1000.0
            break
    return arrival, overshoot, settle


def frequency_response(cfg: EngineConfig, fps: float, field: IntentField, freqs: list[float]) -> list[tuple[float, float, float]]:
    out = []
    for f in freqs:
        lag, ratio = sine_lag(cfg, fps, f, field)
        out.append((f, lag, ratio))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--flat-field", action="store_true", help="use a uniform intent field (worst-case hesitation)")
    ap.add_argument("--freqs", default="0.5,1.0,1.5,2.0,3.0")
    ap.add_argument("--tremor", action="store_true", help="also measure cursor RMS while the hand is stationary but noisy")
    ap.add_argument("--sweep-euro", action="store_true", help="sweep One-Euro min_cutoff/beta instead of the controller list")
    args = ap.parse_args()

    base = load_config(args.config)
    field = _field(flat=args.flat_field)
    label = "flat field" if args.flat_field else "peaked field"
    freqs = [float(x) for x in args.freqs.split(",")]

    print(f"fps={args.fps:g}  controller={base.cursor.controller}  intent field={label}")
    print()

    variants: list[tuple[str, EngineConfig]] = []
    if args.sweep_euro:
        for cutoff in (1.1, 2.0, 3.0, 4.5):
            for beta in (0.05, 0.3, 0.6, 1.2):
                c = copy.deepcopy(base)
                c.cursor.euro_min_cutoff = cutoff
                c.cursor.euro_beta = beta
                variants.append((f"euro fc={cutoff:g} beta={beta:g}", c))
    else:
        variants.append(("as-configured (spring+euro)", copy.deepcopy(base)))

        c = copy.deepcopy(base)
        c.cursor.controller = "gain"
        variants.append(("gain (plain EMA a=0.35)", c))

        c = copy.deepcopy(base)
        c.cursor.controller = "velocity_curve"
        variants.append(("velocity_curve (V1 adaptive)", c))

        for omega in (45.0, 60.0, 90.0):
            c = copy.deepcopy(base)
            c.cursor.spring_omega_fast = omega
            variants.append((f"spring omega_fast={omega:g}", c))

        c = copy.deepcopy(base)
        c.cursor.feedforward = False
        c.cursor.prediction_horizon = 0.045
        variants.append(("positional lead (old behaviour)", c))

        c = copy.deepcopy(base)
        c.cursor.intent_adaptive = False
        variants.append(("spring, intent-gain off", c))

    # -------- flick / step summary -------- #
    print(f"{'variant':<34s} {'arriv':>7s} {'over':>8s} {'settle':>8s}")
    for name, cfg in variants:
        arrival, overshoot, settle = flick(cfg, args.fps, field)
        st = f"{settle:.0f}ms" if not math.isnan(settle) else "   >1s"
        print(f"{name:<34s} {arrival * 100:6.1f}% {overshoot:6.0f}px {st:>8s}")

    # -------- frequency response -------- #
    header = "".join(f"{f:>7.1f}Hz" for f in freqs)
    print()
    print(f"{'variant':<34s} {header}")
    print(f"{'':<34s} " + "".join(f"{'lag/amp':>9s}" for _ in freqs))
    for name, cfg in variants:
        cells = []
        for f, lag, ratio in frequency_response(cfg, args.fps, field, freqs):
            cells.append(f"{lag:>4.0f}/{ratio:.2f}")
        print(f"{name:<34s} " + "".join(f"{c:>9s}" for c in cells))

    if args.tremor:
        print()
        print("tremor at rest (hand = 9.5 Hz / 0.006 + white 0.003):")
        print(f"  {'variant':<32s} {'RMS':>8s} {'peak':>8s}")
        for name, cfg in variants:
            rms, peak = tremor(cfg, args.fps, field)
            print(f"  {name:<32s} {rms:6.2f}px {peak:6.2f}px")


if __name__ == "__main__":
    main()
