"""Benchmark metrics.

Every metric here is defined explicitly, because "accuracy" on its own is a
meaningless number for a real-time interaction system.  The metrics that actually
matter for this project are:

* **detection latency** — how long after the user starts a movement does the
  engine recognise it;
* **false activation rate** — how often the engine does something the user did
  not ask for (the metric that kills gesture UIs in practice);
* **interaction effort** — distance moved, corrective movements, time taken.  The
  goal is not to recognise the gesture, it is to make the interaction cheap;
* **cursor jitter and lag** — the two ends of the smoothing trade-off.

All of them are computed from a recorded replay, so two algorithm versions can be
compared on the *same* input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import FrameReport

#: Minimum normalised cross-correlation for a cursor-lag estimate to be reported.
MIN_LAG_QUALITY = 0.5


@dataclass
class SegmentTruth:
    t0: float
    t1: float
    label: str
    intent: str
    tags: dict = field(default_factory=dict)

    def contains(self, t: float) -> bool:
        return self.t0 <= t <= self.t1


@dataclass
class Evaluation:
    name: str = "run"
    frames: int = 0
    duration: float = 0.0
    fps: float = 0.0
    # gesture
    gesture_hits: int = 0
    gesture_misses: int = 0
    gesture_false: int = 0
    gesture_latency_ms: list[float] = field(default_factory=list)
    # intent
    intent_hits: int = 0
    intent_misses: int = 0
    # commits
    commits: int = 0
    commits_true: int = 0
    commits_false: int = 0
    # cursor
    jitter_px: float = 0.0
    lag_ms: float = 0.0
    lag_quality: float = 0.0
    # effort
    path_length: float = 0.0
    corrective_movements: int = 0
    path_efficiency: float = 0.0
    mean_speed: float = 0.0
    # latency
    pipeline_ms_p50: float = 0.0
    pipeline_ms_p95: float = 0.0
    stage_ms: dict[str, float] = field(default_factory=dict)
    # interaction
    actions: dict[str, int] = field(default_factory=dict)

    def as_dict(self, include_timing: bool = True) -> dict:
        n_gestures = self.gesture_hits + self.gesture_misses
        d = {
            "name": self.name,
            "frames": self.frames,
            "duration_s": round(self.duration, 3),
            "fps": round(self.fps, 2),
            "gesture_recall": round(self.gesture_hits / n_gestures, 4) if n_gestures else None,
            "gesture_hits": self.gesture_hits,
            "gesture_misses": self.gesture_misses,
            "gesture_false_per_s": round(self.gesture_false / self.duration, 3) if self.duration else 0.0,
            "gesture_latency_ms": round(float(np.median(self.gesture_latency_ms)), 1) if self.gesture_latency_ms else None,
            "gesture_latency_p95_ms": round(float(np.percentile(self.gesture_latency_ms, 95)), 1) if self.gesture_latency_ms else None,
            "intent_recall": round(self.intent_hits / max(self.intent_hits + self.intent_misses, 1), 4),
            "commits": self.commits,
            # None (not 0.0) when nothing committed: "no commits" and "every
            # commit was wrong" are different results and must not be conflated.
            "commit_precision": round(self.commits_true / self.commits, 4) if self.commits else None,
            "commit_false": self.commits_false,
            "cursor_jitter_px": round(self.jitter_px, 3),
            # None, not 0.0, when the estimate is not trustworthy: reporting a
            # number for a noise fit is worse than reporting nothing.
            "cursor_lag_ms": round(self.lag_ms, 1) if self.lag_quality >= MIN_LAG_QUALITY else None,
            "cursor_lag_quality": round(self.lag_quality, 3),
            "path_length": round(self.path_length, 3),
            "corrective_movements": self.corrective_movements,
            "path_efficiency": round(self.path_efficiency, 4),
            "mean_speed": round(self.mean_speed, 3),
            "actions": dict(self.actions),
        }
        if include_timing:
            d["pipeline_ms_p50"] = round(self.pipeline_ms_p50, 2)
            d["pipeline_ms_p95"] = round(self.pipeline_ms_p95, 2)
            d["stage_ms"] = {k: round(v, 2) for k, v in self.stage_ms.items()}
        return d


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _pipeline_ms(report: FrameReport) -> float:
    keys = ("tracking", "features", "motion", "gesture", "intent", "policy", "cursor", "dispatch")
    return float(sum(report.stage_ms.get(k, 0.0) for k in keys))


def _cursor_series(reports: list[FrameReport]) -> np.ndarray:
    pts = [r.cursor for r in reports if r.cursor is not None]
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2))


def _hand_series(reports: list[FrameReport]) -> np.ndarray:
    pts = [r.motion.position for r in reports if r.motion is not None]
    return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2))


def _tracking_mask(reports: list[FrameReport]) -> np.ndarray:
    """Frames inside a continuous tracking episode.

    Session boundaries are excluded.  The cursor deliberately teleports when it
    is first activated (there is nothing on screen to teleport *away* from), and
    that single step contributes more to a second-difference RMS than a whole
    second of tremor — it made every variant report ~143 px of "jitter".
    """
    modes = [str(r.notes.get("cursor_mode", "")) for r in reports]
    mask = np.ones(len(reports), dtype=bool)
    for i, mode in enumerate(modes):
        if mode in ("", "PAUSED"):
            mask[i] = False
        elif i > 0 and modes[i - 1] == "PAUSED":
            mask[i] = False  # first frame after (re)activation
    return mask


def cursor_jitter(
    reports: list[FrameReport],
    screen: tuple[int, int] = (1920, 1080),
    max_hand_speed: float = 0.12,
    settle_frames: int = 5,
) -> float:
    """RMS of the cursor's high-frequency residual while the hand is still, in px.

    The residual is ``|p - moving_average_3(p)|`` — the same jitter definition
    used by :func:`gesture_engine.motion.trajectory.analyse`, which keeps one
    notion of "jitter" in the codebase instead of two.

    Why not the second difference over all tracked frames (the previous
    version): a second difference only vanishes for *constant-velocity* motion,
    and a real reach is accelerate-cruise-decelerate, so the metric ended up
    dominated by the cursor's own acceleration.  It reported 60 px for the
    responsive controller while an isolated probe measured 0.41 px of actual
    tremor — i.e. it was scoring responsiveness as if it were shake.

    Three guards, each of which changes the answer:

    * only frames where the **hand** is still are counted (that is what "jitter"
      means for a pointing device),
    * the first ``settle_frames`` of each still period are dropped, because the
      cursor is still decelerating out of the previous movement,
    * the residual is computed **within** each contiguous still run, never
      across a gap.
    """
    mask = _tracking_mask(reports)
    speeds = np.asarray([r.motion.speed if r.motion is not None else 1.0 for r in reports], dtype=np.float64)
    mask = mask & (speeds <= max_hand_speed)

    residuals: list[np.ndarray] = []
    kernel = np.ones(3) / 3.0
    for start, stop in _runs(mask):
        if stop - start < settle_frames + 4:
            continue
        pts = [reports[i].cursor for i in range(start + settle_frames, stop) if reports[i].cursor is not None]
        if len(pts) < 4:
            continue
        px = np.asarray(pts, dtype=np.float64) * np.asarray(screen, dtype=np.float64)
        smooth = np.stack([np.convolve(px[:, 0], kernel, mode="valid"), np.convolve(px[:, 1], kernel, mode="valid")], axis=1)
        residuals.append(np.linalg.norm(px[1:-1] - smooth, axis=1))

    if not residuals:
        return 0.0
    allres = np.concatenate(residuals)
    return float(np.sqrt((allres**2).mean()))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """All contiguous ``True`` runs as ``(start, stop)`` (stop exclusive)."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask)))
    return out


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """``(start, stop)`` of the longest contiguous ``True`` run (stop exclusive)."""
    best = (0, 0)
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    if start is not None and len(mask) - start > best[1] - best[0]:
        best = (start, len(mask))
    return best


def cursor_lag(
    reports: list[FrameReport],
    screen: tuple[int, int] = (1920, 1080),
    max_lag_frames: int = 15,
    min_speed: float = 0.15,
    min_run_frames: int = 15,
    min_runs: int = 3,
    min_correlation: float = 0.3,
) -> tuple[float, float]:
    """Delay between the hand and the cursor.  Returns ``(lag_ms, quality)``.

    Positive lag = the cursor trails the hand.  The estimate is the
    length-weighted mean of the per-stroke velocity cross-correlation lag.

    **Read the quality before the number.**  It is
    ``min(mean correlation, inter-stroke agreement)`` and it declines whenever
    the strokes do not agree on a single lag.  Below ~0.5 there is no lag to
    report and the caller is expected to print "unknown".

    The domain of validity is narrow, and stating it *is* the function rather
    than a caveat on it.  The estimator is exact — it recovers twelve synthetic
    known delays to the frame — **when the cursor is approximately a delayed
    copy of the hand**.  On a real controller it is not, because the cursor is a
    *filter*, not a delay: the One-Euro pre-filter, the speed-scheduled spring
    and the feed-forward each contribute frequency-dependent phase, so "the lag"
    is not one number, it is a transfer function.  Two measured consequences:

    * **Strokes disagree.**  On the ``click`` scenario the per-stroke estimates
      scatter over ``[2, -8, 2, 3, 3, 8, 2, -8, 3, -7, 2, -7, 2]`` frames — the
      whole search range — and the weighted mean lands near zero only because
      positive and negative errors cancel.  That is not a measurement, it is
      noise averaging into a plausible-looking number.  The agreement term
      catches it and the caller prints "unknown".
    * **A periodic stimulus is systematically wrong.**  On a continuous 0.5 Hz
      sinusoid all eleven strokes return exactly 10 frames (333 ms, agreement
      1.0) while the phase lag measured against the same stimulus is 57 ms:
      perfectly repeatable, six times too large.  Agreement cannot detect this,
      and no cheap statistic separates it — moving fraction, stroke length, gap
      length and the hand's autocorrelation all overlap between a sinusoid and
      the stroke scenarios.  A continuous sweep is not a session and must not be
      passed to this function.

    **The instrument of record for cursor latency is the isolated frequency
    response** in ``tools/_lag_probe.py``: it drives the controller with a known
    stimulus and measures phase and amplitude directly, which is well posed
    whatever the controller happens to be.  This metric exists to catch a
    session in which the cursor has become a delayed copy — a broken filter, a
    bypassed controller — and to give the benchmark a way to say "no".

    Three measurement problems, each of which silently produced a confident
    wrong answer in an earlier implementation:

    * **The cursor is not an affine image of the hand.**  ``active_region``
      remaps the hand rectangle onto the screen, the controller adds a dead zone
      and the spring adds dynamics.  Comparing raw positions with a squared
      error therefore measures the coordinate mismatch, not the lag — it pinned
      at the search boundary (+400 ms) for every variant.  The comparison
      therefore runs on *velocity*, which is unaffected by the offset part of
      the mapping.
    * **One stroke is not a sample.**  Correlating only the *longest* moving run
      discarded most of the session and made the answer depend on which single
      stroke happened to be longest: it reported 133 ms for a controller whose
      isolated phase lag is 60-76 ms, and it could not tell the variants apart.
      Every stroke of at least ``min_run_frames`` now contributes, weighted by
      its length, which also gives the metric something to compute a dispersion
      from.
    * **Peak alignment does not work on real movements.**  "Delay between the
      hand's peak speed and the cursor's peak speed" sounds more physical, but a
      hand movement is accelerate-cruise-decelerate and the peak sits on a
      plateau, so ``argmax`` lands on an arbitrary frame of the plateau — it
      returned 0 ms for every stroke and every variant.  Cross-correlation uses
      the whole stroke instead of one sample of it.

    * **The window must not move with the lag.**  A stroke is a run of frames in
      which the *hand* is moving; the cursor is sampled at ``frame + lag`` from
      the **full series** and is allowed to fall outside the window, because
      that is precisely what a delay means.  Shifting the window itself (the
      obvious implementation) mixes in the cursor's trailing tail — the frames
      where the hand has already stopped but the cursor is still arriving — and
      biases the answer by up to 1.6 frames: against synthetic sessions with an
      exactly known delay it returned 153 ms for 100 ms and 187 ms for 133 ms.
      Sampling by index recovers all twelve of the known delays exactly.
    * **Peak alignment does not work on real movements.**  "Delay between the
      hand's peak speed and the cursor's peak speed" sounds more physical, but a
      hand movement is accelerate-cruise-decelerate and the peak sits on a
      plateau, so ``argmax`` lands on an arbitrary frame of the plateau — it
      returned 0 ms for every stroke and every variant.  Cross-correlation uses
      the whole stroke instead of one sample of it.

    A stroke is a contiguous run of frames whose hand speed exceeds
    ``min_speed``.  Runs whose best correlation is below ``min_correlation`` are
    dropped as noise, and fewer than ``min_runs`` survivors means "unknown".
    """
    hand = _hand_series(reports)
    cur = _cursor_series(reports)
    speeds = np.asarray([r.motion.speed if r.motion is not None else 0.0 for r in reports], dtype=np.float64)
    n = min(hand.shape[0], cur.shape[0], speeds.shape[0])
    if n < 20:
        return 0.0, 0.0
    hand, cur, speeds = hand[:n], cur[:n], speeds[:n]

    ts = np.asarray([r.timestamp for r in reports][:n], dtype=np.float64)
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 1 / 30.0
    if dt <= 0:
        return 0.0, 0.0

    v_hand_full = np.diff(hand, axis=0)
    v_cur_full = np.diff(cur, axis=0)
    n_v = len(v_cur_full)

    lags: list[float] = []
    weights: list[float] = []
    scores: list[float] = []
    for start, stop in _runs(speeds > min_speed):
        if stop - start < min_run_frames:
            continue
        index = np.arange(start, stop - 1)
        a = v_hand_full[index]
        a = a - a.mean(axis=0)

        best_lag, best_score = 0, -2.0
        for lag in range(-max_lag_frames, max_lag_frames + 1):
            j = index + lag
            inside = (j >= 0) & (j < n_v)
            if int(inside.sum()) < 8:
                continue
            x = a[inside]
            y = v_cur_full[j[inside]]
            y = y - y.mean(axis=0)
            den = float(np.linalg.norm(x) * np.linalg.norm(y))
            if den < 1e-12:
                continue
            score = float((x * y).sum()) / den
            if score > best_score:
                best_score, best_lag = score, lag
        if best_score < min_correlation:
            continue
        lags.append(float(best_lag))
        weights.append(float(stop - start))
        scores.append(best_score)

    if len(lags) < min_runs:
        return 0.0, 0.0

    arr = np.asarray(lags, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    lag_frames = float(np.average(arr, weights=w))
    mean_correlation = float(np.average(np.asarray(scores, dtype=np.float64), weights=w))

    # Agreement: how tightly the strokes cluster.  A genuine delay produces a
    # tight cluster; a filtering controller produces a scatter, and a scatter
    # has no mean.
    spread = float(np.percentile(arr, 75) - np.percentile(arr, 25))
    agreement = max(0.0, 1.0 - spread / max(float(max_lag_frames), 1.0))
    quality = min(mean_correlation, agreement)
    return lag_frames * dt * 1000.0, quality


def gesture_false_runs(reports: list[FrameReport], truth: list[SegmentTruth]) -> int:
    """Number of *contiguous runs* of a confirmed gesture that contradict truth.

    Counting per-frame and dividing by an assumed run length (the previous
    approach) is not a measurement: it makes the false-activation rate depend on
    the vote window rather than on how often the engine actually did the wrong
    thing.  A run is what a user perceives as one wrong action.
    """
    runs = 0
    previous: str | None = None
    for r in reports:
        name = r.gesture.gesture
        active = next((s for s in truth if s.contains(r.timestamp)), None)
        contradicts = (
            name not in ("NONE", "")
            and (active is None or (active.label not in ("NONE", "") and active.label != name))
        )
        if contradicts and name != previous:
            runs += 1
        previous = name if name not in ("NONE", "") else None
    return runs


def corrective_movements(reports: list[FrameReport], speed_threshold: float = 0.35) -> int:
    """Count of direction reversals during movement — a proxy for 'aiming again'."""
    count = 0
    prev_dir = None
    for r in reports:
        if r.motion is None or r.motion.speed < speed_threshold:
            continue
        d = r.motion.direction
        if prev_dir is not None and abs((d - prev_dir + np.pi) % (2 * np.pi) - np.pi) > np.radians(120):
            count += 1
        prev_dir = d
    return count


# --------------------------------------------------------------------------- #
# Main evaluation
# --------------------------------------------------------------------------- #


def evaluate(
    reports: list[FrameReport],
    truth: list[SegmentTruth] | None = None,
    name: str = "run",
    screen: tuple[int, int] = (1920, 1080),
    latency_tolerance: float = 0.60,
) -> Evaluation:
    ev = Evaluation(name=name)
    if not reports:
        return ev

    ev.frames = len(reports)
    ev.duration = float(reports[-1].timestamp - reports[0].timestamp)
    ev.fps = float(ev.frames / ev.duration) if ev.duration > 0 else 0.0

    # ---- gesture detection per ground-truth segment -------------------- #
    if truth:
        for seg in truth:
            if seg.label in ("NONE", ""):
                continue
            found_at = None
            for r in reports:
                if r.timestamp < seg.t0:
                    continue
                if r.timestamp > seg.t1 + latency_tolerance:
                    break
                if r.gesture.gesture == seg.label:
                    found_at = r.timestamp
                    break
            if found_at is None:
                ev.gesture_misses += 1
            else:
                ev.gesture_hits += 1
                ev.gesture_latency_ms.append(max(found_at - seg.t0, 0.0) * 1000.0)

        # false gestures: a confirmed gesture whose label is not the active truth
        ev.gesture_false = gesture_false_runs(reports, truth)

        # ---- intent ---------------------------------------------------- #
        for seg in truth:
            if seg.intent in ("UNKNOWN", ""):
                continue
            ok = any(
                r.intents.committed is not None
                and r.intents.committed.value == seg.intent
                and seg.t0 - 0.1 <= r.timestamp <= seg.t1 + latency_tolerance
                for r in reports
            )
            if ok:
                ev.intent_hits += 1
            else:
                ev.intent_misses += 1

        # ---- commits ---------------------------------------------------- #
        commit_tags = [s for s in truth if s.tags.get("commit")]
        for r in reports:
            if r.commit is not None and r.commit.committed:
                ev.commits += 1
                if any(s.t0 - 0.4 <= r.timestamp <= s.t1 + 0.5 for s in commit_tags):
                    ev.commits_true += 1
                else:
                    ev.commits_false += 1

    # ---- cursor quality ------------------------------------------------- #
    ev.jitter_px = cursor_jitter(reports, screen)
    ev.lag_ms, ev.lag_quality = cursor_lag(reports, screen)

    # ---- effort ---------------------------------------------------------- #
    hand = _hand_series(reports)
    if hand.shape[0] > 1:
        steps = np.linalg.norm(np.diff(hand, axis=0), axis=1)
        ev.path_length = float(steps.sum())
        ev.path_efficiency = float(np.linalg.norm(hand[-1] - hand[0]) / max(ev.path_length, 1e-6))
    speeds = [r.motion.speed for r in reports if r.motion is not None]
    ev.mean_speed = float(np.mean(speeds)) if speeds else 0.0
    ev.corrective_movements = corrective_movements(reports)

    # ---- latency --------------------------------------------------------- #
    pipeline = np.asarray([_pipeline_ms(r) for r in reports], dtype=np.float64)
    ev.pipeline_ms_p50 = float(np.percentile(pipeline, 50))
    ev.pipeline_ms_p95 = float(np.percentile(pipeline, 95))
    stages: dict[str, list[float]] = {}
    for r in reports:
        for k, v in r.stage_ms.items():
            stages.setdefault(k, []).append(v)
    ev.stage_ms = {k: float(np.mean(v)) for k, v in stages.items()}

    # ---- actions --------------------------------------------------------- #
    for r in reports:
        for a in r.actions:
            ev.actions[a.kind] = ev.actions.get(a.kind, 0) + 1

    return ev


def compare(results: list[Evaluation], include_timing: bool = False) -> dict:
    """Tabulate several runs (e.g. V1 vs V2, fixed vs adaptive smoothing).

    ``include_timing=False`` by default on purpose: every metric in this table is
    a pure function of the input stream and the algorithm version, so two runs of
    the same variant must produce *bit-identical* numbers.  Wall-clock latency is
    not such a function, so it is reported separately (:meth:`timing_table`) —
    mixing the two made the benchmark table non-reproducible.
    """
    keys = [
        "gesture_recall",
        "gesture_latency_ms",
        "gesture_false_per_s",
        "intent_recall",
        "commit_precision",
        "cursor_jitter_px",
        "cursor_lag_ms",
        "cursor_lag_quality",
        "path_length",
        "corrective_movements",
    ]
    if include_timing:
        keys += ["pipeline_ms_p50", "pipeline_ms_p95"]
    table: dict[str, dict] = {}
    for ev in results:
        d = ev.as_dict(include_timing=include_timing)
        table[ev.name] = {k: d.get(k) for k in keys}
    return table


def timing_table(results: list[Evaluation]) -> dict[str, dict]:
    """Per-stage wall-clock costs.  Explicitly *not* part of :func:`compare`."""
    return {
        ev.name: {
            "pipeline_ms_p50": round(ev.pipeline_ms_p50, 2),
            "pipeline_ms_p95": round(ev.pipeline_ms_p95, 2),
            "stage_ms": {k: round(v, 2) for k, v in ev.stage_ms.items()},
        }
        for ev in results
    }


def format_table(table: dict[str, dict]) -> str:
    """Markdown table, one row per metric, one column per run."""
    if not table:
        return "(no results)"
    names = list(table)
    keys = list(next(iter(table.values())).keys())
    header = "| metric | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 1)
    rows = [header, sep]
    for k in keys:
        cells = []
        for n in names:
            v = table[n].get(k)
            cells.append("-" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
        rows.append(f"| {k} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
