"""Scripted synthetic hand source.

Plays a timeline of segments (pose + trajectory + ground-truth label) and emits
``Observation`` objects through the same interface as the MediaPipe tracker.  The
engine cannot tell the difference — that is the point.

Ground-truth labels travel in ``Observation.camera['label']`` so the recorder
stores them and the benchmark can compute accuracy, latency and false-activation
rates without a human in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import EngineConfig
from ..types import Landmarks, Observation
from .landmark_model import HandPose, build_hand, make_pose, wrist_for_tip


@dataclass(slots=True)
class Segment:
    """One scripted interval."""

    duration: float
    pose: str
    start: tuple[float, float] = (0.5, 0.65)
    end: tuple[float, float] | None = None
    easing: str = "linear"  # linear | smooth | decel | accel
    rotation_from: float = 0.0
    rotation_to: float = 0.0
    scale: float = 0.22
    scale_to: float | None = None
    label: str = "NONE"
    intent: str = "UNKNOWN"
    noise: float = 0.0
    #: Seconds over which the finger configuration morphs from the previous
    #: segment's pose into this one.  Without this the synthetic hand would
    #: teleport between poses, which injects a huge fake velocity spike into the
    #: control point and makes every motion-based decision meaningless.
    blend: float = 0.09
    tags: dict = field(default_factory=dict)


def _ease(kind: str, u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    if kind == "smooth":
        return float(u * u * (3.0 - 2.0 * u))
    if kind == "decel":
        return float(1.0 - (1.0 - u) ** 3)
    if kind == "accel":
        return float(u**3)
    return u


def tip(
    duration: float,
    pose: str,
    start: tuple[float, float],
    end: tuple[float, float] | None = None,
    *,
    rotation_deg: float = 0.0,
    scale: float = 0.22,
    **kwargs: object,
) -> Segment:
    """Build a :class:`Segment` from **control-point** waypoints.

    ``Segment.start`` / ``Segment.end`` are wrist coordinates, but a scenario is
    written in terms of where the cursor should be.  This converts one to the
    other so the ground-truth label ("the hand is on T0") actually describes the
    point the engine steers with.
    """
    s = wrist_for_tip(pose, start, rotation_deg=rotation_deg, scale=scale)
    e = wrist_for_tip(pose, end, rotation_deg=rotation_deg, scale=scale) if end is not None else None
    return Segment(duration, pose, start=s, end=e, rotation_from=rotation_deg, scale=scale, **kwargs)  # type: ignore[arg-type]


def _lerp_pose(a: HandPose, b: HandPose, u: float) -> HandPose:
    """Blend two poses (curl, spread and rotation only)."""
    u = float(np.clip(u, 0.0, 1.0))
    return HandPose(
        curl=tuple(float(a.curl[i] + (b.curl[i] - a.curl[i]) * u) for i in range(5)),  # type: ignore[arg-type]
        rotation_deg=float(a.rotation_deg + (b.rotation_deg - a.rotation_deg) * u),
        position=a.position,
        scale=float(a.scale + (b.scale - a.scale) * u),
        force_pinch=b.force_pinch if u > 0.5 else a.force_pinch,
        spread_deg=tuple(float(a.spread_deg[i] + (b.spread_deg[i] - a.spread_deg[i]) * u) for i in range(5)),  # type: ignore[arg-type]
    )


class SyntheticTracker:
    """Timeline-driven replacement for the camera tracker."""

    def __init__(self, cfg: EngineConfig, segments: list[Segment] | None = None, loop: bool = True) -> None:
        self.cfg = cfg
        self.segments = segments if segments is not None else default_scenario()
        self.loop = loop
        self.duration = float(sum(s.duration for s in self.segments))
        self._rng = np.random.default_rng(12345)
        self._t0: float | None = None
        self._frame = 0

    # ------------------------------------------------------------------ #
    def reset(self, t0: float | None = None) -> None:
        self._t0 = t0
        self._frame = 0

    def _locate(self, t: float) -> tuple[int, Segment, float, float]:
        """Return ``(index, segment, u, segment_start_time)`` for local time ``t``."""
        total = self.duration if self.duration > 0 else 1.0
        tt = t % total if self.loop else min(t, total - 1e-6)
        acc = 0.0
        for i, seg in enumerate(self.segments):
            if tt < acc + seg.duration:
                return i, seg, (tt - acc) / max(seg.duration, 1e-6), acc
            acc += seg.duration
        return len(self.segments) - 1, self.segments[-1], 1.0, max(total - self.segments[-1].duration, 0.0)

    def pose_at(self, t: float) -> tuple[np.ndarray, Segment]:
        idx, seg, u, seg_start = self._locate(t)
        e = _ease(seg.easing, u)
        start = np.asarray(seg.start, dtype=np.float64)
        end = np.asarray(seg.end if seg.end is not None else seg.start, dtype=np.float64)
        xy = start + (end - start) * e
        rot = seg.rotation_from + (seg.rotation_to - seg.rotation_from) * e
        scale = seg.scale if seg.scale_to is None else seg.scale + (seg.scale_to - seg.scale) * e

        base: HandPose = make_pose(seg.pose, rotation_deg=rot, position=(float(xy[0]), float(xy[1])), scale=scale)

        # Morph the finger configuration in from the previous segment.
        if seg.blend > 0.0 and idx > 0:
            in_seg = t - seg_start
            if 0.0 <= in_seg < seg.blend:
                prev = self.segments[idx - 1]
                prev_pose = make_pose(prev.pose, rotation_deg=rot, position=(float(xy[0]), float(xy[1])), scale=scale)
                base = _lerp_pose(prev_pose, base, in_seg / seg.blend)

        pts = build_hand(base, noise=seg.noise, rng=self._rng)
        return pts, seg

    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray | None, timestamp: float, frame_id: int = 0, camera: dict | None = None) -> Observation:
        if self._t0 is None:
            self._t0 = timestamp
        local_t = timestamp - self._t0
        pts, seg = self.pose_at(local_t)
        meta = dict(camera or {})
        meta.update({"synthetic": True, "label": seg.label, "intent_label": seg.intent, "tags": dict(seg.tags)})
        obs = Observation(timestamp=timestamp, frame_id=frame_id, camera=meta)
        obs.hand = Landmarks(points=pts, name="hand")
        obs.hand_confidence = 0.98
        return obs

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #


def default_scenario() -> list[Segment]:
    """A full interaction loop: activate -> travel -> click -> drag -> pause -> cancel.

    Written in control-point coordinates (see :func:`tip`), and every segment
    chains into the next: a jump between segments is a fake velocity spike that
    every motion-based decision in the pipeline will happily believe.
    """
    return [
        tip(0.60, "OPEN_PALM", (0.50, 0.72), label="OPEN_PALM", intent="PAUSE"),
        tip(0.50, "INDEX_UP", (0.50, 0.70), label="INDEX_UP", intent="MOVE_CURSOR", noise=0.0008),
        tip(0.90, "INDEX_UP", (0.50, 0.70), (0.30, 0.45), easing="smooth", label="INDEX_UP", intent="MOVE_CURSOR", noise=0.0008),
        tip(0.35, "INDEX_UP", (0.30, 0.45), (0.28, 0.42), easing="decel", label="INDEX_UP", intent="MOVE_CURSOR", noise=0.0008),
        tip(0.30, "INDEX_UP", (0.28, 0.42), label="INDEX_UP", intent="SELECT", noise=0.0008, tags={"commit": True}),
        tip(0.25, "PINCH", (0.28, 0.42), label="PINCH", intent="SELECT", noise=0.0008),
        tip(0.60, "PINCH", (0.28, 0.42), (0.62, 0.55), easing="smooth", label="PINCH", intent="DRAG", noise=0.0008),
        tip(0.30, "INDEX_UP", (0.62, 0.55), label="INDEX_UP", intent="MOVE_CURSOR", noise=0.0008),
        tip(0.60, "OPEN_PALM", (0.62, 0.55), (0.45, 0.55), easing="smooth", label="OPEN_PALM", intent="WINDOW_SWITCH"),
        tip(0.45, "FIST", (0.45, 0.55), label="FIST", intent="CANCEL"),
    ]


def click_study_scenario(repeats: int = 8) -> list[Segment]:
    """Travel to a target, decelerate, micro-pause, pinch: a click trial.

    Used by the commit-point experiment: the target entry is the commit signal
    under test, the pinch is the reference mechanism.

    Two properties this scenario must have to be worth anything:

    * the **control point** lands on the target (see :func:`tip`) — otherwise
      ``TARGET_ENTRY``, approach detection and the gravity field never fire;
    * the timeline is *continuous*.  Each trial starts where the previous one
      ended and travels back to the home position as a real segment; the earlier
      version restarted every trial at home, teleporting the hand once per cycle,
      which the commit detector read as a thrust.
    """
    home = (0.50, 0.70)
    targets = [(0.25, 0.35), (0.72, 0.30), (0.30, 0.68), (0.68, 0.62), (0.50, 0.28), (0.20, 0.55)]
    segs: list[Segment] = [tip(0.60, "INDEX_UP", home, label="INDEX_UP", intent="MOVE_CURSOR")]
    cursor = home
    for i in range(repeats):
        tx, ty = targets[i % len(targets)]
        segs.append(tip(0.75, "INDEX_UP", cursor, (tx, ty), easing="smooth", label="INDEX_UP", intent="MOVE_CURSOR", noise=0.001))
        # The commit tag covers the *arrival*: the deceleration into the target
        # and the settle are one event, and the kinematic commit fires at the
        # start of it (when the control point enters the target), not 0.2 s later
        # when the dwell "officially" begins.  Tagging only the dwell marked a
        # correct TARGET_ENTRY commit as a false positive.
        segs.append(
            tip(0.22, "INDEX_UP", (tx, ty), (tx, ty + 0.005), easing="decel", label="INDEX_UP", intent="MOVE_CURSOR", noise=0.001, tags={"commit": True})
        )
        segs.append(tip(0.28, "INDEX_UP", (tx, ty + 0.005), label="INDEX_UP", intent="SELECT", noise=0.001, tags={"commit": True, "target": f"T{i}"}))
        segs.append(tip(0.22, "PINCH", (tx, ty + 0.005), label="PINCH", intent="SELECT", noise=0.001))
        segs.append(tip(0.50, "INDEX_UP", (tx, ty + 0.005), home, easing="smooth", label="INDEX_UP", intent="MOVE_CURSOR", noise=0.001))
        cursor = home
    return segs


def swipe_scenario() -> list[Segment]:
    """Activate with the index, then swipe horizontally.

    Two things matter here and both were wrong at first:

    * the original version swiped with an open palm.  That is unplayable: the
      activation pose is INDEX_UP, so the FSM never leaves IDLE and the swipe can
      never become an action — the benchmark was measuring a scenario in which
      nothing could happen;
    * the segments must *chain* in position.  A settle segment anchored at
      ``(0.50, 0.50)`` between two swipes that end at 0.24 teleports the hand
      0.26 units in one frame — a fake 7.8 u/s spike that the recogniser reads as
      a swipe in the direction of the teleport.
    """
    return [
        tip(0.60, "INDEX_UP", (0.70, 0.50), label="INDEX_UP", intent="MOVE_CURSOR"),
        tip(0.40, "INDEX_UP", (0.70, 0.50), label="INDEX_UP", intent="MOVE_CURSOR"),
        tip(0.40, "INDEX_UP", (0.70, 0.50), (0.24, 0.50), easing="smooth", label="SWIPE_LEFT", intent="WINDOW_SWITCH"),
        tip(0.50, "INDEX_UP", (0.24, 0.50), label="INDEX_UP", intent="MOVE_CURSOR"),
        tip(0.40, "INDEX_UP", (0.24, 0.50), (0.74, 0.50), easing="smooth", label="SWIPE_RIGHT", intent="WINDOW_SWITCH"),
        tip(0.40, "INDEX_UP", (0.74, 0.50), label="INDEX_UP", intent="MOVE_CURSOR"),
    ]


def scroll_scenario() -> list[Segment]:
    """Slow, deliberate vertical strokes while pointing.

    Strokes are 0.9 s for 0.30 units (~0.33 u/s): a scroll is a slow gesture, and
    the scroll-versus-travel discrimination is a *speed* test, so the scenario has
    to be physically honest about that.

    The rest position is the start of the first stroke, so the timeline chains:
    the previous version rested at y=0.45 and then started the first stroke at
    y=0.62, teleporting the hand 0.17 units in one frame.  That spike sat inside
    the speed window for a full second and suppressed every scroll score for the
    whole first stroke.
    """
    return [
        tip(0.60, "INDEX_UP", (0.50, 0.62), label="INDEX_UP", intent="MOVE_CURSOR"),
        tip(0.40, "INDEX_UP", (0.50, 0.62), label="INDEX_UP", intent="MOVE_CURSOR"),
        tip(0.90, "INDEX_UP", (0.50, 0.62), (0.50, 0.32), easing="smooth", label="SCROLL_UP", intent="SCROLL"),
        tip(0.30, "INDEX_UP", (0.50, 0.32), label="INDEX_UP", intent="MOVE_CURSOR"),
        tip(0.90, "INDEX_UP", (0.50, 0.32), (0.50, 0.62), easing="smooth", label="SCROLL_DOWN", intent="SCROLL"),
        tip(0.30, "INDEX_UP", (0.50, 0.62), label="INDEX_UP", intent="MOVE_CURSOR"),
    ]


def tracking_loss_scenario() -> list[Segment]:
    """Arm the cursor, then lose the hand mid-movement.

    The first segment must be still and long enough for the activation dwell,
    otherwise the FSM never leaves IDLE and the loss is never exercised.
    """
    return [
        tip(0.60, "INDEX_UP", (0.50, 0.60), label="INDEX_UP", intent="MOVE_CURSOR"),
        tip(0.40, "INDEX_UP", (0.50, 0.60), label="INDEX_UP", intent="MOVE_CURSOR"),
        tip(0.60, "INDEX_UP", (0.50, 0.60), (0.62, 0.45), easing="smooth", label="INDEX_UP", intent="MOVE_CURSOR"),
        tip(0.35, "FIST", (0.62, 0.45), (0.30, 0.35), easing="smooth", label="FIST", intent="CANCEL"),
    ]


SCENARIOS: dict[str, object] = {
    "demo": default_scenario,
    "click": click_study_scenario,
    "swipe": swipe_scenario,
    "scroll": scroll_scenario,
    "loss": tracking_loss_scenario,
}


def get_scenario(name: str) -> list[Segment]:
    factory = SCENARIOS.get(name)
    if factory is None:
        raise KeyError(f"unknown scenario {name!r}; known: {sorted(SCENARIOS)}")
    return factory()  # type: ignore[operator]
