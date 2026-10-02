"""Recording and deterministic replay.

This is the single most valuable piece of infrastructure in the project.  Without
it every algorithm change is evaluated by squinting at the screen; with it, a
recorded session can be replayed through version A and version B and the two
event streams diffed.

File format: JSON Lines.

* line 1 — ``header``: full config snapshot, camera profile, code versions,
  feature-vector version.  A recording without its config is worthless.
* lines 2..N — ``frame``: timestamp, landmarks, derived features, motion state,
  gesture, intent field, cursor position, events, commit signal.
* last line — ``footer``: frame count and duration.

The header is what makes a recording *self-describing*: :func:`load_recording`
rebuilds an :class:`EngineConfig` from it, so an old recording replays with the
thresholds it was captured under, not the current ones.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..config import EngineConfig, to_dict
from ..types import Event, EventType, Landmarks, Observation

log = logging.getLogger(__name__)

RECORDING_FORMAT_VERSION = 2


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays into JSON-native values.

    The in-memory :class:`~gesture_engine.types.FrameReport` deliberately keeps
    arrays (the overlay wants them), so the serialisation boundary — not the
    producer — is the right place to flatten them.  Without this the recorder
    dies on the first frame that carries landmarks.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #


@dataclass
class Recorder:
    cfg: EngineConfig
    name: str = "session"
    directory: str | Path | None = None
    enabled: bool = True
    label: str = ""
    _fh: Any = None
    _path: Path | None = None
    _frames: int = 0
    _started: float = 0.0
    _last_flush: float = 0.0

    def open(self) -> Path | None:
        if not self.enabled:
            return None
        directory = Path(self.directory or self.cfg.record.directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self._path = directory / f"{self.name}-{stamp}.jsonl"
        self._fh = self._path.open("w", encoding="utf-8")
        self._started = time.time()
        self._write(
            {
                "type": "header",
                "format": RECORDING_FORMAT_VERSION,
                "created": self._started,
                "label": self.label or self.name,
                "camera_profile": self.cfg.capture.profile,
                "camera": {
                    "device": self.cfg.capture.device,
                    "width": self.cfg.capture.width,
                    "height": self.cfg.capture.height,
                    "fps": self.cfg.capture.fps,
                    "undistorted": self.cfg.calibration.enabled,
                    "lens": self.cfg.active_profile().lens,
                },
                "tracking_backend": self.cfg.tracking.backend,
                "config": to_dict(self.cfg),
            }
        )
        return self._path

    # ------------------------------------------------------------------ #
    def _write(self, obj: dict) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(_jsonable(obj), ensure_ascii=False, separators=(",", ":")) + "\n")

    def record(self, report: Any) -> None:
        """Append one frame from a :class:`~gesture_engine.types.FrameReport`."""
        if self._fh is None:
            return
        self._frames += 1
        rec: dict[str, Any] = {
            "type": "frame",
            "i": report.frame_id,
            "t": round(report.timestamp, 6),
            "state": report.state,
            "gesture": {"name": report.gesture.gesture, "conf": report.gesture.confidence, "stab": report.gesture.stability},
            "intent": report.intents.as_dict(),
            "intent_committed": report.intents.committed.value if report.intents.committed else None,
            "intent_entropy": round(report.intents.entropy, 4),
        }
        if report.motion is not None:
            rec["motion"] = {
                "p": [round(float(v), 6) for v in report.motion.position],
                "v": [round(float(v), 6) for v in report.motion.velocity],
                "a": [round(float(v), 6) for v in report.motion.acceleration],
                "speed": round(report.motion.speed, 6),
                "peak": round(report.motion.speed_peak, 6),
                "pause": round(report.motion.pause_duration, 6),
                "trend": round(report.motion.speed_trend, 6),
                "conf": round(report.motion.confidence, 4),
            }
        if report.features is not None and self.cfg.record.include_frames:
            rec["features"] = report.features.as_dict()
        if report.cursor is not None:
            rec["cursor"] = [round(float(v), 3) for v in report.cursor]
        if report.events:
            rec["events"] = [{"type": e.type.value, "t": round(e.timestamp, 6), "conf": round(e.confidence, 3)} for e in report.events]
        if report.commit is not None:
            rec["commit"] = {"committed": report.commit.committed, "score": round(report.commit.score, 4), "kind": report.commit.kind}
        if report.target_beliefs:
            rec["targets"] = [
                {"id": b.target.id, "conf": round(b.confidence, 4), "dist": round(b.distance, 4), "state": b.target.state.value}
                for b in report.target_beliefs
            ]
        if report.actions:
            rec["actions"] = [{"kind": a.kind, "intent": a.intent.value, **{k: v for k, v in a.payload.items() if k != "target"}} for a in report.actions]

        # The landmark stream is written as its own record type (format v2) so it
        # can be replayed into a *different* algorithm version.  When it is
        # enabled the points are removed from `notes` to avoid storing 21x3 floats
        # twice per frame.
        points = report.notes.get("hand_points") if report.notes else None
        hands = report.notes.get("hands") if report.notes else None
        separate_landmarks = bool(self.cfg.record.include_landmarks) and (points is not None or hands is not None)
        if report.notes:
            rec["notes"] = (
                {k: v for k, v in report.notes.items() if k != "hand_points"} if separate_landmarks else report.notes
            )
        self._write(rec)
        if separate_landmarks:
            hand_map = None
            if hands:
                hand_map = {k: np.asarray(v, dtype=np.float64) for k, v in hands.items()}
            self.record_landmarks(
                report.timestamp,
                report.frame_id,
                None if hand_map else (None if points is None else np.asarray(points, dtype=np.float64)),
                float((report.notes or {}).get("hand_confidence", 1.0)),
                meta={
                    "label": (report.notes or {}).get("label", ""),
                    "hands": hand_map,
                    "hand_confidences": (report.notes or {}).get("hand_confidences", {}),
                    "primary_hand": (report.notes or {}).get("primary_hand"),
                },
            )

        now = time.time()
        if now - self._last_flush > 2.0:
            self._fh.flush()
            self._last_flush = now

    def record_landmarks(self, timestamp: float, frame_id: int, hand: np.ndarray | None, confidence: float, meta: dict | None = None) -> None:
        if self._fh is None:
            return
        self._write(
            {
                "type": "landmarks",
                "i": frame_id,
                "t": round(timestamp, 6),
                "conf": round(confidence, 4),
                "points": None if hand is None else [[round(float(v), 6) for v in row] for row in hand],
                "hands": (
                    {k: [[round(float(v), 6) for v in row] for row in np.asarray(arr, dtype=np.float64)] for k, arr in ((meta or {}).get("hands") or {}).items()}
                    if meta and isinstance(meta.get("hands"), dict) else None
                ),
                "meta": meta or {},
            }
        )

    def close(self) -> None:
        if self._fh is None:
            return
        self._write({"type": "footer", "frames": self._frames, "ended": time.time(), "duration": time.time() - self._started})
        self._fh.close()
        self._fh = None
        log.info("recording saved: %s (%d frames)", self._path, self._frames)

    def __enter__(self) -> "Recorder":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def path(self) -> Path | None:
        return self._path


# --------------------------------------------------------------------------- #
# Reader / replay
# --------------------------------------------------------------------------- #


@dataclass
class Recording:
    path: Path
    header: dict = field(default_factory=dict)
    frames: list[dict] = field(default_factory=list)
    landmarks: dict[int, dict] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if not self.frames:
            return 0.0
        return float(self.frames[-1]["t"] - self.frames[0]["t"])

    @property
    def fps(self) -> float:
        d = self.duration
        return float(len(self.frames) / d) if d > 0 else 0.0

    def config(self) -> EngineConfig | None:
        data = self.header.get("config")
        if not data:
            return None
        from ..config import _build  # local import: config internals are private

        return _build(EngineConfig, data)

    def labels(self) -> list[str]:
        return [f.get("notes", {}).get("label", "") for f in self.frames]

    def ground_truth(self) -> list[str]:
        return [f.get("notes", {}).get("intent_label", "") for f in self.frames]


def load_recording(path: str | Path) -> Recording:
    p = Path(path)
    header: dict = {}
    frames: list[dict] = []
    landmarks: dict[int, dict] = {}
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            kind = obj.get("type")
            if kind == "header":
                header = obj
            elif kind == "frame":
                frames.append(obj)
            elif kind == "landmarks":
                landmarks[int(obj.get("i", 0))] = obj
    return Recording(path=p, header=header, frames=frames, landmarks=landmarks)


class ReplayTracker:
    """Drop-in replacement for the MediaPipe tracker that reads a recording.

    Implements the same ``process(frame, timestamp)`` interface, so the engine
    loop is unchanged.  Timestamps come from the *recording*, which is what makes
    replay bit-for-bit deterministic regardless of machine speed.
    """

    def __init__(self, recording: Recording, use_landmarks: bool = True, resample_fps: float | None = None) -> None:
        self.recording = recording
        self.use_landmarks = use_landmarks
        self.resample_fps = resample_fps
        self.index = 0
        self._emitted = 0
        self._t0 = recording.frames[0]["t"] if recording.frames else 0.0

    def reset(self) -> None:
        self.index = 0
        self._emitted = 0

    @property
    def exhausted(self) -> bool:
        return self.index >= len(self.recording.frames)

    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray | None, timestamp: float, frame_id: int = 0, camera: dict | None = None) -> Observation | None:
        if self.exhausted:
            return None
        rec = self.recording.frames[self.index]
        self.index += 1
        t = float(rec["t"]) - self._t0

        obs = Observation(timestamp=t, frame_id=int(rec.get("i", self.index)), camera=dict(camera or {}))
        obs.camera.update(
            {
                "replay": True,
                "label": rec.get("notes", {}).get("label", ""),
                "intent_label": rec.get("notes", {}).get("intent_label", ""),
                "recorded_state": rec.get("state"),
            }
        )

        lm = self.recording.landmarks.get(int(rec.get("i", -1)))
        notes = rec.get("notes", {}) or {}
        if self.use_landmarks:
            points = None
            conf = 1.0
            if lm and lm.get("hands"):
                for label, arr in lm.get("hands", {}).items():
                    obs.hands[str(label)] = Landmarks(points=np.asarray(arr, dtype=np.float64), name=str(label))
                meta = lm.get("meta", {}) or {}
                confs = meta.get("hand_confidences", {}) or {}
                obs.hand_confidences = {str(k): float(v) for k, v in confs.items()}
                obs.primary_hand = meta.get("primary_hand") or (next(iter(obs.hands)) if obs.hands else None)
                if obs.primary_hand in obs.hands:
                    obs.hand = obs.hands[obs.primary_hand]
                    obs.hand_confidence = float(obs.hand_confidences.get(obs.primary_hand, lm.get("conf", 1.0)))
                return obs
            if lm and lm.get("points"):
                points = np.asarray(lm["points"], dtype=np.float64)
                conf = float(lm.get("conf", 1.0))
            elif notes.get("hand_points") is not None:
                # Fall back to the inline copy so recordings written with
                # `include_landmarks: false` still replay their landmark stream.
                points = np.asarray(notes["hand_points"], dtype=np.float64)
                conf = float(notes.get("hand_confidence", 1.0))
            if points is not None:
                obs.hand = Landmarks(points=points, name="hand")
                obs.hand_confidence = conf
                obs.hands = {"hand": obs.hand}
                obs.hand_confidences = {"hand": conf}
                obs.primary_hand = "hand"
        return obs

    def close(self) -> None:
        pass

    def __len__(self) -> int:
        return len(self.recording.frames)


def iter_recordings(directory: str | Path) -> Iterator[Path]:
    p = Path(directory)
    yield from sorted(p.glob("*.jsonl"))


def diff_sessions(a: Recording, b: Recording) -> dict:
    """Compare the per-frame decisions of two recordings of the same session.

    Used by the V1-vs-V2 comparison: replay the *same* landmark stream through
    both algorithms and count where they disagree.
    """
    n = min(len(a.frames), len(b.frames))
    gesture_disagreements = 0
    intent_disagreements = 0
    state_disagreements = 0
    for i in range(n):
        fa, fb = a.frames[i], b.frames[i]
        if fa.get("gesture", {}).get("name") != fb.get("gesture", {}).get("name"):
            gesture_disagreements += 1
        if fa.get("intent_committed") != fb.get("intent_committed"):
            intent_disagreements += 1
        if fa.get("state") != fb.get("state"):
            state_disagreements += 1
    return {
        "frames": n,
        "gesture_disagreement_rate": gesture_disagreements / n if n else 0.0,
        "intent_disagreement_rate": intent_disagreements / n if n else 0.0,
        "state_disagreement_rate": state_disagreements / n if n else 0.0,
    }
