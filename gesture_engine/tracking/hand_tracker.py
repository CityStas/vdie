"""Hand tracking adapter.

MediaPipe is an *implementation detail of this module only*.  Everything
downstream sees :class:`~gesture_engine.types.Observation`.

MediaPipe 1.x dropped the legacy ``mp.solutions.hands`` API; the Tasks API
(``mediapipe.tasks.python.vision.HandLandmarker``) is the only supported path.
It requires a ``.task`` model bundle, which is fetched once and cached locally.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from ..config import EngineConfig
from ..types import Landmarks, Observation

log = logging.getLogger(__name__)


class TrackerError(RuntimeError):
    pass


def resolve_delegate(use_gpu: bool):
    """Return the MediaPipe ``BaseOptions.Delegate`` enum member.

    MediaPipe's Tasks API takes an **enum**, not a string: ``to_ctypes()`` does
    ``self.delegate.value``, so passing ``"CPU"`` fails with
    ``AttributeError: 'str' object has no attribute 'value'`` at landmarker
    construction time.  The failure only appears on the real-camera path, which
    is why it survived the synthetic-only test suite.  Keeping the choice in one
    place makes it testable without a camera.
    """
    from mediapipe.tasks.python import BaseOptions

    return BaseOptions.Delegate.GPU if use_gpu else BaseOptions.Delegate.CPU


def ensure_model(path: Path, url: str, auto_download: bool = True) -> Path:
    """Resolve the model bundle, downloading it once if necessary."""
    if path.exists() and path.stat().st_size > 0:
        return path
    if not auto_download:
        raise TrackerError(
            f"MediaPipe model not found at {path}. Download it manually from {url} "
            f"or set tracking.auto_download: true."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    log.info("downloading MediaPipe hand model -> %s", path)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, tmp.open("wb") as fh:  # noqa: S310
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise TrackerError(f"failed to download model from {url}: {exc}") from exc
    tmp.replace(path)
    return path


class HandTracker:
    """MediaPipe Tasks hand landmarker behind a tiny, stable interface."""

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._landmarker: Any = None
        self._mode = cfg.tracking.running_mode
        self._missing = 0
        self._last_primary_label: str | None = None

    # ------------------------------------------------------------------ #
    def _lazy_init(self) -> None:
        if self._landmarker is not None:
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover
            raise TrackerError(
                "mediapipe is not installed. `pip install mediapipe`, or set "
                "tracking.backend to 'synthetic' to run without a camera."
            ) from exc

        tcfg = self.cfg.tracking
        model = ensure_model(
            Path(tcfg.model_path),
            tcfg.model_url,
            auto_download=tcfg.auto_download,
        )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model), delegate=resolve_delegate(tcfg.use_gpu)),
            running_mode=vision.RunningMode.VIDEO if self._mode == "video" else vision.RunningMode.IMAGE,
            num_hands=max(1, tcfg.max_hands),
            min_hand_detection_confidence=tcfg.min_detection_confidence,
            min_hand_presence_confidence=tcfg.min_detection_confidence,
            min_tracking_confidence=tcfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray, timestamp: float, frame_id: int = 0, camera: dict | None = None) -> Observation:
        """Run detection on a BGR frame and return a generic Observation."""
        obs = Observation(timestamp=timestamp, frame_id=frame_id, camera=camera or {})
        if frame is None:
            return obs

        self._lazy_init()
        assert self._landmarker is not None
        import mediapipe as mp

        rgb = frame[:, :, ::-1] if frame.ndim == 3 and frame.shape[2] == 3 else frame
        rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._mode == "video":
            result = self._landmarker.detect_for_video(mp_image, int(timestamp * 1000))
        else:
            result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            self._missing += 1
            return obs

        self._missing = 0
        handedness = getattr(result, "handedness", None) or getattr(result, "handednesses", None)
        best_idx = 0
        best_conf = -1.0
        used: set[str] = set()
        for i, raw in enumerate(result.hand_landmarks):
            points = np.asarray([[lm.x, lm.y, lm.z] for lm in raw], dtype=np.float64)
            vis = np.asarray([getattr(lm, "visibility", 1.0) or 1.0 for lm in raw], dtype=np.float64)
            label = None
            confidence = 1.0
            if handedness and i < len(handedness):
                try:
                    cat = handedness[i][0]
                    label = str(getattr(cat, "category_name", None) or getattr(cat, "display_name", None) or "").lower()
                    confidence = float(getattr(cat, "score", 1.0))
                except Exception:  # noqa: BLE001
                    pass
            if label not in {"left", "right"}:
                label = f"hand_{i}"
            if label in used:
                label = f"{label}_{i}"
            used.add(label)
            obs.hands[label] = Landmarks(points=points, visibility=vis, name=label)
            obs.hand_confidences[label] = confidence
            if confidence > best_conf:
                best_idx, best_conf = i, confidence

        labels = list(obs.hands.keys())
        highest_label = labels[best_idx] if labels and best_idx < len(labels) else (labels[0] if labels else None)
        # Prefer the previously selected physical hand when it is still visible
        # and its confidence is close to the strongest hand. This prevents the
        # WHERE/WHAT roles from flipping because MediaPipe scores two hands by a
        # few hundredths differently from frame to frame.
        primary_label = highest_label
        if self._last_primary_label in obs.hands:
            last_conf = float(obs.hand_confidences.get(self._last_primary_label, 0.0))
            best_conf_value = float(obs.hand_confidences.get(highest_label, 0.0)) if highest_label else 0.0
            if last_conf >= best_conf_value - 0.10:
                primary_label = self._last_primary_label
        self._last_primary_label = primary_label
        obs.primary_hand = primary_label
        if primary_label is not None:
            obs.hand = obs.hands[primary_label]
            obs.hand_confidence = float(obs.hand_confidences.get(primary_label, best_conf if best_conf >= 0 else 1.0))
        return obs

    # ------------------------------------------------------------------ #
    def _pick_hand(self, result: Any) -> int:
        """Compatibility helper: return the most confident detected hand."""
        handedness = getattr(result, "handedness", None) or getattr(result, "handednesses", None)
        if not handedness:
            return 0
        scores = [float(c[0].score) if c else 0.0 for c in handedness]
        return int(np.argmax(scores)) if scores else 0

    def close(self) -> None:
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:  # noqa: BLE001
                pass
            self._landmarker = None

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullTracker:
    """Always reports "no hand".  Useful for benchmarking the pipeline overhead."""

    def __init__(self, cfg: EngineConfig | None = None) -> None:
        self.cfg = cfg

    def process(self, frame: np.ndarray, timestamp: float, frame_id: int = 0, camera: dict | None = None) -> Observation:
        return Observation(timestamp=timestamp, frame_id=frame_id, camera=camera or {})

    def close(self) -> None:
        pass
