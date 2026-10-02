"""Camera abstraction.

Camera *configuration* lives in ``config.yaml`` (``capture`` + ``profiles`` +
``calibration``), never in code.  That separation is what makes the "camera
configuration vs algorithm version" benchmark split possible: the same recording
harness can be run with ``profile: g85_12_60`` and with ``profile: g85_fisheye``
and the two results are attributable.

Optional fisheye undistortion is deliberately **off by default**.  Undistortion
costs a full-frame remap per frame and, more importantly, it changes the landmark
geometry — so it is a separate experimental condition, not a default.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import EngineConfig

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    pass


@dataclass
class Frame:
    image: np.ndarray | None
    timestamp: float
    frame_id: int
    meta: dict = field(default_factory=dict)


def _backend_code(cv2, name: str) -> int:
    return {
        "dshow": getattr(cv2, "CAP_DSHOW", 700),
        "msmf": getattr(cv2, "CAP_MSMF", 1400),
        "any": getattr(cv2, "CAP_ANY", 0),
    }.get(name, getattr(cv2, "CAP_ANY", 0))


class Undistorter:
    """Optional fisheye / pinhole undistortion driven by a calibration file."""

    def __init__(self, cfg: EngineConfig) -> None:
        self.enabled = False
        self._map1 = None
        self._map2 = None
        self._calib: dict = {}
        ccfg = cfg.calibration
        if not ccfg.enabled:
            return
        path = Path(ccfg.path)
        if not path.exists():
            log.warning("calibration enabled but %s not found; undistortion disabled", path)
            return
        data = np.load(path)
        self._calib = {k: data[k] for k in data.files}
        self._size = tuple(int(v) for v in self._calib.get("image_size", (cfg.capture.width, cfg.capture.height)))
        self._balance = float(ccfg.balance)
        self._fov_scale = float(ccfg.fov_scale)
        self.enabled = True

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        import cv2

        if self._map1 is None:
            self._build_maps(frame)
        return cv2.remap(frame, self._map1, self._map2, interpolation=cv2.INTER_LINEAR)

    def _build_maps(self, frame: np.ndarray) -> None:
        import cv2

        k = self._calib.get("K")
        d = self._calib.get("D")
        if k is None:
            raise CameraError("calibration file must contain K")
        h, w = frame.shape[:2]
        new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            k.astype(np.float64),
            d.astype(np.float64),
            (w, h),
            np.eye(3),
            balance=self._balance,
            fov_scale=self._fov_scale,
        )
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            k.astype(np.float64),
            d.astype(np.float64),
            np.eye(3),
            new_k,
            (w, h),
            cv2.CV_16SC2,
        )


class CameraSource:
    """OpenCV capture with a configuration profile."""

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._cap = None
        self._frame_id = 0
        self._failures = 0
        self._t0 = time.perf_counter()
        self.undistorter = Undistorter(cfg)

    # ------------------------------------------------------------------ #
    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise CameraError("opencv-python is required for camera capture") from exc

        c = self.cfg.capture
        cap = cv2.VideoCapture(c.device, _backend_code(cv2, c.backend))
        if not cap.isOpened():
            raise CameraError(f"cannot open camera device {c.device!r} with backend {c.backend!r}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
        cap.set(cv2.CAP_PROP_FPS, c.fps)
        if c.buffer_size:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, c.buffer_size)
        if c.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.fourcc))
        self._cap = cap
        self._t0 = time.perf_counter()
        log.info("camera open: %sx%s @%s profile=%s", c.width, c.height, c.fps, c.profile)

    # ------------------------------------------------------------------ #
    @property
    def profile(self) -> str:
        return self.cfg.capture.profile

    def read(self) -> Frame:
        if self._cap is None:
            self.open()
        assert self._cap is not None
        ok, image = self._cap.read()
        timestamp = time.perf_counter() - self._t0
        if not ok or image is None:
            self._failures += 1
            if self._failures > self.cfg.capture.max_read_failures:
                raise CameraError("camera stopped delivering frames")
            return Frame(image=None, timestamp=timestamp, frame_id=self._frame_id)
        self._failures = 0
        if self.cfg.capture.flip_horizontal:
            image = image[:, ::-1]
        image = self.undistorter.apply(image)
        self._frame_id += 1
        return Frame(
            image=image,
            timestamp=timestamp,
            frame_id=self._frame_id,
            meta={
                "profile": self.profile,
                "lens": self.cfg.active_profile().lens,
                "undistorted": self.undistorter.enabled,
                "size": (image.shape[1], image.shape[0]),
            },
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraSource":
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def list_cameras(
    max_index: int = 6,
    backend: str = "any",
    probe_frames: int = 20,
    width: int = 0,
    height: int = 0,
    fps: int = 0,
) -> list[dict]:
    """Probe indices 0..max_index-1 and report what actually delivers frames.

    Probing with one hard-coded backend answers the wrong question.  On Windows
    ``DSHOW`` reports ``isOpened() == True`` for indices it cannot capture from
    and then returns the *default* 640x480 with ``fps == -1``; for devices it can
    open it may still deliver a fraction of the frames (measured on this machine:
    MSMF 28.7 fps vs DSHOW 9.3 fps at 1280x720).  So the probe reads frames and
    times them, and takes the delivered resolution from the frame itself rather
    than from the requested property.

    ``width``/``height``/``fps`` are requested before probing when non-zero, so
    the numbers reported are the ones the engine will actually see — a probe at
    the device default would report 640x480 for a camera the engine drives at
    1280x720, which is a different (and equally misleading) answer.
    """
    import cv2

    found: list[dict] = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, _backend_code(cv2, backend))
        entry: dict = {"index": i, "backend": backend, "opened": bool(cap.isOpened())}
        try:
            if entry["opened"]:
                if width:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                if height:
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                if fps:
                    cap.set(cv2.CAP_PROP_FPS, fps)
                ok, frame = cap.read()
                if not ok or frame is None:
                    entry.update(frames=0, fps=0.0, width=0, height=0)
                else:
                    t0 = time.perf_counter()
                    n = 1
                    for _ in range(max(0, probe_frames - 1)):
                        good, _f = cap.read()
                        n += 1 if good else 0
                    dt = time.perf_counter() - t0
                    entry.update(
                        height=int(frame.shape[0]),
                        width=int(frame.shape[1]),
                        frames=n,
                        fps=round(n / dt, 1) if dt > 0 else 0.0,
                    )
            found.append(entry)
        finally:
            cap.release()
    return found


class NullCamera:
    """Frame source for replay / headless runs."""

    def __init__(self, cfg: EngineConfig, size: tuple[int, int] = (640, 480)) -> None:
        self.cfg = cfg
        self.size = size
        self._frame_id = 0
        self._t0 = time.perf_counter()

    def open(self) -> None:
        pass

    def read(self) -> Frame:
        self._frame_id += 1
        return Frame(
            image=None,
            timestamp=time.perf_counter() - self._t0,
            frame_id=self._frame_id,
            meta={"profile": "null"},
        )

    def close(self) -> None:
        pass
