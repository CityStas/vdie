"""Real-hardware path: MediaPipe Tasks wiring and camera probing.

Everything here is written so it runs **without a camera and without a network**,
because the defects it guards against are exactly the ones the synthetic-only
test suite could not see.  The delegate bug is the reference example: the whole
synthetic pipeline passed while ``--source mediapipe`` crashed on the first
frame, because MediaPipe's ``BaseOptions`` takes an enum and the code passed a
string.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gesture_engine.capture.camera import CameraError, CameraSource, _backend_code, list_cameras
from gesture_engine.config import load_config
from gesture_engine.tracking.hand_tracker import TrackerError, ensure_model, resolve_delegate

# --------------------------------------------------------------------------- #
# MediaPipe Tasks wiring
# --------------------------------------------------------------------------- #


def test_resolve_delegate_returns_the_enum_not_a_string():
    """Regression: a string delegate crashed landmarker construction.

    MediaPipe's ``BaseOptions.to_ctypes()`` does ``self.delegate.value``, so
    ``delegate="CPU"`` raised ``AttributeError: 'str' object has no attribute
    'value'`` inside ``HandLandmarker.create_from_options``.  The string never
    reached a test because no test built the landmarker.
    """
    mp = pytest.importorskip("mediapipe")
    from mediapipe.tasks.python import BaseOptions

    cpu = resolve_delegate(False)
    gpu = resolve_delegate(True)

    assert not isinstance(cpu, str), "delegate must be the enum, not a string"
    assert cpu is BaseOptions.Delegate.CPU
    assert gpu is BaseOptions.Delegate.GPU
    # The attribute the library actually dereferences.
    assert isinstance(cpu.value, int)
    assert isinstance(gpu.value, int)
    assert cpu.value != gpu.value
    assert mp is not None


def test_base_options_accepts_the_resolved_delegate():
    """The value must survive the real constructor, not just the type check."""
    pytest.importorskip("mediapipe")
    from mediapipe.tasks.python import BaseOptions

    opts = BaseOptions(model_asset_path="does-not-need-to-exist.task", delegate=resolve_delegate(False))
    # to_ctypes is the exact call that failed before; it must not raise.
    ctypes_opts = opts.to_ctypes()
    assert ctypes_opts.delegate == BaseOptions.Delegate.CPU.value


def test_ensure_model_uses_an_existing_file_without_network(tmp_path: Path):
    model = tmp_path / "hand_landmarker.task"
    model.write_bytes(b"not really a model, but non-empty")
    # auto_download=False proves the file was used: a download attempt would raise.
    assert ensure_model(model, "http://127.0.0.1:1/unreachable", auto_download=False) == model


def test_ensure_model_refuses_to_download_when_disabled(tmp_path: Path):
    missing = tmp_path / "absent.task"
    with pytest.raises(TrackerError, match="not found"):
        ensure_model(missing, "http://127.0.0.1:1/unreachable", auto_download=False)


def test_ensure_model_treats_an_empty_file_as_missing(tmp_path: Path):
    empty = tmp_path / "empty.task"
    empty.touch()
    with pytest.raises(TrackerError):
        ensure_model(empty, "http://127.0.0.1:1/unreachable", auto_download=False)


# --------------------------------------------------------------------------- #
# Camera probing
# --------------------------------------------------------------------------- #


def test_backend_names_map_to_distinct_opencv_codes():
    cv2 = pytest.importorskip("cv2")

    codes = {_backend_code(cv2, name) for name in ("any", "msmf", "dshow")}
    assert len(codes) == 3, "backend names must not silently collapse to one code"
    # An unknown name degrades to the platform default rather than raising.
    assert _backend_code(cv2, "nonsense") == getattr(cv2, "CAP_ANY", 0)


def test_list_cameras_with_no_indices_returns_nothing():
    pytest.importorskip("cv2")
    assert list_cameras(max_index=0) == []


def test_list_cameras_reports_delivered_resolution_not_requested():
    """The probe must report what arrived, not what was asked for.

    A probe that echoes ``CAP_PROP_FRAME_WIDTH`` reports the default 640x480 for
    a device that never opened, which is how a broken backend looks healthy.
    """
    pytest.importorskip("cv2")
    for cam in list_cameras(max_index=6):
        assert "backend" in cam and "opened" in cam
        if cam["opened"] and cam.get("frames"):
            assert cam["width"] > 0 and cam["height"] > 0
            assert cam["fps"] > 0.0
        else:
            # A device that delivers nothing must not report a plausible size.
            assert cam.get("frames", 0) == 0


def test_camera_source_rejects_a_device_that_does_not_exist():
    """A wrong --device must fail loudly instead of producing black frames."""
    pytest.importorskip("cv2")
    cfg = load_config()
    cfg.capture.device = 97  # far outside any real enumeration
    cfg.capture.backend = "any"
    with pytest.raises(CameraError):
        CameraSource(cfg).open()


def test_camera_source_does_not_distort_when_calibration_is_off():
    cfg = load_config()
    cfg.calibration.enabled = False
    src = CameraSource(cfg)
    assert src.undistorter.enabled is False
    assert src.profile == cfg.capture.profile
