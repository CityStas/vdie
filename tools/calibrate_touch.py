"""9-point camera -> desktop homography calibration using the real tracker.

Run on Windows:
    calibrate_touch.cmd

For each red guide point, put the *index fingertip* on the guide and press SPACE.
The utility averages the last stable samples from MediaPipe and fits a 3x3
homography from camera-normalized coordinates to physical desktop pixels.

ESC cancels. The generated JSON is consumed by the touch-surface controller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gesture_engine.config import load_config
from gesture_engine.capture.camera import CameraSource
from gesture_engine.tracking.hand_tracker import HandTracker
from gesture_engine.win32 import make_process_dpi_aware


def fit_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if src.shape != (9, 2) or dst.shape != (9, 2):
        raise ValueError("expected 9 source and 9 destination 2D points")
    A: list[list[float]] = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(A, dtype=np.float64))
    H = vt[-1].reshape(3, 3)
    return H / max(abs(H[2, 2]), 1e-12)


def _screen_size() -> tuple[int, int]:
    import ctypes

    user32 = ctypes.windll.user32
    return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))


def main() -> int:
    make_process_dpi_aware()
    ap = argparse.ArgumentParser(description="9-point index-tip touch-surface calibration")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--output", default="config/calibration/touch_surface.json")
    ap.add_argument("--device", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg.capture.device = args.device
    cfg.tracking.max_hands = max(1, cfg.tracking.max_hands)

    camera = CameraSource(cfg)
    tracker = HandTracker(cfg)
    camera.open()

    width, height = _screen_size()
    grid = [
        (0.08, 0.10), (0.50, 0.10), (0.92, 0.10),
        (0.08, 0.50), (0.50, 0.50), (0.92, 0.50),
        (0.08, 0.90), (0.50, 0.90), (0.92, 0.90),
    ]
    desktop = np.asarray([(x * (width - 1), y * (height - 1)) for x, y in grid], dtype=np.float64)
    source: list[np.ndarray] = []
    index = 0
    win = "touch surface calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    print("Move INDEX TIP to each red guide and press SPACE. Hold it still before capture.")
    print("ESC cancels. The ninth point saves the calibration automatically.")

    try:
        while index < len(grid):
            packet = camera.read()
            if packet.image is None:
                continue
            obs = tracker.process(packet.image, packet.timestamp, packet.frame_id, packet.meta)
            frame = packet.image.copy()
            h, w = frame.shape[:2]
            gx, gy = grid[index]
            guide = (int(gx * w), int(gy * h))
            cv2.circle(frame, guide, 18, (0, 0, 255), 2)
            cv2.circle(frame, guide, 3, (0, 0, 255), -1)
            cv2.putText(frame, f"Point {index + 1}/9 - index tip on red guide - SPACE to capture", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            tip = None
            if obs.hand is not None and obs.hand_confidence >= 0.55:
                tip = np.asarray(obs.hand.points[8, :2], dtype=np.float64)
                cv2.circle(frame, (int(tip[0] * w), int(tip[1] * h)), 9, (255, 255, 0), 2)
                cv2.putText(frame, f"tip {tip[0]:.3f}, {tip[1]:.3f}", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            key = cv2.waitKey(1) & 0xFF
            cv2.imshow(win, frame)
            if key == 27:
                return 2
            if key == 32:
                samples: list[np.ndarray] = []
                # Collect a short burst after SPACE so one noisy frame cannot define
                # the homography point. The live camera remains visible while sampling.
                for _ in range(12):
                    packet = camera.read()
                    if packet.image is None:
                        continue
                    obs2 = tracker.process(packet.image, packet.timestamp, packet.frame_id, packet.meta)
                    if obs2.hand is not None and obs2.hand_confidence >= 0.55:
                        samples.append(np.asarray(obs2.hand.points[8, :2], dtype=np.float64))
                    display = packet.image.copy()
                    hh, ww = display.shape[:2]
                    cv2.circle(display, (int(gx * ww), int(gy * hh)), 18, (0, 0, 255), 2)
                    cv2.putText(display, "Sampling... keep index tip still", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    cv2.imshow(win, display)
                    if cv2.waitKey(1) & 0xFF == 27:
                        return 2
                if len(samples) >= 6:
                    median = np.median(np.asarray(samples), axis=0)
                    spread = float(np.median(np.linalg.norm(np.asarray(samples) - median, axis=1)))
                    if spread <= 0.025:
                        source.append(median)
                        index += 1
                        print(f"captured {index}/9: {median.tolist()}, median spread={spread:.4f}")
                    else:
                        print(f"capture rejected: fingertip moved too much (median spread={spread:.4f}). Try again.")
                else:
                    print("capture rejected: no stable index fingertip detected")
    finally:
        cv2.destroyAllWindows()
        tracker.close()
        camera.close()

    if len(source) != 9:
        print("Calibration cancelled/incomplete.")
        return 2

    src = np.asarray(source, dtype=np.float64)
    H = fit_homography(src, desktop)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "homography_v1",
        "homography": H.tolist(),
        "source_points": src.tolist(),
        "desktop_points": desktop.tolist(),
        "desktop_size": [width, height],
        "camera_size": [cfg.capture.width, cfg.capture.height],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
