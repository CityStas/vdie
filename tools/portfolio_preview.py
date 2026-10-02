"""Portfolio preview renderer for the Vision Desktop Interaction Engine.

The engine's own debug window is a plain camera feed with an engineering HUD on
top.  That is the right tool while tuning and the wrong picture for a portfolio:
the camera is whatever the room looks like, the hand is wherever the operator's
hand happened to be, and the result is not reproducible.

This tool keeps the *data* authentic and replaces only the *scene*:

* a scripted two-hand timeline is fed through the **real** engine
  (features -> motion -> gesture -> FSM -> targets -> commit -> intent field ->
  policy -> cursor -> bimanual).  Every number drawn on the canvas comes out of a
  ``FrameReport``: state, gesture, intent probabilities, target lock, commit
  kind, bimanual action, cursor pixel.  Nothing is mocked.
* the two hands are the engine's own analytic hand model
  (:mod:`gesture_engine.tracking.landmark_model`) — the same 21 landmarks the
  recogniser was written against, so the pose that is drawn is the pose that was
  classified.
* the scene is drawn on a perspective plane: a virtual touch surface seen at an
  angle, with both hand rigs lying on it, a trajectory trail, the absolute
  cursor, and the desktop target map beside it.

Output is a still (PNG + JPG) per scripted moment, plus an optional mp4 of the
whole timeline.

    python -m tools.portfolio_preview --out preview
    python -m tools.portfolio_preview --list
    python -m tools.portfolio_preview --moment drag --clip
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gesture_engine import GestureEngine, build_dispatcher, build_target_provider, load_config
from gesture_engine.config import apply_overrides
from gesture_engine.debug.overlay import HAND_CONNECTIONS
from gesture_engine.tracking.landmark_model import POSES, HandPose, build_hand
from gesture_engine.types import Landmarks, Observation

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: .venv\\Scripts\\pip install pillow") from exc


# --------------------------------------------------------------------------- #
# Palette (dark, instrument-panel)
# --------------------------------------------------------------------------- #

BG_TOP = (7, 8, 11)
BG_BOT = (13, 16, 21)
PLANE_FILL = (9, 12, 17)
PLANE_GRID = (26, 33, 45)
PLANE_EDGE = (44, 55, 72)
PANEL = (16, 19, 25)
PANEL_EDGE = (38, 46, 60)
TXT = (230, 237, 245)
TXT_DIM = (138, 148, 166)
TXT_FAINT = (74, 83, 97)
ACCENT = (53, 224, 192)      # primary hand / WHERE
ACCENT2 = (255, 174, 51)     # secondary hand / WHAT
ACCENT3 = (108, 140, 255)    # desktop map / targets
TRAIL = (255, 210, 74)
DANGER = (255, 92, 92)

STATE_COLOR = {
    "IDLE": (140, 150, 165),
    "ACTIVATING": (0, 200, 255),
    "ARMED": (0, 220, 255),
    "CURSOR": ACCENT,
    "PINCH_DOWN": (255, 160, 0),
    "DRAGGING": (255, 120, 40),
    "SCROLL": (255, 0, 200),
    "PAUSED": (0, 165, 255),
    "COOLDOWN": (180, 180, 0),
    "EMERGENCY_CANCEL": DANGER,
}

FONT_DIR = Path("C:/Windows/Fonts")
FONT_FILES = {
    "reg": "Inter-Regular.otf",
    "med": "Inter-Medium.otf",
    "semi": "Inter-SemiBold.otf",
    "bold": "Inter-Bold.otf",
    "mono": "consola.ttf",
    "monob": "consolab.ttf",
}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    path = FONT_DIR / FONT_FILES[kind]
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# Scripted two-hand timeline
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Key:
    """One hand pose held from ``t`` onwards (until the next key)."""

    t: float
    pose: str
    tip: tuple[float, float]
    scale: float = 0.22
    rot: float = 0.0


#: Primary hand = WHERE (the pointer).
#:
#: The movement budget is set by the recogniser, not by taste.  A fast travel
#: becomes SWIPE_*, a slow *vertical* stroke becomes SCROLL_* — those are the
#: engine's own thresholds (``swipe_min_speed`` 0.55, ``scroll_max_speed`` 1.10,
#: ``scroll_min_distance`` 0.10) and they are what a real operator fights too.
#: The only travel that stays INDEX_UP end to end is slow *and* horizontal:
#: mean speed ~0.27 u/s keeps swipe's speed confidence at zero, and a near-zero
#: dy keeps the scroll stroke confidence at zero.
PRIMARY_KEYS: tuple[Key, ...] = (
    Key(0.00, "INDEX_UP", (0.40, 0.32), 0.19, 4.0),
    Key(0.85, "INDEX_UP", (0.40, 0.32), 0.19, 4.0),
    Key(2.05, "INDEX_UP", (0.72, 0.30), 0.19, 2.0),
    Key(2.35, "INDEX_UP", (0.72, 0.295), 0.19, 1.0),
    Key(3.30, "INDEX_UP", (0.72, 0.295), 0.19, 1.0),
    Key(4.45, "INDEX_UP", (0.40, 0.33), 0.19, -2.0),
    Key(4.60, "PINCH", (0.52, 0.34), 0.19, -6.0),
    Key(5.00, "PINCH", (0.42, 0.32), 0.19, -6.0),
    Key(5.60, "INDEX_UP", (0.40, 0.32), 0.19, 4.0),
)

#: Secondary hand = WHAT (the modifier).  Short pinch/release = left click,
#: held pinch + primary movement = drag, both pinched = zoom.
#:
#: Each pinch is held well past ``POSE_BLEND_S``: the pose morph deliberately
#: drops ``force_pinch`` while the fingers are still moving, so a key pair
#: closer than ~0.2 s never produces a *closed* pinch for long enough to be
#: recognised — the click edge then fires on noise instead of on the gesture.
SECONDARY_KEYS: tuple[Key, ...] = (
    Key(0.00, "OPEN_PALM", (0.28, 0.62), 0.16, -20.0),
    Key(2.55, "OPEN_PALM", (0.28, 0.62), 0.16, -20.0),
    Key(2.60, "PINCH", (0.28, 0.62), 0.16, -20.0),
    Key(2.88, "OPEN_PALM", (0.28, 0.62), 0.16, -20.0),
    Key(3.12, "OPEN_PALM", (0.30, 0.60), 0.16, -22.0),
    Key(3.15, "PINCH", (0.30, 0.60), 0.16, -22.0),
    Key(4.48, "OPEN_PALM", (0.30, 0.60), 0.16, -22.0),
    Key(4.60, "PINCH", (0.28, 0.60), 0.16, -18.0),
    Key(5.00, "PINCH", (0.10, 0.78), 0.16, -18.0),
    Key(5.60, "OPEN_PALM", (0.28, 0.62), 0.16, -20.0),
)

POSE_BLEND_S = 0.10


def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _lerp_curl(a: str, b: str, u: float):
    ca, cb = POSES[a].curl, POSES[b].curl
    return tuple(float(ca[i] + (cb[i] - ca[i]) * u) for i in range(5))  # type: ignore[return-value]


def pose_at(keys: tuple[Key, ...], t: float) -> HandPose:
    """Interpolate position smoothly and morph the finger configuration.

    The steady pose inside the interval ``[keys[i-1].t, keys[i].t)`` is
    ``keys[i-1].pose`` — that is the key that *established* it.  Over the first
    ``POSE_BLEND_S`` the configuration morphs from ``keys[i-2].pose`` into it, so
    a key pair separated by less than the blend never produces a closed pose at
    all (which is why every pinch below is held far longer than the blend).
    """
    if t <= keys[0].t:
        k = keys[0]
        return _pose(k, k.pose)
    for i in range(1, len(keys)):
        if t < keys[i].t:
            a, b = keys[i - 1], keys[i]
            span = max(b.t - a.t, 1e-6)
            u = (t - a.t) / span
            pos = (
                a.tip[0] + (b.tip[0] - a.tip[0]) * _smooth(u),
                a.tip[1] + (b.tip[1] - a.tip[1]) * _smooth(u),
            )
            scale = a.scale + (b.scale - a.scale) * _smooth(u)
            rot = a.rot + (b.rot - a.rot) * _smooth(u)
            into = t - a.t
            if into < POSE_BLEND_S and i >= 2 and keys[i - 2].pose != a.pose:
                curl = _lerp_curl(keys[i - 2].pose, a.pose, into / POSE_BLEND_S)
                return HandPose(curl=curl, rotation_deg=rot, position=pos, scale=scale, force_pinch=False)
            return _pose(Key(a.t, a.pose, pos, scale, rot), a.pose)
    k = keys[-1]
    return _pose(k, k.pose)


def _pose(k: Key, name: str) -> HandPose:
    base = POSES[name]
    return HandPose(
        curl=base.curl,
        rotation_deg=k.rot,
        position=k.tip,
        scale=k.scale,
        force_pinch=base.force_pinch,
        spread_deg=base.spread_deg,
    )


def landmarks_for(keys: tuple[Key, ...], t: float, rng: np.random.Generator, noise: float = 0.0007) -> np.ndarray:
    """21 landmarks in image-normalized coords, index tip exactly on the key path."""
    p = pose_at(keys, t)
    # The model is affine in position, so placing the wrist at ``tip - offset``
    # puts landmark 8 exactly on the scripted point.
    probe = HandPose(curl=p.curl, rotation_deg=p.rotation_deg, position=(0.0, 0.0), scale=p.scale,
                     force_pinch=p.force_pinch, spread_deg=p.spread_deg)
    offset = build_hand(probe)[8, :2]
    wrist = (float(p.position[0] - offset[0]), float(p.position[1] - offset[1]))
    final = HandPose(curl=p.curl, rotation_deg=p.rotation_deg, position=wrist, scale=p.scale,
                     force_pinch=p.force_pinch, spread_deg=p.spread_deg)
    return build_hand(final, noise=noise, rng=rng)


class TwoHandTracker:
    """Emits two hands (primary = Right / WHERE, secondary = Left / WHAT)."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(20261002)
        self._t0: float | None = None

    def reset(self, t0: float | None = None) -> None:
        self._t0 = t0

    def process(self, frame, timestamp, frame_id=0, camera=None):
        if self._t0 is None:
            self._t0 = timestamp
        t = timestamp - self._t0
        primary = landmarks_for(PRIMARY_KEYS, t, self._rng)
        secondary = landmarks_for(SECONDARY_KEYS, t, self._rng)
        obs = Observation(timestamp=timestamp, frame_id=frame_id, camera=dict(camera or {}))
        obs.hands = {
            "Right": Landmarks(points=primary, name="Right"),
            "Left": Landmarks(points=secondary, name="Left"),
        }
        obs.hand_confidences = {"Right": 0.97, "Left": 0.95}
        obs.primary_hand = "Right"
        obs.hand = obs.hands["Right"]
        obs.hand_confidence = 0.97
        return obs

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Engine run
# --------------------------------------------------------------------------- #


@dataclass
class Sample:
    t: float
    report: object
    hands: dict
    trail: np.ndarray


def build_engine(cfg, tracker):
    dispatcher = build_dispatcher(cfg, dry_run=True)
    engine = GestureEngine(
        cfg,
        tracker,
        dispatcher=dispatcher,
        target_provider=build_target_provider(cfg, screen=dispatcher.mouse.screen_size()),
    )
    return engine, dispatcher


def run_timeline(fps: int = 30, duration: float = 5.9):
    cfg = load_config()
    apply_overrides(cfg, {
        "control.enabled": False,
        "debug.overlay": False,
        "targets.provider": "static",
        "bimanual.enabled": True,
        "cursor.controller": "touch_surface",
        "cursor.relative_mode": False,
    })
    tracker = TwoHandTracker(cfg)
    engine, dispatcher = build_engine(cfg, tracker)
    engine.start()
    samples: list[Sample] = []
    frames = int(duration * fps)
    for i in range(frames):
        ts = i / fps
        report = engine.process_frame(None, ts, i, {"profile": "synthetic"})
        if report is None:
            continue
        hands = {k: np.asarray(v, dtype=np.float64) for k, v in report.notes.get("hands", {}).items()}
        trail = engine.motion.recent_points(1.2)
        samples.append(Sample(t=ts, report=report, hands=hands, trail=np.asarray(trail, dtype=np.float64) if trail is not None else np.zeros((0, 2))))
    engine.stop()
    return samples


# --------------------------------------------------------------------------- #
# Plane geometry
# --------------------------------------------------------------------------- #


@dataclass
class Plane:
    """Perspective quad for the virtual touch surface, plus its homography."""

    quad: np.ndarray  # 4x2, order: TL(far-left) TR(far-right) BR(near-right) BL(near-left)
    H: np.ndarray

    @staticmethod
    def build(box: tuple[int, int, int, int], tilt: float = 0.22) -> "Plane":
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        inset = w * tilt
        quad = np.array([
            [x0 + inset, y0],
            [x1 - inset, y0],
            [x1, y1],
            [x0, y1],
        ], dtype=np.float32)
        src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        import cv2

        H = cv2.getPerspectiveTransform(src, quad)
        return Plane(quad=quad.astype(np.float64), H=np.asarray(H, dtype=np.float64))

    def map(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64)
        ones = np.ones((p.shape[0], 1))
        q = np.hstack([p[:, :2], ones]) @ self.H.T
        return q[:, :2] / q[:, 2:3]

    def map_one(self, u: float, v: float) -> tuple[float, float]:
        q = self.map(np.array([[u, v]]))[0]
        return (float(q[0]), float(q[1]))


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #


def vgrad(size, top, bot) -> Image.Image:
    w, h = size
    col = np.zeros((h, 1, 3), dtype=np.float64)
    for i in range(3):
        col[:, 0, i] = np.linspace(top[i], bot[i], h)
    return Image.fromarray(np.repeat(col, w, axis=1).astype(np.uint8), "RGB")


def radial_glow(size, center, radius, color, strength=0.5) -> Image.Image:
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / max(radius, 1)
    a = np.clip(1.0 - d, 0.0, 1.0) ** 2 * strength
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[..., 0], layer[..., 1], layer[..., 2] = color
    layer[..., 3] = (a * 255).astype(np.uint8)
    return Image.fromarray(layer, "RGBA")


def panel(img: Image.Image, box, radius=14, fill=PANEL, edge=PANEL_EDGE, alpha=232, width=1):
    x0, y0, x1, y1 = [int(v) for v in box]
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill + (alpha,), outline=edge + (255,), width=width)
    return Image.alpha_composite(img.convert("RGBA"), layer)


def chip(img, xy, text, fnt, color, *, fill=None, pad=(10, 5), radius=8, edge=None, text_color=None):
    d = ImageDraw.Draw(img)
    tw = d.textlength(text, font=fnt)
    th = fnt.size
    x, y = xy
    box = [x, y, x + tw + pad[0] * 2, y + th + pad[1] * 2]
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dd = ImageDraw.Draw(layer)
    f = (color + (28,)) if fill is None else (fill + (40,))
    e = (color + (170,)) if edge is None else (edge + (200,))
    dd.rounded_rectangle(box, radius=radius, fill=f, outline=e, width=1)
    out = Image.alpha_composite(img.convert("RGBA"), layer)
    ImageDraw.Draw(out).text((x + pad[0], y + pad[1] - 1), text, font=fnt, fill=text_color or color)
    return out, box


def bar(img, box, value, color, *, track=(30, 36, 46), radius=5):
    x0, y0, x1, y1 = [int(v) for v in box]
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=track)
    w = int((x1 - x0) * max(0.0, min(1.0, value)))
    if w > 2:
        d.rounded_rectangle([x0, y0, x0 + w, y1], radius=radius, fill=color)
    return img


def dashed_line(img, p0, p1, color, *, width=2, dash=9, gap=7, alpha=150):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    total = float(np.linalg.norm(p1 - p0))
    if total < 1:
        return img
    direction = (p1 - p0) / total
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    s = 0.0
    while s < total:
        e = min(s + dash, total)
        a = p0 + direction * s
        b = p0 + direction * e
        d.line([tuple(a), tuple(b)], fill=color + (alpha,), width=width)
        s = e + gap
    return Image.alpha_composite(img.convert("RGBA"), layer)


def glow_layer(img, draw_fn, radius=10, strength=0.85):
    """Render ``draw_fn`` onto a transparent layer, blur it, composite."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    blurred = layer.filter(ImageFilter.GaussianBlur(radius))
    a = np.asarray(blurred, dtype=np.float64)
    a[..., 3] *= strength
    blurred = Image.fromarray(a.astype(np.uint8), "RGBA")
    return Image.alpha_composite(Image.alpha_composite(img.convert("RGBA"), blurred), layer)


# --------------------------------------------------------------------------- #
# Frame renderer
# --------------------------------------------------------------------------- #


class Renderer:
    def __init__(self, w: int = 1600, h: int = 1000, budget_ms: float = 1000.0 / 30.0) -> None:
        self.w, self.h = w, h
        self.budget_ms = budget_ms
        self.f_title = font("semi", 20)
        self.f_sub = font("reg", 15)
        self.f_mono = font("mono", 15)
        self.f_monob = font("monob", 15)
        self.f_mono_s = font("mono", 13)
        self.f_tiny = font("mono", 11)
        self.f_h = font("bold", 17)
        self.f_lbl = font("med", 13)
        self.f_lbls = font("med", 12)
        self.f_big = font("bold", 26)

    # -- layout ---------------------------------------------------------- #
    def render(self, s: Sample) -> Image.Image:
        rep = s.report
        W, H = self.w, self.h
        img = vgrad((W, H), BG_TOP, BG_BOT).convert("RGBA")
        img = Image.alpha_composite(img, radial_glow((W, H), (W * 0.32, H * 0.42), W * 0.52, (24, 60, 60), 0.55))
        img = Image.alpha_composite(img, radial_glow((W, H), (W * 0.86, H * 0.30), W * 0.30, (28, 34, 60), 0.45))

        header_h = 66
        footer_h = 46
        pad = 26
        left_w = int(W * 0.60)
        plane_box = (pad, header_h + pad, pad + left_w - 18, H - footer_h - pad)
        right_x0 = plane_box[2] + 22
        right_x1 = W - pad

        plane = Plane.build(plane_box, tilt=0.20)
        img = self._draw_plane(img, plane, s)
        img = self._draw_hands(img, plane, s)
        self._map_cursor = None
        img = self._draw_hud(img, rep, s, right_x0, right_x1, header_h + pad, H - footer_h - pad)

        # Mapping connector: the primary index tip on the surface -> the desktop
        # coordinate the absolute touch mapper produced from it.
        primary = rep.notes.get("primary_hand")
        tip_pts = s.hands.get(primary)
        if self._map_cursor is not None and tip_pts is not None:
            tip = plane.map(tip_pts[8:9, :2])[0]
            img = dashed_line(img, (float(tip[0]), float(tip[1])), self._map_cursor,
                              (120, 132, 150), width=1, dash=5, gap=7, alpha=110)

        img = self._draw_header(img, rep, s, pad, header_h)
        img = self._draw_footer(img, rep, s, pad, W - pad, H - footer_h)
        return img

    # -- plane ----------------------------------------------------------- #
    def _draw_plane(self, img, plane: Plane, s: Sample) -> Image.Image:
        quad = [tuple(p) for p in plane.quad]

        # soft highlight in the middle of the surface so it reads as a plane and
        # not as a flat rectangle
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(layer).polygon(quad, fill=PLANE_FILL + (240,))
        img = Image.alpha_composite(img, layer)
        cx, cy = plane.map_one(0.5, 0.55)
        img = Image.alpha_composite(img, radial_glow((self.w, self.h), (cx, cy), self.w * 0.24, (26, 62, 66), 0.30))

        # grid
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for i in range(1, 20):
            u = i / 20.0
            major = (i % 2 == 0)
            col = PLANE_GRID + (255 if major else 130,)
            d.line([plane.map_one(u, 0.0), plane.map_one(u, 1.0)], fill=col, width=1)
            d.line([plane.map_one(0.0, u), plane.map_one(1.0, u)], fill=col, width=1)
        img = Image.alpha_composite(img, layer)

        # edges + corner ticks
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.line(quad + [quad[0]], fill=PLANE_EDGE + (255,), width=2, joint="curve")
        for cx_, cy_ in quad:
            d.ellipse([cx_ - 3, cy_ - 3, cx_ + 3, cy_ + 3], fill=PLANE_EDGE + (255,))
        img = Image.alpha_composite(img, layer)

        # surface caption + mapping readout, in the far-left corner of the plane
        d = ImageDraw.Draw(img)
        lx, ly = plane.map_one(0.06, 0.10)
        d.text((lx, ly), "VIRTUAL TOUCH SURFACE", font=self.f_lbls, fill=TXT_FAINT)
        d.text((lx, ly + 16), "camera 1280x720  ->  desktop 1920x1080  (absolute)", font=self.f_mono_s, fill=(58, 66, 80))

        # trajectory trail of the control point
        trail = s.trail
        if trail is not None and len(trail) > 1:
            pts = [tuple(plane.map(trail[i:i + 1])[0]) for i in range(0, len(trail), 2)]
            if len(pts) > 1:
                img = glow_layer(img, lambda dd: dd.line(pts, fill=TRAIL + (140,), width=3, joint="curve"), radius=8, strength=0.55)
                d = ImageDraw.Draw(img)
                d.line(pts, fill=TRAIL + (200,), width=2, joint="curve")
                for i, p in enumerate(pts[::4]):
                    d.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=TRAIL + (150,))
        return img

    # -- hands ----------------------------------------------------------- #
    def _draw_hands(self, img, plane: Plane, s: Sample) -> Image.Image:
        rep = s.report
        primary = rep.notes.get("primary_hand")
        order = sorted(s.hands.keys(), key=lambda k: (k == primary))  # secondary first
        for label in order:
            pts = s.hands[label]
            if pts is None or len(pts) < 21:
                continue
            color = ACCENT if label == primary else ACCENT2
            proj = plane.map(pts[:, :2])
            conf = float(rep.notes.get("hand_confidences", {}).get(label, 0.0))
            img = self._draw_one_hand(img, proj, color, label, label == primary, conf)
        return img

    def _draw_one_hand(self, img, proj, color, label, is_primary, conf) -> Image.Image:
        def depth(i: int) -> float:
            return 0.55 + 0.75 * float(proj[i, 1] / self.h)

        # palm fill (convex hull of wrist + MCPs)
        palm_idx = [0, 1, 5, 9, 13, 17]

        # contact shadow: the rig lies *on* the surface, not above it
        def shadow(d):
            off = np.array([3.0, 9.0])
            for a, b in HAND_CONNECTIONS:
                d.line([tuple(proj[a] + off), tuple(proj[b] + off)], fill=(0, 0, 0, 255), width=max(3, int(11 * depth(a))))
            d.polygon([tuple(proj[i] + off) for i in palm_idx], fill=(0, 0, 0, 190))

        img = glow_layer(img, shadow, radius=11, strength=0.55)

        def paint(d):
            for a, b in HAND_CONNECTIONS:
                d.line([tuple(proj[a]), tuple(proj[b])], fill=color + (255,), width=max(2, int(9 * depth(a))))
            d.polygon([tuple(proj[i]) for i in palm_idx], fill=color + (70,))

        img = glow_layer(img, paint, radius=13, strength=0.85)

        # crisp bones
        d = ImageDraw.Draw(img)
        for a, b in HAND_CONNECTIONS:
            d.line([tuple(proj[a]), tuple(proj[b])], fill=color + (255,), width=max(2, int(3 * depth(a))), joint="curve")
        d.polygon([tuple(proj[i]) for i in palm_idx], outline=color + (255,), width=2)

        # joints
        for i in range(21):
            r = max(2, int(3.4 * depth(i)))
            if i in (4, 8, 12, 16, 20):
                d.ellipse([proj[i][0] - r, proj[i][1] - r, proj[i][0] + r, proj[i][1] + r], fill=(255, 255, 255, 235), outline=color + (255,), width=2)
            else:
                d.ellipse([proj[i][0] - r, proj[i][1] - r, proj[i][0] + r, proj[i][1] + r], fill=color + (255,), outline=(255, 255, 255, 150), width=1)

        # index tip emphasis
        tip = tuple(proj[8])
        img = glow_layer(img, lambda dd: dd.ellipse([tip[0] - 13, tip[1] - 13, tip[0] + 13, tip[1] + 13], outline=color + (255,), width=3), radius=8, strength=0.9)
        d = ImageDraw.Draw(img)
        d.ellipse([tip[0] - 12, tip[1] - 12, tip[0] + 12, tip[1] + 12], outline=(255, 255, 255, 220), width=2)
        d.ellipse([tip[0] - 3, tip[1] - 3, tip[0] + 3, tip[1] + 3], fill=(255, 255, 255, 255))

        # label chip near the wrist, clamped so it never leaves the canvas
        wrist = tuple(proj[0])
        text = f"{'WHERE' if is_primary else 'WHAT'}  {label}  {conf:.2f}"
        fnt = self.f_lbls
        tw = d.textlength(text, font=fnt)
        bx0 = min(max(wrist[0] - tw / 2 - 9, 10.0), self.w - tw - 28.0)
        by0 = min(max(wrist[1] + 16, 10.0), self.h - fnt.size - 24.0)
        d.rounded_rectangle([bx0, by0, bx0 + tw + 18, by0 + fnt.size + 10], radius=8, fill=(8, 10, 14, 225), outline=color + (200,), width=1)
        d.text((bx0 + 9, by0 + 5), text, font=fnt, fill=color + (255,))
        return img

    # -- HUD ------------------------------------------------------------- #
    @staticmethod
    def _target_readout(rep):
        """Which target the cursor is inside, and the strongest belief.

        ``touch_surface`` runs with ``cursor.precision_near_target=false`` (it
        costs ~30 ms of lag for no measured gain), so ``cursor_target`` is always
        None in this mode.  The target *model* still produces beliefs and gravity
        every frame, so the readout is derived from those instead of from a lock
        that this controller deliberately does not take.
        """
        inside = best = None
        for b in rep.target_beliefs:
            if best is None or b.confidence > best.confidence:
                best = b
            if rep.cursor is not None and b.target.contains(rep.cursor):
                if inside is None or b.confidence > inside.confidence:
                    inside = b
        return inside, best

    def _draw_hud(self, img, rep, s: Sample, x0, x1, y0, y1) -> Image.Image:
        img = panel(img, (x0, y0, x1, y1), radius=16)
        d = ImageDraw.Draw(img)
        pad = 18
        y = y0 + pad

        # state + gesture
        st_col = STATE_COLOR.get(rep.state, TXT)
        d.text((x0 + pad, y), "STATE", font=self.f_lbls, fill=TXT_FAINT)
        d.text((x0 + pad, y + 18), rep.state, font=self.f_big, fill=st_col)
        d.text((x1 - pad - d.textlength(f"pipeline {rep.fps:4.0f} fps", font=self.f_mono_s), y + 8),
               f"pipeline {rep.fps:4.0f} fps", font=self.f_mono_s, fill=TXT_FAINT)
        y += 56

        g = rep.gesture
        d.text((x0 + pad, y), "GESTURE", font=self.f_lbls, fill=TXT_FAINT)
        d.text((x0 + pad, y + 17), g.gesture, font=self.f_h, fill=TXT)
        bar(img, (x0 + pad, y + 42, x1 - pad, y + 48), g.confidence, ACCENT, track=(30, 36, 46))
        d.text((x1 - pad - d.textlength(f"{g.confidence:.2f}", font=self.f_mono_s), y + 17), f"{g.confidence:.2f}", font=self.f_mono_s, fill=TXT_DIM)
        y += 74

        # intent field
        d.text((x0 + pad, y), "INTENT FIELD", font=self.f_lbls, fill=TXT_FAINT)
        d.text((x1 - pad - d.textlength(f"H {rep.intents.entropy:.2f}", font=self.f_mono_s), y), f"H {rep.intents.entropy:.2f}", font=self.f_mono_s, fill=TXT_FAINT)
        y += 20
        top = rep.intents.top_k(5)
        for intent, prob in top:
            committed = rep.intents.committed is intent
            col = ACCENT if committed else (ACCENT3 if prob > 0.12 else (86, 96, 112))
            d.text((x0 + pad, y), intent.value, font=self.f_mono_s, fill=TXT if committed else TXT_DIM)
            d.text((x0 + pad + 116, y), f"{prob:.2f}", font=self.f_mono_s, fill=col)
            bar(img, (x0 + pad + 152, y + 2, x1 - pad, y + 10), prob, col)
            y += 20
        y += 10

        # commit + bimanual
        bim = rep.notes.get("bimanual") or {}
        commit_kind = rep.commit.kind if rep.commit else "NONE"
        commit_score = rep.commit.score if rep.commit else 0.0
        d.text((x0 + pad, y), "COMMIT", font=self.f_lbls, fill=TXT_FAINT)
        d.text((x0 + pad + 66, y - 2), commit_kind, font=self.f_monob, fill=TXT if commit_kind != "NONE" else TXT_FAINT)
        d.text((x1 - pad - d.textlength(f"{commit_score:.2f}", font=self.f_mono_s), y - 2), f"{commit_score:.2f}", font=self.f_mono_s, fill=TXT_DIM)
        y += 22
        inside, best = self._target_readout(rep)
        d.text((x0 + pad, y), "TARGET", font=self.f_lbls, fill=TXT_FAINT)
        if inside is not None:
            d.text((x0 + pad + 66, y - 2), f"INSIDE {inside.target.id}", font=self.f_monob, fill=ACCENT3)
            d.text((x1 - pad - d.textlength(f"b {inside.confidence:.2f}", font=self.f_mono_s), y - 2),
                   f"b {inside.confidence:.2f}", font=self.f_mono_s, fill=TXT_DIM)
        elif best is not None:
            d.text((x0 + pad + 66, y - 2), f"NEAREST {best.target.id}", font=self.f_monob, fill=TXT_DIM)
            d.text((x1 - pad - d.textlength(f"d {best.distance:.2f}", font=self.f_mono_s), y - 2),
                   f"d {best.distance:.2f}", font=self.f_mono_s, fill=TXT_FAINT)
        else:
            d.text((x0 + pad + 66, y - 2), "NONE", font=self.f_monob, fill=TXT_FAINT)
        y += 22
        sec_action = str(bim.get("secondary_action", "NONE"))
        if sec_action == "NONE":
            # Zoom/selection/pan own the secondary channel but do not write
            # ``secondary_action``, so the readout would say NONE while the
            # engine is mid-gesture.
            if bim.get("zoom_active"):
                sec_action = "ZOOM"
            elif bim.get("selection_active"):
                sec_action = "SELECTION"
            elif bim.get("pan_active"):
                sec_action = "PAN"
        d.text((x0 + pad, y), "WHAT", font=self.f_lbls, fill=TXT_FAINT)
        d.text((x0 + pad + 66, y - 2), sec_action, font=self.f_monob, fill=ACCENT2 if sec_action != "NONE" else TXT_FAINT)
        gate = str(bim.get("gate_reason", ""))
        d.text((x0 + pad, y + 20), gate, font=self.f_mono_s, fill=TXT_FAINT)
        y += 48

        # desktop map
        map_h = 225
        mbox = (x0 + pad, y, x1 - pad, y + map_h)
        img = self._draw_desktop_map(img, rep, mbox)
        y = mbox[3] + 18

        # actions
        d = ImageDraw.Draw(img)
        d.text((x0 + pad, y), "ACTIONS", font=self.f_lbls, fill=TXT_FAINT)
        y += 18
        actions = rep.actions
        if not actions:
            d.text((x0 + pad, y), "—", font=self.f_mono_s, fill=TXT_FAINT)
        for a in actions[:4]:
            src = str(a.payload.get("source", ""))
            label = f"{a.kind:<12} {a.intent.value}"
            d.text((x0 + pad, y), label, font=self.f_mono_s, fill=ACCENT if src.startswith("bimanual") else TXT)
            y += 17
        y += 12

        # pipeline waterfall: per-stage cost of this exact frame
        stages = [(k, float(v)) for k, v in rep.stage_ms.items() if float(v) > 0.01]
        if stages:
            total = max(sum(v for _, v in stages), 1e-6)
            d.text((x0 + pad, y), "PIPELINE  (ms, this frame)", font=self.f_lbls, fill=TXT_FAINT)
            y += 18
            bar_x0 = x0 + pad + 86
            bar_x1 = x1 - pad - 40
            vmax = max(v for _, v in stages)
            for name, ms in stages[:10]:
                d.text((x0 + pad, y - 1), name, font=self.f_tiny, fill=TXT_FAINT)
                d.text((x1 - pad - d.textlength(f"{ms:.2f}", font=self.f_tiny), y - 1), f"{ms:.2f}", font=self.f_tiny, fill=TXT_DIM)
                w = int((bar_x1 - bar_x0) * (ms / vmax))
                d.rectangle([bar_x0, y + 1, bar_x0 + max(w, 1), y + 7], fill=(ACCENT3 if ms < vmax else ACCENT))
                y += 12
            d.text((x0 + pad, y + 2),
                   f"total {total:.2f} ms   budget {self.budget_ms:.1f} ms @30 fps   headroom {self.budget_ms / total:.1f}x",
                   font=self.f_tiny, fill=TXT_FAINT)
        return img
    def _draw_desktop_map(self, img, rep, box) -> Image.Image:
        x0, y0, x1, y1 = [int(v) for v in box]
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        d.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=(9, 12, 17, 245), outline=(44, 54, 70, 255), width=1)
        d.text((x0 + 12, y0 + 9), "DESKTOP / SPATIAL INTENT", font=self.f_lbls, fill=TXT_DIM)
        d.text((x0 + 12, y1 - 20), f"screen {int(rep.notes.get('screen', [1920, 1080])[0])}x{int(rep.notes.get('screen', [1920, 1080])[1])}",
               font=self.f_mono_s, fill=TXT_FAINT)
        img = Image.alpha_composite(img.convert("RGBA"), layer)

        ix0, iy0, ix1, iy1 = x0 + 10, y0 + 30, x1 - 10, y1 - 28
        iw, ih = ix1 - ix0, iy1 - iy0

        def mp(p):
            return (ix0 + float(np.clip(p[0], 0, 1)) * iw, iy0 + float(np.clip(p[1], 0, 1)) * ih)

        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for i in range(1, 8):
            d.line([(ix0 + iw * i / 8, iy0), (ix0 + iw * i / 8, iy1)], fill=(24, 30, 40, 255))
            d.line([(ix0, iy0 + ih * i / 8), (ix1, iy0 + ih * i / 8)], fill=(24, 30, 40, 255))
        d.rectangle([ix0, iy0, ix1, iy1], outline=(40, 50, 66, 255))

        locked = rep.notes.get("cursor_target")
        inside, best = self._target_readout(rep)
        focus = locked or (inside.target.id if inside is not None else None)
        for b in rep.target_beliefs:
            bx0, by0, bx1, by1 = b.target.bounds
            q0, q1 = mp((bx0, by0)), mp((bx1, by1))
            active = b.target.id == focus
            d.rounded_rectangle([q0, q1], radius=5,
                                fill=(ACCENT3[0], ACCENT3[1], ACCENT3[2], 55 if active else 22),
                                outline=(ACCENT3 if active else (66, 82, 116)) + (255,), width=2 if active else 1)
            # Only the focused target carries a label: six labels in a map this
            # size collide and the result reads as noise, not as targets.
            if active:
                d.text((q0[0] + 5, q0[1] + 3), b.target.id, font=self.f_mono_s, fill=(255, 255, 255, 240))
        img = Image.alpha_composite(img.convert("RGBA"), layer)

        if rep.cursor is not None:
            c = mp(rep.cursor)
            self._map_cursor = (float(c[0]), float(c[1]))
            img = glow_layer(img, lambda dd: dd.ellipse([c[0] - 12, c[1] - 12, c[0] + 12, c[1] + 12], outline=ACCENT + (255,), width=3), radius=8, strength=0.9)
            d = ImageDraw.Draw(img)
            d.line([(c[0] - 16, c[1]), (c[0] + 16, c[1])], fill=(255, 255, 255, 210), width=1)
            d.line([(c[0], c[1] - 16), (c[0], c[1] + 16)], fill=(255, 255, 255, 210), width=1)
            d.ellipse([c[0] - 3, c[1] - 3, c[0] + 3, c[1] + 3], fill=(255, 255, 255, 255))
            if inside is not None:
                d.text((x0 + 12, y0 + 30 + ih + 4),
                       f"INSIDE {inside.target.id}   belief {inside.confidence:.2f}", font=self.f_monob, fill=ACCENT)
            elif best is not None:
                d.text((x0 + 12, y0 + 30 + ih + 4),
                       f"NEAREST {best.target.id}  d={best.distance:.2f}  b={best.confidence:.2f}", font=self.f_monob, fill=TXT_DIM)
            else:
                d.text((x0 + 12, y0 + 30 + ih + 4), "FREE TRAVEL", font=self.f_monob, fill=TXT_DIM)
        return img

    # -- chrome ---------------------------------------------------------- #
    def _draw_header(self, img, rep, s: Sample, pad, header_h) -> Image.Image:
        d = ImageDraw.Draw(img)
        y = pad + 6
        d.text((pad, y), "VISION DESKTOP INTERACTION ENGINE", font=self.f_title, fill=TXT)
        d.text((pad + d.textlength("VISION DESKTOP INTERACTION ENGINE", font=self.f_title) + 14, y + 6), "v0.3.1", font=self.f_sub, fill=TXT_FAINT)

        x = self.w - pad
        for text, color in [("synthetic hand model", TXT_DIM), ("2 hands", ACCENT2)]:
            tw = d.textlength(text, font=self.f_mono_s)
            x -= tw + 22
            d.rounded_rectangle([x, y + 2, x + tw + 16, y + 24], radius=7, fill=(18, 22, 29, 235), outline=color + (120,), width=1)
            d.text((x + 8, y + 6), text, font=self.f_mono_s, fill=color)
        return img

    def _draw_footer(self, img, rep, s: Sample, x0, x1, y0) -> Image.Image:
        d = ImageDraw.Draw(img)
        d.line([(x0, y0), (x1, y0)], fill=(30, 36, 46, 255), width=1)
        notes = rep.notes
        lat = rep.latency_ms
        parts = [
            f"t {rep.timestamp:5.2f}s",
            f"frame {rep.frame_id}",
            f"latency {lat:5.2f} ms",
            f"hands {notes.get('hand_count', 0)}",
            f"primary {notes.get('primary_hand')}",
            f"index_tip {notes.get('index_tip_raw')[0]:.3f},{notes.get('index_tip_raw')[1]:.3f}",
            f"cursor {notes.get('cursor_pixel')[0]:.0f},{notes.get('cursor_pixel')[1]:.0f} px",
            f"mode {notes.get('cursor_mode')}",
        ]
        x = x0
        for i, p in enumerate(parts):
            d.text((x, y0 + 14), p, font=self.f_mono_s, fill=TXT_DIM if i < 4 else TXT_FAINT)
            x += d.textlength(p, font=self.f_mono_s) + 26
        d.text((x1 - d.textlength("synthetic hand model · real pipeline", font=self.f_mono_s), y0 + 14),
               "synthetic hand model · real pipeline", font=self.f_mono_s, fill=TXT_FAINT)
        return img


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

#: Named moments to capture: (auto-rule, caption).  A moment is picked from the
#: run by a *decision*, not by a timestamp — retiming the timeline must not
#: silently start capturing a frame in which nothing happens.
MOMENTS: dict[str, tuple[str, str]] = {
    "commit": ("commit_target_entry", "Predictive target acquisition: the cursor enters T1 and the kinematic commit point fires TARGET_ENTRY"),
    "click": ("click", "Short secondary pinch = left click, generated on the pinch edge, not on a dwell"),
    "drag": ("drag", "Secondary pinch held + primary movement = drag (the WHAT hand owns the button)"),
    "zoom": ("zoom", "Both hands pinching = Ctrl+wheel zoom, armed and deadbanded on the separation delta"),
}

DEFAULT_RULES: dict[str, object] = {}


def auto_pick(samples: list[Sample], rule: str) -> Sample:
    if not samples:
        raise SystemExit("no frames produced")
    if isinstance(rule, (int, float)):
        return min(samples, key=lambda s: abs(s.t - float(rule)))
    for s in samples:
        r = s.report
        if rule == "commit_target_entry" and r.commit and r.commit.kind == "TARGET_ENTRY":
            return s
        if rule == "click" and any(a.kind == "click" for a in r.actions):
            return s
        if rule == "drag" and str((r.notes.get("bimanual") or {}).get("secondary_action")) == "DRAG":
            return s
    if rule == "zoom":
        # Pick the frame with the largest separation change *that actually
        # emitted*: the pose is identical at every zoom frame, the delta is the
        # point of the shot, but a frame without the action would show
        # ``ACTIONS —`` while the panel claims a zoom is in progress.
        best_z: tuple[float, Sample] | None = None
        for s in samples:
            b = s.report.notes.get("bimanual") or {}
            if b.get("zoom_active") and any(a.kind == "zoom" for a in s.report.actions):
                d = abs(float(b.get("zoom_delta_ratio") or 0.0))
                if best_z is None or d > best_z[0]:
                    best_z = (d, s)
        if best_z is not None:
            return best_z[1]
    if rule == "lock":
        best: tuple[float, Sample] | None = None
        for s in samples:
            r = s.report
            if r.cursor is None:
                continue
            for b in r.target_beliefs:
                if b.target.contains(r.cursor) and (best is None or b.confidence > best[0]):
                    best = (b.confidence, s)
        if best is not None:
            return best[1]
    print(f"  ! no frame matched rule {rule!r}; falling back to the middle of the run")
    return samples[len(samples) // 2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("portfolio-preview", description="Render portfolio previews of the VDIE engine")
    ap.add_argument("--out", type=str, default="preview")
    ap.add_argument("--moment", type=str, default=None, help="render one moment by name")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--clip", action="store_true", help="also write an mp4 of the full timeline")
    ap.add_argument("--clip-width", type=int, default=0, help="clip width (default: --width)")
    ap.add_argument("--clip-height", type=int, default=0, help="clip height (default: --height)")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args(argv)

    if args.list:
        for name, (rule, cap) in MOMENTS.items():
            print(f"{name:<7} rule={rule:<20} {cap}")
        return 0

    samples = run_timeline(fps=args.fps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(args.width, args.height)

    names = [args.moment] if args.moment else list(MOMENTS)
    for name in names:
        rule, _ = MOMENTS[name]
        s = auto_pick(samples, rule)
        img = renderer.render(s).convert("RGB")
        png = out / f"vdie-{name}.png"
        img.save(png)
        img.save(out / f"vdie-{name}.jpg", quality=90, optimize=True)
        print(f"{name:<7} t={s.t:4.2f} state={s.report.state:<10} gesture={s.report.gesture.gesture:<10} "
              f"commit={(s.report.commit.kind if s.report.commit else '-'):<13} -> {png}")

    if args.clip:
        import cv2

        cw = args.clip_width or args.width
        ch = args.clip_height or args.height
        clip_renderer = renderer if (cw, ch) == (args.width, args.height) else Renderer(cw, ch)
        path = out / "vdie-timeline.mp4"
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (cw, ch))
        for s in samples:
            frame = np.asarray(clip_renderer.render(s).convert("RGB"))
            vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        vw.release()
        size_mb = path.stat().st_size / 1e6
        print(f"clip -> {path}  ({len(samples)} frames @ {args.fps} fps, {cw}x{ch}, {size_mb:.1f} MB)")
        print("     webm/H.264 re-encode needs a real ffmpeg in PATH "
              "(winget install --id Gyan.FFmpeg -e); the portfolio's tools/clip.js does the same.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
