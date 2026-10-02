"""Pure geometry helpers.  No state, no I/O, fully unit-testable."""

from __future__ import annotations

import numpy as np

EPS = 1e-9


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance in the xy-plane (V1 definition)."""
    d = np.asarray(a, dtype=np.float64)[:2] - np.asarray(b, dtype=np.float64)[:2]
    return float(np.sqrt(d[0] * d[0] + d[1] * d[1]))


def distance3(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]
    return float(np.linalg.norm(d))


def distance_matrix(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)[:, :2]
    diff = p[:, None, :] - p[None, :, :]
    return np.sqrt((diff * diff).sum(axis=-1))


def safe_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros_like(v)
    return v / n


def unit_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit vector from ``a`` to ``b`` in 2-D."""
    return safe_normalize(np.asarray(b, dtype=np.float64)[:2] - np.asarray(a, dtype=np.float64)[:2])


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex ``b`` for the chain ``a -> b -> c``, in radians.

    Uses the 2-D projection; MediaPipe's z is too noisy for reliable 3-D angles.
    """
    ba = np.asarray(a, dtype=np.float64)[:2] - np.asarray(b, dtype=np.float64)[:2]
    bc = np.asarray(c, dtype=np.float64)[:2] - np.asarray(b, dtype=np.float64)[:2]
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba < EPS or nbc < EPS:
        return 0.0
    cos_t = float(np.dot(ba, bc) / (nba * nbc))
    return float(np.arccos(np.clip(cos_t, -1.0, 1.0)))


def vector_angle(v: np.ndarray) -> float:
    """Angle of a vector in radians, ``atan2(y, x)``."""
    v = np.asarray(v, dtype=np.float64)
    return float(np.arctan2(v[1], v[0]))


def angle_difference(a: float, b: float) -> float:
    """Smallest signed difference ``a - b`` wrapped to ``(-pi, pi]``."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return float(d)


def orientation_from_vertical(v: np.ndarray) -> float:
    """Tilt of a vector from the *up* direction, in degrees.

    ``0`` = pointing up, ``+90`` = right, ``-90`` = left, ``+-180`` = down.
    This is exactly the master-prompt formula ``atan2(Vx, -Vy)``.
    """
    v = np.asarray(v, dtype=np.float64)
    return float(np.degrees(np.arctan2(v[0], -v[1])))


def project_onto(v: np.ndarray, axis: np.ndarray) -> float:
    u = safe_normalize(axis)
    return float(np.dot(np.asarray(v, dtype=np.float64)[:2], u))


def perpendicular_distance(point: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    """Signed distance of ``point`` from the line through ``origin`` along ``axis``."""
    u = safe_normalize(axis)
    rel = np.asarray(point, dtype=np.float64)[:2] - np.asarray(origin, dtype=np.float64)[:2]
    return float(rel[0] * (-u[1]) + rel[1] * u[0])


def polyline_length(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float64)[:, :2]
    if p.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def straightness(points: np.ndarray) -> float:
    """``|end - start| / path_length``; 1.0 = perfectly straight."""
    p = np.asarray(points, dtype=np.float64)[:, :2]
    if p.shape[0] < 2:
        return 1.0
    path = polyline_length(p)
    if path < EPS:
        return 1.0
    return float(np.linalg.norm(p[-1] - p[0]) / path)


def curvature_radius(points: np.ndarray) -> float:
    """Signed radius of curvature from three last points; ``inf`` when collinear."""
    p = np.asarray(points, dtype=np.float64)[:, :2]
    if p.shape[0] < 3:
        return float("inf")
    a, b, c = p[-3], p[-2], p[-1]
    ab = np.linalg.norm(b - a)
    bc = np.linalg.norm(c - b)
    ca = np.linalg.norm(a - c)
    if min(ab, bc, ca) < EPS:
        return float("inf")
    cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(cross) < EPS:
        return float("inf")
    return float((ab * bc * ca) / (2.0 * cross))


def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate_2d(points: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate an (N, 2+) array about the origin; extra columns are preserved."""
    p = np.asarray(points, dtype=np.float64).copy()
    r = rotation_matrix_2d(angle_rad)
    p[:, :2] = p[:, :2] @ r.T
    return p


def bounding_box(points: np.ndarray) -> tuple[float, float, float, float]:
    p = np.asarray(points, dtype=np.float64)[:, :2]
    return (
        float(p[:, 0].min()),
        float(p[:, 1].min()),
        float(p[:, 0].max()),
        float(p[:, 1].max()),
    )


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def lerp(a: np.ndarray | float, b: np.ndarray | float, t: float) -> np.ndarray | float:
    return a + (b - a) * t


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite interpolation; returns 0 below ``edge0``, 1 above ``edge1``."""
    if abs(edge1 - edge0) < EPS:
        return 1.0 if x >= edge1 else 0.0
    t = float(np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)
