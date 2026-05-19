"""Stent generation and 2D projection utilities."""

import math
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import trimesh
except ImportError:  # pragma: no cover - handled at runtime
    trimesh = None


def _require_trimesh():
    if trimesh is None:
        raise ImportError('trimesh is required for stent generation')


def generate_stent_mesh(diameter_mm: float, length_mm: float, radial_segments: int = 48) -> 'trimesh.Trimesh':
    """Generate a simple cylindrical stent mesh centered at the origin."""
    _require_trimesh()
    radius = float(diameter_mm) * 0.5
    height = float(length_mm)
    segments = max(12, int(radial_segments))
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=segments)
    mesh.apply_translation([0.0, 0.0, 0.0])
    return mesh


def _rotation_matrix(lao_deg: float, cran_deg: float) -> np.ndarray:
    """Return rotation matrix matching DRR convention (LAO around Z, CRAN around X)."""
    lao = math.radians(float(lao_deg))
    cran = math.radians(float(cran_deg))
    cz, sz = math.cos(lao), math.sin(lao)
    cx, sx = math.cos(cran), math.sin(cran)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    return rx @ rz


def _pixel_size_mm(fov_mm: Optional[float], sid_mm: float, sod_mm: float, size: int) -> float:
    fov = float(fov_mm) if fov_mm and float(fov_mm) > 0 else 220.0
    sid = float(sid_mm) if sid_mm and float(sid_mm) > 0 else 1020.0
    sod = float(sod_mm) if sod_mm and float(sod_mm) > 0 else 510.0
    return (fov * (sid / sod)) / float(size)


def project_stent_mask(
    mesh: 'trimesh.Trimesh',
    lao_deg: float,
    cran_deg: float,
    size: int,
    sid_mm: float,
    sod_mm: float,
    fov_mm: Optional[float] = None,
) -> np.ndarray:
    """Project a stent mesh to a 2D mask using a simple cone-beam model."""
    if mesh is None:
        raise ValueError('mesh is None')

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    verts = (_rotation_matrix(lao_deg, cran_deg) @ verts.T).T

    src = np.array([0.0, -float(sod_mm), 0.0], dtype=np.float32)
    det_y = float(sid_mm) - float(sod_mm)
    pix_mm = _pixel_size_mm(fov_mm, sid_mm, sod_mm, int(size))
    if pix_mm <= 0:
        raise ValueError('pixel size is invalid')

    dy = verts[:, 1] - src[1]
    valid = np.abs(dy) > 1e-6
    t = np.zeros_like(dy)
    t[valid] = (det_y - src[1]) / dy[valid]
    valid &= t > 0

    proj = src + (verts - src) * t[:, None]
    u = proj[:, 0]
    v = proj[:, 2]

    px = (float(size) * 0.5) + (u / pix_mm)
    py = (float(size) * 0.5) - (v / pix_mm)

    coords = np.column_stack([px, py]).astype(np.int32)
    mask = np.zeros((int(size), int(size)), dtype=np.uint8)

    faces = np.asarray(mesh.faces, dtype=np.int64)
    for tri in faces:
        if not (valid[tri[0]] and valid[tri[1]] and valid[tri[2]]):
            continue
        pts = coords[tri]
        if np.any(pts[:, 0] < -5) or np.any(pts[:, 0] > size + 5):
            continue
        if np.any(pts[:, 1] < -5) or np.any(pts[:, 1] > size + 5):
            continue
        cv2.fillConvexPoly(mask, pts, 255)

    return (mask > 0).astype(np.float32)


def mask_center(mask: np.ndarray) -> Tuple[float, float]:
    """Return (x, y) center of mass for a binary mask."""
    if mask is None:
        return 0.0, 0.0
    ys, xs = np.where(mask > 0.5)
    if xs.size == 0:
        return float(mask.shape[1]) * 0.5, float(mask.shape[0]) * 0.5
    return float(xs.mean()), float(ys.mean())


def transform_mask(
    mask: np.ndarray,
    center_xy: Tuple[float, float],
    angle_deg: float,
    base_center: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Rotate and translate a mask around its base center."""
    if mask is None:
        return None
    h, w = mask.shape[:2]
    if base_center is None:
        base_center = mask_center(mask)
    dx = float(center_xy[0]) - float(base_center[0])
    dy = float(center_xy[1]) - float(base_center[1])

    rot = cv2.getRotationMatrix2D((base_center[0], base_center[1]), float(angle_deg), 1.0)
    rot[0, 2] += dx
    rot[1, 2] += dy

    out = cv2.warpAffine(mask.astype(np.float32), rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return (out > 0.5).astype(np.float32)
