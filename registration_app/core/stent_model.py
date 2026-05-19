"""Geometrie 3D du stent tresse (fils helicoidaux).

Cylindre simple (silhouette) est conserve dans core.stent_placement pour la
projection masque -> 2D ; cette geometrie tubulaire fil-a-fil sert au rendu
3D et a la projection inverse depuis la fluoro.
"""

import math
import numpy as np


def wire_centerlines(diameter_mm: float, length_mm: float,
                     n_wires: int = 16, braid_angle_deg: float = 45.0,
                     n_pts: int = 120):
    """Retourne {'plus': [pts(n_pts,3)...], 'minus': [...]} des centerlines des fils."""
    R = diameter_mm / 2.0
    pitch = math.pi * diameter_mm / math.tan(math.radians(braid_angle_deg))
    n_half = max(1, n_wires // 2)
    t = np.linspace(0, 1, int(n_pts))
    z = (t - 0.5) * length_mm

    def helix(i, d):
        phase = 2 * math.pi * i / n_half
        theta = d * 2 * math.pi * (length_mm / pitch) * t + phase
        return np.column_stack([R * np.cos(theta), R * np.sin(theta), z])

    return {s: [helix(i, d) for i in range(n_half)]
            for s, d in (("plus", 1), ("minus", -1))}


def _tube_mesh(pts: np.ndarray, radius: float, sides: int = 8):
    """Mesh tubulaire autour d'une courbe 3D. Retourne (verts (N*sides, 3), faces (M, 3))."""
    N = len(pts)
    ang = np.linspace(0, 2 * math.pi, sides, endpoint=False)

    T = np.diff(pts, axis=0, append=pts[[-2]])
    T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-12)

    arb = np.array([0., 0., 1.]) if abs(T[0, 2]) < .9 else np.array([1., 0., 0.])
    N0 = np.cross(T[0], arb); N0 /= np.linalg.norm(N0)
    Ns = [N0]
    for i in range(1, N):
        b = np.cross(T[i - 1], T[i]); bn = np.linalg.norm(b)
        if bn < 1e-10:
            Ns.append(Ns[-1])
        else:
            b /= bn
            th = math.acos(np.clip(T[i - 1] @ T[i], -1, 1))
            c, s = math.cos(th), math.sin(th)
            x, y, z = b
            R_ = np.array([[c + x*x*(1-c), x*y*(1-c) - z*s, x*z*(1-c) + y*s],
                           [y*x*(1-c) + z*s, c + y*y*(1-c), y*z*(1-c) - x*s],
                           [z*x*(1-c) - y*s, z*y*(1-c) + x*s, c + z*z*(1-c)]])
            Ns.append(R_ @ Ns[-1])
    Ns = np.array(Ns)
    Bs = np.cross(T, Ns)

    verts = (pts[:, None]
             + radius * np.cos(ang)[None, :, None] * Ns[:, None]
             + radius * np.sin(ang)[None, :, None] * Bs[:, None]).reshape(-1, 3)

    i = np.arange(N - 1); j = np.arange(sides)
    ii, jj = np.meshgrid(i, j, indexing="ij")
    a = (ii * sides + jj).ravel()
    b = (ii * sides + (jj + 1) % sides).ravel()
    c = ((ii + 1) * sides + jj).ravel()
    d = ((ii + 1) * sides + (jj + 1) % sides).ravel()
    return verts, np.concatenate([np.c_[a, b, d], np.c_[a, d, c]])


def braided_stent_mesh(diameter_mm: float, length_mm: float,
                       n_wires: int = 24, braid_angle_deg: float = 45.0,
                       wire_radius_mm: float = 0.12, n_pts: int = 90,
                       tube_sides: int = 5):
    """Genere le mesh complet (vertices, faces) d'un stent tresse 3D.

    Le mesh est centre a l'origine, axe le long de +Z.
    """
    wires = wire_centerlines(diameter_mm, length_mm, n_wires, braid_angle_deg, n_pts)
    vl, fl, offset = [], [], 0
    for pts in wires["plus"] + wires["minus"]:
        v, f = _tube_mesh(pts, wire_radius_mm, tube_sides)
        vl.append(v); fl.append(f + offset); offset += len(v)
    return np.concatenate(vl), np.concatenate(fl)


def _rotation_align_z_to(axis_unit: np.ndarray) -> np.ndarray:
    """Rotation 3x3 alignant +Z avec axis_unit (formule de Rodrigues)."""
    z = np.array([0.0, 0.0, 1.0])
    a = np.asarray(axis_unit, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return np.eye(3)
    a = a / n
    d = float(np.dot(z, a))
    if d > 0.9999:
        return np.eye(3)
    if d < -0.9999:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, a)
    s = float(np.linalg.norm(v))
    vx = np.array([[0.0, -v[2], v[1]],
                   [v[2], 0.0, -v[0]],
                   [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + (vx @ vx) * ((1.0 - d) / (s * s))


def braided_stent_world(diameter_mm: float, length_mm: float,
                        center_world: np.ndarray, axis_world: np.ndarray,
                        **kw):
    """Retourne (verts_world, faces) du stent tresse positionne dans le repere CT-monde."""
    verts, faces = braided_stent_mesh(diameter_mm, length_mm, **kw)
    R = _rotation_align_z_to(np.asarray(axis_world, dtype=np.float64))
    verts_w = (R @ verts.T).T + np.asarray(center_world, dtype=np.float64)
    return verts_w, faces
