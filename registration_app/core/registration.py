"""
core/registration.py
Recalage 2D/3D basé sur IoU de masques — 4 DOF (tx, ty, θ, scale).
Optimisation : centroïde init → Differential Evolution → Nelder-Mead.
"""

import numpy as np
import cv2
from scipy.optimize import differential_evolution, minimize
from typing import Callable, Optional


def apply_transform(mask_2d: np.ndarray, tx: float, ty: float,
                    angle_deg: float, center=None,
                    scale: float = 1.0) -> np.ndarray:
    """
    Transformation similitude 2D : mise à l'échelle + rotation + translation.
    center : centre de rotation/zoom (cx, cy) en pixels.
    scale  : facteur d'échelle (1.0 = identité).
    """
    h, w = mask_2d.shape
    if center is None:
        center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    out = cv2.warpAffine(
        mask_2d.astype(np.float32), M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return (out > 0.5).astype(np.float32)


def iou_score(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a * b).sum())
    union = float(np.clip(a + b, 0, 1).sum())
    return inter / (union + 1e-8)


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a * b).sum())
    return 2.0 * inter / (float(a.sum()) + float(b.sum()) + 1e-8)


def centroid(mask: np.ndarray):
    """Retourne (cx, cy) du masque binaire."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def register(
    mask_moving: np.ndarray,
    mask_fixed: np.ndarray,
    search_tx_px: float = None,
    search_ty_px: float = None,
    search_rot_deg: float = 30.0,
    search_scale_range: float = 0.4,
    progress_cb: Optional[Callable[[float, float], None]] = None,
) -> dict:
    """
    Recale mask_moving sur mask_fixed par maximisation IoU.
    Les bornes de recherche sont calculées automatiquement si None.

    Returns:
        dict avec clés : tx, ty, angle, scale, iou, dice,
                         iou_init, iou_centered, history, mask_registered
    """
    h, w = mask_fixed.shape[:2]
    if search_tx_px is None:
        search_tx_px = w * 0.5
    if search_ty_px is None:
        search_ty_px = h * 0.5
    cx_mov, cy_mov = centroid(mask_moving)
    cx_fix, cy_fix = centroid(mask_fixed)

    # ── Init par centroïde ────────────────────────────────────────────────────
    tx_init = cx_fix - cx_mov
    ty_init = cy_fix - cy_mov
    proj_centered = apply_transform(mask_moving, tx_init, ty_init, 0.0,
                                    center=(cx_mov, cy_mov))
    iou_init     = iou_score(mask_moving, mask_fixed)
    iou_centered = iou_score(proj_centered, mask_fixed)

    # ── Objectif ──────────────────────────────────────────────────────────────
    iou_history = []
    call_count = [0]
    max_calls_est = 300 * 15   # maxiter × popsize

    def objective(params):
        tx, ty, angle, scale = params
        moved = apply_transform(mask_moving, tx, ty, angle, (cx_mov, cy_mov), scale)
        score = iou_score(moved, mask_fixed)
        iou_history.append(score)
        call_count[0] += 1
        if progress_cb is not None and call_count[0] % 50 == 0:
            frac = min(call_count[0] / max_calls_est, 0.95)
            progress_cb(frac, score)
        return -score

    # ── Stage 1 : Differential Evolution ─────────────────────────────────────
    bounds = [
        (tx_init - search_tx_px,  tx_init + search_tx_px),
        (ty_init - search_ty_px,  ty_init + search_ty_px),
        (-search_rot_deg,          search_rot_deg),
        (max(0.1, 1.0 - search_scale_range), 1.0 + search_scale_range),
    ]
    res_de = differential_evolution(
        objective, bounds=bounds,
        maxiter=300, popsize=15, seed=42,
        tol=1e-5, mutation=(0.5, 1.0), recombination=0.7,
        workers=1, disp=False,
    )

    # ── Stage 2 : Nelder-Mead raffinement ────────────────────────────────────
    res_nm = minimize(
        objective, x0=res_de.x, method='Nelder-Mead',
        options={'xatol': 0.05, 'fatol': 1e-6, 'maxiter': 3000, 'disp': False}
    )

    tx_f, ty_f, rot_f, scale_f = res_nm.x
    mask_reg = apply_transform(mask_moving, tx_f, ty_f, rot_f, (cx_mov, cy_mov), scale_f)

    if progress_cb is not None:
        progress_cb(1.0, iou_score(mask_reg, mask_fixed))

    return {
        'tx': tx_f,
        'ty': ty_f,
        'angle': rot_f,
        'scale': scale_f,
        'iou': iou_score(mask_reg, mask_fixed),
        'dice': dice_score(mask_reg, mask_fixed),
        'iou_init': iou_init,
        'iou_centered': iou_centered,
        'history': iou_history,
        'mask_registered': mask_reg,
        'center': (cx_mov, cy_mov),
    }
