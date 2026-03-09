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


# ──────────────────────────────────────────────────────────────────────────────
# Elastic / deformable registration (Free-Form Deformation on coarse grid)
# ──────────────────────────────────────────────────────────────────────────────

def _ctrl_to_dense(ctrl_dx_flat: np.ndarray, ctrl_dy_flat: np.ndarray,
                   ng: int, h: int, w: int):
    """Resize a coarse control-point grid to a full-image displacement field."""
    dx = cv2.resize(ctrl_dx_flat.reshape(ng, ng).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(ctrl_dy_flat.reshape(ng, ng).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR)
    return dx, dy


def _apply_ctrl_pts(img_f32: np.ndarray, params: np.ndarray,
                    ng: int, h: int, w: int) -> np.ndarray:
    """Apply a coarse FFD grid to a float32 image and threshold to binary."""
    n = ng * ng
    dx, dy = _ctrl_to_dense(params[:n], params[n:], ng, h, w)
    x, y = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    out = cv2.remap(img_f32.astype(np.float32), x + dx, y + dy,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return (out > 0.5).astype(np.float32)


def apply_full_transform(img_f32: np.ndarray, result: dict) -> np.ndarray:
    """
    Apply the complete transform stored in a registration result dict:
    rigid similarity (tx, ty, rotation, scale) then, if elastic keys are
    present, the stored dense displacement field.

    Returns a float32 image — caller is responsible for thresholding if needed.
    """
    tx = result['tx']; ty = result['ty']
    angle = result['angle']; scale = result.get('scale', 1.0)
    cx, cy = result['center']
    h, w = img_f32.shape[:2]

    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    M[0, 2] += tx; M[1, 2] += ty
    warped = cv2.warpAffine(img_f32.astype(np.float32), M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if result.get('elastic') and 'disp_x' in result:
        dx = result['disp_x'].astype(np.float32)
        dy = result['disp_y'].astype(np.float32)
        if dx.shape != (h, w):
            dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_LINEAR)
            dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_LINEAR)
        x, y = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        warped = cv2.remap(warped, x + dx, y + dy,
                           cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped


def register_elastic(
    mask_moving: np.ndarray,
    mask_fixed: np.ndarray,
    grid_size: int = 4,
    max_disp_frac: float = 0.20,
    smooth_weight: float = 0.05,
    progress_cb: Optional[Callable[[float, float], None]] = None,
) -> dict:
    """
    Two-stage elastic registration:
      1. Rigid similarity (tx, ty, rot, scale) via Differential Evolution + Nelder-Mead.
      2. Free-Form Deformation (FFD) on a ``grid_size x grid_size`` control-point
         grid optimised with Powell to maximise IoU on top of the rigid result.

    Returns the same dict as ``register`` extended with:
        elastic  : True
        disp_x   : dense X-displacement map (H, W) float32
        disp_y   : dense Y-displacement map (H, W) float32
    """
    h, w = mask_fixed.shape[:2]

    # ── Stage 1: rigid ────────────────────────────────────────────────────────
    cb_rigid = (lambda f, s: progress_cb(f * 0.5, s)) if progress_cb else None
    rigid = register(mask_moving, mask_fixed, progress_cb=cb_rigid)
    rigid_mask = rigid['mask_registered'].astype(np.float32)

    # ── Stage 2: elastic FFD ──────────────────────────────────────────────────
    ng = grid_size
    n_pts = ng * ng
    n_params = 2 * n_pts
    max_disp = max(h, w) * max_disp_frac
    call_count = [0]

    def objective(params):
        moved = _apply_ctrl_pts(rigid_mask, params, ng, h, w)
        score = iou_score(moved, mask_fixed)
        # Smoothness + magnitude regularisation
        dx = params[:n_pts].reshape(ng, ng)
        dy = params[n_pts:].reshape(ng, ng)
        reg = smooth_weight * (
            float(np.sum(dx ** 2) + np.sum(dy ** 2)) +
            float(np.sum(np.diff(dx, axis=0) ** 2) + np.sum(np.diff(dx, axis=1) ** 2) +
                  np.sum(np.diff(dy, axis=0) ** 2) + np.sum(np.diff(dy, axis=1) ** 2))
        ) / (max_disp ** 2 + 1e-8)
        call_count[0] += 1
        if progress_cb and call_count[0] % 30 == 0:
            frac = min(0.5 + call_count[0] / 5000 * 0.5, 0.98)
            progress_cb(frac, score)
        return -score + reg

    res = minimize(objective, x0=np.zeros(n_params), method='Powell',
                   options={'maxiter': 5000, 'maxfev': 60000,
                            'ftol': 1e-6, 'disp': False})

    best = res.x
    dx_dense, dy_dense = _ctrl_to_dense(best[:n_pts], best[n_pts:], ng, h, w)
    elastic_mask = _apply_ctrl_pts(rigid_mask, best, ng, h, w)

    if progress_cb:
        progress_cb(1.0, iou_score(elastic_mask, mask_fixed))

    return {
        **rigid,
        'elastic': True,
        'iou':  iou_score(elastic_mask, mask_fixed),
        'dice': dice_score(elastic_mask, mask_fixed),
        'disp_x': dx_dense,
        'disp_y': dy_dense,
        'mask_registered': elastic_mask,
    }
