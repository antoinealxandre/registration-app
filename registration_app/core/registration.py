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


def ncc_score(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Cross-Correlation on float images [0,1]."""
    a_norm = a.astype(np.float32)
    b_norm = b.astype(np.float32)
    a_mean = a_norm.mean()
    b_mean = b_norm.mean()
    a_c = a_norm - a_mean
    b_c = b_norm - b_mean
    denom = (np.std(a_c) * np.std(b_c) * float(a_norm.size)) + 1e-8
    return float(np.sum(a_c * b_c) / denom)


def gradient_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Gradient correlation - robust to cross-modality intensity differences."""
    a_u8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    b_u8 = (np.clip(b, 0, 1) * 255).astype(np.uint8)
    ga_x = cv2.Sobel(a_u8, cv2.CV_64F, 1, 0)
    ga_y = cv2.Sobel(a_u8, cv2.CV_64F, 0, 1)
    gb_x = cv2.Sobel(b_u8, cv2.CV_64F, 1, 0)
    gb_y = cv2.Sobel(b_u8, cv2.CV_64F, 0, 1)
    ga = ga_x + ga_y
    gb = gb_x + gb_y
    return ncc_score(ga, gb)


def centroid(mask: np.ndarray):
    """Retourne (cx, cy) du masque binaire."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def _tight_bounds(x_best: np.ndarray, tx0: float, ty0: float, factor: float = 0.25):
    """Tighten search bounds around current best solution for next pyramid level."""
    tx_b, ty_b = x_best[0], x_best[1]
    angle_b, scale_b = x_best[2], x_best[3]

    delta_tx = abs(tx_b - tx0) + 1.0
    delta_ty = abs(ty_b - ty0) + 1.0
    delta_angle = 3.0
    delta_scale = 0.05

    return [
        (tx_b - delta_tx * factor, tx_b + delta_tx * factor),
        (ty_b - delta_ty * factor, ty_b + delta_ty * factor),
        (angle_b - delta_angle, angle_b + delta_angle),
        (max(0.1, scale_b - delta_scale), scale_b + delta_scale),
    ]


def register(
    mask_moving: np.ndarray,
    mask_fixed: np.ndarray,
    search_tx_px: float = None,
    search_ty_px: float = None,
    search_rot_deg: float = 30.0,
    search_scale_range: float = 0.4,
    progress_cb: Optional[Callable[[float, float], None]] = None,
    drr_float: Optional[np.ndarray] = None,
    fluoro_float: Optional[np.ndarray] = None,
    metric: str = 'iou',
) -> dict:
    """
    Recale mask_moving sur mask_fixed par maximisation métrique.
    Les bornes de recherche sont calculées automatiquement si None.

    Args:
        drr_float: Optional DRR image [0,1] for intensity-based metrics
        fluoro_float: Optional fluoroscopy image [0,1] for intensity-based metrics
        metric: 'iou' | 'combined' (60% IoU + 30% NCC + 10% gradient correlation)

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
        iou = iou_score(moved, mask_fixed)

        if metric == 'combined' and drr_float is not None and fluoro_float is not None:
            # Apply same transform to DRR for intensity-based metrics
            h, w = drr_float.shape[:2]
            M = cv2.getRotationMatrix2D((cx_mov, cy_mov), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            drr_warped = cv2.warpAffine(
                drr_float.astype(np.float32), M, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            ncc = ncc_score(drr_warped, fluoro_float)
            grad_corr = gradient_correlation(drr_warped, fluoro_float)
            score = 0.6 * iou + 0.3 * ncc + 0.1 * grad_corr
        else:
            score = iou

        iou_history.append(iou)
        call_count[0] += 1
        if progress_cb is not None and call_count[0] % 50 == 0:
            frac = min(call_count[0] / max_calls_est, 0.95)
            progress_cb(frac, iou)
        return -score

    # ── Multi-resolution pyramid (64 → 128 → 512 px) ────────────────────────────
    # Cascade resolutions with tightening bounds
    pyramid = [(64, 150, 12), (128, 100, 8), (512, 50, 6)]
    tx_best, ty_best, rot_best, scale_best = tx_init, ty_init, 0.0, 1.0
    bounds_current = [
        (tx_init - search_tx_px, tx_init + search_tx_px),
        (ty_init - search_ty_px, ty_init + search_ty_px),
        (-search_rot_deg, search_rot_deg),
        (max(0.1, 1.0 - search_scale_range), 1.0 + search_scale_range),
    ]

    cv2.setNumThreads(2)  # Avoid OpenCV/joblib contention with workers=4

    for res, maxiter, popsize in pyramid:
        # Resize masks to current resolution
        scale_factor = res / 512.0
        mf_r = cv2.resize(mask_fixed, (res, res), interpolation=cv2.INTER_NEAREST)
        mm_r = cv2.resize(mask_moving, (res, res), interpolation=cv2.INTER_NEAREST)

        # Rescale images if provided
        drr_r = drr_float
        fluoro_r = fluoro_float
        if drr_float is not None:
            drr_r = cv2.resize(drr_float, (res, res), interpolation=cv2.INTER_LINEAR)
        if fluoro_float is not None:
            fluoro_r = cv2.resize(fluoro_float, (res, res), interpolation=cv2.INTER_LINEAR)

        # Update centroid for scaled resolution
        cx_mov_r, cy_mov_r = centroid(mm_r)

        # Rescale bounds for current resolution
        bounds_r = [
            (bounds_current[0][0] * scale_factor, bounds_current[0][1] * scale_factor),
            (bounds_current[1][0] * scale_factor, bounds_current[1][1] * scale_factor),
            bounds_current[2],  # Angle invariant
            bounds_current[3],  # Scale invariant
        ]

        def objective_r(params):
            tx, ty, angle, scale = params
            moved = apply_transform(mm_r, tx, ty, angle, (cx_mov_r, cy_mov_r), scale)
            iou = iou_score(moved, mf_r)

            if metric == 'combined' and drr_r is not None and fluoro_r is not None:
                h, w = drr_r.shape[:2]
                M = cv2.getRotationMatrix2D((cx_mov_r, cy_mov_r), angle, scale)
                M[0, 2] += tx
                M[1, 2] += ty
                drr_warped = cv2.warpAffine(
                    drr_r.astype(np.float32), M, (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0
                )
                ncc = ncc_score(drr_warped, fluoro_r)
                grad_corr = gradient_correlation(drr_warped, fluoro_r)
                score = 0.6 * iou + 0.3 * ncc + 0.1 * grad_corr
            else:
                score = iou

            iou_history.append(iou)
            call_count[0] += 1
            if progress_cb is not None and call_count[0] % 50 == 0:
                frac = min(call_count[0] / (sum(m*p for _, m, p in pyramid) * 15), 0.95)
                progress_cb(frac, iou)
            return -score

        # Differential Evolution at this resolution
        res_de = differential_evolution(
            objective_r, bounds=bounds_r,
            maxiter=maxiter, popsize=popsize, seed=42,
            tol=1e-5, mutation=(0.5, 1.0), recombination=0.7,
            workers=4, disp=False,
        )

        # Extract best solution and rescale back to 512px coordinates
        tx_best = res_de.x[0] / scale_factor if res < 512 else res_de.x[0]
        ty_best = res_de.x[1] / scale_factor if res < 512 else res_de.x[1]
        rot_best = res_de.x[2]
        scale_best = res_de.x[3]

        # Tighten bounds for next level
        if res < 512:
            bounds_current = _tight_bounds(
                np.array([tx_best, ty_best, rot_best, scale_best]),
                tx_init, ty_init, factor=0.25
            )

    # ── Final Nelder-Mead refinement at 512px ───────────────────────────────────
    res_nm = minimize(
        objective, x0=np.array([tx_best, ty_best, rot_best, scale_best]),
        method='Nelder-Mead',
        options={'xatol': 0.5, 'fatol': 1e-6, 'maxiter': 3000, 'disp': False}
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
    grid_size: int = 8,
    max_disp_frac: float = 0.05,
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
        # Smoothness + bending energy regularisation
        dx = params[:n_pts].reshape(ng, ng)
        dy = params[n_pts:].reshape(ng, ng)
        # L2 norm of displacements
        l2 = float(np.sum(dx ** 2) + np.sum(dy ** 2))
        # First-order derivatives (gradient smoothness)
        grad = float(np.sum(np.diff(dx, axis=0) ** 2) + np.sum(np.diff(dx, axis=1) ** 2) +
                     np.sum(np.diff(dy, axis=0) ** 2) + np.sum(np.diff(dy, axis=1) ** 2))
        # Second-order derivatives (bending energy - curvature)
        ddx = np.diff(dx, n=2, axis=0) if ng > 2 else np.array([[0.0]])
        ddy = np.diff(dx, n=2, axis=1) if ng > 2 else np.array([[0.0]])
        ddx_y = np.diff(dy, n=2, axis=0) if ng > 2 else np.array([[0.0]])
        ddy_y = np.diff(dy, n=2, axis=1) if ng > 2 else np.array([[0.0]])
        bending = float(np.sum(ddx ** 2) + np.sum(ddy ** 2) +
                       np.sum(ddx_y ** 2) + np.sum(ddy_y ** 2))
        reg = smooth_weight * (l2 + grad + 0.1 * bending) / (max_disp ** 2 + 1e-8)
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
