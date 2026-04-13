"""
core/refinement.py
Stage 2 : Recalage IoU de la silhouette du coeur.

À chaque évaluation : projette la segmentation 3D du coeur aux angles (LAO, CRAN)
candidats, applique la transformation (tx, ty, angle_stage1, scale_stage1), compare
par IoU avec l'annotation manuelle sur la fluoroscopie.

Stratégie : grille 5×5 sur (LAO, CRAN) → Nelder-Mead sur (LAO, CRAN, tx, ty).
Durée estimée : ~1–2 s pour (25 + ~150) évaluations × ~10 ms chacune.
"""

import numpy as np
import cv2
from scipy.optimize import minimize
from typing import Callable, Optional

from core.drr_generator import project_mask_3d
from core.registration import apply_transform, iou_score


def refine_heart(
    heart_mask_3d: np.ndarray,
    fluoro_heart_mask: np.ndarray,
    ct_aff: Optional[np.ndarray],
    lao_deg: float,
    cran_deg: float,
    table_angle: float,
    output_size: int,
    stage1_result: dict,
    search_range_deg: float = 10.0,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """
    Recale la silhouette du coeur par maximisation d'IoU.

    Returns dict : best_lao, best_cran, tx, ty, angle, scale, center,
                   initial_iou, best_iou, heart_proj_before, heart_proj_after,
                   n_evaluations.
    """
    # ── Préparer le masque cible (annotation fluoro) ───────────────────────
    target = (fluoro_heart_mask > 0.5).astype(np.float32)
    if target.shape[:2] != (output_size, output_size):
        target = cv2.resize(target, (output_size, output_size),
                            interpolation=cv2.INTER_NEAREST)

    angle1 = stage1_result['angle']
    scale1 = stage1_result.get('scale', 1.0)
    cx1, cy1 = stage1_result.get('center', (output_size / 2.0, output_size / 2.0))
    tx1, ty1 = stage1_result['tx'], stage1_result['ty']

    n_eval = [0]
    best = {'iou': -1.0, 'params': None, 'proj': None}

    def _evaluate(lao, cran, tx, ty):
        proj = project_mask_3d(
            heart_mask_3d, ct_aff, None,
            lao_deg=lao, cran_deg=cran + 180,
            table_angle=table_angle, output_size=output_size,
        )
        moved = apply_transform(proj, tx, ty, angle1, (cx1, cy1), scale1)
        score = iou_score(moved, target)
        n_eval[0] += 1
        if score > best['iou']:
            best.update({'iou': score, 'params': (lao, cran, tx, ty), 'proj': moved})
        return score

    def _obj(p):
        return -_evaluate(*p)

    # ── Projection initiale (référence "avant") ────────────────────────────
    if progress_cb:
        progress_cb(2, 'Projection initiale…')

    proj_init = project_mask_3d(
        heart_mask_3d, ct_aff, None,
        lao_deg=lao_deg, cran_deg=cran_deg + 180,
        table_angle=table_angle, output_size=output_size,
    )
    proj_init_aligned = apply_transform(proj_init, tx1, ty1, angle1, (cx1, cy1), scale1)
    iou_init = iou_score(proj_init_aligned, target)

    # ── Grille 5×5 sur (LAO, CRAN) ────────────────────────────────────────
    if progress_cb:
        progress_cb(5, f'IoU initial={iou_init:.3f} — Recherche angulaire (25 pts)…')

    half = search_range_deg / 2.0
    angles = np.linspace(-half, half, 5)
    done = 0
    for dlao in angles:
        for dcran in angles:
            _evaluate(lao_deg + dlao, cran_deg + dcran, tx1, ty1)
            done += 1
            if progress_cb and done % 5 == 0:
                progress_cb(5 + int(50 * done / 25),
                            f'Grid {done}/25  IoU={best["iou"]:.3f}')

    best_init_lao = best['params'][0] if best['params'] else lao_deg
    best_init_cran = best['params'][1] if best['params'] else cran_deg

    if progress_cb:
        progress_cb(58, f'Grid done — LAO={best_init_lao:.1f}°  CRAN={best_init_cran:.1f}°  '
                        f'IoU={best["iou"]:.3f} — Nelder-Mead…')

    # ── Nelder-Mead : (LAO, CRAN, tx, ty) ─────────────────────────────────
    x0 = [best_init_lao, best_init_cran, tx1, ty1]
    minimize(_obj, x0=x0, method='Nelder-Mead',
             options={'xatol': 0.1, 'fatol': 1e-5, 'maxiter': 300,
                      'adaptive': True})

    lao_f, cran_f, tx_f, ty_f = best['params']
    iou_f = best['iou']

    if progress_cb:
        pct = (iou_f - iou_init) / (iou_init + 1e-8) * 100
        progress_cb(100, f'Terminé — IoU: {iou_init:.3f} → {iou_f:.3f} ({pct:+.1f}%)  '
                         f'LAO={lao_f:.1f}°  CRAN={cran_f:.1f}°  '
                         f'{n_eval[0]} éval.')

    return {
        'best_lao': lao_f,
        'best_cran': cran_f,
        'tx': tx_f,
        'ty': ty_f,
        'angle': angle1,
        'scale': scale1,
        'center': (cx1, cy1),
        'initial_iou': iou_init,
        'best_iou': iou_f,
        'heart_proj_before': proj_init_aligned,
        'heart_proj_after': best['proj'],
        'n_evaluations': n_eval[0],
    }
