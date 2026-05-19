"""TAVI risk metrics : membranous septum length & pacemaker dependency.

Référence clinique : Nai Fovino et al., "Anatomical Predictors of Pacemaker
Dependency After TAVR", Circ Arrhythm Electrophysiol. 2021;14:e009028.

Le risque dépend principalement de ΔMSID = ID − MS_length (cut-off 3 mm,
OR 7.58, sensibilité 84 %, spécificité 69 %).
"""

import numpy as np


DELTA_MSID_THRESHOLD_MM = 3.0
PM_DEPENDENCY_LOW = 0.027   # ΔMSID < 3 mm
PM_DEPENDENCY_HIGH = 0.70   # ΔMSID ≥ 3 mm


def world_from_voxel(voxel_xyz, affine):
    """Index voxel (x,y,z) -> coords mondiales (mm) via affine NIfTI."""
    v = np.asarray(voxel_xyz, dtype=np.float64)
    if affine is None:
        return v.copy()
    return (np.asarray(affine) @ np.append(v, 1.0))[:3]


def voxel_from_world(world_mm, affine):
    """Inverse de world_from_voxel."""
    w = np.asarray(world_mm, dtype=np.float64)
    if affine is None:
        return w.copy()
    return (np.linalg.inv(np.asarray(affine)) @ np.append(w, 1.0))[:3]


def voxel_from_view_pixel(plane, col, row, vol_shape, slice_idx):
    """Pixel (col,row) d'une vue ``np.rot90(slice, 1)`` -> voxel (x,y,z).

    Les coupes sont rendues avec une rotation CCW pour mettre le crânien en haut.
    Inverse : rotated[r,c] vient de original[c, W−1−r] (W = largeur originale).
    """
    sx, sy, sz = (int(v) for v in vol_shape)
    c = int(round(col)); r = int(round(row))
    if plane == 'axial':            # vol[:, :, idx] shape (sx,sy) -> rot90 (sy,sx)
        x, y, z = c, sy - 1 - r, slice_idx
    elif plane == 'coronal':        # vol[:, idx, :] shape (sx,sz) -> rot90 (sz,sx)
        x, y, z = c, slice_idx, sz - 1 - r
    else:                           # sagittal vol[idx, :, :] shape (sy,sz) -> rot90 (sz,sy)
        x, y, z = slice_idx, c, sz - 1 - r
    return (max(0, min(sx - 1, x)),
            max(0, min(sy - 1, y)),
            max(0, min(sz - 1, z)))


def view_pixel_from_voxel(plane, voxel_xyz, vol_shape):
    """Forward : voxel (x,y,z) -> (col, row) sur la vue rot90 + index natif de coupe."""
    sx, sy, sz = (int(v) for v in vol_shape)
    x, y, z = (int(round(v)) for v in voxel_xyz)
    if plane == 'axial':
        return float(x), float(sy - 1 - y), z
    if plane == 'coronal':
        return float(x), float(sz - 1 - z), y
    return float(y), float(sz - 1 - z), x


def ms_length_from_hinges(hinge1_mm, hinge2_mm, ms_point_mm):
    """Distance perpendiculaire 3D du point MS à la ligne joignant les deux hinges.

    Le plan annulaire est défini cliniquement par les nadirs des cusps ; sur
    coupe coronale ses deux hinges visibles donnent une droite — la distance
    perpendiculaire du MS apex à cette droite est notre estimation du MS length.
    """
    h1 = np.asarray(hinge1_mm, dtype=np.float64)
    h2 = np.asarray(hinge2_mm, dtype=np.float64)
    p = np.asarray(ms_point_mm, dtype=np.float64)
    axis = h2 - h1
    n = float(np.linalg.norm(axis))
    if n < 1e-6:
        return float(np.linalg.norm(p - h1))
    u = axis / n
    rel = p - h1
    return float(np.linalg.norm(rel - np.dot(rel, u) * u))


def risk_assessment(ms_length_mm, implantation_depth_mm=None):
    """Évalue le risque PM-dependency post-TAVR.

    Sans ID -> aucune stratification, seul MS rapporté.
    """
    out = {
        'ms_length_mm': round(float(ms_length_mm), 2),
        'implantation_depth_mm': None,
        'delta_msid_mm': None,
        'risk_level': 'unknown',
        'pm_dependency_rate': None,
        'reason': 'Marquer le cusp non-coronaire sur la fluoro (apres pose du stent) pour le score complet.',
    }
    if implantation_depth_mm is None:
        return out
    d = float(implantation_depth_mm) - float(ms_length_mm)
    out['implantation_depth_mm'] = round(float(implantation_depth_mm), 2)
    out['delta_msid_mm'] = round(d, 2)
    if d >= DELTA_MSID_THRESHOLD_MM:
        out['risk_level'] = 'HIGH'
        out['pm_dependency_rate'] = PM_DEPENDENCY_HIGH
        out['reason'] = (f'DeltaMSID = {d:.1f} mm >= {DELTA_MSID_THRESHOLD_MM:g} mm : '
                         f'OR 7.58, ~{PM_DEPENDENCY_HIGH:.0%} de PM dependency.')
    else:
        out['risk_level'] = 'LOW'
        out['pm_dependency_rate'] = PM_DEPENDENCY_LOW
        out['reason'] = (f'DeltaMSID = {d:.1f} mm < {DELTA_MSID_THRESHOLD_MM:g} mm : '
                         f'risque faible (~{PM_DEPENDENCY_LOW:.1%}).')
    return out
