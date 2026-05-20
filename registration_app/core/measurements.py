"""TAVI risk metrics : membranous septum length & pacemaker dependency.

Référence clinique : Nai Fovino et al., "Anatomical Predictors of Pacemaker
Dependency After TAVR", Circ Arrhythm Electrophysiol. 2021;14:e009028.

Le risque dépend principalement de ΔMSID = ID − MS_length (cut-off 3 mm,
OR 7.58, sensibilité 84 %, spécificité 69 %).

REPÈRES ANATOMIQUES (Landmarks) :
═════════════════════════════════════════════════════════════════════════════

1. HINGE1 (Hinge gauche / Left hinge)
   - Localisation : Base de la valve aortique côté gauche
   - Anatomie : Jonction cusp-paroi ventriculaire du côté gauche
   - Vue idéale : Coupe coronale (projection AP avec angulation craniale)
   - Objectif : Marquer le pôle gauche de la ligne annulaire
   - Risque associé : Proximité au système de conduction auriculo-ventriculaire

2. HINGE2 (Hinge droit / Right hinge)
   - Localisation : Base de la valve aortique côté droit
   - Anatomie : Jonction cusp-paroi ventriculaire du côté droit
   - Vue idéale : Même coupe que HINGE1 pour délimiter l'anneau valve
   - Objectif : Complète HINGE1 pour tracer la ligne annulaire
   - Distance HINGE1-HINGE2 : Longueur de l'anneau valvulaire (~15-25 mm)

3. MS (Membranous Septum point)
   - Localisation : Apex du septum membraneux / début du septum musculaire
   - Anatomie : Point critique juste au-dessus du septum musculaire
   - Vue idéale : Coupe coronale ou petit-axe (perpendiculaire à annulus)
   - Objectif : Référence supérieure pour le système de conduction
   - Rôle clinique : Sommet du risque PM (pacemaker) / MS_length = distance annulus→MS

4. NCC (Non-Coronary Cusp base)
   - Localisation : Base/jonction inférieure du cusp non-coronaire
   - Anatomie : Pointe inférieure du cusp non-coronaire (le 3e cusp aortique)
   - Vue idéale : Coupe petit-axe ou coronale où le cusp est maximal
   - Objectif : Référence pour profondeur d'implantation (ID = distance stent→NCC)
   - Relation : NCC est généralement 2-5 mm sous la ligne hinge-hinge

INTERPRÉTATION DU RISQUE PM :
═════════════════════════════════════════════════════════════════════════════
- MS_length : Distance annulaire → apex septum membraneux (5-12 mm, normal)
- ID : Distance distal stent → NCC (implantation profonde, 0-8 mm)
- ΔMSID = MS_length − ID (cut-off 3 mm)
  • ΔMSID < 3 mm : Risque ÉLEVÉ (70% PM dépendance à 6 mois)
  • ΔMSID ≥ 3 mm : Risque BAS (2.7% PM dépendance à 6 mois)

CONSEILS DE PLACEMENT :
═════════════════════════════════════════════════════════════════════════════
- Commencez par HINGE1 et HINGE2 sur coupe coronale pour anchorer l'anneau
- Localisez MS perpendiculairement au milieu de la ligne hinge-hinge
- Finalisez avec NCC (le point de repère le plus difficile, à chercher sous l'anneau)
- Utilisez le zoom et pan de la vue CT pour améliorer la précision
- Chaque point est projeté sur les coupes après recalage pour vérification visuelle
"""

import math
import numpy as np


DELTA_MSID_THRESHOLD_MM = 3.0
PM_DEPENDENCY_LOW = 0.027   # ΔMSID < 3 mm
PM_DEPENDENCY_HIGH = 0.70   # ΔMSID ≥ 3 mm


def inverse_register_point_2d(point_xy, reg_result):
    """Inverse de la similitude 2D ``apply_full_transform`` appliquee a un point.

    register() retourne (tx, ty, angle, scale, center) qui transforme le DRR
    pour matcher la fluoroscopie. Pour mapper fluoro -> DRR, on inverse la
    matrice 2x3 de cv2.getRotationMatrix2D etendue en 3x3.
    """
    if reg_result is None:
        return tuple(float(v) for v in point_xy)
    import cv2
    angle = float(reg_result.get('angle', 0.0))
    scale = float(reg_result.get('scale', 1.0))
    cx, cy = reg_result.get('center', (0.0, 0.0))
    tx = float(reg_result.get('tx', 0.0))
    ty = float(reg_result.get('ty', 0.0))
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, scale)
    M[0, 2] += tx; M[1, 2] += ty
    M3 = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64)
    p = np.array([float(point_xy[0]), float(point_xy[1]), 1.0])
    q = np.linalg.inv(M3) @ p
    return float(q[0]), float(q[1])


def inverse_register_axis_2d(axis_deg, reg_result):
    """L'angle d'axe d'un objet : appliquer −angle_reg pour revenir au repere DRR."""
    if reg_result is None:
        return float(axis_deg)
    return float(axis_deg) - float(reg_result.get('angle', 0.0))


def register_point_2d(point_xy, reg_result):
    """Applique la similitude (repere DRR -> repere fluoro) a un point 2D."""
    if reg_result is None:
        return tuple(float(v) for v in point_xy)
    import cv2
    angle = float(reg_result.get('angle', 0.0))
    scale = float(reg_result.get('scale', 1.0))
    cx, cy = reg_result.get('center', (0.0, 0.0))
    tx = float(reg_result.get('tx', 0.0))
    ty = float(reg_result.get('ty', 0.0))
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), angle, scale)
    M[0, 2] += tx; M[1, 2] += ty
    p = np.array([float(point_xy[0]), float(point_xy[1]), 1.0])
    q = M @ p
    return float(q[0]), float(q[1])


def project_world_to_fluoro_pixel(world_pt, ct_center_world, image_size,
                                  lao_deg, cran_deg, sid_mm, sod_mm, fov_mm,
                                  registration_result=None):
    """Projette un point 3D du repere CT-monde vers le pixel fluoro.

    1) Translation -> centre du volume CT (qui sert d'iso-centre cone-beam),
    2) Rotation DRR (LAO autour Z, CRAN autour X),
    3) Projection cone-beam (source a y=-SOD, detecteur a y=SID-SOD),
    4) Mise a l'echelle pixel,
    5) Recalage 2D (DRR -> fluoro) si registration_result fourni.
    """
    p_c = np.asarray(world_pt, dtype=np.float64) - np.asarray(ct_center_world, dtype=np.float64)
    rot = _drr_rotation(lao_deg, cran_deg)
    p_cam = rot @ p_c
    src_y = -float(sod_mm)
    det_y = float(sid_mm) - float(sod_mm)
    dy = p_cam[1] - src_y
    if abs(dy) < 1e-6:
        return None
    t = (det_y - src_y) / dy
    if t <= 0:
        return None
    u_det = p_cam[0] * t
    v_det = p_cam[2] * t
    size = float(image_size)
    pix_mm = (float(fov_mm) * (float(sid_mm) / float(sod_mm))) / size
    px_drr = size * 0.5 + u_det / pix_mm
    py_drr = size * 0.5 - v_det / pix_mm
    return register_point_2d((px_drr, py_drr), registration_result)


def _drr_rotation(lao_deg: float, cran_deg: float) -> np.ndarray:
    """Rotation appliquée par ``project_stent_mask`` (LAO autour Z, CRAN autour X).

    R_cam_from_ct = Rx(CRAN) @ Rz(LAO). Inverse = transposée (rotation orthogonale).
    """
    lao = math.radians(float(lao_deg))
    cran = math.radians(float(cran_deg))
    cz, sz = math.cos(lao), math.sin(lao)
    cx, sx = math.cos(cran), math.sin(cran)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    return rx @ rz


def stent_pose_in_ct(center_px, axis_deg_2d, image_size,
                     lao_deg, cran_deg, sid_mm, sod_mm, fov_mm,
                     ct_center_world, depth_mm: float = 0.0):
    """Reconstruit la pose 3D du stent (centre + axe unitaire) en coords CT-monde.

    Le centre est positionne le long du rayon source -> pixel detecteur,
    a la profondeur ``depth_mm`` (mm le long de l'axe Y du repere camera,
    positif = vers le detecteur, 0 = iso-centre).

    Returns
    -------
    (center_world, axis_world_unit) : tuple of np.ndarray shape (3,)
    """
    size = float(image_size)
    sid = float(sid_mm); sod = float(sod_mm)
    pix_mm = (float(fov_mm) * (sid / sod)) / size
    # Pixel -> coord détecteur (mm) dans le repère caméra
    u_det = (float(center_px[0]) - size * 0.5) * pix_mm
    v_det = (size * 0.5 - float(center_px[1])) * pix_mm
    # Source a y_cam = -SOD, detecteur a y_cam = SID-SOD. Le rayon passe par source et (u_det, SID-SOD, v_det).
    # A la profondeur y_target dans le repere camera, le point sur le rayon est :
    #   (u_det * (y_target + SOD) / SID, y_target, v_det * (y_target + SOD) / SID).
    y_target = float(depth_mm)
    t = (y_target + sod) / sid
    center_cam = np.array([u_det * t, y_target, v_det * t], dtype=np.float64)
    # Axe 2D : convention ``atan2(-dy_image, dx_image)`` -> direction camera (cos, 0, sin).
    a = math.radians(float(axis_deg_2d))
    axis_cam = np.array([math.cos(a), 0.0, math.sin(a)], dtype=np.float64)
    # Repère caméra -> repère CT (rotation inverse) puis translation a l'iso-centre CT
    rot = _drr_rotation(lao_deg, cran_deg)
    center_ct = rot.T @ center_cam + np.asarray(ct_center_world, dtype=np.float64)
    axis_ct = rot.T @ axis_cam
    n = float(np.linalg.norm(axis_ct))
    if n > 1e-9:
        axis_ct /= n
    return center_ct, axis_ct


def implantation_depth_3d(stent_center_world, stent_axis_world, stent_length_mm, ncc_world):
    """ID = distance le long de l'axe stent entre l'extrémité basse et le NCC.

    L'extrémité « basse » est définie comme celle la plus proche du NCC
    (la cuspe non-coronaire est sous le THV côté LVOT). On projette ensuite
    le vecteur (NCC − bas) sur l'axe stent pour avoir une mesure parallèle au
    long axis de la racine aortique (convention article Nai Fovino).
    """
    c = np.asarray(stent_center_world, dtype=np.float64)
    u = np.asarray(stent_axis_world, dtype=np.float64)
    n = float(np.linalg.norm(u))
    if n < 1e-9:
        return None
    u = u / n
    half = 0.5 * float(stent_length_mm)
    end1 = c + u * half
    end2 = c - u * half
    ncc = np.asarray(ncc_world, dtype=np.float64)
    bot = end1 if np.linalg.norm(end1 - ncc) < np.linalg.norm(end2 - ncc) else end2
    # Distance signée le long de l'axe (toujours positive : on prend |.|)
    return float(abs(np.dot(ncc - bot, u)))


def stent_endpoints_world(stent_center_world, stent_axis_world, stent_length_mm):
    """Retourne les deux extrémités 3D du stent en coords monde."""
    c = np.asarray(stent_center_world, dtype=np.float64)
    u = np.asarray(stent_axis_world, dtype=np.float64)
    n = float(np.linalg.norm(u))
    if n > 1e-9:
        u = u / n
    half = 0.5 * float(stent_length_mm)
    return c + u * half, c - u * half


def ray_from_drr_pixel(pixel_xy, image_size, lao_deg, cran_deg,
                       sid_mm, sod_mm, fov_mm, ct_center_world):
    """Construit le rayon source 3D -> pixel DRR detecteur en coords CT-monde.

    Retourne (origin_world, direction_unit_world).
    """
    size = float(image_size)
    sid = float(sid_mm); sod = float(sod_mm)
    pix_mm = (float(fov_mm) * (sid / sod)) / size
    u_det = (float(pixel_xy[0]) - size * 0.5) * pix_mm
    v_det = (size * 0.5 - float(pixel_xy[1])) * pix_mm
    src_cam = np.array([0.0, -sod, 0.0], dtype=np.float64)
    det_cam = np.array([u_det, sid - sod, v_det], dtype=np.float64)
    inv_rot = _drr_rotation(lao_deg, cran_deg).T
    ct_c = np.asarray(ct_center_world, dtype=np.float64)
    src_world = inv_rot @ src_cam + ct_c
    det_world = inv_rot @ det_cam + ct_c
    direction = det_world - src_world
    n = float(np.linalg.norm(direction))
    if n > 1e-9:
        direction /= n
    return src_world, direction


def closest_point_on_ray(anchor_world, ray_origin, ray_direction):
    """Point du rayon ``origin + t * direction`` le plus proche de ``anchor`` (t >= 0)."""
    a = np.asarray(anchor_world, dtype=np.float64)
    o = np.asarray(ray_origin, dtype=np.float64)
    d = np.asarray(ray_direction, dtype=np.float64)
    t = float(np.dot(a - o, d))
    return o + max(t, 0.0) * d


def stent_3d_pose_from_fluoro(stent_center_px_fluoro, stent_axis_deg_fluoro,
                              stent_length_mm, image_size,
                              lao_deg, cran_deg, sid_mm, sod_mm, fov_mm,
                              ct_center_world, anchor_world,
                              registration_result=None):
    """Reconstruit (center_world, axis_unit_world) de la pose 3D du stent.

    1) Inverse(recalage) : fluoro -> DRR pixel,
    2) Rayon back-projecte du pixel DRR,
    3) Profondeur fixee par projection orthogonale de ``anchor_world`` sur le rayon,
    4) Axe : direction 2D back-projetee dans le plan perpendiculaire au rayon.
    """
    cx_drr, cy_drr = inverse_register_point_2d(stent_center_px_fluoro, registration_result)
    ang_drr = inverse_register_axis_2d(stent_axis_deg_fluoro, registration_result)
    src_w, dir_w = ray_from_drr_pixel(
        (cx_drr, cy_drr), image_size, lao_deg, cran_deg,
        sid_mm, sod_mm, fov_mm, ct_center_world,
    )
    center_w = closest_point_on_ray(anchor_world, src_w, dir_w)
    # Direction de l'axe 2D en frame camera (cos(a), 0, sin(a)) puis rotation inverse
    a = math.radians(float(ang_drr))
    axis_cam = np.array([math.cos(a), 0.0, math.sin(a)], dtype=np.float64)
    inv_rot = _drr_rotation(lao_deg, cran_deg).T
    axis_w = inv_rot @ axis_cam
    n = float(np.linalg.norm(axis_w))
    if n > 1e-9:
        axis_w /= n
    return center_w, axis_w


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
