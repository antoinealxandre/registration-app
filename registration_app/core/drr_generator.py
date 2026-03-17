"""
core/drr_generator.py
GéGénération de DRR via DiffDRR (cone-beam, géométrie C-arm réaliste).
Trois backends disponibles :
  - "siddon"     : DiffDRR Siddon ray tracing (exact, recommandé)
  - "trilinear"  : DiffDRR trilinear interpolation (lisse, légèrement plus rapide)
  - "cpu"        : projection CPU scipy ray-sum (rapide, sans GPU/DiffDRR requis)

NOTE ANGLES : DiffDRR attend des radians par défaut.
On passe toujours degrees=True pour rester en degrés côté API.
Bug courant : 0.6 radians ≈ 34° ≠ 0.6°.
"""

import numpy as np
import nibabel as nib
import cv2
import tempfile, os  # kept for potential external callers
from scipy.ndimage import rotate as _ndrot

try:
    import torch
    from diffdrr.drr import DRR
    from diffdrr.data import read as diffdrr_read
    HAS_DIFFDRR = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_DIFFDRR = False
    _DEVICE = None


def load_ct(ct_path: str):
    """Charge un CT NIfTI, retourne (volume HU, voxel_mm, affine, nib_img, ap_axis, codes)."""
    img = nib.load(ct_path)
    vol = img.get_fdata().astype(np.float32)
    aff = img.affine
    voxel_mm = np.abs(np.diag(aff)[:3])

    codes = nib.aff2axcodes(aff)
    try:
        ap_axis = list(codes).index('A')
    except ValueError:
        try:
            ap_axis = list(codes).index('P')
        except ValueError:
            ap_axis = 1

    return vol, voxel_mm, aff, img, ap_axis, codes


# ══════════════════════════════════════════════════════════════════════════════
# DiffDRR backend — projection cone-beam réaliste
# ══════════════════════════════════════════════════════════════════════════════

def _compute_delx(fov_mm: float, sid_mm: float, sod_mm: float,
                  output_size: int) -> float:
    """Calcule le pixel spacing détecteur (mm/px) depuis le FOV à l'isocentre.
    
    fov_mm  = champ de vue physique à l'isocentre
    Au détecteur, le FOV est magnifié : fov_det = fov_mm * (sid/sod)
    delx = fov_det / output_size
    """
    if fov_mm and fov_mm > 0 and sid_mm and sod_mm and sod_mm > 0:
        fov_det = fov_mm * (sid_mm / sod_mm)
        return fov_det / output_size
    # Défaut raisonnable : FOV détecteur ~400mm pour 512px
    return 400.0 / output_size


def generate_drr(
    ct_path: str,
    lao_deg: float = 0.0,
    cran_deg: float = 0.0,
    table_angle: float = 0.0,
    output_size: int = 512,
    sid_mm: float = 1020.0,
    sod_mm: float = 510.0,
    fov_mm: float = None,
    renderer: str = "siddon",
    progress_cb=None,
) -> np.ndarray:
    """
    Génère un DRR.

    - ct_path     : chemin vers le NIfTI du CT
    - lao_deg     : rotation LAO (+) / RAO (-) [deg]
    - cran_deg    : rotation crâniale (+) / caudale (-) [deg]
    - table_angle : rotation de la table [deg]
    - output_size : résolution de sortie [px]
    - sid_mm      : Source-to-Detector distance [mm]
    - sod_mm      : Source-to-Object (isocentre) distance [mm]
    - fov_mm      : champ de vue à l'isocentre [mm]
    - renderer    : "siddon" | "trilinear" | "cpu"
    - progress_cb : callback(pct, msg)

    Retourne : image float32 [0, 1]
    """
    if renderer == "cpu":
        return _generate_drr_cpu(
            ct_path, lao_deg, cran_deg, table_angle, output_size, progress_cb)

    if not HAS_DIFFDRR:
        raise RuntimeError(
            'DiffDRR non disponible — installez-le : pip install diffdrr')

    if progress_cb:
        progress_cb(5, f'Chargement CT ({renderer})…')

    subject = diffdrr_read(ct_path, orientation="AP")
    delx = _compute_delx(fov_mm, sid_mm, sod_mm, output_size)

    drr_module = DRR(
        subject,
        sdd=sid_mm,
        height=output_size,
        width=output_size,
        delx=delx,
        renderer=renderer,          # "siddon" ou "trilinear"
    ).to(_DEVICE)

    if progress_cb:
        progress_cb(30, 'Projection cone-beam…')

    # Convention DiffDRR ZXY : [cran, lao, table]  — degrees=True (IMPORTANT)
    # DiffDRR attend des radians par défaut ; degrees=True convertit en interne.
    rot = torch.tensor([[cran_deg, lao_deg, table_angle]],
                       dtype=torch.float32, device=_DEVICE)
    tra = torch.tensor([[0.0, sod_mm, 0.0]],
                       dtype=torch.float32, device=_DEVICE)

    with torch.no_grad():
        img_tensor = drr_module(rot, tra,
                                parameterization="euler_angles",
                                convention="ZXY",
                                degrees=True)   # ← FIX : angles en degrés

    img_np = img_tensor.squeeze().cpu().numpy()

    if progress_cb:
        progress_cb(70, 'Post-traitement…')

    return _postprocess(img_np)


def project_mask_3d(
    mask_3d: np.ndarray,
    ct_affine: np.ndarray,
    ct_path: str,
    lao_deg: float = 0.0,
    cran_deg: float = 0.0,
    table_angle: float = 0.0,
    output_size: int = 512,
    sid_mm: float = 1020.0,
    sod_mm: float = 510.0,
    fov_mm: float = None,
    threshold_frac: float = 0.05,
    renderer: str = "siddon",
) -> np.ndarray:
    """
    Projette un masque 3D binaire → masque 2D.

    Entièrement en mémoire (scipy rotate + max-projection sur l'axe AP).
    Aucune I/O disque, aucune dépendance DiffDRR → ~10 ms au lieu de minutes.
    La signature est identique pour rétro-compatibilité ; ct_path non utilisé.
    """
    vol = (mask_3d > 0).astype(np.float32)

    # Downsample ×2 sur chaque axe : 8× moins de voxels, qualité identique pour un masque binaire
    if vol.shape[0] > 64:
        vol = vol[::2, ::2, ::2]

    # Même convention d'axes que _generate_drr_cpu :
    # axis0=R-L  axis1=A-P  axis2=S-I
    # order=0 (plus proche voisin) : correct pour binaire et 3× plus rapide que order=1
    if abs(lao_deg) > 0.1:
        vol = _ndrot(vol, -lao_deg,    axes=(0, 1), reshape=False, order=0)
    if abs(cran_deg) > 0.1:
        vol = _ndrot(vol,  cran_deg,   axes=(1, 2), reshape=False, order=0)
    if abs(table_angle) > 0.1:
        vol = _ndrot(vol,  table_angle, axes=(0, 2), reshape=False, order=0)

    # Max-projection le long de l'axe AP : vrai si AU MOINS un voxel est dans le masque
    proj = vol.max(axis=1).astype(np.float32)   # (X, Z)
    proj = np.ascontiguousarray(proj.T)          # (Z, X) → même orientation que le DRR

    proj = cv2.resize(proj, (output_size, output_size),
                      interpolation=cv2.INTER_LINEAR)
    return (proj > 0.3).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers internes
# ══════════════════════════════════════════════════════════════════════════════

def _postprocess(img_np: np.ndarray) -> np.ndarray:
    """Percentile norm + CLAHE → float32 [0, 1]."""
    p2, p98 = np.percentile(img_np, (2, 98))
    if p98 > p2:
        img_np = np.clip((img_np - p2) / (p98 - p2), 0, 1)
    else:
        img_np = np.zeros_like(img_np)
    img_u8 = (img_np * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_u8).astype(np.float32) / 255.0


def _generate_drr_cpu(
    ct_path: str,
    lao_deg: float = 0.0,
    cran_deg: float = 0.0,
    table_angle: float = 0.0,
    output_size: int = 256,
    progress_cb=None,
) -> np.ndarray:
    """
    Projection CPU ray-sum (scipy.ndimage) — pas besoin de DiffDRR/GPU.
    Parallèle (pas de cone-beam) mais suffisant pour comparaison rapide.
    Angles en degrés.
    """
    if progress_cb:
        progress_cb(10, 'Chargement CT (CPU)…')

    img = nib.load(ct_path)
    vol = img.get_fdata().astype(np.float32)
    # Clamp : on ne projette que l'atténuation positive (os, tissus)
    vol = np.clip(vol, 0, None)

    if progress_cb:
        progress_cb(25, 'Rotation volume CPU…')

    # NIfTI convention typique : axis0=R-L, axis1=A-P, axis2=S-I
    # AP projection = sum axis 1
    # LAO  = rotation dans le plan axial (axes 0,1)
    # CRAN = tilt dans le plan sagittal (axes 1,2)
    # Table = rotation dans le plan coronal (axes 0,2)
    v = vol
    if abs(lao_deg) > 0.1:
        v = _ndrot(v, -lao_deg, axes=(0, 1), reshape=False, order=1)
    if abs(cran_deg) > 0.1:
        v = _ndrot(v,  cran_deg, axes=(1, 2), reshape=False, order=1)
    if abs(table_angle) > 0.1:
        v = _ndrot(v,  table_angle, axes=(0, 2), reshape=False, order=1)

    if progress_cb:
        progress_cb(70, 'Projection CPU…')

    proj = v.sum(axis=1).astype(np.float32)   # (X, Z) après somme AP
    proj = np.ascontiguousarray(proj.T)        # (Z, X) → affichage standard

    proj = cv2.resize(proj, (output_size, output_size),
                      interpolation=cv2.INTER_LINEAR)

    if progress_cb:
        progress_cb(85, 'Post-traitement CPU…')

    return _postprocess(proj)
