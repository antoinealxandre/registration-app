"""
core/drr_generator.py
Génération de DRR (Digitally Reconstructed Radiograph) par ray casting CPU.
Vue AP (antéro-postérieure) — rendu fluoroscopie-like.
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import rotate, zoom
import cv2


def load_ct(ct_path: str):
    """Charge un CT NIfTI, retourne (volume HU, voxel_mm, affine, nib_img)."""
    img = nib.load(ct_path)
    vol = img.get_fdata().astype(np.float32)
    aff = img.affine
    voxel_mm = np.abs(np.diag(aff)[:3])

    # Détecter l'axe AP depuis l'affine
    codes = nib.aff2axcodes(aff)
    try:
        ap_axis = list(codes).index('A')
    except ValueError:
        try:
            ap_axis = list(codes).index('P')
        except ValueError:
            ap_axis = 1   # fallback

    return vol, voxel_mm, aff, img, ap_axis, codes


def generate_drr(
    ct_vol: np.ndarray,
    voxel_mm: np.ndarray,
    ap_axis: int = 1,
    lao_deg: float = 0.0,
    cran_deg: float = 0.0,
    output_size: int = 512,
    hu_min: float = -500,
    hu_max: float = 2000,
    mu_water: float = 0.02,
    invert: bool = True,
) -> np.ndarray:
    """
    Génère un DRR fluoroscopie-like par intégration de rayons (Beer-Lambert).

    Paramètres clés :
    - ap_axis   : axe antéro-postérieur dans le volume NIfTI
    - lao_deg   : rotation LAO (+) / RAO (-) en degrés
    - cran_deg  : inclinaison crâniale (+) / caudale (-) en degrés
    - hu_min/max: fenêtre HU pour la projection
    - mu_water  : coefficient d'atténuation de l'eau (mm⁻¹)
    - invert    : True = os sombres sur fond clair (convention fluoroscopie)

    Retourne : image float32 [0, 1]
    """
    # 1. Fenêtrage HU et conversion en coefficients d'atténuation linéaire
    #    μ(x) = μ_water × (HU(x)/1000 + 1)  — approximation linéaire
    vol = np.clip(ct_vol, hu_min, hu_max)
    mu_vol = mu_water * (vol / 1000.0 + 1.0)
    mu_vol = np.clip(mu_vol, 0, None)   # pas de valeurs négatives

    # 2. Rotations C-arm
    if lao_deg != 0.0:
        mu_vol = rotate(mu_vol, angle=lao_deg, axes=(0, 1),
                        reshape=False, order=1, cval=0.0)
    if cran_deg != 0.0:
        mu_vol = rotate(mu_vol, angle=cran_deg, axes=(1, 2),
                        reshape=False, order=1, cval=0.0)

    # 3. Intégration le long de l'axe AP (Beer-Lambert : I = I0 × exp(-∫μ dl))
    #    On intègre μ × dl (dl = épaisseur de voxel en mm)
    dl = voxel_mm[ap_axis]
    integral = mu_vol.sum(axis=ap_axis) * dl   # shape: (X, Z) ou (Y, Z)...

    # 4. Correction d'aspect ratio (voxels anisotropes)
    ax_keep = [i for i in range(3) if i != ap_axis]
    ratio = voxel_mm[ax_keep[0]] / voxel_mm[ax_keep[1]]
    if abs(ratio - 1.0) > 0.02:
        integral = zoom(integral, zoom=(ratio, 1.0), order=1)

    # 5. Orientation : supérieur en haut (convention radiologique)
    proj = np.flipud(integral.T).astype(np.float32)

    # 6. Resize
    proj = cv2.resize(proj, (output_size, output_size),
                      interpolation=cv2.INTER_LINEAR)

    # 7. Conversion Beer-Lambert → intensité transmise
    #    I/I0 = exp(-∫μ dl) — plus c'est dense, plus c'est atténué
    transmitted = np.exp(-proj)

    # 8. Normalisation [0, 1]
    t_min, t_max = transmitted.min(), transmitted.max()
    if t_max > t_min:
        transmitted = (transmitted - t_min) / (t_max - t_min)

    # 9. Convention fluoroscopie : os = sombre, air = clair
    if invert:
        transmitted = 1.0 - transmitted

    # 10. CLAHE pour améliorer le contraste (comme une vraie fluoro)
    transmitted_u8 = (transmitted * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    transmitted_u8 = clahe.apply(transmitted_u8)
    transmitted = transmitted_u8.astype(np.float32) / 255.0

    return transmitted


def project_mask_3d(
    mask_3d: np.ndarray,
    voxel_mm: np.ndarray,
    ap_axis: int = 1,
    lao_deg: float = 0.0,
    cran_deg: float = 0.0,
    output_size: int = 512,
    threshold_frac: float = 0.05,
) -> np.ndarray:
    """
    Projette un masque 3D binaire en 2D (même géométrie que generate_drr).
    Retourne un masque binaire float32 [0, 1].
    """
    vol = mask_3d.astype(np.float32)

    if lao_deg != 0.0:
        vol = rotate(vol, angle=lao_deg, axes=(0, 1),
                     reshape=False, order=1, cval=0)
    if cran_deg != 0.0:
        vol = rotate(vol, angle=cran_deg, axes=(1, 2),
                     reshape=False, order=1, cval=0)

    proj = vol.sum(axis=ap_axis)

    ax_keep = [i for i in range(3) if i != ap_axis]
    ratio = voxel_mm[ax_keep[0]] / voxel_mm[ax_keep[1]]
    if abs(ratio - 1.0) > 0.02:
        proj = zoom(proj, zoom=(ratio, 1.0), order=1)

    proj = np.flipud(proj.T).astype(np.float32)
    proj = cv2.resize(proj, (output_size, output_size),
                      interpolation=cv2.INTER_LINEAR)

    thresh = proj.max() * threshold_frac
    return (proj > thresh).astype(np.float32)
