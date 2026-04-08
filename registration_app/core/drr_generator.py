"""
core/drr_generator.py
Génération de DRR – cone-beam, géométrie C-arm réaliste.

Backends (par ordre de préférence) :
  - "nanodrr"    : nanoDRR (rapide, HU → LAC physique, torch.compile)
  - "siddon"     : DiffDRR Siddon ray tracing (legacy)
  - "trilinear"  : DiffDRR trilinear interpolation (legacy)
  - "cpu"        : projection CPU scipy ray-sum (sans GPU)

Le backend GPU est choisi automatiquement :
  nanoDRR si disponible, sinon DiffDRR, sinon CPU.
"""

import numpy as np
import nibabel as nib
import cv2
from scipy.ndimage import rotate as _ndrot

_DEVICE = None

# ── nanoDRR (preferred) ──────────────────────────────────────────────────────
try:
    import torch
    from nanodrr.data import Subject as _NanoSubject
    from nanodrr.camera import make_k_inv as _make_k_inv, make_rt_inv as _make_rt_inv
    from nanodrr.drr import render as _nanodrr_render
    HAS_NANODRR = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_NANODRR = False

# ── DiffDRR (fallback) ───────────────────────────────────────────────────────
try:
    import torch
    from diffdrr.drr import DRR as _DiffDRR
    from diffdrr.data import read as _diffdrr_read
    HAS_DIFFDRR = True
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    HAS_DIFFDRR = False


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
    - renderer    : "nanodrr" | "siddon" | "trilinear" | "cpu"
    - progress_cb : callback(pct, msg)

    Retourne : image float32 [0, 1]
    """
    if renderer == "cpu":
        return _generate_drr_cpu(
            ct_path, lao_deg, cran_deg, table_angle, output_size, progress_cb)

    # ── nanoDRR (preferred GPU backend) ───────────────────────────────────
    if HAS_NANODRR and renderer != "trilinear":
        return _generate_drr_nanodrr(
            ct_path, lao_deg, cran_deg, table_angle,
            output_size, sid_mm, sod_mm, fov_mm, progress_cb)

    # ── DiffDRR fallback ──────────────────────────────────────────────────
    if HAS_DIFFDRR:
        return _generate_drr_diffdrr(
            ct_path, lao_deg, cran_deg, table_angle,
            output_size, sid_mm, sod_mm, fov_mm,
            renderer if renderer in ("siddon", "trilinear") else "siddon",
            progress_cb)

    raise RuntimeError(
        'Aucun backend GPU disponible — installez nanodrr ou diffdrr')


# ══════════════════════════════════════════════════════════════════════════════
# nanoDRR backend — projection cone-beam avec conversion HU → LAC physique
# ══════════════════════════════════════════════════════════════════════════════

def _generate_drr_nanodrr(
    ct_path, lao_deg, cran_deg, table_angle,
    output_size, sid_mm, sod_mm, fov_mm, progress_cb,
):
    if progress_cb:
        progress_cb(5, 'Chargement CT (nanoDRR)…')

    # mu_bone augmenté (×2) pour renforcer le contraste osseux
    from nanodrr.data.preprocess import MU_BONE
    subject = _NanoSubject.from_filepath(
        ct_path, mu_bone=MU_BONE * 2.0,
    ).to(_DEVICE)

    delx = _compute_delx(fov_mm, sid_mm, sod_mm, output_size)
    k_inv = _make_k_inv(
        sdd=sid_mm, delx=delx, dely=delx,
        x0=0.0, y0=0.0,
        height=output_size, width=output_size,
        device=_DEVICE,
    )
    sdd_t = torch.tensor([sid_mm], device=_DEVICE)

    # Extrinsics — ZXY Euler angles in degrees, same convention as DiffDRR
    rt_inv = _make_rt_inv(
        torch.tensor([[cran_deg, lao_deg, table_angle]],
                      dtype=torch.float32, device=_DEVICE),
        torch.tensor([[0.0, sod_mm, 0.0]],
                      dtype=torch.float32, device=_DEVICE),
        orientation="AP",
        isocenter=subject.isocenter,
    )

    if progress_cb:
        progress_cb(30, 'Projection cone-beam (nanoDRR)…')

    with torch.no_grad():
        img_tensor = _nanodrr_render(
            subject, k_inv, rt_inv, sdd_t,
            output_size, output_size,
        )

    # Sum across label channels if present, take first batch
    img_np = img_tensor.sum(dim=1).squeeze(0).cpu().numpy()

    if progress_cb:
        progress_cb(70, 'Post-traitement…')

    return _postprocess(img_np)


# ══════════════════════════════════════════════════════════════════════════════
# DiffDRR backend (legacy fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _generate_drr_diffdrr(
    ct_path, lao_deg, cran_deg, table_angle,
    output_size, sid_mm, sod_mm, fov_mm, renderer, progress_cb,
):
    if progress_cb:
        progress_cb(5, f'Chargement CT (DiffDRR {renderer})…')

    subject = _diffdrr_read(ct_path, orientation="AP")
    delx = _compute_delx(fov_mm, sid_mm, sod_mm, output_size)

    drr_module = _DiffDRR(
        subject,
        sdd=sid_mm,
        height=output_size,
        width=output_size,
        delx=delx,
        renderer=renderer,
    ).to(_DEVICE)

    if progress_cb:
        progress_cb(30, 'Projection cone-beam (DiffDRR)…')

    rot = torch.tensor([[cran_deg, lao_deg, table_angle]],
                       dtype=torch.float32, device=_DEVICE)
    tra = torch.tensor([[0.0, sod_mm, 0.0]],
                       dtype=torch.float32, device=_DEVICE)

    with torch.no_grad():
        img_tensor = drr_module(rot, tra,
                                parameterization="euler_angles",
                                convention="ZXY",
                                degrees=True)

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
# Générateur rapide (CT pré-chargé) — pour boucle d'optimisation
# ══════════════════════════════════════════════════════════════════════════════

def create_fast_generator(ct_path: str, output_size: int = 256,
                          sid_mm: float = 1020.0, sod_mm: float = 510.0,
                          fov_mm: float = None, table_angle: float = 0.0):
    """
    Retourne un callable ``gen(lao_deg, cran_deg) -> float32 [0,1]``
    qui génère un DRR *sans* recharger le CT à chaque appel.

    Les angles attendus sont les valeurs UI brutes (PAS +180 pour cran).
    Le +180 est ajouté automatiquement en interne.

    Backends (par priorité) : nanoDRR → DiffDRR → CPU (lent).
    """
    delx = _compute_delx(fov_mm, sid_mm, sod_mm, output_size)

    # ── nanoDRR ───────────────────────────────────────────────────────────
    if HAS_NANODRR:
        from nanodrr.data.preprocess import MU_BONE
        subject = _NanoSubject.from_filepath(
            ct_path, mu_bone=MU_BONE * 2.0,
        ).to(_DEVICE)
        k_inv = _make_k_inv(
            sdd=sid_mm, delx=delx, dely=delx,
            x0=0.0, y0=0.0,
            height=output_size, width=output_size,
            device=_DEVICE,
        )
        sdd_t = torch.tensor([sid_mm], device=_DEVICE)

        def _gen_nano(lao_deg, cran_deg):
            rt_inv = _make_rt_inv(
                torch.tensor([[cran_deg + 180, lao_deg, table_angle]],
                              dtype=torch.float32, device=_DEVICE),
                torch.tensor([[0.0, sod_mm, 0.0]],
                              dtype=torch.float32, device=_DEVICE),
                orientation="AP",
                isocenter=subject.isocenter,
            )
            with torch.no_grad():
                img_t = _nanodrr_render(
                    subject, k_inv, rt_inv, sdd_t,
                    output_size, output_size)
            return _postprocess(img_t.sum(dim=1).squeeze(0).cpu().numpy())

        return _gen_nano

    # ── DiffDRR ───────────────────────────────────────────────────────────
    if HAS_DIFFDRR:
        from diffdrr.pose import convert as _pose_convert
        subject = _diffdrr_read(ct_path, orientation="AP")
        drr_module = _DiffDRR(
            subject, sdd=sid_mm,
            height=output_size, width=output_size,
            delx=delx, renderer="siddon",
        ).to(_DEVICE)

        def _gen_diffdrr(lao_deg, cran_deg):
            rot = torch.tensor([[cran_deg + 180, lao_deg, table_angle]],
                               dtype=torch.float32, device=_DEVICE)
            tra = torch.tensor([[0.0, sod_mm, 0.0]],
                               dtype=torch.float32, device=_DEVICE)
            with torch.no_grad():
                img_t = drr_module(rot, tra,
                                   parameterization="euler_angles",
                                   convention="ZXY", degrees=True)
            return _postprocess(img_t.squeeze().cpu().numpy())

        return _gen_diffdrr

    # ── CPU fallback (lent) ───────────────────────────────────────────────
    def _gen_cpu(lao_deg, cran_deg):
        return generate_drr(
            ct_path=ct_path,
            lao_deg=lao_deg,
            cran_deg=cran_deg + 180,
            table_angle=table_angle,
            output_size=output_size,
            sid_mm=sid_mm,
            sod_mm=sod_mm,
            fov_mm=fov_mm,
            renderer="cpu",
        )

    return _gen_cpu


# ══════════════════════════════════════════════════════════════════════════════
# Helpers internes
# ══════════════════════════════════════════════════════════════════════════════

def _postprocess(img_np: np.ndarray) -> np.ndarray:
    """
    Post-traitement réaliste du DRR brut → float32 [0, 1] aspect radio.

    Pipeline :
      1. Log-transform (loi de Beer-Lambert) — compresse la dynamique,
         réduit la surbrillance de l'aorte et des zones denses.
      2. Normalisation robuste (percentiles serrés p0.5/p99.5).
      3. Inversion — os clairs sur fond sombre (convention radio positive).
      4. Gamma correction (γ = 0.6) — réhausse le contraste dans les tons moyens.
      5. CLAHE adapté médical — fait ressortir les vertèbres.
      6. Unsharp-mask léger — netteté des contours osseux.
    """
    # 1. Log-transform : line-integral → "épaisseur d'atténuation" logarithmique
    #    eps évite log(0) ; log1p = log(1 + x) compresse naturellement les hautes valeurs
    img = np.log1p(np.clip(img_np, 0, None))

    # 2. Normalisation robuste — écrêtage des outliers (metal, bord CT)
    p_lo, p_hi = np.percentile(img, (0.5, 99.5))
    if p_hi > p_lo:
        img = np.clip((img - p_lo) / (p_hi - p_lo), 0, 1)
    else:
        img = np.zeros_like(img)

    # 3. Flip horizontal — C-arm PA : l'image est miroitée gauche-droite
    #    pour correspondre à la fluoro
    img = np.fliplr(img)

    # 4. Gamma correction — γ < 1 éclaircit les tons moyens,
    #    fait mieux ressortir les vertèbres par rapport au fond
    gamma = 0.6
    img = np.power(img, gamma)

    # 5. CLAHE à contraste adaptatif — rehausse les structures locales (vertèbres)
    img_u8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_u8 = clahe.apply(img_u8)

    # 6. Unsharp-mask léger pour la netteté des contours
    blur = cv2.GaussianBlur(img_u8, (0, 0), sigmaX=1.5)
    img_u8 = cv2.addWeighted(img_u8, 1.3, blur, -0.3, 0)

    return img_u8.astype(np.float32) / 255.0


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
