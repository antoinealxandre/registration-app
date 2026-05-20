"""DICOM and metadata CSV readers for fluoroscopy geometry."""

import re as _re

import numpy as np
import pandas as pd
import cv2

try:
    import pydicom
except ImportError:  # pragma: no cover - optional dependency
    pydicom = None

from ui.theme import DEFAULT_FOV_MM


def _frame_to_gray_float(frame: np.ndarray) -> np.ndarray:
    """Return a grayscale float32 view of a DICOM frame."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.ndim == 3:
        if arr.shape[-1] == 1:
            return arr[..., 0].astype(np.float32)
        if arr.shape[-1] >= 3:
            rgb = arr[..., :3].astype(np.float32)
            return (0.299 * rgb[..., 0]
                    + 0.587 * rgb[..., 1]
                    + 0.114 * rgb[..., 2]).astype(np.float32)
        return arr.mean(axis=-1).astype(np.float32)
    raise ValueError(f'Frame DICOM invalide (ndim={arr.ndim})')


def _normalize_dicom_frame(frame: np.ndarray, photometric: str = 'MONOCHROME2') -> np.ndarray:
    """Normalize one fluoroscopy frame to uint8 for display/annotation."""
    gray = _frame_to_gray_float(frame)
    finite = np.isfinite(gray)
    if not np.any(finite):
        return np.zeros(gray.shape[:2], dtype=np.uint8)

    values = gray[finite]
    lo, hi = np.percentile(values, (0.5, 99.7))
    if hi <= lo:
        lo = float(values.min())
        hi = float(values.max())

    if hi > lo:
        gray = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
    else:
        gray = np.zeros_like(gray, dtype=np.float32)

    if str(photometric).upper() == 'MONOCHROME1':
        gray = 1.0 - gray

    return np.clip(gray * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _select_best_frame_sharpness(frames_u8: np.ndarray) -> int:
    """Select sharpest frame (end-diastole) by Laplacian variance (minimal cardiac motion)."""
    if frames_u8.shape[0] <= 1:
        return 0
    scores = []
    for frame in frames_u8:
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        sharpness = float(laplacian.var())
        scores.append(sharpness)
    return int(np.argmax(scores))


def read_dicom_fluoro_series(path: str):
    """
    Read a fluoroscopy DICOM series and return normalized uint8 frames + metadata.

    Returns:
      {
        'frames_u8': uint8 array shaped (N, H, W),
        'meta': compatible metadata dict,
      }
    """
    if pydicom is None:
        raise RuntimeError('pydicom non disponible — installez-le : pip install pydicom')

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    samples_per_pixel = int(getattr(ds, 'SamplesPerPixel', 1) or 1)
    photometric = str(getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2') or 'MONOCHROME2')

    if arr.ndim == 2:
        frame_list = [arr]
    elif arr.ndim == 3 and samples_per_pixel > 1 and arr.shape[-1] in (3, 4):
        frame_list = [arr]
    elif arr.ndim == 3:
        frame_list = [arr[i] for i in range(arr.shape[0])]
    elif arr.ndim == 4:
        frame_list = [arr[i] for i in range(arr.shape[0])]
    else:
        raise ValueError(f'Format DICOM non pris en charge (shape={arr.shape})')

    frames_u8 = np.stack(
        [_normalize_dicom_frame(frame, photometric=photometric) for frame in frame_list],
        axis=0,
    )

    n_frames = int(frames_u8.shape[0])
    # Select sharpest frame (end-diastole, minimal cardiac motion) via Laplacian variance
    frame_idx = _select_best_frame_sharpness(frames_u8)
    img_uint8 = frames_u8[frame_idx]

    def _get(tag, default):
        val = getattr(ds, tag, None)
        if val is None:
            return default
        try:
            return float(str(val).split('\\')[0])
        except Exception:
            return default

    def _get_str(tag, default=''):
        val = getattr(ds, tag, None)
        return str(val).strip() if val is not None else default

    def _get_multi(tag):
        val = getattr(ds, tag, None)
        if val is None:
            return None
        try:
            return [float(v) for v in str(val).split('\\')]
        except Exception:
            return None

    def _get_private(group, elem, default=None):
        try:
            de = ds[group, elem]
            return float(de.value)
        except Exception:
            return default

    def _get_float(tag, default):
        val = getattr(ds, tag, None)
        if val is None: return default
        try: return float(str(val).split('\\')[0])
        except: return default

    # GE Specific
    # "AngleValuePArm" often corresponds to RAO/LAO. Positive is usually RAO, negative is LAO.
    # "AngleValueCArm" usually CRA/CAU. Positive CRA, negative CAU.
    arm_p = _get_private(0x0019, 0x1002, None)
    arm_c = _get_private(0x0019, 0x1003, None)

    # Standard Primary/Secondary angles (DICOM standard convention)
    # Primary = LAO/RAO (left/right anterior oblique)
    # Secondary = CRA/CAUD (cranial/caudal)
    primary = _get_float('PositionerPrimaryAngle', 0.0)
    secondary = _get_float('PositionerSecondaryAngle', 0.0)

    # Determine manufacturer to apply correct angle convention
    manufacturer = _get_str('Manufacturer', '').upper()
    use_ge_convention = 'GE' in manufacturer and (arm_c is not None or arm_p is not None)

    if use_ge_convention:
        # GE C-arms: convention swap (GE private tags encode angles differently)
        lao = arm_c if arm_c is not None else secondary
        cran = arm_p if arm_p is not None else primary
    else:
        # Standard DICOM convention (Siemens, Philips, Canon, etc.)
        lao = primary
        cran = secondary

    sid = _get_float('DistanceSourceToDetector', 1020.0)
    sod = _get_float('DistanceSourceToPatient', 510.0)
    mag = _get_float('EstimatedRadiographicMagnificationFactor', sid / sod if sod > 0 else 1.0)

    ips = _get_multi('ImagerPixelSpacing')
    pixel_mm = ips[0] if ips else 0.2

    rows = int(getattr(ds, 'Rows', img_uint8.shape[0]))
    cols = int(getattr(ds, 'Columns', img_uint8.shape[1]))
    
    # Check GE private zoom
    zoom_factor = _get_private(0x0019, 0x1018, 1.0)
    # The FOV in mm at detector is typically pixel_mm * rows
    if zoom_factor is None or zoom_factor <= 0:
        zoom_factor = 1.0
        
    fov_det = rows * pixel_mm / zoom_factor
    fov_mm = dict(
        fov_dim_mm=(fov_det, fov_det),
        fov_mm = fov_det / mag if mag > 0 else fov_det
    )
    
    intensifier_mm = _get_float('IntensifierSize', 0.0)

    cx = _get_private(0x0019, 0x1019, 0.0)
    cy = _get_private(0x0019, 0x101a, 0.0)

    # Calculate actual FOV geometry and centers taking zoom into account
    fov_shape = _get_str('FieldOfViewShape', '')
    fov_origin_raw = _get_multi('FieldOfViewOrigin')
    fov_origin = tuple(int(v) for v in fov_origin_raw) if fov_origin_raw else None

    table_angle = _get_float('TableAngle', 0.0)

    arm_l = _get_private(0x0019, 0x1001, None)

    # Extract centers
    cx = _get_private(0x0019, 0x1019, 0.0)
    cy = _get_private(0x0019, 0x101a, 0.0)

    table_angle = _get_float('TableAngle', 0.0)


    arm_l = _get_private(0x0019, 0x1001, None)
    arm_p = _get_private(0x0019, 0x1002, None)
    arm_c = _get_private(0x0019, 0x1003, None)

    patient_pos = _get_str('PatientPosition', 'HFS')

    shutter = {
        'left': int(_get('ShutterLeftVerticalEdge', 0)),
        'right': int(_get('ShutterRightVerticalEdge', cols)),
        'upper': int(_get('ShutterUpperHorizontalEdge', 0)),
        'lower': int(_get('ShutterLowerHorizontalEdge', rows)),
    }
    collimator = {
        'left': int(_get('CollimatorLeftVerticalEdge', 0)),
        'right': int(_get('CollimatorRightVerticalEdge', cols)),
        'upper': int(_get('CollimatorUpperHorizontalEdge', 0)),
        'lower': int(_get('CollimatorLowerHorizontalEdge', rows)),
    }

    manufacturer = _get_str('Manufacturer', '')
    model = _get_str('ManufacturerModelName', '')
    frame_time_ms = _get_float('FrameTime', 0.0)
    cine_rate_fps = _get_float('CineRate', _get_float('RecommendedDisplayFrameRate', 0.0))
    if (cine_rate_fps <= 0) and frame_time_ms > 0:
        cine_rate_fps = 1000.0 / frame_time_ms

    meta = dict(
        lao=lao,
        cran=cran,
        sid_mm=sid,
        sod_mm=sod,
        magnification=mag,
        pixel_mm=pixel_mm,
        fov_mm=fov_mm['fov_mm'],
        fov_dim_mm=fov_mm['fov_dim_mm'],
        intensifier_mm=intensifier_mm,
        fov_shape=fov_shape,
        fov_origin=fov_origin,
        table_angle=table_angle,
        arm_l=arm_l,
        arm_p=arm_p,
        arm_c=arm_c,
        patient_pos=patient_pos,
        shutter=shutter,
        collimator=collimator,
        rows=rows,
        cols=cols,
        n_frames=n_frames,
        frame_used=frame_idx + 1,
        frame_time_ms=frame_time_ms,
        cine_rate_fps=cine_rate_fps,
        manufacturer=manufacturer,
        model=model,
    )
    return {'frames_u8': frames_u8, 'meta': meta}


def read_dicom_fluoro(path: str, frame_index: int = None):
    """
    Read one fluoroscopy frame and extract C-arm geometry.

    Returns (img_uint8, meta) where meta contains geometry and acquisition fields.
    """
    series = read_dicom_fluoro_series(path)
    frames_u8 = series['frames_u8']
    meta = dict(series['meta'])
    idx = int(meta.get('frame_used', 1)) - 1 if frame_index is None else int(frame_index)
    idx = max(0, min(idx, frames_u8.shape[0] - 1))
    meta['frame_used'] = idx + 1
    return frames_u8[idx], meta


def read_metadata_csv(path: str):
    """
    Read DICOM metadata CSV exported by 3D Slicer.

    Expected columns are typically Tag, Name, Value, VR, Length.
    Returns a metadata dict compatible with read_dicom_fluoro().
    """
    df = pd.read_csv(path)

    if _re.match(r'\[?[0-9a-fA-F]{8}\]?$', str(df.columns[0]).strip()):
        df = pd.read_csv(
            path,
            header=None,
            names=['Tag', 'Name', 'Value', 'VR', 'Length'][:None],
        )
        if len(df.columns) < 3:
            raise ValueError(f'CSV metadata : au moins 3 colonnes attendues, {len(df.columns)} trouvées')

    cols_lower = {c.strip().lower(): c for c in df.columns}
    name_col = cols_lower.get('name', cols_lower.get('keyword', df.columns[1]))
    val_col = cols_lower.get('value', df.columns[2])

    lookup = {}
    for _, row in df.iterrows():
        key = str(row[name_col]).strip()
        val = str(row[val_col]).strip()
        if key:
            lookup[key] = val

    def _f(key, default=0.0):
        v = lookup.get(key)
        if v is None:
            return default
        try:
            return float(v.split('\\')[0].split(',')[0].strip())
        except Exception:
            return default

    def _flist(key):
        v = lookup.get(key)
        if v is None:
            return None
        try:
            parts = v.replace('\\', ',').split(',')
            return [float(p.strip()) for p in parts if p.strip()]
        except Exception:
            return None

    primary = _f('PositionerPrimaryAngle', 0.0)
    secondary = _f('PositionerSecondaryAngle', 0.0)

    # Determine manufacturer to apply correct angle convention
    # CSV may not have GE private tags, so we rely on Manufacturer field only
    manufacturer = str(lookup.get('Manufacturer', '')).upper()
    use_ge_convention = 'GE' in manufacturer

    if use_ge_convention:
        # GE convention: Primary = CRA/CAUD, Secondary = LAO/RAO (reversed from standard)
        lao = secondary
        cran = primary
    else:
        # Standard DICOM convention: Primary = LAO/RAO, Secondary = CRA/CAUD
        lao = primary
        cran = secondary
    sid = _f('DistanceSourceToDetector', 1020.0)
    sod = _f('DistanceSourceToPatient', 510.0)
    mag = _f('EstimatedRadiographicMagnificationFactor', sid / sod if sod > 0 else 1.0)

    ips = _flist('ImagerPixelSpacing')
    pixel_mm = ips[0] if ips else 0.2

    rows = int(_f('Rows', 1000))
    cols = int(_f('Columns', 1000))
    fov_dim = _flist('FieldOfViewDimensions')
    if fov_dim:
        fov_dim_mm = tuple(fov_dim)
        fov_mm = fov_dim_mm[0] * (sod / sid) if sid > 0 else fov_dim_mm[0]
    else:
        fov_mm = DEFAULT_FOV_MM
        fov_det = fov_mm * (sid / sod) if sid > 0 and sod > 0 else (DEFAULT_FOV_MM * 2.0)
        fov_dim_mm = (fov_det, fov_det)

    intensifier_mm = _f('IntensifierSize', 0.0)
    if not np.isfinite(fov_mm) or fov_mm <= 0:
        fov_mm = DEFAULT_FOV_MM

    table_angle = _f('TableAngle', 0.0)
    arm_l = _f('AngleValueLArm', None) if 'AngleValueLArm' in lookup else None
    arm_p = _f('AngleValuePArm', None) if 'AngleValuePArm' in lookup else None
    arm_c = _f('AngleValueCArm', None) if 'AngleValueCArm' in lookup else None

    patient_pos = lookup.get('PatientPosition', 'HFS')

    shutter = {
        'left': int(_f('ShutterLeftVerticalEdge', 0)),
        'right': int(_f('ShutterRightVerticalEdge', cols)),
        'upper': int(_f('ShutterUpperHorizontalEdge', 0)),
        'lower': int(_f('ShutterLowerHorizontalEdge', rows)),
    }
    collimator = {
        'left': int(_f('CollimatorLeftVerticalEdge', 0)),
        'right': int(_f('CollimatorRightVerticalEdge', cols)),
        'upper': int(_f('CollimatorUpperHorizontalEdge', 0)),
        'lower': int(_f('CollimatorLowerHorizontalEdge', rows)),
    }

    n_frames = int(_f('NumberOfFrames', 1))
    frame_used = int(_f('RepresentativeFrameNumber', (n_frames + 1) // 2))

    manufacturer = lookup.get('Manufacturer', '')
    model = lookup.get('ManufacturerModelName', '')
    fov_shape = lookup.get('FieldOfViewShape', '')
    fov_origin_raw = _flist('FieldOfViewOrigin')
    fov_origin = tuple(int(v) for v in fov_origin_raw) if fov_origin_raw else None

    return dict(
        lao=lao,
        cran=cran,
        sid_mm=sid,
        sod_mm=sod,
        magnification=mag,
        pixel_mm=pixel_mm,
        fov_mm=fov_mm,
        fov_dim_mm=fov_dim_mm,
        intensifier_mm=intensifier_mm,
        fov_shape=fov_shape,
        fov_origin=fov_origin,
        table_angle=table_angle,
        arm_l=arm_l,
        arm_p=arm_p,
        arm_c=arm_c,
        patient_pos=patient_pos,
        shutter=shutter,
        collimator=collimator,
        rows=rows,
        cols=cols,
        n_frames=n_frames,
        frame_used=frame_used,
        manufacturer=manufacturer,
        model=model,
    )
