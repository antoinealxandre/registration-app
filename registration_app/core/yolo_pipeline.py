"""
core/yolo_pipeline.py
Pipeline YOLO pour detection automatique de vertebres.
Filtre automatiquement les classes non-vertebres (ex: scoliosis spine).
"""

import numpy as np
import cv2

_YOLO_MODEL = None


def load_yolo_model(path: str):
    global _YOLO_MODEL
    from ultralytics import YOLO
    _YOLO_MODEL = YOLO(path)
    return _YOLO_MODEL


def is_model_loaded() -> bool:
    return _YOLO_MODEL is not None


def fluoro_to_xray(img: np.ndarray, gamma: float = 1.4,
                   clahe_clip: float = 2.5, unsharp_strength: float = 0.5,
                   invert: bool = False, contrast: float = 1.0) -> np.ndarray:
    """Transforme une fluoroscopie en pseudo-Rx."""
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    gray = img.copy()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)
    if abs(contrast - 1.0) > 0.01:
        mean = gray.mean()
        gray = np.clip((gray.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)
    gray = cv2.bilateralFilter(gray, 7, 40, 40)
    inv_gamma = 1.0 / max(0.01, gamma)
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** inv_gamma * 255
    gray = cv2.LUT(gray, lut.astype(np.uint8))
    clahe = cv2.createCLAHE(clipLimit=max(0.1, clahe_clip), tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    if unsharp_strength > 0:
        blurred = cv2.GaussianBlur(gray, (0, 0), 1.5)
        gray = cv2.addWeighted(gray, 1.0 + unsharp_strength,
                               blurred, -unsharp_strength, 0)
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    hist_eq = cv2.equalizeHist(gray)
    gray = cv2.addWeighted(gray, 0.7, hist_eq, 0.3, 0)
    return gray.astype(np.uint8)


def detection_points(det: dict) -> np.ndarray:
    """Retourne les sommets XY d'une detection sous forme int32 (N, 2)."""
    pts = det.get('points')
    if pts is not None and len(pts) >= 3:
        arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    else:
        x1 = float(det.get('x1', 0))
        y1 = float(det.get('y1', 0))
        x2 = float(det.get('x2', x1))
        y2 = float(det.get('y2', y1))
        arr = np.array([
            (x1, y1),
            (x2, y1),
            (x2, y2),
            (x1, y2),
        ], dtype=np.float32)
    return np.rint(arr).astype(np.int32)


def normalize_detection(det: dict, w: int = None, h: int = None) -> dict:
    """Normalise une detection en representation polygonale + bbox derivee."""
    out = dict(det)
    pts = detection_points(det).astype(np.float32)
    if pts.shape[0] < 3:
        return out
    if w is not None:
        pts[:, 0] = np.clip(pts[:, 0], 0, max(0, int(w) - 1))
    if h is not None:
        pts[:, 1] = np.clip(pts[:, 1], 0, max(0, int(h) - 1))
    pts_i = np.rint(pts).astype(np.int32)
    x1 = int(np.min(pts_i[:, 0]))
    y1 = int(np.min(pts_i[:, 1]))
    x2 = int(np.max(pts_i[:, 0]))
    y2 = int(np.max(pts_i[:, 1]))
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    area = float(abs(cv2.contourArea(pts.astype(np.float32))))
    out.update({
        'points': [(int(x), int(y)) for x, y in pts_i],
        'shape': det.get('shape') or ('quad' if len(pts_i) == 4 else 'polygon'),
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'cx': cx,
        'cy': cy,
        'area': area,
    })
    return out


def normalize_detections(detections: list, w: int = None, h: int = None) -> list:
    return [normalize_detection(det, w=w, h=h) for det in detections]


def detection_bounds(det: dict):
    det_n = normalize_detection(det)
    return det_n['x1'], det_n['y1'], det_n['x2'], det_n['y2']


def detection_center(det: dict):
    det_n = normalize_detection(det)
    return float(det_n['cx']), float(det_n['cy'])


def sort_detections_vertical(detections: list) -> list:
    return sorted(detections, key=lambda det: (detection_center(det)[1], detection_center(det)[0]))


def scale_detection(det: dict, sx: float, sy: float = None,
                    w: int = None, h: int = None) -> dict:
    """Met a l'echelle une detection polygonale."""
    sy = sx if sy is None else sy
    pts = detection_points(det).astype(np.float32)
    pts[:, 0] *= float(sx)
    pts[:, 1] *= float(sy)
    out = dict(det)
    out['points'] = [(float(x), float(y)) for x, y in pts]
    return normalize_detection(out, w=w, h=h)


def make_manual_detection(points: list, cls_name: str = 'manual',
                          conf: float = 1.0, shape: str = 'polygon') -> dict:
    """Construit une detection manuelle a partir d'une liste de points."""
    det = {
        'points': [(int(x), int(y)) for x, y in points],
        'cls_name': cls_name,
        'conf': float(conf),
        'shape': shape,
        'manual': True,
    }
    return normalize_detection(det)


def detect_vertebrae(img: np.ndarray, conf: float = 0.25,
                     iou: float = 0.45, imgsz: int = 288,
                     preprocess: bool = True,
                     preprocess_params: dict = None) -> dict:
    """
    Detecte les vertebres (filtre les autres classes comme 'scoliosis spine').
    Retourne {boxes, mask, infer_img, n_detections}.
    boxes triées par position verticale (haut en bas).
    """
    if _YOLO_MODEL is None:
        raise RuntimeError('Aucun modele YOLO charge.')

    if img.dtype != np.uint8:
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    else:
        img_u8 = img.copy()
    if img_u8.ndim == 3:
        if img_u8.shape[2] == 1:
            img_u8 = img_u8[:, :, 0]
        else:
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

    if preprocess:
        pp = preprocess_params or {}
        infer_img = fluoro_to_xray(
            img_u8,
            gamma=pp.get('gamma', 1.4),
            clahe_clip=pp.get('clahe_clip', 2.5),
            unsharp_strength=pp.get('unsharp_strength', 0.5),
            invert=pp.get('invert', False),
            contrast=pp.get('contrast', 1.0))
    else:
        infer_img = img_u8

    img_rgb = cv2.cvtColor(infer_img, cv2.COLOR_GRAY2RGB)
    effective_imgsz = max(infer_img.shape[:2]) if imgsz == 0 else max(32, imgsz)

    results = _YOLO_MODEL.predict(img_rgb, conf=max(0.01, conf),
                                  iou=max(0.01, iou), imgsz=effective_imgsz,
                                  verbose=False)

    h, w = infer_img.shape[:2]
    boxes_list = []

    if results:
        r = results[0]
        names = _YOLO_MODEL.names if hasattr(_YOLO_MODEL, 'names') else {}
        if r.boxes is not None:
            xyxy = r.boxes.xyxy.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            for i, box in enumerate(xyxy):
                cid = cls_ids[i]
                cls_name = names.get(int(cid), str(cid))
                if 'vertebr' not in cls_name.lower():
                    continue
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                boxes_list.append(normalize_detection({
                    'x1': int(x1),
                    'y1': int(y1),
                    'x2': int(x2),
                    'y2': int(y2),
                    'points': [(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                    'conf': float(confs[i]),
                    'cls_id': int(cid),
                    'cls_name': cls_name,
                    'shape': 'quad',
                }, w=w, h=h))

    boxes_list = sort_detections_vertical(boxes_list)
    mask = boxes_to_mask(boxes_list, h, w)

    return {'boxes': boxes_list, 'mask': mask,
            'infer_img': infer_img, 'n_detections': len(boxes_list)}


def boxes_to_mask(boxes: list, h: int, w: int) -> np.ndarray:
    """Construit un masque binaire uint8 a partir d'une liste de formes."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for det in boxes:
        pts = detection_points(det)
        if pts.shape[0] < 3:
            continue
        pts = normalize_detection({'points': pts.tolist()}, w=w, h=h)['points']
        cv2.fillPoly(mask, [np.asarray(pts, dtype=np.int32)], 255)
    return mask


def draw_detections(img: np.ndarray, boxes: list) -> np.ndarray:
    """Dessine les detections sur l'image — retourne RGB uint8."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img.copy()
    if vis.dtype != np.uint8:
        vis = (np.clip(vis, 0, 1) * 255).astype(np.uint8) if vis.max() <= 1.0 else vis.astype(np.uint8)
    palette = [(80, 220, 130), (255, 100, 100), (100, 180, 255),
               (255, 200, 80), (200, 130, 255), (130, 255, 200)]
    for idx, det in enumerate(boxes):
        color = palette[idx % len(palette)]
        pts = detection_points(det)
        x1, y1, _, _ = detection_bounds(det)
        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, color, 2, lineType=cv2.LINE_AA)
        label = f"V{idx+1} {det.get('conf', 0.0):.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(th + 6, y1)
        cv2.rectangle(vis, (x1, top - th - 6), (x1 + tw + 4, top), color, -1)
        cv2.putText(vis, label, (x1 + 2, top - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis
