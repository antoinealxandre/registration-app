"""
core/yolo_pipeline.py
Pipeline YOLO pour détection automatique de vertèbres.
Filtre automatiquement les classes non-vertèbres (ex: scoliosis spine).
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
                   invert: bool = False) -> np.ndarray:
    """Transforme une fluoroscopie en pseudo-Rx."""
    if img.dtype != np.uint8:
        img = np.clip(img * 255, 0, 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    gray = img.copy()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)
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


def detect_vertebrae(img: np.ndarray, conf: float = 0.25,
                     iou: float = 0.45, imgsz: int = 288,
                     preprocess: bool = True,
                     preprocess_params: dict = None) -> dict:
    """
    Détecte les vertèbres (filtre les autres classes comme 'scoliosis spine').
    Retourne {boxes, mask, infer_img, n_detections}.
    boxes trié par position verticale (y1 croissant = haut en bas).
    """
    if _YOLO_MODEL is None:
        raise RuntimeError('Aucun modèle YOLO chargé.')

    if img.dtype != np.uint8:
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    else:
        img_u8 = img.copy()
    if img_u8.ndim == 3:
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

    if preprocess:
        pp = preprocess_params or {}
        infer_img = fluoro_to_xray(
            img_u8,
            gamma=pp.get('gamma', 1.4),
            clahe_clip=pp.get('clahe_clip', 2.5),
            unsharp_strength=pp.get('unsharp_strength', 0.5),
            invert=pp.get('invert', False))
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
                # Filtrer : ne garder que les classes contenant 'vertebr'
                if 'vertebr' not in cls_name.lower():
                    continue
                x1, y1, x2, y2 = box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                boxes_list.append({
                    'x1': int(x1), 'y1': int(y1),
                    'x2': int(x2), 'y2': int(y2),
                    'conf': float(confs[i]),
                    'cls_id': int(cid),
                    'cls_name': cls_name,
                })

    # Trier par position verticale (haut en bas)
    boxes_list.sort(key=lambda b: b['y1'])

    # Construire le masque à partir des boîtes filtrées
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes_list:
        cv2.rectangle(mask, (b['x1'], b['y1']), (b['x2'], b['y2']), 255, -1)

    return {'boxes': boxes_list, 'mask': mask,
            'infer_img': infer_img, 'n_detections': len(boxes_list)}


def boxes_to_mask(boxes: list, h: int, w: int) -> np.ndarray:
    """Construit un masque binaire uint8 à partir d'une liste de boîtes."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        cv2.rectangle(mask, (b['x1'], b['y1']), (b['x2'], b['y2']), 255, -1)
    return mask


def draw_detections(img: np.ndarray, boxes: list) -> np.ndarray:
    """Dessine les boîtes sur l'image — retourne RGB uint8."""
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if img.ndim == 2 else img.copy()
    if vis.dtype != np.uint8:
        vis = (np.clip(vis, 0, 1) * 255).astype(np.uint8) if vis.max() <= 1.0 else vis.astype(np.uint8)
    palette = [(80, 220, 130), (255, 100, 100), (100, 180, 255),
               (255, 200, 80), (200, 130, 255), (130, 255, 200)]
    for idx, det in enumerate(boxes):
        color = palette[idx % len(palette)]
        x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"V{idx+1} {det['conf']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return vis
