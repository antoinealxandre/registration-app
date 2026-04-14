"""Background workers for DRR generation, YOLO detection, and registration."""

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

from core.drr_generator import generate_drr, project_mask_3d
from core.registration import register, register_elastic
from core.yolo_pipeline import detect_vertebrae, is_model_loaded as yolo_ready
from ui.theme import AUTO_PIPELINE_FOV_MM


def _match_centroids(boxes_drr: list, boxes_fl: list, img_size: int):
    """
    Pair each DRR box with the nearest fluoroscopy box centroid.

    Threshold is 30% of image size. Returns (matched_drr, matched_fl).
    If no pair is found, returns all boxes from both sides.
    """
    thr = img_size * 0.30

    def _cx(box):
        return ((box['x1'] + box['x2']) / 2.0, (box['y1'] + box['y2']) / 2.0)

    used = set()
    matched_drr, matched_fl = [], []
    for drr_box in boxes_drr:
        cd = _cx(drr_box)
        best_d, best_j = float('inf'), -1
        for j, fl_box in enumerate(boxes_fl):
            if j in used:
                continue
            cf = _cx(fl_box)
            d = ((cd[0] - cf[0]) ** 2 + (cd[1] - cf[1]) ** 2) ** 0.5
            if d < best_d:
                best_d, best_j = d, j
        if best_j >= 0 and best_d <= thr:
            matched_drr.append(drr_box)
            matched_fl.append(boxes_fl[best_j])
            used.add(best_j)

    if not matched_drr:
        return boxes_drr[:], boxes_fl[:]
    return matched_drr, matched_fl


class WorkerThread(QThread):
    progress = pyqtSignal(int, str)
    result = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, task, kw):
        super().__init__()
        self.task = task
        self.kw = kw

    def run(self):
        try:
            {
                'drr': self._drr,
                'register': self._reg,
                'yolo_detect': self._yolo_detect,
                'auto_pipeline': self._auto_pipeline,
                'auto_phase2': self._auto_phase2,
            }[self.task]()
        except Exception as ex:
            import traceback
            self.error.emit(f'{ex}\n{traceback.format_exc()}')

    def _yolo_detect(self):
        kw = self.kw
        self.progress.emit(10, f'Detection YOLO ({kw["target"]})...')
        det = detect_vertebrae(
            kw['img'],
            conf=kw.get('conf', 0.25),
            iou=kw.get('iou', 0.45),
            imgsz=kw.get('imgsz', 288),
            preprocess=kw.get('preprocess', True),
            preprocess_params=kw.get('pp', {}),
        )
        self.progress.emit(100, f"{det['n_detections']} vertebre(s) - {kw['target']}")
        self.result.emit({'task': 'yolo_detect', 'target': kw['target'], **det})

    def _drr(self):
        def pcb(pct, msg):
            self.progress.emit(pct, msg)

        pcb(5, 'Generation DRR (DiffDRR cone-beam)...')
        ct_path = self.kw['ct_path']
        renderer = self.kw.get('renderer', 'siddon')
        drr = generate_drr(
            ct_path=ct_path,
            lao_deg=self.kw['lao_deg'],
            cran_deg=self.kw['cran_deg'],
            table_angle=self.kw.get('table_angle', 0.0),
            output_size=self.kw['output_size'],
            sid_mm=self.kw.get('sid_mm', 1020.0),
            sod_mm=self.kw.get('sod_mm', 510.0),
            fov_mm=self.kw.get('fov_mm'),
            renderer=renderer,
            progress_cb=pcb,
        )
        pcb(80, 'Projection segmentations...')
        masks_out = {}
        ct_aff = self.kw.get('ct_aff')
        for name, mask in self.kw.get('masks', {}).items():
            if mask is None or mask.sum() == 0:
                continue
            masks_out[name] = project_mask_3d(
                mask_3d=mask,
                ct_affine=ct_aff,
                ct_path=ct_path,
                lao_deg=self.kw['lao_deg'],
                cran_deg=self.kw['cran_deg'],
                table_angle=self.kw.get('table_angle', 0.0),
                output_size=self.kw['output_size'],
                sid_mm=self.kw.get('sid_mm', 1020.0),
                sod_mm=self.kw.get('sod_mm', 510.0),
                fov_mm=self.kw.get('fov_mm'),
                renderer=renderer,
            )
        pcb(100, 'DRR pret')
        self.result.emit({'drr': drr, 'masks': masks_out})

    def _reg(self):
        elastic = self.kw.get('elastic', False)

        def cb(fraction, iou_val):
            if elastic:
                stage = 'Rigide' if fraction < 0.5 else 'Elastique'
                self.progress.emit(int(5 + fraction * 90), f'{stage} - IoU={iou_val:.3f}')
            else:
                self.progress.emit(int(5 + fraction * 90), f'Optimisation - IoU={iou_val:.3f}')

        if elastic:
            res = register_elastic(
                mask_moving=self.kw['moving'],
                mask_fixed=self.kw['fixed'],
                progress_cb=cb,
            )
        else:
            res = register(
                mask_moving=self.kw['moving'],
                mask_fixed=self.kw['fixed'],
                progress_cb=cb,
            )
        self.progress.emit(100, f"IoU={res['iou']:.3f}")
        self.result.emit(res)

    def _auto_pipeline(self):
        kw = self.kw
        ct_path = kw['ct_path']
        fluoro = kw['fluoro']
        reg_size = kw['output_size']
        lao_deg = kw['lao_deg']
        cran_deg = kw['cran_deg']
        table_angle = kw.get('table_angle', 0.0)
        yolo_kw = kw.get('yolo_kw', {})

        geom_kw = dict(lao_deg=lao_deg, cran_deg=cran_deg,
                       table_angle=table_angle, output_size=reg_size)
        for k in ('sid_mm', 'sod_mm', 'fov_mm'):
            if k in kw and kw[k] is not None:
                geom_kw[k] = kw[k]

        if not yolo_ready():
            raise RuntimeError('Chargez un modele YOLO (.pt) d\'abord.')

        self.progress.emit(5, f'Generation DRR (FOV={geom_kw.get("fov_mm", AUTO_PIPELINE_FOV_MM):.0f} mm)...')
        drr_image = generate_drr(
            ct_path=ct_path,
            renderer='siddon',
            progress_cb=lambda p, m: self.progress.emit(5 + int(p * 0.18), m),
            **geom_kw,
        )

        all_proj_masks = {}
        seg_masks = kw.get('seg_masks', {})
        if seg_masks:
            self.progress.emit(23, f'Projection de {len(seg_masks)} segmentation(s)...')
            for seg_name, seg_vol in seg_masks.items():
                if seg_vol is None or seg_vol.sum() == 0:
                    continue
                all_proj_masks[seg_name] = project_mask_3d(
                    mask_3d=seg_vol,
                    ct_affine=kw.get('ct_aff'),
                    ct_path=ct_path,
                    lao_deg=lao_deg,
                    cran_deg=cran_deg,
                    table_angle=table_angle,
                    output_size=reg_size,
                    sid_mm=geom_kw.get('sid_mm', 1020.0),
                    sod_mm=geom_kw.get('sod_mm', 510.0),
                    fov_mm=geom_kw.get('fov_mm'),
                )

        self.progress.emit(25, 'YOLO sur le DRR...')
        drr_u8 = (np.clip(drr_image, 0, 1) * 255).astype(np.uint8)
        drr_rs = cv2.resize(drr_u8, (reg_size, reg_size), interpolation=cv2.INTER_LANCZOS4)
        det_drr = detect_vertebrae(
            drr_rs,
            conf=yolo_kw.get('conf', 0.25),
            iou=yolo_kw.get('iou', 0.45),
            imgsz=yolo_kw.get('imgsz', 288),
            preprocess=False,
        )
        boxes_drr = det_drr['boxes']

        self.progress.emit(38, 'YOLO sur la fluoroscopie...')
        yolo_size = 1024
        fl_u8 = (np.clip(fluoro, 0, 1) * 255).astype(np.uint8)
        fl_rs = cv2.resize(fl_u8, (yolo_size, yolo_size), interpolation=cv2.INTER_LANCZOS4)
        det_fl = detect_vertebrae(
            fl_rs,
            conf=yolo_kw.get('conf', 0.25),
            iou=yolo_kw.get('iou', 0.45),
            imgsz=yolo_kw.get('imgsz', 288),
            preprocess=True,
            preprocess_params=yolo_kw.get('pp', {}),
        )
        scale_f = reg_size / yolo_size
        boxes_fl = [
            {
                'x1': int(b['x1'] * scale_f),
                'y1': int(b['y1'] * scale_f),
                'x2': int(b['x2'] * scale_f),
                'y2': int(b['y2'] * scale_f),
                'conf': b['conf'],
                'cls_name': b.get('cls_name', ''),
            }
            for b in det_fl['boxes']
        ]

        if not boxes_drr or not boxes_fl:
            self.progress.emit(50, 'Detection insuffisante - selection manuelle...')
            self.result.emit({
                '_phase': 'select_vertebrae',
                'drr_image': drr_image,
                'all_proj_masks': all_proj_masks,
                'drr_boxes': boxes_drr,
                'det_fl': det_fl,
                'det_drr': det_drr,
                'boxes_fl': boxes_fl,
                'reg_size': reg_size,
            })
            return

        matched_drr, matched_fl = _match_centroids(boxes_drr, boxes_fl, reg_size)
        self.progress.emit(48, f'{len(matched_drr)} paire(s) - recalage...')

        mask_drr = np.zeros((reg_size, reg_size), dtype=np.float32)
        for b in matched_drr:
            mask_drr[b['y1']:b['y2'], b['x1']:b['x2']] = 1.0
        mask_fl = np.zeros((reg_size, reg_size), dtype=np.float32)
        for b in matched_fl:
            mask_fl[b['y1']:b['y2'], b['x1']:b['x2']] = 1.0

        use_elastic = kw.get('elastic', True)
        if use_elastic:
            self.progress.emit(50, 'Recalage elastique (rigide + FFD)...')

            def _cb(frac, iou_val):
                stage = 'Rigide' if frac < 0.5 else 'Elastique'
                self.progress.emit(50 + int(frac * 44), f'{stage} - IoU={iou_val:.3f}')

            res = register_elastic(mask_moving=mask_drr, mask_fixed=mask_fl, progress_cb=_cb)
        else:
            def _cb(frac, iou_val):
                self.progress.emit(50 + int(frac * 44), f'Recalage IoU={iou_val:.3f}')

            res = register(mask_moving=mask_drr, mask_fixed=mask_fl, progress_cb=_cb)

        res.update(
            _auto=True,
            _phase='done',
            drr_image=drr_image,
            proj_masks=all_proj_masks,
            mask_fl=mask_fl,
            mask_drr=mask_drr,
            n_fluoro_sel=len(matched_fl),
            n_drr_sel=len(matched_drr),
        )
        self.progress.emit(100, f"Termine - IoU={res['iou']:.3f}")
        self.result.emit(res)

    def _auto_phase2(self):
        kw = self.kw
        boxes_fl = kw['boxes_fl']
        boxes_drr = kw['boxes_drr']
        reg_size = kw['reg_size']
        drr_image = kw['drr_image']
        all_proj_masks = kw['all_proj_masks']

        n_fl = len(boxes_fl)
        n_drr = len(boxes_drr)

        if n_fl == 0:
            raise RuntimeError('Aucune vertebre fluoro selectionnee.')
        if n_drr == 0:
            raise RuntimeError('Aucune vertebre DRR selectionnee.')

        self.progress.emit(55, 'Construction des masques...')

        mask_drr = np.zeros((reg_size, reg_size), dtype=np.float32)
        for b in boxes_drr:
            mask_drr[b['y1']:b['y2'], b['x1']:b['x2']] = 1.0

        mask_fl = np.zeros((reg_size, reg_size), dtype=np.float32)
        for b in boxes_fl:
            mask_fl[b['y1']:b['y2'], b['x1']:b['x2']] = 1.0

        self.progress.emit(58, 'Recalage elastique (rigide + FFD)...')

        def reg_cb(frac, iou_val):
            stage = 'Rigide' if frac < 0.5 else 'Elastique'
            self.progress.emit(58 + int(frac * 35), f'{stage} - IoU={iou_val:.4f}')

        res = register_elastic(mask_moving=mask_drr, mask_fixed=mask_fl, progress_cb=reg_cb)

        self.progress.emit(95, f"Recalage termine - IoU={res['iou']:.4f}  Dice={res['dice']:.4f}")

        res['_auto'] = True
        res['_phase'] = 'done'
        res['drr_image'] = drr_image
        res['proj_masks'] = all_proj_masks
        res['n_fluoro_sel'] = n_fl
        res['n_drr_sel'] = n_drr
        res['mask_fl'] = mask_fl
        res['mask_drr'] = mask_drr
        self.progress.emit(100, f"Pipeline termine - IoU={res['iou']:.4f}  Dice={res['dice']:.4f}")
        self.result.emit(res)
