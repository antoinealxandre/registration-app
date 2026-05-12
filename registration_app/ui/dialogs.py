"""Dialogs and auxiliary views used by the registration UI."""

import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QDialog,
    QWidget,
    QLabel,
    QPushButton,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy,
    QCheckBox,
    QGroupBox,
    QSlider,
    QFrame,
    QFileDialog,
    QMessageBox,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from core.registration import apply_full_transform
from core.yolo_pipeline import (
    detection_bounds,
    detection_center,
    detection_points,
    make_manual_detection,
    normalize_detections,
    sort_detections_vertical,
)
from ui.theme import (
    STYLE,
    DARK_BG,
    PANEL_BG,
    CARD_BG,
    BORDER,
    BORDER2,
    ACCENT,
    ACCENT2,
    TEXT,
    TEXT_DIM,
    TEXT_MID,
    WARN,
    ERR,
    SEG_PALETTE as _SEG_PALETTE,
    YOLO_BOX_PALETTE as _YOLO_BOX_PALETTE,
)

def _img_to_rgb(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    elif out.dtype != np.uint8:
        out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return np.ascontiguousarray(out)


def _fit_rgb_to_label(img: np.ndarray, label_widget: QLabel):
    lw = max(64, label_widget.width())
    lh = max(64, label_widget.height())
    h, w = img.shape[:2]
    scale = min(lw / max(1, w), lh / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    img_r = np.ascontiguousarray(img_r)
    qi = QImage(img_r.data, nw, nh, nw * 3, QImage.Format_RGB888)
    label_widget.setPixmap(QPixmap.fromImage(qi).copy())


def _draw_detection(img: np.ndarray, det: dict, color: tuple,
                    selected: bool = True, label: str = None) -> np.ndarray:
    pts = detection_points(det).astype(np.int32).reshape(-1, 1, 2)
    x1, y1, x2, y2 = detection_bounds(det)
    col = color if selected else tuple(max(25, c // 3) for c in color)
    out = img
    if selected:
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        out = cv2.addWeighted(overlay, 0.16, out, 0.84, 0)
    cv2.polylines(out, [pts], True, col, 2 if selected else 1, cv2.LINE_AA)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        top = max(th + 6, y1)
        cv2.rectangle(out, (x1, top - th - 6), (x1 + tw + 8, top), col, -1)
        cv2.putText(out, label, (x1 + 4, top - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    if selected and len(pts) == 4:
        corner_len = min(16, max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 4))
        corners = pts.reshape(-1, 2)
        for i, p in enumerate(corners):
            pn = corners[(i + 1) % len(corners)]
            pp = corners[(i - 1) % len(corners)]
            d1 = pn - p
            d2 = pp - p
            n1 = max(1.0, float(np.linalg.norm(d1)))
            n2 = max(1.0, float(np.linalg.norm(d2)))
            q1 = tuple(np.rint(p + d1 / n1 * corner_len).astype(int))
            q2 = tuple(np.rint(p + d2 / n2 * corner_len).astype(int))
            p_t = tuple(int(v) for v in p)
            cv2.line(out, p_t, q1, col, 2, cv2.LINE_AA)
            cv2.line(out, p_t, q2, col, 2, cv2.LINE_AA)
    return out


class DetectionShapeCanvas(QLabel):
    detections_changed = pyqtSignal()
    selection_changed = pyqtSignal(int)
    hint_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 420)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._img = None
        self._detections = []
        self._selected = -1
        self._mode = 'select'
        self._draw_pts = []
        self._cursor_pos = None

    def set_data(self, img: np.ndarray, detections: list):
        self._img = img.copy()
        h, w = self._img.shape[:2]
        self._detections = sort_detections_vertical(normalize_detections(detections, w=w, h=h))
        self._selected = min(self._selected, len(self._detections) - 1) if self._detections else -1
        self._draw_pts = []
        self._refresh()
        self.detections_changed.emit()

    def get_detections(self):
        return list(self._detections)

    def set_mode(self, mode: str):
        self._mode = mode
        self._draw_pts = []
        self._refresh()
        if mode == 'polygon':
            self.hint_changed.emit('Cliquez pour poser les sommets. Double-cliquez ou revenez au premier point pour fermer.')
        else:
            self.hint_changed.emit('Cliquez une forme pour la sélectionner.')

    def set_selected_index(self, idx: int):
        self._selected = idx if 0 <= idx < len(self._detections) else -1
        self._refresh()
        self.selection_changed.emit(self._selected)

    def delete_selected(self):
        if not (0 <= self._selected < len(self._detections)):
            return
        self._detections.pop(self._selected)
        self._selected = min(self._selected, len(self._detections) - 1) if self._detections else -1
        self._refresh()
        self.selection_changed.emit(self._selected)
        self.detections_changed.emit()

    def clear_all(self):
        self._detections = []
        self._selected = -1
        self._draw_pts = []
        self._refresh()
        self.selection_changed.emit(-1)
        self.detections_changed.emit()

    def cancel_drawing(self):
        self._draw_pts = []
        self._refresh()

    def _w2i(self, qx, qy):
        if self._img is None:
            return 0, 0
        h, w = self._img.shape[:2]
        ww, wh = self.width(), self.height()
        sc = min(ww / max(1, w), wh / max(1, h))
        ox = (ww - w * sc) / 2.0
        oy = (wh - h * sc) / 2.0
        ix = int(np.clip((qx - ox) / max(sc, 1e-6), 0, w - 1))
        iy = int(np.clip((qy - oy) / max(sc, 1e-6), 0, h - 1))
        return ix, iy

    def _find_hit(self, ix, iy):
        for i in range(len(self._detections) - 1, -1, -1):
            pts = detection_points(self._detections[i]).astype(np.float32)
            if cv2.pointPolygonTest(pts, (float(ix), float(iy)), False) >= 0:
                return i
        return -1

    def _commit_polygon(self):
        if self._img is None or len(self._draw_pts) < 3:
            return
        det = make_manual_detection(self._draw_pts, cls_name='manual', conf=1.0)
        h, w = self._img.shape[:2]
        self._detections.append(det)
        self._detections = sort_detections_vertical(normalize_detections(self._detections, w=w, h=h))
        self._selected = -1
        for i, cur in enumerate(self._detections):
            if cur.get('points') == det.get('points'):
                self._selected = i
                break
        self._draw_pts = []
        self._refresh()
        self.selection_changed.emit(self._selected)
        self.detections_changed.emit()

    def _refresh(self):
        if self._img is None:
            self.clear()
            return
        img = _img_to_rgb(self._img)
        for i, det in enumerate(self._detections):
            color = _YOLO_BOX_PALETTE[i % len(_YOLO_BOX_PALETTE)]
            shape_txt = 'Quad' if len(detection_points(det)) == 4 else f'Poly {len(detection_points(det))}'
            label = f'V{i+1} {shape_txt}'
            img = _draw_detection(img, det, color, selected=(i == self._selected), label=label if i == self._selected else None)

        if self._mode == 'polygon' and self._draw_pts:
            pts_a = np.asarray(self._draw_pts, dtype=np.int32)
            draw_col = (255, 245, 245)
            if len(pts_a) >= 2:
                cv2.polylines(img, [pts_a.reshape(-1, 1, 2)], False, draw_col, 2, cv2.LINE_AA)
            for p in self._draw_pts:
                cv2.circle(img, tuple(p), 5, draw_col, -1)
            if self._cursor_pos:
                cv2.line(img, tuple(self._draw_pts[-1]), self._cursor_pos, draw_col, 1, cv2.LINE_AA)
                d0 = np.linalg.norm(np.array(self._cursor_pos, float) - np.array(self._draw_pts[0], float))
                close_col = (120, 255, 160) if d0 < 14 and len(self._draw_pts) >= 3 else (255, 245, 245)
                cv2.circle(img, tuple(self._draw_pts[0]), 8, close_col, 2)
                if len(self._draw_pts) >= 3 and d0 < 14:
                    cv2.putText(img, 'Fermer', (self._draw_pts[0][0] + 10, self._draw_pts[0][1] + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, close_col, 1, cv2.LINE_AA)

        _fit_rgb_to_label(img, self)

    def mousePressEvent(self, e):
        if self._img is None:
            return
        ix, iy = self._w2i(e.x(), e.y())
        self._cursor_pos = (ix, iy)
        if self._mode == 'select':
            if e.button() == Qt.LeftButton:
                self.set_selected_index(self._find_hit(ix, iy))
            return
        if e.button() == Qt.RightButton:
            if self._draw_pts:
                self._draw_pts.pop()
                self._refresh()
            return
        if e.button() != Qt.LeftButton:
            return
        if (self._draw_pts and len(self._draw_pts) >= 3 and
                np.linalg.norm(np.array((ix, iy), float) - np.array(self._draw_pts[0], float)) < 14):
            self._commit_polygon()
            return
        if not self._draw_pts or (ix, iy) != self._draw_pts[-1]:
            self._draw_pts.append((ix, iy))
        self._refresh()

    def mouseMoveEvent(self, e):
        if self._img is None:
            return
        self._cursor_pos = self._w2i(e.x(), e.y())
        self._refresh()

    def mouseDoubleClickEvent(self, e):
        if self._mode == 'polygon' and len(self._draw_pts) >= 3:
            self._commit_polygon()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh()


class DetectionShapeEditorDialog(QDialog):
    def __init__(self, image: np.ndarray, detections: list, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 760)
        self.setStyleSheet(STYLE)
        self._image = image
        self._build_ui()
        self._canvas.set_data(image, detections)
        self._canvas.set_mode('select')
        self._refresh_list()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        left = QVBoxLayout()
        self._canvas = DetectionShapeCanvas()
        self._canvas.setStyleSheet(f'background:{DARK_BG};border:1px solid {BORDER};border-radius:6px;')
        self._canvas.detections_changed.connect(self._refresh_list)
        self._canvas.selection_changed.connect(self._on_selection_changed)
        self._canvas.hint_changed.connect(self._set_hint)
        left.addWidget(self._canvas, 1)

        self._lbl_hint = QLabel('Cliquez une forme pour la sélectionner.')
        self._lbl_hint.setObjectName('dim')
        self._lbl_hint.setWordWrap(True)
        self._lbl_hint.setAlignment(Qt.AlignCenter)
        left.addWidget(self._lbl_hint)
        root.addLayout(left, 3)

        right_w = QWidget()
        right_w.setFixedWidth(340)
        right_w.setStyleSheet(f'background:{PANEL_BG};border:1px solid {BORDER};border-radius:6px;')
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(10, 10, 10, 10)
        right_l.setSpacing(8)

        title = QLabel('FORMES')
        title.setStyleSheet(f'color:{ACCENT};font-size:12px;font-weight:700;letter-spacing:1px;')
        right_l.addWidget(title)

        self._btn_select = QPushButton('Selection')
        self._btn_select.setCheckable(True)
        self._btn_poly = QPushButton('Polygone / quad')
        self._btn_poly.setCheckable(True)
        mode_row = QHBoxLayout()
        mode_row.setSpacing(4)
        mode_row.addWidget(self._btn_select, 1)
        mode_row.addWidget(self._btn_poly, 1)
        right_l.addLayout(mode_row)
        self._btn_select.clicked.connect(lambda: self._set_mode('select'))
        self._btn_poly.clicked.connect(lambda: self._set_mode('polygon'))

        tip = QLabel(
            'Polygone : cliquez pour poser les sommets. Un quadrilatere de 4 points '
            'permet de tracer des boxes obliques / parallelogrammes.'
        )
        tip.setObjectName('dim')
        tip.setWordWrap(True)
        right_l.addWidget(tip)

        action_row = QHBoxLayout()
        action_row.setSpacing(4)
        btn_delete = QPushButton('Supprimer')
        btn_delete.setObjectName('warn')
        btn_delete.clicked.connect(self._canvas.delete_selected)
        btn_clear = QPushButton('Tout effacer')
        btn_clear.setObjectName('danger')
        btn_clear.clicked.connect(self._canvas.clear_all)
        action_row.addWidget(btn_delete, 1)
        action_row.addWidget(btn_clear, 1)
        right_l.addLayout(action_row)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet('QScrollArea{background:transparent;border:none;}')
        self._list_inner = QWidget()
        self._list_inner.setStyleSheet('background:transparent;')
        self._list_layout = QVBoxLayout(self._list_inner)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(6)
        self._scroll.setWidget(self._list_inner)
        right_l.addWidget(self._scroll, 1)

        self._lbl_summary = QLabel('')
        self._lbl_summary.setObjectName('dim')
        self._lbl_summary.setAlignment(Qt.AlignCenter)
        right_l.addWidget(self._lbl_summary)

        btn_row = QHBoxLayout()
        btn_ok = QPushButton('Appliquer')
        btn_ok.setObjectName('success')
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton('Annuler')
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_ok, 1)
        btn_row.addWidget(btn_cancel)
        right_l.addLayout(btn_row)

        root.addWidget(right_w)

    def _set_hint(self, text):
        self._lbl_hint.setText(text)

    def _set_mode(self, mode):
        self._btn_select.setChecked(mode == 'select')
        self._btn_poly.setChecked(mode == 'polygon')
        self._canvas.set_mode(mode)

    def _on_selection_changed(self, idx):
        self._refresh_list()

    def _refresh_list(self):
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        detections = self._canvas.get_detections()
        selected_idx = self._canvas._selected
        if not detections:
            lbl = QLabel('Aucune forme')
            lbl.setObjectName('dim')
            lbl.setAlignment(Qt.AlignCenter)
            self._list_layout.addWidget(lbl)
        else:
            for i, det in enumerate(detections):
                color = _YOLO_BOX_PALETTE[i % len(_YOLO_BOX_PALETTE)]
                r, g, b = color
                x1, y1, x2, y2 = detection_bounds(det)
                cx, cy = detection_center(det)
                n_pts = len(detection_points(det))
                kind = 'Quad' if n_pts == 4 else f'Poly {n_pts}'
                btn = QPushButton(
                    f'V{i+1}  {kind}\nCentre ({int(cx)}, {int(cy)})  |  {x2 - x1}x{y2 - y1}px'
                )
                btn.setCheckable(True)
                btn.setChecked(i == selected_idx)
                btn.setStyleSheet(
                    f'QPushButton{{text-align:left;padding:8px;border-radius:6px;'
                    f'background:{CARD_BG};border:1px solid rgb({r},{g},{b});}}'
                    f'QPushButton:checked{{background:rgba({r},{g},{b},120);'
                    f'border:2px solid rgb({r},{g},{b});color:#fff;font-weight:700;}}'
                )
                btn.clicked.connect(lambda checked, idx=i: self._canvas.set_selected_index(idx))
                self._list_layout.addWidget(btn)
        self._list_layout.addStretch()
        self._lbl_summary.setText(f'{len(detections)} forme(s)')

    def get_detections(self):
        return self._canvas.get_detections()


class YoloDetectionPanel(QDialog):
    """
    Panneau de revue YOLO avec sélection des detections.
    """
    def __init__(self, det_result: dict, target: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Detections YOLO - {target.upper()}')
        self.resize(1120, 720)
        self.setStyleSheet(STYLE)
        self._target = target
        self._infer_img = det_result['infer_img']
        h, w = self._infer_img.shape[:2]
        self._boxes = sort_detections_vertical(normalize_detections(det_result.get('boxes', []), w=w, h=h))
        self._selected = list(range(len(self._boxes)))
        self._chk_list = []
        self._build_ui()
        self._rebuild_det_cards()
        self._render_image()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        left = QVBoxLayout()
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setMinimumSize(500, 500)
        self._img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img_label.setStyleSheet(f'background:{DARK_BG};border-radius:6px;border:2px solid {BORDER};')
        left.addWidget(self._img_label, 1)
        self._info_label = QLabel('')
        self._info_label.setStyleSheet(f'color:{ACCENT};font-size:12px;font-weight:600;background:transparent;')
        self._info_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self._info_label)
        root.addLayout(left, 3)

        right_w = QWidget()
        right_w.setFixedWidth(330)
        right_w.setStyleSheet(f'background:{PANEL_BG};border:1px solid {BORDER};border-radius:6px;')
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(10, 10, 10, 10)
        right_l.setSpacing(8)

        title = QLabel('DETECTIONS')
        title.setStyleSheet(f'color:{ACCENT};font-size:12px;font-weight:700;letter-spacing:1px;')
        right_l.addWidget(title)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        btn_all = QPushButton('Toutes')
        btn_all.clicked.connect(self._select_all)
        btn_none = QPushButton('Aucune')
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        right_l.addLayout(btn_row)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet('QScrollArea{background:transparent;border:none;}')
        self._inner = QWidget()
        self._inner.setStyleSheet('background:transparent;')
        self._inner_l = QVBoxLayout(self._inner)
        self._inner_l.setSpacing(6)
        self._inner_l.setContentsMargins(0, 0, 0, 0)
        self._scroll.setWidget(self._inner)
        right_l.addWidget(self._scroll, 1)

        self._lbl_summary = QLabel('')
        self._lbl_summary.setStyleSheet(f'color:{TEXT_MID};font-size:11px;')
        self._lbl_summary.setAlignment(Qt.AlignCenter)
        right_l.addWidget(self._lbl_summary)

        btn_row2 = QHBoxLayout()
        btn_ok = QPushButton('Appliquer')
        btn_ok.setObjectName('success')
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton('Annuler')
        btn_cancel.clicked.connect(self.reject)
        btn_row2.addWidget(btn_ok, 1)
        btn_row2.addWidget(btn_cancel)
        right_l.addLayout(btn_row2)

        root.addWidget(right_w)

    def _rebuild_det_cards(self):
        self._chk_list = []
        while self._inner_l.count():
            item = self._inner_l.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for i, box in enumerate(self._boxes):
            color = _YOLO_BOX_PALETTE[i % len(_YOLO_BOX_PALETTE)]
            self._inner_l.addWidget(self._make_det_card(i, box, color))
        self._inner_l.addStretch()
        n_sel = len(self._selected)
        total = len(self._boxes)
        self._lbl_summary.setText(f'{n_sel} selectionnee(s) / {total}')
        self._info_label.setText(f'{self._target.upper()}  -  {total} detection(s)')

    def _make_det_card(self, idx, box, color):
        r, g, b = color
        card = QWidget()
        card.setStyleSheet(
            f'background:{CARD_BG};border:1px solid rgb({r},{g},{b});border-radius:6px;'
        )
        cl = QVBoxLayout(card)
        cl.setContentsMargins(8, 6, 8, 6)
        cl.setSpacing(3)

        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        chk = QCheckBox()
        chk.setChecked(idx in self._selected)
        chk.toggled.connect(lambda checked, i=idx: self._on_toggle(i, checked))
        self._chk_list.append(chk)
        hdr.addWidget(chk)

        pts_n = len(detection_points(box))
        shape_txt = 'Quad' if pts_n == 4 else f'Poly {pts_n}'
        lbl_name = QLabel(f'V{idx+1}  {shape_txt}')
        lbl_name.setStyleSheet(f'color:rgb({r},{g},{b});font-size:13px;font-weight:700;')
        hdr.addWidget(lbl_name)

        conf_pct = int(float(box.get('conf', 0.0)) * 100)
        conf_color = ACCENT2 if conf_pct >= 70 else (WARN if conf_pct >= 40 else ERR)
        badge = QLabel(f'{conf_pct}%')
        badge.setFixedSize(42, 20)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            f'background:{conf_color};color:#fff;font-size:10px;font-weight:700;border-radius:10px;'
        )
        hdr.addWidget(badge)
        hdr.addStretch()
        cl.addLayout(hdr)

        cls_lbl = QLabel(box.get('cls_name', 'vertebra'))
        cls_lbl.setStyleSheet(f'color:{TEXT_MID};font-size:10px;')
        cl.addWidget(cls_lbl)

        x1, y1, x2, y2 = detection_bounds(box)
        cx, cy = detection_center(box)
        info = QLabel(f'Centre ({int(cx)}, {int(cy)})  |  {x2 - x1}x{y2 - y1} px')
        info.setStyleSheet(f'color:{TEXT_DIM};font-size:10px;')
        cl.addWidget(info)
        return card

    def _on_toggle(self, idx, checked):
        if checked and idx not in self._selected:
            self._selected.append(idx)
        elif not checked and idx in self._selected:
            self._selected.remove(idx)
        self._selected.sort()
        self._lbl_summary.setText(f'{len(self._selected)} selectionnee(s) / {len(self._boxes)}')
        self._render_image()

    def _select_all(self):
        for chk in self._chk_list:
            chk.setChecked(True)

    def _select_none(self):
        for chk in self._chk_list:
            chk.setChecked(False)

    def _render_image(self):
        img = _img_to_rgb(self._infer_img)
        for i, box in enumerate(self._boxes):
            color = _YOLO_BOX_PALETTE[i % len(_YOLO_BOX_PALETTE)]
            label = f'V{i+1} {float(box.get("conf", 0.0)):.0%}'
            img = _draw_detection(img, box, color, selected=(i in self._selected), label=label if i in self._selected else None)
        _fit_rgb_to_label(img, self._img_label)

    def get_selection(self):
        return sorted(self._selected)

    def get_selected_detections(self):
        return [self._boxes[i] for i in sorted(self._selected) if 0 <= i < len(self._boxes)]

    def get_detections(self):
        return list(self._boxes)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render_image()


# ══════════════════════════════════════════════════════════════════════════════
# Panneau dual : sélection vertèbres Fluoro (gauche) + DRR (droite)
# ══════════════════════════════════════════════════════════════════════════════

class DualYoloSelectionDialog(QDialog):
    def __init__(self, det_fl: dict, det_drr: dict,
                 boxes_fl: list, named_drr_boxes: list,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle('Selection des vertebres - Fluoro / DRR')
        self.resize(1500, 840)
        self.setStyleSheet(STYLE)

        h_fl, w_fl = det_fl['infer_img'].shape[:2]
        h_drr, w_drr = det_drr['infer_img'].shape[:2]
        self._det_fl = det_fl
        self._det_drr = det_drr
        self._boxes_fl = sort_detections_vertical(normalize_detections(boxes_fl, w=w_fl, h=h_fl))
        self._boxes_drr = sort_detections_vertical(normalize_detections(named_drr_boxes, w=w_drr, h=h_drr))
        self._sel_fl = list(range(len(self._boxes_fl)))
        self._sel_drr = list(range(len(self._boxes_drr)))
        self._side_ui = {}
        self._build_ui()
        self._rebuild_side('fl')
        self._rebuild_side('drr')

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        title = QLabel('Selectionnez les vertebres : Fluoro (gauche) / DRR (droite)')
        title.setStyleSheet(f'color:{ACCENT};font-size:13px;font-weight:700;background:transparent;padding:4px;')
        outer.addWidget(title)

        hint = QLabel('Selectionnez les detections a conserver pour le recalage.')
        hint.setObjectName('dim')
        hint.setWordWrap(True)
        outer.addWidget(hint)

        row = QHBoxLayout()
        row.setSpacing(0)
        row.addWidget(self._build_side('fl', 'FLUORO', self._det_fl['infer_img']), 1)
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f'color:{BORDER};margin:0px;width:2px;')
        row.addWidget(sep)
        row.addWidget(self._build_side('drr', 'DRR', self._det_drr['infer_img']), 1)
        outer.addLayout(row, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()
        btn_ok = QPushButton('Lancer le recalage')
        btn_ok.setObjectName('success')
        btn_ok.setFixedHeight(38)
        btn_ok.setMinimumWidth(170)
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton('Annuler')
        btn_cancel.setFixedHeight(38)
        btn_cancel.setMinimumWidth(120)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()
        outer.addLayout(btn_row)

    def _build_side(self, side_key, title_text, base_img):
        container = QWidget()
        vl = QVBoxLayout(container)
        vl.setContentsMargins(6, 0, 6, 0)
        vl.setSpacing(6)

        title = QLabel(title_text)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f'color:{ACCENT};font-size:11px;font-weight:700;letter-spacing:2px;padding:4px;')
        vl.addWidget(title)

        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setMinimumSize(320, 320)
        img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_label.setStyleSheet(f'background:{DARK_BG};border:1px solid {BORDER};border-radius:4px;')
        vl.addWidget(img_label, 1)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(4)
        btn_all = QPushButton('Toutes')
        btn_none = QPushButton('Aucune')
        btn_all.clicked.connect(lambda: self._set_all(side_key, True))
        btn_none.clicked.connect(lambda: self._set_all(side_key, False))
        ctrl_row.addWidget(btn_all)
        ctrl_row.addWidget(btn_none)
        vl.addLayout(ctrl_row)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(110)
        scroll.setStyleSheet(
            f'QScrollArea{{background:transparent;border:none;}}'
            f'QScrollBar:vertical{{width:6px;background:{PANEL_BG};}}'
            f'QScrollBar::handle:vertical{{background:{BORDER};border-radius:3px;}}'
        )
        inner_w = QWidget()
        inner_w.setStyleSheet('background:transparent;')
        inner_l = QHBoxLayout(inner_w)
        inner_l.setSpacing(4)
        inner_l.setContentsMargins(2, 2, 2, 2)
        scroll.setWidget(inner_w)
        vl.addWidget(scroll)

        summary = QLabel('')
        summary.setObjectName('dim')
        summary.setAlignment(Qt.AlignCenter)
        vl.addWidget(summary)

        self._side_ui[side_key] = {
            'img': img_label,
            'layout': inner_l,
            'summary': summary,
            'buttons': [],
            'base_img': base_img,
        }
        return container

    def _boxes_for(self, side_key):
        return self._boxes_fl if side_key == 'fl' else self._boxes_drr

    def _sel_for(self, side_key):
        return self._sel_fl if side_key == 'fl' else self._sel_drr

    def _set_sel_for(self, side_key, values):
        if side_key == 'fl':
            self._sel_fl = values
        else:
            self._sel_drr = values

    def _rebuild_side(self, side_key):
        ui = self._side_ui[side_key]
        layout = ui['layout']
        ui['buttons'] = []
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        boxes = self._boxes_for(side_key)
        sel_list = self._sel_for(side_key)
        for i, box in enumerate(boxes):
            color = _YOLO_BOX_PALETTE[i % len(_YOLO_BOX_PALETTE)]
            r, g, b = color
            pts_n = len(detection_points(box))
            kind = 'Quad' if pts_n == 4 else f'P{pts_n}'
            btn = QPushButton(f'V{i+1}\n{kind}')
            btn.setCheckable(True)
            btn.setChecked(i in sel_list)
            btn.setFixedHeight(34)
            btn.setMinimumWidth(52)
            btn.setFont(QFont('monospace', 9))

            def update_style(cur_btn=btn, rr=r, gg=g, bb=b):
                if cur_btn.isChecked():
                    cur_btn.setStyleSheet(
                        f'background-color:rgb({rr},{gg},{bb});'
                        f'color:white;border:2px solid rgb({rr},{gg},{bb});'
                        f'border-radius:4px;font-weight:700;'
                    )
                else:
                    cur_btn.setStyleSheet(
                        f'background-color:{CARD_BG};'
                        f'color:{TEXT_MID};border:1px solid {BORDER};'
                        f'border-radius:4px;'
                    )

            def on_toggled(checked, idx=i, cur_btn=btn):
                current = [j for j in self._sel_for(side_key) if j < len(self._boxes_for(side_key))]
                if checked and idx not in current:
                    current.append(idx)
                elif not checked and idx in current:
                    current.remove(idx)
                current.sort()
                self._set_sel_for(side_key, current)
                update_style()
                self._render_side(side_key)
                self._update_summary(side_key)

            btn.toggled.connect(on_toggled)
            update_style()
            ui['buttons'].append(btn)
            layout.addWidget(btn)

        layout.addStretch()
        self._update_summary(side_key)
        self._render_side(side_key)

    def _update_summary(self, side_key):
        boxes = self._boxes_for(side_key)
        sel_list = self._sel_for(side_key)
        self._side_ui[side_key]['summary'].setText(f'{len(sel_list)} selectionnee(s) / {len(boxes)}')

    def _set_all(self, side_key, state):
        for btn in self._side_ui[side_key]['buttons']:
            btn.setChecked(state)

    def _render_side(self, side_key):
        ui = self._side_ui[side_key]
        img = _img_to_rgb(ui['base_img'])
        boxes = self._boxes_for(side_key)
        sel_list = self._sel_for(side_key)
        for i, box in enumerate(boxes):
            color = _YOLO_BOX_PALETTE[i % len(_YOLO_BOX_PALETTE)]
            label = f'V{i+1} {float(box.get("conf", 0.0)):.0%}'
            img = _draw_detection(img, box, color, selected=(i in sel_list), label=label if i in sel_list else None)
        _fit_rgb_to_label(img, ui['img'])

    def get_selections(self):
        return sorted(self._sel_fl), sorted(self._sel_drr)

    def get_selected_detections(self):
        selected_fl = [self._boxes_fl[i] for i in sorted(self._sel_fl) if 0 <= i < len(self._boxes_fl)]
        selected_drr = [self._boxes_drr[i] for i in sorted(self._sel_drr) if 0 <= i < len(self._boxes_drr)]
        return selected_fl, selected_drr

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render_side('fl')
        self._render_side('drr')


class VertebralDetectionWindow(QDialog):
    """Fenêtre simplifiée : sélection des vertèbres par clic direct sur les images.
    Fluoro (gauche) + DRR (droite), rectangles cliquables, juste les pourcentages."""
    
    def __init__(self, det_fl: dict, det_drr: dict,
                 boxes_fl: list, boxes_drr: list,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle('Détection des vertèbres')
        self.resize(1300, 650)
        self.setStyleSheet(STYLE)

        h_fl, w_fl = det_fl['infer_img'].shape[:2]
        h_drr, w_drr = det_drr['infer_img'].shape[:2]
        self._det_fl = det_fl
        self._det_drr = det_drr
        self._boxes_fl = sort_detections_vertical(normalize_detections(boxes_fl, w=w_fl, h=h_fl))
        self._boxes_drr = sort_detections_vertical(normalize_detections(boxes_drr, w=w_drr, h=h_drr))
        self._sel_fl = set(range(len(self._boxes_fl)))
        self._sel_drr = set(range(len(self._boxes_drr)))
        self._side_ui = {}
        self._build_ui()
        self._render_side('fl')
        self._render_side('drr')

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        title = QLabel('Cliquez sur les rectangles pour les activer/désactiver')
        title.setStyleSheet(f'color:{ACCENT};font-size:13px;font-weight:700;background:transparent;padding:4px;')
        outer.addWidget(title)

        # Layout dual : Fluoro | DRR
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(self._build_side('fl', 'FLUORO', self._det_fl['infer_img']), 1)
        row.addWidget(self._build_side('drr', 'DRR', self._det_drr['infer_img']), 1)
        outer.addLayout(row, 1)

        # Boutons d'action
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()
        btn_ok = QPushButton('Valider')
        btn_ok.setObjectName('success')
        btn_ok.setFixedHeight(36)
        btn_ok.setMinimumWidth(140)
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton('Annuler')
        btn_cancel.setFixedHeight(36)
        btn_cancel.setMinimumWidth(120)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()
        outer.addLayout(btn_row)

    def _build_side(self, side_key, title_text, base_img):
        container = QWidget()
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(6)

        title = QLabel(title_text)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f'color:{ACCENT};font-size:11px;font-weight:700;letter-spacing:2px;padding:4px;')
        vl.addWidget(title)

        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setMinimumSize(300, 300)
        img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_label.setStyleSheet(f'background:{DARK_BG};border:1px solid {BORDER};border-radius:4px;')
        img_label.setMouseTracking(True)
        
        def on_click(e, sk=side_key):
            self._on_image_click(e, sk, img_label)
        
        img_label.mousePressEvent = on_click
        vl.addWidget(img_label, 1)

        summary = QLabel('')
        summary.setObjectName('dim')
        summary.setAlignment(Qt.AlignCenter)
        vl.addWidget(summary)

        self._side_ui[side_key] = {
            'img': img_label,
            'summary': summary,
            'base_img': base_img,
        }
        return container

    def _on_image_click(self, e, side_key, img_label):
        """Détecte quel rectangle a été cliqué et bascule sa sélection."""
        if not hasattr(self, '_side_ui') or side_key not in self._side_ui:
            return
        
        boxes = self._boxes_fl if side_key == 'fl' else self._boxes_drr
        base_img = self._side_ui[side_key]['base_img']
        h, w = base_img.shape[:2]
        
        # Conversion coordonnées widget -> image
        lw = img_label.width()
        lh = img_label.height()
        scale = min(lw / max(1, w), lh / max(1, h))
        offset_x = (lw - w * scale) / 2.0
        offset_y = (lh - h * scale) / 2.0
        ix = int((e.x() - offset_x) / max(scale, 1e-6))
        iy = int((e.y() - offset_y) / max(scale, 1e-6))
        
        # Test point dans polygone pour chaque rectangle
        for i, box in enumerate(boxes):
            pts = detection_points(box).astype(np.float32)
            if cv2.pointPolygonTest(pts, (float(ix), float(iy)), False) >= 0:
                sel_set = self._sel_fl if side_key == 'fl' else self._sel_drr
                if i in sel_set:
                    sel_set.discard(i)
                else:
                    sel_set.add(i)
                self._render_side(side_key)
                self._update_summary(side_key)
                return

    def _render_side(self, side_key):
        """Affiche l'image avec les rectangles et pourcentages."""
        ui = self._side_ui[side_key]
        img = _img_to_rgb(ui['base_img'])
        boxes = self._boxes_fl if side_key == 'fl' else self._boxes_drr
        sel_set = self._sel_fl if side_key == 'fl' else self._sel_drr
        
        for i, box in enumerate(boxes):
            color = _YOLO_BOX_PALETTE[i % len(_YOLO_BOX_PALETTE)]
            conf_pct = int(float(box.get('conf', 0.0)) * 100)
            label = f'{conf_pct}%'
            img = _draw_detection(img, box, color, selected=(i in sel_set), label=label)
        
        _fit_rgb_to_label(img, ui['img'])

    def _update_summary(self, side_key):
        boxes = self._boxes_fl if side_key == 'fl' else self._boxes_drr
        sel_set = self._sel_fl if side_key == 'fl' else self._sel_drr
        self._side_ui[side_key]['summary'].setText(f'{len(sel_set)} selectionnée(s) / {len(boxes)}')

    def get_selected_detections(self):
        """Retourne les détections sélectionnées."""
        selected_fl = [self._boxes_fl[i] for i in sorted(self._sel_fl) if 0 <= i < len(self._boxes_fl)]
        selected_drr = [self._boxes_drr[i] for i in sorted(self._sel_drr) if 0 <= i < len(self._boxes_drr)]
        return selected_fl, selected_drr

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render_side('fl')
        self._render_side('drr')


class SegOverlayWindow(QDialog):
    """Fenêtre optimisée : fluoroscopie recalée + segmentations, contrôles avancés."""

    def __init__(self, fluoro: np.ndarray, proj_masks: dict, result: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Segmentations 2D — Vue fluoroscopie recalée')
        self.resize(1200, 800)
        self.setStyleSheet(STYLE)
        self._fluoro = np.clip(fluoro, 0, 1).astype(np.float32)
        self._proj_masks = proj_masks
        self._result = result
        self._alpha_global = 0.40
        self._alpha_per_struct = {}
        self._zoom = 1.0
        self._pan_x, self._pan_y = 0, 0
        names = list(proj_masks.keys())
        self._colors = {n: _SEG_PALETTE[i % len(_SEG_PALETTE)] for i, n in enumerate(names)}
        self._chks: dict = {}
        self._opacity_sliders = {}
        self._line_width = 2
        self._full_image = None  # Cache pour les rendus sans zoom/pan
        self._build_ui()
        self._render()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(10)

        # ── Panneau gauche : image ────────────────────────────────────────────
        left = QVBoxLayout()
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setMinimumSize(480, 480)
        self._img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img_label.setStyleSheet(f'background:{DARK_BG};border-radius:4px;border:2px solid {BORDER};')
        self._img_label.wheelEvent = self._wheel_zoom
        self._img_label.mousePressEvent = self._pan_start
        self._img_label.mouseMoveEvent = self._pan_move
        self._img_label.mouseReleaseEvent = self._pan_end
        self._pan_active = False
        self._pan_start_pos = None
        left.addWidget(self._img_label, 1)

        # Zoom + reset
        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel('Zoom :'))
        self._zoom_slider = QSlider(Qt.Horizontal); self._zoom_slider.setRange(50, 400); self._zoom_slider.setValue(100)
        self._zoom_slider.valueChanged.connect(lambda v: (setattr(self, '_zoom', v/100), self._render()))
        zoom_row.addWidget(self._zoom_slider)
        self._lbl_zoom = QLabel('100%'); self._lbl_zoom.setMinimumWidth(40)
        zoom_row.addWidget(self._lbl_zoom)
        btn_rst = QPushButton('Réinitialiser'); btn_rst.clicked.connect(self._reset_view)
        zoom_row.addWidget(btn_rst)
        left.addLayout(zoom_row)

        # ── Panneau droit : contrôles et structures ────────────────────────────
        right = QVBoxLayout()

        # Transparence globale
        trans_grp = QGroupBox('Affichage global')
        trans_l = QVBoxLayout(trans_grp)
        hrow = QHBoxLayout()
        hrow.addWidget(QLabel('Transparence :'))
        sl_a = QSlider(Qt.Horizontal); sl_a.setRange(0, 100); sl_a.setValue(40)
        sl_a.valueChanged.connect(lambda v: (setattr(self, '_alpha_global', v/100), self._render()))
        hrow.addWidget(sl_a)
        trans_l.addLayout(hrow)

        # Épaisseur contours
        lw_row = QHBoxLayout()
        lw_row.addWidget(QLabel('Épaisseur contours :'))
        sl_lw = QSlider(Qt.Horizontal); sl_lw.setRange(1, 8); sl_lw.setValue(2)
        sl_lw.valueChanged.connect(lambda v: (setattr(self, '_line_width', v), self._render()))
        lw_row.addWidget(sl_lw)
        trans_l.addLayout(lw_row)

        trans_grp.setMaximumHeight(90)
        right.addWidget(trans_grp)

        # Structures avec checkboxes, couleurs et sliders d'opacité
        struct_grp = QGroupBox('STRUCTURES')
        struct_l = QVBoxLayout(struct_grp)

        # All / None buttons
        hdr = QHBoxLayout()
        btn_all  = QPushButton('Toutes');  btn_all.clicked.connect(self._select_all)
        btn_none = QPushButton('Aucune');  btn_none.clicked.connect(self._select_none)
        hdr.addWidget(btn_all); hdr.addWidget(btn_none); hdr.addStretch()
        struct_l.addLayout(hdr)

        # Scroll area pour les structures
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f'QScrollArea{{background:{PANEL_BG};border:none;}}')
        inner = QWidget(); inner.setStyleSheet(f'background:{PANEL_BG};')
        inner_l = QVBoxLayout(inner); inner_l.setSpacing(6); inner_l.setContentsMargins(4,4,4,4)

        for name in self._proj_masks.keys():
            r, g, b = self._colors[name]
            row = QHBoxLayout(); row.setSpacing(4)
            
            # Checkbox
            chk = QCheckBox(); chk.setChecked(True)
            chk.toggled.connect(self._render)
            self._chks[name] = chk
            row.addWidget(chk)

            # Swatch couleur
            swatch = QLabel()
            swatch.setFixedSize(20, 20)
            swatch.setStyleSheet(f'background:rgb({r},{g},{b});border-radius:3px;border:1px solid white;')
            row.addWidget(swatch)

            # Nom
            lbl_name = QLabel(name); lbl_name.setMinimumWidth(80)
            row.addWidget(lbl_name)

            # Opacité per-struct
            sl_op = QSlider(Qt.Horizontal); sl_op.setRange(0, 100); sl_op.setValue(100)
            sl_op.setMaximumWidth(80)
            sl_op.valueChanged.connect(lambda v, n=name: (
                self._alpha_per_struct.__setitem__(n, v/100), self._render()))
            self._opacity_sliders[name] = sl_op
            self._alpha_per_struct[name] = 1.0
            row.addWidget(sl_op)

            lbl_pct = QLabel('100%'); lbl_pct.setMinimumWidth(30); lbl_pct.setAlignment(Qt.AlignRight)
            row.addWidget(lbl_pct)
            sl_op.valueChanged.connect(lambda v, lp=lbl_pct: lp.setText(f'{v}%'))

            row.addStretch()
            inner_l.addLayout(row)

        inner_l.addStretch()
        scroll.setWidget(inner)
        struct_l.addWidget(scroll)
        right.addWidget(struct_grp)

        # Boutons d'action
        btn_row = QHBoxLayout()
        btn_export = QPushButton('Exporter'); btn_export.clicked.connect(self._export_image)
        btn_3d = QPushButton('Vue 3D'); btn_3d.clicked.connect(self._open_3d_view)
        btn_close = QPushButton('Fermer'); btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_export); btn_row.addWidget(btn_3d); btn_row.addWidget(btn_close)
        right.addLayout(btn_row)

        # Layout principal
        root.addLayout(left, 2)
        root.addLayout(right, 1)

    def _select_all(self):
        for chk in self._chks.values(): chk.setChecked(True)

    def _select_none(self):
        for chk in self._chks.values(): chk.setChecked(False)

    def _wheel_zoom(self, e):
        delta = e.angleDelta().y()
        if delta > 0: self._zoom = min(self._zoom * 1.1, 4.0)
        else: self._zoom = max(self._zoom / 1.1, 0.5)
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(int(self._zoom * 100))
        self._zoom_slider.blockSignals(False)
        self._render()
        e.accept()

    def _pan_start(self, e):
        self._pan_active = True
        self._pan_start_pos = (e.x(), e.y())

    def _pan_move(self, e):
        if not self._pan_active or self._pan_start_pos is None: return
        dx = (e.x() - self._pan_start_pos[0]) * 0.5
        dy = (e.y() - self._pan_start_pos[1]) * 0.5
        self._pan_x += dx
        self._pan_y += dy
        self._pan_start_pos = (e.x(), e.y())
        self._render()

    def _pan_end(self, e):
        self._pan_active = False

    def _reset_view(self):
        self._zoom = 1.0
        self._pan_x, self._pan_y = 0, 0
        self._zoom_slider.blockSignals(True)
        self._zoom_slider.setValue(100)
        self._zoom_slider.blockSignals(False)
        self._render()

    def _render(self):
        S = int(self._fluoro.shape[0])
        fl = (self._fluoro * 255).astype(np.uint8)
        rgb = cv2.cvtColor(fl, cv2.COLOR_GRAY2RGB).astype(np.float32)

        for name, mask in self._proj_masks.items():
            if not self._chks.get(name, QCheckBox()).isChecked():
                continue
            warped = apply_full_transform(mask.astype(np.float32), self._result)
            # Resize warped to match fluoro dimensions
            if warped.shape[:2] != (S, rgb.shape[1]):
                warped = cv2.resize(warped, (rgb.shape[1], S), interpolation=cv2.INTER_LINEAR)
            r, g, b = self._colors[name]
            struct_alpha = self._alpha_per_struct.get(name, 1.0)
            ov = rgb.copy(); ov[warped > 0.5] = [r, g, b]
            blend_alpha = self._alpha_global * struct_alpha
            rgb = cv2.addWeighted(rgb, 1 - blend_alpha, ov, blend_alpha, 0)
            
            # Contours
            m8 = (warped * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rgb8 = np.clip(rgb, 0, 255).astype(np.uint8)
            cv2.drawContours(rgb8, cnts, -1, (r, g, b), self._line_width, lineType=cv2.LINE_AA)
            rgb = rgb8.astype(np.float32)

        self._full_image = np.clip(rgb, 0, 255).astype(np.uint8)

        # Appliquer zoom et pan
        h, w = self._full_image.shape[:2]
        if abs(self._zoom - 1.0) > 0.01 or abs(self._pan_x) > 0.5 or abs(self._pan_y) > 0.5:
            M = cv2.getRotationMatrix2D((w*0.5, h*0.5), 0, self._zoom)
            M[0, 2] += self._pan_x
            M[1, 2] += self._pan_y
            rgb_t = cv2.warpAffine(self._full_image, M, (w, h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(20,20,46))
        else:
            rgb_t = self._full_image

        # Affichage dans le label
        lw = max(48, self._img_label.width()); lh = max(48, self._img_label.height())
        rgb_r = cv2.resize(rgb_t, (lw, lh), interpolation=cv2.INTER_LINEAR)
        h2, w2 = rgb_r.shape[:2]
        qi = QImage(rgb_r.data, w2, h2, w2 * 3, QImage.Format_RGB888)
        self._img_label.setPixmap(QPixmap.fromImage(qi).copy())

        # Mettre à jour l'affichage du zoom
        self._lbl_zoom.setText(f'{self._zoom*100:.0f}%')

    def _export_image(self):
        if self._full_image is None: return
        p, _ = QFileDialog.getSaveFileName(
            self, 'Exporter image', '',
            'PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tiff)')
        if not p: return
        try:
            rgb_bgr = cv2.cvtColor(self._full_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(p, rgb_bgr)
            QMessageBox.information(self, 'Export', f'Sauvegardé :\n{p}')
        except Exception as ex:
            QMessageBox.warning(self, 'Erreur', f'Export échoué : {ex}')

    def _open_3d_view(self):
        QMessageBox.information(self, 'Vue 3D',
                                'La vue 3D interactive n\'est pas disponible.\n'
                                'Utilisez 3D Slicer pour visualiser les segmentations en 3D.')

    def resizeEvent(self, e):
        super().resizeEvent(e); self._render()


# ══════════════════════════════════════════════════════════════════════════════
# Lecture DICOM fluoroscopie — extraction des paramètres géométriques
# ══════════════════════════════════════════════════════════════════════════════

class ComparisonDialog(QDialog):
    """Dialogue comparant les itérations DRR + recalage côte à côte."""

    def __init__(self, iterations, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Comparaison des iterations')
        self.resize(1100, 700)
        self.setStyleSheet(STYLE)
        self._iters = iterations
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10); root.setSpacing(8)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        inner = QWidget(); inner.setStyleSheet(f'background:{PANEL_BG};')
        grid = QGridLayout(inner); grid.setSpacing(8)

        headers = ['#', 'DRR', 'LAO', 'CRAN', 'Table', 'IoU', 'Dice',
                   'Tx', 'Ty', 'Rot', 'Scale']
        for j, hdr in enumerate(headers):
            lbl = QLabel(hdr)
            lbl.setStyleSheet(f'color:{ACCENT};font-weight:700;font-size:11px;'
                              f'border:none;background:transparent;')
            lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(lbl, 0, j)

        for i, it in enumerate(self._iters):
            row = i + 1
            idx_lbl = QLabel(f'{i+1}')
            idx_lbl.setAlignment(Qt.AlignCenter)
            grid.addWidget(idx_lbl, row, 0)

            thumb_lbl = QLabel()
            thumb_lbl.setFixedSize(64, 64)
            thumb_lbl.setAlignment(Qt.AlignCenter)
            drr = it.get('drr_image')
            if drr is not None:
                u8 = (np.clip(drr, 0, 1) * 255).astype(np.uint8)
                t = cv2.resize(u8, (64, 64), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(t, cv2.COLOR_GRAY2RGB)
                qi = QImage(rgb.data, 64, 64, 64 * 3, QImage.Format_RGB888)
                thumb_lbl.setPixmap(QPixmap.fromImage(qi).copy())
            thumb_lbl.setStyleSheet(f'border:1px solid {BORDER2};border-radius:4px;')
            grid.addWidget(thumb_lbl, row, 1, alignment=Qt.AlignCenter)

            for j, key in enumerate(['lao', 'cran', 'table']):
                val_lbl = QLabel(f'{it[key]:.1f}\u00b0')
                val_lbl.setAlignment(Qt.AlignCenter)
                grid.addWidget(val_lbl, row, 2 + j)

            res = it.get('result')
            if res:
                iou_v = res['iou']; dice_v = res['dice']
                col = ACCENT2 if iou_v > 0.5 else (WARN if iou_v > 0.25 else ERR)
                iou_lbl = QLabel(f'{iou_v:.3f}')
                iou_lbl.setStyleSheet(f'color:{col};font-weight:700;')
                iou_lbl.setAlignment(Qt.AlignCenter)
                dice_lbl = QLabel(f'{dice_v:.3f}')
                dice_lbl.setStyleSheet(f'color:{col};font-weight:700;')
                dice_lbl.setAlignment(Qt.AlignCenter)
                grid.addWidget(iou_lbl, row, 5)
                grid.addWidget(dice_lbl, row, 6)
                grid.addWidget(QLabel(f'{res["tx"]:+.1f}'), row, 7, alignment=Qt.AlignCenter)
                grid.addWidget(QLabel(f'{res["ty"]:+.1f}'), row, 8, alignment=Qt.AlignCenter)
                grid.addWidget(QLabel(f'{res["angle"]:+.2f}\u00b0'), row, 9, alignment=Qt.AlignCenter)
                grid.addWidget(QLabel(f'{res["scale"]:.3f}'), row, 10, alignment=Qt.AlignCenter)
            else:
                na = QLabel('\u2014')
                na.setStyleSheet(f'color:{TEXT_DIM};')
                na.setAlignment(Qt.AlignCenter)
                grid.addWidget(na, row, 5, 1, 6)

        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        btn_close = QPushButton('Fermer')
        btn_close.clicked.connect(self.accept)
        root.addWidget(btn_close)



