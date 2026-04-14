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
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

from core.registration import apply_full_transform
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

class YoloDetectionPanel(QDialog):
    """
    Panneau affichant l'image avec les détections YOLO style,
    et un panneau latéral listant chaque détection avec checkbox,
    confiance, classe, dimensions.
    Retourne la liste des indices sélectionnés via get_selection().
    """
    def __init__(self, det_result: dict, target: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Détections YOLO — {target.upper()}')
        self.resize(1100, 700)
        self.setStyleSheet(STYLE)
        self._det = det_result
        self._target = target
        self._boxes = det_result['boxes']
        self._infer_img = det_result['infer_img']
        self._mask = det_result['mask']
        self._selected = list(range(len(self._boxes)))  # all selected by default
        self._chk_list = []
        self._build_ui()
        self._render_image()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(10)

        # ── Left: image with detections ───────────────────────────────────────
        left = QVBoxLayout()
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setMinimumSize(500, 500)
        self._img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img_label.setStyleSheet(f'background:{DARK_BG};border-radius:6px;border:2px solid {BORDER};')
        left.addWidget(self._img_label, 1)

        # Info bar
        info = QLabel(f'{self._target.upper()}  —  {len(self._boxes)} détection(s)')
        info.setStyleSheet(f'color:{ACCENT};font-size:12px;font-weight:600;background:transparent;')
        info.setAlignment(Qt.AlignCenter)
        left.addWidget(info)
        root.addLayout(left, 3)

        # ── Right: detection list panel ────────────────────────────────────────
        right_w = QWidget()
        right_w.setFixedWidth(320)
        right_w.setStyleSheet(f'background:{PANEL_BG};border:1px solid {BORDER};border-radius:6px;')
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(10, 10, 10, 10); right_l.setSpacing(8)

        title = QLabel('DETECTIONS')
        title.setStyleSheet(f'color:{ACCENT};font-size:12px;font-weight:700;letter-spacing:1px;border:none;background:transparent;')
        right_l.addWidget(title)

        # All / None
        btn_row = QHBoxLayout(); btn_row.setSpacing(4)
        btn_all = QPushButton('Toutes'); btn_all.setFixedHeight(26)
        btn_all.clicked.connect(self._select_all)
        btn_none = QPushButton('Aucune'); btn_none.setFixedHeight(26)
        btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(btn_all); btn_row.addWidget(btn_none)
        right_l.addLayout(btn_row)

        # Scroll area for detections
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f'QScrollArea{{background:transparent;border:none;}}')
        inner = QWidget(); inner.setStyleSheet('background:transparent;')
        inner_l = QVBoxLayout(inner); inner_l.setSpacing(6); inner_l.setContentsMargins(0, 0, 0, 0)

        for i, box in enumerate(self._boxes):
            color = _YOLO_BOX_PALETTE[i % len(_YOLO_BOX_PALETTE)]
            card = self._make_det_card(i, box, color)
            inner_l.addWidget(card)

        inner_l.addStretch()
        scroll.setWidget(inner)
        right_l.addWidget(scroll, 1)

        # Summary
        self._lbl_summary = QLabel(f'{len(self._boxes)} sélectionnée(s)')
        self._lbl_summary.setStyleSheet(f'color:{TEXT_MID};font-size:11px;border:none;background:transparent;')
        self._lbl_summary.setAlignment(Qt.AlignCenter)
        right_l.addWidget(self._lbl_summary)

        # Buttons
        btn_row2 = QHBoxLayout(); btn_row2.setSpacing(4)
        btn_ok = QPushButton('Appliquer'); btn_ok.setObjectName('success')
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton('Annuler')
        btn_cancel.clicked.connect(self.reject)
        btn_row2.addWidget(btn_ok, 1); btn_row2.addWidget(btn_cancel)
        right_l.addLayout(btn_row2)

        root.addWidget(right_w)

    def _make_det_card(self, idx, box, color):
        r, g, b = color
        card = QWidget()
        card.setStyleSheet(
            f'background:{CARD_BG};border:1px solid rgb({r},{g},{b});'
            f'border-radius:6px;')
        cl = QVBoxLayout(card)
        cl.setContentsMargins(8, 6, 8, 6); cl.setSpacing(3)

        # Header row: checkbox + label + confidence badge
        hdr = QHBoxLayout(); hdr.setSpacing(6)
        chk = QCheckBox(); chk.setChecked(True)
        chk.toggled.connect(lambda checked, i=idx: self._on_toggle(i, checked))
        self._chk_list.append(chk)
        hdr.addWidget(chk)

        lbl_name = QLabel(f'V{idx+1}')
        lbl_name.setStyleSheet(
            f'color:rgb({r},{g},{b});font-size:13px;font-weight:700;'
            f'border:none;background:transparent;')
        hdr.addWidget(lbl_name)

        conf_pct = int(box['conf'] * 100)
        conf_color = ACCENT2 if conf_pct >= 70 else (WARN if conf_pct >= 40 else ERR)
        badge = QLabel(f'{conf_pct}%')
        badge.setFixedSize(42, 20)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            f'background:{conf_color};color:#fff;font-size:10px;font-weight:700;'
            f'border-radius:10px;border:none;')
        hdr.addWidget(badge)
        hdr.addStretch()
        cl.addLayout(hdr)

        # Class name
        cls_lbl = QLabel(box.get('cls_name', 'vertebra'))
        cls_lbl.setStyleSheet(f'color:{TEXT_MID};font-size:10px;border:none;background:transparent;')
        cl.addWidget(cls_lbl)

        # Box info
        bw = box['x2'] - box['x1']
        bh = box['y2'] - box['y1']
        cx = (box['x1'] + box['x2']) // 2
        cy = (box['y1'] + box['y2']) // 2
        info = QLabel(f'Centre ({cx}, {cy})  |  {bw}×{bh} px')
        info.setStyleSheet(f'color:{TEXT_DIM};font-size:10px;border:none;background:transparent;')
        cl.addWidget(info)

        return card

    def _on_toggle(self, idx, checked):
        if checked and idx not in self._selected:
            self._selected.append(idx)
            self._selected.sort()
        elif not checked and idx in self._selected:
            self._selected.remove(idx)
        n = len(self._selected)
        self._lbl_summary.setText(f'{n} sélectionnée(s) / {len(self._boxes)}')
        self._render_image()

    def _select_all(self):
        for chk in self._chk_list: chk.setChecked(True)

    def _select_none(self):
        for chk in self._chk_list: chk.setChecked(False)

    def _render_image(self):
        img = self._infer_img.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        for i, box in enumerate(self._boxes):
            color = _YOLO_BOX_PALETTE[i % len(_YOLO_BOX_PALETTE)]
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            is_sel = i in self._selected
            thickness = 2 if is_sel else 1
            alpha_col = color if is_sel else tuple(c // 3 for c in color)

            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), alpha_col, thickness, lineType=cv2.LINE_AA)

            if is_sel:
                # Label background
                label = f'V{i+1} {box["conf"]:.0%}'
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 8, y1), alpha_col, -1)
                cv2.putText(img, label, (x1 + 4, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Corner accents (YOLO style)
                corner_len = min(15, (x2 - x1) // 4, (y2 - y1) // 4)
                ct = 3
                # Top-left
                cv2.line(img, (x1, y1), (x1 + corner_len, y1), alpha_col, ct, cv2.LINE_AA)
                cv2.line(img, (x1, y1), (x1, y1 + corner_len), alpha_col, ct, cv2.LINE_AA)
                # Top-right
                cv2.line(img, (x2, y1), (x2 - corner_len, y1), alpha_col, ct, cv2.LINE_AA)
                cv2.line(img, (x2, y1), (x2, y1 + corner_len), alpha_col, ct, cv2.LINE_AA)
                # Bottom-left
                cv2.line(img, (x1, y2), (x1 + corner_len, y2), alpha_col, ct, cv2.LINE_AA)
                cv2.line(img, (x1, y2), (x1, y2 - corner_len), alpha_col, ct, cv2.LINE_AA)
                # Bottom-right
                cv2.line(img, (x2, y2), (x2 - corner_len, y2), alpha_col, ct, cv2.LINE_AA)
                cv2.line(img, (x2, y2), (x2, y2 - corner_len), alpha_col, ct, cv2.LINE_AA)

        # Fit to label
        lw = max(64, self._img_label.width())
        lh = max(64, self._img_label.height())
        h, w = img.shape[:2]
        scale = min(lw / w, lh / h)
        nw, nh = int(w * scale), int(h * scale)
        img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img_r = np.ascontiguousarray(img_r)
        qi = QImage(img_r.data, nw, nh, nw * 3, QImage.Format_RGB888)
        self._img_label.setPixmap(QPixmap.fromImage(qi).copy())

    def get_selection(self):
        return sorted(self._selected)

    def resizeEvent(self, e):
        super().resizeEvent(e); self._render_image()


# ══════════════════════════════════════════════════════════════════════════════
# Panneau dual : sélection vertèbres Fluoro (gauche) + DRR (droite)
# ══════════════════════════════════════════════════════════════════════════════

class DualYoloSelectionDialog(QDialog):
    """
    Fenêtre split : fluoroscopie à gauche, DRR à droite.
    Chaque côté affiche les détections YOLO avec checkboxes.
    L'utilisateur coche les vertèbres à utiliser pour le recalage.
    Retourne (fluoro_indices, drr_indices) via get_selections().
    """

    def __init__(self, det_fl: dict, det_drr: dict,
                 boxes_fl: list, named_drr_boxes: list,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle('Sélection des vertèbres — Fluoro / DRR')
        self.resize(1500, 800)
        self.setStyleSheet(STYLE)

        self._det_fl = det_fl
        self._det_drr = det_drr
        self._boxes_fl = boxes_fl
        self._named_drr_boxes = named_drr_boxes

        # Sélections (indices)
        self._sel_fl = list(range(len(det_fl['boxes'])))
        self._sel_drr = list(range(len(named_drr_boxes)))

        self._chks_fl: list = []
        self._chks_drr: list = []

        self._build_ui()
        self._render_fl()
        self._render_drr()

    # ── Construction UI ───────────────────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ── Titre ──────────────────────────────────────────────────────────────
        title = QLabel('Sélectionnez les vertèbres : Fluoro (gauche) / DRR (droite)')
        title.setStyleSheet(
            f'color:{ACCENT};font-size:13px;font-weight:700;'
            f'background:transparent;padding:4px;')
        outer.addWidget(title)

        # ── Ligne principale : fluoro | sep | DRR ─────────────────────────────
        row = QHBoxLayout()
        row.setSpacing(0)

        # --- Gauche : Fluoro ---
        row.addWidget(self._build_side(
            'FLUORO', self._det_fl['boxes'], self._chks_fl,
            self._sel_fl, 'fl'), 1)

        # --- Séparateur central ---
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f'color:{BORDER};margin:0px;width:2px;')
        row.addWidget(sep)

        # --- Droite : DRR ---
        row.addWidget(self._build_side(
            'DRR', self._named_drr_boxes, self._chks_drr,
            self._sel_drr, 'drr'), 1)

        outer.addLayout(row, 1)

        # ── Boutons bas ───────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()
        btn_ok = QPushButton('✓ Lancer le recalage')
        btn_ok.setObjectName('success')
        btn_ok.setFixedHeight(38)
        btn_ok.setMinimumWidth(160)
        btn_ok.setFont(QFont('monospace', 10, QFont.Bold))
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton('✕ Annuler')
        btn_cancel.setFixedHeight(38)
        btn_cancel.setMinimumWidth(120)
        btn_cancel.setFont(QFont('monospace', 10))
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()
        outer.addLayout(btn_row)

    def _build_side(self, title_text, boxes, chk_list, sel_list, side_key):
        """Panneau dédié à un côté : image (grande) + liste très compacte."""
        container = QWidget()
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(4)

        # Titre du côté
        title = QLabel(title_text)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f'color:{ACCENT};font-size:11px;font-weight:700;'
            f'background:transparent;letter-spacing:2px;padding:4px;')
        vl.addWidget(title)

        # ── Image (grande) ───────────────────────────────────────────────────
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setMinimumSize(300, 300)
        img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_label.setStyleSheet(
            f'background:{DARK_BG};border:1px solid {BORDER};border-radius:4px;')
        vl.addWidget(img_label, 1)
        setattr(self, f'_img_{side_key}', img_label)

        # ── Liste compacte (boutons toggles, pas cards) ──────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(80)
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

        for i, box in enumerate(boxes):
            color = _YOLO_BOX_PALETTE[i % len(_YOLO_BOX_PALETTE)]
            tb = self._make_toggle_btn(i, box, color, chk_list, sel_list, side_key)
            inner_l.addWidget(tb)

        inner_l.addStretch()
        scroll.setWidget(inner_w)
        vl.addWidget(scroll)

        return container

    def _make_toggle_btn(self, idx, box, color, chk_list, sel_list, side_key):
        """Bouton toggle compact pour une détection."""
        r, g, b = color
        btn = QPushButton()
        btn.setFixedHeight(28)
        btn.setMinimumWidth(40)
        btn.setCheckable(True)
        btn.setChecked(True)
        
        name = box.get('cls_name', f'V{idx+1}')
        conf_pct = int(box.get('conf', 0) * 100)
        btn.setText(f'{name}\n{conf_pct}%')
        btn.setFont(QFont('monospace', 9))
        
        # Style : fond coloré si sélectionné, sinon grisé
        def update_style():
            is_checked = btn.isChecked()
            if is_checked:
                style = (f'background-color:rgb({r},{g},{b});'
                        f'color:white;border:2px solid rgb({r},{g},{b});'
                        f'border-radius:4px;font-weight:700;')
            else:
                style = (f'background-color:{CARD_BG};'
                        f'color:{TEXT_MID};border:1px solid {BORDER};'
                        f'border-radius:4px;')
            btn.setStyleSheet(style)

        def on_toggled(checked):
            if checked and idx not in sel_list:
                sel_list.append(idx)
                sel_list.sort()
            elif not checked and idx in sel_list:
                sel_list.remove(idx)
            update_style()
            if side_key == 'fl':
                self._render_fl()
            else:
                self._render_drr()

        btn.toggled.connect(on_toggled)
        chk_list.append(btn)  # stocke les boutons au lieu des checkboxes
        update_style()
        return btn

    def _set_all(self, chk_list, state):
        """chk_list contient maintenant des boutons, pas des checkboxes."""
        for btn in chk_list:
            btn.setChecked(state)

    # ── Rendu images ──────────────────────────────────────────────────────────
    def _render_fl(self):
        self._render_side(
            self._det_fl['infer_img'], self._det_fl['boxes'],
            self._sel_fl, self._img_fl)

    def _render_drr(self):
        self._render_side(
            self._det_drr['infer_img'], self._named_drr_boxes,
            self._sel_drr, self._img_drr)

    def _render_side(self, base_img, boxes, sel_list, label_widget):
        img = base_img.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)

        for i, box in enumerate(boxes):
            color = _YOLO_BOX_PALETTE[i % len(_YOLO_BOX_PALETTE)]
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            is_sel = i in sel_list
            thickness = 2 if is_sel else 1
            col = color if is_sel else tuple(c // 3 for c in color)

            cv2.rectangle(img, (x1, y1), (x2, y2), col, thickness, cv2.LINE_AA)
            if is_sel:
                name = box.get('cls_name', f'V{i+1}')
                label = f'{name} {box.get("conf", 0):.0%}'
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(img, (x1, y1 - th - 6),
                              (x1 + tw + 6, y1), col, -1)
                cv2.putText(img, label, (x1 + 3, y1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 0), 1, cv2.LINE_AA)

        lw = max(64, label_widget.width())
        lh = max(64, label_widget.height())
        h, w = img.shape[:2]
        scale = min(lw / w, lh / h)
        nw, nh = int(w * scale), int(h * scale)
        img_r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        img_r = np.ascontiguousarray(img_r)
        qi = QImage(img_r.data, nw, nh, nw * 3, QImage.Format_RGB888)
        label_widget.setPixmap(QPixmap.fromImage(qi).copy())

    def get_selections(self):
        """Retourne (fluoro_indices, drr_indices)."""
        return sorted(self._sel_fl), sorted(self._sel_drr)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render_fl()
        self._render_drr()


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



