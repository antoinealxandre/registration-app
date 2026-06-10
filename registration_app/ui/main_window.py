"""Main window orchestration for the registration application."""

import json
import math
import os
from datetime import datetime

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import SimpleITK as sitk
except ImportError:
    sitk = None

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QSlider,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QProgressBar,
    QTabWidget,
    QSizePolicy,
    QStatusBar,
    QFrame,
    QMessageBox,
    QGridLayout,
    QCheckBox,
    QDialog,
    QScrollArea,
    QButtonGroup,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont

from core.drr_generator import load_ct, DRR_POSTPROCESS_PRESETS, enhance_drr_image
from core.measurements import (
    ms_length_from_hinges,
    project_world_to_fluoro_pixel,
    risk_assessment,
    view_pixel_from_voxel,
    voxel_from_view_pixel,
    voxel_from_world,
    world_from_voxel,
)
from core.registration import apply_full_transform
from core.stent_placement import (
    generate_stent_mesh,
    project_stent_mask,
    mask_center,
    transform_mask,
)
from core.totalseg_runner import export_multilabel_segmentation
from core.yolo_pipeline import (
    load_yolo_model as yolo_load,
    is_model_loaded as yolo_ready,
    boxes_to_mask,
)
from services.pipeline_workers import WorkerThread
from ui.theme import (
    STYLE,
    SIDEBAR_W,
    DEFAULT_FOV_MM,
    AUTO_PIPELINE_FOV_MM,
    STRUCT,
    color_for_structure,
    BORDER,
    BORDER2,
    ACCENT,
    ACCENT2,
    CARD_BG,
    DARK_BG,
    PANEL_BG,
    TEXT,
    TEXT_DIM,
    TEXT_MID,
    WARN,
    ERR,
)
from ui.widgets.annotation_widgets import (
    DropZone,
    CollapsibleSection,
    ImageCard,
    AnnotationCanvas,
    BusyOverlay,
    ResultPanel,
    SegmentationReviewPanel,
    FinalOverlayPanel,
    _make_svg_icon,
)
from ui.dialogs import (
    YoloDetectionPanel,
    DualYoloSelectionDialog,
    VertebralDetectionWindow,
    ComparisonDialog,
)
from utils.dicom_io import read_dicom_fluoro_series, read_metadata_csv


TOTALSEG_LICENSE_KEY = 'aca_17SH96R9RRZYEV'

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('2D/3D Registration')
        self.setMinimumSize(1280, 800)
        self.setStyleSheet(STYLE)
        self.ct_vol = self.voxel_mm = self.ct_aff = None
        self.ct_path = None
        self.ap_axis = 1
        self.seg_masks = {}; self.proj_masks = {}; self.drr_image = None
        self.seg_affine = None
        self.tseg_output_dir = None
        self.tseg_multilabel_path = None
        self.dicom_meta = {}
        self.fluoro_image = None; self.result = None
        self._drr_base_image = None
        self._loaded_images = []; self._pending_csv = None
        self._fixed_image_index = None
        self._mobile_image_index = None
        self._iterations = []
        self._current_iter_idx = -1
        self.stent_mesh = None
        self._stent_base_mask = None
        self._stent_base_center = None
        self._stent_proj_params = None
        self._stent_center_px = None
        self._stent_axis_deg = 0.0
        self._stent_mode_active = False
        self._ms_world = {}              # {'hinge1','hinge2','ms'} -> np.array world mm
        self._ms_click_target = None     # nom du prochain point à capturer
        self._build_ui()
        self._build_menu_bar()
        self._on_drr_preset_changed(self.cb_drr_preset.currentIndex())
        self._load_yolo_default()  # Charger modèle auto au démarrage
        self._status('Deposez vos fichiers pour commencer')

    # ─────────────────────────────────────────────────────────────────────────
    def _sep(self):
        f = QFrame(); f.setObjectName('sep'); f.setFrameShape(QFrame.HLine)
        return f

    def _section_label(self, text):
        l = QLabel(text); l.setObjectName('section-title')
        return l

    def _build_ui(self):
        cw = QWidget(); self.setCentralWidget(cw)
        root = QHBoxLayout(cw)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Panneau gauche scrollable ─────────────────────────────────────────
        left_outer = QWidget()
        left_outer.setFixedWidth(SIDEBAR_W)
        left_outer.setStyleSheet(f'background:{PANEL_BG};border-right:1px solid {BORDER};')
        left_vbox = QVBoxLayout(left_outer)
        left_vbox.setContentsMargins(0, 0, 0, 0)
        left_vbox.setSpacing(0)

        # Logo/titre
        tl = QLabel('2D/3D Registration')
        tl.setAlignment(Qt.AlignCenter)
        tl.setFixedHeight(44)
        tl.setStyleSheet(
            f'color:{ACCENT};font-size:13px;font-weight:700;letter-spacing:1px;'
            f'background:{PANEL_BG};border-bottom:1px solid {BORDER};'
        )
        left_vbox.addWidget(tl)

        # Scroll container
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_content = QWidget()
        scroll_content.setStyleSheet(f'background:{PANEL_BG};')
        self._ll = QVBoxLayout(scroll_content)
        self._ll.setContentsMargins(10, 10, 10, 10)
        self._ll.setSpacing(8)
        scroll.setWidget(scroll_content)
        left_vbox.addWidget(scroll, 1)
        root.addWidget(left_outer)
        self._left_outer = left_outer   # reference pour le dock (afficher/masquer)

        ll = self._ll   # shorthand

        # ── DONNEES ───────────────────────────────────────────────────────────
        sec_data = CollapsibleSection('DONNEES', starts_open=True)
        self._drop_zone = DropZone()
        self._drop_zone.files_dropped.connect(self._on_files_dropped)
        sec_data.addWidget(self._drop_zone)

        btn_browse = QPushButton('Parcourir...')
        btn_browse.clicked.connect(self._on_browse)
        sec_data.addWidget(btn_browse)

        self._chk_indicators = {}

        def _dot_row(label_widget, key):
            row = QHBoxLayout(); row.setSpacing(6); row.setContentsMargins(0, 0, 0, 0)
            dot = QLabel('●'); dot.setFixedWidth(16)
            dot.setStyleSheet(f'color:{TEXT_DIM};font-size:14px;border:none;background:transparent;')
            self._chk_indicators[key] = dot
            row.addWidget(dot); row.addWidget(label_widget, 1)
            return row

        self.lbl_ct = QLabel('CT : --'); self.lbl_ct.setObjectName('dim'); self.lbl_ct.setWordWrap(True)
        self.lbl_seg = QLabel('Seg : --'); self.lbl_seg.setObjectName('dim'); self.lbl_seg.setWordWrap(True)
        self.lbl_fluoro_meta = QLabel('Fluoro : --'); self.lbl_fluoro_meta.setObjectName('dim'); self.lbl_fluoro_meta.setWordWrap(True)
        sec_data.addLayout(_dot_row(self.lbl_ct, 'ct'))
        sec_data.addLayout(_dot_row(self.lbl_seg, 'seg'))
        sec_data.addLayout(_dot_row(self.lbl_fluoro_meta, 'fluoro'))

        self._fluoro_frame_box = QWidget()
        self._fluoro_frame_box.setStyleSheet('background:transparent;')
        frame_box = QVBoxLayout(self._fluoro_frame_box)
        frame_box.setContentsMargins(0, 0, 0, 0)
        frame_box.setSpacing(3)
        frame_row = QHBoxLayout(); frame_row.setSpacing(6); frame_row.setContentsMargins(0, 0, 0, 0)
        self.lbl_fluoro_frame = QLabel('Frame :')
        self.lbl_fluoro_frame.setObjectName('mid')
        self.sl_fluoro_frame = QSlider(Qt.Horizontal)
        self.sl_fluoro_frame.setRange(1, 1)
        self.sl_fluoro_frame.setValue(1)
        self.sl_fluoro_frame.valueChanged.connect(self._on_fluoro_frame_slider)
        self.lbl_fluoro_frame_value = QLabel('-- / --')
        self.lbl_fluoro_frame_value.setObjectName('dim')
        self.lbl_fluoro_frame_value.setFixedWidth(54)
        frame_row.addWidget(self.lbl_fluoro_frame)
        frame_row.addWidget(self.sl_fluoro_frame, 1)
        frame_row.addWidget(self.lbl_fluoro_frame_value)
        frame_box.addLayout(frame_row)
        self.lbl_fluoro_frame_hint = QLabel('Molette sur la fluoroscopie pour changer de frame')
        self.lbl_fluoro_frame_hint.setObjectName('dim')
        self.lbl_fluoro_frame_hint.setWordWrap(True)
        frame_box.addWidget(self.lbl_fluoro_frame_hint)
        self._fluoro_frame_box.hide()
        sec_data.addWidget(self._fluoro_frame_box)

        # Dots fantômes pour DRR / YOLO / Reg (non affichés, mis à jour par _update_checklist)
        for _key in ('drr', 'yolo', 'reg'):
            _d = QLabel('●'); _d.setFixedWidth(16)
            _d.setStyleSheet(f'color:{TEXT_DIM};font-size:14px;border:none;background:transparent;')
            _d.hide()
            self._chk_indicators[_key] = _d

        ll.addWidget(sec_data)

        # ── SEGMENTATION AUTO (TotalSegmentator) ───────────────────────────
        sec_seg_auto = CollapsibleSection('SEGMENTATION AUTO', starts_open=False)

        seg_info = QLabel(
            'Segmentation automatique TotalSegmentator\n'
            'Source : le meme CT NIfTI que celui charge pour le DRR')
        seg_info.setObjectName('dim'); seg_info.setWordWrap(True)
        sec_seg_auto.addWidget(seg_info)

        self.lbl_tseg_src = QLabel('Source : CT courant')
        self.lbl_tseg_src.setObjectName('dim')
        self.lbl_tseg_src.setWordWrap(True)
        sec_seg_auto.addWidget(self.lbl_tseg_src)

        def _lbl_mid(t):
            l = QLabel(t)
            l.setObjectName('mid')
            return l

        tseg_grid = QWidget(); tseg_grid.setStyleSheet('background:transparent;')
        tgl = QGridLayout(tseg_grid)
        tgl.setContentsMargins(0, 0, 0, 0); tgl.setSpacing(4)
        tgl.setColumnStretch(1, 1)

        tgl.addWidget(_lbl_mid('Tache'), 0, 0)
        self.cb_tseg_task = QComboBox()
        self.cb_tseg_task.addItem('total')
        self.cb_tseg_task.addItem('heartchambers_highres')
        tgl.addWidget(self.cb_tseg_task, 0, 1)

        tgl.addWidget(_lbl_mid('Device'), 1, 0)
        self.cb_tseg_device = QComboBox()
        self.cb_tseg_device.addItems(['gpu', 'cpu'])
        tgl.addWidget(self.cb_tseg_device, 1, 1)

        self.chk_tseg_fast = QCheckBox('Mode rapide (--fast)')
        self.chk_tseg_fast.setChecked(False)
        tgl.addWidget(self.chk_tseg_fast, 2, 0, 1, 2)

        tgl.addWidget(_lbl_mid('Licence'), 3, 0)
        lbl_lic = QLabel(f'Integree: {TOTALSEG_LICENSE_KEY}')
        lbl_lic.setObjectName('dim')
        lbl_lic.setWordWrap(True)
        tgl.addWidget(lbl_lic, 3, 1)

        sec_seg_auto.addWidget(tseg_grid)

        self.btn_tseg_run = QPushButton('Segmenter automatiquement')
        self.btn_tseg_run.setObjectName('primary')
        self.btn_tseg_run.clicked.connect(self._run_totalseg)
        sec_seg_auto.addWidget(self.btn_tseg_run)

        exp_row = QHBoxLayout(); exp_row.setSpacing(4)
        self.btn_tseg_export = QPushButton('Telecharger segmentation (fichier unique)')
        self.btn_tseg_export.setEnabled(False)
        self.btn_tseg_export.clicked.connect(self._export_totalseg_masks)
        exp_row.addWidget(self.btn_tseg_export, 1)
        sec_seg_auto.addLayout(exp_row)

        self.btn_tseg_3d = QPushButton('Vue 3D segmentation')
        self.btn_tseg_3d.setEnabled(False)
        self.btn_tseg_3d.clicked.connect(self._open_segmentation_3d)
        sec_seg_auto.addWidget(self.btn_tseg_3d)

        self.lbl_tseg_status = QLabel('')
        self.lbl_tseg_status.setObjectName('dim')
        self.lbl_tseg_status.setWordWrap(True)
        sec_seg_auto.addWidget(self.lbl_tseg_status)

        ll.addWidget(sec_seg_auto)

        # ── IMAGES ────────────────────────────────────────────────────────────
        self._sec_images = CollapsibleSection('IMAGES', starts_open=False)
        img_actions = QHBoxLayout(); img_actions.setSpacing(6); img_actions.setContentsMargins(0, 0, 0, 0)
        self.btn_clear_images = QPushButton('Tout retirer')
        self.btn_clear_images.setObjectName('danger')
        self.btn_clear_images.setEnabled(False)
        self.btn_clear_images.clicked.connect(self._remove_all_images)
        img_actions.addWidget(self.btn_clear_images)
        img_actions.addStretch()
        self._sec_images.addLayout(img_actions)
        self.lbl_no_images = QLabel('Aucune image chargee')
        self.lbl_no_images.setObjectName('dim')
        self._sec_images.addWidget(self.lbl_no_images)
        self._images_container = QWidget()
        self._images_container.setStyleSheet('background:transparent;')
        self._images_vbox = QVBoxLayout(self._images_container)
        self._images_vbox.setContentsMargins(0, 0, 0, 0)
        self._images_vbox.setSpacing(6)
        self._sec_images.addWidget(self._images_container)
        ll.addWidget(self._sec_images)

        # ── PARAMETRES DRR ────────────────────────────────────────────────────
        self.sec_drr = sec_drr = CollapsibleSection('PARAMETRES DRR', starts_open=False)
        drr_grid = QWidget(); drr_grid.setStyleSheet('background:transparent;')
        gdl = QGridLayout(drr_grid)
        gdl.setContentsMargins(0, 0, 0, 0); gdl.setSpacing(5)
        gdl.setColumnStretch(1, 1)

        def _lbl(t):
            l = QLabel(t); l.setObjectName('mid'); return l

        gdl.addWidget(_lbl('Cran/Caud (deg)'), 0, 0)
        self.sp_lao = QDoubleSpinBox(); self.sp_lao.setRange(-180, 180); self.sp_lao.setValue(0); self.sp_lao.setSingleStep(0.5)
        gdl.addWidget(self.sp_lao, 0, 1)
        gdl.addWidget(_lbl('LAO/RAO (deg)'), 1, 0)
        self.sp_cran = QDoubleSpinBox(); self.sp_cran.setRange(-180, 180); self.sp_cran.setValue(0); self.sp_cran.setSingleStep(0.5)
        gdl.addWidget(self.sp_cran, 1, 1)
        gdl.addWidget(_lbl('Table (deg)'), 2, 0)
        self.sp_table = QDoubleSpinBox(); self.sp_table.setRange(-45, 45); self.sp_table.setValue(0); self.sp_table.setSingleStep(0.5)
        gdl.addWidget(self.sp_table, 2, 1)
        gdl.addWidget(_lbl('FOV isoctr (mm)'), 3, 0)
        self.sp_fov = QDoubleSpinBox(); self.sp_fov.setRange(50, 500); self.sp_fov.setValue(DEFAULT_FOV_MM); self.sp_fov.setSingleStep(10)
        gdl.addWidget(self.sp_fov, 3, 1)
        gdl.addWidget(_lbl('Resolution (px)'), 4, 0)
        self.sp_size = QSpinBox(); self.sp_size.setRange(128, 1024); self.sp_size.setValue(512); self.sp_size.setSingleStep(64)
        gdl.addWidget(self.sp_size, 4, 1)
        gdl.addWidget(_lbl('Preset rendu'), 5, 0)
        self.cb_drr_preset = QComboBox()
        self.cb_drr_preset.addItem('Equilibre', 'balanced')
        self.cb_drr_preset.addItem('Contours osseux', 'bone')
        self.cb_drr_preset.addItem('Contraste doux', 'soft')
        self.cb_drr_preset.setCurrentIndex(1)
        self.cb_drr_preset.currentIndexChanged.connect(self._on_drr_preset_changed)
        gdl.addWidget(self.cb_drr_preset, 5, 1)
        self.chk_drr_soft = QCheckBox('Attenuer tissus mous')
        self.chk_drr_soft.setChecked(True)
        gdl.addWidget(self.chk_drr_soft, 6, 0, 1, 2)
        self.chk_drr_clahe = QCheckBox('Contraste local (CLAHE)')
        self.chk_drr_clahe.setChecked(True)
        gdl.addWidget(self.chk_drr_clahe, 7, 0, 1, 2)
        self.chk_drr_edges = QCheckBox('Renforcer gradient et contours')
        self.chk_drr_edges.setChecked(True)
        gdl.addWidget(self.chk_drr_edges, 8, 0, 1, 2)
        gdl.addWidget(_lbl('Gain os'), 9, 0)
        self.sp_drr_bone = QDoubleSpinBox(); self.sp_drr_bone.setRange(0.2, 2.5); self.sp_drr_bone.setValue(1.0); self.sp_drr_bone.setSingleStep(0.1)
        gdl.addWidget(self.sp_drr_bone, 9, 1)
        gdl.addWidget(_lbl('Gain contours'), 10, 0)
        self.sp_drr_edges = QDoubleSpinBox(); self.sp_drr_edges.setRange(0.0, 2.5); self.sp_drr_edges.setValue(1.0); self.sp_drr_edges.setSingleStep(0.1)
        gdl.addWidget(self.sp_drr_edges, 10, 1)
        gdl.addWidget(_lbl('Gamma'), 11, 0)
        self.sp_drr_gamma = QDoubleSpinBox(); self.sp_drr_gamma.setRange(0.3, 1.6); self.sp_drr_gamma.setValue(0.74); self.sp_drr_gamma.setSingleStep(0.02)
        gdl.addWidget(self.sp_drr_gamma, 11, 1)
        sec_drr.addWidget(drr_grid)
        lbl_drr_hint = QLabel(
            'La generation DRR revient au rendu precedent.\n'
            'Les filtres ci-dessous s appliquent apres generation sur le DRR courant.')
        lbl_drr_hint.setObjectName('dim')
        lbl_drr_hint.setWordWrap(True)
        sec_drr.addWidget(lbl_drr_hint)
        row_drr_filters = QHBoxLayout(); row_drr_filters.setSpacing(6)
        self.btn_drr_apply_filters = QPushButton('Appliquer filtres')
        self.btn_drr_apply_filters.clicked.connect(self._apply_drr_filters)
        self.btn_drr_apply_filters.setEnabled(False)
        row_drr_filters.addWidget(self.btn_drr_apply_filters, 1)
        self.btn_drr_reset_filters = QPushButton('DRR de base')
        self.btn_drr_reset_filters.clicked.connect(self._reset_drr_filters)
        self.btn_drr_reset_filters.setEnabled(False)
        row_drr_filters.addWidget(self.btn_drr_reset_filters, 1)
        sec_drr.addLayout(row_drr_filters)
        self.btn_drr = QPushButton('Generer DRR'); self.btn_drr.setObjectName('primary')
        self.btn_drr.clicked.connect(self.generate_drr); self.btn_drr.setEnabled(False)
        sec_drr.addWidget(self.btn_drr)
        self.sp_lao.valueChanged.connect(self._update_stent_overlay)
        self.sp_cran.valueChanged.connect(self._update_stent_overlay)
        self.sp_fov.valueChanged.connect(self._update_stent_overlay)
        self.sp_fov.valueChanged.connect(lambda _v: self._ms_recompute())
        ll.addWidget(sec_drr)

        # ── ANNOTATION ────────────────────────────────────────────────────────
        sec_ann = CollapsibleSection('ANNOTATION', starts_open=False)

        self.chk_use_seg = QCheckBox('Utiliser la segmentation CT (contours auto sur DRR)')
        self.chk_use_seg.setChecked(False)
        self.chk_use_seg.setToolTip(
            'Coche : les masques de segmentation 3D sont auto-projetes sur le DRR apres generation.\n'
            'Decoche : le DRR est genere vierge, annoter manuellement les vertebres.')
        sec_ann.addWidget(self.chk_use_seg)

        # Boutons outils avec icônes Material Icons
        tr = QHBoxLayout(); tr.setSpacing(4); tr.setContentsMargins(0, 0, 0, 0)
        _pencil_d = ('M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25z'
                     'M20.71 7.04c.39-.39.39-1.02 0-1.41l-2.34-2.34c-.39-.39'
                     '-1.02-.39-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z')
        _rect_d   = 'M3 3v18h18V3H3zm16 16H5V5h14v14z'
        _poly_d   = ('M4 6a2 2 0 1 0 0.01 0zM10 4a2 2 0 1 0 0.01 0z'
                     'M18 7a2 2 0 1 0 0.01 0zM16 18a2 2 0 1 0 0.01 0z'
                     'M6 17a2 2 0 1 0 0.01 0zM5.7 7.5l2.7-1.8M11.9 5.3l4.2 1.1'
                     'M17.2 8.7l-1.1 7M14.4 17.8l-6.8-.3M5.7 15.3l-.8-6')
        _eraser_d = ('M15.14 3c-.51 0-1.02.2-1.41.59L2.59 14.73c-.78.77-.78 2.04'
                     ' 0 2.83L5.03 20h7.66l8.72-8.73c.79-.78.79-2.05 0-2.83l-4.85'
                     '-4.85c-.39-.39-.9-.59-1.42-.59zM6.04 19l-2.06-2.07 5.23-5.22'
                     ' 2.07 2.07L6.04 19z')

        self.btn_pencil = QPushButton(); self.btn_pencil.setObjectName('tool')
        self.btn_pencil.setCheckable(True); self.btn_pencil.setChecked(True)
        self.btn_pencil.setFixedSize(34, 34); self.btn_pencil.setToolTip('Crayon (tracé libre)')
        _ic = _make_svg_icon(_pencil_d)
        if _ic: self.btn_pencil.setIcon(_ic); self.btn_pencil.setIconSize(QSize(20, 20))
        else: self.btn_pencil.setText('✏')

        self.btn_rect = QPushButton(); self.btn_rect.setObjectName('tool')
        self.btn_rect.setCheckable(True)
        self.btn_rect.setFixedSize(34, 34); self.btn_rect.setToolTip('Rectangle')
        _ic = _make_svg_icon(_rect_d)
        if _ic: self.btn_rect.setIcon(_ic); self.btn_rect.setIconSize(QSize(20, 20))
        else: self.btn_rect.setText('⬜')

        self.btn_poly = QPushButton(); self.btn_poly.setObjectName('tool')
        self.btn_poly.setCheckable(True)
        self.btn_poly.setFixedSize(34, 34); self.btn_poly.setToolTip('Polygone point par point')
        _ic = _make_svg_icon(_poly_d)
        if _ic: self.btn_poly.setIcon(_ic); self.btn_poly.setIconSize(QSize(20, 20))
        else: self.btn_poly.setText('P')

        self.btn_eraser = QPushButton(); self.btn_eraser.setObjectName('tool')
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.setFixedSize(34, 34); self.btn_eraser.setToolTip('Gomme (effacer)')
        _ic = _make_svg_icon(_eraser_d)
        if _ic: self.btn_eraser.setIcon(_ic); self.btn_eraser.setIconSize(QSize(20, 20))
        else: self.btn_eraser.setText('⌫')

        self.btn_pencil.clicked.connect(lambda: self._set_tool('pencil'))
        self.btn_rect.clicked.connect(lambda: self._set_tool('rectangle'))
        self.btn_poly.clicked.connect(lambda: self._set_tool('polygon'))
        self.btn_eraser.clicked.connect(lambda: self._set_tool('eraser'))
        tr.addWidget(self.btn_pencil); tr.addWidget(self.btn_rect); tr.addWidget(self.btn_poly); tr.addWidget(self.btn_eraser)
        tr.addStretch()
        sec_ann.addLayout(tr)

        rp = QHBoxLayout(); rp.setSpacing(6); rp.setContentsMargins(0, 0, 0, 0)
        rp.addWidget(_lbl('Epaisseur :'))
        self.sl_pen = QSlider(Qt.Horizontal); self.sl_pen.setRange(1, 15); self.sl_pen.setValue(2)
        self.lbl_pen = QLabel('2px'); self.lbl_pen.setObjectName('dim'); self.lbl_pen.setFixedWidth(30)
        self.sl_pen.valueChanged.connect(self._on_pen)
        rp.addWidget(self.sl_pen, 1); rp.addWidget(self.lbl_pen)
        sec_ann.addLayout(rp)

        act = QHBoxLayout(); act.setSpacing(6); act.setContentsMargins(0, 0, 0, 0)
        b_undo = QPushButton('Annuler'); b_undo.setObjectName('warn')
        b_all = QPushButton('Tout effacer'); b_all.setObjectName('danger')
        b_undo.clicked.connect(self._undo)
        b_all.clicked.connect(self._clear_all)
        act.addWidget(b_undo, 1); act.addWidget(b_all, 1)
        sec_ann.addLayout(act)
        ll.addWidget(sec_ann)

        # ── STENT ──────────────────────────────────────────────────────────
        sec_stent = CollapsibleSection('STENT', starts_open=False)
        stent_info = QLabel(
            'Generer un stent (diametre + longueur) puis placer le centre et l axe sur la fluoroscopie.')
        stent_info.setObjectName('dim'); stent_info.setWordWrap(True)
        sec_stent.addWidget(stent_info)

        lbl_sapien = QLabel('Edwards SAPIEN 3 Transcatheter Heart Valve')
        lbl_sapien.setWordWrap(True)
        sec_stent.addWidget(lbl_sapien)

        self.cb_stent_preset = QComboBox()
        self.cb_stent_preset.addItem('20 mm × 15.5 mm', (20.0, 15.5))
        self.cb_stent_preset.addItem('23 mm × 18 mm',   (23.0, 18.0))
        self.cb_stent_preset.addItem('26 mm × 20 mm',   (26.0, 20.0))
        self.cb_stent_preset.addItem('29 mm × 22.5 mm', (29.0, 22.5))
        self.cb_stent_preset.addItem('Custom', None)
        self.cb_stent_preset.currentIndexChanged.connect(self._on_stent_preset_changed)
        sec_stent.addWidget(self.cb_stent_preset)

        stent_params = QHBoxLayout(); stent_params.setSpacing(6)
        stent_params.addWidget(_lbl('Diam (mm)'), 0)
        self.sp_stent_D = QDoubleSpinBox(); self.sp_stent_D.setRange(3, 40)
        self.sp_stent_D.setValue(20.0); self.sp_stent_D.setSingleStep(0.5)
        stent_params.addWidget(self.sp_stent_D, 0)
        stent_params.addWidget(_lbl('Long (mm)'), 0)
        self.sp_stent_L = QDoubleSpinBox(); self.sp_stent_L.setRange(5, 60)
        self.sp_stent_L.setValue(15.5); self.sp_stent_L.setSingleStep(1)
        self.sp_stent_L.valueChanged.connect(lambda _v: self._ms_recompute())
        stent_params.addWidget(self.sp_stent_L, 0)
        stent_params.addStretch()
        sec_stent.addLayout(stent_params)
        self._on_stent_preset_changed(0)

        stent_btns = QHBoxLayout(); stent_btns.setSpacing(6)
        self.btn_stent_gen = QPushButton('Generer stent')
        self.btn_stent_gen.setObjectName('primary')
        self.btn_stent_gen.clicked.connect(self._stent_generate)
        self.btn_stent_place = QPushButton('Placer sur fluoro')
        self.btn_stent_place.setCheckable(True)
        self.btn_stent_place.setEnabled(False)
        self.btn_stent_place.toggled.connect(self._toggle_stent_mode)
        stent_btns.addWidget(self.btn_stent_gen, 1)
        stent_btns.addWidget(self.btn_stent_place, 1)
        sec_stent.addLayout(stent_btns)

        stent_row = QHBoxLayout(); stent_row.setSpacing(6)
        self.chk_stent_show = QCheckBox('Afficher')
        self.chk_stent_show.setChecked(True)
        self.chk_stent_show.toggled.connect(self._update_stent_overlay)
        stent_row.addWidget(self.chk_stent_show)
        stent_row.addStretch()
        sec_stent.addLayout(stent_row)

        self.lbl_stent_status = QLabel('--')
        self.lbl_stent_status.setObjectName('dim')
        self.lbl_stent_status.setWordWrap(True)
        sec_stent.addWidget(self.lbl_stent_status)

        ll.addWidget(sec_stent)

        # ── TAVI RISK (MS length + ΔMSID + risque PM-dependency) ─────────────
        sec_tavi = CollapsibleSection('TAVI RISK', starts_open=False)
        tavi_info = QLabel(
            'Auto : les 2 hinges (ligne annulaire) sont deduits des nadirs\n'
            'des cusps L1/L2/L3 segmentes. Reste a cliquer le MS apex.\n'
            'ID = distance ligne annulaire -> base du stent (pas de NCC).')
        tavi_info.setObjectName('dim'); tavi_info.setWordWrap(True)
        sec_tavi.addWidget(tavi_info)

        self.btn_auto_hinges = QPushButton('Auto-placer les hinges')
        self.btn_auto_hinges.setObjectName('primary')
        self.btn_auto_hinges.setToolTip(
            'Detecte le nadir (point le plus ventriculaire) de chaque cusp segmente '
            '(L1/L2/L3) et place hinge L / hinge R = les 2 nadirs les plus ecartes. '
            'Necessite au moins 2 cusps/leaflets segmentes.')
        self.btn_auto_hinges.clicked.connect(self._auto_place_hinges)
        sec_tavi.addWidget(self.btn_auto_hinges)

        self.btn_ms_h1 = QPushButton('1. Hinge L (cusp gauche)'); self.btn_ms_h1.setCheckable(True)
        self.btn_ms_h2 = QPushButton('2. Hinge R (cusp droit)'); self.btn_ms_h2.setCheckable(True)
        self.btn_ms_apex = QPushButton('3. MS apex (septum)'); self.btn_ms_apex.setCheckable(True)
        for btn, key in ((self.btn_ms_h1, 'hinge1'),
                          (self.btn_ms_h2, 'hinge2'),
                          (self.btn_ms_apex, 'ms')):
            btn.toggled.connect(lambda c, k=key: self._ms_arm(k if c else None))
            btn.setToolTip(self._MS_HINT.get(key, ''))
            sec_tavi.addWidget(btn)

        self.btn_ms_reset = QPushButton('Reinitialiser les points')
        self.btn_ms_reset.setObjectName('warn')
        self.btn_ms_reset.clicked.connect(self._ms_reset)
        sec_tavi.addWidget(self.btn_ms_reset)

        self.lbl_ms_value = QLabel('MS length : --')
        self.lbl_ms_value.setObjectName('metric')
        self.lbl_ms_value.setAlignment(Qt.AlignCenter)
        sec_tavi.addWidget(self.lbl_ms_value)

        self.lbl_id = QLabel('ID : --'); self.lbl_id.setObjectName('dim')
        self.lbl_delta = QLabel('DeltaMSID : --'); self.lbl_delta.setObjectName('dim')
        self.lbl_risk = QLabel('Risque PM : --')
        self.lbl_risk.setWordWrap(True)
        sec_tavi.addWidget(self.lbl_id)
        sec_tavi.addWidget(self.lbl_delta)
        sec_tavi.addWidget(self.lbl_risk)

        ll.addWidget(sec_tavi)

        # ── DETECTION YOLO ────────────────────────────────────────────────────
        sec_yolo = CollapsibleSection('DETECTION YOLO', starts_open=False)

        row_yolo_load = QHBoxLayout(); row_yolo_load.setSpacing(4)
        self.btn_load_yolo = QPushButton('Charger un autre modèle')
        self.btn_load_yolo.clicked.connect(self._load_yolo_model)
        row_yolo_load.addWidget(self.btn_load_yolo, 1)
        self.lbl_yolo = QLabel(''); self.lbl_yolo.setObjectName('dim')
        row_yolo_load.addWidget(self.lbl_yolo)
        sec_yolo.addLayout(row_yolo_load)

        yolo_grid = QWidget(); yolo_grid.setStyleSheet('background:transparent;')
        ygl = QGridLayout(yolo_grid); ygl.setContentsMargins(0,0,0,0); ygl.setSpacing(4)
        ygl.addWidget(_lbl('Confiance (%)'), 0, 0)
        self.sp_yolo_conf = QSpinBox(); self.sp_yolo_conf.setRange(1, 95); self.sp_yolo_conf.setValue(25)
        ygl.addWidget(self.sp_yolo_conf, 0, 1)
        ygl.addWidget(_lbl('IoU NMS (%)'), 1, 0)
        self.sp_yolo_iou = QSpinBox(); self.sp_yolo_iou.setRange(1, 95); self.sp_yolo_iou.setValue(45)
        ygl.addWidget(self.sp_yolo_iou, 1, 1)
        ygl.addWidget(_lbl('imgsz (px)'), 2, 0)
        self.sp_yolo_imgsz = QSpinBox(); self.sp_yolo_imgsz.setRange(0, 2048); self.sp_yolo_imgsz.setValue(288)
        ygl.addWidget(self.sp_yolo_imgsz, 2, 1)
        ygl.addWidget(_lbl('Gamma'), 3, 0)
        self.sp_yolo_gamma = QDoubleSpinBox(); self.sp_yolo_gamma.setRange(0.1, 5.0); self.sp_yolo_gamma.setValue(0.65); self.sp_yolo_gamma.setSingleStep(0.1)
        ygl.addWidget(self.sp_yolo_gamma, 3, 1)
        ygl.addWidget(_lbl('Contraste'), 4, 0)
        self.sp_yolo_contrast = QDoubleSpinBox(); self.sp_yolo_contrast.setRange(0.1, 5.0); self.sp_yolo_contrast.setValue(1.5); self.sp_yolo_contrast.setSingleStep(0.1)
        ygl.addWidget(self.sp_yolo_contrast, 4, 1)
        self.chk_yolo_invert = QCheckBox('Inverser niveaux'); self.chk_yolo_invert.setChecked(True); ygl.addWidget(self.chk_yolo_invert, 5, 0, 1, 2)
        sec_yolo.addWidget(yolo_grid)

        row_det = QHBoxLayout(); row_det.setSpacing(4)
        self.btn_detect_fl = QPushButton('Détecter Fluoro'); self.btn_detect_fl.setObjectName('primary')
        self.btn_detect_fl.clicked.connect(self._detect_fluoro)
        row_det.addWidget(self.btn_detect_fl, 1)
        self.btn_detect_drr = QPushButton('Détecter DRR'); self.btn_detect_drr.setObjectName('primary')
        self.btn_detect_drr.clicked.connect(self._detect_drr)
        row_det.addWidget(self.btn_detect_drr, 1)
        sec_yolo.addLayout(row_det)

        self.lbl_yolo_status = QLabel(''); self.lbl_yolo_status.setObjectName('dim')
        self.lbl_yolo_status.setWordWrap(True)
        sec_yolo.addWidget(self.lbl_yolo_status)

        # État détections
        self._yolo_det_fl = None    # résultat detect_vertebrae fluoro
        self._yolo_det_drr = None   # résultat detect_vertebrae DRR

        ll.addWidget(sec_yolo)

        # ── RECALAGE ──────────────────────────────────────────────────────────
        sec_reg = CollapsibleSection('RECALAGE', starts_open=False)

        self.btn_reg = QPushButton('Lancer le recalage'); self.btn_reg.setObjectName('success')
        self.btn_reg.setEnabled(False); self.btn_reg.clicked.connect(self.run_registration)
        sec_reg.addWidget(self.btn_reg)
        self.chk_elastic = QCheckBox('Recalage elastique (deformation libre)')
        self.chk_elastic.setToolTip(
            'Active une deformation libre (FFD 4x4) apres le recalage rigide.\n'
            'Ameliore le IoU sur les formes complexes — plus lent (~15-30 s).')
        sec_reg.addWidget(self.chk_elastic)
        ll.addWidget(sec_reg)

        # ── ACTIONS ──────────────────────────────────────────────────────────
        sec_act = CollapsibleSection('ACTIONS', starts_open=False)

        btn_exp = QPushButton('Exporter resultats')
        btn_exp.clicked.connect(self.export_results)
        sec_act.addWidget(btn_exp)
        ll.addWidget(sec_act)

        # ── ITERATIONS ────────────────────────────────────────────────────────
        sec_iter = CollapsibleSection('ITERATIONS', starts_open=False)

        self._iter_container = QWidget()
        self._iter_container.setStyleSheet('background:transparent;')
        self._iter_vbox = QVBoxLayout(self._iter_container)
        self._iter_vbox.setContentsMargins(0, 0, 0, 0)
        self._iter_vbox.setSpacing(4)
        self._lbl_no_iter = QLabel('Aucune iteration')
        self._lbl_no_iter.setObjectName('dim')
        self._iter_vbox.addWidget(self._lbl_no_iter)
        sec_iter.addWidget(self._iter_container)

        self.btn_compare = QPushButton('Comparer les iterations')
        self.btn_compare.setObjectName('primary')
        self.btn_compare.setEnabled(False)
        self.btn_compare.clicked.connect(self._open_comparison)
        sec_iter.addWidget(self.btn_compare)

        ll.addWidget(sec_iter)

        # ── PIPELINE AUTO (bouton magique) ────────────────────────────────────
        sec_auto = CollapsibleSection('PIPELINE AUTO', starts_open=True)

        auto_info = QLabel(
            'DRR + Détection YOLO + Appariement auto + Recalage élastique\n'
            'Nécessite : CT, Fluoroscopie DICOM et modèle YOLO.\n'
            'Fallback : sélection manuelle si YOLO échoue.')
        auto_info.setObjectName('dim'); auto_info.setWordWrap(True)
        sec_auto.addWidget(auto_info)

        self.btn_auto = QPushButton('LANCER LE PIPELINE COMPLET')
        self.btn_auto.setObjectName('success')
        self.btn_auto.setStyleSheet(
            f'QPushButton{{background:#0d2a1a;border:2px solid {ACCENT2};color:{ACCENT2};'
            f'font-weight:700;font-size:13px;min-height:42px;border-radius:6px;letter-spacing:1px;}}'
            f'QPushButton:hover{{background:#1a3d28;color:#fff;}}'
            f'QPushButton:disabled{{background:{DARK_BG};border-color:{TEXT_DIM};color:{TEXT_DIM};}}')
        self.btn_auto.clicked.connect(self._run_auto_pipeline)
        sec_auto.addWidget(self.btn_auto)

        self.btn_detect_vertebrae = QPushButton('DÉTECTER VERTÈBRES')
        self.btn_detect_vertebrae.setObjectName('info')
        self.btn_detect_vertebrae.setStyleSheet(
            f'QPushButton{{background:#1a2a3a;border:2px solid {ACCENT};color:{ACCENT};'
            f'font-weight:700;font-size:12px;min-height:36px;border-radius:6px;}}'
            f'QPushButton:hover{{background:#2a3a4a;color:#fff;}}'
            f'QPushButton:disabled{{background:{DARK_BG};border-color:{TEXT_DIM};color:{TEXT_DIM};}}')
        self.btn_detect_vertebrae.clicked.connect(self._detect_vertebrae_manual)
        sec_auto.addWidget(self.btn_detect_vertebrae)

        self.lbl_auto_status = QLabel('')
        self.lbl_auto_status.setObjectName('dim')
        self.lbl_auto_status.setWordWrap(True)
        sec_auto.addWidget(self.lbl_auto_status)

        ll.addWidget(sec_auto)

        ll.addStretch()


        # ── RESULTATS ─────────────────────────────────────────────────────────
        sec_res = CollapsibleSection('RESULTATS', starts_open=True)

        metrics_row = QHBoxLayout(); metrics_row.setSpacing(10)
        m_iou = QWidget(); m_iou.setStyleSheet(f'background:{CARD_BG};border-radius:6px;border:1px solid {BORDER2};')
        iou_v = QVBoxLayout(m_iou); iou_v.setContentsMargins(8, 6, 8, 6); iou_v.setSpacing(1)
        iou_v.addWidget(QLabel('IoU'), alignment=Qt.AlignCenter)
        self.lbl_iou = QLabel('--'); self.lbl_iou.setObjectName('metric'); self.lbl_iou.setAlignment(Qt.AlignCenter)
        iou_v.addWidget(self.lbl_iou)

        m_dice = QWidget(); m_dice.setStyleSheet(f'background:{CARD_BG};border-radius:6px;border:1px solid {BORDER2};')
        dice_v = QVBoxLayout(m_dice); dice_v.setContentsMargins(8, 6, 8, 6); dice_v.setSpacing(1)
        dice_v.addWidget(QLabel('Dice'), alignment=Qt.AlignCenter)
        self.lbl_dice = QLabel('--'); self.lbl_dice.setObjectName('metric'); self.lbl_dice.setAlignment(Qt.AlignCenter)
        dice_v.addWidget(self.lbl_dice)

        metrics_row.addWidget(m_iou, 1); metrics_row.addWidget(m_dice, 1)
        sec_res.addLayout(metrics_row)

        self.lbl_tx = QLabel('tx : --'); self.lbl_tx.setObjectName('dim')
        self.lbl_ty = QLabel('ty : --'); self.lbl_ty.setObjectName('dim')
        self.lbl_rot = QLabel('rot : --'); self.lbl_rot.setObjectName('dim')
        self.lbl_scale = QLabel('scale : --'); self.lbl_scale.setObjectName('dim')
        for l in [self.lbl_tx, self.lbl_ty, self.lbl_rot, self.lbl_scale]:
            sec_res.addWidget(l)
        ll.addWidget(sec_res)



        
        # ── PROGRESSION ──────────────────────────────────────────────────────
        self.prog_bar = QProgressBar(); self.prog_bar.setRange(0, 100); self.prog_bar.setValue(0)
        ll.addWidget(self.prog_bar)
        self.lbl_prog = QLabel('')
        self.lbl_prog.setObjectName('dim'); self.lbl_prog.setAlignment(Qt.AlignCenter)
        self.lbl_prog.setWordWrap(True)
        ll.addWidget(self.lbl_prog)


        # ── Onglets ───────────────────────────────────────────────────────────
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        self.cv_fl = AnnotationCanvas(); self.cv_fl.mask_updated.connect(self._on_mask_upd)
        self.cv_fl.wheel_scrolled.connect(self._step_fluoro_frame)
        self.cv_fl.stent_pose_changed.connect(self._on_stent_pose_changed)
        self.cv_fl.point_picked.connect(self._on_point_picked)
        self.cv_drr = AnnotationCanvas(); self.cv_drr.mask_updated.connect(self._on_mask_upd)
        self.result_panel = ResultPanel()
        self.seg_review_panel = SegmentationReviewPanel()
        self.seg_review_panel.view_clicked.connect(self._on_seg_view_clicked)

        for cv, hint, label in [
            (self.cv_fl, 'Image fixe -- dessiner les structures a recaler', 'Fixe'),
            (self.cv_drr, 'Image mobile -- dessiner les memes structures', 'Mobile'),
        ]:
            self.tabs.addTab(self._wrap(cv, hint), label)
        self.tabs.addTab(self.seg_review_panel, 'Seg CT')
        self.seg_review_panel.request_3d_view.connect(self._open_segmentation_3d)
        self.tabs.addTab(self.result_panel, 'Resultat')

        self.overlay_panel = FinalOverlayPanel()
        self.tabs.addTab(self.overlay_panel, 'Overlay')

        for cv in [self.cv_fl, self.cv_drr]:
            cv.set_tool('pencil'); cv.set_active('vertebrae')

        root.addWidget(self.tabs, 1)

        # Dock lateral d'icones a l'extreme gauche (ouvre/ferme des panneaux).
        self._build_tool_dock()
        root.insertWidget(0, self._tool_dock)

        self._busy_overlay = BusyOverlay(cw)
        self._busy_overlay.setGeometry(cw.rect())
        self.setStatusBar(QStatusBar())

    def _wrap(self, cv, hint):
        w = QWidget(); w.setStyleSheet(f'background:{PANEL_BG};')
        l = QVBoxLayout(w); l.setContentsMargins(4, 4, 4, 4); l.setSpacing(6)
        
        # ── Sélection de structure à annoter ──────────────────────────────────
        struct_ctrl = QHBoxLayout(); struct_ctrl.setSpacing(6)
        struct_lbl = QLabel('Structure :')
        struct_lbl.setObjectName('dim')
        struct_ctrl.addWidget(struct_lbl)
        
        struct_btns = QButtonGroup()
        for i, (name, info) in enumerate(STRUCT.items()):
            btn = QPushButton(info['label'])
            btn.setObjectName('tool')
            btn.setCheckable(True)
            btn.setChecked(i == 0)  # vertebrae par défaut
            r, g, b = info['rgb']
            btn.setStyleSheet(
                f"QPushButton {{"
                f"  border: 2px solid {BORDER2}; border-radius: 4px;"
                f"  min-height: 24px; padding: 2px 8px;"
                f"  background: rgba({r-40},{g-40},{b-40},255);"
                f"}}"
                f"QPushButton:checked {{"
                f"  border: 2px solid rgb({r},{g},{b});"
                f"  color: rgb({r},{g},{b}); font-weight: 600;"
                f"}}")
            btn.clicked.connect(lambda checked, name=name: self._set_active_struct(cv, name))
            struct_btns.addButton(btn)
            struct_ctrl.addWidget(btn)
        cv._struct_btns = struct_btns
        
        struct_ctrl.addStretch()
        l.addLayout(struct_ctrl)
        
        l.addWidget(cv, 1)
        return w

    # ── Barre de menus + dock lateral (refonte "Photoshop medical") ────────────

    def _build_menu_bar(self):
        """Barre de menus standardisee, sobre, cablee aux actions existantes."""
        mb = self.menuBar()
        mb.setStyleSheet(
            f'QMenuBar{{background:{PANEL_BG};color:{TEXT};border-bottom:1px solid {BORDER};'
            f'padding:2px 4px;}}'
            f'QMenuBar::item{{background:transparent;padding:5px 12px;border-radius:4px;}}'
            f'QMenuBar::item:selected{{background:{CARD_BG};color:{ACCENT};}}'
            f'QMenu{{background:{PANEL_BG};color:{TEXT};border:1px solid {BORDER2};padding:4px;}}'
            f'QMenu::item{{padding:6px 24px 6px 14px;border-radius:4px;}}'
            f'QMenu::item:selected{{background:{CARD_BG};color:{ACCENT};}}'
            f'QMenu::separator{{height:1px;background:{BORDER};margin:4px 8px;}}'
        )

        # Fichier
        m_file = mb.addMenu('Fichier')
        m_file.addAction('Ouvrir des fichiers...', self._on_browse)
        m_file.addAction('Parcourir...', self._on_browse)
        m_file.addSeparator()
        m_file.addAction('Exporter les masques', self._export_totalseg_masks)
        m_file.addAction('Sauvegarder l\'iteration', self._save_iteration)
        m_file.addSeparator()
        m_file.addAction('Quitter', self.close)

        # Edition
        m_edit = mb.addMenu('Edition')
        m_edit.addAction('Annuler', self._undo)
        m_edit.addAction('Effacer les annotations', self._clear_all)

        # Outils
        m_tools = mb.addMenu('Outils')
        m_tools.addAction('Segmentation (TotalSegmentator)', self._run_totalseg)
        m_tools.addAction('Pipeline automatique', self._run_auto_pipeline)

        # Affichage : navigation entre les onglets + sidebar
        m_view = mb.addMenu('Affichage')
        tab_names = ['Image fixe', 'Image mobile', 'Segmentation CT',
                     'Resultat', 'Overlay']
        for i, label in enumerate(tab_names):
            m_view.addAction(label, lambda _=False, idx=i: self.tabs.setCurrentIndex(idx))
        m_view.addSeparator()
        m_view.addAction('Afficher / masquer le panneau lateral',
                         self._toggle_sidebar)

        # Fenetres : vues 3D
        m_win = mb.addMenu('Fenetres')
        m_win.addAction('Vue 3D segmentation CT', self._open_segmentation_3d)
        m_win.addAction('Vue 3D overlay recale',
                        lambda: self.overlay_panel._open_3d_view())

        # Aide
        m_help = mb.addMenu('Aide')
        m_help.addAction('A propos', self._show_about)

    def _toggle_sidebar(self):
        if hasattr(self, '_left_outer'):
            # isHidden() reflete l'etat explicite (independant des ancetres).
            self._left_outer.setVisible(self._left_outer.isHidden())

    def _show_about(self):
        QMessageBox.about(
            self, 'A propos',
            '2D/3D Registration\n\n'
            'Visualisation et recalage 2D/3D pour la planification TAVI.\n'
            'Proof of Concept clinique.')

    # Chemins d'icones Material (flat, monochrome, viewBox 0 0 24 24)
    _ICON_PANEL = 'M3 5h18v2H3V5m0 6h18v2H3v-2m0 6h12v2H3v-2z'

    def _build_tool_dock(self):
        """Dock lateral minimaliste : juste le toggle sidebar. Navigation via menu."""
        dock = QWidget()
        dock.setFixedWidth(52)
        dock.setStyleSheet(f'background:{DARK_BG};border-right:1px solid {BORDER};')
        v = QVBoxLayout(dock)
        v.setContentsMargins(6, 8, 6, 8)
        v.setSpacing(6)

        def _tool_btn(path_d, tip, slot):
            b = QPushButton()
            ic = _make_svg_icon(path_d, color=TEXT_MID, size=22)
            if ic is not None:
                b.setIcon(ic)
                b.setIconSize(QSize(22, 22))
            else:
                b.setText(tip[:1])
            b.setToolTip(tip)
            b.setFixedSize(40, 40)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                f'QPushButton{{background:transparent;border:none;border-radius:8px;}}'
                f'QPushButton:hover{{background:{CARD_BG};}}'
                f'QPushButton:pressed{{background:{BORDER2};}}'
            )
            b.clicked.connect(slot)
            v.addWidget(b)
            return b

        _tool_btn(self._ICON_PANEL, 'Afficher / masquer le panneau lateral',
                  self._toggle_sidebar)
        v.addStretch()

        self._tool_dock = dock

    # ── Slots UI ──────────────────────────────────────────────────────────────

    def _set_tool(self, tool):
        if self._stent_mode_active:
            self._set_stent_mode(False, update_button=True)
        for btn, t in [
            (self.btn_pencil, 'pencil'),
            (self.btn_rect, 'rectangle'),
            (self.btn_poly, 'polygon'),
            (self.btn_eraser, 'eraser'),
        ]:
            btn.setChecked(t == tool)
        for cv in [self.cv_fl, self.cv_drr]:
            cv.set_tool(tool)
        hints = {
            'pencil':    'Tracer le contour en continu, relacher pour valider',
            'rectangle': 'Cliquer-glisser pour dessiner un rectangle',
            'polygon':   'Cliquer pour poser les sommets, double-cliquer ou revenir au premier point pour fermer',
            'eraser':    'Cliquer-glisser pour effacer des annotations',
        }
        self._status(hints.get(tool, ''))

    def _set_active_struct(self, cv, struct_name):
        """Change la structure active pour annoter."""
        cv.set_active(struct_name)
        info = STRUCT.get(struct_name, {})
        hint = f'Structure active : {info.get("label", struct_name)}'
        self._status(hint)
        cv._refresh()

    def _on_pen(self,v): self.lbl_pen.setText(f'{v}px'); [cv.set_pen_radius(v) for cv in [self.cv_fl,self.cv_drr]]
    def _undo(self): self._active_cv().undo()
    def _clear_all(self): self._active_cv().clear_all()

    def _on_drr_preset_changed(self, _idx):
        preset_name = self.cb_drr_preset.currentData() or 'balanced'
        preset = DRR_POSTPROCESS_PRESETS.get(preset_name, DRR_POSTPROCESS_PRESETS['balanced'])
        self.sp_drr_bone.blockSignals(True)
        self.sp_drr_edges.blockSignals(True)
        self.sp_drr_gamma.blockSignals(True)
        self.sp_drr_bone.setValue(1.0)
        self.sp_drr_edges.setValue(1.0)
        self.sp_drr_gamma.setValue(float(preset.get('gamma', 1.0)))
        self.sp_drr_bone.blockSignals(False)
        self.sp_drr_edges.blockSignals(False)
        self.sp_drr_gamma.blockSignals(False)

    def _drr_postprocess_kw(self):
        preset_name = self.cb_drr_preset.currentData() or 'balanced'
        preset = dict(DRR_POSTPROCESS_PRESETS.get(preset_name, DRR_POSTPROCESS_PRESETS['balanced']))
        preset['bone_boost'] = float(preset.get('bone_boost', 0.0)) * self.sp_drr_bone.value()
        preset['gradient_boost'] = float(preset.get('gradient_boost', 0.0)) * self.sp_drr_edges.value()
        preset['gamma'] = self.sp_drr_gamma.value()
        if not self.chk_drr_soft.isChecked():
            preset['soft_tissue_suppression'] = 0.0
        if not self.chk_drr_clahe.isChecked():
            preset['clahe_enabled'] = False
        if not self.chk_drr_edges.isChecked():
            preset['gradient_boost'] = 0.0
            preset['tophat_enabled'] = False
            preset['unsharp_amount'] = min(float(preset.get('unsharp_amount', 0.0)), 0.18)
        return preset

    def _update_drr_filter_buttons(self):
        has_drr = self._drr_base_image is not None
        self.btn_drr_apply_filters.setEnabled(has_drr)
        self.btn_drr_reset_filters.setEnabled(has_drr)

    def _set_drr_base_image(self, img):
        self._drr_base_image = None if img is None else img.astype(np.float32).copy()
        self._update_drr_filter_buttons()

    def _refresh_drr_after_filter(self, status_msg=''):
        if self.drr_image is None:
            return
        self.cv_drr.set_image(self.drr_image, preserve_masks=True)
        if 0 <= self._current_iter_idx < len(self._iterations):
            self._iterations[self._current_iter_idx]['drr_image'] = self.drr_image.copy()
            self._refresh_iter_list()
        if self.result is not None:
            self._build_result(self.result)
        if status_msg:
            self._status(status_msg)

    def _apply_drr_filters(self):
        if self._drr_base_image is None:
            self._err('Generez ou chargez un DRR d abord.'); return
        self.drr_image = enhance_drr_image(self._drr_base_image, self._drr_postprocess_kw())
        self._refresh_drr_after_filter('Filtres appliques sur le DRR courant')

    def _reset_drr_filters(self):
        if self._drr_base_image is None:
            self._err('Aucun DRR de base disponible.'); return
        self.drr_image = self._drr_base_image.copy()
        self._refresh_drr_after_filter('Retour au DRR genere')

    def _save_mask(self):
        cv=self._active_cv(); m=cv.get_mask()
        if m is None or m.sum()==0: self._err('Masque vide'); return
        p,_=QFileDialog.getSaveFileName(self,'Sauvegarder','mask.png','PNG (*.png)')
        if p: cv2.imwrite(p,(m*255).astype(np.uint8)); self._status(f'Masque sauvegardé → {p}')

    def _active_cv(self): return self.cv_fl if self.tabs.currentIndex()==0 else self.cv_drr

    def _show_busy_overlay(self, title, message='Preparation...'):
        if not hasattr(self, '_busy_overlay'):
            return
        self._busy_overlay.show_busy(
            title=title,
            message=message,
            progress=self.prog_bar.value(),
            snapshot_widget=self.tabs,
        )

    def _hide_busy_overlay(self):
        if hasattr(self, '_busy_overlay'):
            self._busy_overlay.hide_busy()

    def _start_worker(self, task, kw, result_slot, error_slot,
                      busy_title=None, busy_message='Preparation...'):
        if busy_title:
            self._show_busy_overlay(busy_title, busy_message)
        self.worker = WorkerThread(task, kw)
        self.worker.progress.connect(self._on_prog)
        self.worker.result.connect(result_slot)
        self.worker.error.connect(error_slot)
        self.worker.start()

    def _on_mask_upd(self):
        mask_fl = self.cv_fl.get_mask()
        mask_drr = self.cv_drr.get_mask()
        has_fl = mask_fl is not None and mask_fl.sum() > 0
        has_drr = mask_drr is not None and mask_drr.sum() > 0
        self.btn_reg.setEnabled(bool(has_fl and has_drr))

    # ── Stent placement ─────────────────────────────────────────────────────

    def _on_stent_preset_changed(self, idx: int):
        dims = self.cb_stent_preset.itemData(idx)
        is_custom = dims is None
        self.sp_stent_D.setEnabled(is_custom)
        self.sp_stent_L.setEnabled(is_custom)
        if not is_custom:
            self.sp_stent_D.setValue(dims[0])
            self.sp_stent_L.setValue(dims[1])

    def _set_stent_mode(self, active: bool, update_button: bool = True):
        self._stent_mode_active = active
        self.cv_fl.set_stent_mode(active)
        if update_button and hasattr(self, 'btn_stent_place'):
            self.btn_stent_place.blockSignals(True)
            self.btn_stent_place.setChecked(active)
            self.btn_stent_place.blockSignals(False)

    def _toggle_stent_mode(self, checked: bool):
        self._set_stent_mode(checked, update_button=False)
        if checked:
            self._status('Mode stent: clic centre, glisser pour definir l axe')

    def _stent_generate(self):
        if self.fluoro_image is None:
            self._err('Chargez une fluoroscopie d abord.')
            return
        try:
            self.stent_mesh = generate_stent_mesh(
                diameter_mm=self.sp_stent_D.value(),
                length_mm=self.sp_stent_L.value(),
            )
        except Exception as ex:
            self._err(str(ex))
            return

        size = self.cv_fl.image_size()
        center = (float(size) * 0.5, float(size) * 0.5)
        self._stent_center_px = center
        self._stent_axis_deg = 0.0
        self._stent_base_mask = None
        self._stent_base_center = None
        self._stent_proj_params = None

        self.cv_fl.set_stent_pose(center, self._stent_axis_deg)
        self._update_stent_axis_length()
        self.btn_stent_place.setEnabled(True)
        self._set_stent_mode(True, update_button=True)
        self._update_stent_overlay()
        self.lbl_stent_status.setText('Stent genere — placez le centre et l axe sur la fluoroscopie')
        self._status('Stent genere. Placez le centre et l axe sur la fluoro.')

    def _on_stent_pose_changed(self, cx: float, cy: float, axis_deg: float):
        self._stent_center_px = (cx, cy)
        self._stent_axis_deg = float(axis_deg)
        self._update_stent_overlay()
        self._ms_recompute()

    def _update_stent_axis_length(self):
        if self.stent_mesh is None:
            self.cv_fl.set_stent_axis_length_px(None)
            return
        size = max(1, self.cv_fl.image_size())
        pix_mm = float(self.sp_fov.value()) / float(size)
        if pix_mm <= 0:
            pix_mm = 1.0
        length_px = float(self.sp_stent_L.value()) / pix_mm
        length_px = max(60.0, min(length_px, float(size) * 0.8))
        self.cv_fl.set_stent_axis_length_px(length_px)

    def _update_stent_overlay(self):
        if self.stent_mesh is None or not hasattr(self, 'chk_stent_show'):
            return
        if not self.chk_stent_show.isChecked():
            self.cv_fl.clear_overlay()
            return
        size = self.cv_fl.image_size()
        if size <= 0:
            return
        self._update_stent_axis_length()
        lao = float(self.sp_lao.value())
        cran = float(self.sp_cran.value()) + 180.0
        sid = float(self.dicom_meta.get('sid_mm', 1020.0))
        sod = float(self.dicom_meta.get('sod_mm', 510.0))
        fov = float(self.sp_fov.value())
        params = (size, lao, cran, sid, sod, fov)

        if self._stent_base_mask is None or self._stent_proj_params != params:
            try:
                base = project_stent_mask(
                    self.stent_mesh,
                    lao_deg=lao,
                    cran_deg=cran,
                    size=size,
                    sid_mm=sid,
                    sod_mm=sod,
                    fov_mm=fov,
                )
            except Exception as ex:
                self._err(f'Erreur projection stent: {ex}')
                return
            self._stent_base_mask = base
            self._stent_base_center = mask_center(base)
            self._stent_proj_params = params

        center = self._stent_center_px or self._stent_base_center
        if center is None:
            center = (float(size) * 0.5, float(size) * 0.5)
        mask = transform_mask(
            self._stent_base_mask,
            center_xy=center,
            angle_deg=self._stent_axis_deg,
            base_center=self._stent_base_center,
        )
        if mask is not None:
            self.cv_fl.set_overlay(mask, color=(255, 200, 80), alpha=0.35, line_width=2)

    def _clear_stent_state(self, message: str = ''):
        self.stent_mesh = None
        self._stent_base_mask = None
        self._stent_base_center = None
        self._stent_proj_params = None
        self._stent_center_px = None
        self._stent_axis_deg = 0.0
        if hasattr(self, 'btn_stent_place'):
            self.btn_stent_place.setEnabled(False)
            self._set_stent_mode(False, update_button=True)
        if hasattr(self, 'chk_stent_show'):
            self.cv_fl.clear_overlay()
        if hasattr(self, 'lbl_stent_status') and message:
            self.lbl_stent_status.setText(message)
        # ID dépend du stent : invalider et rafraîchir le panneau TAVI
        if hasattr(self, 'lbl_ms_value'):
            self._ms_recompute()

    # ── TAVI risk (MS length + ΔMSID + risque PM-dependency) ────────────────

    _MS_BUTTONS = ('btn_ms_h1', 'btn_ms_h2', 'btn_ms_apex')
    _MS_ORDER = ('hinge1', 'hinge2', 'ms')
    _MS_HINT = {
        'hinge1': 'Hinge L (cusp gauche) — base de la valve aortique côté gauche. Repérez sur coupe coronale (axe vertical) : c\'est la jonction valve-paroi ventriculaire. Cliquez pour marquer.',
        'hinge2': 'Hinge R (cusp droit) — base de la valve aortique côté droit. Même coupe coronale que Hinge L. Complète la ligne annulaire (plan de référence pour MS et ID).',
        'ms':     'MS (Membranous Septum) — apex/pointe du septum membraneux à la base du cusp non-coronaire. C\'est le début du septum musculaire. Marque le haut du risque de conduction cardiaque.',
    }
    _MS_MARKER = {
        'hinge1': ((100, 200, 255), 'Hinge L'),
        'hinge2': ((255, 200, 100), 'Hinge R'),
        'ms':     ((255, 100, 100), 'MS'),
    }

    def _ms_arm(self, key):
        """Arme le prochain clic sur la coupe pour ``key`` (hinge1/hinge2/ms)."""
        for name, k in zip(self._MS_BUTTONS, self._MS_ORDER):
            btn = getattr(self, name)
            btn.blockSignals(True)
            btn.setChecked(k == key)
            btn.blockSignals(False)
        self._ms_click_target = key
        if hasattr(self, 'seg_review_panel'):
            self.seg_review_panel.set_click_mode(key)
        if key:
            self.tabs.setCurrentWidget(self.seg_review_panel)
            self._status(self._MS_HINT[key])

    def _ms_reset(self):
        self._ms_world = {}
        if hasattr(self, 'seg_review_panel'):
            self.seg_review_panel.clear_markers()
        self._ms_arm(None)
        self._ms_update_labels(None, None)
        self._status('Mesures TAVI reinitialisees.')

    def _on_seg_view_clicked(self, plane, col, row, slice_idx):
        if not self._ms_click_target or self.ct_vol is None:
            return
        voxel = voxel_from_view_pixel(plane, col, row, self.ct_vol.shape, slice_idx)
        world = world_from_voxel(voxel, self.ct_aff)
        self._set_ms_point(self._ms_click_target, np.asarray(world, dtype=np.float64))
        # Auto-advance vers le prochain point manquant
        next_key = next((k for k in self._MS_ORDER if k not in self._ms_world), None)
        self._ms_arm(next_key)

    def _set_ms_point(self, key, world_mm):
        """Stocke un point MS/NCC en coords monde et met à jour le marker 3D."""
        if key not in self._MS_MARKER:
            return
        self._ms_world[key] = np.asarray(world_mm, dtype=np.float64)
        color, label = self._MS_MARKER[key]
        if self.ct_aff is not None:
            voxel = voxel_from_world(world_mm, self.ct_aff)
            self.seg_review_panel.set_marker(key, voxel, color, label, world_mm=world_mm)
        self._ms_recompute()

    def _on_point_picked(self, name, x, y):
        # Le clic sur fluoro ne sert plus aux points TAVI ; conservé si un autre
        # mode pick s'ajoute plus tard.
        pass

    def _ct_center_world(self):
        if self.ct_vol is None or self.ct_aff is None:
            return None
        sx, sy, sz = (int(v) for v in self.ct_vol.shape)
        return world_from_voxel(((sx - 1) * 0.5, (sy - 1) * 0.5, (sz - 1) * 0.5), self.ct_aff)

    def _aortic_anchor_world(self):
        """Centre 3D-monde de la region aortique pour ancrer la profondeur du stent.

        Priorite : leaflets (cusp/valve/L1-L3/aortic_root) -> aorte (tiers caudal)
        -> centre du volume CT (fallback ultime).
        """
        if self.seg_masks and self.ct_aff is not None:
            import re
            kw_strict = ('leaflet', 'cusp', 'valve', 'aortic_root')
            l_token = re.compile(r'(?:^|[^a-z0-9])l[123](?:$|[^a-z0-9])')
            combined = None
            for name, m in self.seg_masks.items():
                n = str(name).lower()
                hit = any(k in n for k in kw_strict) or bool(l_token.search(n))
                if hit and getattr(m, 'ndim', 0) == 3 and (m > 0).any():
                    combined = (m > 0) if combined is None else (combined | (m > 0))
            if combined is None:
                # Repli : tiers caudal de l'aorte (≈ racine aortique)
                for name, m in self.seg_masks.items():
                    n = str(name).lower()
                    if ('aorta' in n and 'pulmonary' not in n
                            and getattr(m, 'ndim', 0) == 3 and (m > 0).any()):
                        arr = (m > 0).astype(bool)
                        idx = np.argwhere(arr)
                        if idx.size:
                            z_min = int(idx[:, 2].min()); z_max = int(idx[:, 2].max())
                            z_cut = z_min + (z_max - z_min) // 3
                            sub = np.zeros_like(arr)
                            sub[:, :, z_min:z_cut + 1] = arr[:, :, z_min:z_cut + 1]
                            combined = sub if combined is None else (combined | sub)
            if combined is not None and combined.any():
                centroid = np.argwhere(combined).mean(axis=0)
                return world_from_voxel(centroid, self.ct_aff)
        return self._ct_center_world()

    def _collect_cusp_masks(self):
        """Masques individuels des cusps/leaflets (L1/L2/L3), root global exclu."""
        if not self.seg_masks:
            return {}
        import re
        l_token = re.compile(r'(?:^|[^a-z0-9])l[123](?:$|[^a-z0-9])')
        kw = ('leaflet', 'cusp')
        cusps = {}
        for name, m in self.seg_masks.items():
            n = str(name).lower()
            if 'aortic_root' in n:           # le root global n'est pas un cusp isole
                continue
            hit = any(k in n for k in kw) or bool(l_token.search(n))
            if hit and getattr(m, 'ndim', 0) == 3 and (m > 0).any():
                cusps[name] = (m > 0)
        return cusps

    def _aortic_axis_world(self, world_cloud):
        """Axe aortique unitaire (normale au plan annulaire), oriente cranialement.

        = plus petite composante PCA du nuage de cusps (les cusps s'etalent dans
        le plan annulaire et sont minces le long de l'axe). Oriente vers l'aorte
        (craniale) si disponible, sinon vers +z monde.
        """
        centroid = world_cloud.mean(axis=0)
        cov = np.cov((world_cloud - centroid).T)
        _, eigvec = np.linalg.eigh(cov)      # colonnes triees par valeur propre croissante
        u = eigvec[:, 0]                     # plus petite variance = normale annulaire
        cranial_ref = None
        if self.ct_aff is not None:
            for name, m in self.seg_masks.items():
                n = str(name).lower()
                if ('aorta' in n and 'pulmonary' not in n
                        and getattr(m, 'ndim', 0) == 3 and (m > 0).any()):
                    a_idx = np.argwhere(m > 0).astype(np.float64)
                    a_world = (np.asarray(self.ct_aff) @ np.column_stack(
                        [a_idx, np.ones(len(a_idx))]).T).T[:, :3]
                    cranial_ref = a_world.mean(axis=0) - centroid
                    break
        if cranial_ref is None:
            cranial_ref = np.array([0.0, 0.0, 1.0])   # +z monde ~ cranial
        if np.dot(u, cranial_ref) < 0:
            u = -u
        n = float(np.linalg.norm(u))
        return u / n if n > 1e-9 else u

    def _auto_place_hinges(self):
        """Place hinge1/hinge2 = nadirs des 2 cusps les plus ecartes (ligne annulaire)."""
        if self.ct_aff is None:
            self._err('Charger un CT + segmentation avant l\'auto-placement.')
            return
        cusps = self._collect_cusp_masks()
        if len(cusps) < 2:
            self._err('Auto-hinges : au moins 2 cusps/leaflets segmentes (L1/L2/L3) requis.')
            return

        aff = np.asarray(self.ct_aff)

        def to_world(vox_idx):
            v = np.column_stack([vox_idx, np.ones(len(vox_idx))])
            return (aff @ v.T).T[:, :3]

        # Nuage combine (sous-echantillonne) pour l'axe aortique
        all_idx = np.vstack([np.argwhere(mk) for mk in cusps.values()]).astype(np.float64)
        if len(all_idx) > 20000:
            sel = np.random.default_rng(0).choice(len(all_idx), 20000, replace=False)
            all_idx = all_idx[sel]
        u = self._aortic_axis_world(to_world(all_idx))

        # Nadir de chaque cusp = voxel le plus caudal (projection minimale sur u)
        nadirs = {}
        for name, mk in cusps.items():
            idx = np.argwhere(mk).astype(np.float64)
            w = to_world(idx)
            nadirs[name] = w[int(np.argmin(w @ u))]

        # Paire de nadirs la plus ecartee = corde annulaire la plus large
        names = list(nadirs)
        best, best_d = None, -1.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                d = float(np.linalg.norm(nadirs[names[i]] - nadirs[names[j]]))
                if d > best_d:
                    best_d, best = d, (names[i], names[j])
        na, nb = nadirs[best[0]], nadirs[best[1]]
        # Etiquetage L/R : x monde croissant = colonne des vues coronales
        if na[0] > nb[0]:
            na, nb = nb, na
        self._set_ms_point('hinge1', na)
        self._set_ms_point('hinge2', nb)
        self.tabs.setCurrentWidget(self.seg_review_panel)
        self._status(
            f'Hinges auto-places (cusps {best[0]} / {best[1]}, ecart {best_d:.1f} mm). '
            f'Cliquez maintenant MS apex pour completer le score.')

    def _stent_pix_mm(self):
        """Retourne le mm/pixel calibre sur le stent affiche (source de verite metrique).

        Le stent rendu sur le canvas fluoro a une longueur en pixels potentiellement
        ecretee par le canvas (cap min/max). On utilise cette longueur affichee
        comme calibre car c'est CE stent que l'utilisateur positionne sur le
        stent reel visible. Fallback sur fov_mm/size si pas de stent.
        """
        size = max(1, self.cv_fl.image_size())
        len_mm = float(self.sp_stent_L.value())
        len_px = getattr(self.cv_fl, '_stent_axis_len', None)
        if len_px and len_px > 0 and len_mm > 0:
            return len_mm / float(len_px)
        # Fallback : pix_mm theorique a partir du FOV DICOM
        fov = float(self.sp_fov.value())
        return (fov / float(size)) if fov > 0 else 1.0

    def _stent_endpoints_fluoro(self):
        """Retourne (top_px, bot_px) extremites du stent en pixels fluoro (canvas).

        On lit la longueur en pixels REELLEMENT affichee sur le canvas (qui peut
        avoir ete ecretee). C'est cette geometrie visible qui sert de reference
        pour toutes les mesures 2D.
        """
        if self._stent_center_px is None or self.stent_mesh is None:
            return None
        len_px = getattr(self.cv_fl, '_stent_axis_len', None)
        if not len_px or len_px <= 0:
            return None
        cx, cy = self._stent_center_px
        ang = math.radians(self._stent_axis_deg)
        half_px = float(len_px) * 0.5
        # AnnotationCanvas convention : direction = (cos, -sin) en pixels
        end1 = (cx + math.cos(ang) * half_px, cy - math.sin(ang) * half_px)
        end2 = (cx - math.cos(ang) * half_px, cy + math.sin(ang) * half_px)
        return end1, end2

    def _project_world_to_fluoro(self, world_pt):
        """Projette un point CT 3D-monde vers le pixel fluoro via cone-beam DRR + recalage."""
        if self.ct_vol is None or self.ct_aff is None:
            return None
        ct_c = self._ct_center_world()
        if ct_c is None:
            return None
        size = max(1, self.cv_fl.image_size())
        reg = self.result if (isinstance(getattr(self, 'result', None), dict)
                              and 'tx' in self.result) else None
        return project_world_to_fluoro_pixel(
            world_pt, ct_c, size,
            lao_deg=float(self.sp_lao.value()),
            cran_deg=float(self.sp_cran.value()) + 180.0,
            sid_mm=float(self.dicom_meta.get('sid_mm', 1020.0)),
            sod_mm=float(self.dicom_meta.get('sod_mm', 510.0)),
            fov_mm=float(self.sp_fov.value()),
            registration_result=reg,
        )

    @staticmethod
    def _id_depth_px(stent_ends, h1, h2, ms_fl=None):
        """Profondeur (px) de la base du stent sous la ligne annulaire h1-h2.

        = distance perpendiculaire signee de l'extremite ventriculaire du stent
        a la ligne annulaire. Le cote ventriculaire est determine par le MS si
        fourni (le septum membraneux descend du cote ventricule) ; sinon on prend
        la plus grande distance perpendiculaire absolue.
        """
        h1 = np.asarray(h1, dtype=np.float64); h2 = np.asarray(h2, dtype=np.float64)
        u = h2 - h1
        nu = float(np.linalg.norm(u))
        if nu < 1e-6:
            return None
        u = u / nu
        n = np.array([-u[1], u[0]])                 # normale a la ligne annulaire
        e1 = np.asarray(stent_ends[0], dtype=np.float64)
        e2 = np.asarray(stent_ends[1], dtype=np.float64)
        if ms_fl is not None:
            mid = 0.5 * (h1 + h2)
            if float(np.dot(np.asarray(ms_fl, dtype=np.float64) - mid, n)) < 0:
                n = -n                              # n pointe vers le ventricule
            d1 = float(np.dot(e1 - h1, n)); d2 = float(np.dot(e2 - h1, n))
            return max(0.0, d1, d2)                 # base = extremite la plus ventriculaire
        d1 = abs(float(np.dot(e1 - h1, n))); d2 = abs(float(np.dot(e2 - h1, n)))
        return max(d1, d2)

    def _compute_id_mm(self):
        """ID = profondeur d'implantation (mm) = distance du plan annulaire
        (ligne hinge1-hinge2) a la base ventriculaire du stent, mesuree
        perpendiculairement a la ligne annulaire, sur la fluoroscopie.

        Le NCC n'est plus requis : la ligne hinge-hinge EST le plan de reference
        de l'article (Nai Fovino 2021). Les hinges CT sont projetes sur la fluoro
        via la camera DRR + recalage, comme les meshes.
        """
        if not all(k in self._ms_world for k in ('hinge1', 'hinge2')):
            return None
        ends = self._stent_endpoints_fluoro()
        if ends is None:
            return None
        h1 = self._project_world_to_fluoro(self._ms_world['hinge1'])
        h2 = self._project_world_to_fluoro(self._ms_world['hinge2'])
        if h1 is None or h2 is None:
            return None
        ms_fl = self._project_world_to_fluoro(self._ms_world['ms']) if 'ms' in self._ms_world else None
        id_px = self._id_depth_px(ends, h1, h2, ms_fl)
        if id_px is None:
            return None
        return id_px * self._stent_pix_mm()

    def _ms_recompute(self):
        if not hasattr(self, 'lbl_ms_value'):
            return
        ms_mm = None
        # ΔMSID requiert seulement hinge1, hinge2, ms (pas NCC) :
        if all(k in self._ms_world for k in ('hinge1', 'hinge2', 'ms')):
            ms_mm = ms_length_from_hinges(
                self._ms_world['hinge1'],
                self._ms_world['hinge2'],
                self._ms_world['ms'])
        id_mm = self._compute_id_mm()
        self._ms_update_labels(ms_mm, id_mm)
        if hasattr(self, 'overlay_panel'):
            self._push_tavi_to_overlay()

    def _ms_update_labels(self, ms_mm, id_mm):
        self.lbl_ms_value.setText('MS length : --' if ms_mm is None else f'MS length : {ms_mm:.2f} mm')
        self.lbl_id.setText('ID : --' if id_mm is None else f'ID : {id_mm:.2f} mm')
        if ms_mm is None:
            self.lbl_delta.setText('DeltaMSID : --')
            self.lbl_risk.setText('Risque PM : --')
            return
        r = risk_assessment(ms_mm, id_mm)
        d = r.get('delta_msid_mm')
        self.lbl_delta.setText('DeltaMSID : --' if d is None else f'DeltaMSID : {d:+.2f} mm')
        level = r.get('risk_level')
        if level == 'HIGH':
            self.lbl_risk.setText(
                f"<span style='color:{ERR};font-weight:700'>Risque PM : HAUT "
                f"(~{r['pm_dependency_rate']:.0%})</span>")
        elif level == 'LOW':
            self.lbl_risk.setText(
                f"<span style='color:{ACCENT2};font-weight:700'>Risque PM : BAS "
                f"(~{r['pm_dependency_rate']:.1%})</span>")
        else:
            self.lbl_risk.setText('Risque PM : --')

    # ── Drag & Drop / Gestion images ─────────────────────────────────────────

    def _on_files_dropped(self, files):
        csv_files = []
        nifti_files = []
        meta_csv_files = []
        other_files = []
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if f.lower().endswith('.nii.gz') or ext in ('.nii', '.nrrd'):
                nifti_files.append(f)
            elif ext == '.csv':
                # Distinguer les CSV de labels (segmentation) des CSV de métadonnées DICOM
                try:
                    df_peek = pd.read_csv(f, nrows=5)
                    cols_lower = [c.strip().lower() for c in df_peek.columns]
                    # Heuristique : colonnes nommées tag/vr OU contenu 1ère colonne = tags DICOM
                    is_meta = 'tag' in cols_lower or 'vr' in cols_lower
                    if not is_meta:
                        import re
                        first_vals = df_peek.iloc[:, 0].astype(str).tolist()
                        col0_name = str(df_peek.columns[0]).strip()
                        all_vals = [col0_name] + first_vals
                        if any(re.match(r'\[?[0-9a-fA-F]{8}\]?$', v.strip()) for v in all_vals):
                            is_meta = True
                    if is_meta:
                        meta_csv_files.append(f)
                    else:
                        csv_files.append(f)
                except Exception:
                    csv_files.append(f)
            else:
                other_files.append(f)
        csv_path = csv_files[0] if csv_files else self._pending_csv
        # Charger les métadonnées DICOM depuis CSV si présent
        if meta_csv_files:
            try:
                meta = read_metadata_csv(meta_csv_files[0])
                self.dicom_meta = meta
                self._apply_dicom_meta(meta)
                self._status(f'Metadonnees DICOM chargees depuis CSV : {os.path.basename(meta_csv_files[0])}')
            except Exception as ex:
                self._err(f'Lecture CSV metadonnees echouee : {ex}')
        def is_seg(fp):
            n = os.path.basename(fp).lower()
            return any(kw in n for kw in ('seg', 'label', 'mask'))
        nifti_files.sort(key=is_seg)
        for f in nifti_files:
            if is_seg(f):
                self._load_seg_auto(f, csv_path)
                csv_path = None
            else:
                self.load_ct(f)
        if csv_files and csv_path:
            self._pending_csv = csv_path
        for f in other_files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.dcm'):
                self._add_image(f)
            elif ext == '':
                # Fichier sans extension — probablement un DICOM (IM0, IM1…)
                self._add_image(f)
        self._status(f'{len(files)} fichier(s) charge(s)')

    def _on_browse(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 'Selectionner des fichiers', '',
            'Tous (*.nii *.nii.gz *.nrrd *.csv *.png *.jpg *.jpeg *.bmp *.tiff *.tif *.dcm *);;'
            'Segmentations/Volumes (*.nii *.nii.gz *.nrrd);;Images (*.png *.jpg *.jpeg *.tiff *.bmp *.dcm);;'
            'DICOM (*.dcm *);;'
            'CSV labels/meta (*.csv)')
        if files:
            self._on_files_dropped(files)

    def _read_segmentation_volume(self, path):
        """Read NIfTI or NRRD segmentation and return (label_volume, affine, label_names)."""
        p = path.lower()
        if p.endswith('.nii') or p.endswith('.nii.gz'):
            seg_img = nib.load(path)
            sv = seg_img.get_fdata().astype(np.int16)
            return sv, seg_img.affine, None

        if not p.endswith('.nrrd'):
            raise ValueError(f'Format segmentation non supporte: {path}')

        if sitk is None:
            raise RuntimeError('Lecture NRRD indisponible: installez SimpleITK (pip install SimpleITK)')

        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)

        aff = np.eye(4, dtype=np.float64)
        try:
            direction = np.array(img.GetDirection(), dtype=np.float64)
            spacing = np.array(img.GetSpacing(), dtype=np.float64)
            origin = np.array(img.GetOrigin(), dtype=np.float64)
            if direction.size >= 9 and spacing.size >= 3 and origin.size >= 3:
                aff[:3, :3] = direction[:9].reshape(3, 3) @ np.diag(spacing[:3])
                aff[:3, 3] = origin[:3]
        except Exception:
            pass

        keys = set(img.GetMetaDataKeys())
        seg_defs = []
        idx = 0
        while f'Segment{idx}_Name' in keys:
            name = img.GetMetaData(f'Segment{idx}_Name').strip() or f'segment_{idx + 1}'
            try:
                layer = int(float(img.GetMetaData(f'Segment{idx}_Layer'))) if f'Segment{idx}_Layer' in keys else idx
            except Exception:
                layer = idx
            try:
                label_val = int(float(img.GetMetaData(f'Segment{idx}_LabelValue'))) if f'Segment{idx}_LabelValue' in keys else 1
            except Exception:
                label_val = 1
            seg_defs.append((name, layer, label_val))
            idx += 1

        if arr.ndim == 3:
            sv = np.transpose(arr, (2, 1, 0)).astype(np.int16)
            label_names = None
            if seg_defs:
                label_names = {}
                for name, _, label_val in seg_defs:
                    label_names[int(label_val)] = name
            return sv, aff, label_names

        if arr.ndim != 4:
            raise ValueError(f'NRRD segmentation invalide (ndim={arr.ndim}), attendu 3D ou 4D')

        if seg_defs:
            layer_count = max((d[1] for d in seg_defs), default=-1) + 1
        else:
            layer_count = arr.shape[0] if arr.shape[0] <= arr.shape[-1] else arr.shape[-1]

        if arr.shape[0] == layer_count:
            layer_axis = 0
            layer_dim = arr.shape[0]
        elif arr.shape[-1] == layer_count:
            layer_axis = 3
            layer_dim = arr.shape[-1]
        else:
            layer_axis = 0 if arr.shape[0] <= arr.shape[-1] else 3
            layer_dim = arr.shape[0] if layer_axis == 0 else arr.shape[-1]

        def _get_layer(layer_idx):
            layer = arr[layer_idx, ...] if layer_axis == 0 else arr[..., layer_idx]
            if layer.ndim != 3:
                raise ValueError('NRRD 4D invalide: impossible d\'extraire une couche 3D')
            return np.transpose(layer, (2, 1, 0))

        sample = _get_layer(0)
        sv = np.zeros(sample.shape, dtype=np.int16)
        label_names = {}
        out_idx = 1

        if seg_defs:
            for name, layer, label_val in seg_defs:
                if layer < 0 or layer >= layer_dim:
                    continue
                lay = _get_layer(layer)
                m = (lay == label_val)
                if m.sum() == 0:
                    m = lay > 0
                if m.sum() == 0:
                    continue
                sv[m] = out_idx
                label_names[out_idx] = name
                out_idx += 1
        else:
            for layer in range(layer_dim):
                lay = _get_layer(layer)
                m = lay > 0
                if m.sum() == 0:
                    continue
                sv[m] = out_idx
                label_names[out_idx] = f'label_{out_idx}'
                out_idx += 1

        return sv, aff, label_names

    def _build_seg_masks(self, seg_volume, csv_path=None, label_names=None):
        """Build per-structure binary masks from a multilabel segmentation volume."""
        masks = {}

        def _unique_name(base):
            root = str(base).strip() or 'label'
            if root not in masks:
                return root
            i = 2
            while f'{root}_{i}' in masks:
                i += 1
            return f'{root}_{i}'

        if csv_path:
            df = pd.read_csv(csv_path)
            cols = df.columns.tolist()
            for _, row in df.iterrows():
                try:
                    idx = int(row[cols[0]])
                except (ValueError, TypeError):
                    continue
                name = str(row[cols[1]]).strip() if len(cols) > 1 else f'label_{idx}'
                if not name or idx == 0:
                    continue
                m = (seg_volume == idx).astype(np.uint8)
                if m.sum() == 0:
                    continue
                masks[_unique_name(name)] = m
            return masks

        if label_names:
            for idx, name in label_names.items():
                try:
                    idx_i = int(idx)
                except Exception:
                    continue
                if idx_i == 0:
                    continue
                m = (seg_volume == idx_i).astype(np.uint8)
                if m.sum() == 0:
                    continue
                masks[_unique_name(name)] = m
            if masks:
                return masks

        for idx in np.unique(seg_volume):
            if idx == 0:
                continue
            m = (seg_volume == idx).astype(np.uint8)
            if m.sum() == 0:
                continue
            masks[_unique_name(f'label_{int(idx)}')] = m
        return masks

    def _load_seg_auto(self, path, csv_path=None):
        try:
            sv, aff, label_names = self._read_segmentation_volume(path)
            self.seg_affine = aff
            self.seg_masks = self._build_seg_masks(sv, csv_path=csv_path, label_names=label_names)
            n = len(self.seg_masks)
            self.lbl_seg.setText(f'Seg : {os.path.basename(path)} ({n})')
            if self.ct_vol is not None and self.seg_masks:
                self.seg_review_panel.set_data(self.ct_vol, self.seg_masks)
            if hasattr(self, 'btn_tseg_3d'):
                self.btn_tseg_3d.setEnabled(bool(self.seg_masks))
            self._update_checklist()
            self._status(f'Segmentation chargee -- {n} structures')
        except Exception as ex:
            self._err(str(ex))

    def _add_image(self, path):
        name = os.path.basename(path)
        for entry in self._loaded_images:
            if entry['path'] == path:
                self._status(f'Image deja chargee : {name}')
                return
        ext = os.path.splitext(path)[1].lower()
        img = None
        dicom_meta = None
        dicom_frames_u8 = None
        frame_index = 0
        # Tenter DICOM pour .dcm ou fichiers sans extension (ex: IM0)
        is_dicom_ext = ext == '.dcm' or ext == ''
        if is_dicom_ext and pydicom is not None:
            try:
                series = read_dicom_fluoro_series(path)
                dicom_frames_u8 = series['frames_u8']
                dicom_meta = dict(series['meta'])
                frame_index = max(0, int(dicom_meta.get('frame_used', 1)) - 1)
                img = dicom_frames_u8[frame_index]
            except Exception:
                img = None  # fallback to cv2
        if img is None:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self._err(f'Impossible de charger : {name}'); return
        img_float = img.astype(np.float32) / 255.0
        idx = len(self._loaded_images)
        entry = {
            'path': path,
            'name': name,
            'array': img_float,
            'role': None,
            'card': None,
            'dicom_meta': dicom_meta,
            'dicom_frames_u8': dicom_frames_u8,
            'frame_index': frame_index,
        }
        self._loaded_images.append(entry)
        self.lbl_no_images.hide()
        card = ImageCard(idx, name, img_float)
        card.role_changed.connect(self._assign_image)
        card.image_delete_requested.connect(self._remove_image)
        entry['card'] = card
        self._images_vbox.addWidget(card)
        self._sync_image_actions()
        if idx == 0:
            card.set_role_external('fixed')
            self._assign_image(idx, 'fixed')

    def _sync_image_actions(self):
        has_images = bool(self._loaded_images)
        self.lbl_no_images.setVisible(not has_images)
        self.btn_clear_images.setEnabled(has_images)

    def _clear_fixed_image(self, reason=''):
        self._fixed_image_index = None
        self.fluoro_image = None
        self.cv_fl.set_image(None)
        self._clear_stent_state('')
        self.dicom_meta = {}
        self.lbl_fluoro_meta.setText('Fluoro : --')
        self._sync_fluoro_frame_controls(None)
        self._invalidate_registration_state(reason or 'Image fixe retiree.')
        self._update_checklist()

    def _clear_mobile_image(self, reason=''):
        self._mobile_image_index = None
        self.drr_image = None
        self._set_drr_base_image(None)
        self.proj_masks = {}
        self.cv_drr.set_image(None)
        self.tabs.setTabText(1, 'Mobile')
        self._invalidate_registration_state(reason or 'Image mobile retiree.')
        self._update_checklist()

    def _remove_image(self, index):
        if index < 0 or index >= len(self._loaded_images):
            return

        entry = self._loaded_images[index]
        name = entry['name']
        role = entry.get('role')
        fixed_index = self._fixed_image_index
        mobile_index = self._mobile_image_index

        if role == 'fixed':
            self._clear_fixed_image(f'Image fixe retiree : {name}')
        elif role == 'mobile':
            self._clear_mobile_image(f'Image mobile retiree : {name}')

        card = entry.get('card')
        if card:
            self._images_vbox.removeWidget(card)
            card.deleteLater()

        self._loaded_images.pop(index)
        if role != 'fixed' and fixed_index is not None and index < fixed_index:
            self._fixed_image_index = fixed_index - 1
        if role != 'mobile' and mobile_index is not None and index < mobile_index:
            self._mobile_image_index = mobile_index - 1
        for i, img_entry in enumerate(self._loaded_images):
            img_entry['card'].index = i

        self._sync_image_actions()
        self._status(f'Image retiree : {name}')

    def _remove_all_images(self):
        if not self._loaded_images:
            return
        for entry in list(self._loaded_images):
            card = entry.get('card')
            if card:
                self._images_vbox.removeWidget(card)
                card.deleteLater()
        self._loaded_images = []
        self._clear_fixed_image('Toutes les images ont ete retirees.')
        self._clear_mobile_image('')
        self._sync_image_actions()
        self._status('Toutes les images ont ete retirees')

    def _assign_image(self, index, role):
        if index >= len(self._loaded_images):
            return
        if role:
            for i, e in enumerate(self._loaded_images):
                if i != index and e['role'] == role:
                    e['role'] = None
                    if e['card']:
                        e['card'].set_role_external(None)
        self._loaded_images[index]['role'] = role or None
        if role == 'fixed':
            self._set_fixed_image(index)
        elif role == 'mobile':
            self._set_mobile_image(index)

    def _set_fixed_image(self, index, frame_changed=False):
        entry = self._loaded_images[index]
        self._fixed_image_index = index
        self.fluoro_image = entry['array']
        self.cv_fl.set_image(self.fluoro_image)
        if not frame_changed:
            self._clear_stent_state('')
        self.tabs.setCurrentIndex(0)
        self._invalidate_registration_state(
            'Frame fluoroscopique changee.'
            if frame_changed else
            f'Nouvelle image fixe : {entry["name"]}')

        meta = entry.get('dicom_meta')
        if meta:
            self.dicom_meta = dict(meta)
            self._apply_dicom_meta(self.dicom_meta, update_controls=not frame_changed)
            self._sync_fluoro_frame_controls(entry)
            fps = float(meta.get('cine_rate_fps', 0.0) or 0.0)
            fps_txt = f' | {fps:.1f} img/s' if fps > 0 else ''
            self._status(
                f'Fluoro DICOM chargee — frame {entry.get("frame_index", 0) + 1}/'
                f'{meta.get("n_frames", 1)}{fps_txt}')
        else:
            self.dicom_meta = {}
            self.lbl_fluoro_meta.setText(f'Fluoro : {entry["name"]}')
            self._sync_fluoro_frame_controls(None)
            self._status(
                f'Image fixe : {entry["name"]}'
                if not frame_changed else
                f'Image fixe mise a jour : {entry["name"]}')
        self._update_checklist()

    def _set_mobile_image(self, index):
        entry = self._loaded_images[index]
        self._mobile_image_index = index
        self.drr_image = entry['array']
        self._set_drr_base_image(self.drr_image)
        self.proj_masks = {}
        self.cv_drr.set_image(self.drr_image)
        self._invalidate_registration_state(f'Nouvelle image mobile : {entry["name"]}')
        self.tabs.setTabText(1, 'Mobile')
        self.tabs.setCurrentIndex(1)
        self._status(f'Image mobile : {entry["name"]}')
        self._update_checklist()

    def _invalidate_registration_state(self, reason=''):
        self.result = None
        self._yolo_det_fl = None
        self.lbl_iou.setText('--')
        self.lbl_iou.setStyleSheet('')
        self.lbl_dice.setText('--')
        self.lbl_dice.setStyleSheet('')
        self.lbl_tx.setText('tx : --')
        self.lbl_ty.setText('ty : --')
        self.lbl_rot.setText('rot : --')
        self.lbl_scale.setText('scale : --')
        self.result_panel.clear_data()
        self.overlay_panel.clear_data()
        self.btn_reg.setEnabled(False)
        self.lbl_auto_status.setText('')
        if self._iterations:
            self._iterations = []
            self._current_iter_idx = -1
            self._refresh_iter_list()
        self._update_checklist()
        if reason:
            self.lbl_prog.setText(reason)

    def _sync_fluoro_frame_controls(self, entry):
        if not entry:
            self.lbl_fluoro_frame_value.setText('-- / --')
            self._fluoro_frame_box.hide()
            return
        frames = entry.get('dicom_frames_u8')
        if frames is None or frames.shape[0] <= 1:
            self.lbl_fluoro_frame_value.setText('-- / --')
            self._fluoro_frame_box.hide()
            return
        n_frames = int(frames.shape[0])
        frame_idx = int(entry.get('frame_index', 0))
        self.sl_fluoro_frame.blockSignals(True)
        self.sl_fluoro_frame.setRange(1, n_frames)
        self.sl_fluoro_frame.setValue(frame_idx + 1)
        self.sl_fluoro_frame.blockSignals(False)
        self.lbl_fluoro_frame_value.setText(f'{frame_idx + 1} / {n_frames}')
        fps = float(entry.get('dicom_meta', {}).get('cine_rate_fps', 0.0) or 0.0)
        if fps > 0:
            self.lbl_fluoro_frame_hint.setText(
                f'Molette sur la fluoroscopie pour changer de frame ({fps:.1f} img/s)')
        else:
            self.lbl_fluoro_frame_hint.setText(
                'Molette sur la fluoroscopie pour changer de frame')
        self._fluoro_frame_box.show()

    def _on_fluoro_frame_slider(self, value):
        self._set_fluoro_frame((value or 1) - 1)

    def _step_fluoro_frame(self, steps):
        entry = None
        if self._fixed_image_index is not None and self._fixed_image_index < len(self._loaded_images):
            entry = self._loaded_images[self._fixed_image_index]
        if not entry:
            return
        frames = entry.get('dicom_frames_u8')
        if frames is None or frames.shape[0] <= 1:
            return
        cur = int(entry.get('frame_index', 0))
        self._set_fluoro_frame(cur + int(steps))

    def _set_fluoro_frame(self, frame_index):
        if self._fixed_image_index is None or self._fixed_image_index >= len(self._loaded_images):
            return
        entry = self._loaded_images[self._fixed_image_index]
        frames = entry.get('dicom_frames_u8')
        if frames is None or frames.shape[0] <= 1:
            return
        new_idx = max(0, min(int(frame_index), frames.shape[0] - 1))
        if new_idx == int(entry.get('frame_index', 0)):
            self._sync_fluoro_frame_controls(entry)
            return

        entry['frame_index'] = new_idx
        entry['array'] = frames[new_idx].astype(np.float32) / 255.0
        if entry.get('dicom_meta') is not None:
            entry['dicom_meta'] = dict(entry['dicom_meta'])
            entry['dicom_meta']['frame_used'] = new_idx + 1
        if entry.get('card'):
            entry['card'].set_array(entry['array'])

        self._sync_fluoro_frame_controls(entry)
        if entry.get('role') == 'fixed':
            self._set_fixed_image(self._fixed_image_index, frame_changed=True)

    def _apply_dicom_meta(self, meta, update_controls=True):
        """Remplit les spinboxes et labels UI depuis un dict de métadonnées."""
        if update_controls:
            self.sp_lao.setValue(meta['lao'])
            self.sp_cran.setValue(meta['cran'])
            self.sp_table.setValue(meta.get('table_angle', 0.0))
            self.sp_fov.setValue(float(meta.get('fov_mm', DEFAULT_FOV_MM) or DEFAULT_FOV_MM))

        fov = meta['fov_mm']
        fps = float(meta.get('cine_rate_fps', 0.0) or 0.0)
        fps_str = f'\nCadence={fps:.1f} img/s' if fps > 0 else ''
        # Label résumé dans DONNEES
        arm_str = ''
        if meta.get('arm_l') is not None:
            arm_str = f'\nBras : L={meta["arm_l"]:.1f}  P={meta["arm_p"]:.1f}  C={meta["arm_c"]:.1f}'
        lbl = (f'Fluoro DICOM : {meta["cols"]}x{meta["rows"]}px | '
               f'{meta["pixel_mm"]:.3f} mm/px\n'
               f'LAO={meta["lao"]:+.1f}deg  CRAN={meta["cran"]:+.1f}deg  '
               f'Table={meta.get("table_angle", 0):+.1f}deg\n'
               f'SID={meta["sid_mm"]:.0f}mm  SOD={meta["sod_mm"]:.0f}mm  '
               f'Mag={meta.get("magnification", 1):.3f}\n'
               f'FOV isoctr={fov:.1f}mm  '
               f'[frame {meta["frame_used"]}/{meta["n_frames"]}]'
               f'{fps_str}'
               f'{arm_str}')
        self.lbl_fluoro_meta.setText(lbl)

        # Ouvrir automatiquement la section PARAMETRES DRR
        self.sec_drr.set_open(True)

    # ── Actions ───────────────────────────────────────────────────────────────

    def load_ct(self, path=None):
        p = path
        if not p:
            p,_=QFileDialog.getOpenFileName(self,'CT NIfTI','','NIfTI (*.nii *.nii.gz)')
        if not p: return
        try:
            self.ct_vol,self.voxel_mm,self.ct_aff,_,self.ap_axis,codes=load_ct(p)
            self.ct_path = p
            self.ct_codes = codes
            self.lbl_ct.setText(f'CT: {os.path.basename(p)}\n  {self.ct_vol.shape} | {self.voxel_mm.round(2)} mm | AP={self.ap_axis} {codes}')
            self.btn_drr.setEnabled(True)
            if self.seg_masks:
                self.seg_review_panel.set_data(self.ct_vol, self.seg_masks)
            self._update_checklist()
            self._status(f'CT chargé — axe AP={self.ap_axis} ({codes})')
        except Exception as ex: self._err(str(ex))

    def load_seg(self):
        p,_=QFileDialog.getOpenFileName(self,'Segmentation','','Segmentations (*.nii *.nii.gz *.nrrd)')
        if not p: return
        cp,_=QFileDialog.getOpenFileName(self,'Label CSV','','CSV (*.csv)')
        self._load_seg_auto(p, cp or None)

    def load_fluoro(self):
        p,_=QFileDialog.getOpenFileName(self,'Fluoroscopie','','Images & DICOM (*.png *.jpg *.tiff *.bmp *.dcm)')
        if not p: return
        self._add_image(p)
        # Assigner automatiquement comme image fixe si pas encore fait
        for i, e in enumerate(self._loaded_images):
            if e['path'] == p and e['role'] != 'fixed':
                self._assign_image(i, 'fixed')
                if e['card']:
                    e['card'].set_role_external('fixed')
                break

    def _run_totalseg(self):
        in_path = self.ct_path
        if not in_path:
            self._err('Chargez d\'abord un CT NIfTI (meme source que le DRR).')
            return

        self.lbl_tseg_src.setText(f'Source : {os.path.basename(in_path)}')

        task_name = self.cb_tseg_task.currentText().strip() or 'total'
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join('data', 'totalseg_runs', f'{task_name}_{stamp}')
        self.tseg_output_dir = out_dir

        self.btn_tseg_run.setEnabled(False)
        self.btn_tseg_export.setEnabled(False)
        self.lbl_tseg_status.setText('Segmentation en cours...')

        kw = dict(
            input_path=in_path,
            output_dir=out_dir,
            task_name=task_name,
            profile='all',
            fast=self.chk_tseg_fast.isChecked(),
            device=self.cb_tseg_device.currentText().strip().lower(),
            license_key=TOTALSEG_LICENSE_KEY,
        )
        self._start_worker(
            'totalseg',
            kw,
            self._on_totalseg_done,
            self._on_totalseg_err,
            busy_title='Segmentation automatique',
            busy_message='Analyse du volume et preparation des structures...',
        )

    def _on_totalseg_done(self, res):
        self._hide_busy_overlay()
        self.btn_tseg_run.setEnabled(True)

        if res.get('task') != 'totalseg':
            return

        self.seg_masks = res.get('masks', {})
        self.seg_affine = res.get('affine')
        self.tseg_output_dir = res.get('output_dir')

        # Si la source est un NIfTI CT et qu'aucun CT n'est charge, on l'importe.
        src = res.get('input_path')
        if self.ct_vol is None and isinstance(src, str) and src.lower().endswith(('.nii', '.nii.gz')):
            try:
                self.load_ct(src)
            except Exception:
                pass

        n = len(self.seg_masks)
        names = ', '.join(list(self.seg_masks.keys())[:6])
        suffix = '...' if n > 6 else ''
        self.lbl_seg.setText(f'Seg : TotalSegmentator ({n})')
        self.lbl_tseg_status.setText(
            f'Tache : {res.get("task_name", "total")}\n'
            f'Sortie : {self.tseg_output_dir}\n'
            f'Structures chargees : {names}{suffix}')

        # Sortie par defaut: un seul fichier multilabel qui contient toutes les segmentations
        self.tseg_multilabel_path = None
        try:
            default_out = os.path.join(self.tseg_output_dir, 'segmentation_multilabel.nii.gz')
            aff = self.ct_aff if self.ct_aff is not None else self.seg_affine
            exp = export_multilabel_segmentation(self.seg_masks, default_out, affine=aff)
            self.tseg_multilabel_path = exp['out_path']
            self.lbl_tseg_status.setText(
                self.lbl_tseg_status.text() +
                f'\nFichier multilabel : {self.tseg_multilabel_path}')

            # Utiliser explicitement ce fichier unique comme segmentation active pour la suite.
            self._load_seg_auto(exp['out_path'], exp.get('labels_csv'))
            self.lbl_tseg_status.setText(
                self.lbl_tseg_status.text() +
                '\nSegmentation active: fichier multilabel unique charge.')
        except Exception as ex:
            self.lbl_tseg_status.setText(
                self.lbl_tseg_status.text() +
                f'\nAttention: export multilabel auto echoue ({ex}).')

        if self.ct_vol is not None and self.seg_masks:
            self.seg_review_panel.set_data(self.ct_vol, self.seg_masks)
            self.tabs.setCurrentIndex(2)
        elif self.seg_masks:
            self.lbl_tseg_status.setText(
                self.lbl_tseg_status.text() +
                '\nCT non charge : la vue Seg CT requiert un CT NIfTI charge.')

        self.btn_tseg_export.setEnabled(bool(self.seg_masks))
        self.btn_tseg_3d.setEnabled(bool(self.seg_masks))
        self._update_checklist()
        self._status(f'Segmentation automatique terminee ({n} structure(s)).')

    def _on_totalseg_err(self, msg):
        self._hide_busy_overlay()
        self.btn_tseg_run.setEnabled(True)
        self.btn_tseg_export.setEnabled(bool(self.seg_masks))
        self.btn_tseg_3d.setEnabled(bool(self.seg_masks))
        self.lbl_tseg_status.setText(f'Erreur : {msg.splitlines()[0]}')
        self._err(msg)

    def _export_totalseg_masks(self):
        if not self.seg_masks:
            self._err('Aucune segmentation a exporter. Lancez une segmentation auto ou chargez une seg.')
            return

        p, filt = QFileDialog.getSaveFileName(
            self,
            'Exporter segmentation multilabel',
            'segmentation_multilabel.nii.gz',
            'NIfTI (*.nii.gz *.nii);;NRRD (*.nrrd)')
        if not p:
            return

        ext = os.path.splitext(p)[1].lower()
        if not ext:
            if 'NRRD' in filt:
                p += '.nrrd'
            else:
                p += '.nii.gz'

        aff = self.ct_aff if self.ct_aff is not None else self.seg_affine
        try:
            exp = export_multilabel_segmentation(self.seg_masks, p, affine=aff)
        except Exception as ex:
            self._err(str(ex))
            return

        self._status(f'Export segmentation -> {exp["out_path"]}')
        QMessageBox.information(
            self,
            'Export segmentation',
            f'Segmentation exportee:\n{exp["out_path"]}\n\nLabels:\n{exp["labels_csv"]}')

    def _open_segmentation_3d(self):
        if not self.seg_masks:
            self._err('Aucune segmentation disponible pour la vue 3D.')
            return

        try:
            import pyvista as pv
            from skimage import measure
        except Exception as ex:
            self._err(f'Vue 3D indisponible: {ex}')
            return

        plotter = pv.Plotter(window_size=(1280, 860))
        plotter.set_background('#12151f')

        spacing = (1.0, 1.0, 1.0)
        if self.ct_aff is not None:
            try:
                spacing = (
                    float(abs(self.ct_aff[0, 0])),
                    float(abs(self.ct_aff[1, 1])),
                    float(abs(self.ct_aff[2, 2])),
                )
            except Exception:
                spacing = (1.0, 1.0, 1.0)
        elif self.voxel_mm is not None:
            try:
                spacing = tuple(float(v) for v in self.voxel_mm)
            except Exception:
                spacing = (1.0, 1.0, 1.0)

        added = 0
        for idx, (name, vol) in enumerate(self.seg_masks.items()):
            if vol is None or getattr(vol, 'ndim', 0) != 3 or np.sum(vol) == 0:
                continue

            arr = (vol > 0).astype(np.uint8)
            stride = 1
            if arr.size > 180_000_000:
                stride = 2
                arr = arr[::2, ::2, ::2]

            try:
                verts, faces, _, _ = measure.marching_cubes(arr, level=0.5)
            except Exception:
                continue
            if verts.size == 0 or faces.size == 0:
                continue

            sx, sy, sz = spacing
            verts = verts * np.array([sx * stride, sy * stride, sz * stride], dtype=np.float32)
            faces_vtk = np.hstack([
                np.full((faces.shape[0], 1), 3, dtype=np.int64),
                faces.astype(np.int64),
            ]).ravel()
            mesh = pv.PolyData(verts.astype(np.float32), faces_vtk)
            mesh = mesh.clean(tolerance=0.0)

            rgb = color_for_structure(name)
            color = tuple(c / 255.0 for c in rgb)
            plotter.add_mesh(mesh, color=color, opacity=0.45, name=f'{name}_{idx}')
            added += 1

        if added == 0:
            plotter.close()
            self._err('Aucun maillage 3D n\'a pu etre reconstruit.')
            return

        # ── Conversion voxel -> coords 3D du plotter (les meshes seg sont dejà
        #    en coords voxel * spacing, donc les markers TAVI doivent suivre).
        sx, sy, sz = spacing

        def vox_to_plot(voxel):
            return np.array([voxel[0] * sx, voxel[1] * sy, voxel[2] * sz], dtype=np.float64)

        def world_to_plot(world_mm):
            if self.ct_aff is None:
                return np.asarray(world_mm, dtype=np.float64)
            v = voxel_from_world(world_mm, self.ct_aff)
            return vox_to_plot(v)

        # NB : pas de rendu du stent ici. Le stent (et le calcul du risque TAVI
        # complet) n'apparaissent qu'apres recalage, dans l'onglet Overlay 3D.

        # ── Rendu/mise a jour des marqueurs + segments + labels (in-place) ──
        def refresh_tavi():
            """Redessine markers, ligne annulaire, segments MS/ID et labels dans le plotter ouvert."""
            present = set()
            marker_pts = {}
            for name in ('hinge1', 'hinge2', 'ms'):
                world = self._ms_world.get(name)
                if world is None:
                    continue
                pt = world_to_plot(world)
                marker_pts[name] = pt
                color = tuple(c / 255.0 for c in self._MS_MARKER[name][0])
                label = self._MS_MARKER[name][1]
                plotter.add_mesh(pv.Sphere(radius=2.6, center=pt), color=color,
                                  name=f'{name}_marker', render=False)
                plotter.add_point_labels(np.array([pt]), [label], font_size=12,
                                          point_color=color, text_color='white',
                                          shape_opacity=0.0, always_visible=True,
                                          name=f'{name}_label', render=False)
                present.add(f'{name}_marker'); present.add(f'{name}_label')

            h1 = marker_pts.get('hinge1'); h2 = marker_pts.get('hinge2'); ms_pt = marker_pts.get('ms')
            if h1 is not None and h2 is not None:
                plotter.add_mesh(pv.Line(h1, h2), color='cyan', line_width=3,
                                  name='annulus_line', render=False)
                present.add('annulus_line')
                if ms_pt is not None:
                    axis = h2 - h1
                    n = float(np.linalg.norm(axis))
                    if n > 1e-6:
                        u = axis / n
                        proj = h1 + np.dot(ms_pt - h1, u) * u
                        plotter.add_mesh(pv.Line(proj, ms_pt), color='red', line_width=4,
                                          name='ms_segment', render=False)
                        ms_mm = float(np.linalg.norm(ms_pt - proj))
                        mid = 0.5 * (proj + ms_pt)
                        plotter.add_point_labels(
                            np.array([mid]), [f'MS = {ms_mm:.2f} mm'],
                            font_size=14, point_color='red', text_color='white',
                            shape='rounded_rect', shape_color='black', shape_opacity=0.55,
                            always_visible=True, name='ms_label', render=False)
                        present.add('ms_segment'); present.add('ms_label')

            # ID n'est pas affiche ici : c'est une mesure 2D sur la fluoroscopie.
            # Voir l'onglet Overlay -> Vue 3D pour la visualisation du stent + ID.

            # Retirer les acteurs orphelins (point supprime via reset)
            for nm in ('hinge1', 'hinge2', 'ms'):
                for suf in ('_marker', '_label'):
                    actor = f'{nm}{suf}'
                    if actor not in present:
                        try: plotter.remove_actor(actor, render=False)
                        except Exception: pass
            for actor in ('annulus_line', 'ms_segment', 'ms_label'):
                if actor not in present:
                    try: plotter.remove_actor(actor, render=False)
                    except Exception: pass
            plotter.render()
            return len(present), marker_pts

        ms_added, _ = refresh_tavi()

        # ── Picker 3D : alimente _ms_world + rafraichit la vue immediatement ──
        def _on_pick(point, picker=None):
            if not self._ms_click_target:
                return
            try:
                voxel = (float(point[0]) / sx, float(point[1]) / sy, float(point[2]) / sz)
                world = world_from_voxel(voxel, self.ct_aff) if self.ct_aff is not None else np.asarray(point)
                key = self._ms_click_target
                self._set_ms_point(key, world)
                next_key = next((k for k in self._MS_ORDER if k not in self._ms_world), None)
                self._ms_arm(next_key)
                refresh_tavi()
                msg = f'{key} place en 3D ({point[0]:.0f}, {point[1]:.0f}, {point[2]:.0f}).'
                if next_key:
                    msg += f' Prochain : {next_key}.'
                self._status(msg)
            except Exception as ex:
                self._status(f'Pick 3D echoue : {ex}')

        try:
            plotter.enable_point_picking(callback=_on_pick, show_message=False,
                                          show_point=True, point_size=12,
                                          left_clicking=False, color='yellow',
                                          tolerance=0.02)
        except Exception:
            pass

        plotter.add_axes(line_width=2)
        title = f'Segmentations 3D ({added})'
        if ms_added:
            title += f' + reperes TAVI ({ms_added} acteurs)'
        title += '  --  P + clic = placer le repere arme dans le sidebar.'
        plotter.add_text(title, font_size=10, name='title')
        plotter.show()

    def _projection_seg_masks(self):
        """Return segmentation masks for DRR projection while preserving class names."""
        if not self.seg_masks:
            return {}

        out = {}
        for name, m in self.seg_masks.items():
            if m is None:
                continue
            if m.sum() == 0:
                continue
            key = str(name).strip() or str(name)
            out[key] = (m > 0).astype(np.uint8)
        return out

    def generate_drr(self):
        if self.ct_path is None: self._err('Charger un CT d\'abord'); return
        self.btn_drr.setEnabled(False)
        kw = dict(
            ct_path=self.ct_path,
            ct_aff=self.ct_aff,
            lao_deg=self.sp_lao.value(),
            cran_deg=self.sp_cran.value() + 180,
            table_angle=self.sp_table.value(),
            output_size=self.sp_size.value(),
            masks=self._projection_seg_masks(),
            fov_mm=self.sp_fov.value(),
            sid_mm=self.dicom_meta.get('sid_mm', 1020.0),
            sod_mm=self.dicom_meta.get('sod_mm', 510.0),
            renderer='siddon',
        )
        self._start_worker(
            'drr',
            kw,
            self._drr_done,
            self._on_err,
            busy_title='Generation du DRR',
            busy_message='Projection du volume CT et preparation du rendu...',
        )

    def _drr_done(self,res):
        self._hide_busy_overlay()
        self.drr_image=res['drr']; self._set_drr_base_image(self.drr_image); self.proj_masks=res.get('masks',{})
        self.cv_drr.set_image(self.drr_image); self.btn_drr.setEnabled(True)
        self.tabs.setTabText(1,'DRR')
        self._update_checklist()
        # Auto-injecter les masques de segmentation projetes sur le canvas DRR
        if self.chk_use_seg.isChecked():
            if 'vertebrae' in self.proj_masks and self.proj_masks['vertebrae'] is not None:
                self.cv_drr.set_mask('vertebrae', self.proj_masks['vertebrae'])
            elif self.proj_masks:
                # Fusionner toutes les structures disponibles en un seul masque vertebres
                fused = None
                for m in self.proj_masks.values():
                    if m is None or m.sum() == 0: continue
                    fused = m.copy() if fused is None else np.clip(fused + m, 0, 1)
                if fused is not None:
                    self.cv_drr.set_mask('vertebrae', fused)
        self._save_iteration()
        self.tabs.setCurrentIndex(1); self._status('DRR genere — contours vertebres injectes auto, annoter la fluoroscopie')

    def run_registration(self):
        # Récupérer TOUS les masques annotés (vertebrae, heart, aorta)
        masks_all_fl = {}
        masks_all_drr = {}
        for struct in STRUCT.keys():
            m_fl = self.cv_fl.get_mask(struct)
            m_drr = self.cv_drr.get_mask(struct)
            if m_fl is not None and m_fl.sum() > 0:
                masks_all_fl[struct] = m_fl
            if m_drr is not None and m_drr.sum() > 0:
                masks_all_drr[struct] = m_drr
        
        if not masks_all_fl:
            self._err('Annoter au moins une structure (vertèbres, cœur, aorte) sur la fluoroscopie'); return
        if not masks_all_drr:
            self._err('Annoter au moins une structure sur le DRR'); return
        
        # Fusionner tous les masques annotés en un seul masque
        def combine_masks(masks_dict):
            combined = None
            for m in masks_dict.values():
                if combined is None:
                    combined = m.copy()
                else:
                    combined = np.clip(combined + m, 0, 1)
            return combined
        
        mf = combine_masks(masks_all_fl)
        md = combine_masks(masks_all_drr)
        
        # Normaliser les deux masques à la même taille de travail pour le recalage
        reg_size = self.sp_size.value()
        if mf.shape[:2] != (reg_size, reg_size):
            mf = cv2.resize(mf, (reg_size, reg_size), interpolation=cv2.INTER_NEAREST)
        if md.shape[:2] != (reg_size, reg_size):
            md = cv2.resize(md, (reg_size, reg_size), interpolation=cv2.INTER_NEAREST)
        
        # Afficher les structures utilisées dans le recalage
        structs_used = list(set(list(masks_all_fl.keys()) + list(masks_all_drr.keys())))
        struct_labels = [STRUCT[s]['label'] for s in structs_used]
        self._status(f'Recalage sur : {", ".join(struct_labels)}')
        
        self.btn_reg.setEnabled(False)
        kw=dict(moving=md, fixed=mf, elastic=self.chk_elastic.isChecked())
        self._start_worker(
            'register',
            kw,
            self._reg_done,
            self._on_err,
            busy_title='Recalage 2D/3D',
            busy_message='Alignement des formes et optimisation en cours...',
        )

    def _reg_done(self,res):
        self._hide_busy_overlay()
        self.result=res; self.btn_reg.setEnabled(True)
        iou=res['iou']; dice=res['dice']
        self._update_checklist()
        col=ACCENT2 if iou>0.5 else (WARN if iou>0.25 else ERR)
        self.lbl_iou.setText(f'{iou:.3f}')
        self.lbl_iou.setStyleSheet(f'font-size:22px;font-weight:700;color:{col};')
        self.lbl_dice.setText(f'{dice:.3f}')
        self.lbl_dice.setStyleSheet(f'font-size:22px;font-weight:700;color:{col};')
        self.lbl_tx.setText(f'tx = {res["tx"]:+.1f} px')
        self.lbl_ty.setText(f'ty = {res["ty"]:+.1f} px')
        self.lbl_rot.setText(f'rot = {res["angle"]:+.2f} deg')
        self.lbl_scale.setText(f'scale = {res["scale"]:.3f}')
        self._build_result(res); self.tabs.setCurrentIndex(3)
        if 0 <= self._current_iter_idx < len(self._iterations):
            self._iterations[self._current_iter_idx]['result'] = res
            self._refresh_iter_list()
        self._update_overlay()
        self._ms_recompute()
        self._status(f'Recalage termine -- IoU={iou:.3f}  Dice={dice:.3f}')

    def _build_result(self,res):
        s=self.sp_size.value()

        # ── Images source ─────────────────────────────────────────────────────
        fig_fl  = self.fluoro_image if self.fluoro_image is not None else np.zeros((s,s),np.float32)
        fig_drr = self.drr_image    if self.drr_image    is not None else np.zeros((s,s),np.float32)
        # Resize au format de recalage pour avoir des dimensions homogènes
        if fig_fl.shape[:2] != (s, s):
            fig_fl = cv2.resize(fig_fl.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)
        if fig_drr.shape[:2] != (s, s):
            fig_drr = cv2.resize(fig_drr.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)
        fig_drr_reg = apply_full_transform(fig_drr.astype(np.float32), res)

        # ── Contours à superposer ─────────────────────────────────────────────
        contours = []
        ct_cols = {'vertebrae': (80, 220, 130)}

        # Contours annotation fluoro (couleurs struct)
        for struct,info in STRUCT.items():
            m=self.cv_fl.get_mask(struct)
            if m is None or m.sum()==0: continue
            if m.shape[:2] != (s, s):
                m = cv2.resize(m.astype(np.uint8), (s, s), interpolation=cv2.INTER_NEAREST)
            cnts,_=cv2.findContours((m*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for cnt in cnts:
                pts=cnt.squeeze()
                if pts.ndim>=2: contours.append((pts, info['rgb']))

        # Contours annotation DRR recales (gris clair)
        for struct in STRUCT:
            m=self.cv_drr.get_mask(struct)
            if m is None or m.sum()==0: continue
            if m.shape[:2] != (s, s):
                m = cv2.resize(m.astype(np.float32), (s, s), interpolation=cv2.INTER_NEAREST)
            mr=(apply_full_transform(m.astype(np.float32), res) > 0.5).astype(np.uint8)
            cnts,_=cv2.findContours((mr*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for cnt in cnts:
                pts=cnt.squeeze()
                if pts.ndim>=2: contours.append((pts, (210,210,210)))

        # Contours projections CT recalees
        for name,proj in self.proj_masks.items():
            if proj.shape[:2] != (s, s):
                proj = cv2.resize(proj.astype(np.float32), (s, s), interpolation=cv2.INTER_NEAREST)
            pr=(apply_full_transform(proj.astype(np.float32), res) > 0.5).astype(np.uint8)
            col=ct_cols.get(name,(200,200,200))
            cnts,_=cv2.findContours((pr*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                pts=cnt.squeeze()
                if pts.ndim>=2: contours.append((pts, col))

        self.result_panel.set_data(fig_fl, fig_drr_reg, contours)

    def export_results(self):
        if self.result is None: self._err('Lancez d\'abord le recalage'); return
        folder=QFileDialog.getExistingDirectory(self,'Dossier export')
        if not folder: return
        for struct in STRUCT:
            for prefix,cv in [('fluoro',self.cv_fl),('drr',self.cv_drr)]:
                m=cv.get_mask(struct)
                if m is not None and m.sum()>0:
                    cv2.imwrite(os.path.join(folder,f'mask_{prefix}_{struct}.png'),(m*255).astype(np.uint8))
        if self.drr_image is not None:
            cv2.imwrite(os.path.join(folder,'drr.png'),(self.drr_image*255).astype(np.uint8))
        with open(os.path.join(folder,'result.json'),'w') as f:
            export = {'tx_px':float(self.result['tx']),'ty_px':float(self.result['ty']),
                       'angle_deg':float(self.result['angle']),
                       'scale':float(self.result.get('scale',1.0)),
                       'iou':float(self.result['iou']),
                       'dice':float(self.result['dice']),
                       'lao_deg':self.sp_lao.value(),
                       'cran_deg':self.sp_cran.value(),
                       'table_angle':self.sp_table.value(),
                       'ap_axis':int(self.ap_axis)}
            if self.dicom_meta:
                # Ajouter les infos DICOM sérialisables
                for k in ('sid_mm','sod_mm','magnification','pixel_mm',
                          'fov_mm','intensifier_mm','patient_pos',
                          'manufacturer','model'):
                    if k in self.dicom_meta:
                        export[f'dicom_{k}'] = self.dicom_meta[k]
            json.dump(export, f, indent=2)
        self._status(f'Exporté → {folder}')
        QMessageBox.information(self,'Export',f'Sauvegardé dans :\n{folder}')

    def _update_overlay(self):
        """Alimente l'onglet Overlay avec les données courantes."""
        if self.fluoro_image is None or self.result is None:
            return
        if not self.proj_masks:
            return
        if self.result.get('_auto'):
            fov_for_projection = AUTO_PIPELINE_FOV_MM
        else:
            fov_for_projection = self.sp_fov.value()
        self.overlay_panel.set_data(
            fluoro=self.fluoro_image,
            proj_masks=self.proj_masks,
            result=self.result,
            reg_size=self.sp_size.value(),
            seg_volumes=self._projection_seg_masks(),
            ct_affine=self.ct_aff,
            lao_deg=self.sp_lao.value(),
            cran_deg=self.sp_cran.value() + 180.0,
            table_angle=self.sp_table.value(),
            fov_mm=fov_for_projection,
        )
        self._push_tavi_to_overlay()
        # Basculer automatiquement sur l'onglet Overlay
        self.tabs.setCurrentIndex(4)

    def _push_tavi_to_overlay(self):
        """Communique au panneau overlay la pose 2D du stent + reperes projetes + valeurs."""
        if not hasattr(self, 'overlay_panel'):
            return
        stent_fluoro = None
        if self._stent_center_px is not None and self.stent_mesh is not None and hasattr(self, 'cv_fl'):
            # pix_mm calibre sur la longueur du stent affiche sur le canvas
            # (source de verite metrique 2D, cf. _stent_pix_mm).
            pix_mm = self._stent_pix_mm()
            stent_fluoro = {
                'center_px': (float(self._stent_center_px[0]), float(self._stent_center_px[1])),
                'axis_deg': float(self._stent_axis_deg),
                'length_mm': float(self.sp_stent_L.value()),
                'diameter_mm': float(self.sp_stent_D.value()),
                'pix_mm': pix_mm,
            }
        ms_mm = None
        if all(k in self._ms_world for k in ('hinge1', 'hinge2', 'ms')):
            ms_mm = ms_length_from_hinges(
                self._ms_world['hinge1'], self._ms_world['hinge2'], self._ms_world['ms'])
        id_mm = self._compute_id_mm()
        # Convertir chaque repere CT-monde en coords voxel CT. La projection
        # vers le plan d'overlay sera faite par le panneau via le MEME pipeline
        # que les meshes de segmentation, ce qui garantit l'alignement.
        ref_voxel = {}
        if self.ct_aff is not None:
            for k, world in self._ms_world.items():
                v = voxel_from_world(world, self.ct_aff)
                ref_voxel[k] = (float(v[0]), float(v[1]), float(v[2]))
        ct_shape = self.ct_vol.shape if self.ct_vol is not None else None
        self.overlay_panel.set_tavi_overlay(
            stent_fluoro=stent_fluoro,
            ms_world=dict(self._ms_world),
            ref_voxel=ref_voxel,
            ct_shape=ct_shape,
            id_mm=id_mm,
            ms_length_mm=ms_mm,
        )

    # ── Pipeline automatique complet ──────────────────────────────────────────

    def _run_auto_pipeline(self):
        if self.ct_path is None:
            self._err('Chargez un CT (NIfTI) d\'abord.'); return
        if self.fluoro_image is None:
            self._err('Chargez une fluoroscopie d\'abord.'); return
        if not yolo_ready():
            self._err('Chargez un modèle YOLO (.pt) d\'abord.'); return

        self.btn_auto.setEnabled(False)
        self.btn_drr.setEnabled(False)
        self.btn_reg.setEnabled(False)
        self.lbl_auto_status.setText('Pipeline en cours…')
        self.prog_bar.setValue(0)

        kw = dict(
            ct_path=self.ct_path,
            ct_aff=self.ct_aff,
            fluoro=self.fluoro_image,
            seg_masks=self._projection_seg_masks(),
            output_size=self.sp_size.value(),
            lao_deg=self.sp_lao.value(),
            cran_deg=self.sp_cran.value() + 180,  # convention UI 0° = PA (180°)
            table_angle=self.sp_table.value(),
            sid_mm=self.dicom_meta.get('sid_mm', 1020.0),
            sod_mm=self.dicom_meta.get('sod_mm', 510.0),
            fov_mm=AUTO_PIPELINE_FOV_MM,   # pipeline complet verrouillé à 220 mm
            renderer='siddon',
            elastic=True,
            yolo_kw=dict(
                conf=self.sp_yolo_conf.value() / 100.0,
                iou=self.sp_yolo_iou.value() / 100.0,
                imgsz=self.sp_yolo_imgsz.value(),
                pp={'gamma': self.sp_yolo_gamma.value(),
                    'contrast': self.sp_yolo_contrast.value(),
                    'invert': self.chk_yolo_invert.isChecked()}),
        )
        self._start_worker(
            'auto_pipeline',
            kw,
            self._auto_done,
            self._on_auto_err,
            busy_title='Pipeline automatique',
            busy_message='Generation du DRR, detection et appariement des vertebres...',
        )

    def _detect_vertebrae_manual(self):
        """Lance la détection YOLO sur fluoro + DRR existants, puis ouvre la fenêtre de sélection."""
        # Vérifications préalables
        if self.drr_image is None:
            self._err('Générez d\'abord un DRR (onglet DRR).'); return
        if self.fluoro_image is None:
            self._err('Chargez une fluoroscopie d\'abord.'); return
        if not yolo_ready():
            self._err('Chargez un modèle YOLO (.pt) d\'abord.'); return

        self.btn_detect_vertebrae.setEnabled(False)
        self.btn_auto.setEnabled(False)
        self.lbl_auto_status.setText('Détection YOLO en cours…')

        # Paramètres YOLO
        yolo_kw = dict(
            conf=self.sp_yolo_conf.value() / 100.0,
            iou=self.sp_yolo_iou.value() / 100.0,
            imgsz=self.sp_yolo_imgsz.value(),
            pp={'gamma': self.sp_yolo_gamma.value(),
                'contrast': self.sp_yolo_contrast.value(),
                'invert': self.chk_yolo_invert.isChecked()},
        )

        # Préparer les images
        fluoro_u8 = (np.clip(self.fluoro_image, 0, 1) * 255).astype(np.uint8)
        drr_u8 = (np.clip(self.drr_image, 0, 1) * 255).astype(np.uint8)

        kw = dict(
            fluoro_img=fluoro_u8,
            drr_img=drr_u8,
            yolo_kw=yolo_kw,
        )
        self._start_worker(
            'detect_vertebrae_dual',
            kw,
            self._on_detect_vertebrae_done,
            self._on_auto_err,
            busy_title='Détection vertèbres',
            busy_message='Analyse YOLO de la fluoroscopie et du DRR…',
        )

    def _on_detect_vertebrae_done(self, res):
        """Callback après détection YOLO dual."""
        self._hide_busy_overlay()
        self.btn_detect_vertebrae.setEnabled(True)
        self.btn_auto.setEnabled(True)

        det_fl = res.get('det_fl')
        det_drr = res.get('det_drr')
        if det_fl is None or det_drr is None:
            self._err('Erreur lors de la détection YOLO.')
            self.lbl_auto_status.setText('Erreur de détection.')
            return

        boxes_fl = det_fl.get('boxes', [])
        boxes_drr = det_drr.get('boxes', [])
        if not boxes_fl or not boxes_drr:
            self._err(f'Aucune vertèbre détectée. Fluoro: {len(boxes_fl)}, DRR: {len(boxes_drr)}')
            self.lbl_auto_status.setText('Aucune détection.')
            return

        # Ouvrir la fenêtre de sélection
        dlg = VertebralDetectionWindow(
            det_fl=det_fl,
            det_drr=det_drr,
            boxes_fl=boxes_fl,
            boxes_drr=boxes_drr,
            parent=self
        )
        if dlg.exec_() != QDialog.Accepted:
            self.lbl_auto_status.setText('Détection annulée.')
            return

        selected_fl, selected_drr = dlg.get_selected_detections()
        n_fl = len(selected_fl)
        n_drr = len(selected_drr)
        self.lbl_auto_status.setText(f'✓ Détections validées : Fluoro {n_fl} | DRR {n_drr}')
        self._status(f'Vertèbres détectées et validées : {n_fl} fluoro, {n_drr} DRR')

    def _auto_done(self, res):
        """Callback quand le pipeline auto émet un résultat (phase 1 ou final)."""
        self._hide_busy_overlay()

        # ── Phase 1 : sélection des vertèbres fluoro + DRR ───────────────────
        if res.get('_phase') == 'select_vertebrae':
            self._auto_intermediate = res
            self.lbl_auto_status.setText('Sélectionnez les vertèbres (fluoro + DRR)…')

            # Injecter le DRR dans l'UI dès maintenant
            self.drr_image = res['drr_image']
            self._set_drr_base_image(self.drr_image)
            self.proj_masks = res.get('all_proj_masks', {})
            self.cv_drr.set_image(self.drr_image)
            self.tabs.setTabText(1, 'DRR')

            # Ouvrir le panneau dual de sélection
            dlg = DualYoloSelectionDialog(
                det_fl=res['det_fl'],
                det_drr=res['det_drr'],
                boxes_fl=res['boxes_fl'],
                named_drr_boxes=res['drr_boxes'],
                parent=self)
            if dlg.exec_() != QDialog.Accepted:
                self.btn_auto.setEnabled(True)
                self.btn_drr.setEnabled(True)
                self.lbl_auto_status.setText('Annulé par l\'utilisateur.')
                return

            selected_fl, selected_drr = dlg.get_selected_detections()
            if not selected_fl or not selected_drr:
                self._on_auto_err('Sélectionnez au moins une vertèbre de chaque côté.')
                return

            # Lancer la phase 2 (recalage élastique) en worker
            kw2 = dict(
                boxes_fl=selected_fl,
                boxes_drr=selected_drr,
                reg_size=res['reg_size'],
                drr_image=res['drr_image'],
                all_proj_masks=res['all_proj_masks'],
            )
            self._start_worker(
                'auto_phase2',
                kw2,
                self._auto_done,
                self._on_auto_err,
                busy_title='Recalage en cours',
                busy_message='Application des formes selectionnees et optimisation elastique...',
            )
            return

        # ── Phase finale : résultat complet ───────────────────────────────────
        self.btn_auto.setEnabled(True)
        self.btn_drr.setEnabled(True)

        # Injecter le DRR généré dans l'UI
        self.drr_image = res['drr_image']
        self._set_drr_base_image(self.drr_image)
        self.proj_masks = res.get('proj_masks', {})
        self.cv_drr.set_image(self.drr_image)
        self.tabs.setTabText(1, 'DRR')

        # Injecter les masques appariés dans les canvas
        mask_fl = res.get('mask_fl')
        mask_drr = res.get('mask_drr')
        if mask_fl is not None:
            self.cv_fl.set_mask('vertebrae', mask_fl)
        if mask_drr is not None:
            self.cv_drr.set_mask('vertebrae', mask_drr)
        self.btn_reg.setEnabled(True)

        # Stocker le résultat de recalage
        self.result = res
        self._save_iteration()
        self._iterations[self._current_iter_idx]['result'] = res
        self._refresh_iter_list()
        self._update_checklist()

        # Mettre à jour les métriques
        iou = res['iou']; dice = res['dice']
        col = ACCENT2 if iou > 0.5 else (WARN if iou > 0.25 else ERR)
        self.lbl_iou.setText(f'{iou:.3f}')
        self.lbl_iou.setStyleSheet(f'font-size:22px;font-weight:700;color:{col};')
        self.lbl_dice.setText(f'{dice:.3f}')
        self.lbl_dice.setStyleSheet(f'font-size:22px;font-weight:700;color:{col};')
        self.lbl_tx.setText(f'tx = {res["tx"]:+.1f} px')
        self.lbl_ty.setText(f'ty = {res["ty"]:+.1f} px')
        self.lbl_rot.setText(f'rot = {res["angle"]:+.2f} deg')
        self.lbl_scale.setText(f'scale = {res["scale"]:.3f}')

        # Construire le panneau résultat visuel
        self._build_result(res)
        self.tabs.setCurrentIndex(3)

        # Mettre à jour l'onglet overlay final
        self._update_overlay()

        # Label résumé
        n_fl = res.get('n_fluoro_sel', '?')
        n_drr = res.get('n_drr_sel', '?')
        self.lbl_auto_status.setText(
            f'IoU={iou:.4f}  Dice={dice:.4f}\n'
            f'Fluoro : {n_fl} vertèbre(s) | DRR : {n_drr} vertèbre(s)')
        self._status(
            f'Pipeline auto terminé — IoU={iou:.4f}  Dice={dice:.4f}')

    def _on_auto_err(self, msg):
        self._hide_busy_overlay()
        self.btn_auto.setEnabled(True)
        self.btn_drr.setEnabled(True)
        self.btn_reg.setEnabled(True)
        self.lbl_auto_status.setText(f'Erreur : {msg.splitlines()[0]}')
        self._err(msg)

    def _on_prog(self,pct,msg):
        self.prog_bar.setValue(pct); self.lbl_prog.setText(msg); self._status(msg)
        self._busy_overlay.update_progress(pct, msg)
    def _on_err(self,msg):
        self._hide_busy_overlay()
        self.btn_drr.setEnabled(True); self.btn_reg.setEnabled(True); self.btn_detect_fl.setEnabled(True); self.btn_detect_drr.setEnabled(True); self._err(msg)
    def _status(self,msg): self.statusBar().showMessage(msg)
    def _err(self,msg): self.statusBar().showMessage(msg); QMessageBox.warning(self,'Erreur',msg)

    def _update_checklist(self):
        """Met à jour les pastilles de la checklist."""
        checks = {
            'ct':    self.ct_vol is not None,
            'seg':   bool(self.seg_masks),
            'fluoro': self.fluoro_image is not None,
            'drr':   self.drr_image is not None,
            'yolo':  yolo_ready(),
            'reg':   self.result is not None,
        }
        for key, ok in checks.items():
            dot = self._chk_indicators.get(key)
            if dot:
                dot.setStyleSheet(
                    f'color:{ACCENT2};font-size:14px;border:none;background:transparent;'
                    if ok else
                    f'color:{TEXT_DIM};font-size:14px;border:none;background:transparent;'
                )

    # ══════════════════════════════════════════════════════════════════════════
    # Gestion des itérations
    # ══════════════════════════════════════════════════════════════════════════

    def _save_iteration(self):
        """Crée une nouvelle itération à partir de l'état courant."""
        it = {
            'lao': self.sp_lao.value(),
            'cran': self.sp_cran.value(),
            'table': self.sp_table.value(),
            'fov': self.sp_fov.value(),
            'size': self.sp_size.value(),
            'drr_image': self.drr_image.copy() if self.drr_image is not None else None,
            'proj_masks': {k: v.copy() for k, v in self.proj_masks.items()},
            'result': None,
        }
        self._iterations.append(it)
        self._current_iter_idx = len(self._iterations) - 1
        self._refresh_iter_list()

    def _refresh_iter_list(self):
        """Met à jour la liste d'itérations dans la sidebar."""
        while self._iter_vbox.count():
            item = self._iter_vbox.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        if not self._iterations:
            lbl = QLabel('Aucune iteration'); lbl.setObjectName('dim')
            self._iter_vbox.addWidget(lbl)
            self.btn_compare.setEnabled(False)
            return
        for i, it in enumerate(self._iterations):
            card = QWidget()
            is_current = (i == self._current_iter_idx)
            border_col = ACCENT if is_current else BORDER2
            card.setStyleSheet(
                f'background:{CARD_BG};border:1px solid {border_col};border-radius:5px;')
            cl = QHBoxLayout(card)
            cl.setContentsMargins(6, 4, 6, 4); cl.setSpacing(6)
            thumb = QLabel(); thumb.setFixedSize(40, 40)
            drr = it.get('drr_image')
            if drr is not None:
                u8 = (np.clip(drr, 0, 1) * 255).astype(np.uint8)
                t = cv2.resize(u8, (40, 40), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(t, cv2.COLOR_GRAY2RGB)
                qi = QImage(rgb.data, 40, 40, 40 * 3, QImage.Format_RGB888)
                thumb.setPixmap(QPixmap.fromImage(qi).copy())
            thumb.setStyleSheet(f'border:1px solid {BORDER2};border-radius:3px;background:{DARK_BG};')
            cl.addWidget(thumb)
            info_w = QWidget(); info_w.setStyleSheet('background:transparent;border:none;')
            info_l = QVBoxLayout(info_w); info_l.setSpacing(1); info_l.setContentsMargins(0, 0, 0, 0)
            lbl_hdr = QLabel(f'#{i+1}  LAO={it["lao"]:+.1f}  CRAN={it["cran"]:+.1f}')
            lbl_hdr.setStyleSheet(f'color:{TEXT};font-size:10px;border:none;background:transparent;')
            info_l.addWidget(lbl_hdr)
            res = it.get('result')
            if res:
                iou_v = res['iou']
                col = ACCENT2 if iou_v > 0.5 else (WARN if iou_v > 0.25 else ERR)
                m_lbl = QLabel(f'IoU={iou_v:.3f}  Dice={res["dice"]:.3f}')
                m_lbl.setStyleSheet(f'color:{col};font-size:10px;font-weight:600;border:none;background:transparent;')
                info_l.addWidget(m_lbl)
            else:
                nr = QLabel('Non recale')
                nr.setStyleSheet(f'color:{TEXT_DIM};font-size:10px;border:none;background:transparent;')
                info_l.addWidget(nr)
            cl.addWidget(info_w, 1)
            btn = QPushButton('\u25b6'); btn.setFixedSize(28, 28)
            btn.setToolTip('Charger cette iteration')
            btn.clicked.connect(lambda checked, idx=i: self._load_iteration(idx))
            cl.addWidget(btn)
            self._iter_vbox.addWidget(card)
        self.btn_compare.setEnabled(len(self._iterations) >= 2)

    def _load_iteration(self, idx):
        """Recharge une itération dans l'UI."""
        if idx < 0 or idx >= len(self._iterations):
            return
        it = self._iterations[idx]
        self._current_iter_idx = idx
        self.sp_lao.setValue(it['lao'])
        self.sp_cran.setValue(it['cran'])
        self.sp_table.setValue(it['table'])
        self.sp_fov.setValue(it['fov'])
        self.sp_size.setValue(it['size'])
        self.drr_image = it['drr_image']
        self._set_drr_base_image(self.drr_image)
        self.proj_masks = it['proj_masks']
        if self.drr_image is not None:
            self.cv_drr.set_image(self.drr_image)
            if self.chk_use_seg.isChecked() and self.proj_masks:
                fused = None
                for m in self.proj_masks.values():
                    if m is None or m.sum() == 0:
                        continue
                    fused = m.copy() if fused is None else np.clip(fused + m, 0, 1)
                if fused is not None:
                    self.cv_drr.set_mask('vertebrae', fused)
        res = it.get('result')
        if res:
            self.result = res
            iou_v, dice_v = res['iou'], res['dice']
            col = ACCENT2 if iou_v > 0.5 else (WARN if iou_v > 0.25 else ERR)
            self.lbl_iou.setText(f'{iou_v:.3f}')
            self.lbl_iou.setStyleSheet(f'font-size:22px;font-weight:700;color:{col};')
            self.lbl_dice.setText(f'{dice_v:.3f}')
            self.lbl_dice.setStyleSheet(f'font-size:22px;font-weight:700;color:{col};')
            self.lbl_tx.setText(f'tx = {res["tx"]:+.1f} px')
            self.lbl_ty.setText(f'ty = {res["ty"]:+.1f} px')
            self.lbl_rot.setText(f'rot = {res["angle"]:+.2f} deg')
            self.lbl_scale.setText(f'scale = {res["scale"]:.3f}')
            self._build_result(res)
            self.tabs.setCurrentIndex(3)
        else:
            self.tabs.setCurrentIndex(1)
        self._refresh_iter_list()
        self._status(f'Iteration #{idx+1} chargee')

    def _open_comparison(self):
        """Ouvre le dialogue de comparaison des itérations."""
        if len(self._iterations) < 2:
            self._err('Au moins 2 iterations necessaires pour comparer.')
            return
        dlg = ComparisonDialog(self._iterations, parent=self)
        dlg.exec_()

    # ══════════════════════════════════════════════════════════════════════════
    # Détection YOLO
    # ══════════════════════════════════════════════════════════════════════════

    def _load_yolo_default(self):
        """Charge automatiquement le modèle YOLO par défaut s'il existe."""
        default_path = 'data/model/best (3).pt'
        if os.path.exists(default_path):
            try:
                yolo_load(default_path)
                self.lbl_yolo.setText(os.path.basename(default_path))
                self._update_checklist()
                self._status(f'Modèle YOLO chargé automatiquement : {os.path.basename(default_path)}')
            except Exception as ex:
                self._status(f'Chargement YOLO auto échoué : {ex}')

    def _load_yolo_model(self):
        p, _ = QFileDialog.getOpenFileName(self, 'Modèle YOLO', '', 'PyTorch (*.pt)')
        if not p:
            return
        try:
            yolo_load(p)
            self.lbl_yolo.setText(os.path.basename(p))
            self._update_checklist()
            self._status(f'YOLO : {os.path.basename(p)}')
        except Exception as ex:
            self._err(str(ex))

    def _yolo_kw(self):
        return dict(
            conf=self.sp_yolo_conf.value() / 100.0,
            iou=self.sp_yolo_iou.value() / 100.0,
            imgsz=self.sp_yolo_imgsz.value(),
            pp={'gamma': self.sp_yolo_gamma.value(),
                'contrast': self.sp_yolo_contrast.value(),
                'invert': self.chk_yolo_invert.isChecked()})

    def _detect_fluoro(self):
        if not yolo_ready():
            self._err('Chargez un modèle YOLO (.pt) d\'abord.'); return
        if self.fluoro_image is None:
            self._err('Chargez une fluoroscopie d\'abord.'); return
        kw = {**self._yolo_kw(), 'img': self.fluoro_image,
              'target': 'fluoro', 'preprocess': True}
        self.worker = WorkerThread('yolo_detect', kw)
        self.worker.progress.connect(self._on_prog)
        self.worker.result.connect(self._on_yolo_result)
        self.worker.error.connect(self._on_err)
        self.worker.start()

    def _detect_drr(self):
        if not yolo_ready():
            self._err('Chargez un modèle YOLO (.pt) d\'abord.'); return
        if self.drr_image is None:
            self._err('Générez ou chargez un DRR d\'abord.'); return
        kw = {**self._yolo_kw(), 'img': self.drr_image,
              'target': 'drr', 'preprocess': False}
        self.worker = WorkerThread('yolo_detect', kw)
        self.worker.progress.connect(self._on_prog)
        self.worker.result.connect(self._on_yolo_result)
        self.worker.error.connect(self._on_err)
        self.worker.start()

    def _on_yolo_result(self, res):
        target = res['target']
        n = res['n_detections']
        if target == 'fluoro':
            self._yolo_det_fl = res
        else:
            self._yolo_det_drr = res

        # Ouvrir le panneau de sélection YOLO
        dlg = YoloDetectionPanel(res, target, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            selected_boxes = dlg.get_selected_detections()
            if not selected_boxes:
                self._err('Aucune détection sélectionnée.'); return
        else:
            selected_boxes = dlg.get_selected_detections() or res['boxes']

        # Construire le masque et l'injecter
        canvas = self.cv_fl if target == 'fluoro' else self.cv_drr
        h, w = res['mask'].shape
        mask = boxes_to_mask(selected_boxes, h, w)
        mask_f32 = mask.astype(np.float32) / 255.0
        canvas.set_mask('vertebrae', mask_f32)

        tab = 0 if target == 'fluoro' else 1
        self.tabs.setCurrentIndex(tab)

        n_sel = len(selected_boxes)
        self.lbl_yolo_status.setText(
            f'{target.upper()} : {n_sel}/{n} vertèbre(s) retenue(s)\n'
            f'Masque injecté dans l\'onglet {"Fixe" if target=="fluoro" else "DRR"}')
        self._status(f'YOLO {target} : {n_sel} vertèbre(s)')

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, '_busy_overlay') and self.centralWidget() is not None:
            self._busy_overlay.setGeometry(self.centralWidget().rect())


# ══════════════════════════════════════════════════════════════════════════════
# Dialogue de comparaison des itérations
# ══════════════════════════════════════════════════════════════════════════════


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    return app.exec_()

