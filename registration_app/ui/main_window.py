"""Main window orchestration for the registration application."""

import json
import os

import cv2
import nibabel as nib
import numpy as np
import pandas as pd

try:
    import pydicom
except ImportError:
    pydicom = None

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

from core.drr_generator import load_ct
from core.registration import apply_full_transform
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
    ResultPanel,
    FinalOverlayPanel,
    _make_svg_icon,
)
from ui.dialogs import (
    YoloDetectionPanel,
    DualYoloSelectionDialog,
    ComparisonDialog,
)
from utils.dicom_io import read_dicom_fluoro, read_metadata_csv

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
        self.dicom_meta = {}
        self.fluoro_image = None; self.result = None
        self._loaded_images = []; self._pending_csv = None
        self._iterations = []
        self._current_iter_idx = -1
        self._build_ui()
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

        # Dots fantômes pour DRR / YOLO / Reg (non affichés, mis à jour par _update_checklist)
        for _key in ('drr', 'yolo', 'reg'):
            _d = QLabel('●'); _d.setFixedWidth(16)
            _d.setStyleSheet(f'color:{TEXT_DIM};font-size:14px;border:none;background:transparent;')
            _d.hide()
            self._chk_indicators[_key] = _d

        ll.addWidget(sec_data)

        # ── IMAGES ────────────────────────────────────────────────────────────
        self._sec_images = CollapsibleSection('IMAGES', starts_open=False)
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
        self.sp_size = QSpinBox(); self.sp_size.setRange(128, 1024); self.sp_size.setValue(256); self.sp_size.setSingleStep(64)
        gdl.addWidget(self.sp_size, 4, 1)
        sec_drr.addWidget(drr_grid)
        self.btn_drr = QPushButton('Generer DRR'); self.btn_drr.setObjectName('primary')
        self.btn_drr.clicked.connect(self.generate_drr); self.btn_drr.setEnabled(False)
        sec_drr.addWidget(self.btn_drr)
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

        self.btn_eraser = QPushButton(); self.btn_eraser.setObjectName('tool')
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.setFixedSize(34, 34); self.btn_eraser.setToolTip('Gomme (effacer)')
        _ic = _make_svg_icon(_eraser_d)
        if _ic: self.btn_eraser.setIcon(_ic); self.btn_eraser.setIconSize(QSize(20, 20))
        else: self.btn_eraser.setText('⌫')

        self.btn_pencil.clicked.connect(lambda: self._set_tool('pencil'))
        self.btn_rect.clicked.connect(lambda: self._set_tool('rectangle'))
        self.btn_eraser.clicked.connect(lambda: self._set_tool('eraser'))
        tr.addWidget(self.btn_pencil); tr.addWidget(self.btn_rect); tr.addWidget(self.btn_eraser)
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

        # ── DETECTION YOLO ────────────────────────────────────────────────────
        sec_yolo = CollapsibleSection('DETECTION YOLO', starts_open=True)

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

        self.lbl_auto_status = QLabel('')
        self.lbl_auto_status.setObjectName('dim')
        self.lbl_auto_status.setWordWrap(True)
        sec_auto.addWidget(self.lbl_auto_status)

        ll.addWidget(sec_auto)

        ll.addStretch()


        # ── RESULTATS ─────────────────────────────────────────────────────────
        sec_res = CollapsibleSection('RESULTATS', starts_open=False)

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
        self.cv_drr = AnnotationCanvas(); self.cv_drr.mask_updated.connect(self._on_mask_upd)
        self.result_panel = ResultPanel()

        for cv, hint, label in [
            (self.cv_fl, 'Image fixe -- dessiner les structures a recaler', 'Fixe'),
            (self.cv_drr, 'Image mobile -- dessiner les memes structures', 'Mobile'),
        ]:
            self.tabs.addTab(self._wrap(cv, hint), label)
        self.tabs.addTab(self.result_panel, 'Resultat')

        self.overlay_panel = FinalOverlayPanel()
        self.tabs.addTab(self.overlay_panel, 'Overlay')

        for cv in [self.cv_fl, self.cv_drr]:
            cv.set_tool('pencil'); cv.set_active('vertebrae')

        root.addWidget(self.tabs, 1)
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

    # ── Slots UI ──────────────────────────────────────────────────────────────

    def _set_tool(self, tool):
        for btn, t in [(self.btn_pencil, 'pencil'), (self.btn_rect, 'rectangle'), (self.btn_eraser, 'eraser')]:
            btn.setChecked(t == tool)
        for cv in [self.cv_fl, self.cv_drr]:
            cv.set_tool(tool)
        hints = {
            'pencil':    'Tracer le contour en continu, relacher pour valider',
            'rectangle': 'Cliquer-glisser pour dessiner un rectangle',
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

    def _save_mask(self):
        cv=self._active_cv(); m=cv.get_mask()
        if m is None or m.sum()==0: self._err('Masque vide'); return
        p,_=QFileDialog.getSaveFileName(self,'Sauvegarder','mask.png','PNG (*.png)')
        if p: cv2.imwrite(p,(m*255).astype(np.uint8)); self._status(f'Masque sauvegardé → {p}')

    def _active_cv(self): return self.cv_fl if self.tabs.currentIndex()==0 else self.cv_drr

    def _on_mask_upd(self):
        mask_fl = self.cv_fl.get_mask()
        mask_drr = self.cv_drr.get_mask()
        has_fl = mask_fl is not None and mask_fl.sum() > 0
        has_drr = mask_drr is not None and mask_drr.sum() > 0
        self.btn_reg.setEnabled(bool(has_fl and has_drr))

    # ── Drag & Drop / Gestion images ─────────────────────────────────────────

    def _on_files_dropped(self, files):
        csv_files = []
        nifti_files = []
        meta_csv_files = []
        other_files = []
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if f.lower().endswith('.nii.gz') or ext == '.nii':
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
            'Tous (*.nii *.nii.gz *.csv *.png *.jpg *.jpeg *.bmp *.tiff *.tif *.dcm *);;'
            'NIfTI (*.nii *.nii.gz);;Images (*.png *.jpg *.jpeg *.tiff *.bmp *.dcm);;'
            'DICOM (*.dcm *);;'
            'CSV labels/meta (*.csv)')
        if files:
            self._on_files_dropped(files)

    def _load_seg_auto(self, path, csv_path=None):
        try:
            sv = nib.load(path).get_fdata().astype(np.int16)
            self.seg_masks = {}
            if csv_path:
                df = pd.read_csv(csv_path)
                c = df.columns.tolist()
                for _, row in df.iterrows():
                    try:
                        idx = int(row[c[0]])
                    except (ValueError, TypeError):
                        continue  # skip non-integer rows (metadata CSV loaded by mistake)
                    name = str(row[c[1]]).strip()
                    if not name or idx == 0: continue
                    m = (sv == idx).astype(np.uint8)
                    if m.sum() == 0: continue
                    self.seg_masks[name] = m
            else:
                for idx in np.unique(sv):
                    if idx == 0: continue
                    m = (sv == idx).astype(np.uint8)
                    if m.sum() == 0: continue
                    self.seg_masks[f'label_{int(idx)}'] = m
            n = len(self.seg_masks)
            self.lbl_seg.setText(f'Seg : {os.path.basename(path)} ({n})')
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
        # Tenter DICOM pour .dcm ou fichiers sans extension (ex: IM0)
        is_dicom_ext = ext == '.dcm' or ext == ''
        if is_dicom_ext and pydicom is not None:
            try:
                img, dicom_meta = read_dicom_fluoro(path)
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            except Exception:
                img = None  # fallback to cv2
        if img is None:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self._err(f'Impossible de charger : {name}'); return
        img_float = img.astype(np.float32) / 255.0
        idx = len(self._loaded_images)
        entry = {'path': path, 'name': name, 'array': img_float, 'role': None, 'card': None,
                 'dicom_meta': dicom_meta}
        self._loaded_images.append(entry)
        self.lbl_no_images.hide()
        card = ImageCard(idx, name, img_float)
        card.role_changed.connect(self._assign_image)
        entry['card'] = card
        self._images_vbox.addWidget(card)
        if idx == 0:
            card.set_role_external('fixed')
            self._assign_image(idx, 'fixed')

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
        img = self._loaded_images[index]['array']
        name = self._loaded_images[index]['name']
        if role == 'fixed':
            self.fluoro_image = img           # résolution native conservée
            self.cv_fl.set_image(img)
            self.tabs.setCurrentIndex(0)
            self._update_checklist()
            # Auto-remplir les paramètres DRR depuis les métadonnées DICOM
            meta = self._loaded_images[index].get('dicom_meta')
            if meta:
                self.dicom_meta = meta
                self._apply_dicom_meta(meta)
                self._status(f'Fluoro DICOM chargee — LAO={meta["lao"]:+.1f}deg  '
                              f'CRAN={meta["cran"]:+.1f}deg  FOV={meta["fov_mm"]:.0f}mm — '
                              'parametres DRR auto-remplis')
            else:
                self.dicom_meta = {}
                self.lbl_fluoro_meta.setText(f'Fluoro : {name}')
                self._status(f'Image fixe : {name}')
        elif role == 'mobile':
            self.drr_image = img
            self.proj_masks = {}
            self.cv_drr.set_image(img)
            self.tabs.setTabText(1, 'Mobile')
            self.tabs.setCurrentIndex(1)
            self._status(f'Image mobile : {name}')

    def _apply_dicom_meta(self, meta):
        """Remplit les spinboxes et labels UI depuis un dict de métadonnées."""
        self.sp_lao.setValue(meta['lao'])
        self.sp_cran.setValue(meta['cran'])
        self.sp_table.setValue(meta.get('table_angle', 0.0))
        self.sp_fov.setValue(float(meta.get('fov_mm', DEFAULT_FOV_MM) or DEFAULT_FOV_MM))

        fov = meta['fov_mm']
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
            self._update_checklist()
            self._status(f'CT chargé — axe AP={self.ap_axis} ({codes})')
        except Exception as ex: self._err(str(ex))

    def load_seg(self):
        p,_=QFileDialog.getOpenFileName(self,'Segmentation','','NIfTI (*.nii *.nii.gz)')
        if not p: return
        cp,_=QFileDialog.getOpenFileName(self,'Label CSV','','CSV (*.csv)')
        try:
            sv=nib.load(p).get_fdata().astype(np.int16)
            self.seg_masks={}

            if cp:
                df=pd.read_csv(cp); c=df.columns.tolist()
                # c[0] = index/valeur voxel, c[1] = nom du label
                for _,row in df.iterrows():
                    idx=int(row[c[0]]); name=str(row[c[1]]).strip()
                    if not name or idx==0: continue   # ignorer fond
                    m=(sv==idx).astype(np.uint8)
                    if m.sum()==0: continue            # ignorer labels vides
                    self.seg_masks[name]=m
            else:
                # Pas de CSV : une structure par valeur unique non nulle
                for idx in np.unique(sv):
                    if idx==0: continue
                    m=(sv==idx).astype(np.uint8)
                    if m.sum()==0: continue
                    self.seg_masks[f'label_{int(idx)}']=m

            n=len(self.seg_masks)
            total=sum(v.sum() for v in self.seg_masks.values())
            self.lbl_seg.setText(f'Seg: {os.path.basename(p)}\n  {n} structures · {total:,} voxels')
            self._update_checklist()
            self._status(f'Segmentation chargée — {n} structures : {", ".join(list(self.seg_masks)[:6])}{"…" if n>6 else ""}')
        except Exception as ex: self._err(str(ex))

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

    def generate_drr(self):
        if self.ct_path is None: self._err('Charger un CT d\'abord'); return
        self.btn_drr.setEnabled(False)
        kw=dict(ct_path=self.ct_path, ct_aff=self.ct_aff,
                lao_deg=self.sp_lao.value(), cran_deg=self.sp_cran.value() + 180,
                table_angle=self.sp_table.value(),
                output_size=self.sp_size.value(), masks=self.seg_masks,
                fov_mm=self.sp_fov.value(),
            sid_mm=self.dicom_meta.get('sid_mm', 1020.0),
            sod_mm=self.dicom_meta.get('sod_mm', 510.0),
                renderer='siddon')
        self.worker=WorkerThread('drr',kw)
        self.worker.progress.connect(self._on_prog); self.worker.result.connect(self._drr_done)
        self.worker.error.connect(self._on_err); self.worker.start()

    def _drr_done(self,res):
        self.drr_image=res['drr']; self.proj_masks=res.get('masks',{})
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
        self.worker=WorkerThread('register',kw)
        self.worker.progress.connect(self._on_prog); self.worker.result.connect(self._reg_done)
        self.worker.error.connect(self._on_err); self.worker.start()

    def _reg_done(self,res):
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
        self._build_result(res); self.tabs.setCurrentIndex(2)
        if 0 <= self._current_iter_idx < len(self._iterations):
            self._iterations[self._current_iter_idx]['result'] = res
            self._refresh_iter_list()
        self._update_overlay()
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
            seg_volumes=self.seg_masks,
            ct_affine=self.ct_aff,
            lao_deg=self.sp_lao.value(),
            cran_deg=self.sp_cran.value() + 180.0,
            table_angle=self.sp_table.value(),
            fov_mm=fov_for_projection,
        )
        # Basculer automatiquement sur l'onglet Overlay
        self.tabs.setCurrentIndex(3)

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
            seg_masks=self.seg_masks,
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
        self.worker = WorkerThread('auto_pipeline', kw)
        self.worker.progress.connect(self._on_prog)
        self.worker.result.connect(self._auto_done)
        self.worker.error.connect(self._on_auto_err)
        self.worker.start()

    def _auto_done(self, res):
        """Callback quand le pipeline auto émet un résultat (phase 1 ou final)."""

        # ── Phase 1 : sélection des vertèbres fluoro + DRR ───────────────────
        if res.get('_phase') == 'select_vertebrae':
            self._auto_intermediate = res
            self.lbl_auto_status.setText('Sélectionnez les vertèbres (fluoro + DRR)…')

            # Injecter le DRR dans l'UI dès maintenant
            self.drr_image = res['drr_image']
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

            sel_fl, sel_drr = dlg.get_selections()
            if not sel_fl or not sel_drr:
                self._on_auto_err('Sélectionnez au moins une vertèbre de chaque côté.')
                return

            selected_fl = [res['boxes_fl'][i] for i in sel_fl
                           if i < len(res['boxes_fl'])]
            selected_drr = [res['drr_boxes'][i] for i in sel_drr
                            if i < len(res['drr_boxes'])]

            # Lancer la phase 2 (recalage élastique) en worker
            kw2 = dict(
                boxes_fl=selected_fl,
                boxes_drr=selected_drr,
                reg_size=res['reg_size'],
                drr_image=res['drr_image'],
                all_proj_masks=res['all_proj_masks'],
            )
            self.worker = WorkerThread('auto_phase2', kw2)
            self.worker.progress.connect(self._on_prog)
            self.worker.result.connect(self._auto_done)
            self.worker.error.connect(self._on_auto_err)
            self.worker.start()
            return

        # ── Phase finale : résultat complet ───────────────────────────────────
        self.btn_auto.setEnabled(True)
        self.btn_drr.setEnabled(True)

        # Injecter le DRR généré dans l'UI
        self.drr_image = res['drr_image']
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
        self.tabs.setCurrentIndex(2)

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
        self.btn_auto.setEnabled(True)
        self.btn_drr.setEnabled(True)
        self.btn_reg.setEnabled(True)
        self.lbl_auto_status.setText(f'Erreur : {msg.splitlines()[0]}')
        self._err(msg)

    def _on_prog(self,pct,msg): self.prog_bar.setValue(pct); self.lbl_prog.setText(msg); self._status(msg)
    def _on_err(self,msg): self.btn_drr.setEnabled(True); self.btn_reg.setEnabled(True); self.btn_detect_fl.setEnabled(True); self.btn_detect_drr.setEnabled(True); self._err(msg)
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
            self.tabs.setCurrentIndex(2)
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
        if n == 0:
            self._err(f'Aucune vertèbre détectée sur {target}.')
            return
        if target == 'fluoro':
            self._yolo_det_fl = res
        else:
            self._yolo_det_drr = res

        # Ouvrir le panneau de sélection YOLO
        dlg = YoloDetectionPanel(res, target, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            sel = dlg.get_selection()
            if not sel:
                self._err('Aucune détection sélectionnée.'); return
            selected_boxes = [res['boxes'][i] for i in sel]
        else:
            selected_boxes = res['boxes']

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


# ══════════════════════════════════════════════════════════════════════════════
# Dialogue de comparaison des itérations
# ══════════════════════════════════════════════════════════════════════════════


def main():
    app = QApplication([])
    win = MainWindow()
    win.show()
    return app.exec_()

