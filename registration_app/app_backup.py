"""
app.py — Application principale de recalage 2D/3D
Annotation manuelle + DRR generation + IoU registration
"""

import sys, os, json, threading
import numpy as np
import cv2
import nibabel as nib
import pandas as pd
from skimage import measure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QPushButton, QFileDialog, QSlider, QGroupBox,
    QSpinBox, QDoubleSpinBox, QProgressBar, QTabWidget, QSizePolicy,
    QScrollArea, QStatusBar, QFrame, QMessageBox, QComboBox, QCheckBox,
    QGridLayout, QToolBar, QAction, QButtonGroup, QRadioButton,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QTimer
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QBrush, QColor, QFont,
    QCursor, QIcon, QPainterPath,
)

# Ajoute le dossier courant au path
sys.path.insert(0, os.path.dirname(__file__))
from core.drr_generator import load_ct, generate_drr, project_mask_3d
from core.registration import register, apply_transform, iou_score, dice_score


# ══════════════════════════════════════════════════════════════════════════════
# Canvas d'annotation — dessin de masques sur image
# ══════════════════════════════════════════════════════════════════════════════

class AnnotationCanvas(QLabel):
    """
    Widget d'affichage d'image avec dessin de polygones/rectangles pour créer
    un masque binaire des vertèbres.
    """
    mask_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(512, 512)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(QCursor(Qt.CrossCursor))

        self._image_np  = None    # image source float32 [0,1]
        self._mask_np   = None    # masque binaire float32
        self._size      = 512

        self._drawing   = False
        self._tool      = 'rectangle'   # 'rectangle' | 'polygon' | 'eraser'
        self._brush_r   = 20

        self._rect_start = None
        self._rect_cur   = None
        self._poly_pts   = []
        self._rects      = []          # historique rectangles
        self._polys      = []          # historique polygones

        self._overlay_alpha = 0.45
        self._mask_color = QColor(50, 220, 120, 180)

    # ── Image ─────────────────────────────────────────────────────────────────
    def set_image(self, img_np: np.ndarray):
        """img_np : float32 [0,1] ou uint8"""
        self._size = img_np.shape[0]
        if img_np.dtype != np.uint8:
            img_u8 = (img_np * 255).clip(0, 255).astype(np.uint8)
        else:
            img_u8 = img_np
        self._image_np = img_u8
        self._mask_np  = np.zeros((self._size, self._size), np.float32)
        self._rects.clear()
        self._polys.clear()
        self._refresh()

    def set_mask(self, mask_np: np.ndarray):
        self._mask_np = mask_np.astype(np.float32)
        self._refresh()

    def get_mask(self) -> np.ndarray:
        return self._mask_np.copy() if self._mask_np is not None else None

    def set_tool(self, tool: str):
        self._tool = tool
        if tool == 'polygon':
            self._poly_pts = []

    def set_brush_radius(self, r: int):
        self._brush_r = r

    def clear_mask(self):
        if self._mask_np is not None:
            self._mask_np[:] = 0
            self._rects.clear()
            self._polys.clear()
            self._poly_pts = []
            self._refresh()
            self.mask_updated.emit()

    def undo(self):
        """Efface le dernier rectangle ou polygone."""
        if self._rects or self._polys:
            # Reconstruit le masque depuis l'historique
            self._mask_np[:] = 0
            if self._rects:
                all_rects = self._rects[:]
                self._rects.clear()
                for r in all_rects[:-1]:
                    self._add_rect(*r)
            elif self._polys:
                all_polys = self._polys[:]
                self._polys.clear()
                for p in all_polys[:-1]:
                    self._fill_polygon(p)
            self._refresh()
            self.mask_updated.emit()

    # ── Dessin interne ────────────────────────────────────────────────────────
    def _add_rect(self, x0, y0, x1, y1):
        x0, x1 = sorted([x0, x1]); y0, y1 = sorted([y0, y1])
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(self._size-1, x1); y1 = min(self._size-1, y1)
        if x1 > x0 and y1 > y0:
            self._mask_np[y0:y1, x0:x1] = 1.0
            self._rects.append((x0, y0, x1, y1))

    def _fill_polygon(self, pts):
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(self._mask_np, [pts_arr], 1.0)
        self._polys.append(pts)

    def _erase_at(self, x, y):
        r = self._brush_r
        cv2.circle(self._mask_np, (x, y), r, 0.0, -1)

    # ── Render ────────────────────────────────────────────────────────────────
    def _refresh(self):
        if self._image_np is None:
            return
        s = self._size

        # Base image RGB
        if self._image_np.ndim == 2:
            base = cv2.cvtColor(self._image_np, cv2.COLOR_GRAY2RGB)
        else:
            base = self._image_np.copy()

        # Overlay masque en vert translucide
        if self._mask_np is not None and self._mask_np.sum() > 0:
            overlay = base.copy()
            overlay[self._mask_np > 0] = [50, 220, 120]
            base = cv2.addWeighted(base, 1 - self._overlay_alpha,
                                   overlay, self._overlay_alpha, 0)
            # Contour du masque
            mask_u8 = (self._mask_np * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base, contours, -1, (50, 220, 120), 2)

        # Rectangle en cours de dessin
        if self._rect_start and self._rect_cur:
            x0, y0 = self._rect_start
            x1, y1 = self._rect_cur
            cv2.rectangle(base, (min(x0,x1), min(y0,y1)),
                          (max(x0,x1), max(y0,y1)), (255, 160, 50), 2)

        # Polygone en cours
        if self._poly_pts:
            pts = np.array(self._poly_pts, dtype=np.int32)
            cv2.polylines(base, [pts], False, (100, 180, 255), 2)
            for p in self._poly_pts:
                cv2.circle(base, p, 4, (100, 180, 255), -1)

        # Convertit en QPixmap et adapte à la taille du widget
        h, w = base.shape[:2]
        qimg = QImage(base.data, w, h, 3*w, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        self.setPixmap(pix.scaled(self.width(), self.height(),
                                   Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # ── Events souris ─────────────────────────────────────────────────────────
    def _widget_to_image(self, qx, qy):
        """Convertit les coordonnées widget → coordonnées image."""
        if self._image_np is None:
            return 0, 0
        s = self._size
        # Taille affichée (KeepAspectRatio)
        ww, wh = self.width(), self.height()
        scale = min(ww / s, wh / s)
        off_x = (ww - s * scale) / 2
        off_y = (wh - s * scale) / 2
        ix = int((qx - off_x) / scale)
        iy = int((qy - off_y) / scale)
        return max(0, min(s-1, ix)), max(0, min(s-1, iy))

    def mousePressEvent(self, e):
        if self._image_np is None: return
        ix, iy = self._widget_to_image(e.x(), e.y())

        if self._tool == 'rectangle':
            self._drawing = True
            self._rect_start = (ix, iy)
            self._rect_cur   = (ix, iy)

        elif self._tool == 'polygon':
            self._poly_pts.append((ix, iy))
            self._refresh()

        elif self._tool == 'eraser':
            self._drawing = True
            self._erase_at(ix, iy)
            self._refresh()

    def mouseMoveEvent(self, e):
        if self._image_np is None: return
        ix, iy = self._widget_to_image(e.x(), e.y())

        if self._tool == 'rectangle' and self._drawing:
            self._rect_cur = (ix, iy)
            self._refresh()

        elif self._tool == 'eraser' and self._drawing:
            self._erase_at(ix, iy)
            self._refresh()

    def mouseReleaseEvent(self, e):
        if self._image_np is None: return
        ix, iy = self._widget_to_image(e.x(), e.y())

        if self._tool == 'rectangle' and self._drawing:
            self._drawing = False
            x0, y0 = self._rect_start
            self._add_rect(x0, y0, ix, iy)
            self._rect_start = None
            self._rect_cur   = None
            self._refresh()
            self.mask_updated.emit()

        elif self._tool == 'eraser' and self._drawing:
            self._drawing = False
            self.mask_updated.emit()

    def mouseDoubleClickEvent(self, e):
        """Double-clic = fermer le polygone."""
        if self._tool == 'polygon' and len(self._poly_pts) >= 3:
            self._fill_polygon(self._poly_pts)
            self._poly_pts = []
            self._refresh()
            self.mask_updated.emit()

    def resizeEvent(self, e):
        self._refresh()


# ══════════════════════════════════════════════════════════════════════════════
# Thread de génération DRR + recalage (pour ne pas bloquer l'UI)
# ══════════════════════════════════════════════════════════════════════════════

class WorkerThread(QThread):
    progress  = pyqtSignal(int, str)    # (percent, message)
    result    = pyqtSignal(dict)
    error     = pyqtSignal(str)

    def __init__(self, task, kwargs):
        super().__init__()
        self.task   = task
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task == 'drr':
                self._run_drr()
            elif self.task == 'register':
                self._run_register()
        except Exception as ex:
            self.error.emit(str(ex))

    def _run_drr(self):
        self.progress.emit(5, 'Chargement CT…')
        ct_vol     = self.kwargs['ct_vol']
        voxel_mm   = self.kwargs['voxel_mm']
        ap_axis    = self.kwargs['ap_axis']
        lao_deg    = self.kwargs['lao_deg']
        cran_deg   = self.kwargs['cran_deg']
        output_size= self.kwargs['output_size']

        self.progress.emit(20, 'Génération DRR (Beer-Lambert)…')
        drr = generate_drr(ct_vol, voxel_mm, ap_axis=ap_axis,
                            lao_deg=lao_deg, cran_deg=cran_deg,
                            output_size=output_size,
                            hu_min=-500, hu_max=2000,
                            invert=True)

        self.progress.emit(70, 'Projection masques segmentation…')
        masks_out = {}
        for name, mask in self.kwargs.get('masks', {}).items():
            masks_out[name] = project_mask_3d(
                mask, voxel_mm, ap_axis=ap_axis,
                lao_deg=lao_deg, cran_deg=cran_deg,
                output_size=output_size)

        self.progress.emit(100, 'DRR généré ✓')
        self.result.emit({'drr': drr, 'masks': masks_out})

    def _run_register(self):
        self.progress.emit(5, 'Initialisation…')

        def progress_cb(frac, iou):
            self.progress.emit(int(5 + frac * 90),
                               f'Optimisation… IoU={iou:.3f}')

        res = register(
            mask_moving=self.kwargs['mask_moving'],
            mask_fixed=self.kwargs['mask_fixed'],
            search_tx_px=self.kwargs['search_tx'],
            search_ty_px=self.kwargs['search_ty'],
            search_rot_deg=self.kwargs['search_rot'],
            progress_cb=progress_cb,
        )
        self.progress.emit(100, f"Recalage terminé — IoU={res['iou']:.3f}")
        self.result.emit(res)


# ══════════════════════════════════════════════════════════════════════════════
# Fenêtre principale
# ══════════════════════════════════════════════════════════════════════════════

DARK_BG    = '#0f1117'
PANEL_BG   = '#1a1d27'
BORDER     = '#2a2d3e'
ACCENT     = '#3d8ef0'
ACCENT2    = '#32c87a'
TEXT       = '#e8eaf0'
TEXT_DIM   = '#7a7d8e'
WARN       = '#f0a832'
ERR        = '#e05555'


STYLE = f"""
QMainWindow, QWidget {{ background: {DARK_BG}; color: {TEXT}; font-family: 'Consolas', monospace; font-size: 12px; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 6px; margin-top: 10px; padding-top: 8px; color: {TEXT_DIM}; font-size: 11px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {ACCENT}; }}
QPushButton {{ background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 5px; padding: 6px 14px; color: {TEXT}; }}
QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
QPushButton:pressed {{ background: {ACCENT}; color: white; }}
QPushButton#primary {{ background: {ACCENT}; border-color: {ACCENT}; color: white; font-weight: bold; }}
QPushButton#primary:hover {{ background: #5ba3f5; }}
QPushButton#success {{ background: #1e4d2b; border-color: {ACCENT2}; color: {ACCENT2}; font-weight: bold; }}
QPushButton#success:hover {{ background: #2a6e3e; }}
QPushButton#warn {{ background: #4d3a10; border-color: {WARN}; color: {WARN}; }}
QPushButton#warn:hover {{ background: #6e5218; }}
QPushButton#danger {{ background: #3d1a1a; border-color: {ERR}; color: {ERR}; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ width: 14px; height: 14px; background: {ACCENT}; border-radius: 7px; margin: -5px 0; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QDoubleSpinBox, QSpinBox, QComboBox {{ background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px; padding: 3px 6px; color: {TEXT}; }}
QProgressBar {{ background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 4px; height: 8px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; background: {PANEL_BG}; border-radius: 6px; }}
QTabBar::tab {{ background: {DARK_BG}; border: 1px solid {BORDER}; padding: 6px 16px; color: {TEXT_DIM}; border-radius: 4px 4px 0 0; }}
QTabBar::tab:selected {{ background: {PANEL_BG}; color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QLabel {{ color: {TEXT}; }}
QLabel#dim {{ color: {TEXT_DIM}; font-size: 11px; }}
QLabel#title {{ color: {ACCENT}; font-size: 13px; font-weight: bold; }}
QLabel#metric {{ font-size: 22px; font-weight: bold; color: {ACCENT2}; }}
QSplitter::handle {{ background: {BORDER}; width: 2px; }}
QStatusBar {{ background: {PANEL_BG}; border-top: 1px solid {BORDER}; color: {TEXT_DIM}; font-size: 11px; }}
QScrollBar:vertical {{ background: {DARK_BG}; width: 8px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 20px; }}
QFrame#separator {{ background: {BORDER}; max-height: 1px; }}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('2D/3D Registration — EP Lab')
        self.setMinimumSize(1400, 900)
        self.setStyleSheet(STYLE)

        # ── État applicatif ────────────────────────────────────────────────────
        self.ct_vol      = None
        self.voxel_mm    = None
        self.ct_aff      = None
        self.ap_axis     = 1
        self.seg_masks   = {}
        self.drr_image   = None
        self.fluoro_image= None
        self.result      = None

        self._build_ui()
        self._status('Bienvenue — chargez un CT scan pour commencer')

    # ══════════════════════════════════════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Panneau gauche (contrôles) ─────────────────────────────────────────
        left = QWidget(); left.setFixedWidth(300)
        left.setStyleSheet(f'background: {PANEL_BG}; border-radius: 8px;')
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(12, 12, 12, 12)
        left_l.setSpacing(8)

        # Titre
        t = QLabel('2D / 3D\nREGISTRATION'); t.setObjectName('title')
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet(f'color: {ACCENT}; font-size: 16px; font-weight: bold; letter-spacing: 2px;')
        left_l.addWidget(t)

        sep = QFrame(); sep.setObjectName('separator'); sep.setFrameShape(QFrame.HLine)
        left_l.addWidget(sep)

        # ── Chargement fichiers ────────────────────────────────────────────────
        grp_files = QGroupBox('FICHIERS')
        gl = QVBoxLayout(grp_files)

        self.btn_load_ct = QPushButton('📂  Charger CT (.nii/.nii.gz)')
        self.btn_load_ct.setObjectName('primary')
        self.btn_load_ct.clicked.connect(self.load_ct)
        gl.addWidget(self.btn_load_ct)

        self.btn_load_seg = QPushButton('📂  Charger Segmentation')
        self.btn_load_seg.clicked.connect(self.load_seg)
        gl.addWidget(self.btn_load_seg)

        self.btn_load_fluoro = QPushButton('📂  Charger Fluoroscopie (.png)')
        self.btn_load_fluoro.clicked.connect(self.load_fluoro)
        gl.addWidget(self.btn_load_fluoro)

        self.lbl_ct    = QLabel('CT : —'); self.lbl_ct.setObjectName('dim')
        self.lbl_seg   = QLabel('Seg: —'); self.lbl_seg.setObjectName('dim')
        self.lbl_fluoro= QLabel('Fluoro: —'); self.lbl_fluoro.setObjectName('dim')
        for l in [self.lbl_ct, self.lbl_seg, self.lbl_fluoro]:
            gl.addWidget(l)
        left_l.addWidget(grp_files)

        # ── Paramètres DRR ────────────────────────────────────────────────────
        grp_drr = QGroupBox('PARAMÈTRES DRR')
        gdl = QGridLayout(grp_drr)

        gdl.addWidget(QLabel('LAO/RAO (°)'), 0, 0)
        self.spin_lao = QDoubleSpinBox()
        self.spin_lao.setRange(-90, 90); self.spin_lao.setValue(0); self.spin_lao.setSingleStep(5)
        gdl.addWidget(self.spin_lao, 0, 1)

        gdl.addWidget(QLabel('Cran/Caud (°)'), 1, 0)
        self.spin_cran = QDoubleSpinBox()
        self.spin_cran.setRange(-45, 45); self.spin_cran.setValue(0); self.spin_cran.setSingleStep(5)
        gdl.addWidget(self.spin_cran, 1, 1)

        gdl.addWidget(QLabel('Résolution (px)'), 2, 0)
        self.spin_size = QSpinBox()
        self.spin_size.setRange(256, 1024); self.spin_size.setValue(512); self.spin_size.setSingleStep(64)
        gdl.addWidget(self.spin_size, 2, 1)

        self.btn_gen_drr = QPushButton('⚡  Générer DRR')
        self.btn_gen_drr.setObjectName('primary')
        self.btn_gen_drr.clicked.connect(self.generate_drr)
        self.btn_gen_drr.setEnabled(False)
        gdl.addWidget(self.btn_gen_drr, 3, 0, 1, 2)
        left_l.addWidget(grp_drr)

        # ── Outils d'annotation ────────────────────────────────────────────────
        grp_ann = QGroupBox('ANNOTATION MASQUE')
        gal = QVBoxLayout(grp_ann)

        tool_row = QHBoxLayout()
        self.btn_tool_rect = QPushButton('▭ Rectangle')
        self.btn_tool_poly = QPushButton('⬠ Polygone')
        self.btn_tool_erase= QPushButton('◌ Gomme')
        self.btn_tool_rect.setCheckable(True); self.btn_tool_rect.setChecked(True)
        self.btn_tool_poly.setCheckable(True)
        self.btn_tool_erase.setCheckable(True)
        for b in [self.btn_tool_rect, self.btn_tool_poly, self.btn_tool_erase]:
            b.setStyleSheet(b.styleSheet())
            tool_row.addWidget(b)
        self.btn_tool_rect.clicked.connect(lambda: self._set_tool('rectangle'))
        self.btn_tool_poly.clicked.connect(lambda: self._set_tool('polygon'))
        self.btn_tool_erase.clicked.connect(lambda: self._set_tool('eraser'))
        gal.addLayout(tool_row)

        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel('Gomme r:'))
        self.sl_brush = QSlider(Qt.Horizontal)
        self.sl_brush.setRange(5, 60); self.sl_brush.setValue(20)
        self.sl_brush.valueChanged.connect(self._on_brush_change)
        brush_row.addWidget(self.sl_brush)
        self.lbl_brush = QLabel('20px'); self.lbl_brush.setObjectName('dim')
        brush_row.addWidget(self.lbl_brush)
        gal.addLayout(brush_row)

        ann_btns = QHBoxLayout()
        btn_undo  = QPushButton('↩ Undo');   btn_undo.setObjectName('warn')
        btn_clear = QPushButton('🗑 Clear');  btn_clear.setObjectName('danger')
        btn_save_mask = QPushButton('💾 Sauver')
        btn_undo.clicked.connect(self._undo)
        btn_clear.clicked.connect(self._clear_mask)
        btn_save_mask.clicked.connect(self._save_mask)
        for b in [btn_undo, btn_clear, btn_save_mask]: ann_btns.addWidget(b)
        gal.addLayout(ann_btns)

        # Sélecteur de canvas actif
        self.combo_active = QComboBox()
        self.combo_active.addItems(['Annoter Fluoroscopie', 'Annoter DRR'])
        self.combo_active.currentIndexChanged.connect(self._on_active_canvas_change)
        self.combo_active.setStyleSheet(f'background: {PANEL_BG}; color: {ACCENT};')
        gal.addWidget(self.combo_active)
        left_l.addWidget(grp_ann)

        # ── Recalage ──────────────────────────────────────────────────────────
        grp_reg = QGroupBox('RECALAGE')
        grl = QVBoxLayout(grp_reg)

        search_grid = QGridLayout()
        search_grid.addWidget(QLabel('Rech. tx (px)'), 0, 0)
        self.spin_stx = QSpinBox(); self.spin_stx.setRange(10, 300); self.spin_stx.setValue(100)
        search_grid.addWidget(self.spin_stx, 0, 1)
        search_grid.addWidget(QLabel('Rech. ty (px)'), 1, 0)
        self.spin_sty = QSpinBox(); self.spin_sty.setRange(10, 300); self.spin_sty.setValue(100)
        search_grid.addWidget(self.spin_sty, 1, 1)
        search_grid.addWidget(QLabel('Rech. rot (°)'), 2, 0)
        self.spin_srot= QSpinBox(); self.spin_srot.setRange(5, 45); self.spin_srot.setValue(20)
        search_grid.addWidget(self.spin_srot, 2, 1)
        grl.addLayout(search_grid)

        self.btn_register = QPushButton('🎯  Lancer le Recalage')
        self.btn_register.setObjectName('success')
        self.btn_register.clicked.connect(self.run_registration)
        self.btn_register.setEnabled(False)
        grl.addWidget(self.btn_register)
        left_l.addWidget(grp_reg)

        # ── Métriques ─────────────────────────────────────────────────────────
        grp_met = QGroupBox('MÉTRIQUES')
        gml = QGridLayout(grp_met)
        self.lbl_iou  = QLabel('—'); self.lbl_iou.setObjectName('metric')
        self.lbl_dice = QLabel('—'); self.lbl_dice.setObjectName('metric')
        self.lbl_tx   = QLabel('tx: —'); self.lbl_tx.setObjectName('dim')
        self.lbl_ty   = QLabel('ty: —'); self.lbl_ty.setObjectName('dim')
        self.lbl_rot  = QLabel('rot: —'); self.lbl_rot.setObjectName('dim')
        gml.addWidget(QLabel('IoU'), 0, 0); gml.addWidget(self.lbl_iou,  0, 1)
        gml.addWidget(QLabel('Dice'),1, 0); gml.addWidget(self.lbl_dice, 1, 1)
        gml.addWidget(self.lbl_tx,   2, 0, 1, 2)
        gml.addWidget(self.lbl_ty,   3, 0, 1, 2)
        gml.addWidget(self.lbl_rot,  4, 0, 1, 2)
        left_l.addWidget(grp_met)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_l.addWidget(self.progress_bar)
        self.lbl_progress = QLabel('')
        self.lbl_progress.setObjectName('dim')
        self.lbl_progress.setAlignment(Qt.AlignCenter)
        left_l.addWidget(self.lbl_progress)

        btn_export = QPushButton('📤  Exporter résultats')
        btn_export.clicked.connect(self.export_results)
        left_l.addWidget(btn_export)

        left_l.addStretch()
        root.addWidget(left)

        # ── Zone centrale : onglets ────────────────────────────────────────────
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Tab 1 : Fluoroscopie
        self.canvas_fluoro = AnnotationCanvas()
        self.canvas_fluoro.mask_updated.connect(self._on_mask_updated)
        self.tabs.addTab(self._wrap_canvas(self.canvas_fluoro, 'Fluoroscopie — dessiner les contours vertèbres'),
                         '🩻 Fluoroscopie')

        # Tab 2 : DRR
        self.canvas_drr = AnnotationCanvas()
        self.canvas_drr.mask_updated.connect(self._on_mask_updated)
        self.tabs.addTab(self._wrap_canvas(self.canvas_drr, 'DRR — dessiner les contours vertèbres'),
                         '📡 DRR')

        # Tab 3 : Résultat
        self.canvas_result = AnnotationCanvas()
        self.canvas_result.setCursor(QCursor(Qt.ArrowCursor))
        self.tabs.addTab(self._wrap_canvas(self.canvas_result, 'Résultat du recalage'),
                         '✅ Résultat')

        root.addWidget(self.tabs, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _wrap_canvas(self, canvas, hint: str) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(4, 4, 4, 4)
        hint_lbl = QLabel(hint)
        hint_lbl.setObjectName('dim')
        hint_lbl.setAlignment(Qt.AlignCenter)
        l.addWidget(hint_lbl)
        l.addWidget(canvas, 1)
        return w

    # ══════════════════════════════════════════════════════════════════════════
    # Actions
    # ══════════════════════════════════════════════════════════════════════════

    def load_ct(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Charger CT', '', 'NIfTI (*.nii *.nii.gz)')
        if not path: return
        self._status('Chargement CT…')
        try:
            self.ct_vol, self.voxel_mm, self.ct_aff, self.ct_nib, self.ap_axis, codes = load_ct(path)
            name = os.path.basename(path)
            self.lbl_ct.setText(f'CT: {name}\n  {self.ct_vol.shape} | {self.voxel_mm.round(2)} mm\n  AP axis={self.ap_axis} {codes}')
            self.btn_gen_drr.setEnabled(True)
            self._status(f'CT chargé — {self.ct_vol.shape} — axe AP={self.ap_axis}')
        except Exception as ex:
            self._error(f'Erreur CT : {ex}')

    def load_seg(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Charger Segmentation', '', 'NIfTI (*.nii *.nii.gz)')
        if not path: return
        csv_path, _ = QFileDialog.getOpenFileName(
            self, 'Charger Label CSV', '', 'CSV (*.csv)')

        try:
            seg_nib = nib.load(path)
            seg_vol = seg_nib.get_fdata().astype(np.int16)

            label2idx = {}
            if csv_path:
                df = pd.read_csv(csv_path)
                cols = df.columns.tolist()
                label2idx = {row[cols[1]].strip().lower(): int(row[cols[0]])
                             for _, row in df.iterrows()}

            def get_mask(names):
                m = np.zeros(seg_vol.shape, bool)
                for n in names:
                    k = n.strip().lower()
                    if k in label2idx:
                        m |= (seg_vol == label2idx[k])
                return m.astype(np.uint8)

            self.seg_masks = {
                'vertebrae': get_mask(['T6 vertebra','T7 vertebra','T8 vertebra',
                                       'T9 vertebra','T10 vertebra','T11 vertebra','T12 vertebra']),
                'heart':  get_mask(['heart']),
                'aorta':  get_mask(['aorta']),
            }
            n = os.path.basename(path)
            self.lbl_seg.setText(f'Seg: {n}\n  {sum(v.sum() for v in self.seg_masks.values()):,} voxels total')
            self._status('Segmentation chargée')
        except Exception as ex:
            self._error(f'Erreur seg : {ex}')

    def load_fluoro(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Charger Fluoroscopie', '', 'Images (*.png *.jpg *.tiff *.bmp)')
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self._error('Impossible de charger la fluoroscopie')
            return
        size = self.spin_size.value()
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
        self.fluoro_image = img.astype(np.float32) / 255.0
        self.canvas_fluoro.set_image(self.fluoro_image)
        self.lbl_fluoro.setText(f'Fluoro: {os.path.basename(path)}\n  {img.shape}')
        self.tabs.setCurrentIndex(0)
        self._update_register_btn()
        self._status('Fluoroscopie chargée — dessinez les vertèbres')

    def generate_drr(self):
        if self.ct_vol is None:
            self._error('Charger un CT d\'abord')
            return
        self.btn_gen_drr.setEnabled(False)
        self._status('Génération DRR en cours…')
        self.progress_bar.setValue(5)

        kwargs = {
            'ct_vol': self.ct_vol,
            'voxel_mm': self.voxel_mm,
            'ap_axis': self.ap_axis,
            'lao_deg': self.spin_lao.value(),
            'cran_deg': self.spin_cran.value(),
            'output_size': self.spin_size.value(),
            'masks': self.seg_masks if self.seg_masks else {},
        }
        self.worker = WorkerThread('drr', kwargs)
        self.worker.progress.connect(self._on_progress)
        self.worker.result.connect(self._on_drr_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_drr_done(self, res):
        self.drr_image = res['drr']
        self.proj_masks = res.get('masks', {})

        self.canvas_drr.set_image(self.drr_image)
        self.btn_gen_drr.setEnabled(True)
        self.tabs.setCurrentIndex(1)
        self._update_register_btn()
        self._status('DRR généré — annotez les vertèbres dans l\'onglet DRR')

    def run_registration(self):
        mask_fluoro = self.canvas_fluoro.get_mask()
        mask_drr    = self.canvas_drr.get_mask()

        # L'image mobile = projection CT des vertèbres (ou masque DRR annoté)
        # La référence = masque annoté sur la fluoro
        if mask_fluoro is None or mask_fluoro.sum() == 0:
            self._error('Dessinez d\'abord le masque sur la FLUOROSCOPIE')
            return
        if mask_drr is None or mask_drr.sum() == 0:
            self._error('Dessinez d\'abord le masque sur le DRR')
            return

        self.btn_register.setEnabled(False)
        self._status('Recalage en cours (Differential Evolution + Nelder-Mead)…')

        kwargs = {
            'mask_moving': mask_drr,      # DRR annoté = image mobile
            'mask_fixed':  mask_fluoro,   # Fluoro annotée = référence fixe
            'search_tx': self.spin_stx.value(),
            'search_ty': self.spin_sty.value(),
            'search_rot': self.spin_srot.value(),
        }
        self.worker = WorkerThread('register', kwargs)
        self.worker.progress.connect(self._on_progress)
        self.worker.result.connect(self._on_register_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_register_done(self, res):
        self.result = res
        self.btn_register.setEnabled(True)

        # Métriques
        iou  = res['iou']
        dice = res['dice']
        color = ACCENT2 if iou > 0.5 else (WARN if iou > 0.25 else ERR)
        self.lbl_iou.setText(f'{iou:.3f}')
        self.lbl_iou.setStyleSheet(f'font-size: 22px; font-weight: bold; color: {color};')
        self.lbl_dice.setText(f'{dice:.3f}')
        self.lbl_dice.setStyleSheet(f'font-size: 22px; font-weight: bold; color: {color};')
        self.lbl_tx.setText(f'tx = {res["tx"]:+.1f} px')
        self.lbl_ty.setText(f'ty = {res["ty"]:+.1f} px')
        self.lbl_rot.setText(f'rot = {res["angle"]:+.2f} °')

        # Construire l'image résultat
        self._build_result_image(res)
        self.tabs.setCurrentIndex(2)
        self._status(f'Recalage terminé — IoU={iou:.3f}  Dice={dice:.3f}')

    def _build_result_image(self, res):
        size = self.spin_size.value()

        # Base = fluoroscopie ou fond noir
        if self.fluoro_image is not None:
            base = self.fluoro_image.copy()
        else:
            base = np.zeros((size, size), np.float32)

        base_u8 = (base * 255).clip(0, 255).astype(np.uint8)
        result_rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2RGB)

        # Masque fluoro annoté = référence (vert)
        mask_fluoro = self.canvas_fluoro.get_mask()
        if mask_fluoro is not None:
            for c in measure.find_contours(mask_fluoro, 0.5):
                pts = np.array([[p[1], p[0]] for p in c], dtype=np.int32)
                cv2.polylines(result_rgb, [pts], True, (50, 220, 120), 2)

        # Masque DRR recalé (orange)
        mask_reg = res['mask_registered']
        for c in measure.find_contours(mask_reg, 0.5):
            pts = np.array([[p[1], p[0]] for p in c], dtype=np.int32)
            cv2.polylines(result_rgb, [pts], True, (255, 140, 50), 2)

        # Projections segmentation CT recalées (si dispo)
        tx, ty, angle = res['tx'], res['ty'], res['angle']
        cx, cy = res['center']

        colors = {
            'vertebrae': (255, 220, 50),
            'heart':     (230, 80, 80),
            'aorta':     (80, 200, 220),
        }
        for name, proj in self.proj_masks.items():
            proj_reg = apply_transform(proj, tx, ty, angle, (cx, cy))
            col = colors.get(name, (200, 200, 200))
            for c in measure.find_contours(proj_reg, 0.5):
                pts = np.array([[p[1], p[0]] for p in c], dtype=np.int32)
                cv2.polylines(result_rgb, [pts], True, col, 2)

        # Légende
        legend_items = [
            ('Masque fluoro (ref)', (50, 220, 120)),
            ('Masque DRR recalé',   (255, 140, 50)),
            ('Vertèbres CT',        (255, 220, 50)),
            ('Cœur CT',             (230, 80, 80)),
            ('Aorte CT',            (80, 200, 220)),
        ]
        for i, (label, col) in enumerate(legend_items):
            y = 20 + i * 22
            cv2.rectangle(result_rgb, (10, y-10), (26, y+4), col, -1)
            cv2.putText(result_rgb, label, (32, y+2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1)

        # IoU overlay
        iou_txt = f'IoU={res["iou"]:.3f}  Dice={res["dice"]:.3f}  tx={res["tx"]:+.0f}px  ty={res["ty"]:+.0f}px  rot={res["angle"]:+.1f}deg'
        cv2.putText(result_rgb, iou_txt, (8, size-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)

        self.canvas_result.set_image(result_rgb)

    def export_results(self):
        if self.result is None:
            self._error('Lancez d\'abord le recalage')
            return
        folder = QFileDialog.getExistingDirectory(self, 'Dossier export')
        if not folder: return

        # Masques
        mask_fluoro = self.canvas_fluoro.get_mask()
        mask_drr    = self.canvas_drr.get_mask()
        if mask_fluoro is not None:
            cv2.imwrite(os.path.join(folder, 'mask_fluoro.png'),
                        (mask_fluoro*255).astype(np.uint8))
        if mask_drr is not None:
            cv2.imwrite(os.path.join(folder, 'mask_drr.png'),
                        (mask_drr*255).astype(np.uint8))
        if self.drr_image is not None:
            cv2.imwrite(os.path.join(folder, 'drr.png'),
                        (self.drr_image*255).astype(np.uint8))

        # Résultats JSON
        out = {
            'tx_px': float(self.result['tx']),
            'ty_px': float(self.result['ty']),
            'angle_deg': float(self.result['angle']),
            'iou': float(self.result['iou']),
            'dice': float(self.result['dice']),
            'iou_init': float(self.result['iou_init']),
            'lao_deg': self.spin_lao.value(),
            'cran_deg': self.spin_cran.value(),
            'ap_axis': int(self.ap_axis),
        }
        with open(os.path.join(folder, 'registration_result.json'), 'w') as f:
            json.dump(out, f, indent=2)

        self._status(f'Résultats exportés → {folder}')
        QMessageBox.information(self, 'Export', f'Résultats sauvegardés dans :\n{folder}')

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers UI
    # ══════════════════════════════════════════════════════════════════════════

    def _set_tool(self, tool):
        for btn, t in [(self.btn_tool_rect,'rectangle'),
                       (self.btn_tool_poly,'polygon'),
                       (self.btn_tool_erase,'eraser')]:
            btn.setChecked(t == tool)
        active = self._active_canvas()
        active.set_tool(tool)
        self._status(f'Outil : {tool}')

    def _active_canvas(self) -> AnnotationCanvas:
        idx = self.combo_active.currentIndex()
        return self.canvas_fluoro if idx == 0 else self.canvas_drr

    def _on_active_canvas_change(self, idx):
        self.tabs.setCurrentIndex(idx)

    def _undo(self):
        self._active_canvas().undo()

    def _clear_mask(self):
        self._active_canvas().clear_mask()

    def _save_mask(self):
        canvas = self._active_canvas()
        mask = canvas.get_mask()
        if mask is None or mask.sum() == 0:
            self._error('Masque vide')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Sauvegarder masque', 'mask.png', 'PNG (*.png)')
        if path:
            cv2.imwrite(path, (mask*255).astype(np.uint8))
            self._status(f'Masque sauvegardé → {path}')

    def _on_brush_change(self, v):
        self.lbl_brush.setText(f'{v}px')
        self.canvas_fluoro.set_brush_radius(v)
        self.canvas_drr.set_brush_radius(v)

    def _on_mask_updated(self):
        self._update_register_btn()

    def _update_register_btn(self):
        mf = self.canvas_fluoro.get_mask()
        md = self.canvas_drr.get_mask()
        ok = (mf is not None and mf.sum() > 0 and
              md is not None and md.sum() > 0)
        self.btn_register.setEnabled(ok)

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.lbl_progress.setText(msg)
        self._status(msg)

    def _on_error(self, msg):
        self._error(msg)
        self.btn_gen_drr.setEnabled(True)
        self.btn_register.setEnabled(True)

    def _status(self, msg):
        self.status_bar.showMessage(msg)

    def _error(self, msg):
        self.status_bar.showMessage(f'⚠ {msg}')
        QMessageBox.warning(self, 'Erreur', msg)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    app = QApplication(sys.argv)
    app.setApplicationName('2D/3D Registration')
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
