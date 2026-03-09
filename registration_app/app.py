"""
app.py — 2D/3D Registration — EP Lab
Outils : crayon lissé (Catmull-Rom) + polygone + rectangle + gomme
Structures : vertèbres, cœur, aorte  — recalage sur n'importe laquelle
"""

import sys, os, json
import numpy as np
import cv2
import nibabel as nib
import pandas as pd
from skimage import measure

try:
    import pyvista as pv
except Exception:
    pv = None

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QGroupBox,
    QSpinBox, QDoubleSpinBox, QProgressBar, QTabWidget, QSizePolicy,
    QStatusBar, QFrame, QMessageBox, QComboBox, QGridLayout, QCheckBox,
    QDialog, QScrollArea, QToolButton, QButtonGroup,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QCursor, QFont

sys.path.insert(0, os.path.dirname(__file__))
from core.drr_generator import load_ct, generate_drr, project_mask_3d
from core.registration import (register, register_elastic,
                               apply_transform, apply_full_transform,
                               iou_score, dice_score)


# ══════════════════════════════════════════════════════════════════════════════
# Thème
# ══════════════════════════════════════════════════════════════════════════════

DARK_BG  = '#0c0e14'
PANEL_BG = '#12151f'
CARD_BG  = '#1a1d2a'
BORDER   = '#1e2235'
BORDER2  = '#2d3250'
ACCENT   = '#4f9cf9'
ACCENT2  = '#2ecc7a'
TEXT     = '#cdd5e8'
TEXT_DIM = '#4d5570'
TEXT_MID = '#8892b0'
WARN     = '#f0b040'
ERR      = '#e05060'

STRUCT = {
    'vertebrae': {'rgb': (80, 220, 130),  'hex': '#50dc82', 'label': 'Vertebres'},
    'heart':     {'rgb': (240,  80,  90), 'hex': '#f0505a', 'label': 'Coeur'},
    'aorta':     {'rgb': (80, 190, 240),  'hex': '#50bef0', 'label': 'Aorte'},
}

SIDEBAR_W = 310

STYLE = f"""
QMainWindow,QWidget{{background:{DARK_BG};color:{TEXT};font-family:'Segoe UI',sans-serif;font-size:12px;}}
QScrollArea{{background:transparent;border:none;}}
QScrollBar:vertical{{background:{DARK_BG};width:5px;border:none;border-radius:2px;}}
QScrollBar::handle:vertical{{background:{BORDER2};border-radius:2px;min-height:20px;}}
QScrollBar::handle:vertical:hover{{background:{ACCENT};}}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{{height:0px;}}
QPushButton{{
  background:{CARD_BG};border:1px solid {BORDER2};border-radius:6px;
  padding:6px 12px;color:{TEXT};font-size:12px;
  min-height:28px;
}}
QPushButton:hover{{border-color:{ACCENT};color:{ACCENT};background:#1e2235;}}
QPushButton:pressed,QPushButton:checked{{background:{ACCENT};border-color:{ACCENT};color:#fff;}}
QPushButton:disabled{{color:{TEXT_DIM};border-color:{BORDER};background:{DARK_BG};}}
QPushButton#primary{{background:{ACCENT};border-color:{ACCENT};color:#fff;font-weight:600;}}
QPushButton#primary:hover{{background:#3a8ae8;border-color:#3a8ae8;}}
QPushButton#primary:disabled{{background:#1f2d40;border-color:{BORDER2};color:{TEXT_DIM};}}
QPushButton#success{{background:#132b1e;border-color:{ACCENT2};color:{ACCENT2};font-weight:600;}}
QPushButton#success:hover{{background:#1a3d28;}}
QPushButton#success:disabled{{color:{TEXT_DIM};border-color:{BORDER};background:{DARK_BG};}}
QPushButton#warn{{background:#2e2208;border-color:{WARN};color:{WARN};}}
QPushButton#danger{{background:#2a0c10;border-color:{ERR};color:{ERR};}}
QPushButton#tool{{
  background:{CARD_BG};border:1px solid {BORDER2};border-radius:6px;
  padding:5px 14px;color:{TEXT_MID};font-size:11px;min-height:28px;
}}
QPushButton#tool:checked{{background:{ACCENT};border-color:{ACCENT};color:#fff;}}
QPushButton#tool:hover:!checked{{border-color:{ACCENT};color:{ACCENT};}}
QSlider::groove:horizontal{{height:3px;background:{BORDER2};border-radius:2px;margin:0 2px;}}
QSlider::handle:horizontal{{width:13px;height:13px;background:{ACCENT};border-radius:7px;margin:-5px 0;}}
QSlider::sub-page:horizontal{{background:{ACCENT};border-radius:2px;}}
QDoubleSpinBox,QSpinBox,QComboBox{{
  background:{CARD_BG};border:1px solid {BORDER2};border-radius:5px;
  padding:4px 8px;color:{TEXT};selection-background-color:{ACCENT};
}}
QComboBox::drop-down{{border:none;}}
QComboBox QAbstractItemView{{background:{CARD_BG};border:1px solid {BORDER2};color:{TEXT};selection-background-color:{ACCENT};}}
QProgressBar{{background:{CARD_BG};border:1px solid {BORDER};border-radius:4px;height:6px;color:transparent;text-align:center;}}
QProgressBar::chunk{{background:{ACCENT};border-radius:4px;}}
QTabWidget::pane{{border:1px solid {BORDER};background:{PANEL_BG};border-radius:0 6px 6px 6px;}}
QTabBar::tab{{background:{DARK_BG};border:1px solid {BORDER};border-bottom:none;padding:7px 20px;color:{TEXT_DIM};border-radius:5px 5px 0 0;margin-right:2px;}}
QTabBar::tab:selected{{background:{PANEL_BG};color:{ACCENT};border-bottom:2px solid {ACCENT};}}
QTabBar::tab:hover:!selected{{color:{TEXT_MID};}}
QLabel{{color:{TEXT};}}
QLabel#dim{{color:{TEXT_DIM};font-size:11px;}}
QLabel#mid{{color:{TEXT_MID};font-size:11px;}}
QLabel#metric{{font-size:22px;font-weight:700;color:{ACCENT2};}}
QLabel#section-title{{color:{ACCENT};font-size:11px;font-weight:600;letter-spacing:1px;}}
QStatusBar{{background:{PANEL_BG};border-top:1px solid {BORDER};color:{TEXT_DIM};font-size:11px;padding:2px 8px;}}
QFrame#sep{{background:{BORDER};max-height:1px;border:none;}}
QCheckBox{{color:{TEXT};spacing:6px;}}
QCheckBox::indicator{{width:14px;height:14px;border:1px solid {BORDER2};border-radius:3px;background:{CARD_BG};}}
QCheckBox::indicator:checked{{background:{ACCENT};border-color:{ACCENT};}}
QGroupBox{{border:none;margin:0;padding:0;}}
QToolButton{{background:transparent;border:none;color:{TEXT_DIM};padding:2px;}}
QToolButton:hover{{color:{ACCENT};}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# Catmull-Rom spline
# ══════════════════════════════════════════════════════════════════════════════

def catmull_rom_chain(points: list, n_interp: int = 8) -> list:
    """Lisse une liste de points avec une spline Catmull-Rom."""
    if len(points) < 2:
        return list(points)
    pts = np.array(points, dtype=np.float64)
    padded = np.vstack([pts[0:1], pts, pts[-1:]])
    result = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i-1], padded[i], padded[i+1], padded[i+2]
        for t in np.linspace(0, 1, n_interp, endpoint=False):
            t2, t3 = t*t, t*t*t
            x = 0.5*((2*p1[0])+(-p0[0]+p2[0])*t+(2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2+(-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
            y = 0.5*((2*p1[1])+(-p0[1]+p2[1])*t+(2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2+(-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
            result.append((int(round(x)), int(round(y))))
    result.append((int(pts[-1][0]), int(pts[-1][1])))
    return result


def _project_stl_to_pyvista(
    points_xyz: np.ndarray,
    ct_affine: np.ndarray,
    voxel_mm: np.ndarray,
    ap_axis: int,
    lao_deg: float,
    cran_deg: float,
    output_size: int,
    reg_result: dict,
) -> np.ndarray:
    """
    Projette le maillage STL (coordonnées mm monde, RAS/LPS) en espace PyVista
    aligné sur la fluoroscopie.

    Pipeline identique à generate_drr / project_mask_3d :
      1. STL world mm → voxel ijk via inv(ct_affine)   ← corrige tous les flips
      2. Rotations C-arm
      3. col = vox_keep0 * mm0,  row = -vox_keep1 * mm1  (équiv. flipud(proj.T))
      4. Scale + transform DRR→fluoro
    """
    # 1. World mm → voxel ijk (même espace que le DRR)
    inv_aff = np.linalg.inv(ct_affine)
    pts = points_xyz.astype(np.float64)
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts = (inv_aff @ pts_h.T).T[:, :3]          # (N,3) voxel ijk
    pts -= pts.mean(axis=0, keepdims=True)

    # 2. Rotations C-arm (même ordre/axes que scipy.ndimage.rotate dans generate_drr)
    if abs(lao_deg) > 1e-6:
        a = np.deg2rad(lao_deg); c, s = np.cos(a), np.sin(a)
        pts = pts @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64).T
    if abs(cran_deg) > 1e-6:
        a = np.deg2rad(cran_deg); c, s = np.cos(a), np.sin(a)
        pts = pts @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64).T

    # 3. Projection 2D identique à np.flipud(integral.T)
    #    integral = vol.sum(axis=ap_axis)  shape (n_keep0, n_keep1)
    #    integral.T    shape (n_keep1, n_keep0)  → rows=keep1, cols=keep0
    #    flipud        images row 0 = max keep1 index  →  row = -keep1
    keep = [i for i in range(3) if i != ap_axis]
    col = pts[:, keep[0]] * float(voxel_mm[keep[0]])   # mm, axe horizontal
    row = -pts[:, keep[1]] * float(voxel_mm[keep[1]])  # mm, axe vertical inversé

    xy = np.column_stack([col, row])
    mn, mx = xy.min(axis=0), xy.max(axis=0)
    span = np.maximum(mx - mn, 1e-6)
    s2d = (output_size - 1) * 0.92 / float(max(span[0], span[1]))
    xy = (xy - mn) * s2d + output_size * 0.04

    # 4. Transform DRR → Fluoroscopie
    a = np.deg2rad(reg_result['angle']); c, s = np.cos(a), np.sin(a)
    sc = reg_result.get('scale', 1.0)
    cx, cy = float(reg_result['center'][0]), float(reg_result['center'][1])
    x, y = xy[:, 0] - cx, xy[:, 1] - cy
    xf = sc * (c * x - s * y) + cx + reg_result['tx']
    yf = sc * (s * x + c * y) + cy + reg_result['ty']

    # 5. Profondeur AP : même échelle que X,Y, centrée, devant le plan fluoro
    depth_mm = pts[:, ap_axis] * float(voxel_mm[ap_axis])
    z = depth_mm * s2d * sc
    z = (z - z.mean()) + output_size * 0.30

    # 6. Flip X uniquement : corrige le miroir gauche/droite du STL
    #    (le recalage 2D est correct, seul l'axe X est inversé dans l'espace PyVista)
    xf = (output_size - 1) - xf

    return np.column_stack([xf, yf, z]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Zone de depot (Drag & Drop)
# ══════════════════════════════════════════════════════════════════════════════

class DropZone(QFrame):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(70)
        self._lbl = QLabel('Deposer vos fichiers ici\nCT / Segmentation / Images / STL')
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setWordWrap(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 12, 8, 12)
        lay.addWidget(self._lbl)
        self._apply_style(False)

    def _apply_style(self, hover):
        bc = ACCENT if hover else BORDER2
        bg = '#1a1d2e' if hover else DARK_BG
        tc = ACCENT if hover else TEXT_DIM
        self.setStyleSheet(f'border:2px dashed {bc};border-radius:8px;background:{bg};')
        self._lbl.setStyleSheet(f'color:{tc};font-size:11px;border:none;background:transparent;')

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._apply_style(True)

    def dragLeaveEvent(self, e):
        self._apply_style(False)

    def dropEvent(self, e):
        self._apply_style(False)
        files = [url.toLocalFile() for url in e.mimeData().urls() if url.toLocalFile()]
        if files:
            self.files_dropped.emit(files)


# ══════════════════════════════════════════════════════════════════════════════
# Section pliable
# ══════════════════════════════════════════════════════════════════════════════

class CollapsibleSection(QWidget):
    def __init__(self, title, parent=None, starts_open=True):
        super().__init__(parent)
        self._open = starts_open
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Header row
        hdr = QWidget()
        hdr.setCursor(Qt.PointingHandCursor)
        hdr.setStyleSheet(f'''
            QWidget{{background:{CARD_BG};border:1px solid {BORDER2};
                    border-radius:6px;padding:0px;}}
        ''')
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(10, 7, 8, 7)
        self._arrow = QLabel(self._arrow_char())
        self._arrow.setStyleSheet(f'color:{ACCENT};font-size:9px;border:none;background:transparent;min-width:10px;')
        self._title_lbl = QLabel(title)
        self._title_lbl.setObjectName('section-title')
        self._title_lbl.setStyleSheet(f'border:none;background:transparent;letter-spacing:1px;font-size:11px;font-weight:600;color:{ACCENT};')
        hl.addWidget(self._arrow)
        hl.addWidget(self._title_lbl, 1)
        root.addWidget(hdr)
        hdr.mousePressEvent = lambda e: self.toggle()

        # Content
        self._content = QWidget()
        self._content.setStyleSheet(f'background:transparent;')
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(0, 6, 0, 2)
        self._content_layout.setSpacing(4)
        root.addWidget(self._content)
        self._update_visibility()

    def _arrow_char(self):
        return 'v' if self._open else '>'

    def toggle(self):
        self._open = not self._open
        self._arrow.setText(self._arrow_char())
        self._update_visibility()

    def _update_visibility(self):
        self._content.setVisible(self._open)

    def layout(self):
        return self._content_layout

    def addWidget(self, w):
        self._content_layout.addWidget(w)

    def addLayout(self, l):
        self._content_layout.addLayout(l)


# ══════════════════════════════════════════════════════════════════════════════
# Vignette image
# ══════════════════════════════════════════════════════════════════════════════

THUMB_SIZE = 64
ROLE_COLORS = {None: BORDER2, 'fixed': '#4f9cf9', 'mobile': '#2ecc7a'}
ROLE_LABELS = {None: '', 'fixed': 'FIXE', 'mobile': 'MOBILE'}

class ImageCard(QWidget):
    role_changed = pyqtSignal(int, str)   # index, role ('fixed'/'mobile'/'')

    def __init__(self, index, name, array, parent=None):
        super().__init__(parent)
        self.index = index
        self.name = name
        self.array = array
        self._role = None
        self.setFixedHeight(THUMB_SIZE + 18)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(name)
        self._build()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        # Thumbnail
        self._thumb_lbl = QLabel()
        self._thumb_lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self._thumb_lbl.setAlignment(Qt.AlignCenter)
        self._thumb_lbl.setStyleSheet(
            f'border:2px solid {BORDER2};border-radius:5px;background:{DARK_BG};'
        )
        self._update_thumb()
        lay.addWidget(self._thumb_lbl)

        # Info + role badge
        right = QWidget()
        right.setStyleSheet('background:transparent;')
        vr = QVBoxLayout(right)
        vr.setContentsMargins(0, 2, 0, 2)
        vr.setSpacing(4)

        short = self.name if len(self.name) <= 20 else self.name[:17] + '...'
        name_lbl = QLabel(short)
        name_lbl.setObjectName('mid')
        name_lbl.setWordWrap(False)
        vr.addWidget(name_lbl)

        # Role buttons
        br = QHBoxLayout()
        br.setSpacing(4)
        br.setContentsMargins(0, 0, 0, 0)
        self._btn_fixed = QPushButton('Fixe')
        self._btn_fixed.setObjectName('tool')
        self._btn_fixed.setCheckable(True)
        self._btn_fixed.setFixedHeight(22)
        self._btn_fixed.setStyleSheet(
            f'QPushButton{{font-size:10px;padding:2px 8px;border-radius:4px;'
            f'background:{CARD_BG};border:1px solid {BORDER2};color:{TEXT_DIM};min-height:20px;}}'
            f'QPushButton:checked{{background:{ACCENT};border-color:{ACCENT};color:#fff;}}'
            f'QPushButton:hover:!checked{{border-color:{ACCENT};color:{ACCENT};}}'
        )
        self._btn_mobile = QPushButton('Mobile')
        self._btn_mobile.setObjectName('tool')
        self._btn_mobile.setCheckable(True)
        self._btn_mobile.setFixedHeight(22)
        self._btn_mobile.setStyleSheet(
            f'QPushButton{{font-size:10px;padding:2px 8px;border-radius:4px;'
            f'background:{CARD_BG};border:1px solid {BORDER2};color:{TEXT_DIM};min-height:20px;}}'
            f'QPushButton:checked{{background:{ACCENT2};border-color:{ACCENT2};color:#fff;}}'
            f'QPushButton:hover:!checked{{border-color:{ACCENT2};color:{ACCENT2};}}'
        )
        self._btn_fixed.clicked.connect(self._on_fixed)
        self._btn_mobile.clicked.connect(self._on_mobile)
        br.addWidget(self._btn_fixed)
        br.addWidget(self._btn_mobile)
        br.addStretch()
        vr.addLayout(br)
        vr.addStretch()
        lay.addWidget(right, 1)

    def _update_thumb(self):
        arr = self.array
        u8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        thumb = cv2.resize(u8, (THUMB_SIZE, THUMB_SIZE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)
        h, w = rgb.shape[:2]
        qi = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pm = QPixmap.fromImage(qi).copy()
        self._thumb_lbl.setPixmap(pm)

    def _on_fixed(self):
        self._btn_mobile.setChecked(False)
        if self._btn_fixed.isChecked():
            self._set_role('fixed')
        else:
            self._set_role(None)

    def _on_mobile(self):
        self._btn_fixed.setChecked(False)
        if self._btn_mobile.isChecked():
            self._set_role('mobile')
        else:
            self._set_role(None)

    def _set_role(self, role):
        self._role = role
        color = ROLE_COLORS.get(role, BORDER2)
        self._thumb_lbl.setStyleSheet(
            f'border:2px solid {color};border-radius:5px;background:{DARK_BG};'
        )
        self.role_changed.emit(self.index, role or '')

    def set_role_external(self, role):
        self._role = role
        self._btn_fixed.blockSignals(True)
        self._btn_mobile.blockSignals(True)
        self._btn_fixed.setChecked(role == 'fixed')
        self._btn_mobile.setChecked(role == 'mobile')
        self._btn_fixed.blockSignals(False)
        self._btn_mobile.blockSignals(False)
        color = ROLE_COLORS.get(role, BORDER2)
        self._thumb_lbl.setStyleSheet(
            f'border:2px solid {color};border-radius:5px;background:{DARK_BG};'
        )


# ══════════════════════════════════════════════════════════════════════════════
# Canvas d'annotation
# ══════════════════════════════════════════════════════════════════════════════

class AnnotationCanvas(QLabel):
    mask_updated = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(512, 512)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)

        self._img_np  = None
        self._size    = 512
        self._masks   = {k: None for k in STRUCT}
        self._active  = 'heart'
        self._history = {k: [] for k in STRUCT}

        self._tool    = 'pencil'
        self._pen_r   = 2
        self._brush_r = 20
        self._drawing = False

        self._raw_pts    = []
        self._poly_pts   = []
        self._rect_start = None
        self._rect_cur   = None
        self._cursor_pos = None
        self._alpha      = 0.40

    # ── API ───────────────────────────────────────────────────────────────────

    def set_image(self, img: np.ndarray):
        s = img.shape[0]; self._size = s
        u8 = (np.clip(img,0,1)*255).astype(np.uint8) if img.dtype != np.uint8 else img.copy()
        self._img_np = u8
        for k in STRUCT:
            self._masks[k] = np.zeros((s,s), np.float32)
            self._history[k] = []
        self._raw_pts=[]; self._poly_pts=[]
        self._refresh()

    def set_tool(self, t):
        self._tool=t; self._raw_pts=[]; self._poly_pts=[]
        self._rect_start=self._rect_cur=None; self._drawing=False
        self.setCursor(QCursor(Qt.BlankCursor if t=='eraser' else Qt.CrossCursor))
        self._refresh()

    def set_active(self, s): self._active=s; self._raw_pts=[]; self._poly_pts=[]
    def set_pen_radius(self, r): self._pen_r=max(1,r)
    def set_brush_radius(self, r): self._brush_r=max(3,r)

    def get_mask(self, struct=None):
        if struct:
            m=self._masks.get(struct); return m.copy() if m is not None else None
        out=None
        for m in self._masks.values():
            if m is None or m.sum()==0: continue
            out=m.copy() if out is None else np.clip(out+m,0,1)
        return out

    def undo(self):
        h=self._history[self._active]
        if h: self._masks[self._active]=h.pop(); self._refresh(); self.mask_updated.emit()

    def clear_struct(self):
        self._push_history(); self._masks[self._active][:]=0
        self._raw_pts=[]; self._poly_pts=[]; self._refresh(); self.mask_updated.emit()

    def clear_all(self):
        for k in STRUCT:
            if self._masks[k] is not None: self._masks[k][:]=0
            self._history[k]=[]
        self._raw_pts=[]; self._poly_pts=[]; self._refresh(); self.mask_updated.emit()

    # ── Historique ────────────────────────────────────────────────────────────

    def _push_history(self):
        m=self._masks[self._active]
        if m is not None:
            h=self._history[self._active]; h.append(m.copy())
            if len(h)>30: h.pop(0)

    def _m(self): return self._masks[self._active]

    # ── Dessin ────────────────────────────────────────────────────────────────

    def _commit_pencil(self):
        if len(self._raw_pts) < 2: self._raw_pts=[]; return
        self._push_history()
        smooth = catmull_rom_chain(self._raw_pts, n_interp=8)
        pts_arr = np.array(smooth, dtype=np.int32)
        # Toujours peindre le trait
        for i in range(len(pts_arr)-1):
            cv2.line(self._m(), tuple(pts_arr[i]), tuple(pts_arr[i+1]), 1.0, self._pen_r*2)
        # Fermeture automatique si dernier ≈ premier (< 20px)
        p0 = np.array(self._raw_pts[0], float)
        p1 = np.array(self._raw_pts[-1], float)
        if np.linalg.norm(p1-p0) < 20:
            cv2.fillPoly(self._m(), [pts_arr], 1.0)
        self._raw_pts=[]

    def _draw_poly(self, pts):
        self._push_history(); cv2.fillPoly(self._m(), [np.array(pts,np.int32)], 1.0)

    def _draw_rect(self, x0, y0, x1, y1):
        self._push_history()
        x0,x1=sorted([x0,x1]); y0,y1=sorted([y0,y1])
        x0=max(0,x0); y0=max(0,y0); x1=min(self._size-1,x1); y1=min(self._size-1,y1)
        if x1>x0 and y1>y0: self._m()[y0:y1,x0:x1]=1.0

    def _erase(self, x, y):
        for m in self._masks.values():
            if m is not None: cv2.circle(m,(x,y),self._brush_r,0.0,-1)

    # ── Rendu ─────────────────────────────────────────────────────────────────

    def _refresh(self):
        if self._img_np is None: return
        img=self._img_np
        base=(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) if img.ndim==2 else img.copy()).astype(np.float32)

        # Masques persistants (remplissage + contour)
        for struct, info in STRUCT.items():
            m=self._masks.get(struct)
            if m is None or m.sum()==0: continue
            r,g,b=info['rgb']
            ov=base.copy(); ov[m>0]=[r,g,b]
            base=cv2.addWeighted(base,1-self._alpha,ov,self._alpha,0)
            m8=(m*255).astype(np.uint8)
            cnts,_=cv2.findContours(m8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            b8=np.clip(base,0,255).astype(np.uint8)
            cv2.drawContours(b8,cnts,-1,(r,g,b),2)
            base=b8.astype(np.float32)

        base=np.clip(base,0,255).astype(np.uint8)
        r,g,b=STRUCT[self._active]['rgb']

        # ── Preview crayon : tracé lissé en temps réel ────────────────────────
        if self._tool=='pencil' and len(self._raw_pts)>=2:
            smooth=catmull_rom_chain(self._raw_pts,n_interp=6)
            if len(smooth)>=2:
                pts_a=np.array(smooth,np.int32)
                for i in range(len(pts_a)-1):
                    cv2.line(base,tuple(pts_a[i]),tuple(pts_a[i+1]),(r,g,b),max(1,self._pen_r*2))
            # Cercle de départ — vert si tracé prêt à se fermer
            if len(self._raw_pts)>5:
                d=np.linalg.norm(np.array(self._raw_pts[0])-np.array(self._raw_pts[-1]))
                col=(60,255,80) if d<20 else (200,200,200)
                cv2.circle(base,tuple(self._raw_pts[0]),8,col,2)
                if d<20:
                    cv2.putText(base,'Fermer ici',(self._raw_pts[0][0]+12,self._raw_pts[0][1]+4),
                                cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1)

        # Preview polygone
        if self._tool=='polygon' and len(self._poly_pts)>=2:
            pts_a=np.array(self._poly_pts,np.int32)
            cv2.polylines(base,[pts_a],False,(r,g,b),2)
            for p in self._poly_pts: cv2.circle(base,p,5,(r,g,b),-1)
            if self._cursor_pos:
                cv2.line(base,self._poly_pts[-1],self._cursor_pos,(r,g,b),1)

        # Preview rectangle
        if self._tool=='rectangle' and self._rect_start and self._rect_cur:
            x0,y0=self._rect_start; x1,y1=self._rect_cur
            cv2.rectangle(base,(min(x0,x1),min(y0,y1)),(max(x0,x1),max(y0,y1)),(r,g,b),2)

        # Curseur gomme
        if self._tool=='eraser' and self._cursor_pos:
            cv2.circle(base,self._cursor_pos,self._brush_r,(200,80,80),2)
            cv2.circle(base,self._cursor_pos,2,(200,80,80),-1)

        # Barre de statut en bas
        labels={'pencil':'CRAYON','polygon':'POLYGONE','rectangle':'RECT','eraser':'GOMME'}
        txt=f"{labels.get(self._tool,'')}  |  {STRUCT[self._active]['label']}"
        cv2.putText(base,txt,(8,self._size-10),cv2.FONT_HERSHEY_SIMPLEX,0.38,(r,g,b),1)

        h,w=base.shape[:2]
        qi=QImage(base.data,w,h,3*w,QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qi).scaled(self.width(),self.height(),
                       Qt.KeepAspectRatio,Qt.SmoothTransformation))

    # ── Coordonnées ───────────────────────────────────────────────────────────

    def _w2i(self,qx,qy):
        if self._img_np is None: return 0,0
        s=self._size; ww,wh=self.width(),self.height()
        sc=min(ww/s,wh/s); ox=(ww-s*sc)/2; oy=(wh-s*sc)/2
        return max(0,min(s-1,int((qx-ox)/sc))),max(0,min(s-1,int((qy-oy)/sc)))

    # ── Événements ────────────────────────────────────────────────────────────

    def mousePressEvent(self,e):
        if self._img_np is None: return
        ix,iy=self._w2i(e.x(),e.y())
        if self._tool=='pencil':
            self._drawing=True; self._raw_pts=[(ix,iy)]; self._refresh()
        elif self._tool=='polygon':
            self._poly_pts.append((ix,iy)); self._refresh()
        elif self._tool=='rectangle':
            self._drawing=True; self._rect_start=self._rect_cur=(ix,iy)
        elif self._tool=='eraser':
            self._drawing=True; self._push_history(); self._erase(ix,iy); self._refresh()

    def mouseMoveEvent(self,e):
        if self._img_np is None: return
        ix,iy=self._w2i(e.x(),e.y()); self._cursor_pos=(ix,iy)
        if self._tool=='pencil' and self._drawing:
            if not self._raw_pts or abs(ix-self._raw_pts[-1][0])>1 or abs(iy-self._raw_pts[-1][1])>1:
                self._raw_pts.append((ix,iy))
            self._refresh()
        elif self._tool=='rectangle' and self._drawing:
            self._rect_cur=(ix,iy); self._refresh()
        elif self._tool=='polygon': self._refresh()
        elif self._tool=='eraser':
            if self._drawing: self._erase(ix,iy)
            self._refresh()

    def mouseReleaseEvent(self,e):
        if self._img_np is None: return
        ix,iy=self._w2i(e.x(),e.y())
        if self._tool=='pencil' and self._drawing:
            self._drawing=False
            if (ix,iy) not in self._raw_pts: self._raw_pts.append((ix,iy))
            self._commit_pencil(); self._refresh(); self.mask_updated.emit()
        elif self._tool=='rectangle' and self._drawing:
            self._drawing=False; self._draw_rect(*self._rect_start,ix,iy)
            self._rect_start=self._rect_cur=None; self._refresh(); self.mask_updated.emit()
        elif self._tool=='eraser' and self._drawing:
            self._drawing=False; self.mask_updated.emit()

    def mouseDoubleClickEvent(self,e):
        if self._tool=='polygon' and len(self._poly_pts)>=3:
            self._draw_poly(self._poly_pts); self._poly_pts=[]
            self._refresh(); self.mask_updated.emit()

    def resizeEvent(self,e): self._refresh()


# ══════════════════════════════════════════════════════════════════════════════
# Panneau de visualisation du recalage
# ══════════════════════════════════════════════════════════════════════════════

class ResultPanel(QWidget):
    """
    Visualisation multi-modes de la superposition Fluoroscopie / DRR recalé.
    Modes : Fusion · Damier · Différence · Avant/Après · Cyan/Magenta
    """
    MODES = [
        ('Fusion',          'Opacité DRR',   50),
        ('Damier',          'Taille tuile',  30),
        ('Différence',      'Contraste ×',   30),
        ('Avant / Après',   'Position (%)',  50),
        ('Cyan / Magenta',  '',               0),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._img_a    = None   # fluoroscopie float32 [0,1]
        self._img_b    = None   # DRR recalé   float32 [0,1]
        self._contours = []     # [(pts_ndarray (N,2) int, (r,g,b)), …]
        self._mode     = 0
        self._param    = 50
        self._show_cnt = True
        self._build_ui()

    # ── Construction ──────────────────────────────────────────────────────────
    def _build_ui(self):
        v = QVBoxLayout(self); v.setContentsMargins(6, 6, 6, 6); v.setSpacing(5)

        ctrl = QHBoxLayout(); ctrl.setSpacing(8)

        self._cmb = QComboBox()
        for name, _, _ in self.MODES: self._cmb.addItem(name)
        self._cmb.setFixedWidth(190)
        self._cmb.currentIndexChanged.connect(self._on_mode)
        ctrl.addWidget(self._cmb)

        self._lbl_p = QLabel(self.MODES[0][1])
        self._lbl_p.setObjectName('dim'); self._lbl_p.setFixedWidth(90)
        ctrl.addWidget(self._lbl_p)

        self._sl = QSlider(Qt.Horizontal)
        self._sl.setRange(0, 100); self._sl.setValue(50); self._sl.setFixedWidth(170)
        self._sl.valueChanged.connect(self._on_param)
        ctrl.addWidget(self._sl)

        self._lbl_v = QLabel('50 %')
        self._lbl_v.setObjectName('dim'); self._lbl_v.setFixedWidth(48)
        ctrl.addWidget(self._lbl_v)

        sep = QFrame(); sep.setFrameShape(QFrame.VLine); sep.setObjectName('sep')
        ctrl.addWidget(sep)

        self._chk_cnt = QCheckBox('Contours'); self._chk_cnt.setChecked(True)
        self._chk_cnt.toggled.connect(self._on_cnt)
        ctrl.addWidget(self._chk_cnt)

        ctrl.addStretch()

        self._lbl_hint = QLabel('')
        self._lbl_hint.setObjectName('dim')
        self._lbl_hint.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ctrl.addWidget(self._lbl_hint)

        v.addLayout(ctrl)

        self._lbl_img = QLabel()
        self._lbl_img.setAlignment(Qt.AlignCenter)
        self._lbl_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._lbl_img.setStyleSheet(f'background:{DARK_BG};border-radius:4px;')
        v.addWidget(self._lbl_img, 1)

    # ── API ───────────────────────────────────────────────────────────────────
    def set_data(self, img_a: np.ndarray, img_b: np.ndarray, contours: list):
        """
        img_a      : fluoroscopie float32 [0,1] — référence
        img_b      : DRR recalé   float32 [0,1] — image mobile recalée
        contours   : liste de ( pts ndarray (N,2) int, (r,g,b) )
        """
        self._img_a    = np.clip(img_a, 0, 1).astype(np.float32)
        self._img_b    = np.clip(img_b, 0, 1).astype(np.float32)
        self._contours = contours
        self._render()

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_mode(self, idx):
        self._mode = idx
        _, plabel, pval = self.MODES[idx]
        self._lbl_p.setText(plabel)
        has_slider = bool(plabel)
        self._sl.setVisible(has_slider); self._lbl_v.setVisible(has_slider)
        if has_slider: self._sl.setValue(pval)
        hints = {
            0: 'Fluoro ←→ image mobile recalée via opacité',
            1: 'Tuiles alternées Fluoro / image mobile',
            2: 'Carte |Fluoro − mobile| (chaud = grande erreur)',
            3: 'Glisser le curseur pour balayer Fluoro vs mobile',
            4: 'Rose=Fluoro · Vert=Mobile · Blanc=overlap',
        }
        self._lbl_hint.setText(hints.get(idx, ''))
        self._render()

    def _on_param(self, v):
        self._param = v
        if   self._mode == 1: self._lbl_v.setText(f'{4+v//2} px')
        elif self._mode == 2: self._lbl_v.setText(f'{1+v/20:.1f} ×')
        else:                 self._lbl_v.setText(f'{v} %')
        self._render()

    def _on_cnt(self, checked): self._show_cnt = checked; self._render()

    def resizeEvent(self, e): super().resizeEvent(e); self._render()

    # ── Rendu ─────────────────────────────────────────────────────────────────
    def _render(self):
        if self._img_a is None or self._img_b is None:
            self._lbl_img.clear(); return

        a = self._img_a; b = self._img_b
        h, w = a.shape[:2]; p = self._param

        # ── Mode 0 : Fusion ───────────────────────────────────────────────────
        if self._mode == 0:
            alpha = p / 100.0
            blended = (1 - alpha) * a + alpha * b
            rgb = cv2.cvtColor((blended * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

        # ── Mode 1 : Damier ───────────────────────────────────────────────────
        elif self._mode == 1:
            tile = max(4, 4 + p // 2)   # 4 .. 54 px
            rows = (np.arange(h) // tile) % 2
            cols = (np.arange(w) // tile) % 2
            mask = (rows[:, None] ^ cols[None, :]).astype(bool)
            a8 = (a * 255).astype(np.uint8); b8 = (b * 255).astype(np.uint8)
            ra = cv2.cvtColor(a8, cv2.COLOR_GRAY2RGB)
            rb = cv2.cvtColor(b8, cv2.COLOR_GRAY2RGB)
            rgb = np.where(mask[:, :, None], ra, rb).astype(np.uint8)

        # ── Mode 2 : Différence ───────────────────────────────────────────────
        elif self._mode == 2:
            gain = 1.0 + p / 20.0          # 1 .. 6 ×
            diff = np.clip(np.abs(a - b) * gain, 0, 1)
            bgr  = cv2.applyColorMap((diff * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # ── Mode 3 : Avant / Après ────────────────────────────────────────────
        elif self._mode == 3:
            split = max(1, int(w * p / 100.0))
            a8 = (a * 255).astype(np.uint8); b8 = (b * 255).astype(np.uint8)
            ra = cv2.cvtColor(a8, cv2.COLOR_GRAY2RGB)
            rb = cv2.cvtColor(b8, cv2.COLOR_GRAY2RGB)
            rgb = ra.copy(); rgb[:, split:] = rb[:, split:]
            cv2.line(rgb, (split, 0), (split, h - 1), (255, 255, 255), 2)
            cv2.putText(rgb, 'FLUORO',   (max(4, split-80), 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 255), 1)
            cv2.putText(rgb, 'DRR reg.', (min(w - 90, split + 6), 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 255, 190), 1)

        # ── Mode 4 : Cyan / Magenta ───────────────────────────────────────────
        else:
            a8 = (a * 255).astype(np.uint8)
            b8 = (b * 255).astype(np.uint8)
            # Fluoro → Magenta (R + B),  DRR → Cyan (G + B)
            # Overlap → Blanc (R+G+B)
            r_ch = a8
            g_ch = b8
            b_ch = np.clip(a8.astype(np.uint16) + b8.astype(np.uint16), 0, 255).astype(np.uint8)
            rgb = np.stack([r_ch, g_ch, b_ch], axis=2)

        # ── Contours superposés ───────────────────────────────────────────────
        if self._show_cnt:
            for pts, color in self._contours:
                if pts is None or len(pts) < 2: continue
                pts32 = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(rgb, [pts32], True, color, 1)

        # ── Redimensionner au widget ──────────────────────────────────────────
        lw = self._lbl_img.width(); lh = self._lbl_img.height()
        side = min(lw, lh)
        if side > 16:
            rgb = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_LINEAR)
        rgb = np.ascontiguousarray(rgb)
        h2, w2 = rgb.shape[:2]
        qimg = QImage(rgb.data, w2, h2, w2 * 3, QImage.Format_RGB888)
        self._lbl_img.setPixmap(QPixmap.fromImage(qimg).copy())


# ══════════════════════════════════════════════════════════════════════════════
# Worker thread
# ══════════════════════════════════════════════════════════════════════════════

class WorkerThread(QThread):
    progress=pyqtSignal(int,str); result=pyqtSignal(dict); error=pyqtSignal(str)
    def __init__(self,task,kw): super().__init__(); self.task=task; self.kw=kw
    def run(self):
        try: {'drr':self._drr,'register':self._reg}[self.task]()
        except Exception as ex:
            import traceback; self.error.emit(f'{ex}\n{traceback.format_exc()}')

    def _drr(self):
        self.progress.emit(10,'Génération DRR…')
        ct_vol = self.kw['ct_vol']
        ap_ax  = self.kw['ap_axis']
        # Slab AP : permet de cibler vertèbres ou cœur selon profondeur
        slab_c = self.kw.get('slab_c', 50.0) / 100.0
        slab_w = self.kw.get('slab_w', 100.0) / 100.0
        n = ct_vol.shape[ap_ax]
        c_idx = int(n * slab_c)
        half  = max(1, int(n * slab_w / 2))
        s0, s1 = max(0, c_idx - half), min(n, c_idx + half)
        sl = [slice(None)] * 3; sl[ap_ax] = slice(s0, s1)
        ct_slab = ct_vol[tuple(sl)]
        drr=generate_drr(ct_slab, self.kw['voxel_mm'], ap_axis=ap_ax,
                          lao_deg=self.kw['lao_deg'],cran_deg=self.kw['cran_deg'],
                          output_size=self.kw['output_size'],hu_min=-500,hu_max=2000,invert=True)
        self.progress.emit(60,'Projection segmentations…')
        masks_out={}
        for name,mask in self.kw.get('masks',{}).items():
            if mask is None or mask.sum()==0: continue
            mask_slab = mask[tuple(sl)]
            if mask_slab.sum()==0: continue
            masks_out[name]=project_mask_3d(mask_slab,self.kw['voxel_mm'],ap_axis=ap_ax,
                                             lao_deg=self.kw['lao_deg'],cran_deg=self.kw['cran_deg'],
                                             output_size=self.kw['output_size'])
        self.progress.emit(100,'DRR prêt'); self.result.emit({'drr':drr,'masks':masks_out})

    def _reg(self):
        elastic = self.kw.get('elastic', False)
        def cb(f, iou):
            if elastic:
                stage = 'Rigide' if f < 0.5 else 'Elastique'
                self.progress.emit(int(5 + f*90), f'{stage} — IoU={iou:.3f}')
            else:
                self.progress.emit(int(5 + f*90), f'Optimisation — IoU={iou:.3f}')
        if elastic:
            res = register_elastic(mask_moving=self.kw['moving'], mask_fixed=self.kw['fixed'],
                                   progress_cb=cb)
        else:
            res = register(mask_moving=self.kw['moving'], mask_fixed=self.kw['fixed'],
                           progress_cb=cb)
        self.progress.emit(100, f"IoU={res['iou']:.3f}"); self.result.emit(res)


# ══════════════════════════════════════════════════════════════════════════════
# Fenêtre segmentations 2D sur fluoroscopie
# ══════════════════════════════════════════════════════════════════════════════

_SEG_PALETTE = [
    (79,220,130),(240,80,90),(80,190,240),(240,180,0),(180,80,240),
    (240,120,60),(60,240,240),(240,60,180),(120,240,60),(60,120,240),
    (200,200,80),(80,200,200),(200,80,200),(160,240,120),(240,160,120),
]

class SegOverlayWindow(QDialog):
    """Fenêtre dédiée : fluoroscopie recalée + segmentations CT projetées, toggle par structure."""

    def __init__(self, fluoro: np.ndarray, proj_masks: dict, result: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Segmentations 2D — Vue fluoroscopie recalée')
        self.resize(720, 820)
        self.setStyleSheet(STYLE)
        self._fluoro = np.clip(fluoro, 0, 1).astype(np.float32)
        self._proj_masks = proj_masks
        self._result = result
        self._alpha = 0.40
        names = list(proj_masks.keys())
        self._colors = {n: _SEG_PALETTE[i % len(_SEG_PALETTE)] for i, n in enumerate(names)}
        self._chks: dict = {}
        self._build_ui()
        self._render()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)

        # Image
        self._lbl = QLabel()
        self._lbl.setAlignment(Qt.AlignCenter)
        self._lbl.setMinimumSize(512, 512)
        self._lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._lbl.setStyleSheet(f'background:{DARK_BG};border-radius:4px;')
        root.addWidget(self._lbl, 1)

        # Alpha slider
        row_a = QHBoxLayout()
        row_a.addWidget(QLabel('Transparence :'))
        sl_a = QSlider(Qt.Horizontal); sl_a.setRange(0, 100); sl_a.setValue(40)
        sl_a.valueChanged.connect(lambda v: (setattr(self, '_alpha', v/100), self._render()))
        row_a.addWidget(sl_a)
        root.addLayout(row_a)

        # Checkboxes panel
        grp = QGroupBox('STRUCTURES — activer / désactiver')
        grp_l = QVBoxLayout(grp)

        hdr = QHBoxLayout()
        btn_all  = QPushButton('Toutes');  btn_all.clicked.connect(self._select_all)
        btn_none = QPushButton('Aucune');  btn_none.clicked.connect(self._select_none)
        hdr.addWidget(btn_all); hdr.addWidget(btn_none); hdr.addStretch()
        grp_l.addLayout(hdr)

        # Scroll area for many structures
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(220); scroll.setStyleSheet(f'background:{PANEL_BG};border:none;')
        inner = QWidget(); inner.setStyleSheet(f'background:{PANEL_BG};')
        gl = QGridLayout(inner); gl.setSpacing(4)

        for i, name in enumerate(self._proj_masks.keys()):
            r, g, b = self._colors[name]
            chk = QCheckBox(name); chk.setChecked(True)
            chk.setStyleSheet(f'QCheckBox{{color: rgb({r},{g},{b});}}')
            chk.toggled.connect(self._render)
            self._chks[name] = chk
            gl.addWidget(chk, i // 2, i % 2)

        scroll.setWidget(inner)
        grp_l.addWidget(scroll)
        root.addWidget(grp)

    def _select_all(self):
        for chk in self._chks.values(): chk.setChecked(True)

    def _select_none(self):
        for chk in self._chks.values(): chk.setChecked(False)

    def _render(self):
        fl = (self._fluoro * 255).astype(np.uint8)
        rgb = cv2.cvtColor(fl, cv2.COLOR_GRAY2RGB).astype(np.float32)

        for name, mask in self._proj_masks.items():
            if not self._chks.get(name, QCheckBox()).isChecked():
                continue
            warped = apply_full_transform(mask.astype(np.float32), self._result)
            r, g, b = self._colors[name]
            ov = rgb.copy(); ov[warped > 0.5] = [r, g, b]
            rgb = cv2.addWeighted(rgb, 1 - self._alpha, ov, self._alpha, 0)
            m8 = (warped * 255).astype(np.uint8)
            cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            rgb8 = np.clip(rgb, 0, 255).astype(np.uint8)
            cv2.drawContours(rgb8, cnts, -1, (r, g, b), 2)
            rgb = rgb8.astype(np.float32)

        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        lw = max(16, self._lbl.width()); lh = max(16, self._lbl.height())
        side = min(lw, lh)
        rgb_r = cv2.resize(rgb, (side, side), interpolation=cv2.INTER_LINEAR)
        h2, w2 = rgb_r.shape[:2]
        qi = QImage(rgb_r.data, w2, h2, w2 * 3, QImage.Format_RGB888)
        self._lbl.setPixmap(QPixmap.fromImage(qi).copy())

    def resizeEvent(self, e):
        super().resizeEvent(e); self._render()


# ══════════════════════════════════════════════════════════════════════════════
# Fenêtre principale
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('2D/3D Registration — EP Lab')
        self.setMinimumSize(1280, 800)
        self.setStyleSheet(STYLE)
        self.ct_vol = self.voxel_mm = self.ct_aff = None
        self.ap_axis = 1
        self.seg_masks = {}; self.proj_masks = {}; self.drr_image = None
        self.full_stl_path = ''; self.full_stl_name = ''
        self.fluoro_image = None; self.result = None
        self._loaded_images = []; self._pending_csv = None
        self._build_ui()
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
        tl = QLabel('EP Lab  |  2D/3D Registration')
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

        self.lbl_ct = QLabel('CT : --'); self.lbl_ct.setObjectName('dim'); self.lbl_ct.setWordWrap(True)
        self.lbl_seg = QLabel('Seg : --'); self.lbl_seg.setObjectName('dim'); self.lbl_seg.setWordWrap(True)
        self.lbl_stl = QLabel('STL : --'); self.lbl_stl.setObjectName('dim'); self.lbl_stl.setWordWrap(True)
        sec_data.addWidget(self.lbl_ct)
        sec_data.addWidget(self.lbl_seg)
        sec_data.addWidget(self.lbl_stl)
        ll.addWidget(sec_data)

        # ── IMAGES ────────────────────────────────────────────────────────────
        self._sec_images = CollapsibleSection('IMAGES', starts_open=True)
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
        sec_drr = CollapsibleSection('PARAMETRES DRR', starts_open=False)
        drr_grid = QWidget(); drr_grid.setStyleSheet('background:transparent;')
        gdl = QGridLayout(drr_grid)
        gdl.setContentsMargins(0, 0, 0, 0); gdl.setSpacing(5)
        gdl.setColumnStretch(1, 1)

        def _lbl(t):
            l = QLabel(t); l.setObjectName('mid'); return l

        gdl.addWidget(_lbl('LAO/RAO (deg)'), 0, 0)
        self.sp_lao = QDoubleSpinBox(); self.sp_lao.setRange(-90, 90); self.sp_lao.setValue(0); self.sp_lao.setSingleStep(5)
        gdl.addWidget(self.sp_lao, 0, 1)
        gdl.addWidget(_lbl('Cran/Caud (deg)'), 1, 0)
        self.sp_cran = QDoubleSpinBox(); self.sp_cran.setRange(-45, 45); self.sp_cran.setValue(0); self.sp_cran.setSingleStep(5)
        gdl.addWidget(self.sp_cran, 1, 1)
        gdl.addWidget(_lbl('Resolution (px)'), 2, 0)
        self.sp_size = QSpinBox(); self.sp_size.setRange(256, 1024); self.sp_size.setValue(512); self.sp_size.setSingleStep(64)
        gdl.addWidget(self.sp_size, 2, 1)
        gdl.addWidget(_lbl('Centre slab (%)'), 3, 0)
        self.sp_slab_c = QDoubleSpinBox(); self.sp_slab_c.setRange(5, 95); self.sp_slab_c.setValue(50); self.sp_slab_c.setSingleStep(5); self.sp_slab_c.setSuffix(' %')
        gdl.addWidget(self.sp_slab_c, 3, 1)
        gdl.addWidget(_lbl('Epaisseur slab (%)'), 4, 0)
        self.sp_slab_w = QDoubleSpinBox(); self.sp_slab_w.setRange(5, 100); self.sp_slab_w.setValue(100); self.sp_slab_w.setSingleStep(5); self.sp_slab_w.setSuffix(' %')
        gdl.addWidget(self.sp_slab_w, 4, 1)

        sec_drr.addWidget(drr_grid)
        self.btn_drr = QPushButton('Generer DRR'); self.btn_drr.setObjectName('primary')
        self.btn_drr.clicked.connect(self.generate_drr); self.btn_drr.setEnabled(False)
        sec_drr.addWidget(self.btn_drr)
        ll.addWidget(sec_drr)

        # ── ANNOTATION ────────────────────────────────────────────────────────
        sec_ann = CollapsibleSection('ANNOTATION', starts_open=True)

        r0 = QHBoxLayout(); r0.setSpacing(6); r0.setContentsMargins(0, 0, 0, 0)
        r0.addWidget(_lbl('Structure :'))
        self.cb_struct = QComboBox()
        for k, info in STRUCT.items():
            self.cb_struct.addItem(info['label'], k)
        self.cb_struct.setCurrentIndex(1)
        self.cb_struct.currentIndexChanged.connect(self._on_struct)
        r0.addWidget(self.cb_struct, 1)
        sec_ann.addLayout(r0)

        r1 = QHBoxLayout(); r1.setSpacing(6); r1.setContentsMargins(0, 0, 0, 0)
        r1.addWidget(_lbl('Annoter :'))
        self.cb_canvas = QComboBox()
        self.cb_canvas.addItems(['Image fixe', 'Image mobile'])
        self.cb_canvas.currentIndexChanged.connect(self._on_canvas)
        r1.addWidget(self.cb_canvas, 1)
        sec_ann.addLayout(r1)

        tr = QHBoxLayout(); tr.setSpacing(6); tr.setContentsMargins(0, 0, 0, 0)
        self.btn_pencil = QPushButton('Crayon'); self.btn_pencil.setObjectName('tool'); self.btn_pencil.setCheckable(True); self.btn_pencil.setChecked(True)
        self.btn_rect = QPushButton('Rectangle'); self.btn_rect.setObjectName('tool'); self.btn_rect.setCheckable(True)
        self.btn_pencil.clicked.connect(lambda: self._set_tool('pencil'))
        self.btn_rect.clicked.connect(lambda: self._set_tool('rectangle'))
        tr.addWidget(self.btn_pencil, 1); tr.addWidget(self.btn_rect, 1)
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

        hint = QLabel('Crayon : tracer le contour puis relacher\nFermeture auto si retour au depart')
        hint.setObjectName('dim'); hint.setWordWrap(True)
        sec_ann.addWidget(hint)
        ll.addWidget(sec_ann)

        # ── RECALAGE ──────────────────────────────────────────────────────────
        sec_reg = CollapsibleSection('RECALAGE', starts_open=True)
        sec_reg.addWidget(_lbl('Structures pour le recalage :'))
        self.chk_structs = {}
        for k, info in STRUCT.items():
            chk = QCheckBox(info['label']); chk.setChecked(k in ('heart', 'vertebrae'))
            chk.setStyleSheet(f'QCheckBox{{color:{info["hex"]};}}'
                              f'QCheckBox::indicator{{width:13px;height:13px;border:1px solid {BORDER2};border-radius:3px;background:{CARD_BG};}}'
                              f'QCheckBox::indicator:checked{{background:{info["hex"]};border-color:{info["hex"]};}}')
            self.chk_structs[k] = chk; sec_reg.addWidget(chk)

        self.btn_reg = QPushButton('Lancer le recalage'); self.btn_reg.setObjectName('success')
        self.btn_reg.setEnabled(False); self.btn_reg.clicked.connect(self.run_registration)
        sec_reg.addWidget(self.btn_reg)
        self.chk_elastic = QCheckBox('Recalage elastique (deformation libre)')
        self.chk_elastic.setToolTip(
            'Active une deformation libre (FFD 4x4) apres le recalage rigide.\n'
            'Ameliore le IoU sur les formes complexes — plus lent (~15-30 s).')
        sec_reg.addWidget(self.chk_elastic)
        ll.addWidget(sec_reg)

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

        # ── ACTIONS ──────────────────────────────────────────────────────────
        sec_act = CollapsibleSection('ACTIONS', starts_open=True)

        self.btn_overlay = QPushButton('Vue 3D (STL)')
        self.btn_overlay.setObjectName('primary')
        self.btn_overlay.setEnabled(False)
        self.btn_overlay.clicked.connect(self.open_overlay)
        sec_act.addWidget(self.btn_overlay)

        self.btn_seg_overlay = QPushButton('Segmentations 2D')
        self.btn_seg_overlay.setObjectName('primary')
        self.btn_seg_overlay.setEnabled(False)
        self.btn_seg_overlay.clicked.connect(self.open_seg_overlay)
        sec_act.addWidget(self.btn_seg_overlay)

        btn_exp = QPushButton('Exporter resultats')
        btn_exp.clicked.connect(self.export_results)
        sec_act.addWidget(btn_exp)
        ll.addWidget(sec_act)

        ll.addStretch()

        root.addWidget(left_outer)

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

        for cv in [self.cv_fl, self.cv_drr]:
            cv.set_tool('pencil'); cv.set_active('heart')

        root.addWidget(self.tabs, 1)
        self.setStatusBar(QStatusBar())

    def _wrap(self, cv, hint):
        w = QWidget(); w.setStyleSheet(f'background:{PANEL_BG};')
        l = QVBoxLayout(w); l.setContentsMargins(4, 4, 4, 4); l.setSpacing(4)
        hl = QLabel(hint); hl.setObjectName('dim'); hl.setAlignment(Qt.AlignCenter)
        l.addWidget(hl); l.addWidget(cv, 1)
        return w

    # ── Slots UI ──────────────────────────────────────────────────────────────

    def _set_tool(self,tool):
        for btn,t in [(self.btn_pencil,'pencil'),(self.btn_rect,'rectangle')]:
            btn.setChecked(t==tool)
        for cv in [self.cv_fl,self.cv_drr]: cv.set_tool(tool)
        hints={'pencil':'Tracer le contour en continu, relacher pour valider',
               'rectangle':'Cliquer-glisser pour dessiner un rectangle'}
        self._status(hints.get(tool,''))

    def _on_struct(self,idx):
        k=self.cb_struct.itemData(idx)
        for cv in [self.cv_fl,self.cv_drr]: cv.set_active(k)

    def _on_canvas(self,idx): self.tabs.setCurrentIndex(idx)
    def _on_pen(self,v): self.lbl_pen.setText(f'{v}px'); [cv.set_pen_radius(v) for cv in [self.cv_fl,self.cv_drr]]
    def _undo(self): self._active_cv().undo()
    def _clear_all(self): self._active_cv().clear_all()

    def _save_mask(self):
        cv=self._active_cv(); m=cv.get_mask()
        if m is None or m.sum()==0: self._err('Masque vide'); return
        p,_=QFileDialog.getSaveFileName(self,'Sauvegarder','mask.png','PNG (*.png)')
        if p: cv2.imwrite(p,(m*255).astype(np.uint8)); self._status(f'Masque sauvegardé → {p}')

    def _active_cv(self): return self.cv_fl if self.cb_canvas.currentIndex()==0 else self.cv_drr

    def _on_mask_upd(self):
        has_fl  = self.cv_fl.get_mask()  is not None and self.cv_fl.get_mask().sum()  > 0
        has_drr = self.cv_drr.get_mask() is not None and self.cv_drr.get_mask().sum() > 0
        self.btn_reg.setEnabled(has_fl and has_drr)

    # ── Drag & Drop / Gestion images ─────────────────────────────────────────

    def _on_files_dropped(self, files):
        csv_files = []
        nifti_files = []
        other_files = []
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if f.lower().endswith('.nii.gz') or ext == '.nii':
                nifti_files.append(f)
            elif ext == '.csv':
                csv_files.append(f)
            else:
                other_files.append(f)
        csv_path = csv_files[0] if csv_files else self._pending_csv
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
            if ext == '.stl':
                self.load_full_stl(f)
            elif ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.dcm'):
                self._add_image(f)
        self._status(f'{len(files)} fichier(s) charge(s)')

    def _on_browse(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 'Selectionner des fichiers', '',
            'Tous (*.nii *.nii.gz *.csv *.stl *.png *.jpg *.jpeg *.bmp *.tiff *.tif *.dcm);;'
            'NIfTI (*.nii *.nii.gz);;Images (*.png *.jpg *.jpeg *.tiff *.bmp *.dcm);;'
            'CSV (*.csv);;STL (*.stl)')
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
                    idx = int(row[c[0]]); name = str(row[c[1]]).strip()
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
        if ext == '.dcm':
            try:
                import pydicom
                ds = pydicom.dcmread(path)
                arr = ds.pixel_array.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                img = (arr * 255).astype(np.uint8)
                if img.ndim == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            except Exception as ex:
                self._err(f'Lecture DICOM echouee : {ex}'); return
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self._err(f'Impossible de charger : {name}'); return
        s = self.sp_size.value()
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LANCZOS4)
        img_float = img.astype(np.float32) / 255.0
        idx = len(self._loaded_images)
        entry = {'path': path, 'name': name, 'array': img_float, 'role': None, 'card': None}
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
            self.fluoro_image = img
            self.cv_fl.set_image(img)
            self.tabs.setCurrentIndex(0)
            self._status(f'Image fixe : {name}')
        elif role == 'mobile':
            self.drr_image = img
            self.proj_masks = {}
            self.cv_drr.set_image(img)
            self.tabs.setTabText(1, 'Mobile')
            self.tabs.setCurrentIndex(1)
            self._status(f'Image mobile : {name}')

    # ── Actions ───────────────────────────────────────────────────────────────

    def load_ct(self, path=None):
        p = path
        if not p:
            p,_=QFileDialog.getOpenFileName(self,'CT NIfTI','','NIfTI (*.nii *.nii.gz)')
        if not p: return
        try:
            self.ct_vol,self.voxel_mm,self.ct_aff,_,self.ap_axis,codes=load_ct(p)
            self.ct_codes = codes   # ('R'/'L', 'A'/'P', 'S'/'I') — utilisé pour l'orientation STL
            self.lbl_ct.setText(f'CT: {os.path.basename(p)}\n  {self.ct_vol.shape} | {self.voxel_mm.round(2)} mm | AP={self.ap_axis} {codes}')
            self.btn_drr.setEnabled(True); self._status(f'CT chargé — axe AP={self.ap_axis} ({codes})')
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
            self._status(f'Segmentation chargée — {n} structures : {", ".join(list(self.seg_masks)[:6])}{"…" if n>6 else ""}')
        except Exception as ex: self._err(str(ex))

    def load_fluoro(self):
        p,_=QFileDialog.getOpenFileName(self,'Fluoroscopie','','Images (*.png *.jpg *.tiff *.bmp)')
        if not p: return
        img=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
        if img is None: self._err('Impossible de charger'); return
        s=self.sp_size.value()
        img=cv2.resize(img,(s,s),interpolation=cv2.INTER_LANCZOS4)
        self.fluoro_image=img.astype(np.float32)/255.0
        self.cv_fl.set_image(self.fluoro_image); self.lbl_fl.setText(f'Fluoro: {os.path.basename(p)}\n  {img.shape}')
        self.tabs.setCurrentIndex(0)
        self._status('Fluoroscopie chargée — sélectionner la structure et dessiner son contour')

    def generate_drr(self):
        if self.ct_vol is None: self._err('Charger un CT d\'abord'); return
        self.btn_drr.setEnabled(False)
        kw=dict(ct_vol=self.ct_vol,voxel_mm=self.voxel_mm,ap_axis=self.ap_axis,
                lao_deg=self.sp_lao.value(),cran_deg=self.sp_cran.value(),
                output_size=self.sp_size.value(),masks=self.seg_masks,
                slab_c=self.sp_slab_c.value(),slab_w=self.sp_slab_w.value())
        self.worker=WorkerThread('drr',kw)
        self.worker.progress.connect(self._on_prog); self.worker.result.connect(self._drr_done)
        self.worker.error.connect(self._on_err); self.worker.start()

    def _drr_done(self,res):
        self.drr_image=res['drr']; self.proj_masks=res.get('masks',{})
        self.cv_drr.set_image(self.drr_image); self.btn_drr.setEnabled(True)
        self.tabs.setTabText(1,'DRR')
        self.tabs.setCurrentIndex(1); self._status('DRR genere -- annoter les memes structures dans l\'onglet Mobile')

    def load_full_stl(self, path=None):
        p = path
        if not p:
            p,_=QFileDialog.getOpenFileName(self,'STL complet','','STL (*.stl)')
        if not p: return
        try:
            if pv is None:
                raise RuntimeError('PyVista non disponible. Installez pyvista.')
            mesh = pv.read(p)
            if mesh is None or mesh.n_points == 0:
                raise ValueError('STL vide ou illisible')
            self.full_stl_path = p
            self.full_stl_name = os.path.basename(p)
            self.lbl_stl.setText(f'STL: {self.full_stl_name}\n  {mesh.n_cells:,} cells | {mesh.n_points:,} pts')
            self.btn_overlay.setEnabled(self.result is not None and self.fluoro_image is not None)
            self._status('STL chargé — prêt pour la vue PyVista après recalage')
        except Exception as ex:
            self._err(f'Chargement STL échoué : {ex}')

    def load_xray(self):
        p,_=QFileDialog.getOpenFileName(self,'X-Ray image mobile','','Images (*.png *.jpg *.jpeg *.tiff *.bmp *.dcm)')
        if not p: return
        # Support DICOM simple via pydicom si disponible, sinon OpenCV
        img=None
        if p.lower().endswith('.dcm'):
            try:
                import pydicom
                ds=pydicom.dcmread(p)
                arr=ds.pixel_array.astype(np.float32)
                # Normaliser
                arr=(arr-arr.min())/(arr.max()-arr.min()+1e-8)
                img=(arr*255).astype(np.uint8)
                if img.ndim==3: img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            except Exception as ex:
                self._err(f'Lecture DICOM échouée : {ex}'); return
        else:
            img=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
        if img is None: self._err('Impossible de charger l\'image'); return
        s=self.sp_size.value()
        img=cv2.resize(img,(s,s),interpolation=cv2.INTER_LANCZOS4)
        self.drr_image=img.astype(np.float32)/255.0
        self.proj_masks={}
        self.cv_drr.set_image(self.drr_image)
        self.lbl_xray.setText(f'X-Ray: {os.path.basename(p)}\n  {img.shape}')
        self.tabs.setTabText(1,'X-Ray')
        self.tabs.setCurrentIndex(1)
        self._status('X-Ray chargé — dessiner les structures dans l\'onglet X-Ray')

    def run_registration(self):
        def fused(cv):
            out=None
            for k in STRUCT:
                if not self.chk_structs[k].isChecked(): continue
                m=cv.get_mask(k)
                if m is None or m.sum()==0: continue
                out=m.copy() if out is None else np.clip(out+m,0,1)
            return out
        mf=fused(self.cv_fl); md=fused(self.cv_drr)
        if mf is None or mf.sum()==0: self._err('Aucune structure annotée sur la fluoroscopie'); return
        if md is None or md.sum()==0: self._err('Aucune structure annotée sur le DRR'); return
        self.btn_reg.setEnabled(False)
        kw=dict(moving=md, fixed=mf, elastic=self.chk_elastic.isChecked())
        self.worker=WorkerThread('register',kw)
        self.worker.progress.connect(self._on_prog); self.worker.result.connect(self._reg_done)
        self.worker.error.connect(self._on_err); self.worker.start()

    def _reg_done(self,res):
        self.result=res; self.btn_reg.setEnabled(True)
        iou=res['iou']; dice=res['dice']
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
        self.btn_overlay.setEnabled(bool(self.full_stl_path) and self.fluoro_image is not None)
        self.btn_seg_overlay.setEnabled(bool(self.proj_masks) and self.fluoro_image is not None)
        self._status(f'Recalage termine -- IoU={iou:.3f}  Dice={dice:.3f}')

    def _build_result(self,res):
        s=self.sp_size.value()

        # ── Images source ─────────────────────────────────────────────────────
        fig_fl  = self.fluoro_image if self.fluoro_image is not None else np.zeros((s,s),np.float32)
        fig_drr = self.drr_image    if self.drr_image    is not None else np.zeros((s,s),np.float32)
        fig_drr_reg = apply_full_transform(fig_drr.astype(np.float32), res)

        # ── Contours à superposer ─────────────────────────────────────────────
        contours = []
        ct_cols={'vertebrae':(80,220,130),'heart':(240,80,90),'aorta':(80,190,240)}

        # Contours annotation fluoro (couleurs struct)
        for struct,info in STRUCT.items():
            m=self.cv_fl.get_mask(struct)
            if m is None or m.sum()==0: continue
            cnts,_=cv2.findContours((m*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for cnt in cnts:
                pts=cnt.squeeze()
                if pts.ndim>=2: contours.append((pts, info['rgb']))

        # Contours annotation DRR recales (gris clair)
        for struct in STRUCT:
            m=self.cv_drr.get_mask(struct)
            if m is None or m.sum()==0: continue
            mr=(apply_full_transform(m.astype(np.float32), res) > 0.5).astype(np.uint8)
            cnts,_=cv2.findContours((mr*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for cnt in cnts:
                pts=cnt.squeeze()
                if pts.ndim>=2: contours.append((pts, (210,210,210)))

        # Contours projections CT recalees
        for name,proj in self.proj_masks.items():
            pr=(apply_full_transform(proj.astype(np.float32), res) > 0.5).astype(np.uint8)
            col=ct_cols.get(name,(200,200,200))
            cnts,_=cv2.findContours((pr*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                pts=cnt.squeeze()
                if pts.ndim>=2: contours.append((pts, col))

        self.result_panel.set_data(fig_fl, fig_drr_reg, contours)

    def open_overlay(self):
        """Affiche le STL recalé sur la fluoroscopie dans une fenêtre PyVista.
        Projection directe (instantanée) — même géométrie que le DRR.
        """
        if self.result is None:       self._err('Lancez d\'abord le recalage'); return
        if self.fluoro_image is None: self._err('Aucune fluoroscopie chargée'); return
        if not self.full_stl_path:   self._err('Chargez un STL d\'abord'); return
        if pv is None:               self._err('pyvista non installé (pip install pyvista)'); return
        try:
            # ── 1. Lire le maillage ───────────────────────────────────────────
            mesh = pv.read(self.full_stl_path).triangulate().clean()
            if mesh.n_points == 0: raise ValueError('STL vide ou invalide')

            # ── 2. Projeter le STL en espace PyVista aligné sur la fluoro ────
            if self.ct_aff is None:
                raise RuntimeError('CT non chargé — impossible de projeter le STL')
            mesh.points = _project_stl_to_pyvista(
                mesh.points,
                ct_affine=self.ct_aff,
                voxel_mm=self.voxel_mm,
                ap_axis=self.ap_axis,
                lao_deg=self.sp_lao.value(),
                cran_deg=self.sp_cran.value(),
                output_size=self.sp_size.value(),
                reg_result=self.result,
            )

            # ── 3. Texture fluoroscopie (VTK : rangée 0 = bas → rangée 0 numpy
            #       apparaît en haut avec up=(0,-1,0) )
            fl = np.clip(self.fluoro_image, 0, 1)
            h, w = fl.shape[:2]
            tex = pv.numpy_to_texture(
                cv2.cvtColor((fl * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            )

            # ── 4. Plan portant la texture (Z=0) ─────────────────────────────
            plane = pv.Plane(
                center=(w * 0.5, h * 0.5, 0.0),
                direction=(0, 0, 1),
                i_size=float(w), j_size=float(h),
                i_resolution=1, j_resolution=1,
            )

            # ── 5. Rendu PyVista ──────────────────────────────────────────────
            pl = pv.Plotter(window_size=(960, 960),
                            title='Fluoroscopie + STL recalé')
            pl.add_mesh(plane, texture=tex, lighting=False)
            pl.add_mesh(mesh, color='tomato', opacity=0.45, smooth_shading=True)
            pl.add_axes()
            pl.enable_parallel_projection()
            # Caméra au-dessus (Z+), up=(0,-1,0) : coordonnées image = coordonnées monde
            pl.camera_position = [
                (w * 0.5, h * 0.5, max(w, h) * 2.5),
                (w * 0.5, h * 0.5, 0.0),
                (0.0, -1.0, 0.0),
            ]
            self._status(f'PyVista ouvert — '
                         f'tx={self.result["tx"]:+.1f} ty={self.result["ty"]:+.1f} '
                         f'θ={self.result["angle"]:+.2f}° s={self.result.get("scale",1):.3f}')
            pl.show()
        except Exception as ex:
            import traceback
            self._err(f'PyVista échoué : {ex}\n{traceback.format_exc()}')

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
            json.dump({'tx_px':float(self.result['tx']),'ty_px':float(self.result['ty']),
                       'angle_deg':float(self.result['angle']),
                       'scale':float(self.result.get('scale',1.0)),
                       'iou':float(self.result['iou']),
                       'dice':float(self.result['dice']),
                       'lao_deg':self.sp_lao.value(),
                       'cran_deg':self.sp_cran.value(),
                       'ap_axis':int(self.ap_axis)},f,indent=2)
        self._status(f'Exporté → {folder}')
        QMessageBox.information(self,'Export',f'Sauvegardé dans :\n{folder}')

    def open_seg_overlay(self):
        if not self.proj_masks:
            self._err('Aucune segmentation projetée — chargez une seg. et générez le DRR')
            return
        if self.result is None:
            self._err('Lancez d\'abord le recalage')
            return
        if self.fluoro_image is None:
            self._err('Aucune fluoroscopie chargée')
            return
        win = SegOverlayWindow(self.fluoro_image, self.proj_masks, self.result, parent=self)
        win.exec_()

    def _on_prog(self,pct,msg): self.prog_bar.setValue(pct); self.lbl_prog.setText(msg); self._status(msg)
    def _on_err(self,msg): self.btn_drr.setEnabled(True); self.btn_reg.setEnabled(True); self._err(msg)
    def _status(self,msg): self.statusBar().showMessage(msg)
    def _err(self,msg): self.statusBar().showMessage(msg); QMessageBox.warning(self,'Erreur',msg)


def main():
    app=QApplication(sys.argv)
    win=MainWindow(); win.show()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
