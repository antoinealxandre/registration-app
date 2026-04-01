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

try:
    import pydicom
except ImportError:
    pydicom = None

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QGroupBox,
    QSpinBox, QDoubleSpinBox, QProgressBar, QTabWidget, QSizePolicy,
    QStatusBar, QFrame, QMessageBox, QComboBox, QGridLayout, QCheckBox,
    QDialog, QScrollArea, QToolButton, QButtonGroup,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap, QCursor, QFont, QIcon, QPainter

sys.path.insert(0, os.path.dirname(__file__))
from core.drr_generator import load_ct, generate_drr, project_mask_3d
from core.registration import (register, register_elastic,
                               apply_transform, apply_full_transform,
                               iou_score, dice_score)
from core.yolo_pipeline import (
    load_yolo_model as yolo_load, is_model_loaded as yolo_ready,
    detect_vertebrae, boxes_to_mask, draw_detections,
)


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
    'autre':     {'rgb': (80, 190, 240),  'hex': '#50bef0', 'label': 'Autre'},
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


# ══════════════════════════════════════════════════════════════════════════════
# Icônes SVG (Material Icons)
# ══════════════════════════════════════════════════════════════════════════════

def _make_svg_icon(path_d: str, color: str = '#cdd5e8', size: int = 20):
    """Crée une QIcon depuis un path SVG Material Icons (viewBox 0 0 24 24)."""
    try:
        from PyQt5.QtSvg import QSvgRenderer
        from PyQt5.QtCore import QByteArray
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
               f'<path fill="{color}" d="{path_d}"/></svg>')
        renderer = QSvgRenderer(QByteArray(svg.encode('utf-8')))
        img = QImage(size, size, QImage.Format_ARGB32)
        img.fill(0)
        p = QPainter(img)
        renderer.render(p)
        p.end()
        return QIcon(QPixmap.fromImage(img))
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Zone de depot (Drag & Drop)
# ══════════════════════════════════════════════════════════════════════════════

class DropZone(QFrame):
    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumHeight(70)
        self._lbl = QLabel('Deposer vos fichiers ici\nCT / Segmentation / Images')
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

    def set_open(self, opened: bool):
        if self._open != opened:
            self.toggle()

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
        self._active  = 'vertebrae'
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
        h, w = img.shape[:2]
        s = max(h, w)
        u8 = (np.clip(img,0,1)*255).astype(np.uint8) if img.dtype != np.uint8 else img.copy()
        # Rendre l'image carrée si nécessaire
        if h != w:
            u8_sq = np.zeros((s, s), dtype=np.uint8) if u8.ndim == 2 else np.zeros((s, s, 3), dtype=np.uint8)
            u8_sq[:h, :w] = u8
            u8 = u8_sq
        self._size = s
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

    def set_mask(self, struct: str, mask: np.ndarray):
        """Injecte un masque externe (ex: projection CT) dans un slot de structure."""
        if struct not in self._masks:
            return
        s = self._size
        if mask.shape != (s, s):
            mask = cv2.resize(mask.astype(np.float32), (s, s), interpolation=cv2.INTER_LINEAR)
        mask = (mask > 0.5).astype(np.float32)
        # Push current state to history for this struct
        cur = self._masks[struct]
        if cur is not None:
            h = self._history[struct]; h.append(cur.copy())
            if len(h) > 30: h.pop(0)
        self._masks[struct] = mask
        self._refresh()
        self.mask_updated.emit()

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
        try: {'drr':self._drr,'register':self._reg,
               'yolo_detect':self._yolo_detect,
               'auto_pipeline':self._auto_pipeline,
               'auto_phase2':self._auto_phase2}[self.task]()
        except Exception as ex:
            import traceback; self.error.emit(f'{ex}\n{traceback.format_exc()}')

    def _yolo_detect(self):
        kw = self.kw
        self.progress.emit(10, f'Détection YOLO ({kw["target"]})…')
        det = detect_vertebrae(
            kw['img'], conf=kw.get('conf', 0.25), iou=kw.get('iou', 0.45),
            imgsz=kw.get('imgsz', 288), preprocess=kw.get('preprocess', True),
            preprocess_params=kw.get('pp', {}))
        self.progress.emit(100, f'{det["n_detections"]} vertèbre(s) — {kw["target"]}')
        self.result.emit({'task': 'yolo_detect', 'target': kw['target'], **det})

    def _drr(self):
        def pcb(pct, msg):
            self.progress.emit(pct, msg)

        pcb(5, 'Génération DRR (DiffDRR cone-beam)…')
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
        pcb(80, 'Projection segmentations…')
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
        pcb(100, 'DRR prêt')
        self.result.emit({'drr': drr, 'masks': masks_out})

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

    # ── Pipeline automatique complet ──────────────────────────────────────────
    def _auto_pipeline(self):
        """
        Pipeline semi-auto :
          1. Génération DRR (avec cran+180 pour symétrie PA)
          2. Projection de toutes les segmentations (pour overlay après recalage)
          3. Détection YOLO sur le DRR (sans preprocessing)
          4. Détection YOLO sur la fluoro (avec preprocessing, 1024×1024)
          → Pause : fenêtre duale de sélection des vertèbres
          Phase 2 : construction masques + recalage élastique
        """
        import re as _re
        kw = self.kw
        ct_path   = kw['ct_path']
        ct_aff    = kw['ct_aff']
        seg_masks = kw['seg_masks']       # dict {name: 3d_mask}
        fluoro    = kw['fluoro']           # float32 [0,1]
        reg_size  = kw['output_size']
        yolo_kw   = kw.get('yolo_kw', {})

        # Angles bruts UI — cran_deg inclut DÉJÀ le +180 (ajouté par le caller)
        lao_deg     = kw['lao_deg']
        cran_deg    = kw['cran_deg']      # déjà UI + 180
        table_angle = kw.get('table_angle', 0.0)

        # ── Kwargs géométriques partagés ──────────────────────────────────────
        geom_kw = dict(lao_deg=lao_deg, cran_deg=cran_deg,
                       table_angle=table_angle, output_size=reg_size)
        for k in ('sid_mm', 'sod_mm', 'fov_mm'):
            if k in kw and kw[k] is not None:
                geom_kw[k] = kw[k]
        renderer = kw.get('renderer', 'siddon')

        # ══════════════════════════════════════════════════════════════════════
        # 1. Génération DRR (symétrie PA avec cran+180 déjà appliquée)
        # ══════════════════════════════════════════════════════════════════════
        self.progress.emit(5, 'Génération DRR (cone-beam)…')
        drr_image = generate_drr(
            ct_path=ct_path, renderer=renderer,
            progress_cb=lambda pct, msg: self.progress.emit(5 + int(pct * 0.12), msg),
            **geom_kw)

        # ══════════════════════════════════════════════════════════════════════
        # 2. Projection de TOUTES les segmentations (pour overlay après recalage)
        # ══════════════════════════════════════════════════════════════════════
        self.progress.emit(20, 'Projection des segmentations…')
        proj_kw = dict(**geom_kw, ct_path=ct_path, ct_affine=ct_aff,
                       renderer=renderer)
        all_proj_masks = {}
        seg_names = list(seg_masks.keys())
        for i, name in enumerate(seg_names):
            proj = project_mask_3d(mask_3d=seg_masks[name], **proj_kw)
            if proj.sum() > 0:
                all_proj_masks[name] = proj
            self.progress.emit(20 + int((i + 1) / max(len(seg_names), 1) * 8),
                               f'Projection {name}…')

        # ══════════════════════════════════════════════════════════════════════
        # 3. Détection YOLO sur le DRR (sans preprocessing)
        # ══════════════════════════════════════════════════════════════════════
        if not yolo_ready():
            raise RuntimeError('Le modèle YOLO n\'est pas chargé.')

        self.progress.emit(30, 'Détection YOLO sur le DRR…')
        drr_u8 = (np.clip(drr_image, 0, 1) * 255).astype(np.uint8)
        drr_resized = cv2.resize(drr_u8, (reg_size, reg_size),
                                 interpolation=cv2.INTER_LANCZOS4)
        det_drr = detect_vertebrae(
            drr_resized,
            conf=yolo_kw.get('conf', 0.25),
            iou=yolo_kw.get('iou', 0.45),
            imgsz=yolo_kw.get('imgsz', 288),
            preprocess=False)
        drr_boxes = det_drr['boxes']
        if not drr_boxes:
            raise RuntimeError(
                'Aucune vertèbre détectée sur le DRR.\n'
                'Essayez de baisser le seuil de confiance YOLO.')
        self.progress.emit(38,
            f'YOLO DRR : {len(drr_boxes)} vertèbre(s) détectée(s)')

        # ══════════════════════════════════════════════════════════════════════
        # 4. Détection YOLO sur la fluoroscopie (preprocessing, 1024×1024)
        # ══════════════════════════════════════════════════════════════════════
        self.progress.emit(40, 'Détection YOLO sur la fluoroscopie (1024×1024)…')
        yolo_size = 1024
        fl_u8 = (np.clip(fluoro, 0, 1) * 255).astype(np.uint8)
        fl_resized = cv2.resize(fl_u8, (yolo_size, yolo_size),
                                interpolation=cv2.INTER_LANCZOS4)
        det_fl = detect_vertebrae(
            fl_resized,
            conf=yolo_kw.get('conf', 0.25),
            iou=yolo_kw.get('iou', 0.45),
            imgsz=yolo_kw.get('imgsz', 288),
            preprocess=True,
            preprocess_params=yolo_kw.get('pp', {}))
        boxes_fl_raw = det_fl['boxes']
        n_fl_raw = len(boxes_fl_raw)
        if n_fl_raw == 0:
            raise RuntimeError(
                'Aucune vertèbre détectée sur la fluoroscopie.\n'
                'Essayez de baisser le seuil de confiance YOLO.')

        # Rescaler les boîtes fluoro de yolo_size → reg_size
        scale_f = reg_size / yolo_size
        boxes_fl = []
        for b in boxes_fl_raw:
            boxes_fl.append({
                'x1': int(b['x1'] * scale_f), 'y1': int(b['y1'] * scale_f),
                'x2': int(b['x2'] * scale_f), 'y2': int(b['y2'] * scale_f),
                'conf': b['conf'], 'cls_name': b.get('cls_name', ''),
            })
        n_fl = len(boxes_fl)
        self.progress.emit(50, f'{n_fl} vertèbre(s) détectée(s) sur la fluoro')

        # ══════════════════════════════════════════════════════════════════════
        # Pause : on renvoie les données au thread principal pour que
        # l'utilisateur sélectionne les vertèbres (fluoro + DRR) via le
        # panneau dual. La phase 2 (recalage élastique) sera relancée après.
        # ══════════════════════════════════════════════════════════════════════
        self.progress.emit(50, 'Sélection des vertèbres…')
        self.result.emit({
            '_phase': 'select_vertebrae',
            'drr_image': drr_image,
            'all_proj_masks': all_proj_masks,
            'drr_boxes': drr_boxes,
            'det_fl': det_fl,
            'det_drr': det_drr,
            'boxes_fl': boxes_fl,
            'reg_size': reg_size,
        })

    # ── Phase 2 du pipeline auto : appariement + recalage ─────────────────
    def _auto_phase2(self):
        """
        Phase 2 : reçoit les boîtes sélectionnées par l'utilisateur
        (fluoro + DRR) et effectue le recalage élastique.
        """
        kw = self.kw
        boxes_fl        = kw['boxes_fl']
        boxes_drr       = kw['boxes_drr']
        reg_size        = kw['reg_size']
        drr_image       = kw['drr_image']
        all_proj_masks  = kw['all_proj_masks']

        n_fl  = len(boxes_fl)
        n_drr = len(boxes_drr)

        if n_fl == 0:
            raise RuntimeError('Aucune vertèbre fluoro sélectionnée.')
        if n_drr == 0:
            raise RuntimeError('Aucune vertèbre DRR sélectionnée.')

        # ══════════════════════════════════════════════════════════════════════
        # Construction des masques (boîtes sélectionnées) + recalage élastique
        # ══════════════════════════════════════════════════════════════════════
        self.progress.emit(55, 'Construction des masques…')

        mask_drr = np.zeros((reg_size, reg_size), dtype=np.float32)
        for b in boxes_drr:
            mask_drr[b['y1']:b['y2'], b['x1']:b['x2']] = 1.0

        mask_fl = np.zeros((reg_size, reg_size), dtype=np.float32)
        for b in boxes_fl:
            mask_fl[b['y1']:b['y2'], b['x1']:b['x2']] = 1.0

        self.progress.emit(58, 'Recalage élastique (rigide + FFD)…')

        def reg_cb(frac, iou_val):
            stage = 'Rigide' if frac < 0.5 else 'Elastique'
            self.progress.emit(58 + int(frac * 35), f'{stage} — IoU={iou_val:.4f}')

        res = register_elastic(mask_moving=mask_drr, mask_fixed=mask_fl,
                               progress_cb=reg_cb)

        self.progress.emit(95,
            f'Recalage terminé — IoU={res["iou"]:.4f}  Dice={res["dice"]:.4f}')

        # ── Résultat complet ──────────────────────────────────────────────────
        res['_auto'] = True
        res['_phase'] = 'done'
        res['drr_image'] = drr_image
        res['proj_masks'] = all_proj_masks
        res['n_fluoro_sel'] = n_fl
        res['n_drr_sel'] = n_drr
        res['mask_fl'] = mask_fl
        res['mask_drr'] = mask_drr
        self.progress.emit(100,
            f'Pipeline terminé — IoU={res["iou"]:.4f}  Dice={res["dice"]:.4f}')
        self.result.emit(res)


# ══════════════════════════════════════════════════════════════════════════════
# Fenêtre segmentations 2D sur fluoroscopie
# ══════════════════════════════════════════════════════════════════════════════

_SEG_PALETTE = [
    (79,220,130),(240,80,90),(80,190,240),(240,180,0),(180,80,240),
    (240,120,60),(60,240,240),(240,60,180),(120,240,60),(60,120,240),
    (200,200,80),(80,200,200),(200,80,200),(160,240,120),(240,160,120),
]


# ══════════════════════════════════════════════════════════════════════════════
# Panneau de résultats détection YOLO
# ══════════════════════════════════════════════════════════════════════════════

_YOLO_BOX_PALETTE = [
    (80, 220, 130), (255, 100, 100), (100, 180, 255),
    (255, 200, 80), (200, 130, 255), (130, 255, 200),
    (255, 160, 100), (100, 255, 255), (255, 100, 200),
    (200, 255, 100),
]


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
        btn_close = QPushButton('Fermer'); btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_export); btn_row.addWidget(btn_close)
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

    def resizeEvent(self, e):
        super().resizeEvent(e); self._render()


# ══════════════════════════════════════════════════════════════════════════════
# Lecture DICOM fluoroscopie — extraction des paramètres géométriques
# ══════════════════════════════════════════════════════════════════════════════

def read_dicom_fluoro(path: str):
    """
    Lit un DICOM de fluoroscopie (XA mono ou multi-frames) et extrait les
    paramètres géométriques complets du C-arm.

    Retourne (img_uint8, meta) avec meta contenant :
        lao, cran       – angles positionneur [deg]
        sid_mm, sod_mm  – distances source-détecteur / source-patient [mm]
        magnification   – facteur de grossissement (SID/SOD)
        pixel_mm        – ImagerPixelSpacing [mm/px]
        fov_mm          – champ de vue à l'isocentre [mm]
        fov_dim_mm      – (float,float) FieldOfViewDimensions au détecteur [mm]
        intensifier_mm  – IntensifierSize [mm]
        table_angle     – angle de la table [deg]
        arm_l, arm_p, arm_c – angles bras L/P/C (tags GE privés)
        patient_pos     – PatientPosition (HFS, FFS, …)
        shutter         – dict (left,right,upper,lower) bords du shutter [px]
        collimator      – dict (left,right,upper,lower) bords du collimateur [px]
        fov_shape        – forme du FOV (RECTANGLE, ROUND, …)
        fov_origin       – (int,int) FieldOfViewOrigin [px]
        rows, cols, n_frames, frame_used
        manufacturer, model
    """
    if pydicom is None:
        raise RuntimeError('pydicom non disponible — installez-le : pip install pydicom')

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array  # (frames, H, W) ou (H, W)

    # ── Sélection de la frame (multi-frame XA) ────────────────────────────
    if arr.ndim == 3:
        n_frames = arr.shape[0]
        rep = int(getattr(ds, 'RepresentativeFrameNumber', (n_frames + 1) // 2))
        frame_idx = max(0, min(rep - 1, n_frames - 1))  # 1-based → 0-based
        frame = arr[frame_idx].astype(np.float32)
    else:
        n_frames, frame_idx = 1, 0
        frame = arr.astype(np.float32)

    # ── Normalisation ────────────────────────────────────────────────────────
    fmin, fmax = frame.min(), frame.max()
    if fmax > fmin:
        frame = (frame - fmin) / (fmax - fmin)
    img_uint8 = (frame * 255).astype(np.uint8)

    # ── Helper lecture tags ───────────────────────────────────────────────────
    def _get(tag, default):
        val = getattr(ds, tag, None)
        if val is None:
            return default
        try:
            v = float(str(val).split('\\')[0])
            return v
        except Exception:
            return default

    def _get_str(tag, default=''):
        val = getattr(ds, tag, None)
        return str(val).strip() if val is not None else default

    def _get_multi(tag):
        """Retourne une liste de floats depuis un tag DS multi-value."""
        val = getattr(ds, tag, None)
        if val is None:
            return None
        try:
            return [float(v) for v in str(val).split('\\')]
        except Exception:
            return None

    def _get_private(group, elem, default=None):
        """Lit un tag privé (ex: [0019,1001])."""
        try:
            de = ds[group, elem]
            return float(de.value)
        except Exception:
            return default

    # ── Angles positionneur ──────────────────────────────────────────────────
    lao  = _get('PositionerPrimaryAngle',   0.0)
    cran = _get('PositionerSecondaryAngle', 0.0)

    # ── Distances et grossissement ───────────────────────────────────────────
    sid  = _get('DistanceSourceToDetector', 1000.0)
    sod  = _get('DistanceSourceToPatient',  750.0)
    mag  = _get('EstimatedRadiographicMagnificationFactor', sid / sod if sod > 0 else 1.0)

    # ── Pixel spacing ────────────────────────────────────────────────────────
    ips = _get_multi('ImagerPixelSpacing')
    pixel_mm = ips[0] if ips else 0.2

    # ── Dimensions détecteur / FOV ───────────────────────────────────────────
    rows = int(getattr(ds, 'Rows', img_uint8.shape[0]))
    cols = int(getattr(ds, 'Columns', img_uint8.shape[1]))
    fov_dim = _get_multi('FieldOfViewDimensions')           # au détecteur
    fov_dim_mm = tuple(fov_dim) if fov_dim else (pixel_mm * cols, pixel_mm * rows)
    intensifier_mm = _get('IntensifierSize', 0.0)
    fov_shape = _get_str('FieldOfViewShape', '')
    fov_origin_raw = _get_multi('FieldOfViewOrigin')
    fov_origin = tuple(int(v) for v in fov_origin_raw) if fov_origin_raw else None

    # Champ de vue à l'isocentre (corrigé du grossissement)
    fov_mm = fov_dim_mm[0] * (sod / sid) if sid > 0 else fov_dim_mm[0]

    # ── Table ────────────────────────────────────────────────────────────────
    table_angle = _get('TableAngle', 0.0)

    # ── Angles bras (tags privés GE) ─────────────────────────────────────────
    arm_l = _get_private(0x0019, 0x1001, None)
    arm_p = _get_private(0x0019, 0x1002, None)
    arm_c = _get_private(0x0019, 0x1003, None)

    # ── Patient position ─────────────────────────────────────────────────────
    patient_pos = _get_str('PatientPosition', 'HFS')

    # ── Shutter / Collimateur ────────────────────────────────────────────────
    shutter = {
        'left':  int(_get('ShutterLeftVerticalEdge',  0)),
        'right': int(_get('ShutterRightVerticalEdge', cols)),
        'upper': int(_get('ShutterUpperHorizontalEdge', 0)),
        'lower': int(_get('ShutterLowerHorizontalEdge', rows)),
    }
    collimator = {
        'left':  int(_get('CollimatorLeftVerticalEdge',  0)),
        'right': int(_get('CollimatorRightVerticalEdge', cols)),
        'upper': int(_get('CollimatorUpperHorizontalEdge', 0)),
        'lower': int(_get('CollimatorLowerHorizontalEdge', rows)),
    }

    # ── Fabricant / Modèle ───────────────────────────────────────────────────
    manufacturer = _get_str('Manufacturer', '')
    model = _get_str('ManufacturerModelName', '')

    meta = dict(
        lao=lao, cran=cran,
        sid_mm=sid, sod_mm=sod,
        magnification=mag,
        pixel_mm=pixel_mm,
        fov_mm=fov_mm,
        fov_dim_mm=fov_dim_mm,
        intensifier_mm=intensifier_mm,
        fov_shape=fov_shape,
        fov_origin=fov_origin,
        table_angle=table_angle,
        arm_l=arm_l, arm_p=arm_p, arm_c=arm_c,
        patient_pos=patient_pos,
        shutter=shutter,
        collimator=collimator,
        rows=rows, cols=cols,
        n_frames=n_frames, frame_used=frame_idx + 1,
        manufacturer=manufacturer, model=model,
    )
    return img_uint8, meta


def read_metadata_csv(path: str):
    """
    Lit un CSV de métadonnées DICOM exporté par 3D Slicer
    (colonnes : Tag, Name, Value, VR, Length).
    Retourne un dict compatible avec read_dicom_fluoro.meta.
    """
    import re as _re
    df = pd.read_csv(path)
    # Si le header ressemble à un tag DICOM, le CSV n'a pas d'en-tête → re-lire
    if _re.match(r'\[?[0-9a-fA-F]{8}\]?$', str(df.columns[0]).strip()):
        df = pd.read_csv(path, header=None,
                         names=['Tag', 'Name', 'Value', 'VR', 'Length'][:None])
        # Ajuster si le nombre de colonnes diffère
        if len(df.columns) < 3:
            raise ValueError(f'CSV metadata : au moins 3 colonnes attendues, {len(df.columns)} trouvées')
    # Normalise les noms de colonnes
    cols_lower = {c.strip().lower(): c for c in df.columns}
    name_col = cols_lower.get('name', cols_lower.get('keyword', df.columns[1]))
    val_col  = cols_lower.get('value', df.columns[2])

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

    lao  = _f('PositionerPrimaryAngle', 0.0)
    cran = _f('PositionerSecondaryAngle', 0.0)
    sid  = _f('DistanceSourceToDetector', 1000.0)
    sod  = _f('DistanceSourceToPatient', 750.0)
    mag  = _f('EstimatedRadiographicMagnificationFactor', sid / sod if sod > 0 else 1.0)

    ips = _flist('ImagerPixelSpacing')
    pixel_mm = ips[0] if ips else 0.2

    rows = int(_f('Rows', 1000))
    cols = int(_f('Columns', 1000))
    fov_dim = _flist('FieldOfViewDimensions')
    fov_dim_mm = tuple(fov_dim) if fov_dim else (pixel_mm * cols, pixel_mm * rows)
    intensifier_mm = _f('IntensifierSize', 0.0)

    fov_mm = fov_dim_mm[0] * (sod / sid) if sid > 0 else fov_dim_mm[0]

    table_angle = _f('TableAngle', 0.0)
    arm_l = _f('AngleValueLArm', None) if 'AngleValueLArm' in lookup else None
    arm_p = _f('AngleValuePArm', None) if 'AngleValuePArm' in lookup else None
    arm_c = _f('AngleValueCArm', None) if 'AngleValueCArm' in lookup else None

    patient_pos = lookup.get('PatientPosition', 'HFS')

    shutter = {
        'left':  int(_f('ShutterLeftVerticalEdge', 0)),
        'right': int(_f('ShutterRightVerticalEdge', cols)),
        'upper': int(_f('ShutterUpperHorizontalEdge', 0)),
        'lower': int(_f('ShutterLowerHorizontalEdge', rows)),
    }
    collimator = {
        'left':  int(_f('CollimatorLeftVerticalEdge', 0)),
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
        lao=lao, cran=cran,
        sid_mm=sid, sod_mm=sod,
        magnification=mag,
        pixel_mm=pixel_mm,
        fov_mm=fov_mm,
        fov_dim_mm=fov_dim_mm,
        intensifier_mm=intensifier_mm,
        fov_shape=fov_shape,
        fov_origin=fov_origin,
        table_angle=table_angle,
        arm_l=arm_l, arm_p=arm_p, arm_c=arm_c,
        patient_pos=patient_pos,
        shutter=shutter,
        collimator=collimator,
        rows=rows, cols=cols,
        n_frames=n_frames, frame_used=frame_used,
        manufacturer=manufacturer, model=model,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Fenêtre principale
# ══════════════════════════════════════════════════════════════════════════════

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
        self.sp_fov = QDoubleSpinBox(); self.sp_fov.setRange(50, 500); self.sp_fov.setValue(300); self.sp_fov.setSingleStep(10)
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
        self.btn_load_yolo = QPushButton('Charger modèle (.pt)')
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

        self.btn_seg_overlay = QPushButton('Segmentations 2D')
        self.btn_seg_overlay.setObjectName('primary')
        self.btn_seg_overlay.setEnabled(False)
        self.btn_seg_overlay.clicked.connect(self.open_seg_overlay)
        sec_act.addWidget(self.btn_seg_overlay)

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
            'DRR + Détection + Appariement + Recalage\n'
            'en un seul clic. Nécessite : CT, Segmentation\n'
            'vertébrale, Fluoroscopie DICOM et modèle YOLO.')
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

        root.addWidget(left_outer)


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

        for cv in [self.cv_fl, self.cv_drr]:
            cv.set_tool('pencil'); cv.set_active('vertebrae')

        root.addWidget(self.tabs, 1)
        self.setStatusBar(QStatusBar())

    def _wrap(self, cv, hint):
        w = QWidget(); w.setStyleSheet(f'background:{PANEL_BG};')
        l = QVBoxLayout(w); l.setContentsMargins(4, 4, 4, 4); l.setSpacing(6)
        
        # ── Sélection de structure à annoter ──────────────────────────────────
        struct_ctrl = QHBoxLayout(); struct_ctrl.setSpacing(6)
        struct_ctrl.addWidget(QLabel('Structure :').setObjectName('dim') or QLabel('Structure :'))
        
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
        has_fl  = self.cv_fl.get_mask()  is not None and self.cv_fl.get_mask().sum()  > 0
        has_drr = self.cv_drr.get_mask() is not None and self.cv_drr.get_mask().sum() > 0
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

    def load_xray(self):
        p,_=QFileDialog.getOpenFileName(self,'X-Ray image mobile','','Images (*.png *.jpg *.jpeg *.tiff *.bmp *.dcm *);;All (*)')
        if not p: return
        img=None
        ext = os.path.splitext(p)[1].lower()
        is_dcm = ext in ('.dcm', '.ima')
        if not is_dcm and ext == '' and pydicom is not None:
            try:
                pydicom.dcmread(p, stop_before_pixels=True)
                is_dcm = True
            except Exception:
                pass
        if is_dcm:
            try:
                img, _ = read_dicom_fluoro(p)
                if img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            except Exception as ex:
                self._err(f'Lecture DICOM echouee : {ex}'); return
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
        self.btn_seg_overlay.setEnabled(bool(self.proj_masks) and self.fluoro_image is not None)
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

    # ── Pipeline automatique complet ──────────────────────────────────────────

    def _run_auto_pipeline(self):
        """Lance le pipeline complet : DRR → Détection → Appariement → Recalage."""
        # Vérifications
        if self.ct_path is None:
            self._err('Chargez un CT (NIfTI) d\'abord.'); return
        if not self.seg_masks:
            self._err('Chargez une segmentation vertébrale (NIfTI + CSV labels).'); return
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
            seg_masks=self.seg_masks,
            fluoro=self.fluoro_image,
            output_size=self.sp_size.value(),
            lao_deg=self.sp_lao.value(),
            cran_deg=self.sp_cran.value() + 180,  # convention UI 0° = PA (180°)
            table_angle=self.sp_table.value(),
            sid_mm=self.dicom_meta.get('sid_mm', 1020.0),
            sod_mm=self.dicom_meta.get('sod_mm', 510.0),
            fov_mm=self.dicom_meta.get('fov_mm'),
            renderer='siddon',
            elastic=self.chk_elastic.isChecked(),
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

        # Activer le bouton segmentations 2D
        self.btn_seg_overlay.setEnabled(
            bool(self.proj_masks) and self.fluoro_image is not None)

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


def main():
    app=QApplication(sys.argv)
    win=MainWindow(); win.show()
    sys.exit(app.exec_())

if __name__=='__main__':
    main()
