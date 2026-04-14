"""Reusable annotation widgets and visualization panels."""

import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider,
    QComboBox, QSizePolicy, QCheckBox, QFrame, QFileDialog, QMessageBox,
    QScrollArea,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QCursor, QPainter, QIcon

from core.registration import apply_full_transform
from ui.theme import (
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
    STRUCT,
    SEG_PALETTE as _SEG_PALETTE,
)

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
# Panneau final : Fluoroscopie + Segmentations 3D projetées et recalées
# ══════════════════════════════════════════════════════════════════════════════

# Palette médicale par nom de structure
_MEDICAL_COLORS = {
    'myocardium':               (206, 110,  84),
    'left atrium':              (203, 108,  81),
    'left ventricle':           (152,  55,  13),
    'left ventricle of heart':  (152,  55,  13),
    'right atrium':             (210, 115,  89),
    'right ventricle':          (181,  85,  57),
    'right ventricle of heart': (181,  85,  57),
    'aorta':                    (224,  97,  76),
    'pulmonary artery':         (  0, 122, 171),
    'pulmonary venous system':  (186,  77,  64),
    'atrial_appendage_left':    (142, 192,  72),
    'left atrial appendage':    (142, 192,  72),
    'superior_vena_cava':       (115, 176, 130),
    'superior vena cava':       (115, 176, 130),
    'inferior vena cava':       (  0, 151, 206),
    'heart':                    (206, 110,  84),
    'spleen':                   (157, 108, 162),
    'liver':                    (221, 130, 101),
    'stomach':                  (216, 132, 105),
    'esophagus':                (211, 171, 143),
    'trachea':                  (182, 228, 255),
    'portal/splenic vein':      (  0, 151, 206),
}

_VERTEBRA_COLOR = (226, 202, 134)   # Jaune os pour les vertèbres


def _color_for_structure(name: str, index: int) -> tuple:
    """Retourne (R,G,B) pour une structure anatomique."""
    key = name.lower().strip()
    if key in _MEDICAL_COLORS:
        return _MEDICAL_COLORS[key]
    # Vertèbres : T6, T7, L1 … pattern
    if 'vertebra' in key or 'vertebr' in key:
        return _VERTEBRA_COLOR
    # Poumons
    if 'lung' in key:
        return (172, 138, 115)
    return _SEG_PALETTE[index % len(_SEG_PALETTE)]


class FinalOverlayPanel(QWidget):
    """
    Onglet intégré : vue fluoroscopie 2D avec segmentations 3D
    projetées et recalées superposées (contours + remplissage).
    """

    RENDER_MODES = ['Contours + Remplissage', 'Contours seuls', 'Remplissage seul']

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fluoro = None
        self._proj_masks = {}
        self._seg_volumes = {}
        self._ct_affine = None
        self._view_lao = 0.0
        self._view_cran = 0.0
        self._view_table = 0.0
        self._view_fov_mm = None
        self._result = None
        self._reg_size = 512
        self._mode = 0
        self._alpha = 0.35
        self._lw = 2
        self._show_labels = True
        self._vis = {}
        self._colors = {}
        self._chks = {}
        self._full_image = None
        self._zoom = 1.0
        self._pan_x = self._pan_y = 0
        self._pan_active = False
        self._pan_start = None
        self._build_ui()

    # ── Construction UI ───────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(5)

        # ── Barre de contrôle supérieure ──────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        self._cmb = QComboBox()
        for m in self.RENDER_MODES:
            self._cmb.addItem(m)
        self._cmb.setFixedWidth(200)
        self._cmb.currentIndexChanged.connect(self._on_mode)
        ctrl.addWidget(self._cmb)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.VLine); sep1.setObjectName('sep')
        ctrl.addWidget(sep1)

        lbl_a = QLabel('Opacité :'); lbl_a.setObjectName('dim')
        ctrl.addWidget(lbl_a)
        self._sl_alpha = QSlider(Qt.Horizontal)
        self._sl_alpha.setRange(5, 80); self._sl_alpha.setValue(35)
        self._sl_alpha.setFixedWidth(110)
        self._sl_alpha.valueChanged.connect(self._on_alpha)
        ctrl.addWidget(self._sl_alpha)
        self._lbl_alpha = QLabel('35 %'); self._lbl_alpha.setObjectName('dim'); self._lbl_alpha.setFixedWidth(35)
        ctrl.addWidget(self._lbl_alpha)

        lbl_c = QLabel('Contour :'); lbl_c.setObjectName('dim')
        ctrl.addWidget(lbl_c)
        self._sl_lw = QSlider(Qt.Horizontal)
        self._sl_lw.setRange(1, 6); self._sl_lw.setValue(2)
        self._sl_lw.setFixedWidth(70)
        self._sl_lw.valueChanged.connect(self._on_lw)
        ctrl.addWidget(self._sl_lw)

        self._chk_labels = QCheckBox('Labels')
        self._chk_labels.setChecked(True)
        self._chk_labels.toggled.connect(lambda _: self._render())
        ctrl.addWidget(self._chk_labels)

        ctrl.addStretch()

        btn_export = QPushButton('Exporter PNG')
        btn_export.clicked.connect(self._export)
        ctrl.addWidget(btn_export)

        btn_3d = QPushButton('Vue 3D')
        btn_3d.clicked.connect(self._open_3d_view)
        ctrl.addWidget(btn_3d)

        root.addLayout(ctrl)

        # ── Zone principale : image + panneau structures ──────────────────────
        body = QHBoxLayout()
        body.setSpacing(8)

        # Image
        self._lbl_img = QLabel()
        self._lbl_img.setAlignment(Qt.AlignCenter)
        self._lbl_img.setMinimumSize(400, 400)
        self._lbl_img.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._lbl_img.setStyleSheet(f'background:{DARK_BG};border-radius:4px;')
        self._lbl_img.wheelEvent = self._on_wheel
        self._lbl_img.mousePressEvent = self._on_press
        self._lbl_img.mouseMoveEvent = self._on_move
        self._lbl_img.mouseReleaseEvent = self._on_release
        body.addWidget(self._lbl_img, 3)

        # Panneau structures
        sp = QWidget()
        sp.setFixedWidth(230)
        sp.setStyleSheet(f'background:{PANEL_BG};border:1px solid {BORDER};border-radius:6px;')
        spl = QVBoxLayout(sp)
        spl.setContentsMargins(8, 8, 8, 8); spl.setSpacing(6)

        title = QLabel('STRUCTURES')
        title.setStyleSheet(f'color:{ACCENT};font-size:11px;font-weight:700;'
                            f'letter-spacing:1px;border:none;background:transparent;')
        spl.addWidget(title)

        hr = QHBoxLayout(); hr.setSpacing(4)
        ba = QPushButton('Toutes'); ba.setFixedHeight(24); ba.clicked.connect(self._show_all)
        bn = QPushButton('Aucune'); bn.setFixedHeight(24); bn.clicked.connect(self._hide_all)
        hr.addWidget(ba); hr.addWidget(bn)
        spl.addLayout(hr)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(f'QScrollArea{{background:transparent;border:none;}}')
        self._inner = QWidget(); self._inner.setStyleSheet('background:transparent;')
        self._inner_l = QVBoxLayout(self._inner)
        self._inner_l.setContentsMargins(0, 0, 0, 0); self._inner_l.setSpacing(3)
        self._inner_l.addStretch()
        self._scroll.setWidget(self._inner)
        spl.addWidget(self._scroll, 1)

        # Zoom
        zr = QHBoxLayout(); zr.setSpacing(4)
        zr.addWidget(QLabel('Zoom :'))
        self._sl_zoom = QSlider(Qt.Horizontal)
        self._sl_zoom.setRange(50, 400); self._sl_zoom.setValue(100)
        self._sl_zoom.valueChanged.connect(self._on_zoom_sl)
        zr.addWidget(self._sl_zoom)
        self._lbl_zoom = QLabel('100 %'); self._lbl_zoom.setFixedWidth(40); self._lbl_zoom.setObjectName('dim')
        zr.addWidget(self._lbl_zoom)
        spl.addLayout(zr)

        btn_rst = QPushButton('Réinitialiser vue')
        btn_rst.setFixedHeight(26)
        btn_rst.clicked.connect(self._reset_view)
        spl.addWidget(btn_rst)

        self._lbl_info = QLabel('En attente de données…')
        self._lbl_info.setObjectName('dim'); self._lbl_info.setWordWrap(True)
        spl.addWidget(self._lbl_info)

        body.addWidget(sp)
        root.addLayout(body, 1)

    # ── API publique ──────────────────────────────────────────────────────────
    def set_data(self, fluoro: np.ndarray, proj_masks: dict,
                 result: dict, reg_size: int,
                 seg_volumes: dict = None,
                 ct_affine: np.ndarray = None,
                 lao_deg: float = 0.0,
                 cran_deg: float = 0.0,
                 table_angle: float = 0.0,
                 fov_mm: float = None):
        """
        fluoro     : float32 [0,1] image fluoroscopie (résolution native)
        proj_masks : {nom_structure: masque 2D float32 à reg_size}
        result     : dict résultat du recalage (tx, ty, angle, scale, center…)
        reg_size   : résolution de travail du recalage [px]
        """
        self._fluoro = np.clip(fluoro, 0, 1).astype(np.float32)
        self._proj_masks = proj_masks or {}
        self._seg_volumes = seg_volumes or {}
        self._ct_affine = ct_affine
        self._view_lao = float(lao_deg)
        self._view_cran = float(cran_deg)
        self._view_table = float(table_angle)
        self._view_fov_mm = fov_mm
        self._result = result
        self._reg_size = reg_size
        self._colors = {}
        self._vis = {}
        for i, name in enumerate(self._proj_masks):
            self._colors[name] = _color_for_structure(name, i)
            self._vis[name] = True
        self._rebuild_list()
        self._zoom = 1.0; self._pan_x = self._pan_y = 0
        self._sl_zoom.blockSignals(True); self._sl_zoom.setValue(100); self._sl_zoom.blockSignals(False)
        self._render()

    def has_data(self):
        return self._fluoro is not None and self._result is not None and bool(self._proj_masks)

    # ── Reconstruction de la liste de structures ──────────────────────────────
    def _rebuild_list(self):
        while self._inner_l.count():
            w = self._inner_l.takeAt(0).widget()
            if w: w.deleteLater()
        self._chks = {}
        for name in self._proj_masks:
            r, g, b = self._colors.get(name, (200, 200, 200))
            row = QWidget(); row.setStyleSheet('background:transparent;')
            rl = QHBoxLayout(row); rl.setContentsMargins(2, 1, 2, 1); rl.setSpacing(4)
            chk = QCheckBox(); chk.setChecked(True)
            chk.toggled.connect(lambda _, n=name: self._toggle(n))
            self._chks[name] = chk
            rl.addWidget(chk)
            sw = QLabel(); sw.setFixedSize(12, 12)
            sw.setStyleSheet(f'background:rgb({r},{g},{b});border-radius:2px;border:none;')
            rl.addWidget(sw)
            display = name if len(name) <= 22 else name[:19] + '…'
            lbl = QLabel(display); lbl.setToolTip(name)
            lbl.setStyleSheet(f'color:{TEXT};font-size:10px;border:none;background:transparent;')
            rl.addWidget(lbl, 1)
            self._inner_l.addWidget(row)
        self._inner_l.addStretch()
        n = len(self._proj_masks)
        self._lbl_info.setText(f'{n} structure(s) projetée(s)')

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _toggle(self, name):
        self._vis[name] = self._chks[name].isChecked()
        self._render()

    def _show_all(self):
        for c in self._chks.values(): c.setChecked(True)

    def _hide_all(self):
        for c in self._chks.values(): c.setChecked(False)

    def _on_mode(self, idx):   self._mode = idx; self._render()
    def _on_alpha(self, v):    self._alpha = v / 100.0; self._lbl_alpha.setText(f'{v} %'); self._render()
    def _on_lw(self, v):       self._lw = v; self._render()

    def _on_zoom_sl(self, v):
        self._zoom = v / 100.0
        self._lbl_zoom.setText(f'{v} %')
        self._render()

    def _on_wheel(self, e):
        if e.angleDelta().y() > 0:
            self._zoom = min(self._zoom * 1.12, 4.0)
        else:
            self._zoom = max(self._zoom / 1.12, 0.5)
        self._sl_zoom.blockSignals(True)
        self._sl_zoom.setValue(int(self._zoom * 100))
        self._sl_zoom.blockSignals(False)
        self._lbl_zoom.setText(f'{int(self._zoom*100)} %')
        self._render()

    def _on_press(self, e):
        self._pan_active = True; self._pan_start = (e.x(), e.y())
    def _on_move(self, e):
        if not self._pan_active or not self._pan_start: return
        self._pan_x += (e.x() - self._pan_start[0]) * 0.5
        self._pan_y += (e.y() - self._pan_start[1]) * 0.5
        self._pan_start = (e.x(), e.y())
        self._render()
    def _on_release(self, e):
        self._pan_active = False

    def _reset_view(self):
        self._zoom = 1.0; self._pan_x = self._pan_y = 0
        self._sl_zoom.blockSignals(True); self._sl_zoom.setValue(100); self._sl_zoom.blockSignals(False)
        self._lbl_zoom.setText('100 %')
        self._render()

    # ── Rendu principal ───────────────────────────────────────────────────────
    def _render(self):
        if self._fluoro is None or self._result is None:
            self._lbl_img.clear(); return

        fl = self._fluoro
        S = fl.shape[0]
        reg_s = self._reg_size

        fl_u8 = (fl * 255).astype(np.uint8)
        rgb = cv2.cvtColor(fl_u8, cv2.COLOR_GRAY2RGB).astype(np.float32)

        self._show_labels = self._chk_labels.isChecked()

        for name, mask in self._proj_masks.items():
            if not self._vis.get(name, True):
                continue

            m = mask.astype(np.float32)
            if m.shape[:2] != (reg_s, reg_s):
                m = cv2.resize(m, (reg_s, reg_s), interpolation=cv2.INTER_LINEAR)

            warped = apply_full_transform(m, self._result)
            if warped.shape[:2] != (S, S):
                warped = cv2.resize(warped, (S, S), interpolation=cv2.INTER_LINEAR)

            binary = (warped > 0.3).astype(np.uint8)
            if binary.sum() == 0:
                continue

            r, g, b = self._colors.get(name, (200, 200, 200))
            alpha = self._alpha

            # Remplissage
            if self._mode in (0, 2):
                ov = rgb.copy(); ov[binary > 0] = [r, g, b]
                rgb = cv2.addWeighted(rgb, 1 - alpha, ov, alpha, 0)

            # Contours
            if self._mode in (0, 1):
                cnts, _ = cv2.findContours(
                    binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
                cv2.drawContours(rgb_u8, cnts, -1, (r, g, b),
                                 self._lw, lineType=cv2.LINE_AA)
                rgb = rgb_u8.astype(np.float32)

            # Labels
            if self._show_labels and binary.sum() > 100:
                ys, xs = np.where(binary > 0)
                cx, cy = int(xs.mean()), int(ys.min()) - 10
                cy = max(16, cy)
                rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
                txt = name if len(name) <= 24 else name[:21] + '…'
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
                lx = max(0, min(cx - tw // 2, S - tw - 4))
                ly = max(th + 4, cy)
                cv2.rectangle(rgb_u8,
                              (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3),
                              (int(r * 0.3), int(g * 0.3), int(b * 0.3)), -1)
                cv2.rectangle(rgb_u8,
                              (lx - 3, ly - th - 3), (lx + tw + 3, ly + 3),
                              (r, g, b), 1)
                cv2.putText(rgb_u8, txt, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (r, g, b), 1, cv2.LINE_AA)
                rgb = rgb_u8.astype(np.float32)

        self._full_image = np.clip(rgb, 0, 255).astype(np.uint8)

        # Zoom et pan
        h, w = self._full_image.shape[:2]
        if abs(self._zoom - 1.0) > 0.01 or abs(self._pan_x) > 0.5 or abs(self._pan_y) > 0.5:
            M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), 0, self._zoom)
            M[0, 2] += self._pan_x; M[1, 2] += self._pan_y
            display = cv2.warpAffine(self._full_image, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(12, 14, 20))
        else:
            display = self._full_image

        lw = max(48, self._lbl_img.width()); lh = max(48, self._lbl_img.height())
        side = min(lw, lh)
        if side > 16:
            display = cv2.resize(display, (side, side), interpolation=cv2.INTER_LINEAR)
        display = np.ascontiguousarray(display)
        h2, w2 = display.shape[:2]
        qi = QImage(display.data, w2, h2, w2 * 3, QImage.Format_RGB888)
        self._lbl_img.setPixmap(QPixmap.fromImage(qi).copy())

    def _export(self):
        if self._full_image is None: return
        p, _ = QFileDialog.getSaveFileName(
            self, 'Exporter overlay', 'overlay_final.png',
            'PNG (*.png);;JPEG (*.jpg);;TIFF (*.tiff)')
        if not p: return
        bgr = cv2.cvtColor(self._full_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(p, bgr)
        QMessageBox.information(self, 'Export', f'Sauvegardé :\n{p}')

    def _apply_registration_to_points(self, pts_xy: np.ndarray) -> np.ndarray:
        """Apply rigid + elastic registration to 2D points in reg_size coordinates."""
        if pts_xy.size == 0 or self._result is None:
            return pts_xy

        reg = self._result
        tx = float(reg.get('tx', 0.0))
        ty = float(reg.get('ty', 0.0))
        angle = float(reg.get('angle', 0.0))
        scale = float(reg.get('scale', 1.0))
        center = reg.get('center', (self._reg_size * 0.5, self._reg_size * 0.5))
        cx, cy = float(center[0]), float(center[1])

        mat = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        mat[0, 2] += tx
        mat[1, 2] += ty

        pts_h = np.column_stack([
            pts_xy.astype(np.float32),
            np.ones((pts_xy.shape[0],), dtype=np.float32),
        ])
        out = (mat @ pts_h.T).T.astype(np.float32)

        if reg.get('elastic') and ('disp_x' in reg) and ('disp_y' in reg):
            dx = reg['disp_x'].astype(np.float32)
            dy = reg['disp_y'].astype(np.float32)
            if dx.shape != (self._reg_size, self._reg_size):
                dx = cv2.resize(dx, (self._reg_size, self._reg_size), interpolation=cv2.INTER_LINEAR)
                dy = cv2.resize(dy, (self._reg_size, self._reg_size), interpolation=cv2.INTER_LINEAR)

            def _sample_bilinear(field: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
                """Bilinear sampling with constant-zero border for arbitrary point count."""
                h, w = field.shape
                x0 = np.floor(xq).astype(np.int32)
                y0 = np.floor(yq).astype(np.int32)
                x1 = x0 + 1
                y1 = y0 + 1

                wx = xq - x0.astype(np.float32)
                wy = yq - y0.astype(np.float32)

                def _gather(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
                    vals = np.zeros((xq.shape[0],), dtype=np.float32)
                    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
                    if np.any(valid):
                        vals[valid] = field[iy[valid], ix[valid]]
                    return vals

                v00 = _gather(x0, y0)
                v10 = _gather(x1, y0)
                v01 = _gather(x0, y1)
                v11 = _gather(x1, y1)

                return (
                    v00 * (1.0 - wx) * (1.0 - wy)
                    + v10 * wx * (1.0 - wy)
                    + v01 * (1.0 - wx) * wy
                    + v11 * wx * wy
                ).astype(np.float32)

            sample_dx = _sample_bilinear(dx, out[:, 0], out[:, 1])
            sample_dy = _sample_bilinear(dy, out[:, 0], out[:, 1])

            # apply_full_transform uses inverse remap; forward point approximation is -disp
            out[:, 0] -= sample_dx
            out[:, 1] -= sample_dy

        return out

    def _fov_scale_for_mask(self, mask_3d: np.ndarray) -> float:
        """Return the same in-plane scale correction used by project_mask_3d."""
        fov = self._view_fov_mm
        if fov is None or float(fov) <= 0 or self._ct_affine is None:
            return 1.0
        try:
            vx = float(abs(self._ct_affine[0, 0]))
            vz = float(abs(self._ct_affine[2, 2]))
            nx = int(mask_3d.shape[0])
            nz = int(mask_3d.shape[2])
            ct_span_mm = max(nx * vx, nz * vz, 1e-6)
            return float(np.clip(ct_span_mm / float(fov), 0.25, 4.0))
        except Exception:
            return 1.0

    def _volume_to_registered_mesh(self, pv, mask_name: str,
                                   mask_3d: np.ndarray, side: int):
        """Build an anatomical 3D mesh from NIfTI segmentation and register its projection."""
        if mask_3d is None:
            return None

        vol = (mask_3d > 0).astype(np.float32)
        if vol.sum() == 0:
            return None

        # Adaptive downsample for interactive meshing on large volumes.
        stride = 1
        while (max(vol.shape) / stride) > 220:
            stride *= 2
        if stride > 1:
            vol = vol[::stride, ::stride, ::stride]

        from scipy.ndimage import rotate as nd_rotate

        if abs(self._view_lao) > 0.1:
            vol = nd_rotate(
                vol,
                -self._view_lao,
                axes=(0, 1),
                reshape=False,
                order=0,
                prefilter=False,
                mode='constant',
                cval=0.0,
            )
        if abs(self._view_cran) > 0.1:
            vol = nd_rotate(
                vol,
                self._view_cran,
                axes=(1, 2),
                reshape=False,
                order=0,
                prefilter=False,
                mode='constant',
                cval=0.0,
            )
        if abs(self._view_table) > 0.1:
            vol = nd_rotate(
                vol,
                self._view_table,
                axes=(0, 2),
                reshape=False,
                order=0,
                prefilter=False,
                mode='constant',
                cval=0.0,
            )

        if vol.sum() < 8:
            return None

        from skimage import measure

        try:
            verts, faces, _, _ = measure.marching_cubes(
                vol,
                level=0.5,
                step_size=1,
                allow_degenerate=False,
            )
        except Exception:
            return None

        if verts.shape[0] < 3 or faces.shape[0] == 0:
            return None

        nx, ny, nz = vol.shape
        reg_size = float(self._reg_size)

        # Projection to DRR in-plane coordinates (same convention as project_mask_3d).
        x_plane = verts[:, 0] * ((reg_size - 1.0) / max(nx - 1, 1))
        y_plane = verts[:, 2] * ((reg_size - 1.0) / max(nz - 1, 1))

        fov_scale = self._fov_scale_for_mask(mask_3d)
        if abs(fov_scale - 1.0) > 0.02:
            c = reg_size * 0.5
            x_plane = c + (x_plane - c) * fov_scale
            y_plane = c + (y_plane - c) * fov_scale

        pts_plane = np.column_stack([x_plane, y_plane]).astype(np.float32)
        pts_plane = self._apply_registration_to_points(pts_plane)

        # Overlay panel displays on fluoroscopy native size.
        if side != self._reg_size:
            sf = float(side) / float(self._reg_size)
            pts_plane *= sf

        # Use AP axis (axis 1) as real depth axis.
        vy = 1.0
        if self._ct_affine is not None:
            try:
                vy = float(abs(self._ct_affine[1, 1])) * float(stride)
            except Exception:
                vy = 1.0
        depth = (verts[:, 1] - ((ny - 1.0) * 0.5)) * vy
        
        # Determine exact XYZ scaling to prevent flattening the 3D model
        # The 2D projection scaled X by (side / nx), we match that depth-wise
        depth_scale = float(side) / max(nx, 1) if nx > 0 else 1.0
        z_world = 10.0 + depth * depth_scale

        points_world = np.column_stack([
            pts_plane[:, 0],
            (side - 1.0) - pts_plane[:, 1],
            z_world,
        ]).astype(np.float32)

        faces_vtk = np.hstack([
            np.full((faces.shape[0], 1), 3, dtype=np.int64),
            faces.astype(np.int64),
        ]).ravel()
        mesh = pv.PolyData(points_world, faces_vtk)
        mesh = mesh.clean(tolerance=0.0)

        if mesh.n_points > 250000:
            try:
                mesh = mesh.decimate_pro(0.65, preserve_topology=True)
            except Exception:
                pass

        return mesh

    def _open_3d_view(self):
        """Open anatomical 3D meshes from NIfTI segmentations over fluoroscopy."""
        if self._fluoro is None or self._result is None or not self._proj_masks:
            QMessageBox.information(
                self,
                'Vue 3D',
                'Lancez un recalage avec segmentations avant d\'ouvrir la vue 3D.',
            )
            return

        if not self._seg_volumes:
            QMessageBox.information(
                self,
                'Vue 3D',
                'Aucune segmentation 3D disponible.\n'
                'Chargez un fichier NIfTI de segmentation pour obtenir des maillages anatomiques.',
            )
            return

        try:
            import pyvista as pv
        except Exception as ex:
            QMessageBox.warning(
                self,
                'Vue 3D',
                f'PyVista est indisponible : {ex}\nInstallez via: pip install pyvista',
            )
            return

        fluoro = np.clip(self._fluoro, 0, 1).astype(np.float32)
        side = int(max(fluoro.shape[:2]))
        fluoro_u8 = (fluoro * 255).astype(np.uint8)
        if fluoro_u8.shape[:2] != (side, side):
            fluoro_u8 = cv2.resize(fluoro_u8, (side, side), interpolation=cv2.INTER_LINEAR)
        fluoro_rgb = cv2.cvtColor(fluoro_u8, cv2.COLOR_GRAY2RGB)

        texture = pv.numpy_to_texture(np.ascontiguousarray(fluoro_rgb))
        plotter = pv.Plotter(window_size=(1280, 900))
        plotter.set_background("#12151f")

        plane = pv.Plane(
            center=(side * 0.5, side * 0.5, 0.0),
            direction=(0.0, 0.0, 1.0),
            i_size=float(side),
            j_size=float(side),
            i_resolution=1,
            j_resolution=1,
        )
        plane.texture_map_to_plane(inplace=True)
        plotter.add_mesh(plane, texture=texture, name='fluoro_plane', lighting=False)

        added_meshes = 0

        for idx, (name, _) in enumerate(self._proj_masks.items()):
            if not self._vis.get(name, True):
                continue

            vol = self._seg_volumes.get(name)
            if vol is None:
                continue

            mesh = self._volume_to_registered_mesh(pv, name, vol, side)
            if mesh is None or mesh.n_points < 3:
                continue

            color = tuple(c / 255.0 for c in self._colors.get(name, (200, 200, 200)))

            plotter.add_mesh(
                mesh,
                color=color,
                opacity=min(0.9, max(0.2, self._alpha + 0.15)),
                smooth_shading=True,
                name=f'{name}_{idx}',
            )
            try:
                edges = mesh.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=False,
                    manifold_edges=False,
                    non_manifold_edges=False,
                )
                if edges.n_points > 0:
                    plotter.add_mesh(
                        edges,
                        color=color,
                        line_width=max(1.2, self._lw),
                        opacity=0.95,
                    )
            except Exception:
                pass

            added_meshes += 1

        if added_meshes == 0:
            plotter.close()
            QMessageBox.information(
                self,
                'Vue 3D',
                'Aucun maillage anatomique n\'a pu etre reconstruit.\n'
                'Verifiez que les structures NIfTI chargees correspondent aux structures visibles.',
            )
            return

        plotter.enable_parallel_projection()
        plotter.add_text('Maillages anatomiques 3D recales sur fluoroscopie', font_size=10)
        plotter.add_axes(line_width=2)
        plotter.camera_position = [
            (side * 0.5, side * 0.5, side * 1.7),
            (side * 0.5, side * 0.5, 0.0),
            (0.0, -1.0, 0.0),
        ]
        plotter.show(title='Vue 3D - Overlay recale', auto_close=True)

    def resizeEvent(self, e):
        super().resizeEvent(e); self._render()


# ══════════════════════════════════════════════════════════════════════════════
# Worker thread
# ══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────

