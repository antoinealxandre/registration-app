"""Reusable annotation widgets and visualization panels."""

import math
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QSlider,
    QComboBox, QGridLayout, QSizePolicy, QCheckBox, QFrame, QFileDialog, QMessageBox,
    QScrollArea, QProgressBar, QGraphicsBlurEffect, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QEvent, QRectF
from PyQt5.QtGui import QImage, QPixmap, QCursor, QPainter, QIcon, QColor

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
    image_delete_requested = pyqtSignal(int)

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
        self._btn_delete = QPushButton()
        self._btn_delete.setObjectName('tool')
        self._btn_delete.setFixedSize(22, 22)
        self._btn_delete.setToolTip('Retirer cette image')
        trash_icon = _make_svg_icon(
            'M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12z'
            'M19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z',
            color='#ef6666',
            size=16,
        )
        if trash_icon:
            self._btn_delete.setIcon(trash_icon)
            self._btn_delete.setIconSize(QSize(16, 16))
        else:
            self._btn_delete.setText('x')
        self._btn_delete.setStyleSheet(
            f'QPushButton{{font-size:10px;padding:2px;border-radius:4px;'
            f'background:{CARD_BG};border:1px solid {BORDER2};color:#ef6666;min-height:20px;}}'
            f'QPushButton:hover{{border-color:#ef6666;background:#2a0a0a;}}'
        )
        self._btn_delete.clicked.connect(self._on_delete)
        br.addStretch()
        br.addWidget(self._btn_delete)
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

    def set_array(self, array):
        self.array = array
        self._update_thumb()

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

    def _on_delete(self):
        self.image_delete_requested.emit(self.index)

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
# Overlay de chargement
# ══════════════════════════════════════════════════════════════════════════════

class BusyOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_widget = None
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        # Fond transparent, le flou est rendu à partir du contenu capturé
        self.setStyleSheet('background:transparent;')
        self.hide()

        self._backdrop = QLabel(self)
        self._backdrop.setScaledContents(True)
        self._backdrop.setStyleSheet('background:rgba(8, 12, 22, 18);')

        self._panel = QFrame(self)
        self._panel.setFixedSize(420, 140)
        self._panel.setStyleSheet(
            'QFrame{'
            'border-radius:14px;'
            'border:1px solid #1e2235;'
            'background:#12151f;'
            '}'
        )
        shadow = QGraphicsDropShadowEffect(self._panel)
        shadow.setBlurRadius(40)
        shadow.setOffset(0, 12)
        shadow.setColor(QColor(0, 0, 0, 140))
        self._panel.setGraphicsEffect(shadow)
        panel_l = QVBoxLayout(self._panel)
        panel_l.setContentsMargins(18, 12, 18, 12)
        panel_l.setSpacing(4)

        self._title = QLabel('CHARGEMENT')
        self._title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._title.setStyleSheet(
            'background:transparent;color:#8fa5cc;font-size:9px;font-weight:700;letter-spacing:0.8px;'
        )
        self._title.setFixedHeight(18)
        panel_l.addWidget(self._title)

        self._message = QLabel('Preparation...')
        self._message.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._message.setWordWrap(False)
        self._message.setStyleSheet(
            'background:transparent;color:#f0f4fb;font-size:14px;font-weight:500;'
        )
        self._message.setFixedHeight(22)
        panel_l.addWidget(self._message)

        self._progress = QProgressBar(self._panel)
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(7)
        self._progress.setStyleSheet(
            'QProgressBar{'
            'background:rgba(255,255,255,0.04);'
            'border:none;'
            'border-radius:3px;'
            'padding:0px;'
            'margin:0px;'
            '}'
            'QProgressBar::chunk{'
            'border-radius:3px;'
            'background:#2ecc7a;'
            'margin:0px;'
            '}'
        )
        panel_l.addWidget(self._progress)

    def show_busy(self, title: str, message: str = '', progress: int = 0, snapshot_widget=None):
        self._source_widget = snapshot_widget or self.parentWidget()
        self._title.setText(str(title or 'Chargement').upper())
        self._message.setText(str(message or 'Preparation...'))
        self._progress.setValue(int(max(0, min(100, progress))))
        self._sync_geometry()
        self.show()
        self.raise_()

    def update_progress(self, progress: int, message: str = None):
        self._progress.setValue(int(max(0, min(100, progress))))
        if message is not None:
            self._message.setText(str(message))

    def hide_busy(self):
        self._source_widget = None
        self.hide()

    def _sync_geometry(self):
        target = self._source_widget or self.parentWidget()
        if target is None:
            return
        rect = target.rect() if target is self.parentWidget() else target.geometry()
        self.setGeometry(rect)
        self._backdrop.setGeometry(self.rect())
        self._refresh_backdrop(target)
        self._center_panel()

    def _refresh_backdrop(self, target_widget):
        pixmap = self._blur_pixmap(target_widget)
        if pixmap.isNull():
            self._backdrop.clear()
        else:
            self._backdrop.setPixmap(pixmap)

    def _blur_pixmap(self, source_widget):
        if source_widget is None:
            return QPixmap()
        source = source_widget.grab()
        if source.isNull():
            return QPixmap()

        scene = QGraphicsScene()
        scene.setSceneRect(QRectF(source.rect()))
        item = QGraphicsPixmapItem()
        item.setPixmap(source)
        blur = QGraphicsBlurEffect()
        blur.setBlurRadius(22)
        item.setGraphicsEffect(blur)
        scene.addItem(item)

        result = QPixmap(source.size())
        result.fill(Qt.transparent)
        painter = QPainter(result)
        try:
            scene.render(painter, QRectF(result.rect()), QRectF(source.rect()))
        finally:
            painter.end()
        return result

    def _center_panel(self):
        px = (self.width() - self._panel.width()) // 2
        py = (self.height() - self._panel.height()) // 2
        self._panel.move(max(0, px), max(0, py))
        self._panel.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._backdrop.setGeometry(self.rect())
        self._center_panel()
        if self.isVisible() and self._source_widget is not None:
            self._refresh_backdrop(self._source_widget)


# ══════════════════════════════════════════════════════════════════════════════
# Canvas d'annotation
# ══════════════════════════════════════════════════════════════════════════════

class AnnotationCanvas(QLabel):
    mask_updated = pyqtSignal()
    wheel_scrolled = pyqtSignal(int)
    stent_pose_changed = pyqtSignal(float, float, float)
    point_picked = pyqtSignal(str, float, float)

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
        self._overlay_mask = None
        self._overlay_color = (255, 200, 80)
        self._overlay_alpha = 0.35
        self._overlay_lw = 2
        self._stent_mode_active = False
        self._stent_center = None
        self._stent_axis_deg = 0.0
        self._stent_axis_len = None
        self._stent_drag_mode = None
        self._stent_drag_offset = None
        self._stent_handle_radius = 12
        self._tool_before_stent = None
        self._pick_mode = None              # nom du point en attente de capture
        self._point_markers = {}            # name -> {'xy', 'color', 'label'}

    # ── API ───────────────────────────────────────────────────────────────────

    def image_size(self) -> int:
        return int(self._size)

    def set_image(self, img: np.ndarray, preserve_masks: bool = False):
        if img is None:
            self._img_np = None
            self._size = 512
            self._masks = {k: np.zeros((self._size, self._size), np.float32) for k in STRUCT}
            self._history = {k: [] for k in STRUCT}
            self._raw_pts = []
            self._poly_pts = []
            self._rect_start = None
            self._rect_cur = None
            self._cursor_pos = None
            self.clear()
            return
        old_masks = None
        old_history = None
        if preserve_masks and self._img_np is not None:
            old_masks = {
                k: (self._masks[k].copy() if self._masks.get(k) is not None else None)
                for k in STRUCT
            }
            old_history = {
                k: [m.copy() for m in self._history.get(k, [])]
                for k in STRUCT
            }
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
            if old_masks and old_masks.get(k) is not None:
                mask = old_masks[k]
                if mask.shape != (s, s):
                    mask = cv2.resize(mask.astype(np.float32), (s, s), interpolation=cv2.INTER_NEAREST)
                    mask = (mask > 0.5).astype(np.float32)
                self._masks[k] = mask.astype(np.float32)
                self._history[k] = old_history.get(k, []) if old_history else []
            else:
                self._masks[k] = np.zeros((s,s), np.float32)
                self._history[k] = []
        self._raw_pts=[]; self._poly_pts=[]
        self._refresh()

    def set_overlay(self, mask: np.ndarray, color=(255, 200, 80), alpha: float = 0.35, line_width: int = 2):
        if mask is None:
            self._overlay_mask = None
        else:
            self._overlay_mask = mask.astype(np.float32)
        self._overlay_color = tuple(int(v) for v in color)
        self._overlay_alpha = float(alpha)
        self._overlay_lw = int(max(1, line_width))
        self._refresh()

    def clear_overlay(self):
        self._overlay_mask = None
        self._refresh()

    def set_stent_mode(self, active: bool):
        if self._stent_mode_active == active:
            return
        self._stent_mode_active = active
        if active:
            self._tool_before_stent = self._tool
            self.setCursor(QCursor(Qt.OpenHandCursor))
        else:
            if self._tool_before_stent:
                self.set_tool(self._tool_before_stent)
                self._tool_before_stent = None
            else:
                self.setCursor(QCursor(Qt.CrossCursor))
        self._refresh()

    def set_stent_pose(self, center_px=None, axis_deg=None):
        if center_px is not None:
            self._stent_center = (float(center_px[0]), float(center_px[1]))
        if axis_deg is not None:
            self._stent_axis_deg = float(axis_deg)
        self._refresh()

    def set_stent_axis_length_px(self, length_px):
        self._stent_axis_len = None if length_px is None else float(length_px)
        self._refresh()

    def stent_pose(self):
        return self._stent_center, float(self._stent_axis_deg)

    def arm_pick(self, name: str):
        """Arme la capture du prochain clic gauche pour le point ``name``."""
        self._pick_mode = name
        self.setCursor(QCursor(Qt.CrossCursor))

    def disarm_pick(self):
        self._pick_mode = None
        if not self._stent_mode_active:
            self.setCursor(QCursor(Qt.BlankCursor if self._tool == 'eraser' else Qt.CrossCursor))

    def set_point_marker(self, name: str, xy, color=(80, 200, 255), label=None):
        """Affiche/efface un marqueur ponctuel (ex: cusp NCC) sur l'image."""
        if xy is None:
            self._point_markers.pop(name, None)
        else:
            self._point_markers[name] = {
                'xy': (float(xy[0]), float(xy[1])),
                'color': tuple(int(c) for c in color),
                'label': str(label or name),
            }
        self._refresh()

    def clear_point_markers(self):
        self._point_markers = {}
        self._refresh()

    def get_point_marker(self, name: str):
        m = self._point_markers.get(name)
        return None if m is None else m['xy']

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

    def wheelEvent(self, e):
        steps = int(e.angleDelta().y() / 120)
        if steps == 0 and e.angleDelta().y() != 0:
            steps = 1 if e.angleDelta().y() > 0 else -1
        if steps != 0:
            self.wheel_scrolled.emit(steps)
            e.accept()
            return
        super().wheelEvent(e)

    # ── Historique ────────────────────────────────────────────────────────────

    def _push_history(self):
        m=self._masks[self._active]
        if m is not None:
            h=self._history[self._active]; h.append(m.copy())
            if len(h)>30: h.pop(0)

    def _emit_stent_pose(self):
        if self._stent_center is None:
            return
        self.stent_pose_changed.emit(
            float(self._stent_center[0]),
            float(self._stent_center[1]),
            float(self._stent_axis_deg),
        )

    def _update_stent_axis_from_point(self, ix: int, iy: int) -> None:
        if self._stent_center is None:
            return
        dx = ix - self._stent_center[0]
        dy = iy - self._stent_center[1]
        if abs(dx) < 1e-3 and abs(dy) < 1e-3:
            return
        self._stent_axis_deg = math.degrees(math.atan2(-dy, dx))

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

    def _commit_polygon(self):
        if len(self._poly_pts) < 3:
            return
        self._draw_poly(self._poly_pts)
        self._poly_pts = []
        self._refresh()
        self.mask_updated.emit()

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

        if self._overlay_mask is not None and self._overlay_mask.sum() > 0:
            om = self._overlay_mask
            if om.shape != (self._size, self._size):
                om = cv2.resize(om.astype(np.float32), (self._size, self._size), interpolation=cv2.INTER_NEAREST)
            binary = (om > 0.5).astype(np.uint8)
            if binary.sum() > 0:
                or_, og, ob = self._overlay_color
                ov = base.copy()
                ov[binary > 0] = [or_, og, ob]
                base = cv2.addWeighted(base, 1 - self._overlay_alpha, ov, self._overlay_alpha, 0)
                cnts, _ = cv2.findContours(binary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(base, cnts, -1, (or_, og, ob), self._overlay_lw, lineType=cv2.LINE_AA)

        if self._stent_center is not None:
            cx = int(round(self._stent_center[0]))
            cy = int(round(self._stent_center[1]))
            axis_len = self._stent_axis_len
            if axis_len is None:
                axis_len = max(60, int(self._size * 0.2))
            half = int(round(axis_len * 0.5))
            ang = math.radians(self._stent_axis_deg)
            dx = int(round(math.cos(ang) * half))
            dy = int(round(-math.sin(ang) * half))
            p1 = (cx - dx, cy - dy)
            p2 = (cx + dx, cy + dy)
            cv2.line(base, p1, p2, (80, 200, 255), 2, lineType=cv2.LINE_AA)
            cv2.circle(base, (cx, cy), 6, (80, 200, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(base, (cx, cy), self._stent_handle_radius, (80, 200, 255), 1, lineType=cv2.LINE_AA)

        # ── Marqueurs ponctuels (cusp NCC, etc.) ────────────────────────────
        for nm, info in self._point_markers.items():
            x, y = int(round(info['xy'][0])), int(round(info['xy'][1]))
            col = info['color']
            cv2.drawMarker(base, (x, y), col, cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
            cv2.circle(base, (x, y), 7, col, 2, cv2.LINE_AA)
            lbl = info['label']
            if lbl:
                cv2.putText(base, lbl, (x + 10, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

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
                d0 = np.linalg.norm(np.array(self._cursor_pos, float) - np.array(self._poly_pts[0], float))
                hint_col = (60, 255, 80) if d0 < 14 and len(self._poly_pts) >= 3 else (215, 230, 240)
                cv2.circle(base, self._poly_pts[0], 8, hint_col, 2)
                if len(self._poly_pts) >= 3 and d0 < 14:
                    cv2.putText(base, 'Fermer ici', (self._poly_pts[0][0] + 12, self._poly_pts[0][1] + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, hint_col, 1)

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
        if self._pick_mode and e.button() == Qt.LeftButton:
            name = self._pick_mode
            self._pick_mode = None
            self.point_picked.emit(name, float(ix), float(iy))
            self._refresh()
            return
        if self._stent_mode_active:
            if e.button() != Qt.LeftButton:
                return
            if self._stent_center is None:
                self._stent_center = (ix, iy)
                self._stent_drag_mode = 'axis'
                self._update_stent_axis_from_point(ix, iy)
            else:
                dx = ix - self._stent_center[0]
                dy = iy - self._stent_center[1]
                if math.hypot(dx, dy) <= self._stent_handle_radius:
                    self._stent_drag_mode = 'center'
                    self._stent_drag_offset = (self._stent_center[0] - ix, self._stent_center[1] - iy)
                else:
                    self._stent_drag_mode = 'axis'
                    self._update_stent_axis_from_point(ix, iy)
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            self._emit_stent_pose()
            self._refresh()
            return
        if self._tool=='pencil':
            if e.button() != Qt.LeftButton:
                return
            self._drawing=True; self._raw_pts=[(ix,iy)]; self._refresh()
        elif self._tool=='polygon':
            if e.button() == Qt.RightButton:
                if self._poly_pts:
                    self._poly_pts.pop()
                    self._refresh()
                return
            if e.button() != Qt.LeftButton:
                return
            if (self._poly_pts and len(self._poly_pts) >= 3 and
                    np.linalg.norm(np.array((ix, iy), float) - np.array(self._poly_pts[0], float)) < 14):
                self._commit_polygon()
                return
            if not self._poly_pts or (ix, iy) != self._poly_pts[-1]:
                self._poly_pts.append((ix,iy))
            self._refresh()
        elif self._tool=='rectangle':
            if e.button() != Qt.LeftButton:
                return
            self._drawing=True; self._rect_start=self._rect_cur=(ix,iy)
        elif self._tool=='eraser':
            if e.button() != Qt.LeftButton:
                return
            self._drawing=True; self._push_history(); self._erase(ix,iy); self._refresh()

    def mouseMoveEvent(self,e):
        if self._img_np is None: return
        ix,iy=self._w2i(e.x(),e.y()); self._cursor_pos=(ix,iy)
        if self._stent_mode_active:
            if self._stent_drag_mode == 'center':
                off = self._stent_drag_offset or (0.0, 0.0)
                self._stent_center = (ix + off[0], iy + off[1])
                self._emit_stent_pose()
                self._refresh()
            elif self._stent_drag_mode == 'axis':
                self._update_stent_axis_from_point(ix, iy)
                self._emit_stent_pose()
                self._refresh()
            return
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
        if self._stent_mode_active:
            if self._stent_drag_mode is not None:
                self._stent_drag_mode = None
                self._stent_drag_offset = None
                self.setCursor(QCursor(Qt.OpenHandCursor))
                self._refresh()
            return
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
            self._commit_polygon()

    def resizeEvent(self,e): self._refresh()


# ══════════════════════════════════════════════════════════════════════════════
# Panneau de visualisation du recalage
# ══════════════════════════════════════════════════════════════════════════════

class SegmentationReviewPanel(QWidget):
    """Segmentation panel: axial/coronal/sagittal + external 3D window."""

    request_3d_view = pyqtSignal()
    view_clicked = pyqtSignal(str, float, float, int)   # plane, col, row, slice_idx

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ct_vol = None
        self._seg_masks = {}
        self._ax_idx = 0
        self._co_idx = 0
        self._sa_idx = 0
        self._alpha = 0.45
        self._lo = -1000.0
        self._hi = 1000.0
        self._struct_checks = {}
        self._struct_order = {}
        self._click_mode = None             # nom du point à capturer (ou None)
        self._markers = {}                  # name -> {'voxel': (x,y,z), 'color': rgb, 'label': str}
        self._view_fit = {}                 # plane -> {'x0','y0','nw','nh','ow','oh'}
        self._zoom = 1.0                    # facteur de zoom (CTRL+molette dans les vues)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(5)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        ctrl.addWidget(QLabel('Opacite :'))
        self._sl_alpha = QSlider(Qt.Horizontal)
        self._sl_alpha.setRange(5, 90)
        self._sl_alpha.setValue(45)
        self._sl_alpha.setFixedWidth(130)
        self._sl_alpha.valueChanged.connect(self._on_alpha)
        ctrl.addWidget(self._sl_alpha)
        self._lbl_alpha = QLabel('45 %')
        self._lbl_alpha.setObjectName('dim')
        self._lbl_alpha.setFixedWidth(44)
        ctrl.addWidget(self._lbl_alpha)

        sep = QFrame(); sep.setFrameShape(QFrame.VLine); sep.setObjectName('sep')
        ctrl.addWidget(sep)

        btn_open_3d = QPushButton('Ouvrir 3D')
        btn_open_3d.clicked.connect(self.request_3d_view.emit)
        ctrl.addWidget(btn_open_3d)

        ctrl.addStretch()
        root.addLayout(ctrl)

        row_sl = QHBoxLayout(); row_sl.setSpacing(8)

        row_sl.addWidget(QLabel('Axial'))
        self._sl_ax = QSlider(Qt.Horizontal)
        self._sl_ax.setRange(0, 0)
        self._sl_ax.valueChanged.connect(self._on_ax)
        row_sl.addWidget(self._sl_ax, 1)
        self._lbl_ax = QLabel('0/0'); self._lbl_ax.setObjectName('dim'); self._lbl_ax.setFixedWidth(60)
        row_sl.addWidget(self._lbl_ax)

        row_sl.addWidget(QLabel('Coronal'))
        self._sl_co = QSlider(Qt.Horizontal)
        self._sl_co.setRange(0, 0)
        self._sl_co.valueChanged.connect(self._on_co)
        row_sl.addWidget(self._sl_co, 1)
        self._lbl_co = QLabel('0/0'); self._lbl_co.setObjectName('dim'); self._lbl_co.setFixedWidth(60)
        row_sl.addWidget(self._lbl_co)

        row_sl.addWidget(QLabel('Sagittal'))
        self._sl_sa = QSlider(Qt.Horizontal)
        self._sl_sa.setRange(0, 0)
        self._sl_sa.valueChanged.connect(self._on_sa)
        row_sl.addWidget(self._sl_sa, 1)
        self._lbl_sa = QLabel('0/0'); self._lbl_sa.setObjectName('dim'); self._lbl_sa.setFixedWidth(60)
        row_sl.addWidget(self._lbl_sa)

        root.addLayout(row_sl)

        body = QHBoxLayout()
        body.setSpacing(8)

        views_wrap = QWidget()
        views_wrap.setStyleSheet('background:transparent;')
        views_row = QHBoxLayout(views_wrap)
        views_row.setContentsMargins(0, 0, 0, 0)
        views_row.setSpacing(6)

        self._ax_view = self._mk_view('Axial', 'axial')
        self._co_view = self._mk_view('Coronal', 'coronal')
        self._sa_view = self._mk_view('Sagittal', 'sagittal')
        views_row.addWidget(self._ax_view['box'], 1)
        views_row.addWidget(self._co_view['box'], 1)
        views_row.addWidget(self._sa_view['box'], 1)
        body.addWidget(views_wrap, 1)

        struct_frame = QFrame()
        struct_frame.setObjectName('structPanel')
        struct_frame.setMinimumWidth(280)
        struct_frame.setMaximumWidth(360)
        struct_frame.setStyleSheet(
            f'#structPanel{{border:1px solid {BORDER2};border-radius:6px;background:transparent;}}'
        )
        sf = QVBoxLayout(struct_frame)
        sf.setContentsMargins(8, 6, 8, 6)
        sf.setSpacing(4)

        sh = QHBoxLayout()
        sh.setSpacing(6)
        lbl_struct = QLabel('Structures visibles')
        lbl_struct.setObjectName('mid')
        sh.addWidget(lbl_struct)
        sh.addStretch()
        self._btn_struct_all = QPushButton('Tout')
        self._btn_struct_all.setObjectName('tool')
        self._btn_struct_all.setFixedHeight(24)
        self._btn_struct_all.clicked.connect(lambda: self._set_all_structures(True))
        sh.addWidget(self._btn_struct_all)
        self._btn_struct_none = QPushButton('Aucun')
        self._btn_struct_none.setObjectName('tool')
        self._btn_struct_none.setFixedHeight(24)
        self._btn_struct_none.clicked.connect(lambda: self._set_all_structures(False))
        sh.addWidget(self._btn_struct_none)
        sf.addLayout(sh)

        self._struct_scroll = QScrollArea()
        self._struct_scroll.setWidgetResizable(True)
        self._struct_scroll.setFrameShape(QFrame.NoFrame)
        self._struct_scroll.setStyleSheet('QScrollArea{background:transparent;border:none;}')

        self._struct_widget = QWidget()
        self._struct_layout = QVBoxLayout(self._struct_widget)
        self._struct_layout.setContentsMargins(0, 0, 0, 0)
        self._struct_layout.setSpacing(2)
        self._struct_scroll.setWidget(self._struct_widget)
        sf.addWidget(self._struct_scroll, 1)
        body.addWidget(struct_frame)

        root.addLayout(body, 1)

        self._lbl_info = QLabel('Chargez un CT et une segmentation. Molette: changer de coupe.')
        self._lbl_info.setObjectName('dim')
        root.addWidget(self._lbl_info)

    def _mk_view(self, title, plane):
        box = QWidget()
        box.setStyleSheet('background:transparent;')
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        lbl_title = QLabel(title)
        lbl_title.setObjectName('dim')
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet(f'background:{DARK_BG};border-radius:4px;')
        lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lbl.setToolTip('Molette souris: coupe precedente/suivante')
        lbl.setProperty('plane', plane)
        lbl.installEventFilter(self)
        lay.addWidget(lbl_title)
        lay.addWidget(lbl, 1)
        return {'box': box, 'label': lbl}

    def _clear_struct_widgets(self):
        while self._struct_layout.count() > 0:
            it = self._struct_layout.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()

    def _rebuild_struct_controls(self):
        self._clear_struct_widgets()
        self._struct_checks = {}
        self._struct_order = {}

        names = sorted(self._seg_masks.keys(), key=lambda x: str(x).lower())
        if not names:
            lbl = QLabel('Aucune structure disponible')
            lbl.setObjectName('dim')
            self._struct_layout.addWidget(lbl)
            self._struct_layout.addStretch(1)
            return

        for i, name in enumerate(names):
            row = QWidget()
            row.setStyleSheet('background:transparent;')
            hl = QHBoxLayout(row)
            hl.setContentsMargins(0, 0, 0, 0)
            hl.setSpacing(6)

            col = _color_for_structure(name, i)
            sw = QFrame()
            sw.setFixedSize(14, 14)
            sw.setStyleSheet(
                f'background: rgb({col[0]}, {col[1]}, {col[2]});'
                'border:1px solid #10131d;border-radius:3px;'
            )
            hl.addWidget(sw)

            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.toggled.connect(self._on_structs_changed)
            hl.addWidget(cb, 1)
            self._struct_layout.addWidget(row)
            self._struct_checks[name] = cb
            self._struct_order[name] = i

        self._struct_layout.addStretch(1)

    def _set_all_structures(self, checked):
        if not self._struct_checks:
            return
        for cb in self._struct_checks.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self._on_structs_changed(checked)

    def _selected_count(self):
        return sum(1 for cb in self._struct_checks.values() if cb.isChecked())

    def _on_structs_changed(self, _checked):
        total = len(self._struct_checks)
        if total:
            self._lbl_info.setText(f'{self._selected_count()}/{total} structure(s) affichee(s) - molette: changer de coupe')
        self._render_all()

    def set_data(self, ct_vol, seg_masks):
        self._ct_vol = ct_vol
        self._seg_masks = seg_masks or {}
        self._rebuild_struct_controls()

        if self._ct_vol is None or self._ct_vol.ndim != 3:
            self._sl_ax.setRange(0, 0)
            self._sl_co.setRange(0, 0)
            self._sl_sa.setRange(0, 0)
            self._ax_idx = self._co_idx = self._sa_idx = 0
            self._lbl_info.setText('CT absent: impossible d\'afficher la segmentation sur le scan.')
            self._render_all()
            return

        sx, sy, sz = [int(v) for v in self._ct_vol.shape]
        self._ax_idx = sz // 2
        self._co_idx = sy // 2
        self._sa_idx = sx // 2

        self._sl_ax.blockSignals(True)
        self._sl_ax.setRange(0, max(0, sz - 1))
        self._sl_ax.setValue(self._ax_idx)
        self._sl_ax.blockSignals(False)

        self._sl_co.blockSignals(True)
        self._sl_co.setRange(0, max(0, sy - 1))
        self._sl_co.setValue(self._co_idx)
        self._sl_co.blockSignals(False)

        self._sl_sa.blockSignals(True)
        self._sl_sa.setRange(0, max(0, sx - 1))
        self._sl_sa.setValue(self._sa_idx)
        self._sl_sa.blockSignals(False)

        try:
            self._lo, self._hi = np.percentile(self._ct_vol, [1, 99])
        except Exception:
            self._lo, self._hi = float(np.min(self._ct_vol)), float(np.max(self._ct_vol))
        if self._hi <= self._lo:
            self._hi = self._lo + 1.0

        total = len(self._struct_checks)
        self._lbl_info.setText(f'{total}/{total} structure(s) affichee(s) - molette: changer de coupe')
        self._render_all()

    def _on_ax(self, v):
        self._ax_idx = int(v)
        self._render_all()

    def _on_co(self, v):
        self._co_idx = int(v)
        self._render_all()

    def _on_sa(self, v):
        self._sa_idx = int(v)
        self._render_all()

    def _on_alpha(self, v):
        self._alpha = v / 100.0
        self._lbl_alpha.setText(f'{v} %')
        self._render_all()

    def _orient(self, sl, _plane):
        # Orientation fix: remove the vertical inversion that flipped anatomy upside-down.
        return np.rot90(sl, 1)

    def _iter_selected_masks(self):
        if not self._seg_masks:
            return []
        if not self._struct_checks:
            return list(self._seg_masks.items())
        out = []
        for name, mask in self._seg_masks.items():
            cb = self._struct_checks.get(name)
            if cb is None or cb.isChecked():
                out.append((name, mask))
        return out

    def _slice_of(self, vol, plane, idx):
        if plane == 'axial':
            return vol[:, :, idx] if idx < vol.shape[2] else None
        if plane == 'coronal':
            return vol[:, idx, :] if idx < vol.shape[1] else None
        return vol[idx, :, :] if idx < vol.shape[0] else None

    def _render_plane(self, plane):
        if self._ct_vol is None or self._ct_vol.ndim != 3:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        if plane == 'axial':
            idx = int(max(0, min(self._ax_idx, self._ct_vol.shape[2] - 1)))
            self._lbl_ax.setText(f'{idx + 1}/{self._ct_vol.shape[2]}')
        elif plane == 'coronal':
            idx = int(max(0, min(self._co_idx, self._ct_vol.shape[1] - 1)))
            self._lbl_co.setText(f'{idx + 1}/{self._ct_vol.shape[1]}')
        else:
            idx = int(max(0, min(self._sa_idx, self._ct_vol.shape[0] - 1)))
            self._lbl_sa.setText(f'{idx + 1}/{self._ct_vol.shape[0]}')

        sl = self._slice_of(self._ct_vol, plane, idx)
        if sl is None:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        sln = np.clip((sl.astype(np.float32) - self._lo) / (self._hi - self._lo), 0, 1)
        base = self._orient((sln * 255).astype(np.uint8), plane)
        rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)

        for i, (name, m3d) in enumerate(self._iter_selected_masks()):
            if m3d is None or getattr(m3d, 'ndim', 0) != 3:
                continue
            msl = self._slice_of(m3d, plane, idx)
            if msl is None:
                continue
            m = self._orient((msl > 0).astype(np.uint8), plane)
            if m.shape != base.shape:
                # Imported masks may be off by a few pixels versus CT shape.
                m = cv2.resize(m, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST)
                m = (m > 0).astype(np.uint8)
            if m.sum() == 0:
                continue

            color = _color_for_structure(name, self._struct_order.get(name, i))
            ov = rgb.copy()
            ov[m > 0] = color
            rgb = cv2.addWeighted(rgb, 1 - self._alpha, ov, self._alpha, 0)

            cnts, _ = cv2.findContours((m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cnts, -1, color, 1, lineType=cv2.LINE_AA)

        self._draw_markers_on(rgb, plane, idx)
        return np.ascontiguousarray(rgb)

    def _get_marker_data(self, plane, slice_idx):
        """Retourne les données de marqueurs (sans appliquer l'échelle de zoom)."""
        if not self._markers or self._ct_vol is None:
            return {}, {}
        from core.measurements import view_pixel_from_voxel
        vol_shape = self._ct_vol.shape

        # Calculer les pixel coords et ms_length une fois
        pix = {}
        for name, m in self._markers.items():
            col, row, native_idx = view_pixel_from_voxel(plane, m['voxel'], vol_shape)
            cx, cy = int(round(col)), int(round(row))
            pix[name] = {'xy': (cx, cy), 'native': native_idx,
                         'color': tuple(int(c) for c in m['color']),
                         'label': m['label']}

        # Pré-calculer les valeurs en mm (utilisées plus tard)
        ms_info = {}
        if 'hinge1' in pix and 'hinge2' in pix and 'ms' in pix:
            h1w = self._markers['hinge1'].get('world_mm')
            h2w = self._markers['hinge2'].get('world_mm')
            msw = self._markers['ms'].get('world_mm')
            if h1w is not None and h2w is not None and msw is not None:
                from core.measurements import ms_length_from_hinges
                mm = ms_length_from_hinges(h1w, h2w, msw)
                ms_info['mm'] = mm
                ms_info['hinge1_xy'] = pix['hinge1']['xy']
                ms_info['hinge2_xy'] = pix['hinge2']['xy']
                ms_info['ms_xy'] = pix['ms']['xy']

        return pix, ms_info

    def _draw_markers_on(self, rgb, plane, slice_idx):
        """Dessine les segments TAVI (ligne annulaire, MS) sur la coupe AVANT le zoom."""
        pix, ms_info = self._get_marker_data(plane, slice_idx)
        if not pix:
            return

        r_h, r_w = rgb.shape[:2]
        pix = {k: v for k, v in pix.items() if 0 <= v['xy'][0] < r_w and 0 <= v['xy'][1] < r_h}

        # ── Ligne annulaire (cyan) + segment MS perpendiculaire (rouge) ─
        if 'hinge1' in pix and 'hinge2' in pix:
            p1 = pix['hinge1']['xy']; p2 = pix['hinge2']['xy']
            cv2.line(rgb, p1, p2, (90, 220, 240), 1, cv2.LINE_AA)
            if 'ms' in pix and ms_info.get('mm') is not None:
                pm = pix['ms']['xy']
                p1v = np.asarray(p1, dtype=np.float64); p2v = np.asarray(p2, dtype=np.float64)
                pmv = np.asarray(pm, dtype=np.float64)
                axis = p2v - p1v
                n = float(np.linalg.norm(axis))
                if n > 1e-6:
                    u = axis / n
                    proj = p1v + np.dot(pmv - p1v, u) * u
                    proj_xy = (int(round(proj[0])), int(round(proj[1])))
                    cv2.arrowedLine(rgb, proj_xy, pm, (90, 90, 255), 2, cv2.LINE_AA, tipLength=0.18)

    def _draw_markers_overlay(self, canvas, plane, slice_idx, fit):
        """Dessine les marqueurs et labels APRÈS le zoom, à taille fixe (en pixels canvas)."""
        if fit is None or fit['nw'] <= 0 or fit['nh'] <= 0:
            return

        pix, ms_info = self._get_marker_data(plane, slice_idx)
        if not pix:
            return

        # Remapper les coordonnées d'origine à l'espace du canvas après zoom
        x0, y0, nw, nh = fit['x0'], fit['y0'], fit['nw'], fit['nh']
        ow, oh = fit['ow'], fit['oh']

        # ── Segment MS avec label en mm (à taille fixe) ────────────────────────
        if ms_info.get('mm') is not None and 'hinge1' in pix and 'hinge2' in pix and 'ms' in pix:
            p1 = np.asarray(pix['hinge1']['xy'], dtype=np.float64)
            p2 = np.asarray(pix['hinge2']['xy'], dtype=np.float64)
            pm = np.asarray(pix['ms']['xy'], dtype=np.float64)

            axis = p2 - p1
            n_axis = float(np.linalg.norm(axis))
            if n_axis > 1e-6:
                u = axis / n_axis
                proj = p1 + np.dot(pm - p1, u) * u

                # Transformer aux coords canvas
                proj_c = np.array([x0 + (proj[0] * nw) / ow, y0 + (proj[1] * nh) / oh])
                pm_c = np.array([x0 + (pm[0] * nw) / ow, y0 + (pm[1] * nh) / oh])

                proj_xy = tuple(int(round(c)) for c in proj_c)
                pm_xy = tuple(int(round(c)) for c in pm_c)

                # Vérifier les limites du canvas
                h_c, w_c = canvas.shape[:2]
                if (0 <= proj_xy[0] < w_c and 0 <= proj_xy[1] < h_c and
                    0 <= pm_xy[0] < w_c and 0 <= pm_xy[1] < h_c):
                    cv2.arrowedLine(canvas, proj_xy, pm_xy, (90, 90, 255), 2, cv2.LINE_AA, tipLength=0.18)
                    mid = ((proj_xy[0] + pm_xy[0]) // 2 + 6, (proj_xy[1] + pm_xy[1]) // 2)
                    if 0 <= mid[0] < w_c and 0 <= mid[1] < h_c:
                        cv2.putText(canvas, f'MS={ms_info["mm"]:.2f}mm', mid,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(canvas, f'MS={ms_info["mm"]:.2f}mm', mid,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 255), 1, cv2.LINE_AA)

        # ── Marqueurs à taille fixe (points + croix + halo + label) ────────────
        h_c, w_c = canvas.shape[:2]
        for name, info in pix.items():
            cx_orig, cy_orig = info['xy']
            native_idx = info['native']
            on_slice = abs(native_idx - slice_idx) <= 1

            # Transformer aux coords canvas (appliquer zoom via fit)
            cx_c = x0 + (cx_orig * nw) / ow
            cy_c = y0 + (cy_orig * nh) / oh

            if not (0 <= cx_c < w_c and 0 <= cy_c < h_c):
                continue

            cx_c = int(round(cx_c))
            cy_c = int(round(cy_c))
            color = info['color']

            if on_slice:
                cv2.circle(canvas, (cx_c, cy_c), 7, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.drawMarker(canvas, (cx_c, cy_c), color, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
                cv2.circle(canvas, (cx_c, cy_c), 6, color, 2, cv2.LINE_AA)
                cv2.putText(canvas, info['label'], (cx_c + 8, cy_c - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(canvas, info['label'], (cx_c + 8, cy_c - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            else:
                overlay = canvas.copy()
                cv2.circle(overlay, (cx_c, cy_c), 4, color, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, dst=canvas)

    def _fit_to_label(self, rgb, lbl, plane=None, slice_idx=None):
        lh = max(96, lbl.height())
        lw = max(96, lbl.width())
        h, w = rgb.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((lh, lw, 3), dtype=np.uint8), None
        s = min(lw / float(w), lh / float(h)) * float(self._zoom)
        nw = max(1, int(round(w * s)))
        nh = max(1, int(round(h * s)))
        resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((lh, lw, 3), 16, dtype=np.uint8)
        y0 = (lh - nh) // 2
        x0 = (lw - nw) // 2
        # Letterbox si zoom < seuil ; recadrage centre si l'image depasse le label
        src_x0 = max(0, -x0); src_y0 = max(0, -y0)
        src_x1 = min(nw, lw - x0); src_y1 = min(nh, lh - y0)
        dst_x0 = max(0, x0); dst_y0 = max(0, y0)
        if src_x1 > src_x0 and src_y1 > src_y0:
            canvas[dst_y0:dst_y0 + (src_y1 - src_y0),
                   dst_x0:dst_x0 + (src_x1 - src_x0)] = resized[src_y0:src_y1, src_x0:src_x1]

        fit = {'x0': x0, 'y0': y0, 'nw': nw, 'nh': nh, 'ow': w, 'oh': h}

        # Dessiner les marqueurs à taille fixe APRÈS le zoom
        if plane is not None and slice_idx is not None:
            self._draw_markers_overlay(canvas, plane, slice_idx, fit)

        # Indicateur de zoom (coin sup. gauche) si zoom != 1
        if abs(self._zoom - 1.0) > 0.05:
            cv2.putText(canvas, f'{self._zoom:.1f}x', (8, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        return canvas, fit

    def _set_lbl_img(self, lbl, rgb, plane=None, slice_idx=None):
        img, fit = self._fit_to_label(rgb, lbl, plane=plane, slice_idx=slice_idx)
        if plane is not None and fit is not None:
            self._view_fit[plane] = fit
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, w * 3, QImage.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qimg).copy())

    def _render_all(self):
        if self._ct_vol is None or self._ct_vol.ndim != 3:
            self._ax_view['label'].clear()
            self._co_view['label'].clear()
            self._sa_view['label'].clear()
            return

        self._set_lbl_img(self._ax_view['label'], self._render_plane('axial'), 'axial', self._ax_idx)
        self._set_lbl_img(self._co_view['label'], self._render_plane('coronal'), 'coronal', self._co_idx)
        self._set_lbl_img(self._sa_view['label'], self._render_plane('sagittal'), 'sagittal', self._sa_idx)

    def _step_slider(self, slider, steps):
        new_val = max(slider.minimum(), min(slider.maximum(), slider.value() + int(steps)))
        if new_val != slider.value():
            slider.setValue(new_val)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and self._ct_vol is not None:
            delta = event.angleDelta().y()
            if delta == 0:
                return True
            # CTRL + molette = zoom sur la coupe (toutes vues partagent le facteur)
            if event.modifiers() & Qt.ControlModifier:
                factor = 1.18 if delta > 0 else 1.0 / 1.18
                self._zoom = max(0.5, min(8.0, self._zoom * factor))
                self._render_all()
                return True
            steps = int(delta / 120)
            if steps == 0:
                steps = 1 if delta > 0 else -1

            if obj is self._ax_view['label']:
                self._step_slider(self._sl_ax, steps)
                return True
            if obj is self._co_view['label']:
                self._step_slider(self._sl_co, steps)
                return True
            if obj is self._sa_view['label']:
                self._step_slider(self._sl_sa, steps)
                return True

        if (event.type() == QEvent.MouseButtonPress
                and self._click_mode is not None
                and event.button() == Qt.LeftButton
                and self._ct_vol is not None):
            plane = obj.property('plane')
            if plane in ('axial', 'coronal', 'sagittal'):
                fit = self._view_fit.get(plane)
                if fit is None:
                    return True
                px = event.pos().x() - fit['x0']
                py = event.pos().y() - fit['y0']
                if 0 <= px < fit['nw'] and 0 <= py < fit['nh'] and fit['nw'] > 0 and fit['nh'] > 0:
                    col = px * fit['ow'] / fit['nw']
                    row = py * fit['oh'] / fit['nh']
                    slice_idx = {'axial': self._ax_idx,
                                 'coronal': self._co_idx,
                                 'sagittal': self._sa_idx}[plane]
                    self.view_clicked.emit(plane, float(col), float(row), int(slice_idx))
                return True

        return super().eventFilter(obj, event)

    # ── API marqueurs ─────────────────────────────────────────────────────────
    def set_click_mode(self, mode):
        """Active la capture du prochain clic gauche pour ce mode (ou None pour arrêter)."""
        self._click_mode = mode
        cursor = Qt.CrossCursor if mode else Qt.ArrowCursor
        for v in (self._ax_view, self._co_view, self._sa_view):
            v['label'].setCursor(QCursor(cursor))

    def set_marker(self, name: str, voxel_xyz, color=(80, 200, 255), label=None, world_mm=None):
        if voxel_xyz is None:
            self._markers.pop(name, None)
        else:
            self._markers[name] = {
                'voxel': tuple(int(round(v)) for v in voxel_xyz),
                'color': tuple(int(c) for c in color),
                'label': str(label or name),
                'world_mm': None if world_mm is None else np.asarray(world_mm, dtype=np.float64),
            }
        self._render_all()

    def clear_markers(self):
        self._markers = {}
        self._render_all()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render_all()

class ResultPanel(QWidget):
    """
    Visualisation multi-modes de la superposition Fluoroscopie / DRR recalé.
    Modes : Fusion · Damier · Différence · Avant/Après · Cyan/Magenta
    """
    MODES = [
        ('Fusion',          'Opacité DRR',   50),
        ('Avant / Après',   'Position (%)',  50),
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

    def clear_data(self):
        self._img_a = None
        self._img_b = None
        self._contours = []
        self._lbl_img.clear()

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
            1: 'Glisser le curseur pour balayer Fluoro vs mobile',
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

        # ── Mode 1 : Avant / Après ────────────────────────────────────────────
        else:
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

def _color_for_structure(name: str, index: int = 0) -> tuple:
    """Couleur (R,G,B) stable d'une structure — delegue a theme.color_for_structure.

    L'argument ``index`` n'est conserve que pour compatibilite d'appel ; la
    couleur ne depend que du NOM, pour rester identique dans toutes les vues.
    """
    from ui.theme import color_for_structure
    return color_for_structure(name)


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
        # Données TAVI optionnelles affichées dans la vue 3D post-recalage
        self._tavi_stent_pose = None        # dict {center_px, axis_deg, length_mm, diameter_mm, pix_mm}
        self._tavi_ms_world = {}
        self._tavi_ref_voxel = {}           # {'hinge1','hinge2','ms'} -> (vx, vy, vz) en coords voxel CT
        self._tavi_ct_shape = None          # forme du volume CT (nx, ny, nz) pour la projection
        self._tavi_id_mm = None
        self._tavi_ms_length = None
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

    def clear_data(self):
        self._fluoro = None
        self._proj_masks = {}
        self._seg_volumes = {}
        self._ct_affine = None
        self._result = None
        self._full_image = None
        self._vis = {}
        self._colors = {}
        self._chks = {}
        self._tavi_stent_pose = None
        self._tavi_ms_world = {}
        self._tavi_ref_voxel = {}
        self._tavi_ct_shape = None
        self._tavi_id_mm = None
        self._tavi_ms_length = None
        self._rebuild_list()
        self._lbl_img.clear()
        self._lbl_info.setText('En attente de données…')

    def set_tavi_overlay(self, stent_fluoro=None, ms_world=None, ref_voxel=None,
                          ct_shape=None, id_mm=None, ms_length_mm=None):
        """Injecte les donnees TAVI a afficher dans la vue 3D recalee.

        stent_fluoro : dict {center_px, axis_deg, length_mm, diameter_mm, pix_mm} pose 2D fluoro.
        ms_world     : reperes CT-monde (informatif, non rendu en 2D ici).
        ref_voxel    : reperes en coordonnees voxel CT. Projetes localement via le
                       MEME pipeline que les meshes de segmentation pour s'aligner
                       parfaitement avec elles dans l'overlay 3D.
        ct_shape     : forme (nx, ny, nz) du volume CT, utile pour la projection.
        id_mm        : ID initial (distance ligne annulaire -> base du stent) ;
                       recalcule en live quand on bouge la profondeur en 3D.
        ms_length_mm : valeur affichee si hinge1/hinge2/MS valides.
        """
        self._tavi_stent_pose = stent_fluoro
        self._tavi_ms_world = dict(ms_world or {})
        self._tavi_ref_voxel = dict(ref_voxel or {})
        self._tavi_ct_shape = tuple(ct_shape) if ct_shape is not None else None
        self._tavi_id_mm = id_mm
        self._tavi_ms_length = ms_length_mm

    def _project_voxel_to_view(self, voxel_xyz, ct_shape, side):
        """Projette un voxel CT vers le pixel canvas overlay (= meme pipeline que les meshes).

        Pipeline identique a ``_volume_to_registered_mesh`` :
          1) voxel (vx, vy, vz) -> coords plan DRR (axis 0 -> x, axis 2 -> y),
          2) flip vertical (convention DRR),
          3) fov_scale autour du centre,
          4) recalage 2D (rigide + elastique),
          5) mise a l'echelle vers le canvas fluoro (side / reg_size).
        """
        if ct_shape is None or len(ct_shape) < 3:
            return None
        nx, _, nz = int(ct_shape[0]), int(ct_shape[1]), int(ct_shape[2])
        reg_size = float(self._reg_size)
        vx, vy, vz = float(voxel_xyz[0]), float(voxel_xyz[1]), float(voxel_xyz[2])
        x_plane = vx * ((reg_size - 1.0) / max(nx - 1, 1))
        y_plane = vz * ((reg_size - 1.0) / max(nz - 1, 1))
        y_plane = (reg_size - 1.0) - y_plane
        # fov_scale (utilise la pleine forme du CT, pas le mask downsample)
        fov = self._view_fov_mm
        if fov is not None and float(fov) > 0 and self._ct_affine is not None:
            try:
                vx_aff = float(abs(self._ct_affine[0, 0]))
                vz_aff = float(abs(self._ct_affine[2, 2]))
                ct_span_mm = max(nx * vx_aff, nz * vz_aff, 1e-6)
                fov_scale = float(np.clip(ct_span_mm / float(fov), 0.25, 4.0))
                if abs(fov_scale - 1.0) > 0.02:
                    c = reg_size * 0.5
                    x_plane = c + (x_plane - c) * fov_scale
                    y_plane = c + (y_plane - c) * fov_scale
            except Exception:
                pass
        pts = np.array([[x_plane, y_plane]], dtype=np.float32)
        pts = self._apply_registration_to_points(pts)
        sf = float(side) / float(reg_size) if reg_size > 0 else 1.0
        return float(pts[0, 0] * sf), float(pts[0, 1] * sf)

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

        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
        # Geometrie TAVI (ligne annulaire, MS, base stent, ID) reportee en 2D.
        self._draw_tavi_2d(rgb_u8, S)
        self._full_image = rgb_u8

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

    def _put_label_2d(self, img, text, pos, color):
        """Petit label avec contour noir, taille fixe, a la position (px) donnee."""
        x = int(round(pos[0])); y = int(round(pos[1]))
        H, W = img.shape[:2]
        x = max(2, min(W - 80, x)); y = max(12, min(H - 4, y))
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def _draw_tavi_2d(self, img, side):
        """Reporte la geometrie de la metrique MSID sur l'overlay 2D.

        Ligne annulaire (hinge1-hinge2), cote MS perpendiculaire, extremites du
        stent (points), base ventriculaire et cote ID perpendiculaire a la ligne
        annulaire — avec fleches, segments, points et labels en mm. Cela rend la
        geometrie du calcul verifiable a l'oeil, exactement comme en 3D.
        """
        refs = self._tavi_ref_voxel
        if not refs or self._tavi_ct_shape is None:
            return
        H, W = img.shape[:2]

        def ipt(p):
            return (int(round(float(p[0]))), int(round(float(p[1]))))

        def ok(p):
            return -4 * side < p[0] < 4 * side and -4 * side < p[1] < 4 * side

        proj = {}
        for k in ('hinge1', 'hinge2', 'ms'):
            if k in refs:
                pp = self._project_voxel_to_view(refs[k], self._tavi_ct_shape, side)
                if pp is not None and ok(pp):
                    proj[k] = np.array([pp[0], pp[1]], dtype=np.float64)

        h1 = proj.get('hinge1'); h2 = proj.get('hinge2'); ms = proj.get('ms')
        pix_mm = self._tavi_pix_mm()

        if h1 is not None and h2 is not None:
            cv2.line(img, ipt(h1), ipt(h2), (90, 220, 240), 2, cv2.LINE_AA)
            u = h2 - h1
            nu = float(np.linalg.norm(u))
            if nu > 1e-6:
                u = u / nu
                n = np.array([-u[1], u[0]])
                # ── MS : perpendiculaire annulus -> MS (valeur = px projetes * pix_mm) ──
                if ms is not None:
                    foot = h1 + np.dot(ms - h1, u) * u
                    if ok(foot):
                        cv2.arrowedLine(img, ipt(foot), ipt(ms), (255, 90, 90), 2,
                                        cv2.LINE_AA, tipLength=0.22)
                        ms_mm = (float(np.linalg.norm(ms - foot)) * pix_mm) if pix_mm else self._tavi_ms_length
                        if ms_mm is not None:
                            self._put_label_2d(img, f'MS={ms_mm:.1f}mm',
                                               0.5 * (foot + ms) + np.array([6, -6]), (255, 120, 120))
                    if np.dot(ms - 0.5 * (h1 + h2), n) < 0:
                        n = -n   # n pointe vers le ventricule (cote MS)
                # ── Stent : extremites + base ventriculaire + cote ID ──
                sp = self._tavi_stent_pose
                if sp and sp.get('center_px') and sp.get('pix_mm', 0) > 0:
                    import math as _m
                    cx, cy = sp['center_px']
                    halfpx = float(sp['length_mm']) * 0.5 / max(float(sp['pix_mm']), 1e-6)
                    a = _m.radians(float(sp['axis_deg']))
                    e1 = np.array([cx + _m.cos(a) * halfpx, cy - _m.sin(a) * halfpx])
                    e2 = np.array([cx - _m.cos(a) * halfpx, cy + _m.sin(a) * halfpx])
                    for e in (e1, e2):
                        if ok(e):
                            cv2.circle(img, ipt(e), 4, (250, 210, 60), -1, cv2.LINE_AA)
                    d1 = float(np.dot(e1 - h1, n)); d2 = float(np.dot(e2 - h1, n))
                    base = e1 if d1 >= d2 else e2
                    foot_id = h1 + np.dot(base - h1, u) * u
                    if ok(base) and ok(foot_id):
                        cv2.arrowedLine(img, ipt(foot_id), ipt(base), (255, 160, 50), 2,
                                        cv2.LINE_AA, tipLength=0.22)
                        id_mm = (float(np.linalg.norm(base - foot_id)) * pix_mm) if pix_mm else self._tavi_id_mm
                        if id_mm is not None:
                            self._put_label_2d(img, f'ID={id_mm:.1f}mm',
                                               0.5 * (foot_id + base) + np.array([6, 6]), (255, 180, 90))

        # Points de repere par-dessus
        for k, c in (('hinge1', (120, 210, 255)), ('hinge2', (255, 210, 120)),
                     ('ms', (255, 110, 110))):
            if k in proj and ok(proj[k]):
                cv2.circle(img, ipt(proj[k]), 6, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(img, ipt(proj[k]), 4, c, -1, cv2.LINE_AA)

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
        if vol.sum() < 8:
            return None

        from skimage import measure

        # Garder les dimensions originales pour les projections
        nx_orig, ny_orig, nz_orig = vol.shape

        # ── Compromis qualité/vitesse adaptatif ──
        # Petites structures (< 20M voxels) : step_size=1 (pleine résolution, rapide)
        # Structures moyennes (20-100M) : step_size=2 (4x plus rapide, détail préservé)
        # Grandes structures (>100M) : sous-échantillonnage voxel (8x plus rapide)
        stride = 1
        step_size = 1
        if vol.size > 100_000_000:
            stride = 2
            vol = vol[::2, ::2, ::2]
            step_size = 1
        elif vol.size > 20_000_000:
            step_size = 2

        try:
            verts, faces, _, _ = measure.marching_cubes(
                vol, level=0.5, step_size=step_size, allow_degenerate=False)
        except Exception:
            return None
        if verts.shape[0] < 3 or faces.shape[0] == 0:
            return None

        # Si le volume a été sous-échantillonné, rescaler les verts à la grille originale.
        if stride > 1:
            verts = verts * stride

        # Important: do not reapply C-arm angles on the 3D mesh here.
        # DRR/projection geometry already encodes LAO/CRAN/Table upstream.

        faces_vtk = np.hstack([
            np.full((faces.shape[0], 1), 3, dtype=np.int64),
            faces.astype(np.int64),
        ]).ravel()
        mesh = pv.PolyData(verts.astype(np.float32), faces_vtk).clean(tolerance=0.0)

        if mesh.n_points < 3:
            return None

        # Projection des sommets (meme convention que project_mask_3d).
        v = np.asarray(mesh.points, dtype=np.float64)
        nx, ny, nz = nx_orig, ny_orig, nz_orig
        reg_size = float(self._reg_size)

        x_plane = v[:, 0] * ((reg_size - 1.0) / max(nx - 1, 1))
        y_plane = v[:, 2] * ((reg_size - 1.0) / max(nz - 1, 1))
        # Convention DRR (offset cran +180 en PA) = flip superieur/inferieur.
        y_plane = (reg_size - 1.0) - y_plane

        fov_scale = self._fov_scale_for_mask(mask_3d)
        if abs(fov_scale - 1.0) > 0.02:
            c = reg_size * 0.5
            x_plane = c + (x_plane - c) * fov_scale
            y_plane = c + (y_plane - c) * fov_scale

        pts_plane = np.column_stack([x_plane, y_plane]).astype(np.float32)
        pts_plane = self._apply_registration_to_points(pts_plane)
        if side != self._reg_size:
            pts_plane *= float(side) / float(self._reg_size)

        # ── Profondeur (axe AP = axis 1) ──────────────────────────────────────
        # CRUCIAL pour ne PAS aplatir : la profondeur doit avoir EXACTEMENT le
        # meme facteur "monde par millimetre" que le plan. Le plan subit
        # (reg_size/nx) * fov_scale * scale_recalage * (side/reg_size) ; si la
        # profondeur n'utilise que side/nx (ancien code) elle rate scale et
        # fov_scale (souvent >1) -> structure en crepe. On reconstruit le facteur
        # in-plane par mm, puis on l'applique a la profondeur physique en mm.
        vx = vy = 1.0
        if self._ct_affine is not None:
            try:
                vx = float(abs(self._ct_affine[0, 0])) or 1.0
                vy = float(abs(self._ct_affine[1, 1])) or 1.0
            except Exception:
                vx = vy = 1.0
        scale_reg = 1.0
        if self._result is not None:
            try:
                scale_reg = float(self._result.get('scale', 1.0)) or 1.0
            except Exception:
                scale_reg = 1.0
        # Unites monde par voxel le long de l'axe 0 in-plane (jusqu'a l'espace side) :
        f_vox0 = (((reg_size - 1.0) / max(nx - 1, 1)) * fov_scale
                  * scale_reg * (float(side) / reg_size))
        # Unites monde par mm in-plane, puis profondeur physique (mm) -> monde.
        world_per_mm = f_vox0 / max(vx, 1e-6)
        depth_mm = (v[:, 1] - ((ny - 1.0) * 0.5)) * vy
        z_world = 10.0 + depth_mm * world_per_mm

        mesh.points = np.column_stack([
            pts_plane[:, 0],
            (side - 1.0) - pts_plane[:, 1],
            z_world,
        ]).astype(np.float32)

        mesh = mesh.clean(tolerance=0.0)
        if mesh.n_points > 300000:
            try:
                mesh = mesh.decimate_pro(0.6, preserve_topology=True)
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
        fluoro_actor = plotter.add_mesh(plane, texture=texture, name='fluoro_plane', lighting=False)

        added_meshes = 0
        # Acteurs par structure pour le gestionnaire de calques (oeil + opacite).
        # On construit TOUTES les structures disponibles afin que le basculement
        # de visibilite soit instantane dans la vue 3D (pas de reconstruction).
        seg_actors = {}           # name -> [fill_actor, edge_actor?]
        seg_base_opacity = min(0.9, max(0.2, self._alpha + 0.15))

        for idx, (name, _) in enumerate(self._proj_masks.items()):
            vol = self._seg_volumes.get(name)
            if vol is None:
                continue

            mesh = self._volume_to_registered_mesh(pv, name, vol, side)
            if mesh is None or mesh.n_points < 3:
                continue

            color = tuple(c / 255.0 for c in self._colors.get(name, (200, 200, 200)))

            fill_actor = plotter.add_mesh(
                mesh,
                color=color,
                opacity=seg_base_opacity,
                smooth_shading=True,
                name=f'{name}_{idx}',
            )
            actors = [fill_actor]
            try:
                edges = mesh.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=False,
                    manifold_edges=False,
                    non_manifold_edges=False,
                )
                if edges.n_points > 0:
                    edge_actor = plotter.add_mesh(
                        edges,
                        color=color,
                        line_width=max(1.2, self._lw),
                        opacity=0.95,
                        name=f'{name}_edge_{idx}',
                    )
                    actors.append(edge_actor)
            except Exception:
                pass

            # Visibilite initiale heritee du panneau 2D.
            visible = bool(self._vis.get(name, True))
            for a in actors:
                try:
                    a.SetVisibility(visible)
                except Exception:
                    pass
            seg_actors[name] = actors
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

        # NB : le stent (et la mesure ID) est rendu APRES les reperes, car il a
        # besoin de ref_3d (hinges + MS) pour orienter la profondeur et l'ID.

        # ── Reperes voxel CT projetes dans le plan fluoro via le pipeline des meshes ──
        # On utilise EXACTEMENT la meme chaine de projection que les segmentations
        # (orthographique + recalage 2D) pour que les reperes suivent les structures.
        ref_added = 0
        ref_3d = {}
        if self._tavi_ref_voxel and self._tavi_ct_shape is not None:
            ref_palette = {
                'hinge1': (100/255, 200/255, 255/255),
                'hinge2': (255/255, 200/255, 100/255),
                'ms':     (255/255, 100/255, 100/255),
            }
            ref_labels = {'hinge1': 'Hinge L', 'hinge2': 'Hinge R', 'ms': 'MS'}
            for k, voxel in self._tavi_ref_voxel.items():
                pix = self._project_voxel_to_view(voxel, self._tavi_ct_shape, side)
                if pix is None:
                    continue
                px, py = pix
                pt = (float(px), float((side - 1.0) - py), 10.0)
                color = ref_palette.get(k, (1.0, 1.0, 1.0))
                plotter.add_mesh(pv.Sphere(radius=max(3.0, side * 0.005), center=pt),
                                  color=color, name=f'ref_{k}')
                plotter.add_point_labels(
                    np.array([pt], dtype=np.float32), [ref_labels.get(k, k)],
                    font_size=12, point_color=color, text_color='white',
                    shape_opacity=0.0, always_visible=True, name=f'ref_label_{k}',
                )
                ref_3d[k] = np.asarray(pt, dtype=np.float64)
                ref_added += 1

        # ── Mesure MS (statique : ligne annulaire + cote perpendiculaire) ──
        self._draw_ms_measurement(plotter, pv, ref_3d, side)

        # ── Stent (fils tresses) + ID, avec profondeur ajustable a la main ──
        stent_added = False
        sp = self._tavi_stent_pose
        if sp and sp.get('center_px') and sp.get('pix_mm', 0) > 0:
            import math as _math
            try:
                from core.stent_model import braided_stent_world
                cx, cy = sp['center_px']
                pix_mm = float(sp['pix_mm'])
                length_px = float(sp['length_mm']) / pix_mm
                diameter_px = float(sp['diameter_mm']) / pix_mm
                ang = _math.radians(float(sp['axis_deg']))
                axis_plot = np.array([_math.cos(ang), _math.sin(ang), 0.0], dtype=np.float64)
                center0 = np.array([float(cx), float((side - 1.0) - cy), 8.0], dtype=np.float64)
                half = length_px * 0.5
                # Direction ventriculaire le long de l'axe (vers le MS si dispo)
                vent = axis_plot.copy()
                ms_pt = ref_3d.get('ms')
                if ms_pt is not None and float(np.dot(np.asarray(ms_pt) - center0, axis_plot)) < 0:
                    vent = -axis_plot
                verts_w, faces = braided_stent_world(
                    diameter_mm=float(diameter_px), length_mm=float(length_px),
                    center_world=center0, axis_world=axis_plot,
                    n_wires=24, braid_angle_deg=45.0,
                    wire_radius_mm=max(0.5, 0.18 / max(pix_mm, 1e-6)),
                    n_pts=110, tube_sides=5)
                faces_pv = np.hstack([
                    np.full((faces.shape[0], 1), 3, dtype=np.int64),
                    faces.astype(np.int64)]).ravel()
                stent_actor = plotter.add_mesh(
                    pv.PolyData(verts_w.astype(np.float32), faces_pv),
                    color=(0.98, 0.85, 0.25), opacity=0.92,
                    smooth_shading=True, name='stent_recale')
                stent_added = True
            except Exception:
                stent_actor = None

        # ── Mesure ID + slider de profondeur (hors du try ci-dessus pour que le
        #    slider ne soit JAMAIS avale par une exception de construction du mesh) ──
        if stent_added:
            # ID initial au repos
            ends0 = (center0 + axis_plot * half, center0 - axis_plot * half)
            id_mm0 = self._id_mm_from_plot(ref_3d, ends0, pix_mm)
            self._draw_id_and_risk(plotter, pv, ref_3d, side, ends0, id_mm0)

            def _update_stent_z(z_val):
                try:
                    stent_actor.SetPosition(0.0, 0.0, float(z_val))
                except Exception:
                    pass
                try:
                    plotter.render()
                except Exception:
                    pass

            try:
                plotter.add_slider_widget(
                    _update_stent_z,
                    rng=[-side * 0.4, side * 0.4], value=0.0,
                    title='Stent Z',
                    pointa=(0.73, 0.13), pointb=(0.97, 0.13),
                    color='#f0d020', style='modern', event_type='always')
            except TypeError:
                plotter.add_slider_widget(
                    _update_stent_z,
                    rng=[-side * 0.4, side * 0.4], value=0.0,
                    title='Stent Z',
                    pointa=(0.73, 0.13), pointb=(0.97, 0.13),
                    color='#f0d020', style='modern')
        else:
            self._draw_id_and_risk(plotter, pv, ref_3d, side, None, self._tavi_id_mm)

        # Slider profondeur fluoroscopie (bas gauche)
        def _set_fluoro_depth(value):
            fluoro_actor.SetPosition(0, 0, float(value))

        try:
            plotter.add_slider_widget(
                _set_fluoro_depth,
                rng=[-side * 0.4, side * 0.4], value=0.0,
                title='Fluoro Z',
                pointa=(0.03, 0.13), pointb=(0.27, 0.13),
                color='#8fa5cc', style='modern', event_type='always')
        except TypeError:
            plotter.add_slider_widget(
                _set_fluoro_depth,
                rng=[-side * 0.4, side * 0.4], value=0.0,
                title='Fluoro Z',
                pointa=(0.03, 0.13), pointb=(0.27, 0.13),
                color='#8fa5cc', style='modern')

        # ── Gestionnaire de calques : visibilite (oeil) + opacite globale ──
        self._add_layer_manager_3d(plotter, seg_actors, seg_base_opacity)

        plotter.enable_parallel_projection()
        title = 'Overlay 3D recale'
        if stent_added:
            title += ' + Stent'
        if ref_added:
            title += f' + {ref_added} reperes TAVI'
        title += '  |  R = vue de face'
        plotter.add_text(title, position='upper_right', font_size=9, color='#8892b0')
        plotter.add_axes(line_width=2)

        # Vue de face : Y vers le haut, X vers la droite, Z = profondeur AP.
        cx, cy = side * 0.5, side * 0.5
        def _reset_view(*args):
            plotter.camera_position = [
                (cx, cy, side * 2.0),
                (cx, cy, 10.0),
                (0.0, 1.0, 0.0),
            ]
            try:
                plotter.camera.zoom(0.55)   # dezoom plus large
            except Exception:
                pass
            plotter.render()

        _reset_view()
        # Bouton cliquable (haut gauche) + raccourci 'r' : recadre la vue de face.
        try:
            plotter.add_checkbox_button_widget(
                _reset_view, value=False, position=(12, 12), size=34,
                color_on='#4f9cf9', color_off='#4f9cf9', border_size=2)
            plotter.add_text('Reset vue de face (ou R)', position=(54, 18),
                             font_size=10, color='#cdd5e8', name='reset_hint')
        except Exception:
            pass
        plotter.add_key_event('r', lambda: _reset_view())
        plotter.show(title='Vue 3D - Overlay recale', auto_close=True)

    def _add_layer_manager_3d(self, plotter, seg_actors, base_opacity):
        """Gestionnaire de calques dans la vue 3D : un bouton oeil colore par
        structure (afficher/masquer) + un slider d'opacite globale. Minimaliste,
        ancre dans la fenetre de rendu sans surcharger la zone anatomique.
        """
        if not seg_actors:
            return

        # En-tete discret.
        try:
            plotter.add_text('CALQUES', position=(14, 868), font_size=10,
                             color='#8fa5cc', name='layers_hdr')
        except Exception:
            pass

        # Un bouton-case colore par structure, empile en haut a gauche.
        y0, dy = 838, 30
        for i, (name, actors) in enumerate(seg_actors.items()):
            rgb = self._colors.get(name, (200, 200, 200))
            hexcol = '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

            def _make_toggle(actor_list):
                def _toggle(state):
                    for a in actor_list:
                        try:
                            a.SetVisibility(bool(state))
                        except Exception:
                            pass
                    plotter.render()
                return _toggle

            yi = y0 - i * dy
            if yi < 150:        # evite de chevaucher les sliders du bas
                break
            try:
                plotter.add_checkbox_button_widget(
                    _make_toggle(actors), value=bool(self._vis.get(name, True)),
                    position=(14, yi), size=20,
                    color_on=hexcol, color_off='#2d3250', border_size=1)
                plotter.add_text(str(name), position=(42, yi + 2),
                                 font_size=9, color='#cdd5e8', name=f'layer_lbl_{i}')
            except Exception:
                pass

        # Slider d'opacite globale des segmentations (bas centre, vert medical).
        def _set_seg_opacity(val):
            for actors in seg_actors.values():
                for a in actors:
                    try:
                        a.GetProperty().SetOpacity(float(val))
                    except Exception:
                        pass
            plotter.render()

        try:
            plotter.add_slider_widget(
                _set_seg_opacity, rng=[0.1, 1.0], value=float(base_opacity),
                title='Opacite segmentations',
                pointa=(0.36, 0.13), pointb=(0.64, 0.13),
                color='#2ecc7a', style='modern', event_type='always')
        except TypeError:
            plotter.add_slider_widget(
                _set_seg_opacity, rng=[0.1, 1.0], value=float(base_opacity),
                title='Opacite segmentations',
                pointa=(0.36, 0.13), pointb=(0.64, 0.13),
                color='#2ecc7a', style='modern')

    def _add_dim_bracket(self, plotter, pv, p_a, p_b, label, color, name_prefix, side,
                         line_width=4, tick_factor=0.014, label_offset_z=4.0):
        """Dessine une "cote" entre p_a et p_b : segment + ticks + label decale.

        Acteurs nommes ``{name_prefix}_main/_tick_a/_tick_b/_label`` : un meme
        prefixe rappele REMPLACE l'ancien (utile pour le recalcul live de l'ID).
        """
        p_a = np.asarray(p_a, dtype=np.float64)
        p_b = np.asarray(p_b, dtype=np.float64)
        plotter.add_mesh(pv.Line(p_a, p_b), color=color, line_width=line_width,
                          name=f'{name_prefix}_main')
        axis = p_b - p_a
        n = float(np.linalg.norm(axis))
        if n > 1e-6:
            u = axis / n
            perp = np.array([-u[1], u[0], 0.0])
            pn = float(np.linalg.norm(perp))
            if pn > 1e-6:
                perp /= pn
            tick_len = max(8.0, side * float(tick_factor))
            for end_pt, suf in ((p_a, 'a'), (p_b, 'b')):
                plotter.add_mesh(pv.Line(end_pt + perp * tick_len, end_pt - perp * tick_len),
                                  color=color, line_width=line_width,
                                  name=f'{name_prefix}_tick_{suf}')
            mid = 0.5 * (p_a + p_b)
            label_pt = mid + perp * (tick_len * 2.2) + np.array([0.0, 0.0, float(label_offset_z)])
        else:
            label_pt = 0.5 * (p_a + p_b) + np.array([0.0, 0.0, float(label_offset_z)])
        plotter.add_point_labels(
            np.array([label_pt], dtype=np.float32), [label],
            font_size=15, point_color=color, text_color='white',
            shape='rounded_rect', shape_color='black', shape_opacity=0.65,
            always_visible=True, name=f'{name_prefix}_label')

    def _draw_ms_measurement(self, plotter, pv, ref_3d, side):
        """Ligne annulaire (cyan) + cote MS perpendiculaire (rouge). Statique."""
        h1 = ref_3d.get('hinge1'); h2 = ref_3d.get('hinge2'); ms_pt = ref_3d.get('ms')
        if h1 is None or h2 is None:
            return
        plotter.add_mesh(pv.Line(h1, h2), color=(90/255, 220/255, 240/255),
                          line_width=3, name='annulus_line')
        axis = h2 - h1
        n_axis = float(np.linalg.norm(axis))
        if n_axis > 1e-6 and ms_pt is not None:
            u_an = axis / n_axis
            proj = h1 + np.dot(ms_pt - h1, u_an) * u_an
            pix_mm = self._tavi_pix_mm()
            ms_mm = self._ms_mm_from_plot(ref_3d, pix_mm) if pix_mm else self._tavi_ms_length
            ms_label = f'MS = {ms_mm:.2f} mm' if ms_mm is not None else 'MS'
            self._add_dim_bracket(plotter, pv, proj, ms_pt, ms_label,
                                  (255/255, 90/255, 90/255), 'ms_dim', side, line_width=4)

    def _tavi_pix_mm(self):
        """mm/pixel calibre sur le stent (None si pas de stent)."""
        sp = self._tavi_stent_pose
        if sp and sp.get('pix_mm', 0) > 0:
            return float(sp['pix_mm'])
        return None

    def _ms_mm_from_plot(self, ref_3d, pix_mm):
        """MS (mm) = distance perpendiculaire du MS a la ligne annulaire, calculee
        dans le MEME repere projete que les points dessines (coherence geometrie<->valeur)."""
        h1 = ref_3d.get('hinge1'); h2 = ref_3d.get('hinge2'); ms = ref_3d.get('ms')
        if h1 is None or h2 is None or ms is None or not pix_mm:
            return None
        h1 = np.asarray(h1, dtype=np.float64)[:2]
        h2 = np.asarray(h2, dtype=np.float64)[:2]
        ms = np.asarray(ms, dtype=np.float64)[:2]
        u = h2 - h1
        nu = float(np.linalg.norm(u))
        if nu < 1e-6:
            return None
        u = u / nu
        rel = ms - h1
        perp = float(np.linalg.norm(rel - np.dot(rel, u) * u))
        return perp * float(pix_mm)

    def _id_mm_from_plot(self, ref_3d, stent_ends, pix_mm):
        """ID (mm) = distance perpendiculaire de la base ventriculaire du stent a
        la ligne annulaire, calculee dans le repere du plotter (px canvas)."""
        h1 = ref_3d.get('hinge1'); h2 = ref_3d.get('hinge2')
        if h1 is None or h2 is None or stent_ends is None:
            return None
        h1 = np.asarray(h1, dtype=np.float64)[:2]
        h2 = np.asarray(h2, dtype=np.float64)[:2]
        e1 = np.asarray(stent_ends[0], dtype=np.float64)[:2]
        e2 = np.asarray(stent_ends[1], dtype=np.float64)[:2]
        u = h2 - h1
        nu = float(np.linalg.norm(u))
        if nu < 1e-6:
            return None
        u = u / nu
        n = np.array([-u[1], u[0]])
        ms_pt = ref_3d.get('ms')
        if ms_pt is not None:
            ms2 = np.asarray(ms_pt, dtype=np.float64)[:2]
            if float(np.dot(ms2 - 0.5 * (h1 + h2), n)) < 0:
                n = -n
            d = max(0.0, float(np.dot(e1 - h1, n)), float(np.dot(e2 - h1, n)))
        else:
            d = max(abs(float(np.dot(e1 - h1, n))), abs(float(np.dot(e2 - h1, n))))
        return d * float(pix_mm)

    def _draw_id_and_risk(self, plotter, pv, ref_3d, side, stent_ends, id_mm):
        """Cote ID (base ventriculaire du stent -> ligne annulaire) + callout risque.

        Redessinable a chaque changement de profondeur : les acteurs portent des
        noms fixes donc PyVista remplace les anciens proprement.
        """
        h1 = ref_3d.get('hinge1'); h2 = ref_3d.get('hinge2'); ms_pt = ref_3d.get('ms')
        if (stent_ends is not None and h1 is not None and h2 is not None
                and id_mm is not None):
            h1a = np.asarray(h1, dtype=np.float64)
            h2a = np.asarray(h2, dtype=np.float64)
            e1 = np.asarray(stent_ends[0], dtype=np.float64)
            e2 = np.asarray(stent_ends[1], dtype=np.float64)
            u = h2a - h1a
            nu = float(np.linalg.norm(u))
            if nu > 1e-6:
                u = u / nu
                perp = np.array([-u[1], u[0], 0.0])
                pn = float(np.linalg.norm(perp))
                if pn > 1e-6:
                    perp /= pn
                if ms_pt is not None and float(np.dot(np.asarray(ms_pt) - 0.5 * (h1a + h2a), perp)) < 0:
                    perp = -perp
                # base ventriculaire = extremite la plus du cote MS
                base = e1 if float(np.dot(e1 - h1a, perp)) >= float(np.dot(e2 - h1a, perp)) else e2
                foot = h1a + np.dot(base - h1a, u) * u   # pied de la perpendiculaire
                foot[2] = base[2]                        # cote bien dans le plan du stent
                self._add_dim_bracket(plotter, pv, foot, base, f'ID = {id_mm:.2f} mm',
                                      (255/255, 160/255, 50/255), 'id_dim', side, line_width=4)
        pix_mm = self._tavi_pix_mm()
        ms_mm = self._ms_mm_from_plot(ref_3d, pix_mm) if pix_mm else self._tavi_ms_length
        self._draw_risk_callout(plotter, ms_mm, id_mm)

    def _draw_risk_callout(self, plotter, ms_v, id_v):
        """Callout texte (haut gauche) : MS / ID / dMSID / niveau de risque PM."""
        if ms_v is None and id_v is None:
            return
        try:
            from core.measurements import risk_assessment, DELTA_MSID_THRESHOLD_MM
        except Exception:
            risk_assessment = None
            DELTA_MSID_THRESHOLD_MM = 3.0
        level = pm_rate = delta = None
        if ms_v is not None and id_v is not None and risk_assessment is not None:
            r = risk_assessment(ms_v, id_v)
            delta = r.get('delta_msid_mm'); level = r.get('risk_level')
            pm_rate = r.get('pm_dependency_rate')

        if level == 'HIGH':
            color = '#e05060'
        elif level == 'LOW':
            color = '#2ecc7a'
        else:
            color = '#cdd5e8'

        lines = []
        if delta is not None:
            lines.append(f'ΔMSID = {delta:+.2f} mm')
            if level == 'HIGH':
                lines.append(f'Risque PM : HAUT  (~{pm_rate:.0%})' if pm_rate else 'Risque PM : HAUT')
            elif level == 'LOW':
                lines.append(f'Risque PM : BAS   (~{pm_rate:.1%})' if pm_rate else 'Risque PM : BAS')
        lines += [
            f'MS  = {ms_v:.2f} mm' if ms_v is not None else 'MS  = --',
            f'ID   = {id_v:.2f} mm' if id_v is not None else 'ID   = --',
        ]
        if delta is None:
            lines.append(f'ΔMSID = -- (< {DELTA_MSID_THRESHOLD_MM:.1f} mm = HAUT)')

        try:
            plotter.add_text('\n'.join(lines), position='upper_left',
                             font_size=14, color=color, shadow=True, name='risk_callout')
        except Exception:
            pass

    def resizeEvent(self, e):
        super().resizeEvent(e); self._render()


# ══════════════════════════════════════════════════════════════════════════════
# Worker thread
# ══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────

