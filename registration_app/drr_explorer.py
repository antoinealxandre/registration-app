"""
drr_explorer.py — Explorateur de paramètres DRR
Application standalone pour calibrer les paramètres de génération DRR
depuis un DICOM de fluoroscopie.

Usage :
    pip install pyqt5 pydicom nibabel scipy opencv-python numpy
    python drr_explorer.py
"""

import sys, os, json, time
import numpy as np
import cv2

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QSlider, QGroupBox,
    QDoubleSpinBox, QProgressBar, QSizePolicy, QStatusBar,
    QFrame, QScrollArea, QSplitter, QGridLayout, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QCheckBox, QSpinBox, QToolButton, QMessageBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor

# ══════════════════════════════════════════════════════════════════════════════
# Thème — style "salle de contrôle" : dark steel + cyan froid
# ══════════════════════════════════════════════════════════════════════════════

BG       = '#0a0c10'
PANEL    = '#0f1219'
CARD     = '#141820'
BORDER   = '#1c2130'
BORDER2  = '#252d40'
CYAN     = '#00d4c8'
CYAN2    = '#007a74'
AMBER    = '#ffb020'
RED      = '#e03c50'
TEXT     = '#c8d4e8'
DIM      = '#3a4460'
MID      = '#6a7890'
WHITE    = '#eaf0fc'

STYLE = f"""
QMainWindow, QWidget, QDialog {{
    background: {BG}; color: {TEXT};
    font-family: 'JetBrains Mono', 'Consolas', 'Courier New', monospace;
    font-size: 12px;
}}
QSplitter::handle {{ background: {BORDER}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{
    background: {BG}; width: 4px; border: none;
}}
QScrollBar::handle:vertical {{
    background: {BORDER2}; border-radius: 2px; min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{ background: {CYAN}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QPushButton {{
    background: {CARD}; border: 1px solid {BORDER2};
    border-radius: 4px; padding: 6px 14px;
    color: {TEXT}; font-size: 11px; min-height: 26px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
}}
QPushButton:hover {{ border-color: {CYAN}; color: {CYAN}; }}
QPushButton:pressed {{ background: {CYAN2}; border-color: {CYAN}; color: {WHITE}; }}
QPushButton:disabled {{ color: {DIM}; border-color: {BORDER}; }}
QPushButton#primary {{
    background: {CYAN2}; border-color: {CYAN}; color: {WHITE}; font-weight: 600;
}}
QPushButton#primary:hover {{ background: {CYAN}; color: {BG}; }}
QPushButton#primary:disabled {{ background: {CARD}; border-color: {DIM}; color: {DIM}; }}
QPushButton#danger {{
    background: #1a0a0d; border-color: {RED}; color: {RED};
}}
QDoubleSpinBox, QSpinBox, QComboBox {{
    background: {CARD}; border: 1px solid {BORDER2};
    border-radius: 3px; padding: 3px 6px;
    color: {TEXT}; selection-background-color: {CYAN2};
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    min-height: 22px;
}}
QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
    border-color: {CYAN};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {CARD}; border: 1px solid {BORDER2};
    color: {TEXT}; selection-background-color: {CYAN2};
}}
QSlider::groove:horizontal {{
    height: 2px; background: {BORDER2}; border-radius: 1px;
}}
QSlider::handle:horizontal {{
    width: 12px; height: 12px; background: {CYAN};
    border-radius: 6px; margin: -5px 0;
}}
QSlider::sub-page:horizontal {{ background: {CYAN}; border-radius: 1px; }}
QProgressBar {{
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 2px; height: 4px; color: transparent;
}}
QProgressBar::chunk {{ background: {CYAN}; border-radius: 2px; }}
QTabWidget::pane {{
    border: 1px solid {BORDER}; background: {PANEL};
    border-radius: 0 4px 4px 4px;
}}
QTabBar::tab {{
    background: {BG}; border: 1px solid {BORDER};
    border-bottom: none; padding: 5px 16px;
    color: {DIM}; border-radius: 4px 4px 0 0; margin-right: 2px;
    font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 11px;
}}
QTabBar::tab:selected {{ background: {PANEL}; color: {CYAN}; border-bottom: 2px solid {CYAN}; }}
QTabBar::tab:hover:!selected {{ color: {MID}; }}
QTableWidget {{
    background: {CARD}; border: 1px solid {BORDER};
    gridline-color: {BORDER}; color: {TEXT};
    font-size: 11px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
}}
QTableWidget::item {{ padding: 2px 6px; }}
QTableWidget::item:selected {{ background: {CYAN2}; color: {WHITE}; }}
QHeaderView::section {{
    background: {PANEL}; border: none; border-bottom: 1px solid {BORDER2};
    padding: 4px 6px; color: {MID}; font-size: 10px; font-weight: 600;
    letter-spacing: 1px;
}}
QLabel {{ color: {TEXT}; }}
QLabel#dim {{ color: {DIM}; font-size: 10px; }}
QLabel#mid {{ color: {MID}; font-size: 11px; }}
QLabel#val {{ color: {CYAN}; font-weight: 600; }}
QLabel#head {{
    color: {CYAN}; font-size: 10px; font-weight: 700;
    letter-spacing: 2px;
}}
QLabel#metric {{
    color: {AMBER}; font-size: 16px; font-weight: 700;
}}
QFrame#sep {{ background: {BORDER}; max-height: 1px; border: none; margin: 4px 0; }}
QStatusBar {{
    background: {PANEL}; border-top: 1px solid {BORDER};
    color: {MID}; font-size: 10px; padding: 2px 8px;
    font-family: 'JetBrains Mono', 'Consolas', monospace;
}}
QCheckBox {{ color: {TEXT}; spacing: 6px; }}
QCheckBox::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {BORDER2}; border-radius: 2px; background: {CARD};
}}
QCheckBox::indicator:checked {{ background: {CYAN}; border-color: {CYAN}; }}
QGroupBox {{
    border: none; margin: 0; padding: 0;
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# DICOM metadata extractor
# ══════════════════════════════════════════════════════════════════════════════

DICOM_TAGS_OF_INTEREST = [
    # Geometry
    ('0018', '1110', 'DistanceSourceToDetector',      'SID (mm)',            'geometry'),
    ('0018', '1111', 'DistanceSourceToPatient',       'SOD (mm)',            'geometry'),
    ('0018', '1114', 'EstimatedRadiographicMagnificationFactor', 'Magnification', 'geometry'),
    ('0018', '1164', 'ImagerPixelSpacing',            'Pixel Spacing (mm)',  'geometry'),
    ('0018', '1162', 'IntensifierSize',               'Intensifier (mm)',    'geometry'),
    ('0018', '1149', 'FieldOfViewDimensions',         'FOV (mm)',            'geometry'),
    ('0018', '7004', 'DetectorType',                  'Detector Type',       'geometry'),
    # Angles
    ('0018', '1510', 'PositionerPrimaryAngle',        'LAO/RAO (°)',         'angles'),
    ('0018', '1511', 'PositionerSecondaryAngle',      'CRAN/CAUD (°)',       'angles'),
    ('0018', '1138', 'TableAngle',                    'Table Angle (°)',     'angles'),
    ('0019', '1001', 'AngleValueLArm',                'L-Arm Angle',        'angles'),
    ('0019', '1002', 'AngleValuePArm',                'P-Arm Angle',        'angles'),
    ('0019', '1003', 'AngleValueCArm',                'C-Arm Angle',        'angles'),
    # Acquisition
    ('0018', '0060', 'KVP',                           'kVp',                 'acquisition'),
    ('0018', '1151', 'XRayTubeCurrent',               'Tube Current (mA)',   'acquisition'),
    ('0018', '1150', 'ExposureTime',                  'Exposure Time (ms)',  'acquisition'),
    ('0018', '115a', 'RadiationMode',                 'Radiation Mode',      'acquisition'),
    ('0018', '1160', 'FilterType',                    'Filter Type',         'acquisition'),
    ('0018', '1155', 'RadiationSetting',              'Radiation Setting',   'acquisition'),
    # Image
    ('0028', '0010', 'Rows',                          'Rows (px)',            'image'),
    ('0028', '0011', 'Columns',                       'Columns (px)',         'image'),
    ('0028', '0008', 'NumberOfFrames',                'Frames',               'image'),
    ('0018', '1147', 'FieldOfViewShape',              'FOV Shape',            'image'),
    ('0018', '7032', 'FieldOfViewRotation',           'FOV Rotation',         'image'),
    ('0018', '5100', 'PatientPosition',               'Patient Position',     'image'),
    ('0018', '0040', 'CineRate',                      'Cine Rate (fps)',       'image'),
    # Device
    ('0008', '0070', 'Manufacturer',                  'Manufacturer',         'device'),
    ('0008', '1090', 'ManufacturerModelName',         'Model',                'device'),
    ('0018', '1008', 'GantryID',                      'Gantry ID',            'device'),
    ('0018', '1000', 'DeviceSerialNumber',            'Serial Number',        'device'),
    ('0018', '1020', 'SoftwareVersions',              'Software Version',     'device'),
]


def extract_dicom_metadata(dcm) -> dict:
    """Extrait les métadonnées pertinentes d'un objet pydicom."""
    meta = {}
    rows = []

    for (g, e, name, label, category) in DICOM_TAGS_OF_INTEREST:
        tag = (int(g, 16), int(e, 16))
        try:
            el = dcm[tag]
            val = el.value
            if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
                val_str = ', '.join(str(v) for v in val)
            else:
                val_str = str(val)
            rows.append({'category': category, 'label': label,
                         'name': name, 'value': val_str, 'raw': val})
        except (KeyError, AttributeError):
            pass

    meta['rows'] = rows

    # Parse geometry for DRR defaults
    def _get_float(name_search):
        for r in rows:
            if r['name'] == name_search:
                try:
                    v = r['raw']
                    if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                        return float(list(v)[0])
                    return float(v)
                except Exception:
                    pass
        return None

    def _get_first_float(name_search):
        """Récupère la première valeur float d'un tag multi-valeurs."""
        for r in rows:
            if r['name'] == name_search:
                try:
                    v = r['raw']
                    if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)):
                        vals = list(v)
                        return float(vals[0])
                    return float(v)
                except Exception:
                    pass
        return None

    meta['sid_mm']       = _get_float('DistanceSourceToDetector')
    meta['sod_mm']       = _get_float('DistanceSourceToPatient')
    meta['magnification'] = _get_float('EstimatedRadiographicMagnificationFactor')
    meta['intensifier_mm'] = _get_float('IntensifierSize')
    meta['lao_deg']      = _get_float('PositionerPrimaryAngle')
    meta['cran_deg']     = _get_float('PositionerSecondaryAngle')
    meta['table_deg']    = _get_float('TableAngle')
    meta['fov_mm']       = _get_first_float('FieldOfViewDimensions')
    meta['pixel_mm']     = _get_first_float('ImagerPixelSpacing')
    meta['rows_px']      = _get_float('Rows')
    meta['cols_px']      = _get_float('Columns')

    # Extract fluoro image
    try:
        px = dcm.pixel_array
        if px.ndim == 3:
            px = px[px.shape[0] // 2]   # frame centrale
        if px.dtype != np.uint8:
            px = ((px - px.min()) / (px.max() - px.min() + 1e-8) * 255).astype(np.uint8)
        meta['fluoro_img'] = px
    except Exception:
        meta['fluoro_img'] = None

    return meta


# ══════════════════════════════════════════════════════════════════════════════
# DRR worker thread
# ══════════════════════════════════════════════════════════════════════════════

class DRRWorker(QThread):
    progress = pyqtSignal(int, str)
    result   = pyqtSignal(np.ndarray)
    error    = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            if not HAS_NIBABEL:
                self.error.emit("nibabel non installé — pip install nibabel")
                return
            from drr_generator import generate_drr

            def cb(pct, msg):
                self.progress.emit(int(pct * 100) if pct <= 1 else int(pct), str(msg))

            p = self.params
            drr = generate_drr(
                ct_path=p['ct_path'],
                lao_deg=p['lao_deg'],
                cran_deg=p['cran_deg'],
                table_angle=p['table_angle'],
                output_size=p['output_size'],
                sid_mm=p['sid_mm'],
                sod_mm=p['sod_mm'],
                fov_mm=p['fov_mm'],
                renderer=p['renderer'],
                progress_cb=cb,
            )
            self.result.emit(drr)
        except Exception as ex:
            self.error.emit(str(ex))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers UI
# ══════════════════════════════════════════════════════════════════════════════

def _sep():
    f = QFrame(); f.setObjectName('sep'); return f

def _lbl(text, obj=None):
    l = QLabel(text)
    if obj: l.setObjectName(obj)
    return l

def _head(text):
    l = QLabel(text); l.setObjectName('head')
    return l

def np_to_pixmap(img: np.ndarray, w: int, h: int) -> QPixmap:
    """Convertit un array numpy (float [0,1] ou uint8) en QPixmap redimensionné."""
    if img is None:
        return QPixmap()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img
    h_src, w_src = img_rgb.shape[:2]
    # Fit avec ratio
    scale = min(w / w_src, h / h_src)
    nw, nh = int(w_src * scale), int(h_src * scale)
    img_res = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    qimg = QImage(img_res.data, nw, nh, 3 * nw, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ══════════════════════════════════════════════════════════════════════════════
# Panneau paramètre : slider + spinbox couplés
# ══════════════════════════════════════════════════════════════════════════════

class ParamRow(QWidget):
    changed = pyqtSignal()

    def __init__(self, label: str, vmin: float, vmax: float, vstep: float,
                 vdefault: float, decimals: int = 1, unit: str = '', parent=None):
        super().__init__(parent)
        self._step = vstep
        self._decimals = decimals
        self._blocking = False

        lo = QHBoxLayout(self)
        lo.setContentsMargins(0, 0, 0, 0)
        lo.setSpacing(8)

        lbl = QLabel(f'{label}')
        lbl.setFixedWidth(180)
        lbl.setObjectName('mid')
        lo.addWidget(lbl)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(vmin / vstep), int(vmax / vstep))
        self.slider.setValue(int(vdefault / vstep))
        self.slider.setFixedWidth(140)
        lo.addWidget(self.slider)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(vmin, vmax)
        self.spin.setSingleStep(vstep)
        self.spin.setDecimals(decimals)
        self.spin.setValue(vdefault)
        self.spin.setSuffix(f' {unit}' if unit else '')
        self.spin.setFixedWidth(90)
        lo.addWidget(self.spin)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, v):
        if self._blocking: return
        self._blocking = True
        self.spin.setValue(round(v * self._step, self._decimals))
        self._blocking = False
        self.changed.emit()

    def _from_spin(self, v):
        if self._blocking: return
        self._blocking = True
        self.slider.setValue(int(v / self._step))
        self._blocking = False
        self.changed.emit()

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v: float):
        self._blocking = True
        self.spin.setValue(v)
        self.slider.setValue(int(v / self._step))
        self._blocking = False


# ══════════════════════════════════════════════════════════════════════════════
# Viewer d'image avec overlay
# ══════════════════════════════════════════════════════════════════════════════

class ImageViewer(QLabel):
    def __init__(self, placeholder='— pas d\'image —', parent=None):
        super().__init__(parent)
        self._img = None
        self._overlay = None
        self._placeholder = placeholder
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f'background: {CARD}; border: 1px solid {BORDER};')
        self.setText(f'<span style="color:{DIM}">{placeholder}</span>')

    def set_image(self, img: np.ndarray):
        self._img = img
        self._render()

    def set_overlay(self, img: np.ndarray):
        self._overlay = img
        self._render()

    def clear_overlay(self):
        self._overlay = None
        self._render()

    def _render(self):
        if self._img is None:
            self.setText(f'<span style="color:{DIM}">{self._placeholder}</span>')
            return
        w, h = self.width() - 4, self.height() - 4
        if w < 2 or h < 2:
            return
        if self._overlay is not None:
            composed = self._compose()
            px = np_to_pixmap(composed, w, h)
        else:
            px = np_to_pixmap(self._img, w, h)
        self.setPixmap(px)

    def _compose(self):
        """Blend DRR + fluoro côte à côte ou en overlay selon mode."""
        drr = self._img
        flu = self._overlay
        if drr is None or flu is None:
            return drr

        # Normalise les deux en uint8
        def to_u8(x):
            if x.dtype != np.uint8:
                x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
            if x.ndim == 3:
                x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            return x

        d = to_u8(drr)
        f = to_u8(flu)
        sz = (max(d.shape[1], f.shape[1]), max(d.shape[0], f.shape[0]))
        d = cv2.resize(d, sz)
        f = cv2.resize(f, sz)

        # Magenta/Vert overlay
        out = np.zeros((*sz[::-1], 3), dtype=np.uint8)
        out[:, :, 1] = d          # DRR → vert
        out[:, :, 2] = f          # Fluoro → rouge
        return out

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._render()


# ══════════════════════════════════════════════════════════════════════════════
# Panneau métadonnées DICOM
# ══════════════════════════════════════════════════════════════════════════════

class MetaPanel(QWidget):
    params_applied = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._meta = None
        lo = QVBoxLayout(self)
        lo.setContentsMargins(6, 6, 6, 6)
        lo.setSpacing(6)

        # Header
        hdr = QHBoxLayout()
        hdr.addWidget(_head('MÉTADONNÉES DICOM'))
        hdr.addStretch()
        self.btn_apply = QPushButton('→ Appliquer aux paramètres')
        self.btn_apply.setObjectName('primary')
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply)
        hdr.addWidget(self.btn_apply)
        lo.addLayout(hdr)

        # Résumé géométrie
        self.geo_box = QFrame()
        self.geo_box.setStyleSheet(f'background:{CARD};border:1px solid {BORDER2};border-radius:4px;')
        glo = QGridLayout(self.geo_box)
        glo.setContentsMargins(10, 8, 10, 8)
        glo.setSpacing(6)
        self._geo_labels = {}
        geo_fields = [
            ('SID',  'sid_mm',       'mm'),
            ('SOD',  'sod_mm',       'mm'),
            ('Mag.', 'magnification', '×'),
            ('FOV',  'fov_mm',       'mm'),
            ('LAO',  'lao_deg',      '°'),
            ('CRAN', 'cran_deg',     '°'),
        ]
        for i, (label, key, unit) in enumerate(geo_fields):
            row, col = i // 3, (i % 3) * 2
            glo.addWidget(_lbl(label, 'mid'), row, col)
            vl = _lbl('—', 'val')
            glo.addWidget(vl, row, col + 1)
            self._geo_labels[key] = (vl, unit)
        lo.addWidget(self.geo_box)

        # Filtre catégorie
        flo = QHBoxLayout()
        flo.addWidget(_lbl('Catégorie :', 'mid'))
        self.cat_combo = QComboBox()
        self.cat_combo.addItems(['Tout', 'geometry', 'angles', 'acquisition', 'image', 'device'])
        self.cat_combo.currentTextChanged.connect(self._filter)
        flo.addWidget(self.cat_combo)
        flo.addStretch()
        lo.addLayout(flo)

        # Table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['PARAMÈTRE', 'VALEUR', 'CATÉGORIE'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lo.addWidget(self.table)

    def load(self, meta: dict):
        self._meta = meta
        self.btn_apply.setEnabled(True)
        self._update_geo()
        self._filter(self.cat_combo.currentText())

    def _update_geo(self):
        if not self._meta: return
        for key, (lbl, unit) in self._geo_labels.items():
            v = self._meta.get(key)
            if v is not None:
                lbl.setText(f'{v:g} {unit}' if isinstance(v, float) else f'{v} {unit}')
            else:
                lbl.setText('—')

    def _filter(self, cat: str):
        if not self._meta: return
        rows = self._meta.get('rows', [])
        if cat != 'Tout':
            rows = [r for r in rows if r['category'] == cat]
        self.table.setRowCount(0)
        for r in rows:
            i = self.table.rowCount()
            self.table.insertRow(i)
            self.table.setItem(i, 0, QTableWidgetItem(r['label']))
            self.table.setItem(i, 1, QTableWidgetItem(r['value']))
            cat_item = QTableWidgetItem(r['category'])
            cat_item.setForeground(QColor(MID))
            self.table.setItem(i, 2, cat_item)
            self.table.setRowHeight(i, 22)

    def _apply(self):
        if not self._meta: return
        self.params_applied.emit(self._meta)


# ══════════════════════════════════════════════════════════════════════════════
# Panneau paramètres DRR
# ══════════════════════════════════════════════════════════════════════════════

class ParamsPanel(QWidget):
    generate_requested = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ct_path = None

        lo = QVBoxLayout(self)
        lo.setContentsMargins(8, 8, 8, 8)
        lo.setSpacing(8)

        # ── CT ──────────────────────────────────────────────────────────────
        lo.addWidget(_head('VOLUME CT'))
        ct_row = QHBoxLayout()
        self.btn_ct = QPushButton('Charger CT (.nii / .nii.gz)')
        self.btn_ct.clicked.connect(self._load_ct)
        ct_row.addWidget(self.btn_ct)
        self.lbl_ct = _lbl('—', 'dim')
        ct_row.addWidget(self.lbl_ct)
        lo.addLayout(ct_row)
        lo.addWidget(_sep())

        # ── Géométrie ────────────────────────────────────────────────────────
        lo.addWidget(_head('GÉOMÉTRIE C-ARM'))

        self.p_sid = ParamRow('SID — Source→Détecteur', 500, 1500, 1.0, 996.0, 1, 'mm')
        self.p_sod = ParamRow('SOD — Source→Isocentre', 300, 1200, 1.0, 720.0, 1, 'mm')
        self.p_fov = ParamRow('FOV à l\'isocentre',      50,  500,  1.0, 200.0, 1, 'mm')
        self.p_ap  = ParamRow('Décalage AP isocentre',  -1000, 1000,  1.0,   0.0, 1, 'mm')

        for p in [self.p_sid, self.p_sod, self.p_fov, self.p_ap]:
            lo.addWidget(p)

        # Indicateur magnification calculée
        self.lbl_mag = _lbl('Magnification : —', 'mid')
        lo.addWidget(self.lbl_mag)
        for p in [self.p_sid, self.p_sod]:
            p.changed.connect(self._update_mag)
        self._update_mag()

        lo.addWidget(_sep())

        # ── Angles ───────────────────────────────────────────────────────────
        lo.addWidget(_head('ANGLES C-ARM'))

        self.p_lao   = ParamRow('LAO (+) / RAO (−)',   -90, 90,  0.5, 0.0, 1, '°')
        self.p_cran  = ParamRow('CRAN (+) / CAUD (−)', -45, 45,  0.5, 0.0, 1, '°')
        self.p_table = ParamRow('Table angle',         -30, 30,  0.5, 0.0, 1, '°')

        for p in [self.p_lao, self.p_cran, self.p_table]:
            lo.addWidget(p)

        lo.addWidget(_sep())

        # ── Rendu ─────────────────────────────────────────────────────────────
        lo.addWidget(_head('RENDU'))
        rend_row = QHBoxLayout()
        rend_row.addWidget(_lbl('Backend :', 'mid'))
        self.combo_renderer = QComboBox()
        self.combo_renderer.addItems(['siddon', 'trilinear', 'nanodrr', 'cpu'])
        rend_row.addWidget(self.combo_renderer)
        rend_row.addStretch()
        rend_row.addWidget(_lbl('Taille :', 'mid'))
        self.spin_size = QSpinBox()
        self.spin_size.setRange(128, 1024)
        self.spin_size.setSingleStep(64)
        self.spin_size.setValue(512)
        self.spin_size.setFixedWidth(80)
        rend_row.addWidget(self.spin_size)
        lo.addLayout(rend_row)

        lo.addWidget(_sep())

        # ── Boutons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_gen = QPushButton('⚡  Générer DRR')
        self.btn_gen.setObjectName('primary')
        self.btn_gen.clicked.connect(self._request_generate)
        self.btn_gen.setEnabled(False)
        btn_row.addWidget(self.btn_gen)

        self.btn_reset = QPushButton('Réinitialiser')
        self.btn_reset.clicked.connect(self._reset)
        btn_row.addWidget(self.btn_reset)

        lo.addLayout(btn_row)

        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        lo.addWidget(self.prog)

        self.lbl_prog = _lbl('—', 'dim')
        lo.addWidget(self.lbl_prog)

        lo.addStretch()

        # ── Historique ────────────────────────────────────────────────────────
        lo.addWidget(_sep())
        lo.addWidget(_head('HISTORIQUE'))
        self.history_table = QTableWidget(0, 5)
        self.history_table.setHorizontalHeaderLabels(
            ['SID', 'SOD', 'FOV', 'AP off.', 'Renderer'])
        for i in range(5):
            self.history_table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.ResizeToContents)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setMaximumHeight(120)
        self.history_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.history_table.doubleClicked.connect(self._restore_history)
        lo.addWidget(self.history_table)
        self._history = []

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _load_ct(self):
        p, _ = QFileDialog.getOpenFileName(
            self, 'Charger CT NIfTI', '',
            'NIfTI (*.nii *.nii.gz);;Tous (*)')
        if p:
            self._ct_path = p
            self.lbl_ct.setText(os.path.basename(p))
            self.btn_gen.setEnabled(True)

    def _update_mag(self):
        sid = self.p_sid.value()
        sod = self.p_sod.value()
        if sod > 0:
            mag = sid / sod
            self.lbl_mag.setText(
                f'Magnification calculée : <span style="color:{CYAN}">{mag:.4f}×</span>'
                f'   (×{sid-sod:.0f} mm détecteur→isocentre)')
            self.lbl_mag.setTextFormat(Qt.RichText)

    def _request_generate(self):
        if not self._ct_path:
            return
        params = self.get_params()
        self._add_history(params)
        self.btn_gen.setEnabled(False)
        self.prog.setValue(0)
        self.lbl_prog.setText('En cours…')
        self.generate_requested.emit(params)

    def get_params(self) -> dict:
        sod_eff = self.p_sod.value() - self.p_ap.value()
        return {
            'ct_path':    self._ct_path,
            'lao_deg':    self.p_lao.value(),
            'cran_deg':   self.p_cran.value(),
            'table_angle': self.p_table.value(),
            'output_size': self.spin_size.value(),
            'sid_mm':     self.p_sid.value(),
            'sod_mm':     sod_eff,
            'fov_mm':     self.p_fov.value(),
            'renderer':   self.combo_renderer.currentText(),
            'ap_offset':  self.p_ap.value(),
        }

    def apply_dicom_meta(self, meta: dict):
        """Pré-remplit les paramètres depuis les métadonnées DICOM."""
        if meta.get('sid_mm'):  self.p_sid.setValue(meta['sid_mm'])
        if meta.get('sod_mm'):  self.p_sod.setValue(meta['sod_mm'])
        if meta.get('fov_mm'):  self.p_fov.setValue(meta['fov_mm'])
        if meta.get('lao_deg') is not None:  self.p_lao.setValue(meta['lao_deg'])
        if meta.get('cran_deg') is not None: self.p_cran.setValue(meta['cran_deg'])
        if meta.get('table_deg') is not None: self.p_table.setValue(meta['table_deg'])
        self._update_mag()

    def _reset(self):
        self.p_sid.setValue(996.0)
        self.p_sod.setValue(720.0)
        self.p_fov.setValue(200.0)
        self.p_ap.setValue(0.0)
        self.p_lao.setValue(0.0)
        self.p_cran.setValue(0.0)
        self.p_table.setValue(0.0)

    def _add_history(self, params: dict):
        self._history.append(params.copy())
        i = self.history_table.rowCount()
        self.history_table.insertRow(i)
        vals = [
            f"{params['sid_mm']:.0f}",
            f"{params['sod_mm']:.0f}",
            f"{params['fov_mm']:.0f}",
            f"{params['ap_offset']:.0f}",
            params['renderer'],
        ]
        for c, v in enumerate(vals):
            item = QTableWidgetItem(v)
            item.setTextAlignment(Qt.AlignCenter)
            self.history_table.setItem(i, c, item)
            self.history_table.setRowHeight(i, 20)
        self.history_table.scrollToBottom()

    def _restore_history(self, idx):
        row = idx.row()
        if 0 <= row < len(self._history):
            p = self._history[row]
            self.p_sid.setValue(p['sid_mm'])
            # sod_mm inclus ap_offset, on recalcule
            self.p_ap.setValue(p['ap_offset'])
            self.p_sod.setValue(p['sod_mm'] + p['ap_offset'])
            self.p_fov.setValue(p['fov_mm'])
            self.p_lao.setValue(p['lao_deg'])
            self.p_cran.setValue(p['cran_deg'])
            self.p_table.setValue(p['table_angle'])
            self.combo_renderer.setCurrentText(p['renderer'])


# ══════════════════════════════════════════════════════════════════════════════
# Fenêtre principale
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('DRR Explorer — calibration paramètres')
        self.resize(1400, 860)
        self._drr_img = None
        self._fluoro_img = None
        self._worker = None
        self._overlay_on = False

        self._build_ui()
        self.setStyleSheet(STYLE)
        self._status('Prêt — chargez un DICOM et un CT NIfTI.')

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_lo = QHBoxLayout(central)
        main_lo.setContentsMargins(0, 0, 0, 0)
        main_lo.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)

        # ── Colonne gauche : contrôles ────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(380)
        left.setStyleSheet(f'background:{PANEL}; border-right: 1px solid {BORDER};')
        llо = QVBoxLayout(left)
        llо.setContentsMargins(0, 0, 0, 0)
        llо.setSpacing(0)

        # Titre
        title_bar = QWidget()
        title_bar.setStyleSheet(f'background:{CARD}; border-bottom:1px solid {BORDER2};')
        tlo = QHBoxLayout(title_bar)
        tlo.setContentsMargins(12, 8, 12, 8)
        t = QLabel('DRR <span style="color:#00d4c8">Explorer</span>')
        t.setTextFormat(Qt.RichText)
        t.setStyleSheet(f'font-size:14px; font-weight:700; color:{WHITE};'
                        f'font-family:"JetBrains Mono","Consolas",monospace;')
        tlo.addWidget(t)
        tlo.addStretch()
        ver = _lbl('v1.0', 'dim')
        tlo.addWidget(ver)
        llо.addWidget(title_bar)

        # Tabs gauche : Params / DICOM
        self.left_tabs = QTabWidget()
        llо.addWidget(self.left_tabs)

        # Tab Paramètres
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.params_panel = ParamsPanel()
        self.params_panel.generate_requested.connect(self._start_drr)
        scroll.setWidget(self.params_panel)
        self.left_tabs.addTab(scroll, 'Paramètres')

        # Tab DICOM
        self.meta_panel = MetaPanel()
        self.meta_panel.params_applied.connect(self._on_meta_applied)
        self.left_tabs.addTab(self.meta_panel, 'DICOM')

        splitter.addWidget(left)

        # ── Colonne droite : viewers ──────────────────────────────────────────
        right = QWidget()
        rlo = QVBoxLayout(right)
        rlo.setContentsMargins(8, 8, 8, 8)
        rlo.setSpacing(8)

        # Toolbar viewers
        tbar = QHBoxLayout()

        self.btn_load_dcm = QPushButton('📂  Charger DICOM (.dcm)')
        self.btn_load_dcm.clicked.connect(self._load_dicom)
        tbar.addWidget(self.btn_load_dcm)

        tbar.addWidget(_lbl('|', 'dim'))

        self.btn_overlay = QPushButton('⊕  Overlay ON/OFF')
        self.btn_overlay.setCheckable(True)
        self.btn_overlay.toggled.connect(self._toggle_overlay)
        tbar.addWidget(self.btn_overlay)

        self.btn_save_drr = QPushButton('💾  Sauver DRR')
        self.btn_save_drr.clicked.connect(self._save_drr)
        self.btn_save_drr.setEnabled(False)
        tbar.addWidget(self.btn_save_drr)

        self.btn_save_params = QPushButton('📋  Exporter paramètres')
        self.btn_save_params.clicked.connect(self._export_params)
        tbar.addWidget(self.btn_save_params)

        tbar.addStretch()

        self.lbl_time = _lbl('', 'dim')
        tbar.addWidget(self.lbl_time)

        rlo.addLayout(tbar)

        # Viewers côte à côte
        viewers_row = QHBoxLayout()
        viewers_row.setSpacing(8)

        flu_col = QVBoxLayout()
        flu_hdr = QHBoxLayout()
        flu_hdr.addWidget(_head('FLUOROSCOPIE'))
        self.lbl_fluoro_info = _lbl('—', 'dim')
        flu_hdr.addWidget(self.lbl_fluoro_info)
        flu_hdr.addStretch()
        flu_col.addLayout(flu_hdr)
        self.viewer_flu = ImageViewer('Chargez un DICOM →')
        flu_col.addWidget(self.viewer_flu)
        viewers_row.addLayout(flu_col)

        drr_col = QVBoxLayout()
        drr_hdr = QHBoxLayout()
        drr_hdr.addWidget(_head('DRR GÉNÉRÉ'))
        self.lbl_drr_info = _lbl('—', 'dim')
        drr_hdr.addWidget(self.lbl_drr_info)
        drr_hdr.addStretch()
        drr_col.addLayout(drr_hdr)
        self.viewer_drr = ImageViewer('Générez un DRR ↓')
        drr_col.addWidget(self.viewer_drr)
        viewers_row.addLayout(drr_col)

        rlo.addLayout(viewers_row, stretch=1)

        # Barre de progression et infos
        self.prog_global = QProgressBar()
        self.prog_global.setRange(0, 100)
        self.prog_global.setValue(0)
        rlo.addWidget(self.prog_global)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        main_lo.addWidget(splitter)

        sb = QStatusBar()
        self.setStatusBar(sb)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _load_dicom(self):
        if not HAS_PYDICOM:
            self._status('ERREUR — pip install pydicom')
            QMessageBox.warning(self, 'pydicom manquant',
                                'Installez pydicom :\n  pip install pydicom')
            return
        p, _ = QFileDialog.getOpenFileName(
            self, 'Charger DICOM', '',
            'DICOM (*.dcm *.DCM);;Tous (*)')
        if not p:
            return
        try:
            dcm = pydicom.dcmread(p, force=True)
            meta = extract_dicom_metadata(dcm)
            self.meta_panel.load(meta)
            self.left_tabs.setCurrentIndex(1)   # basculer sur l'onglet DICOM

            if meta.get('fluoro_img') is not None:
                self._fluoro_img = meta['fluoro_img']
                self.viewer_flu.set_image(self._fluoro_img)
                sz = self._fluoro_img.shape
                self.lbl_fluoro_info.setText(f'{sz[1]}×{sz[0]} px')

            sid = meta.get('sid_mm', '—')
            sod = meta.get('sod_mm', '—')
            mag = meta.get('magnification', '—')
            self._status(
                f'DICOM chargé — SID={sid} mm  SOD={sod} mm  Mag={mag}  '
                f'| {os.path.basename(p)}')
        except Exception as ex:
            self._status(f'Erreur DICOM : {ex}')
            QMessageBox.critical(self, 'Erreur DICOM', str(ex))

    def _on_meta_applied(self, meta: dict):
        self.params_panel.apply_dicom_meta(meta)
        self.left_tabs.setCurrentIndex(0)
        self._status('Paramètres mis à jour depuis le DICOM.')

    def _start_drr(self, params: dict):
        if self._worker and self._worker.isRunning():
            self._status('DRR déjà en cours…')
            return
        self._t0 = time.time()
        self._worker = DRRWorker(params)
        self._worker.progress.connect(self._on_progress)
        self._worker.result.connect(self._on_drr_result)
        self._worker.error.connect(self._on_drr_error)
        self._worker.start()

    def _on_progress(self, pct: int, msg: str):
        self.params_panel.prog.setValue(pct)
        self.params_panel.lbl_prog.setText(msg)
        self.prog_global.setValue(pct)
        self._status(f'{pct}% — {msg}')

    def _on_drr_result(self, drr: np.ndarray):
        elapsed = time.time() - self._t0
        self._drr_img = drr
        self.viewer_drr.set_image(drr)
        sz = drr.shape
        self.lbl_drr_info.setText(f'{sz[1]}×{sz[0]} px')
        self.params_panel.prog.setValue(100)
        self.params_panel.lbl_prog.setText(f'Terminé en {elapsed:.1f}s')
        self.prog_global.setValue(100)
        self.btn_save_drr.setEnabled(True)
        self.params_panel.btn_gen.setEnabled(True)
        self.lbl_time.setText(f'⏱ {elapsed:.1f}s')

        if self._overlay_on and self._fluoro_img is not None:
            self.viewer_drr.set_overlay(self._fluoro_img)

        self._status(f'DRR généré en {elapsed:.1f}s — {sz[1]}×{sz[0]}')

    def _on_drr_error(self, msg: str):
        self.params_panel.prog.setValue(0)
        self.params_panel.lbl_prog.setText('Erreur')
        self.params_panel.btn_gen.setEnabled(True)
        self._status(f'ERREUR : {msg}')
        QMessageBox.critical(self, 'Erreur DRR', msg)

    def _toggle_overlay(self, checked: bool):
        self._overlay_on = checked
        if checked and self._drr_img is not None and self._fluoro_img is not None:
            self.viewer_drr.set_overlay(self._fluoro_img)
            self.viewer_flu.set_overlay(self._drr_img)
        else:
            self.viewer_drr.clear_overlay()
            self.viewer_flu.clear_overlay()

    def _save_drr(self):
        if self._drr_img is None:
            return
        p, _ = QFileDialog.getSaveFileName(
            self, 'Sauver DRR', 'drr.png',
            'PNG (*.png);;TIFF (*.tiff);;NumPy (*.npy)')
        if not p:
            return
        if p.endswith('.npy'):
            np.save(p, self._drr_img)
        else:
            img_u8 = (np.clip(self._drr_img, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(p, img_u8)
        self._status(f'DRR sauvegardé → {p}')

    def _export_params(self):
        params = self.params_panel.get_params()
        p, _ = QFileDialog.getSaveFileName(
            self, 'Exporter paramètres', 'drr_params.json', 'JSON (*.json)')
        if not p:
            return
        export = {k: v for k, v in params.items() if k != 'ct_path'}
        export['ct_path'] = params.get('ct_path', '')
        with open(p, 'w') as f:
            json.dump(export, f, indent=2)
        self._status(f'Paramètres exportés → {p}')

    def _status(self, msg: str):
        self.statusBar().showMessage(msg)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Patch pour que drr_generator soit trouvé dans le même dossier ou core/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [script_dir, os.path.join(script_dir, 'core')]:
        if os.path.isfile(os.path.join(candidate, 'drr_generator.py')):
            sys.path.insert(0, candidate)
            break

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Dark palette de base pour éviter les flash blancs
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(CARD))
    pal.setColor(QPalette.AlternateBase, QColor(PANEL))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(CARD))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()