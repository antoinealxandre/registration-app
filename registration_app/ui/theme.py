"""UI theme tokens and shared visual constants."""

DARK_BG = '#0c0e14'
PANEL_BG = '#12151f'
CARD_BG = '#1a1d2a'
BORDER = '#1e2235'
BORDER2 = '#2d3250'
ACCENT = '#4f9cf9'
ACCENT2 = '#2ecc7a'
TEXT = '#cdd5e8'
TEXT_DIM = '#4d5570'
TEXT_MID = '#8892b0'
WARN = '#f0b040'
ERR = '#e05060'

STRUCT = {
    'vertebrae': {'rgb': (80, 220, 130), 'hex': '#50dc82', 'label': 'Vertebres'},
    'heart': {'rgb': (240, 80, 90), 'hex': '#f0505a', 'label': 'Coeur'},
    'autre': {'rgb': (80, 190, 240), 'hex': '#50bef0', 'label': 'Autre'},
}

SIDEBAR_W = 310
DEFAULT_FOV_MM = 220.0
AUTO_PIPELINE_FOV_MM = 220.0

THUMB_SIZE = 64
ROLE_COLORS = {None: BORDER2, 'fixed': '#4f9cf9', 'mobile': '#2ecc7a'}
ROLE_LABELS = {None: '', 'fixed': 'FIXE', 'mobile': 'MOBILE'}

SEG_PALETTE = [
    (79, 220, 130), (240, 80, 90), (80, 190, 240), (240, 180, 0), (180, 80, 240),
    (240, 120, 60), (60, 240, 240), (240, 60, 180), (120, 240, 60), (60, 120, 240),
    (200, 200, 80), (80, 200, 200), (200, 80, 200), (160, 240, 120), (240, 160, 120),
]

YOLO_BOX_PALETTE = [
    (80, 220, 130), (255, 100, 100), (100, 180, 255),
    (255, 200, 80), (200, 130, 255), (130, 255, 200),
    (255, 160, 100), (100, 255, 255), (255, 100, 200),
    (200, 255, 100),
]

MEDICAL_COLORS = {
    'myocardium': (206, 110, 84),
    'left atrium': (203, 108, 81),
    'left ventricle': (152, 55, 13),
    'left ventricle of heart': (152, 55, 13),
    'right atrium': (210, 115, 89),
    'right ventricle': (181, 85, 57),
    'right ventricle of heart': (181, 85, 57),
    'aorta': (224, 97, 76),
    'pulmonary artery': (0, 122, 171),
    'pulmonary venous system': (186, 77, 64),
    'atrial_appendage_left': (142, 192, 72),
    'left atrial appendage': (142, 192, 72),
    'superior_vena_cava': (115, 176, 130),
    'superior vena cava': (115, 176, 130),
    'inferior vena cava': (0, 151, 206),
    'heart': (206, 110, 84),
    'spleen': (157, 108, 162),
    'liver': (221, 130, 101),
    'stomach': (216, 132, 105),
    'esophagus': (211, 171, 143),
    'trachea': (182, 228, 255),
    'portal/splenic vein': (0, 151, 206),
}

VERTEBRA_COLOR = (226, 202, 134)

_CSV_COLOR_OVERRIDE: dict = {}


def set_csv_colors(colors: dict) -> None:
    """Enregistre les couleurs issues d'un NRRD/CSV Slicer (efface si dict vide)."""
    global _CSV_COLOR_OVERRIDE
    _CSV_COLOR_OVERRIDE = {str(k).lower().strip(): tuple(v) for k, v in colors.items()}


def color_for_structure(name: str) -> tuple:
    """Couleur (R,G,B) STABLE et DISTINCTE pour une structure, identique partout.

    Priorite : couleurs NRRD/CSV chargees > MEDICAL_COLORS > hash FNV-1a.
    """
    import colorsys
    key = str(name).lower().strip()
    if key in _CSV_COLOR_OVERRIDE:
        return _CSV_COLOR_OVERRIDE[key]
    if key in MEDICAL_COLORS:
        return MEDICAL_COLORS[key]
    if 'vertebra' in key or 'vertebr' in key:
        return VERTEBRA_COLOR
    if 'lung' in key:
        return (172, 138, 115)
    # FNV-1a : melange fortement meme pour des chaines tres proches.
    h = 2166136261
    for ch in key:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    # Teinte etalee sur tout le cercle via un multiplicateur de Knuth.
    hue = ((h * 2654435761) & 0xFFFFFFFF) / 4294967296.0
    sat = 0.60 + ((h >> 9) % 28) / 100.0     # 0.60 .. 0.87
    val = 0.82 + ((h >> 17) % 16) / 100.0    # 0.82 .. 0.97
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

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
