import sys, os, csv
import numpy as np
import nibabel as nib
import cv2
from scipy.ndimage import rotate as _ndrot

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QDoubleSpinBox, QCheckBox,
    QFrame, QFileDialog, QScrollArea, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent, QFont

# ─────────────────────────────────────────────────────────────────────────────
# GPU backends
# ─────────────────────────────────────────────────────────────────────────────
_DEVICE = None
HAS_DIFFDRR = False
HAS_NANODRR = False

try:
    import torch
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from diffdrr.drr import DRR as _DiffDRR
    from diffdrr.data import read as _diffdrr_read
    HAS_DIFFDRR = True
except ImportError: pass

try:
    import torch
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from nanodrr.data import Subject as _NanoSubject
    from nanodrr.camera import make_k_inv as _make_k_inv, make_rt_inv as _make_rt_inv
    from nanodrr.drr import render as _nanodrr_render
    HAS_NANODRR = True
except ImportError: pass

# ─────────────────────────────────────────────────────────────────────────────
# DRR generation & Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _postprocess(img: np.ndarray) -> np.ndarray:
    """Transforme les données brutes en image visible [0,1]."""
    if img.max() == img.min():
        return np.zeros_like(img)
    
    # Log transform pour compresser la dynamique (effet rayons X)
    img = np.log1p(np.clip(img, 0, None))
    
    # Normalisation robuste
    lo, hi = np.percentile(img, (1, 99))
    if hi > lo:
        img = np.clip((img - lo) / (hi - lo), 0, 1)
    else:
        img = (img - img.min()) / (img.max() - img.min() + 1e-7)

    img = np.flipud(img) # Correction orientation
    img = np.fliplr(img)
    
    # Amélioration du contraste (CLAHE)
    u8 = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    u8 = clahe.apply(u8)
    
    return u8.astype(np.float32) / 255.0

def generate_drr(ct_path: str, lao: float, cran: float, table: float,
                 size: int, sid: float, sod: float, fov_mm: float = None) -> np.ndarray:
    
    # ── nanoDRR (Prioritaire) ────────────────────────────────────────────────
    if HAS_NANODRR:
        try:
            from nanodrr.data.preprocess import MU_BONE
            # Chargement volume
            subj = _NanoSubject.from_filepath(ct_path, mu_bone=MU_BONE).to(_DEVICE)
            
            # Calcul du champ de vision
            # Si auto, on prend ~400mm
            fov = fov_mm if (fov_mm and fov_mm > 0) else 400.0
            delx = (fov * (sid / sod)) / size
            
            k_inv = _make_k_inv(sdd=sid, delx=delx, dely=delx, x0=0, y0=0, 
                                height=size, width=size, device=_DEVICE)
            
            rt_inv = _make_rt_inv(
                torch.tensor([[cran, lao, table]], dtype=torch.float32, device=_DEVICE),
                torch.tensor([[0.0, sod, 0.0]], dtype=torch.float32, device=_DEVICE),
                orientation="PA", # Source devant
                isocenter=subj.isocenter,
            )
            
            with torch.no_grad():
                t = _nanodrr_render(subj, k_inv, rt_inv, torch.tensor([sid], device=_DEVICE))
                # nanoDRR sort (B, C, H, W). On prend le premier canal.
                res = t.squeeze().cpu().numpy()
                print(f"[Debug nanoDRR] Raw range: {res.min():.4f} to {res.max():.4f}")
                return _postprocess(res)
        except Exception as e:
            print(f"[nanoDRR Error] {e}")

    # ── CPU Fallback (Ray-summation simple) ──────────────────────────────────
    img = nib.load(ct_path)
    vol = np.clip(img.get_fdata().astype(np.float32), 0, None)
    if abs(lao) > 0.1: vol = _ndrot(vol, -lao, axes=(0, 1), reshape=False, order=1)
    if abs(cran) > 0.1: vol = _ndrot(vol, cran, axes=(1, 2), reshape=False, order=1)
    proj = vol.sum(axis=1).astype(np.float32).T
    proj = cv2.resize(proj, (size, size))
    return _postprocess(proj)

def _rotate_projection_2d(proj2d: np.ndarray, lao: float, cran: float) -> np.ndarray:
    h, w = proj2d.shape

    # LAO/RAO = rotation autour axe vertical → flip horizontal + compression X
    if abs(lao) > 0.1:
        # Compression horizontale selon cos(angle)
        scale_x = max(abs(np.cos(np.radians(lao))), 0.1)
        new_w = int(w * scale_x)
        if new_w > 0:
            shrunk = cv2.resize(proj2d, (new_w, h), interpolation=cv2.INTER_LINEAR)
            proj2d = np.zeros((h, w), dtype=proj2d.dtype)
            x0 = (w - new_w) // 2
            proj2d[:, x0:x0 + new_w] = shrunk
        # Flip horizontal si LAO > 90° ou RAO > 90°
        if abs(lao) > 90:
            proj2d = np.fliplr(proj2d)

    # CRAN/CAUD = rotation autour axe horizontal → compression verticale
    if abs(cran) > 0.1:
        scale_y = max(abs(np.cos(np.radians(cran))), 0.1)
        new_h = int(h * scale_y)
        if new_h > 0:
            shrunk = cv2.resize(proj2d, (w, new_h), interpolation=cv2.INTER_LINEAR)
            proj2d = np.zeros((h, w), dtype=proj2d.dtype)
            y0 = (h - new_h) // 2
            proj2d[y0:y0 + new_h, :] = shrunk   
        # Flip vertical si CRAN > 90°
        if abs(cran) > 90:
            proj2d = np.flipud(proj2d)

    return proj2d

def _project_mask(mask3d: np.ndarray, lao: float, cran: float) -> np.ndarray:
    """Projette le masque sur le bon axe selon les angles."""
    vol = (mask3d > 0).astype(np.float32)
    
    lao_r = np.radians(lao)
    cran_r = np.radians(cran)

    # Direction de la ligne de visée en coordonnées (X, Y, Z)
    # LAO tourne autour de Z, CRAN autour de X
    dx = np.sin(lao_r) * np.cos(cran_r)
    dy = np.cos(lao_r) * np.cos(cran_r)
    dz = np.sin(cran_r)

    # On choisit l'axe dominant pour la projection
    adx, ady, adz = abs(dx), abs(dy), abs(dz)

    if ady >= adx and ady >= adz:
        # Projection sur axe Y (vue PA/AP, cas de base)
        proj = vol.max(axis=1).T
        proj = np.flipud(np.fliplr(proj))
        proj = _rotate_projection_2d(proj, lao, cran)

    elif adx >= ady and adx >= adz:
        # Projection sur axe X (vue latérale, LAO/RAO ~90°)
        proj = vol.max(axis=0).T
        proj = np.flipud(proj)
        if dx < 0:
            proj = np.fliplr(proj)
        # Compression CRAN résiduelle
        if abs(cran) > 0.1:
            scale_y = max(abs(np.cos(cran_r)), 0.1)
            h, w = proj.shape
            new_h = int(h * scale_y)
            if new_h > 0:
                shrunk = cv2.resize(proj, (w, new_h), interpolation=cv2.INTER_LINEAR)
                proj = np.zeros((h, w), dtype=proj.dtype)
                y0 = (h - new_h) // 2
                proj[y0:y0 + new_h, :] = shrunk

    else:
        # Projection sur axe Z (vue craniale, CRAN ~90°)
        proj = vol.max(axis=2).T
        proj = np.flipud(proj)
        if dz < 0:
            proj = np.fliplr(proj)

    return proj

# ─────────────────────────────────────────────────────────────────────────────
# GUI Components
# ─────────────────────────────────────────────────────────────────────────────

class Worker(QThread):
    done = pyqtSignal(object)
    masks = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, ct_path, seg_masks, active_segs, lao, cran, table, size, sid, sod, fov_mm):
        super().__init__()
        self.params = (ct_path, lao, cran, table, size, sid, sod, fov_mm)
        self.seg_masks = seg_masks
        self.active = active_segs
        self.size = size
        self.lao = lao
        self.cran = cran

    def run(self):
        try:
            drr = generate_drr(*self.params)
            self.done.emit(drr)

            out_masks = {}
            for name, mask3d in self.seg_masks.items():
                if name not in self.active:
                    continue

                # 1. Projection max (rapide, pas de rotation 3D)
                vol = (mask3d > 0).astype(np.float32)
                m_proj = vol.max(axis=1).astype(np.float32).T

                # 2. Même orientation que _postprocess
                m_proj = np.flipud(np.fliplr(m_proj))

                # 3. Rotation 2D légère pour suivre les angles
                m_proj = _project_mask(mask3d, self.lao, self.cran)
                out_masks[name] = cv2.resize(
                    m_proj, (self.size, self.size),
                    interpolation=cv2.INTER_NEAREST
                )

            self.masks.emit(out_masks)
        except Exception as e:
            self.error.emit(str(e))

class DRRMini(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DRR Explorer - Fixed")
        self.setMinimumSize(1000, 750)
        
        self.ct_path = None
        self.seg_masks = {} # {Nom: Array3D}
        self.drr_arr = None
        self.mask_arrs = {} # {Nom: Array2D}
        self._seg_checks = {}
        
        self._build_ui()
        self.setStyleSheet(STYLE)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Panel Gauche
        panel = QWidget()
        panel.setFixedWidth(300)
        lay = QVBoxLayout(panel)
        
        lay.addWidget(QLabel("<b>CHARGEMENT</b>"))
        self.btn_ct = QPushButton("📁 Charger CT (NIfTI)")
        self.btn_ct.clicked.connect(self._pick_ct)
        lay.addWidget(self.btn_ct)
        
        self.btn_seg = QPushButton("📁 Charger Masque Labels")
        self.btn_seg.clicked.connect(self._pick_seg)
        lay.addWidget(self.btn_seg)

        lay.addWidget(_sep())
        lay.addWidget(QLabel("<b>STRUCTURES</b>"))
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.seg_list_layout = QVBoxLayout(self.scroll_content)
        self.scroll.setWidget(self.scroll_content)
        lay.addWidget(self.scroll)

        lay.addWidget(_sep())
        lay.addWidget(QLabel("<b>GEOMETRIE (Angles)</b>"))
        self.sp_lao = self._add_spin(lay, "LAO/RAO", -180, 180, 0)
        self.sp_cran = self._add_spin(lay, "CRAN/CAUD", -90, 90, 0)
        
        lay.addWidget(QLabel("<b>DISTANCE (mm)</b>"))
        self.sp_sid = self._add_spin(lay, "Source-Det (SID)", 500, 1500, 1000)
        self.sp_sod = self._add_spin(lay, "Source-Obj (SOD)", 200, 1000, 500)

        self.btn_run = QPushButton("▶ GÉNÉRER DRR")
        self.btn_run.setMinimumHeight(50)
        self.btn_run.clicked.connect(self._run_task)
        lay.addWidget(self.btn_run)

        self.lbl_status = QLabel("Prêt")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        # Panel Droit (Visualisation)
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setStyleSheet("background-color: black; border: 2px solid #333;")
        
        main_layout.addWidget(panel)
        main_layout.addWidget(self.view, 1)

    def _add_spin(self, layout, label, lo, hi, val):
        h = QHBoxLayout()
        h.addWidget(QLabel(label))
        s = QDoubleSpinBox()
        s.setRange(lo, hi); s.setValue(val)
        h.addWidget(s)
        layout.addLayout(h)
        return s

    def _pick_ct(self):
        path, _ = QFileDialog.getOpenFileName(self, "CT", "", "*.nii*")
        if path:
            self.ct_path = path
            self.lbl_status.setText(f"CT chargé : {os.path.basename(path)}")

    def _pick_seg(self):
        path, _ = QFileDialog.getOpenFileName(self, "Masque", "", "*.nii*")
        if not path: return
        
        try:
            mask_data = nib.load(path).get_fdata()
            csv_path = path.replace(".nii.gz", ".csv").replace(".nii", ".csv")
            labels = {}
            if os.path.exists(csv_path):
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    labels = {int(row[0]): row[1] for row in reader if row}

            unique_ids = np.unique(mask_data).astype(int)
            for uid in unique_ids:
                if uid == 0: continue
                name = labels.get(uid, f"Label_{uid}")
                self.seg_masks[name] = (mask_data == uid).astype(np.uint8)
                
                chk = QCheckBox(name)
                chk.setChecked(True)
                self._seg_checks[name] = chk
                self.seg_list_layout.addWidget(chk)
            
            self.lbl_status.setText(f"Masque chargé : {len(self.seg_masks)} structures.")
        except Exception as e:
            self.lbl_status.setText(f"Erreur masque : {e}")

    def _run_task(self):
        if not self.ct_path:
            self.lbl_status.setText("Erreur : Chargez un CT d'abord")
            return
        
        active = [n for n, c in self._seg_checks.items() if c.isChecked()]
        self.btn_run.setEnabled(False)
        self.lbl_status.setText("Calcul en cours...")

        self.worker = Worker(self.ct_path, self.seg_masks, active, 
                             self.sp_lao.value(), self.sp_cran.value(), 0, 
                             512, self.sp_sid.value(), self.sp_sod.value(), 400)
        self.worker.done.connect(self._display_drr)
        self.worker.masks.connect(self._display_masks)
        self.worker.error.connect(lambda e: self.lbl_status.setText(f"Erreur : {e}"))
        self.worker.finished.connect(lambda: self.btn_run.setEnabled(True))
        self.worker.start()

    def _display_drr(self, arr):
        self.drr_arr = arr
        self._refresh_view()

    def _display_masks(self, masks):
        self.mask_arrs = masks
        self._refresh_view()

    def _refresh_view(self):
        if self.drr_arr is None: return
        
        # Convertir en RGB pour l'overlay
        img_rgb = cv2.cvtColor((self.drr_arr * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Couleurs pour les segmentations
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255)]
        
        for i, (name, m2d) in enumerate(self.mask_arrs.items()):
            color = colors[i % len(colors)]
            contours, _ = cv2.findContours(m2d.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_rgb, contours, -1, color, 1)
            
        h, w, ch = img_rgb.shape
        # IMPORTANT: np.ascontiguousarray et .copy() pour éviter le crash/image noire
        qimg = QImage(np.ascontiguousarray(img_rgb).data, w, h, w*ch, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self.view.setPixmap(pix.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.lbl_status.setText("Rendu terminé.")

# Style Minimaliste Dark
STYLE = """
QMainWindow, QWidget { background-color: #121212; color: #E0E0E0; font-family: Segoe UI; }
QPushButton { background-color: #333; border: 1px solid #555; padding: 5px; border-radius: 4px; }
QPushButton:hover { background-color: #444; }
QDoubleSpinBox { background-color: #222; border: 1px solid #444; padding: 3px; }
QScrollArea { border: none; background-color: #121212; }
"""

def _sep():
    f = QFrame(); f.setFrameShape(QFrame.HLine); f.setStyleSheet("background-color: #333;"); return f

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DRRMini()
    win.show()
    sys.exit(app.exec_())