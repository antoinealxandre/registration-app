"""TotalSegmentator integration helpers (run, import, export)."""

from __future__ import annotations

import glob
import importlib.util
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np


ProgressCb = Optional[Callable[[int, str], None]]


_HEART_DETAILED = {
    "myocardium",
    "atrium_left",
    "atrium_right",
    "ventricle_left",
    "ventricle_right",
    "aorta",
    "pulmonary_artery",
}


def _emit(progress_cb: ProgressCb, pct: int, msg: str) -> None:
    if progress_cb is not None:
        progress_cb(int(max(0, min(100, pct))), msg)


def _strip_nii_suffix(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    if base.endswith(".nii"):
        return base[:-4]
    return os.path.splitext(base)[0]


def _load_binary_mask(path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    return (arr > 0).astype(np.uint8), img.affine


def _collect_mask_files(output_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(output_dir, "*.nii")))
    files.extend(sorted(glob.glob(os.path.join(output_dir, "*.nii.gz"))))
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in files:
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        out.append(p)
    return out


def _resolve_totalsegmentator_command() -> List[str]:
    """Resolve executable command for TotalSegmentator across PATH and module mode."""
    for name in ("TotalSegmentator", "totalsegmentator"):
        exe = shutil.which(name)
        if exe:
            return [exe]

    if importlib.util.find_spec("totalsegmentator.bin.TotalSegmentator") is not None:
        return [sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator"]

    raise RuntimeError(
        "Commande TotalSegmentator introuvable. "
        f"Interpreteur actif: {sys.executable}. "
        "Installez dans cet environnement: "
        f"{sys.executable} -m pip install -U TotalSegmentator"
    )


def _resolve_totalseg_set_license_command() -> List[str]:
    """Resolve command for totalseg_set_license across PATH and module mode."""
    exe = shutil.which("totalseg_set_license")
    if exe:
        return [exe]

    if importlib.util.find_spec("totalsegmentator.bin.totalseg_set_license") is not None:
        return [sys.executable, "-m", "totalsegmentator.bin.totalseg_set_license"]

    raise RuntimeError(
        "Commande totalseg_set_license introuvable. "
        f"Interpreteur actif: {sys.executable}. "
        "Verifiez l'installation de TotalSegmentator dans cet environnement."
    )


def run_totalsegmentator_cli(
    input_path: str,
    output_dir: str,
    task_name: str = "total",
    fast: bool = False,
    device: str = "gpu",
    license_key: str = "",
    progress_cb: ProgressCb = None,
) -> Dict:
    """Run TotalSegmentator CLI and return output folder metadata."""
    if not input_path:
        raise RuntimeError("Chemin d'entree TotalSegmentator manquant.")

    exe_cmd = _resolve_totalsegmentator_command()

    os.makedirs(output_dir, exist_ok=True)

    lic = (license_key or "").strip()
    if lic:
        set_lic_cmd = _resolve_totalseg_set_license_command()
        _emit(progress_cb, 3, "Activation de la licence TotalSegmentator...")
        proc_lic = subprocess.run(
            [*set_lic_cmd, "-l", lic],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_lic.returncode != 0:
            err = (proc_lic.stderr or proc_lic.stdout or "").strip()
            raise RuntimeError(f"Activation licence echouee: {err}")

    cmd = [*exe_cmd, "-i", input_path, "-o", output_dir, "-ta", task_name]
    if fast:
        cmd.append("--fast")
    if device and device.lower() in {"cpu", "gpu"}:
        cmd.extend(["--device", device.lower()])

    _emit(progress_cb, 5, f"Lancement TotalSegmentator ({task_name})...")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    pct = 8
    while True:
        line = proc.stdout.readline() if proc.stdout else ""
        if not line and proc.poll() is not None:
            break
        if not line:
            continue
        line = line.strip()
        if not line:
            continue
        if "download" in line.lower():
            pct = min(35, pct + 2)
        elif "predict" in line.lower() or "nnunet" in line.lower():
            pct = min(85, pct + 3)
        elif "save" in line.lower() or "writing" in line.lower():
            pct = min(95, pct + 2)
        else:
            pct = min(93, pct + 1)
        _emit(progress_cb, pct, line)

    code = proc.wait()
    if code != 0:
        raise RuntimeError("TotalSegmentator a echoue. Consultez le log de progression.")

    mask_files = _collect_mask_files(output_dir)
    if not mask_files:
        raise RuntimeError(
            "Aucun masque NIfTI trouve dans le dossier de sortie TotalSegmentator."
        )

    _emit(progress_cb, 98, f"Segmentation terminee ({len(mask_files)} masque(s)).")
    return {
        "output_dir": output_dir,
        "task_name": task_name,
        "mask_files": mask_files,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }


def build_seg_masks_from_totalseg(
    output_dir: str,
    profile: str = "all",
    progress_cb: ProgressCb = None,
) -> Dict:
    """
    Build app-compatible seg_masks dict from TotalSegmentator outputs.

    Profiles:
      - all: loads all non-empty masks
      - registration: returns keys vertebrae / heart / autre
      - cardiac: returns detailed heart structures when available
    """
    files = _collect_mask_files(output_dir)
    if not files:
        raise RuntimeError("Dossier de sortie vide: aucun masque NIfTI detecte.")

    affine_ref = None
    masks: Dict[str, np.ndarray] = {}
    p = (profile or "all").strip().lower()

    if p == "all":
        for i, fp in enumerate(files):
            name = _strip_nii_suffix(fp)
            m, aff = _load_binary_mask(fp)
            if m.sum() == 0:
                continue
            masks[name] = m
            if affine_ref is None:
                affine_ref = aff
            _emit(progress_cb, 65 + int(30 * (i + 1) / max(1, len(files))), f"Import {name}")

    elif p == "cardiac":
        wanted = set(_HEART_DETAILED)
        for i, fp in enumerate(files):
            name = _strip_nii_suffix(fp)
            if name not in wanted:
                continue
            m, aff = _load_binary_mask(fp)
            if m.sum() == 0:
                continue
            masks[name] = m
            if affine_ref is None:
                affine_ref = aff
            _emit(progress_cb, 65 + int(30 * (i + 1) / max(1, len(files))), f"Import {name}")

        # Fallback for tasks that only include a global heart label.
        if not masks:
            fallback = {"heart", "aorta", "pulmonary_artery", "pulmonary_vein"}
            for i, fp in enumerate(files):
                name = _strip_nii_suffix(fp)
                if name not in fallback:
                    continue
                m, aff = _load_binary_mask(fp)
                if m.sum() == 0:
                    continue
                masks[name] = m
                if affine_ref is None:
                    affine_ref = aff
                _emit(
                    progress_cb,
                    65 + int(30 * (i + 1) / max(1, len(files))),
                    f"Import {name}",
                )

    else:
        vertebrae = None
        heart = None
        other = None

        heart_names = {
            "heart",
            "myocardium",
            "atrium_left",
            "atrium_right",
            "ventricle_left",
            "ventricle_right",
        }
        other_names = {"aorta", "pulmonary_artery", "pulmonary_vein"}

        for i, fp in enumerate(files):
            name = _strip_nii_suffix(fp)
            add_to = None
            if name.startswith("vertebrae_") or name == "sacrum":
                add_to = "vertebrae"
            elif name in heart_names:
                add_to = "heart"
            elif name in other_names:
                add_to = "autre"

            if add_to is None:
                continue

            m, aff = _load_binary_mask(fp)
            if affine_ref is None:
                affine_ref = aff
            if m.sum() == 0:
                continue

            if add_to == "vertebrae":
                vertebrae = m if vertebrae is None else np.logical_or(vertebrae > 0, m > 0).astype(np.uint8)
            elif add_to == "heart":
                heart = m if heart is None else np.logical_or(heart > 0, m > 0).astype(np.uint8)
            else:
                other = m if other is None else np.logical_or(other > 0, m > 0).astype(np.uint8)

            _emit(progress_cb, 65 + int(30 * (i + 1) / max(1, len(files))), f"Fusion {add_to}")

        if vertebrae is not None and vertebrae.sum() > 0:
            masks["vertebrae"] = vertebrae
        if heart is not None and heart.sum() > 0:
            masks["heart"] = heart
        if other is not None and other.sum() > 0:
            masks["autre"] = other

    if not masks:
        raise RuntimeError(
            "Aucun masque exploitable pour le profil selectionne. "
            "Essayez un autre profil ou une autre tache."
        )

    shape = next(iter(masks.values())).shape
    return {
        "masks": masks,
        "affine": affine_ref,
        "shape": shape,
        "mask_names": list(masks.keys()),
    }


def export_multilabel_segmentation(
    seg_masks: Dict[str, np.ndarray],
    out_path: str,
    affine: Optional[np.ndarray] = None,
) -> Dict:
    """Export current seg_masks to multilabel NIfTI or NRRD + labels CSV."""
    if not seg_masks:
        raise RuntimeError("Aucune segmentation a exporter.")
    if not out_path:
        raise RuntimeError("Chemin de sortie manquant.")

    names = list(seg_masks.keys())
    ref = seg_masks[names[0]]
    vol = np.zeros(ref.shape, dtype=np.uint16)
    for i, name in enumerate(names, start=1):
        m = seg_masks[name]
        if m.shape != ref.shape:
            raise RuntimeError("Toutes les segmentations doivent avoir la meme shape.")
        vol[m > 0] = i

    aff = affine if affine is not None else np.eye(4, dtype=np.float32)
    ext = os.path.splitext(out_path.lower())[1]
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    labels_csv = os.path.splitext(out_path)[0] + ".labels.csv"
    with open(labels_csv, "w", encoding="utf-8") as f:
        f.write("id,label\n")
        for i, name in enumerate(names, start=1):
            f.write(f"{i},{name}\n")

    if ext == ".nrrd":
        try:
            import nrrd  # type: ignore
        except Exception as ex:
            raise RuntimeError(
                f"Export NRRD indisponible ({ex}). Installez: pip install pynrrd"
            )
        nrrd.write(out_path, vol.astype(np.uint16))
    else:
        if not (out_path.lower().endswith(".nii") or out_path.lower().endswith(".nii.gz")):
            out_path = out_path + ".nii.gz"
        nii = nib.Nifti1Image(vol, aff)
        nib.save(nii, out_path)

    return {
        "out_path": out_path,
        "labels_csv": labels_csv,
        "n_labels": len(names),
        "shape": vol.shape,
    }
