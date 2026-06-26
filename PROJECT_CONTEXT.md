# Project Overview

**Projet_Recalage** est une application PyQt5 spécialisée dans le **recalage 2D/3D d'images médicales**, avec focus sur l'alignement de stents endovasculaires en fluoroscopie. Elle combine traitement d'images, ML (YOLO, TotalSegmentator), et visualisation 3D.

**Point d'entrée**: `registration_app/app.py` → `ui/main_window.py`

---

# Architecture

## Structure des dossiers

| Dossier | Rôle |
|---------|------|
| `registration_app/core/` | Algorithmes: recalage, génération DRR, YOLO, segmentation |
| `registration_app/ui/` | Interface PyQt5: main_window, widgets, dialogs, theme |
| `registration_app/services/` | Workers threaded (pipeline_workers.py) pour traitements asynchrones |
| `registration_app/utils/` | Utilitaires: DICOM/Nifti I/O (dicom_io.py) |
| `data/ct_scan/DICOM1/` | Données test (fichiers DICOM) |
| `stent.py` | Générateur paramétrique de stent tressé (standalone) |

## Stack technique

**Image médicale**: nibabel, SimpleITK, pydicom, NRRD  
**Calcul scientifique**: NumPy, SciPy, scikit-image, OpenCV  
**UI**: PyQt5, pyqtgraph  
**Visualisation 3D**: PyVista, trimesh  
**ML**: YOLO (via yolo_pipeline), TotalSegmentator  
**DRR**: nanoDRR (GPU) ou DiffDRR/CPU (fallback)  

---

# Key Modules

## 1. **core/registration.py** — Optimisation du recalage 2D/3D
- **Fonction centrale**: `register(mask_moving, mask_fixed)` 
- Approche: maximisation IoU en 4 DOF (tx, ty, rotation θ, scale)
- Pipeline: **init centroïde** → **Differential Evolution** → **Nelder-Mead**
- Retourne: transformations optimales + scores (IoU, Dice) + historique
- `apply_transform()`: similitude 2D (rotation + zoom + translation via OpenCV)

## 2. **core/drr_generator.py** — Génération de DRR (Digital Radiography)
- `generate_drr()`: crée une radiographie 2D à partir d'un volume 3D CT
- Backends: nanoDRR (GPU, rapide) → DiffDRR (Siddon/trilinear) → CPU (scipy ray-sum)
- Géométrie C-arm réaliste (cone-beam, prise en compte du HU → LAC physique)
- Post-processing presets: balanced, bone, soft-tissue (CLAHE, tophat, unsharp mask)
- `load_ct()`: charge fichiers Nifti/DICOM

## 3. **core/yolo_pipeline.py** — Détection d'objets
- `load_yolo_model()`, `detect_vertebrae()`: détection en temps réel
- `boxes_to_mask()`: convertit boîtes de détection en masques binaires
- Pair détections DRR ↔ fluoroscopie par centroïde (matching)

## 4. **core/totalseg_runner.py** — Segmentation automatique
- `run_totalsegmentator_cli()`: interface CLI vers TotalSegmentator
- `build_seg_masks_from_totalseg()`: extraction de masques par label
- Export multilabel NIfTI

## 5. **core/stent_placement.py** — Modélisation du stent
- `generate_stent_mesh()`: mesh géométrique du stent
- `project_stent_mask()`: projection 2D du stent 3D
- `transform_mask()`: applique transformations au stent

## 6. **services/pipeline_workers.py** — Asynchrone
- `WorkerThread`: QThread pour exécution longue (DRR, segmentation, registration)
- Signaux PyQt5 pour retour UI sans blocage
- `_match_centroids()`: appairage détections DRR ↔ fluoroscopie

## 7. **ui/main_window.py** — Orchestration UI
- Onglets: Image chargée, Detection, Registration, Final overlay
- Gestion des fichiers (DICOM/Nifti load/save)
- Sliders pour paramètres (FOV, rotation, translation live)
- Panneaux: DropZone, ImageCard, AnnotationCanvas, ResultPanel, SegmentationReviewPanel

## 8. **ui/widgets/annotation_widgets.py** — Composants visuels
- `AnnotationCanvas`: affichage image + overlay + détection interactive
- `DropZone`: drag-drop de fichiers
- `BusyOverlay`: indicateur de chargement
- `ImageCard`: vignette avec métadonnées

---

# Data Flow

```
Utilisateur charge image (DICOM/Nifti)
    ↓
[Auto-détection] YOLO ou TotalSegmentator (thread)
    ↓
Masque 2D segmenté affiché
    ↓
Génération DRR du volume 3D CT (nanoDRR, thread)
    ↓
[Recalage 2D/3D] optimisation IoU (Differential Evolution + Nelder-Mead, thread)
    ↓
Transformation optimale appliquée → superposition DRR/fluoroscopie
    ↓
Export résultat (fichiers, images, transformations JSON)
```

**Points d'entrée UI**:
- Onglet "Detection": YOLO sur fluoroscopie
- Onglet "Registration": recalage semi-automatique via sliders
- Onglet "Auto-pipeline": detection + registration full-auto
- Onglet "Final overlay": ajustement fin + export

---

# Coding Conventions

1. **Commentaires**: French, spécifiques au "pourquoi" (pas du "quoi")
2. **Nommage**: 
   - Masques: suffixe `_mask` (float32, 0-1)
   - Transformations: (tx, ty, angle_deg, scale)
   - Box YOLO: (x, y, w, h) ou (x1, y1, x2, y2)
3. **Threading**: signaux PyQt5 au lieu de joins directs
4. **Gestion erreurs**: try/except avec messages utilisateur via QMessageBox/status bar
5. **Types**: Optional, Callable importés depuis typing
6. **Conventions fichiers**:
   - Nifti: `.nii.gz`
   - DICOM: dossier avec multiples `.dcm`
   - Export: JSON (transformations), PNG (images), STL (mesh)

---

# Important Files

| Fichier | Priorité | Raison |
|---------|----------|--------|
| `registration_app/core/registration.py` | ★★★ | Cœur du recalage, algo critique |
| `registration_app/ui/main_window.py` | ★★★ | Orchestration complète de l'app |
| `registration_app/core/drr_generator.py` | ★★★ | Génération DRR (YOLO, recalage en dépendent) |
| `registration_app/services/pipeline_workers.py` | ★★ | Asynchronie UI, signaux |
| `registration_app/core/stent_placement.py` | ★★ | Projection stent 3D → 2D |
| `registration_app/core/yolo_pipeline.py` | ★★ | Détection YOLO |
| `registration_app/ui/widgets/annotation_widgets.py` | ★★ | Composants visuels, AnnotationCanvas |
| `stent.py` | ★ | Modèle stent (standalone, optionnel) |

---

# How to Extend the Project

## Ajouter une nouvelle segmentation
1. Implémenter fonction `run_[method]_segmentation()` dans `core/`
2. Ajouter bouton/option UI dans `ui/main_window.py`
3. Retourner masque float32 [0, 1]
4. Intégrer dans WorkerThread

## Ajouter un backend DRR
1. Implémenter `generate_drr_[backend]()` dans `core/drr_generator.py`
2. Ajouter choix dropdown en UI
3. Tester avec `load_ct()` existant
4. Fallback vers CPU si GPU indisponible

## Optimiser le recalage
1. Modifier bounds/algorithme dans `register()` (core/registration.py)
2. Tester avec `score_history` et `progress_cb`
3. Adapter search bounds pour le domaine médical spécifique

## Ajouter format d'export
1. Fonction `export_[format]()` dans `utils/` ou `core/`
2. Intégrer dans bouton "Save" de main_window
3. Gérer via WorkerThread si lourd

## Nouveau modèle YOLO
1. Place `.pt` dans dossier models
2. Update `yolo_pipeline.load_yolo_model()` avec chemin
3. Rétester `boxes_to_mask()` et centroid matching
