# 2D/3D Registration App — EP Lab

Application de recalage semi-automatique fluoroscopie ↔ CT-scan.
Architecture modulaire (UI, services, utilitaires, coeur de calcul) avec pipeline DRR + annotation + YOLO + recalage.

## Installation

```bash
pip install -r requirements.txt
```

> PyQt5 peut nécessiter : `pip install PyQt5 PyQt5-sip`
> Sur macOS : `brew install pyqt5` si pip échoue

## Lancement

```bash
cd registration_app
python app.py
```

## Workflow

```
1. Charger CT (.nii / .nii.gz)
   → L'axe AP est détecté automatiquement depuis l'affine NIfTI

2. [Optionnel] Charger Segmentation + Label CSV
   → Les masques vertèbres / cœur / aorte sont extraits
   → Ils seront projetés et superposés dans le résultat final

3. Charger Fluoroscopie (.png)
   → Affichée dans l'onglet "Fluoroscopie"

4. Annoter la fluoroscopie
   → Onglet "Fluoroscopie" — dessiner les contours des structures
   → Outils disponibles : crayon, rectangle, gomme

5. Générer le DRR
   → Ajuster LAO/RAO et Cran/Caud si la vue n'est pas PA pure
   → Cliquer "Générer DRR"
   → Rendu Beer-Lambert + CLAHE (aspect fluoroscopie)

6. Annoter le DRR
   → Onglet "DRR" — dessiner les mêmes vertèbres

7. Lancer le Recalage
   → Recalage rigide (optionnellement élastique)
   → Résultat dans l'onglet "Résultat"

8. Exporter
   → JSON avec transformation (tx, ty, angle) + IoU/Dice
   → PNGs des masques et du DRR
```

## Structure (Refactor)

```
registration_app/
├── app.py                         # Point d'entrée (launcher léger)
├── requirements.txt
├── core/                          # Coeur scientifique (DRR, recalage, YOLO)
│   ├── drr_generator.py
│   ├── registration.py
│   ├── refinement.py
│   └── yolo_pipeline.py
├── services/
│   └── pipeline_workers.py        # Workers QThread (DRR/YOLO/recalage)
├── ui/
│   ├── main_window.py             # Orchestration UI principale
│   ├── dialogs.py                 # Fenêtres annexes (YOLO, comparaison...)
│   ├── theme.py                   # Design tokens + stylesheet global
│   └── widgets/
│       └── annotation_widgets.py  # Canvas et panneaux réutilisables
├── utils/
│   └── dicom_io.py                # Lecture DICOM et CSV metadata
└── README.md
```

## Validation rapide

```bash
python -m compileall registration_app
```

```bash
cd registration_app
python -c "from ui.main_window import MainWindow; print('ok')"
```
