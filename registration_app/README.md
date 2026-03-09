# 2D/3D Registration App — EP Lab

Application de recalage semi-automatique fluoroscopie ↔ CT-scan.
Annotation manuelle + DRR Beer-Lambert + IoU optimization (3 DOF).

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
   → Onglet "Fluoroscopie" — dessiner les contours des vertèbres
   → Outil Rectangle : cliquer-glisser
   → Outil Polygone  : cliquer les points, double-clic pour fermer
   → Outil Gomme     : effacer des zones du masque

5. Générer le DRR
   → Ajuster LAO/RAO et Cran/Caud si la vue n'est pas PA pure
   → Cliquer "Générer DRR"
   → Rendu Beer-Lambert + CLAHE (aspect fluoroscopie)

6. Annoter le DRR
   → Onglet "DRR" — dessiner les mêmes vertèbres

7. Lancer le Recalage
   → Differential Evolution (global) + Nelder-Mead (local)
   → Durée : 1-5 minutes selon la complexité
   → Résultat dans l'onglet "Résultat"

8. Exporter
   → JSON avec transformation (tx, ty, angle) + IoU/Dice
   → PNGs des masques et du DRR
```

## Paramètres DRR clés

| Paramètre | Valeur recommandée | Description |
|---|---|---|
| LAO/RAO | 0° pour PA pure | Rotation dans le plan coronal |
| Cran/Caud | 0° pour PA pure | Inclinaison craniale/caudale |
| Résolution | 512 px | 256 = rapide, 512 = qualité |

## Structure

```
registration_app/
├── app.py                  # Application PyQt5 principale
├── requirements.txt
├── core/
│   ├── drr_generator.py    # Génération DRR (Beer-Lambert + CLAHE)
│   └── registration.py     # Recalage IoU 3-DOF
└── README.md
```

## Interprétation des résultats

- **IoU > 0.6** : excellent recalage
- **IoU 0.35–0.6** : bon, vérifier visuellement
- **IoU < 0.35** : vérifier l'axe de projection ou redessiner les masques

Si le DRR a la colonne à l'horizontale :
→ Essayer LAO = 90° ou modifier `ap_axis` manuellement dans `core/drr_generator.py`
