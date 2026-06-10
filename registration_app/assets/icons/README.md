# Icônes custom de la barre latérale

Dépose ici tes icônes personnalisées. Le logiciel les charge **automatiquement**
au démarrage si elles portent le bon nom. Si un fichier est absent, une icône
placeholder (Material Icons monochrome) est utilisée à la place.

## Noms de fichiers attendus

| Fichier            | Bouton                                   |
|--------------------|------------------------------------------|
| `panel.*`          | Afficher / masquer la barre latérale     |
| `data.*`           | Importation des données uniquement       |
| `drr.*`            | Génération DRR uniquement                |
| `annot.*`          | Annotations uniquement                   |
| `tavi.*`           | TAVI (stent + risque) uniquement         |
| `reg.*`            | Recalage uniquement                      |
| `all.*`            | Tous les menus (défaut)                  |

`*` = extension. Ordre de priorité : **`.svg` > `.png` > `.ico`**.

## Formats recommandés

- **SVG** (recommandé) : vectoriel, net à toute taille. Garde un `viewBox`
  carré (ex. `0 0 24 24`). La couleur du fichier est conservée telle quelle
  (contrairement aux placeholders monochromes).
- **PNG** : transparent, **carré**, idéalement **48×48** ou **64×64** px
  (affiché à 22×22, le surplus sert au rendu net sur écrans HiDPI).
- **ICO** : accepté mais moins souple ; préfère SVG/PNG.

## Conseils de design

- Icônes **monochromes claires** (~`#8892b0` / blanc cassé) pour rester
  lisibles sur le fond sombre `#0c0e14`.
- Garde une marge interne (~10 %) : l'icône ne doit pas toucher les bords.
- Style **flat / outline**, cohérent entre les 6 icônes.

## Où trouver des icônes gratuites

- Material Symbols : https://fonts.google.com/icons (export SVG)
- Lucide : https://lucide.dev (SVG, style outline cohérent)
- Tabler Icons : https://tabler.io/icons

Après avoir déposé les fichiers, **relance l'application** pour les voir.
