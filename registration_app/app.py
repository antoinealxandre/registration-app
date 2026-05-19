"""Application launcher.

Run this file to start the 2D/3D registration UI.
"""

import sys
import types


def _patch_vtk_matplotlib():
    """Stube ``vtkRenderingMatplotlib`` quand sa DLL est manquante.

    Sur certaines installations VTK + Python 3.13 (Windows), la DLL
    ``vtkRenderingMatplotlib`` est absente du paquet ; PyVista l'importe
    aveuglément au chargement de ``pyvista.plotting`` et plante. Ce binding
    ne sert que pour les labels math-text en 3D — non critique.
    """
    name = 'vtkmodules.vtkRenderingMatplotlib'
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = types.ModuleType(name)


_patch_vtk_matplotlib()

from ui.main_window import main


if __name__ == '__main__':
    sys.exit(main())
