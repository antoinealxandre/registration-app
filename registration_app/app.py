import sys
import types


def _patch_vtk_matplotlib():

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
