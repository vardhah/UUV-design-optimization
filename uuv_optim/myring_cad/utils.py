import os
import sys


def verify_freecad_existence():
    freecad_libs = [
        "/usr/local/lib/FreeCAD.so",
        "/usr/lib/freecad-python3/lib/FreeCAD.so",
    ]

    for lib in freecad_libs:
        if os.path.exists(lib):
            path = os.path.dirname(lib)
            if path not in sys.path:
                sys.path.append(path)
            break
    else:
        raise ValueError("FreeCAD library was not found!")
