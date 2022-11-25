__all__ = [
    "alphashape",
    "meshio",
    "pyvista",
    "Trimesh",
    "trimesh_interfaces"
    "vedo",
    ]

import lazy_loader as lazy

alphashape = lazy.load('alphashape')
meshio = lazy.load('meshio')
pyvista = lazy.load('pyvista')
Trimesh = lazy.load('trimesh.Trimesh')
trimesh_interfaces = lazy.load('trimesh.interfaces')
vedo = lazy.load('vedo')

#__getattr__, __dir__, __stub__ = lazy.attach_stub(__name__, __file__)
#__all__.append(__stub__)
