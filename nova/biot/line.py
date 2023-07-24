import numpy as np

from nova.frame.coilset import CoilSet
from nova.geometry.polyline import PolyLine


coilset = CoilSet(available=["vtk"], delta=-1)

theta = np.linspace(0, 6 * np.pi, 200)
path = np.c_[2 * np.cos(theta), np.linspace(0, 3, len(theta)), 3 * np.sin(theta)]

coilset.winding.insert({"e": [0, 0, 0.25, 0.2]}, path, part="pf")

# coilset.frame.vtkplot()

polyline = PolyLine(path)
