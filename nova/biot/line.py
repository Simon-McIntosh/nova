import numpy as np

from nova.frame.coilset import CoilSet


coilset = CoilSet(available=["vtk"], delta=-50)


coilset.coil.insert(4, 0.3, 0.1, 0.1)

theta = np.linspace(0, 2 * np.pi)
path = np.c_[2 * np.cos(theta), 2 * np.sin(theta), np.zeros_like(theta)]

coilset.winding.insert({"rectangle": [0, 0, 0.5, 1.2]}, path, part="pf")

coilset.frame.vtkplot()
