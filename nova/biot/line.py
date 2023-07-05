from nova.frame.coilset import CoilSet

coilset = CoilSet(available=["vtk"])


coilset.coil.insert(4, 0.3, 0.1, 0.1)

coilset.frame.vtkplot()
