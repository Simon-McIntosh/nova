
import pyvista as pv

from nova.structural.ansyspost import AnsysPost
from nova.structural.fiducialcoil import FiducialCoil
from nova.structural.morph import Morph

# morph.mesh = morph.mesh.slice(normal=[0, 0, 1])

fiducialcoil = FiducialCoil('fiducial', 10)
#fiducialcoil.mesh = fiducialcoil.mesh.slice((0, 0, 1))

base = AnsysPost('TFCgapsG10', 'k0', 'case_il')

#base.mesh = base.mesh.slice(normal=[0, 0, 1])

morph = Morph(fiducialcoil.mesh, base.mesh)

#morph.mesh = morph.mesh.slice(normal=[0, 0, 1])
morph.mesh.plot()


'''
plotter = pv.Plotter()
plotter.add_mesh(fiducialcoil.mesh.warp_by_vector('delta', factor=500),
                 color='b')
plotter.add_mesh(morph.mesh.warp_by_vector('delta', factor=500),
                 color='r')
plotter.show()
'''
