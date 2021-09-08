
import numpy as np
import pyvista as pv
import scipy.interpolate

from nova.structural.ansyspost import AnsysPost
from nova.structural.fiducial import FiducialData

ansys = AnsysPost('TFCgapsG10', 'k0', 'E_TF1')
fiducial = FiducialData()

fea_mesh = pv.PolyData()
for i in range(1, 19):
    if i in fiducial.data.coil:
        fea_mesh += ansys.mesh
    ansys.mesh.rotate_z(20)

index = np.unique(fiducial.mesh.points, axis=0, return_index=True)[1]
rbf = scipy.interpolate.RBFInterpolator(
    fiducial.mesh.points[index], fiducial.mesh['delta'][index])

fea_mesh['delta'] = rbf(fea_mesh.points)

plotter = pv.Plotter()
plotter.add_mesh(fea_mesh, color='w', opacity=0.05)
warp = fea_mesh.warp_by_vector('delta', factor=500)
plotter.add_mesh(warp)

#plotter.add_mesh(fiducial.mesh, line_width=8)
plotter.show()
