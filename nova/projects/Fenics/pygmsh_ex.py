
import io
import os

import numpy as np

import pygmsh
import pyvista as pv

from nova.definitions import root_dir

fname = os.path.join(root_dir,
                     'input/geometry/ITER/sheild/EQ_RIB_A_row37.stp')


with pygmsh.occ.Geometry() as geom:
    
    geom.import_shapes(fname)
    geom.characteristic_length_max = 1000
    mesh = geom.generate_mesh()


mesh.write('tmp.vtk')
mesh = pv.read('tmp.vtk')

#mesh = mesh.decimate_boundary(target_reduction=0.5)

mesh.plot(show_edges=True, color='w')
#with io.BytesIO() as f:
#    mesh.write(f, 'vtk')
#    mesh = pv.read(f)
