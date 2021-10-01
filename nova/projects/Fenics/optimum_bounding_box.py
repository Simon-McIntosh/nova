
import os

import gmsh
import numpy as np
import open3d as o3d
import pyvista as pv
from scipy.spatial.transform import Rotation

from nova.definitions import root_dir

fname = os.path.join(root_dir,
                     'input/geometry/ITER/sheild/EQ_sheild_A_row37.stp')


gmsh.initialize()
gmsh.clear()
shapes = gmsh.model.occ.importShapes(fname, format="step")
gmsh.model.occ.synchronize()
gmsh.model.mesh.generate(3)

points = pv.PolyData()
mesh = pv.PolyData()
for part in gmsh.model.getEntities(3):
    nodes = gmsh.model.mesh.get_nodes(*part, True)[1].reshape(-1, 3)
    points += pv.PolyData(nodes)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nodes)
    obb = pcd.get_oriented_bounding_box()
    #rotvec = Rotation.from_matrix(np.linalg.inv(obb.R)).as_rotvec(True)
    factor = (gmsh.model.occ.get_mass(*part) / np.prod(obb.extent))**(1 / 3)
    print(factor)
    extent = factor * obb.extent
    bounds = np.zeros(6)
    bounds[::2] = -extent/2
    bounds[1::2] = extent/2
    box = pv.Box(bounds)
    box.points = box.points @ np.linalg.inv(obb.R)
    box.points += obb.center
    #box.rotate_vector(rotvec, 2*np.pi*np.linalg.norm(rotvec), obb.center)

    mesh += box

    #o3d.visualization.draw_geometries([obb])

plotter = pv.Plotter()
plotter.add_mesh(points)
plotter.add_mesh(mesh, opacity=0.2)
plotter.show()

#gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)

#
#gmsh.fltk.run()
