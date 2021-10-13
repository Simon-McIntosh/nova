from dataclasses import dataclass, field
import os

import numpy as np
import pyacvd
import pyvista as pv
import scipy.spatial
import tetgen
import vedo

from nova.definitions import root_dir
from nova.utilities.time import clock


@dataclass 
class MultiBlock:
    
    file: str 
    path: str 
    
    def __post_init__(self):
        """Split extension from vtmb path."""
        self.path = os.path.splitext(self.path)[0]

    def read(self, index):
        """Return vtu mesh."""
        return pv.read(os.path.join(self.path, f'{self.file}_{index}.vtu'))
    
    
@dataclass
class Shield:

    file: str = 'IWS_S6_BLOCKS'
    path: str = None
    mesh: pv.PolyData = field(init=False, repr=False)
    geom: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        if self.path is None:
            self.path = os.path.join(root_dir, 'input/geometry/ITER/shield')
        self.load_mesh()
        self.load_geom()

    @property
    def vtk_file(self):
        """Retun full vtk filename."""
        return os.path.join(self.path, f'{self.file}.vtk')

    @property
    def vtmb_file(self):
        """Retun full vtmb filename."""
        return os.path.join(self.path, f'{self.file}.vtmb')

    @property
    def stl_file(self):
        """Return full stl filename."""
        return os.path.join(self.path, f'{self.file}.stl')

    def load_mesh(self):
        """Load mesh."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            self.mesh = self.load_vtk()

    def load_vtk(self):
        """Load vtk mesh from file."""
        mesh = pv.read(self.stl_file)
        mesh.save(self.vtk_file)
        return mesh
        
    def vtu_file(self, index):
        return os.path.join(os.path.splitext(self.vtmb_file)[0],
                            f'{self.file}_{index}.vtu')

    def load_geom(self):
        """Extract convexhull for each pannel."""
        self.mesh = pv.PolyData()
        self.box = pv.PolyData()
        multiblockmesh = self.load_multiblock()
        
        """
        multiblock = MultiBlock(self.file, self.vtmb_file)
        
        n_mesh = len(multiblockmesh)
        n_mesh = 1
        tick = clock(n_mesh, header='loading orientated bounding boxes')

        for i in range(n_mesh):
            
            mesh = multiblock.read(i)
            triangles = np.column_stack(
                (3*np.ones((mesh.n_cells, 1), dtype=int), 
                 mesh.cell_connectivity.reshape(-1, 3))).flatten()
            mesh = pv.UnstructuredGrid(triangles, 5*np.ones(mesh.n_cells),
                                       mesh.points)
            mesh = pv.PolyData(mesh.points, triangles)
            mesh.compute_normals(inplace=True)  
            
            print(mesh['Normals'].shape)
            warp = mesh.warp_by_scalar('Normals', factor=50)
            
            warp.plot()
            '''
            mesh = mesh.extract_surface()
            mesh.
            
            vertex = dict(a=mesh.points[mesh.cell_connectivity[::3]],
                          b=mesh.points[mesh.cell_connectivity[1::3]], 
                          c=mesh.points[mesh.cell_connectivity[2::3]])
            tet_volume = np.einsum("ij, ij->i", vertex['a'], 
                                   np.cross(vertex['b'], vertex['c']))
            volume = np.sum(tet_volume)
            tet_volume.shape = -1, 1
            tet_center = (vertex['a'] + vertex['b'] + vertex['c']) / 4

            center = np.sum(tet_center * tet_volume, axis=0) / volume

            
            #center_of_mass[i] = mesh.center_of_mass()
            #volume[i] = mesh.volume

            hull = scipy.spatial.ConvexHull(mesh.points)
            faces = np.column_stack((3*np.ones((len(hull.simplices), 1),
                                               dtype=int), hull.simplices))
            qhull = pv.PolyData(hull.points, faces.flatten())
            
            
            points = qhull.cell_centers().points
            qhull = qhull.compute_cell_sizes(length=False, volume=False)
            covariance = np.cov(points, rowvar=False, ddof=0,
                                aweights=qhull['Area'])
            eigen_vectors = np.linalg.eigh(covariance)[1]
    
            points = points @ eigen_vectors
            extent = np.max(points, axis=0) - np.min(points, axis=0)
    
            bounds = np.zeros(6)
            bounds[::2] = -extent/2
            bounds[1::2] = extent/2
            box = pv.Box(bounds)
            box.points = box.points @ eigen_vectors.T
            box.points += center
            self.box += box
            self.mesh += mesh
            tick.tock()
        #self.cell = pv.PolyData(center_of_mass)
        #self.cell['volume'] = volume
        '''
        """
        
    @staticmethod
    def center(mesh: vedo.Mesh):
        """Return center of mass."""
        tet = tetgen.TetGen(pv.PolyData(mesh.polydata()))
        tet.tetrahedralize(order=1, quality=False)
        grid = tet.grid.compute_cell_sizes(length=False, area=False)
        return np.sum(grid['Volume'].reshape(-1, 1) * 
                      grid.cell_centers().points, axis=0) / grid.volume
    
    @staticmethod 
    def rotate(mesh: vedo.Mesh):
        """Return PCA rotational transform."""
        points = mesh.points()
        triangles = np.array(mesh.cells())
        vertex = dict(a=points[triangles[:, 0]],
                      b=points[triangles[:, 1]], 
                      c=points[triangles[:, 2]])
        normal = np.cross(vertex['b']-vertex['a'], vertex['c']-vertex['a'])
        
        covariance = np.cov(normal, rowvar=False, ddof=0,
                            aweights=np.linalg.norm(normal, axis=1))
        eigen_vectors = np.linalg.eigh(covariance)[1]
        return scipy.spatial.transform.Rotation.from_matrix(eigen_vectors.T)
        
        
    def load_multiblock(self):
        """Retun multiblock mesh."""
        mesh = vedo.Mesh(self.vtk_file)
        mesh = mesh.splitByConnectivity(1)[0]

        center = self.center(mesh)
        rotate = self.rotate(mesh)
        


        points = points @ eigen_vectors
        extent = np.max(points, axis=0) - np.min(points, axis=0)
        extent *= (mesh.volume() / np.prod(extent))**(1 / 3)

        bounds = np.zeros(6)
        bounds[::2] = -extent/2
        bounds[1::2] = extent/2
        box = pv.Box(bounds)
        box.points = box.points @ eigen_vectors.T
        box.points += center
        box = vedo.Mesh(box).opacity(0.5)
        
        center = vedo.Point(center).c('b')

        
        vedo.show(mesh.opacity(0.5).c('p'), center, box)
        
        #grid.plot(show_edges=True)

        '''
        self.mesh = mesh
        
        
        mesh = mesh.opacity(0.5).wireframe()
        
        center_of_mass = vedo.Point(mesh.centerOfMass(), c='y')
        
        
        
        vedo.show(mesh, center_of_mass, center)
        
        '''
        
        
        '''
        try:
            return pv.read(self.vtmb_file)
        except FileNotFoundError:
            mesh = self.load_vtk()
            multiblock = mesh.split_bodies()
            multiblock.save(self.vtmb_file)
            return multiblock
        '''

    def plot(self):
        """Plot mesh."""
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh, color='r', opacity=1)
        plotter.add_mesh(self.box, color='g', opacity=0.75, show_edges=True)

        #plotter.add_mesh(self.cell, color='b', opacity=1)
        plotter.show()


if __name__ == '__main__':

    shield = Shield('IWS_S6_BLOCKS')
    #shield.plot()
    #shield.load_stl()
