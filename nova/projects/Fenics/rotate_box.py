    @staticmethod
    def rotate(mesh: vedo.Mesh):
        """Return PCA rotational transform."""
        mesh = mesh.fillHoles()
        points = mesh.points()
        triangles = np.array(mesh.cells())
        vertex = dict(a=points[triangles[:, 0]],
                      b=points[triangles[:, 1]],
                      c=points[triangles[:, 2]])
        normal = np.cross(vertex['b']-vertex['a'], vertex['c']-vertex['a'])
        l2norm = np.linalg.norm(normal, axis=1)
        covariance = np.cov(normal, rowvar=False, aweights=l2norm**5)
        eigen = np.linalg.eigh(covariance)[1]
        eigen /= np.linalg.det(eigen)
        return Rotation.from_matrix(eigen)

    @staticmethod
    def extent(mesh: vedo.Mesh, rotate: Rotation):
        """Return box extent."""
        points = rotate.inv().apply(mesh.points())
        extent = np.max(points, axis=0) - np.min(points, axis=0)
        extent *= (mesh.volume() / np.prod(extent))**(1 / 3)
        return extent

    @staticmethod
    def box(center: npt.ArrayLike, extent: npt.ArrayLike, rotate: Rotation):
        """Return pannel bounding box."""
        bounds = np.zeros(6)
        bounds[::2] = -extent/2
        bounds[1::2] = extent/2
        box = pv.Box(bounds)
        box.points = rotate.apply(box.points)
        box.points += center
        return vedo.Mesh(box)
    '''
    def load_mesh(self, mesh: Union[pv.PolyData, vedo.Mesh]):
        """Return pv.PolyData mesh apply requested filters (qhull, ect.)."""
        if not isinstance(mesh, (pv.PolyData, vedo.Mesh)):
            raise TypeError(f'type(mesh) {type(self.mesh)} not in '
                            '[pv.PolyData, vedo.Mesh]')
        mesh = self._polydata(mesh) 
        if self.qhull:
            return self.convex_hull(mesh)
        mesh = vedo.Mesh(mesh).decimate(
                N=6, method='pro', boundaries=True)
        return self._polydata(mesh)

    @staticmethod
    def generate_grid(mesh: pv.PolyData):
        """Generate grid."""
        tet = tetgen.TetGen(mesh)
        tet.tetrahedralize(order=1, quality=False)
        return tet.grid.compute_cell_sizes(length=False, area=False)
        
    @staticmethod
    def _polydata(mesh: Union[pv.PolyData, vedo.Mesh]):
        """Return mesh as pyvista.PolyData."""
        if isinstance(mesh, vedo.Mesh):
            return pv.PolyData(mesh.polydata())
        return mesh
    '''
    
    
