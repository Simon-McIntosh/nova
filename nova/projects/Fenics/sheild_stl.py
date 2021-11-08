from dataclasses import dataclass, field
import os

import pyvista as pv
import vedo

from nova.definitions import root_dir
from nova.electromagnetic.framespace import FrameSpace
from nova.utilities.time import clock


@dataclass
class Shield:
    """Manage shield sector."""

    file: str = 'IWS_FM_PLATE_S4'
    path: str = None
    mesh: pv.PolyData = field(init=False, repr=False)
    geom: pv.PolyData = field(init=False, repr=False)

    def __post_init__(self):
        """Load datasets."""
        if self.path is None:
            self.path = os.path.join(root_dir, 'input/geometry/ITER/shield')
        self.load_mesh()
        self.load_frame()

    @property
    def vtk_file(self):
        """Retun full vtk filename."""
        return os.path.join(self.path, f'{self.file}.vtk')

    @property
    def stl_file(self):
        """Return full stl filename."""
        return os.path.join(self.path, f'{self.file}.stl')

    def load_mesh(self):
        """Load mesh."""
        try:
            self.mesh = pv.read(self.vtk_file)
        except FileNotFoundError:
            self.mesh = self.read_stl()

    def read_stl(self):
        """Read stl file."""
        mesh = pv.read(self.stl_file)
        mesh.save(self.vtk_file)
        return mesh

    def load_frame(self):
        """Retun multiblock mesh."""
        mesh = vedo.Mesh(self.vtk_file).scale(1e-3)
        parts = mesh.splitByConnectivity(3)

        frame = FrameSpace(label='fi', body='panel')
        
        #parts = [vedo.Mesh(pv.read('tmp.vtk'))]
        blocks = []

        tick = clock(len(parts), header='loading decimated convex hulls')
        for i, part in enumerate(parts):
            #pv.PolyData(part.polydata()).save('tmp.vtk')
            #tri = TriPanel(part)

            frame += dict(vtk=part, body='stl') #tri.frame

            #blocks.append(tri.mesh.opacity(1).c(i))
            #blocks.append(tri.panel.opacity(1).c(i+1))
            #blocks.append(tri.panel.opacity(1).c(i+2))

            #tet = TetPanel(tri.panel)

            #print(tri.volume, tet.volume)
            #print(block.center_mass)
            '''
            part.cap()
            convex_hull = self.convex_hull(part)
            blocks.append(convex_hull.c('b').opacity(1))
            try:
                center = self.center(part)
            except RuntimeError:  # Failed to tetrahedralize (non-manifold)
                self.part = part.clone()
                center = self.center(part.decimate(0.1))
            #center = self.center(convex_hull)
            #center = part.centerOfMass()
            #rotate = self.rotate(m)
            #extent = self.extent(m, rotate)
            #box.append(self.box(center, extent, rotate))
            '''
            tick.tock()
        self.frame = frame
        #self.frame.store('tmp', 'frame')
        #self.frame.load('tmp', 'frame')
        vedo.show(frame.vtk)

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
