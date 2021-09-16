"""Custom pyvista plotting methods."""
import numpy as np
import pyvista as pv

from nova.utilities.time import clock


class Plotter:
    """Custom pyvista plotting methods."""

    def warp(self, factor=75, opacity=0.5, displace='delta', scalars=None,
             plotter=None, show_edges=False, color=None):
        """Plot warped with mesh."""
        if scalars is None:
            scalars = displace
        if plotter is None:
            plotter = pv.Plotter()
        if opacity > 0:
            plotter.add_mesh(self.mesh, scalars=None, color='w',
                             opacity=opacity, smooth_shading=True,
                             line_width=3)

        #self.mesh['disp'] = self.mesh[displace] - self.mesh['disp-5']
        #self.mesh[scalars] = 1e-6*(self.mesh[scalars] - self.mesh['vm-5'])
        warp = self.mesh.warp_by_vector(displace, factor=factor)
        plotter.add_mesh(warp, line_width=3,
                         show_edges=show_edges, opacity=0.5,
                         color=color)
        #, smooth_shading=True, show_scalar_bar=False, clim=[-25, 25])
        if plotter is None:
            plotter.show()
        return plotter

    def animate(self, filename: str, scalars: str, max_factor=100, frames=31,
                view='iso', opacity=0.5):
        """Animate warped displacments."""

        plotter = pv.Plotter(notebook=False, off_screen=True)

        #clip = [0, 20, 0, 20, 0, 20]
        reference = self.mesh.copy()
        #reference = reference.clip_box(clip)
        mesh = self.mesh.copy()
        #mesh = mesh.clip_box(clip)

        if opacity > 0:
            plotter.add_mesh(reference, color='w', opacity=opacity,
                             smooth_shading=True)
        plotter.add_mesh(mesh, scalars=scalars, smooth_shading=False)
        plotter.camera.zoom(1.5)
        if view != 'iso':
            getattr(plotter, f'view_{view}')()
            plotter.camera.zoom(5)
        filename += f'_{view}'
        plotter.open_gif(f'{filename}.gif')

        # Update Z and write a frame for each updated position
        factors = np.linspace(0, max_factor, frames // 2)
        factors = np.append(factors, factors[::-1][1:])
        tick = clock(len(factors), header='Generating displacement gif.')
        for factor in factors:
            warp = reference.warp_by_vector(scalars, factor=factor)
            plotter.update_coordinates(warp.points, render=False)
            # plotter.update_scalars(warp[scalars], render=False)
            # plotter.mesh.compute_normals(cell_normals=False, inplace=True)
            plotter.render()
            plotter.write_frame()
            tick.tock()
        plotter.close()
