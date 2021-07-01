"""Custom pyvista plotting methods."""
import numpy as np
import pyvista as pv

from nova.utilities.time import clock


class Plotter:
    """Custom pyvista plotting methods."""

    def warp(self, displace: str, scalars: str,
             factor=75, opacity=0.5, plotter=None, **camera):
        """Plot warped with mesh."""
        if plotter is None:
            plotter = pv.Plotter()
        #for key in camera:
        #    setattr(plotter.camera, key, camera[key])
        if opacity > 0:
            plotter.add_mesh(self.mesh, scalars=None, color='w',
                             opacity=opacity, smooth_shading=True)

        self.mesh['disp'] = self.mesh[displace] - self.mesh['disp-5']
        self.mesh[scalars] = 1e-6*(self.mesh[scalars] - self.mesh['vm-5'])
        warp = self.mesh.warp_by_vector('disp', factor=factor)
        plotter.add_mesh(warp, scalars=scalars, smooth_shading=True,
                         show_scalar_bar=False, clim=[-25, 25])
        #plotter.show()
        return plotter

    def animate(self, filename: str, scalars: str, max_factor=100, frames=31,
                view='iso'):
        """Animate warped displacments."""
        plotter = pv.Plotter(notebook=False, off_screen=True)
        #window_size=[400, 400], multi_samples=8,
        #line_smoothing=True

        reference = self.mesh.copy()
        plotter.add_mesh(reference, color='w', opacity=0.5,
                         smooth_shading=True)
        plotter.add_mesh(self.mesh, scalars=scalars, smooth_shading=False)
        #plotter.camera.zoom(1.5)
        #if view != 'iso':
        #    getattr(plotter, f'view_{view}')()
        #    plotter.camera.zoom(5)
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
