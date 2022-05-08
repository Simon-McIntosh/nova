"""Custom pyvista plotting methods."""
import numpy as np
import pyvista as pv

from nova.utilities.time import clock


class Plotter:
    """Custom pyvista plotting methods."""

    def diff(self, displace: str, reference: str):
        """Diffrence array and return name."""
        name = f'{displace}-{reference}'
        if name not in self.mesh.array_names:
            self.mesh[name] = self.mesh[displace] - self.mesh[reference]
            self.mesh.set_active_scalars(name)
        return name

    def plot(self, scalars=None, opacity=1):
        """Plot mesh."""
        plotter = pv.Plotter()
        plotter.add_mesh(self.mesh, scalars=scalars, color='w',
                         opacity=opacity, smooth_shading=False,
                         line_width=3)
        #plotter.show_bounds()
        plotter.show()
        #return plotter

    def warp(self, factor=75, opacity=0.5, displace=None, scalars=None,
             plotter=None, show_edges=False, color=None, show=False,
             view='iso', clim=None):
        """Plot warped with mesh."""
        if displace is None:
            displace = self.mesh.active_scalars_name
        if scalars is None:
            scalars = displace
        if plotter is None:
            plotter = pv.Plotter()
            show = True
        if opacity > 0:
            plotter.add_mesh(self.mesh, scalars=None, color='w',
                             opacity=opacity, smooth_shading=True,
                             line_width=3, clim=[1000, 2000])

        #self.mesh['disp'] = self.mesh[displace] - self.mesh['disp-5']
        #self.mesh[scalars] = 1e-6*(self.mesh[scalars] - self.mesh['vm-5'])
        warp = self.mesh.warp_by_vector(displace, factor=factor)
        plotter.add_mesh(warp, scalars=scalars, line_width=3,
                         show_edges=show_edges, opacity=1,
                         color=color)
        #, smooth_shading=True, show_scalar_bar=False, clim=[-25, 25])
        if view != 'iso':
            getattr(plotter, f'view_{view}')()
        if show:
            plotter.show()
        return plotter

    def animate(self, filename: str, scalars: str, max_factor=100, frames=31,
                view='iso', opacity=0.5, zoom=1.5):
        """Animate warped displacments."""

        plotter = pv.Plotter(notebook=False, off_screen=True)
        plotter.set_background('white')

        #clip = [0, 20, 0, 20, 0, 20]
        reference = self.mesh.copy()
        #reference = reference.clip_box(clip)
        mesh = self.mesh.copy()
        #mesh = mesh.clip_box(clip)

        if opacity > 0:
            plotter.add_mesh(reference, color='k', opacity=opacity,
                             smooth_shading=True)
        plotter.add_mesh(mesh, scalars=scalars, smooth_shading=False,
                         show_scalar_bar=False)
        plotter.camera.zoom(zoom)
        if view != 'iso':
            getattr(plotter, f'view_{view}')()
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
        #plotter.close()
