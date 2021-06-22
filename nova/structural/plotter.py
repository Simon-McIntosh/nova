"""Custom pyvista plotting methods."""
import pyvista as pv


class Plotter:
    """Custom pyvista plotting methods."""

    def warp(self, scalars: str, factor=75, opacity=0.5):
        """Plot warped with mesh."""
        if factor == 0:
            return self.mesh.plot(scalars=scalars)
        plotter = pv.Plotter()
        if opacity > 0:
            plotter.add_mesh(self.mesh, scalars=None, color='w',
                             opacity=opacity)
        warp = self.mesh.warp_by_vector(scalars, factor=factor)
        plotter.add_mesh(warp, scalars=scalars)
        plotter.show()
