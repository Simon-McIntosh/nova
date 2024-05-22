"""Calculate contour trees using the TTK."""

import numpy as np
import pyvista

from nova.imas.equilibrium import EquilibriumData

pulse, run = 135013, 2

equilibrium = EquilibriumData(pulse, run, occurrence=0)

equilibrium.time = 300
equilibrium.plot_2d("psi", mask=0)

grid = pyvista.StructuredGrid(
    equilibrium.data.r2d.data,
    np.zeros_like(equilibrium.data.r2d.data),
    equilibrium.data.z2d.data,
)
grid["psi"] = equilibrium["psi2d"].flatten("F")
grid = grid.triangulate()

contour = grid.contour(51)

plotter = pyvista.Plotter()
plotter.add_mesh(grid, opacity=0.85, show_edges=True)
plotter.add_mesh(contour, color="black", line_width=5)
plotter.show()

grid.save("contour.vtk")
