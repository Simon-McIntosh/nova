import numpy as np
import pyvista as pv
import vedo
import xarray


from nova.imas.coils_non_axisymmetric import CoilsNonAxisymmetric


# ids = CoilsNonAxisymmetric(115001, 2, field_attrs=["Bx", "By", "Bz", "Br"])
ids = CoilsNonAxisymmetric(111006, 1, field_attrs=["Bx", "By", "Bz", "Br", "Bphi"])


ids.saloc["Ic"] = 1


x = np.linspace(-4, 4, 250)
X, Y = np.meshgrid(x, x, indexing="xy")

grid = xarray.Dataset(coords=dict(zip("xyz", [x, x, [0.0]])))
grid["X"] = ("x", "y", "z"), X[..., np.newaxis]
grid["Y"] = ("x", "y", "z"), Y[..., np.newaxis]
grid["Z"] = ("x", "y", "z"), np.zeros_like(X)[..., np.newaxis]


solve = True
if solve:
    ids._clear()
    ids.grid.solve(grid=grid)
    ids.store()

ids.plot()
ids.set_axes("2d")
ids.grid.plot("bphi", coords="xy", index=(..., 0), levels=500)
ids.grid.axes.plot(0, 0, "r.")


points = np.stack(
    [
        ids.grid.data.X[..., 0],
        ids.grid.data.Y[..., 0],
        np.zeros_like(ids.grid.data.X[..., 0]),
    ],
    axis=-1,
).reshape(-1, 3)

mesh = pv.PolyData(points).delaunay_2d()
contours = mesh.contour(isosurfaces=31, scalars=ids.grid.bphi.reshape(-1))

ids.frame.vtkplot()  # index=["EU9B", "EE9B", "EL9B"])
vedo.Mesh(contours, c="black").show(new=False)
