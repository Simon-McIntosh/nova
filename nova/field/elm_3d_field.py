from tqdm import tqdm

from nova.frame.coilset import CoilSet
from nova.imas.coils_non_axisymmetric import CoilsNonAxisymmetric

attrs = ["Bx", "By", "Bz", "Br", "Bphi"]
datasource = {
    "CC": (111003, 3),
    "VS3": (115003, 2),
    "ELM": (115001, 2),
    "CS": (111004, 2),
    "PF": (111005, 1),
    "TF": (111002, 2),
}

coilset = CoilSet(filename="overlap", field_attrs=attrs)

try:
    coilset.load()
except FileNotFoundError:
    for coil, pulse_run in tqdm(datasource.items()):
        coilset += CoilsNonAxisymmetric(*pulse_run, field_attrs=attrs)
    coilset.store()

coilset.frame.vtkplot()


"""
ids = CoilsNonAxisymmetric(111003, 3, field_attrs=["Bx", "By", "Bz", "Br", "Bphi"])

# ids = CoilsNonAxisymmetric(115001, 2, field_attrs=["Bx", "By", "Bz", "Br", "Bphi"])
# ids = CoilsNonAxisymmetric(111006, 1, field_attrs=["Bx", "By", "Bz", "Br", "Bphi"])
# ids = CoilsNonAxisymmetric(111004, 2, field_attrs=["Bx", "By", "Bz", "Br", "Bphi"])


ids.saloc["Ic"] = 1


x = np.linspace(-12, 12, 100)
X, Y = np.meshgrid(x, x, indexing="xy")

grid = xarray.Dataset(coords=dict(zip("xyz", [x, x, [0.0]])))
grid["X"] = ("x", "y", "z"), X[..., np.newaxis]
grid["Y"] = ("x", "y", "z"), Y[..., np.newaxis]
grid["Z"] = ("x", "y", "z"), np.zeros_like(X)[..., np.newaxis]


solve = True
if solve:
    ids.grid.solve(grid=grid)
    # ids.grid.solve(4e3, 1)
    # ids.store()

ids.plot()
ids.set_axes("2d")
ids.grid.plot("bphi", coords="xy", index=(..., 0), levels=100)
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
contours = mesh.contour(isosurfaces=750, scalars=ids.grid.bphi.reshape(-1))

ids.frame.vtkplot(new=True)  # index=["EU9B", "EE9B", "EL9B"])
vedo.Mesh(contours, c="black").show(new=False)
"""
