"""Decimate Gauss coilset."""

import appdirs
import numpy as np
import pathlib
import pandas as pd
import xarray as xr
import pyvista as pv
import vedo

from nova.geometry.frenet import Frenet
from nova.geometry.polygeom import Polygon
from nova.geometry.polyline import PolyLine
from nova.frame.coilset import CoilSet

path = pathlib.Path(appdirs.user_data_dir("gauss", "nova"))

with pd.ExcelFile(path / "helias4-filaments.xlsx") as xls:
    coil_data = pd.read_excel(
        xls, "coils", header=None, names=("x", "y", "z", "coil"), usecols=[0, 1, 2, 4]
    )
    coil_index = np.array(
        [
            i + 1
            for i, coil in enumerate(coil_data.loc[:, "coil"])
            if isinstance(coil, str)
        ]
    )


coil_data.loc[:, "coil"] = coil_data.coil.bfill().apply(
    lambda coil: int(coil.split()[0])
)
coil_data.loc[:, "name"] = np.searchsorted(coil_index, coil_data.index, side="right")
coil_data.loc[:, "point"] = coil_data.index - (
    coil_index[coil_data.name.values] - coil_index[0]
)
coil_data.set_index(["name", "point"], inplace=True)

# coil = vedo.CSpline(coil_data.loc[0, ["x", "y", "z"]].values, res=60, closed=True)
# coil.show()

arc_eps = 6e-3
minimum_arc_nodes = 4

coilset = CoilSet(field_attrs=["Bx", "By", "Bz", "Br", "Bphi"])
for coil_index in coil_data.index.get_level_values(0).unique()[:2]:
    coilset.winding.insert(
        coil_data.loc[coil_index, ["x", "y", "z"]].values,
        Polygon({"rectangle": [0, 0, 0.4, 0.6]}),
        arc_eps=arc_eps,
        minimum_arc_nodes=minimum_arc_nodes,
        align="twist",
        # name=coil_name,
        # part=part_name(coil_name),
        delim="-",
        # **self.polyline_attrs,
    )

print(coilset.frame)

polyline = PolyLine(
    coil_data.loc[0, ["x", "y", "z"]].values,
    arc_eps=arc_eps,
    line_eps=5e-2,
    rdp_eps=1e-3,
    minimum_arc_nodes=minimum_arc_nodes,
)

polyline.plot()
polyline.axes.set_title("rg")
polyline.plt.show()


axes = polyline.set_axes("1d", nrows=3)

polyline.frenet.plot_components(axes=axes)

path_length = np.r_[0, np.cumsum(polyline.path[1:] - polyline.path[:-1])]
frenet = Frenet(polyline.path)
# frenet.plot_components(axes=axes)
# frenet.plt.show()


# frenet.set_axes("1d")
# frenet.axes.plot(frenet.parametric_length, frenet.twist)
# frenet.plt.show()

# coilset.frame.vtkplot()
# frenet.plot(scale=3)


coilset.grid.solve(5e3, 0.5)

coilset.grid.set_axes("2d")
coilset.plot()
coilset.sloc["Ic"] = 1
coilset.grid.plot("bz", coords="xz", levels=100)
coilset.grid.plt.show()


"""
pl = pv.Plotter()
for name in coil_data.index.levels[0]:
    spline = pv.Spline(coil_data.loc[name, ["x", "y", "z"]], 250)
    pl.add_mesh(
        spline,
        render_lines_as_tubes=True,
        line_width=10,
    )
pl.show()
"""
