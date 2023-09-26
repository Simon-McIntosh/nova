"""Manage CAD centerlines."""
from dataclasses import dataclass, field
import os

import pandas
import xarray

from nova.geometry.polygeom import Polygon
from nova.geometry.polyline import PolyLine
from nova.graphics.plot import Plot
from nova.imas.coil import part_name
from nova.imas.machine import CoilDatabase


@dataclass
class Centerline(Plot, CoilDatabase):
    r"""Extract coil centerlines from CAD traces.

    Centerline source data is recived via email and and stored in a shared folder at:
    \\\\io-ws-ccstore1\\ANSYS_Data\\mcintos\\coil_centerlines

    datadir : str
        Data directory. Set as mount point location to access IO shared folder
    """

    filename: str = ""
    datadir: str = "/mnt/share/coil_centerlines"
    polygon: Polygon = field(default_factory=lambda: Polygon({"o": [0, 0, 0.1]}))

    @property
    def xls_file(self):
        """Return xls filename."""
        return os.path.join(self.datadir, f"{self.filename}.xlsx")

    def build(self):
        """Load points from file and build coil centerline."""
        self.data = 1e-3 * xarray.Dataset()
        self.data.coords["point"] = list("xyz")
        with pandas.ExcelFile(self.xls_file, engine="openpyxl") as xls:
            points = self._read_sheet(xls).to_numpy()
            self.data["points"] = ("index", "point"), points
        self.polyline = PolyLine(points)
        self.winding.insert(
            self.polygon, points, name=self.filename, part=part_name(self.filename)
        )

    def _read_sheet(self, xls, sheet_name=0):
        """Read excel worksheet."""
        sheet = pandas.read_excel(xls, sheet_name, usecols=[2, 3, 4])
        columns = {"X Coord": "x", "Y Coord": "y", "Z Coord": "z"}
        sheet.rename(columns=columns, inplace=True)
        return sheet


if __name__ == "__main__":
    centerline = Centerline(filename="CC1-4")
    # centerline.build()
