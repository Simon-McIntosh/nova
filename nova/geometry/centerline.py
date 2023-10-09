"""Manage CAD centerlines."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar
import os

import pandas
from tqdm import tqdm
import xarray

from nova.geometry.polygeom import Polygon
from nova.graphics.plot import Plot
from nova.imas.coil import part_name
from nova.imas.database import Database, Ids, IdsEntry
from nova.imas.machine import CoilDatabase
from nova.imas.metadata import Metadata, Properties


@dataclass
class Centerline(Plot, Properties, CoilDatabase):
    r"""Extract coil centerlines from CAD traces.

    Centerline source data is recived via email and and stored in a shared folder at:
    \\\\io-ws-ccstore1\\ANSYS_Data\\mcintos\\coil_centerlines

    datadir : str
        Data directory. Set as mount point location to access IO shared folder
    """

    filename: str = ""
    cross_section: Polygon | dict[str, list] = field(
        default_factory=lambda: {"o": [0, 0, 0.1, 0.1, 3]}
    )
    minimum_arc_nodes: int = 4
    quad_segs: int = 32
    arc_eps: float = 1e-3
    line_eps: float = 2e-3
    rdp_eps: float = 1e-4
    datadir: str = "/mnt/share/coil_centerlines"

    centerline_attrs: ClassVar[list[str]] = [
        "pulse",
        "run",
        "machine",
        "occurrence",
        "comment",
        "source",
        "provider",
    ]

    def __post_init__(self):
        """Format cross_section."""
        if isinstance(self.cross_section, dict):
            self.cross_section = Polygon(self.cross_section)
        for attr in self.centerline_attrs:
            if (value := getattr(self, attr)) is not None and value != 0:
                self.ids_metadata[attr] = value
        self._check_status()
        super().__post_init__()

    @cached_property
    def ids_metadata(self):
        """Return ids metadata."""
        metadata = {
            "CC_EXTRATED_CENTERLINES": {
                "status": "active",
                "pulse": 111004,
                "run": 1,
                "occurrence": 0,
                "name": "coils_non_axisymmetric",
                "system": "correction_coils",
                "comment": "Ex-Vessel Coils (EVC) Systems (CC) - conductor centerlines",
                "source": [
                    "Reference: DET-07879",
                    "Objects: Correction Coils + Feeders Centerlines Extraction for "
                    "IMAS database",
                    "Filename: CC_EXTRATED_CENTERLINES.xls",
                    "Date: 05/10/2023",
                    "Provider: Vincent Bontemps, vincent.bontemps@iter.org",
                    "Contact: Guillaume Davin, Guillaume.Davin@iter.org",
                ],
            }
        }
        try:
            return metadata[self.filename]
        except KeyError as error:
            raise KeyError(
                f"Entry for {self.filename} not present in self.metadata"
            ) from error

    def _check_status(self):
        """Assert that machine description status status is set."""
        assert len(self.ids_metadata["status"]) > 0

    @property
    def _base_yaml(self):
        """Return base yaml dict."""
        return {
            attr: self.ids_metadata.get(attr, "")
            for attr in ["status", "replaced_by", "replaces", "reason_for_replacement"]
        } | {"backend": self.backend}

    @property
    def _correction_coils_yaml(self):
        """Return correction coil machine description yaml metadata."""
        return {
            "ids": "coils_non_axisymmetric",
            "pbs": "PBS-11",
            "data_provider": self.provider.split(", ")[0],
            "data_provider_email": self.provider.split(", ")[1],
            "ro": "Fabrice Simon",
            "ro_email": "Fabrice.Simon@iter.org",
            "description": self.ids_metadata["comment"],
            "provenance": self.ids_metadata["source"][0].split()[-1],
        }

    @property
    def yaml(self):
        """Return machine description yaml metadata."""
        system = self.ids_metadata["system"]
        return getattr(self, f"_{system}_yaml") | self._base_yaml

    @property
    def ids_attrs(self):
        """Return ids attributes."""
        return super().ids_attrs | {
            attr: self.ids_metadata[attr]
            for attr in ["pulse", "run", "occurrence", "name"]
        }

    @property
    def polyline_attrs(self):
        """Return polyline attributes."""
        return {
            "minimum_arc_nodes": self.minimum_arc_nodes,
            "quad_segs": self.quad_segs,
            "arc_eps": self.arc_eps,
            "line_eps": self.line_eps,
            "rdp_eps": self.rdp_eps,
        }

    @property
    def group_attrs(self) -> dict:
        """
        Return group attrs.

        Replaces :func:`~nova.imas.database.CoilDatabase`.
        """
        assert isinstance(self.cross_section, Polygon)
        return (
            self.ids_attrs
            | self.polyline_attrs
            | {
                "cross_section": self.cross_section.points,
            }
        )

    @property
    def xls_file(self):
        """Return xls filename."""
        return os.path.join(self.datadir, f"{self.filename}.xlsx")

    @property
    def metadata(self):
        """Return netCDF metadata."""
        return self.ids_attrs | self.polyline_attrs | self.ids_metadata | self.yaml

    def build(self):
        """Load points from file and build coil centerlines."""
        self.data = xarray.Dataset()
        self.data.coords["point"] = list("xyz")
        self.data["cross_section"] = (
            "cross_section_index",
            "point",
        ), self.cross_section.points
        with pandas.ExcelFile(self.xls_file, engine="openpyxl") as xls:
            self.data.coords["coil_name"] = xls.sheet_names
            xls_points = {}
            for coil_name in tqdm(xls.sheet_names, "loading coils"):
                xls_points[coil_name] = (
                    1e-3 * self._read_sheet(xls, coil_name).to_numpy()
                )
                self.winding.insert(
                    xls_points[coil_name],
                    self.cross_section,
                    name=coil_name,
                    part=part_name(coil_name),
                    delim="",
                    **self.polyline_attrs,
                )
        self._store_point_data(xls_points)
        self._store_segment_data()

        self.data.attrs = self.metadata
        # self.store()

    def _read_sheet(self, xls, sheet_name=0):
        """Read excel worksheet."""
        sheet = pandas.read_excel(xls, sheet_name, usecols=[2, 3, 4], nrows=20)
        columns = {"X Coord": "x", "Y Coord": "y", "Z Coord": "z"}
        sheet.rename(columns=columns, inplace=True)
        return sheet

    def _store_point_data(self, xls_points):
        """Store xls point data to dataset."""
        self.data["point_number"] = "coil_name", [
            len(points) for points in xls_points.values()
        ]
        self.data.coords["point_index"] = range(self.data.point_number.max().data)
        self.data["points"] = xarray.DataArray(
            0.0,
            coords=[self.data.coil_name, self.data.point_index, self.data.point],
            dims=["coil_name", "point_index", "point"],
        )
        for coil_name, points in xls_points.items():
            self.data["points"].sel(coil_name=coil_name)[: len(points)] = points

    def _store_segment_data(self):
        """Store subframe segment data to dataset."""
        self.data.coords["segment_point"] = ["start_point", "intermediate_point"]
        self.data["segment_number"] = "coil_name", [
            len(self.loc[self.loc["frame"] == coil_name, :])
            for coil_name in self.data.coil_name
        ]
        self.data.coords["segment_index"] = range(self.data.segment_number.max().data)
        self.data["segment_type"] = xarray.DataArray(
            "",
            coords=[self.data.coil_name, self.data.segment_index],
            dims=["coil_name", "segment_index"],
        ).astype("<U20")
        for attr in ["start_point", "intermediate_point", "end_point", "center"]:
            self.data[attr] = xarray.DataArray(
                0.0,
                coords=[self.data.coil_name, self.data.segment_index, self.data.point],
                dims=["coil_name", "segment_index", "point"],
            )
        points = {
            "start_point": ["x1", "y1", "z1"],
            "end_point": ["x2", "y2", "z2"],
            "center": ["x", "y", "z"],
        }
        for coil_index in range(self.data.dims["coil_name"]):
            index = self.loc["frame"] == self.data.coil_name[coil_index]
            number = sum(index)
            self.data["segment_type"][coil_index, :number] = self.loc[index, "segment"]
            for point, cols in points.items():
                self.data[point][coil_index, :number] = self.loc[index, cols]

    # def _get_point(self, index, columns):

    def update_metadata(self, ids_entry: IdsEntry):
        """Update ids with instance metadata."""
        metadata = Metadata(ids_entry.ids_data)
        comment = self.ids_metadata["comment"]
        provenance = self.ids_metadata["source"]
        metadata.put_properties(comment, homogeneous_time=2, provenance=provenance)
        code_parameters = self.polyline_attrs
        metadata.put_code(code_parameters)

    @cached_property
    def coils_non_axisymmetric_ids(self) -> Ids:
        """Return coils non axisymmetric ids."""
        ids_entry = IdsEntry(**self.ids_attrs)
        self.update_metadata(ids_entry)
        """
        with ids_entry.node("time_slice:global_quantities.*"):
            for attr in ["li_3", "psi_axis", "psi_boundary"]:
                data = self._data[attr].data
                if "psi" in attr:
                    ids_entry[attr, :] = -data  # COCOS
                else:
                    ids_entry[attr, :] = data
        with ids_entry.node("time_slice:global_quantities.magnetic_axis.*"):
            for i, attr in enumerate("rz"):
                ids_entry[attr, :] = self._data.magnetic_axis.data[:, i]
        with ids_entry.node("time_slice:boundary_separatrix.*"):
            for attr in self._data.attrs_0d:
                ids_entry[attr, :] = self._data[attr].data
            ids_entry["type", :] = self._data["boundary_type"].data
            ids_entry["psi", :] = -self._data["psi_boundary"].data  # COCOS
        with ids_entry.node("time_slice:boundary_separatrix.outline.*"):
            for i, attr in enumerate("rz"):
                ids_entry[attr, :] = self._data["boundary"].data[..., i]
        with ids_entry.node("time_slice:boundary_separatrix.geometric_axis.*"):
            for i, attr in enumerate("rz"):
                ids_entry[attr, :] = self._data["geometric_axis"].data[:, i]
        for itime in range(self.data.dims["time"]):
            boundary = ids_entry.ids_data.time_slice[itime].boundary_separatrix
            # boundary x_point
            if self._data.x_point_number[itime].data > 0:
                x_point = self._data.x_point[itime].data
                boundary.x_point.resize(1)
                boundary.x_point[0].r = x_point[0]
                boundary.x_point[0].z = x_point[1]
            # divertor strike points
            if (number := self._data.strike_point_number[itime].data) > 0:
                strike_point = self._data.strike_point[itime].data
                boundary.strike_point.resize(number)
                for point in range(number):
                    boundary.strike_point[point].r = strike_point[point, 0]
                    boundary.strike_point[point].z = strike_point[point, 1]
            # profiles 1D

            # profiles 2D
            profiles_2d = ids_entry.ids_data.time_slice[itime].profiles_2d
            profiles_2d.resize(1)
            profiles_2d[0].type.name = "total"
            profiles_2d[0].type.index = 0
            profiles_2d[0].type.name = "total field and flux"
            profiles_2d[0].grid_type.name = "rectangular"
            profiles_2d[0].grid_type.index = 1
            profiles_2d[0].grid_type.description = "cylindrical grid"
            profiles_2d[0].grid.dim1 = self._data.r.data
            profiles_2d[0].grid.dim2 = self._data.z.data
            profiles_2d[0].r = self._data.r2d.data
            profiles_2d[0].z = self._data.z2d.data
            profiles_2d[0].psi = -self._data.psi2d[itime].data  # COCOS
            # only write field for high order plasma elements
            if self.tplasma == "rectangle":
                profiles_2d[0].b_field_r = self._data.br2d[itime].data
                profiles_2d[0].b_field_z = self._data.bz2d[itime].data
            """

        return ids_entry.ids_data

    def write_ids(self, **ids_attrs):
        """Write pulse design data to equilibrium ids."""
        if ids_attrs["occurrence"] is None:
            ids_attrs["occurrence"] = Database(**ids_attrs).next_occurrence()
        ids_attrs |= {"name": "equilibrium"}
        ids_entry = IdsEntry(ids_data=self.equilibrium_ids, **ids_attrs)
        ids_entry.put_ids()


if __name__ == "__main__":
    centerline = Centerline(
        filename="CC_EXTRATED_CENTERLINES", cross_section={"s": [0, 0, 0.0148]}
    )
    # centerline.build()
