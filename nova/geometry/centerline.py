"""Manage CAD centerlines."""
from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar
import os

import numpy as np
import pandas
from tqdm import tqdm
import xarray


from nova.biot.biotframe import Source
from nova.geometry.polygeom import Polygon
from nova.geometry.section import Section
from nova.graphics.plot import Plot
from nova.imas.coil import part_name
from nova.imas.database import Ids, IdsEntry
from nova.imas.datasource import CAD, DataSource
from nova.imas.machine import CoilDatabase
from nova.imas.metadata import Code

datasource = {
    "CC_EXTRATED_CENTERLINES": DataSource(
        pulse=111003,
        run=2,
        name="coils_non_axisymmetric",
        pbs=11,
        provider="Simon McIntosh, simon.mcintosh@iter.org",
        officer="Fabrice Simon, fabrice.simon@iter.org",
        status="active",
        replaces="111003/1",
        reason_for_replacement="resolve conductor centerlines and include coil feeders",
        cad=CAD(
            cross_section={"square": [0, 0, 0.0148]},
            reference="DET-07879",
            objects="Correction Coils + Feeders Centerlines Extraction "
            "for IMAS database",
            filename="CC_EXTRATED_CENTERLINES.xls",
            date="05/10/2023",
            provider="Vincent Bontemps, vincent.bontemps@iter.org",
            contact="Guillaume Davin, Guillaume.Davin@iter.org",
        ),
    ),
    "CS1L": DataSource(
        pulse=0,
        run=0,
        name="coils_non_axisymmetric",
        # system="central_solenoid",
        # comment="* - conductor centerlines",
        pbs=11,
        provider="Simon McIntosh, simon.mcintosh@iter.org",
        officer="Fabrice Simon, fabrice.simon@iter.org",
        status="active",
        replaces="",
        reason_for_replacement="",
        cad=CAD(
            cross_section={"square": [0, 0, 0.0148]},
            reference="DET-*",
            objects="Correction Coils + Feeders Centerlines Extraction "
            "for IMAS database",
            filename="CS1L.xls",
            date="12/10/2023",
            provider="Vincent Bontemps, vincent.bontemps@iter.org",
            contact="Guillaume Davin, Guillaume.Davin@iter.org",
        ),
    ),
}

"""
    "system": "correction_coils",
    "comment": "Ex-Vessel Coils (EVC) Systems (CC) - conductor centerlines"
"""


@dataclass
class PolylineAttrs:
    """Group polyline attributes."""

    minimum_arc_nodes: int = 4
    quadrant_segments: int = 32
    arc_eps: float = 1e-3
    line_eps: float = 2e-3
    rdp_eps: float = 1e-4

    @property
    def polyline_attrs(self):
        """Return polyline attributes."""
        return {
            "minimum_arc_nodes": self.minimum_arc_nodes,
            "quadrant_segments": self.quadrant_segments,
            "arc_eps": self.arc_eps,
            "line_eps": self.line_eps,
            "rdp_eps": self.rdp_eps,
        }


@dataclass
class Centerline(Plot, PolylineAttrs, CoilDatabase):
    r"""Extract coil centerlines from CAD traces.

    Centerline source data is recived via email and and stored in a shared folder at:
    \\\\io-ws-ccstore1\\ANSYS_Data\\mcintos\\coil_centerlines

    datadir : str
        Data directory. Set as mount point location to access IO shared folder
    """

    filename: str = ""
    datadir: str = "/mnt/share/coil_centerlines"

    description: ClassVar[str] = (
        "An algoritum for the aproximation of CAD generated "
        "multi-point conductor centerlines by a sequence of "
        "straight-line and arc segments. "
        "See Also: nova.geometry.centerline.Centerline"
    )
    subframe_attrs: ClassVar[dict[str, list[str]]] = {
        "start_points": ["x1", "y1", "z1"],
        "end_points": ["x2", "y2", "z2"],
        "centres": ["x", "y", "z"],
        "axis": ["ax", "ay", "az"],
        "normal": ["nx", "ny", "nz"],
    }

    def __post_init__(self):
        """Format cross_section."""
        self.datasource = datasource[self.filename]
        self.ids_attrs = self.datasource.ids_attrs
        super().__post_init__()

    @cached_property
    def datasource(self) -> DataSource:
        """Return datasource instance."""
        try:
            return datasource[self.filename]
        except KeyError as error:
            raise KeyError(
                f"filename {self.filename} not defined in "
                f"datasource {datasource.keys()}"
            ) from error

    @cached_property
    def cross_section(self):
        """Return conductor cross-section from datasource."""
        return Polygon(self.datasource.cad.cross_section)

    @property
    def code(self):
        """Return code instance."""
        return Code(description=self.description, parameter_dict=self.polyline_attrs)

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
    def netcdf_attrs(self):
        """Return netcdf attrs."""
        return self.ids_attrs | self.polyline_attrs  # | self.ids_metadata | self.yaml

    def build(self):
        """Load points from file and build coil centerlines."""
        self.data = xarray.Dataset()
        self.data.coords["point"] = list("xyz")
        self.data.coords["cross_section_index"] = range(len(self.cross_section.points))
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

        self.data.attrs = self.netcdf_attrs
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
        for attr in list(self.subframe_attrs) + ["intermediate_points"]:
            self.data[attr] = xarray.DataArray(
                0.0,
                coords=[self.data.coil_name, self.data.segment_index, self.data.point],
                dims=["coil_name", "segment_index", "point"],
            )
        intermediate_point = Source(self.subframe).space.intermediate_point
        for coil_index in range(self.data.dims["coil_name"]):
            index = self.loc["frame"] == self.data.coil_name[coil_index]
            number = sum(index)
            self.data["segment_type"][coil_index, :number] = self.loc[index, "segment"]
            for point, cols in self.subframe_attrs.items():
                self.data[point][coil_index, :number] = self.loc[index, cols]
            self.data["intermediate_points"][coil_index, :number] = intermediate_point[
                index
            ]

    def _set_ids_points(self, ids_node, name, points):
        """Fill element points."""
        ids_points = getattr(ids_node, name)
        ids_points.r = np.linalg.norm(points[:, :2], axis=1)
        ids_points.phi = np.arctan2(points[:, 1], points[:, 0])
        ids_points.height = points[:, 2]

    @cached_property
    def coils_non_axisymmetric_ids(self) -> Ids:
        """Return populated coils non axisymmetric ids."""
        ids_entry = IdsEntry(ids_node="coil", **self.ids_attrs)
        self.update_metadata(ids_entry)
        ids_entry.ids.resize(self.data.dims["coil_name"])
        section = Section(self.data.cross_section.data)

        for coil_index, ids_coil in enumerate(ids_entry.ids):
            ids_coil.name = self.data.coil_name[coil_index].data
            ids_coil.conductor.resize(1)
            ids_coil.conductor[0].turns = 1
            ids_elements = ids_coil.conductor[0].elements
            segment_number = self.data.segment_number[coil_index].data
            ids_elements.types = self.data.segment_type[
                coil_index, :segment_number
            ].data
            for point_name in [
                "start_points",
                "intermediate_points",
                "end_points",
                "centres",
            ]:
                points = getattr(self.data, point_name)[
                    coil_index, :segment_number
                ].data
                self._set_ids_points(ids_elements, point_name, points)
            ids_cross_section = ids_coil.conductor[0].cross_section
            normal = self.data.normal[coil_index, 0]
            axis = self.data.axis[coil_index, 0]
            start_axes = np.c_[normal, np.cross(axis, normal), axis].T
            section.to_axes(start_axes)

            points = (
                section.points
                + self.data.start_points[coil_index, 0].data[np.newaxis, :]
            )
            self.set_axes("3d")
            self.axes.plot(*points.T)
            print(coil_index, section, ids_cross_section)

        # print(ids_entry.ids_data.coil[0].conductor[0].elements.start_points.r)

        #    ids_entry["name", :] = self.data.coil_name.data

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
        ids_attrs = self.ids_attrs | ids_attrs
        # ids_entry = IdsEntry(ids_data=self.coils_non_axisymmetric_ids, **ids_attrs)

        # ids_entry.put_ids()
        # self.write_yaml(**ids_attrs)


if __name__ == "__main__":
    filename = "CC_EXTRATED_CENTERLINES"

    # filename, cross_section = "CS1L"
    centerline = Centerline(filename=filename)
    # centerline.coils_non_axisymmetric_ids
    from nova.geometry.polyline import PolyLine

    polyline = PolyLine(centerline.data.points[0].values)
    polyline.plot()
