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
from nova.graphics.plot import Plot
from nova.imas.coil import full_coil_name, part_name
from nova.imas.database import Ids, IdsEntry, IdsIndex
from nova.imas.datasource import CAD, DataSource
from nova.imas.machine import CoilDatabase

datasource = {
    "CC": DataSource(
        pulse=111003,
        run=3,
        description="Ex-Vessel Coils (EVC) Systems (CC) - conductor centerlines",
        provider="Simon McIntosh, simon.mcintosh@iter.org",
        officer="Fabrice Simon, fabrice.simon@iter.org",
        pbs=11,
        status="active",
        replaces="111003/2",
        reason_for_replacement="Upgrade conductor cross-section description inline "
        "with changes to the coils_non_axysymmetric IDS for DD verisions >= 3.34.0",
        cad=CAD(
            reference="DET-07879",
            objects="Correction Coil + Feeder Centerline Extraction "
            "for IMAS database",
            date="05/10/2023",
            provider="Vincent Bontemps, vincent.bontemps@iter.org",
            contact="Guillaume Davin, Guillaume.Davin@iter.org",
        ),
        attributes={"cross_section": {"square": [0, 0, 0.0148]}},
    ),
    "CS1L": DataSource(
        pulse=111006,
        run=1,
        description="Central Solenoid Modules - conductor centerlines",
        provider="Simon McIntosh, simon.mcintosh@iter.org",
        officer="Thierry Schild, thierry.schild@iter.org",
        pbs=11,
        status="active",
        replaces="",
        reason_for_replacement="",
        cad=CAD(
            reference="DET-*",
            objects="Central Solenoid Module CS1L + Feeder Centerline Extraction "
            "for IMAS database",
            date="12/10/2023",
            provider="Vincent Bontemps, vincent.bontemps@iter.org",
            contact="Guillaume Davin, Guillaume.Davin@iter.org",
        ),
        attributes={"cross_section": {"circle": [0, 0, 0.0326, 0.0326, 2]}},
    ),
    "CS": DataSource(
        pulse=111004,
        run=2,
        description="Central Solenoid Modules - conductor centerlines",
        provider="Simon McIntosh, simon.mcintosh@iter.org",
        officer="Thierry Schild, thierry.schild@iter.org",
        pbs=11,
        status="active",
        replaces="111004,1",
        reason_for_replacement="Correction to conductor radius.",
        cad=CAD(
            reference="DET-07879-A",
            objects="Central Solenoid + Feeder Centerline Extraction "
            "for IMAS database",
            date="19/10/2023",
            provider="Vincent Bontemps, vincent.bontemps@iter.org",
            contact="Guillaume Davin, Guillaume.Davin@iter.org",
        ),
        attributes={"cross_section": {"circle": [0, 0, 0.0326, 0.0326, 2]}},
    ),
    "PF": DataSource(
        pulse=111005,
        run=1,
        description="Poloidal Field Coils - conductor centerlines",
        provider="Simon McIntosh, simon.mcintosh@iter.org",
        officer="Fabrice Simon, fabrice.simon@iter.org",
        pbs=11,
        status="active",
        replaces="",
        reason_for_replacement="",
        cad=CAD(
            reference="DET-078779-A",
            objects="Poloidal Field Coil + Feeder Centerline Extraction "
            "for IMAS database",
            date="29/11/2023",
            provider="Vincent Bontemps, vincent.bontemps@iter.org",
            contact="Nelson Gatos, nelson.gatos@iter.org",
        ),
        attributes={"cross_section": {"circle": [0, 0, 0.035, 0.035, 2]}},
    ),
    "VS3": DataSource(
        pulse=115003,
        run=2,
        description="Vertical Stability In-Vessel Coils - conductor centerlines",
        provider="Simon McIntosh, simon.mcintosh@iter.org",
        officer="Nicola Mariani, nicola.mariani@iter.org",
        pbs=15,
        status="active",
        replaces="115003/1",
        reason_for_replacement="Geometry update and inclusion of arc elements.",
        cad=CAD(
            reference="IVC_EXTRACTION_bump.xls",
            objects="Vertical Stablilty loop #3 Coil + Feeder Centerline Extraction "
            "for IMAS database",
            date="05/02/2019",
            provider="Vincent Bontemps, vincent.bontemps@iter.org",
            contact="Vincent Bontemps, vincent.bontemps@iter.org",
        ),
        attributes={"cross_section": {"annulus": [0, 0, 0.046, 1 - 0.0339 / 0.046, 2]}},
    ),
    "ELM": DataSource(
        pulse=115001,
        run=2,
        description="ELM Coils - conductor centerlines",
        provider="Simon McIntosh, simon.mcintosh@iter.org",
        officer="Julien Laquiere, julien.laquiere@iter.org",
        pbs=15,
        status="active",
        replaces="115001/1",
        reason_for_replacement="Update corrects conductor radius and "
        "includes arc elements extracted from source IDS multiline.",
        cad=CAD(
            reference="115001/1",
            objects="ELM Coil Centerline Extraction " "for IMAS database",
            date="23/01/2023",
            provider="Masanari Hosokawa, masanari.hosokawa@iter.org",
            contact="Simon McIntosh, simon.mcintosh@iter.org",
        ),
        attributes={"cross_section": {"annulus": [0, 0, 0.046, 1 - 0.0339 / 0.046, 2]}},
    ),
    "TF": DataSource(
        pulse=111002,
        run=2,
        description="TF Coils - conductor centerlines",
        provider="Simon McIntosh, simon.mcintosh@iter.org",
        officer="Sebastien Koczorowski, sebastien.koczorowski@iter.org",
        pbs=11,
        status="active",
        replaces="111002/1",
        reason_for_replacement="Update includes full centerline winding with feeders.",
        cad=CAD(
            reference="DET-07879-C",
            objects="Toroidal Field Coil + Feeder Centerline Extraction "
            "for IMAS database",
            date="09/02/2024",
            provider="Vincent Bontemps, vincent.bontemps@iter.org",
            contact="Nelson Gatos, nelson.gatos@iter.org",
        ),
        attributes={"cross_section": {"circle": [0, 0, 0.0397, 0.0397, 2]}},
    ),
}


@dataclass
class PolylineAttrs:
    """Group polyline attributes."""

    minimum_arc_nodes: int = 4
    quadrant_segments: int = 7
    arc_resolution: float = 1.5
    arc_eps: float = 1e-3
    line_eps: float = 1e-1
    rdp_eps: float = 1e-4

    @property
    def polyline_attrs(self):
        """Return polyline attributes."""
        return {
            "minimum_arc_nodes": self.minimum_arc_nodes,
            "quadrant_segments": self.quadrant_segments,
            "arc_resolution": self.arc_resolution,
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
    name: str = "coils_non_axisymmetric"
    datadir: str = "/mnt/share/coil_centerlines"

    description: ClassVar[str] = (
        "An algoritum for the aproximation of CAD generated "
        "multi-point conductor centerlines by a sequence of "
        "straight-line and arc segments. "
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
        self.datasource.name = self.name
        self.datasource.cad.filename = f"{self.filename}.xls"
        self.datasource.code(
            description=self.description, parameter_dict=self.polyline_attrs
        )
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
        return Polygon(self.datasource.attributes["cross_section"])

    @property
    def yaml_attrs(self) -> dict:
        """Return datasource yaml attributes."""
        return self.datasource.yaml_attrs

    @property
    def xls_file(self):
        """Return xls filename."""
        return os.path.join(self.datadir, f"{self.filename}.xlsx")

    @property
    def netcdf_attrs(self):
        """Return netcdf attrs."""
        return self.ids_attrs | self.polyline_attrs | self.yaml_attrs

    @property
    def group_attrs(self) -> dict:
        """
        Return group attrs.

        Replaces :func:`~nova.imas.database.CoilDatabase`.
        """
        return self.netcdf_attrs | self.datasource.attributes

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
                    delim="-",
                    **self.polyline_attrs,
                )
        self._store_point_data(xls_points)
        self._store_segment_data()
        self.data.attrs = self.netcdf_attrs
        self.store()

    def _read_sheet(self, xls, sheet_name=0):
        """Read excel worksheet."""
        sheet = pandas.read_excel(xls, sheet_name, usecols=[2, 3, 4])
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
        for coil_index in range(self.data.sizes["coil_name"]):
            index = self.loc["frame"] == self.data.coil_name[coil_index]
            number = sum(index)
            self.data["segment_type"][coil_index, :number] = self.loc[index, "segment"]
            for point, cols in self.subframe_attrs.items():
                self.data[point][coil_index, :number] = self.loc[index, cols]
            self.data["intermediate_points"][coil_index, :number] = intermediate_point[
                index
            ]

    def _set_ids_points(self, ids_node, name, points, attrs=["r", "phi", "z"]):
        """Fill element points."""
        ids_points = getattr(ids_node, name)
        setattr(ids_points, attrs[0], np.linalg.norm(points[:, :2], axis=1))
        setattr(ids_points, attrs[1], np.arctan2(points[:, 1], points[:, 0]))
        setattr(ids_points, attrs[2], points[:, 2])

    @cached_property
    def coils_non_axisymmetric_ids(self) -> Ids:
        """Return populated coils non axisymmetric ids."""
        ids_entry = IdsEntry(**self.ids_attrs, ids_node="coil")
        ids_entry.ids.resize(self.data.sizes["coil_name"])
        coil_name = [str(name) for name in self.data.coil_name.data]
        ids_entry["identifier", :] = coil_name
        ids_entry["name", :] = [full_coil_name(identifier) for identifier in coil_name]
        ids_entry["turns", :] = np.ones(self.data.sizes["coil_name"], float)
        element_type = {"line": 1, "arc": 2}
        for name, coil in zip(coil_name, ids_entry.ids):
            coil.conductor.resize(1)
            conductor = coil.conductor[0]
            elements = IdsIndex(conductor, "elements")
            coil_data = self.data.sel(coil_name=name)
            segment_number = coil_data.segment_number.data
            elements["names"] = [f"{name}_{i}" for i in range(segment_number)]
            elements["types"] = np.array(
                [
                    element_type[segment_type]
                    for segment_type in coil_data.segment_type[:segment_number].data
                ]
            )
            for attr in [
                "start_points",
                "intermediate_points",
                "end_points",
                "centres",
            ]:
                points = coil_data[attr][:segment_number].data
                self._set_ids_points(elements.ids, attr, points)

            conductor.cross_section.resize(1)
            cross_section = conductor.cross_section[0]
            match self.cross_section.name:
                case "disc":
                    cross_section.geometry_type.name = "circular"
                    cross_section.geometry_type.index = 2
                    cross_section.geometry_type.description = "Circle"
                    cross_section.width = self.cross_section.width
                case "square":
                    cross_section.geometry_type.name = "square"
                    cross_section.geometry_type.index = 4
                    cross_section.geometry_type.description = "Square"
                    cross_section.width = self.cross_section.width
                case "skin" | "annulus":
                    cross_section.geometry_type.name = "annulus"
                    cross_section.geometry_type.index = 5
                    cross_section.geometry_type.description = "Annulus"
                    cross_section.width = self.cross_section.width
                    radius = self.cross_section.width / 2
                    cross_section.radius_inner = radius * (
                        1 - self.cross_section.metadata["thickness"]
                    )
                case _:
                    raise NotImplementedError(
                        f"cross seciton {self.cross_section.name} not implemented"
                    )

        # update code and ids_properties nodes
        self.datasource.update(ids_entry.ids_data)
        return ids_entry.ids_data

    def write_ids(self, **ids_attrs):
        """Write pulse design data to equilibrium ids."""
        ids_attrs = self.ids_attrs | ids_attrs
        IdsEntry(ids_data=self.coils_non_axisymmetric_ids, **ids_attrs).put_ids()
        self.datasource.write_yaml()


if __name__ == "__main__":
    # filename = "CS"
    # filename = "PF"
    # filename = "CC"
    # filename = "VS3"
    # filename = "ELM"
    # filename = "TF"
    # centerline = Centerline(filename=filename)
    # centerline.plot()
    centerline = Centerline(filename="ELM")
    centerline.plot()
    # centerline.write_ids()
    """
    for filename in ["CS", "PF", "VS3", "ELM", "CC", "TF"]:
        print(filename)
        centerline = Centerline(filename=filename)
        centerline.plot()
        centerline.write_ids()

    """

    # centerline = Centerline(filename=filename)
    # centerline.plot()
    # centerline.write_ids()

    """

    from nova.frame.coilset import CoilSet

    coilset = CoilSet()
    for filename in ["CC", "CS", "PF"]:
        coilset += Centerline(filename=filename)

    coilset.plot()
    """
