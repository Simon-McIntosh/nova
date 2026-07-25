"""Manage poloidal grids."""

from dataclasses import dataclass, field

import numpy as np

from nova.frame.columnar import is_list_like
from nova.frame.coilsetattrs import GridAttrs
from nova.frame.polygrid import PolyGrid
from nova.geometry.polygeom import PolyGeom


@dataclass
class PoloidalGrid(GridAttrs):
    """Generate subframe poloidal grids from frame input."""

    trim: bool = True
    fill: bool = False
    gridattrs: dict = field(
        init=False, default_factory=lambda: dict.fromkeys(["tile", "trim", "fill"])
    )
    required_columns: list = field(
        init=False, default_factory=lambda: ["poly", "delta", "turn", "nturn"]
    )
    additional_columns: list = field(
        init=False, default_factory=lambda: ["scale", "skin"]
    )

    def insert(self, *args, required=None, iloc=None, **additional):
        """
        Insert frame(s).

        Parameters
        ----------
        *args : Union[DataFrame, dict, list]
            Required input.
        required : list[str]
            Required attribute names (args). The default is None.
        iloc : int, optional
            Index before which coils are inserted. The default is None (-1).
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        index : pandas.Index
            FrameSpace index.

        """
        if isinstance(args[0], dict):
            additional |= PolyGeom(args[0]).geometry
        self.attrs = additional
        with self.insert_required(required):
            index = self.frame.insert(*args, iloc=iloc, **self.attrs)
            self.subframe_insert(index)
        if self.link:
            self.linkframe(index)
        self.linksubframe(index)
        self.update_loc_indexer()
        return index

    @staticmethod
    def thick_filament_segment(data, polygrid) -> dict:
        """Return a segment override coupling polygonal plasma cells exactly.

        A point-filament ring is log-singular at its own location, so a plasma
        cell evaluated at or next to itself picks up a spurious near-field
        spike. Where the mesh is polygonal throughout — hexagonal cells plus the
        polygons the first wall clips them into — the cells are handed to the
        exact polygon-section element instead
        (:class:`nova.biot.polysection.PolySection`).

        Only plasma meshes are promoted. Other conductors meshed into polygonal
        turns keep the coupling they were built with: promoting them is
        defensible on the same grounds but moves their numbers and their cost,
        which is a separate decision from the plasma mesh.
        """
        if not data.get("plasma", False):
            return {}
        section = set(np.asarray(polygrid.frame["section"]).tolist())
        if not section or not section <= {"hexagon", "polygon"}:
            return {}
        return {"segment": "polysection"}

    def subframe_insert(self, index):
        """
        Insert subframe(s).

        - Store filaments in subframe.
        - Link turns.

        """
        length = len(index)
        subattrs = dict(self.subattrs)
        turncurrent = subattrs.pop("It", None)
        griddata_columns = self.required_columns + [
            attr for attr in self.additional_columns if attr in self.frame
        ]
        griddata = self.row_records(self.frame, index, griddata_columns)
        framedata = self.row_records(self.frame, index)
        subframe = []
        for i, name in enumerate(index):
            polygrid = PolyGrid(**griddata[i], **self.gridattrs)
            data = framedata[i] | {
                "label": name,
                "frame": name,
                "delim": "_",
                "link": True,
            }
            data |= self.thick_filament_segment(data, polygrid)
            if turncurrent is not None:
                current = (
                    turncurrent[i]
                    if is_list_like(turncurrent) and len(turncurrent) == length
                    else turncurrent
                )
                data["It"] = current * polygrid.frame["nturn"] / polygrid.nturn
            subframe.append(
                self.subframe.assemble(
                    polygrid.frame, **data, **self.broadcast_row(subattrs, i, length)
                )
            )
        self.subframe.concatenate(*subframe)
