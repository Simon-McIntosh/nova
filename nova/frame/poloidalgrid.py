"""Manage poloidal grids."""

from dataclasses import dataclass, field


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
