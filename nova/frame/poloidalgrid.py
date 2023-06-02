"""Manage poloidal grids."""
from dataclasses import dataclass, field

import pandas

from nova.frame.coilsetattrs import GridAttrs
from nova.frame.polygrid import PolyGrid


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
        frame = self.frame.loc[index, :]
        griddata = frame.loc[
            :,
            self.required_columns
            + [attr for attr in self.additional_columns if attr in self.frame],
        ]
        subframe = []
        subattrs = pandas.DataFrame(self.subattrs, index=index)
        try:
            turncurrent = subattrs.pop("It")
        except KeyError:
            turncurrent = None
        for i, name in enumerate(index):
            polygrid = PolyGrid(**griddata.iloc[i].to_dict(), **self.gridattrs)
            data = frame.iloc[i].to_dict()
            data |= {"label": name, "frame": name, "delim": "_", "link": True}
            if turncurrent is not None:
                data["It"] = (
                    turncurrent.iloc[i] * polygrid.frame["nturn"] / polygrid.nturn
                )
            subframe.append(
                self.subframe.assemble(polygrid.frame, **data, **subattrs.iloc[i])
            )
        self.subframe.concatenate(*subframe)
