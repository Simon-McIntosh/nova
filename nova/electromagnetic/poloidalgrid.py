"""Manage poloidal grids."""
from dataclasses import dataclass, field

import pandas

from nova.electromagnetic.frameattrs import FrameAttrs
from nova.electromagnetic.polygrid import PolyGrid


@dataclass
class PoloidalGrid(FrameAttrs):
    """Generate subframe poloidal grids from frame input."""

    trim: bool = True
    fill: bool = False
    grid: dict = field(init=False, default_factory=lambda: dict.fromkeys([
        'tile', 'trim', 'fill']))
    required: list = field(init=False, default_factory=lambda: [
        'poly', 'delta', 'turn', 'nturn'])
    additional: list = field(init=False, default_factory=lambda: [
        'scale', 'skin'])

    def insert(self, *required, iloc=None, **additional):
        """
        Insert frame(s).

        Parameters
        ----------
        *required : Union[DataFrame, dict, list]
            Required input.
        iloc : int, optional
            Index before which coils are inserted. The default is None (-1).
        **additional : dict[str, Any]
            Additional input.

        Returns
        -------
        index : pandas.Index
            Frame index.

        """
        self.attrs = additional
        index = self.frame.insert(*required, iloc=iloc, **self.attrs)
        self.subframe_insert(index)
        if self.link:
            self.subframe.multipoint.link(index, expand=True)
        return index

    def subframe_insert(self, index):
        """
        Insert subframe(s).

        - Store filaments in subframe.
        - Link turns.

        """
        frame = self.frame.loc[index, :]
        griddata = frame.loc[:, self.required +
                             [attr for attr in self.additional
                              if attr in self.frame]]
        subframe = []
        subattrs = pandas.DataFrame(self.subattrs, index=index)
        try:
            turncurrent = subattrs.pop('It')
        except KeyError:
            turncurrent = None
        for i, name in enumerate(index):
            polygrid = PolyGrid(**griddata.iloc[i].to_dict(), **self.grid)
            data = frame.iloc[i].to_dict()
            data |= {'label': name, 'frame': name, 'delim': '_', 'link': True}
            if turncurrent is not None:
                data['It'] = turncurrent.iloc[i] * \
                    polygrid.frame['nturn'] / polygrid.nturn
            subframe.append(self.subframe.assemble(
                polygrid.frame, **data, **subattrs.iloc[i]))
        self.subframe.concatenate(*subframe)
