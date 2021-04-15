"""Manage poloidal grids."""
from dataclasses import dataclass, field
from typing import Union

from nova.electromagnetic.frame import Frame
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
        None.

        """
        self.attrs = additional
        index = self.frame.insert(*required, iloc=iloc, **self.attrs)
        self.subframe_insert(index)

    def subframe_insert(self, index):
        """
        Insert subframe(s).

        - Store filaments in subframe.
        - Link turns.

        """
        columns = self.required + [attr for attr in self.additional
                                   if attr in self.frame]
        frame = self.frame.loc[index, columns]
        part = self.frame.loc[index, 'part']
        subframe = []
        for i, name in enumerate(index):
            data = PolyGrid(**frame.iloc[i].to_dict(), **self.grid).frame
            subframe.append(self.subframe.assemble(
                data, label=name, delim='_', link=True,
                part=part[i]))
        self.subframe.concatenate(*subframe)
