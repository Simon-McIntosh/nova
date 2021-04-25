"""Extend pandas.DataFrame to manage coil and subcoil data."""
from dataclasses import dataclass, field

import pandas

from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.frameloc import FrameLoc
from nova.electromagnetic.select import Select


@dataclass
class FramePack(FrameLoc):
    """Package FrameSpace instances. Manage boolean methods."""

    required: list[str] = field(repr=False, default_factory=lambda: [
        'x', 'z', 'dl', 'dt'])
    additional: list[str] = field(repr=False, default_factory=lambda: [
        'turn', 'frame'])
    available: list[str] = field(repr=False, default_factory=lambda: [
        'link', 'part', 'frame', 'dx', 'dz', 'area',
        'delta', 'section', 'turn', 'scale', 'nturn', 'nfilament',
        'Ic', 'It', 'Psi', 'Bx', 'Bz', 'B', 'acloss'])
    subspace: list[str] = field(repr=False, default_factory=lambda: [
        'Ic'])
    array: list[str] = field(repr=False, default_factory=lambda: [
        'Ic'])

    def __post_init__(self):
        """Init coil and subcoil."""
        self.frame = FrameSpace(
            required=self.required, additional=self.additional,
            available=self.available, subspace=[],
            exclude=['frame', 'Ic', 'It',
                     'active', 'plasma', 'fix', 'feedback'], array=[])
        self.subframe = FrameSpace(
            required=self.required, additional=self.additional,
            available=self.available, subspace=self.subspace,
            exclude=['turn', 'scale', 'nfilament', 'delta'],
            array=self.array, delim='_')
        self.subframe.frame_attr(Select, ['Ic'])

    def __str__(self):
        """Return string representation of coilset frame."""
        columns = [col for col in ['link', 'part', 'section', 'turn',
                                   'delta', 'nturn']
                   if col in self.frame]
        superframe = pandas.DataFrame(self.Loc[:, columns])
        superframe['Ic'] = self.sloc['Ic'][self.frame.subref]
        superframe['It'] = superframe['Ic'] * superframe['nturn']
        return superframe.__str__()

    def link(self, index, factor=1):
        """Apply multipoint link to subframe."""
        self.frame.multipoint.link(index, factor)
        self.subframe.multipoint.link(index, factor, expand=True)

    def drop(self, index=None):
        """
        Remove frame and subframe.

        Parameters
        ----------
        index : int or list or pandas.Index, optional
            Index of coils to be removed. The default is None (all coils).

        Returns
        -------
        loc : [int, int]
            Location index of first removed [frame, subframe].

        """
        if index is None:  # drop all coils
            index = self.coil.index
        if not pandas.api.types.is_list_like(index):
            index = [index]
        loc = self.get_loc(index)
        for name in index:
            if name in self.frame.index:
                self.subframe.drop(self.coil.loc[name, 'subindex'])
                self.frame.drop(name)
        return loc

    def translate(self, index=None, dx=0, dz=0):
        """
        Translate coil in polidal plane.

        Parameters
        ----------
        index : int or array-like or Index, optional
            Coil index. The default is None (all coils).
        dx : float, optional
            x-coordinate translation. The default is 0.
        dz : float, optional
            z-coordinate translation. The default is 0.

        Returns
        -------
        None.

        """
        if index is None:
            index = self.coil.index
        elif not pandas.api.types.is_list_like(index):
            index = [index]
        self.coil.translate(index, dx, dz)
        for name in index:
            self.subcoil.translate(self.coil.loc[name, 'subindex'], dx, dz)

    def _get_iloc(self, index):
        iloc = [None, None]
        for name in index:
            if name in self.coil.index:
                iloc[0] = self.coil.index.get_loc(index[0])
                subindex = self.coil.subindex[index[0]][0]
                iloc[1] = self.subcoil.index.get_loc(subindex)
                break
        return iloc


if __name__ == '__main__':

    framepack = FramePack(required=['rms'])
    framepack.subframe.insert([2, 4])
    print(framepack.subframe)
