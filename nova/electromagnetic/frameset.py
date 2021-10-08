"""Extend pandas.DataFrame to manage coil and subcoil data."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import os

import pandas

from nova.definitions import root_dir
from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.electromagnetic.select import Select


@dataclass
class Frame:
    """Frame container."""

    frame: FrameSpace = field(repr=False)
    subframe: FrameSpace = field(repr=False)
    delta: float = -1


@dataclass
class Frames(FrameSetLoc, Frame):
    """Manage frame / subframe pair."""

    frame: FrameSpace = field(default=None, repr=False)
    subframe: FrameSpace = field(default=None, repr=False)
    path: str = field(default=None)

    def __post_init__(self):
        """Init path."""
        if self.path is None:
            self.path = os.path.join(root_dir, 'data/Nova/coilsets')

    @contextmanager
    def insert_required(self, required=None):
        """Manage local required arguments."""
        _required = self.frame.metaframe.required.copy()
        if required is None:
            required = self.required
        self.update_required(required)
        yield
        self.update_required(_required)

    def update_required(self, required):
        """Update frame and subframe required arguments."""
        for frame in self.frames:
            frame.update_metaframe(dict(Required=required))

    @property
    def frames(self):
        """Return frame and subframe."""
        return self.frame, self.subframe

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

    def _path(self, path):
        """Return self.path if path is None."""
        if path is None:
            return self.path
        return path

    def file(self, name, path=None, extension='.nc'):
        """Return full netCDF file path."""
        if not os.path.splitext(name)[1]:
            name += extension
        return os.path.join(self._path(path), name)

    def store(self, name: str, path=None):
        """Store frame and subframe as groups within hdf file."""
        file = self.file(name, path)
        self.frame.store(file, 'frame', mode='w')
        self.subframe.store(file, 'subframe', mode='a')

    def load(self, name: str, path=None):
        """Load frameset from file."""
        file = self.file(name, path)
        self.frame.load(file, 'frame')
        self.subframe.load(file, 'subframe')


@dataclass
class FrameSet(Frames):
    """Package FrameSpace instances. Manage boolean methods."""

    base: list[str] = field(repr=False, default_factory=lambda: [
        'x', 'y', 'z', 'dx', 'dy', 'dz'])
    required: list[str] = field(repr=False, default_factory=lambda: [])
    additional: list[str] = field(repr=False, default_factory=lambda: [
        'turn', 'frame'])
    available: list[str] = field(repr=False, default_factory=lambda: [
        'link', 'part', 'frame', 'dx', 'dy', 'dz', 'area', 'volume',
        'delta', 'section', 'turn', 'scale', 'nturn', 'nfilament',
        'Ic', 'It', 'Psi', 'Bx', 'Bz', 'B', 'acloss'])
    subspace: list[str] = field(repr=False, default_factory=lambda: [
        'Ic'])
    array: list[str] = field(repr=False, default_factory=lambda: [
        'Ic'])

    def __post_init__(self):
        """Init coil and subcoil."""
        super().__post_init__()
        self.frame = FrameSpace(
            base=self.base, required=self.required, additional=self.additional,
            available=self.available, subspace=[],
            exclude=['frame', 'Ic', 'It',
                     'active', 'plasma', 'fix', 'feedback'], array=[])
        self.subframe = FrameSpace(
            base=self.base, required=self.required, additional=self.additional,
            available=self.available, subspace=self.subspace,
            exclude=['turn', 'scale', 'nfilament', 'delta'],
            array=self.array, delim='_')
        self.subframe.frame_attr(Select, ['Ic'])


if __name__ == '__main__':

    frameset = FrameSet(required=['rms'], additional=['Ic'])
    frameset.subframe.insert([2, 4], It=6, link=True)
    print(frameset.subframe.rms)

    frameset.store('tmp')
    del frameset
    frameset = FrameSet()
    frameset.load('tmp')

    print('')
    print(frameset.subframe.rms)
    print('')
