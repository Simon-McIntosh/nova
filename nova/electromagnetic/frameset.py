"""Extend pandas.DataFrame to manage coil and subcoil data."""
from dataclasses import dataclass, field

import pandas

from nova.electromagnetic.filepath import FilePath
from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.select import Select


@dataclass
class FrameSet(FrameSetLoc, FilePath):
    """Manage FrameSet instances."""

    delta: float = -1
    base: list[str] = field(repr=False, default_factory=lambda: [
        'x', 'y', 'z', 'dx', 'dy', 'dz'])
    required: list[str] = field(repr=False, default_factory=lambda: [])
    additional: list[str] = field(repr=False, default_factory=lambda: [
        'turn', 'frame', 'plasma', 'Ic', 'nturn'])
    available: list[str] = field(repr=False, default_factory=lambda: [
        'link', 'part', 'frame', 'dx', 'dy', 'dz', 'area', 'volume', 'vtk',
        'delta', 'section', 'turn', 'scale', 'nturn', 'nfilament',
        'Ic', 'It', 'Psi', 'Bx', 'Bz', 'B', 'acloss'])
    subspace: list[str] = field(repr=False, default_factory=lambda: [
        'Ic'])
    array: list[str] = field(repr=False, default_factory=lambda: [
        'Ic', 'nturn'])

    def __post_init__(self):
        """Init coil and subcoil."""
        super().__post_init__()
        self.frame = FrameSpace(
            base=self.base, required=self.required, additional=self.additional,
            available=self.available, subspace=[],
            exclude=['frame', 'Ic', 'It',
                     'active', 'plasma', 'fix', 'feedback'], array=[],
            version=['index'])
        self.subframe = FrameSpace(
            base=self.base, required=self.required, additional=self.additional,
            available=self.available, subspace=self.subspace,
            exclude=['turn', 'scale', 'nfilament', 'delta'],
            array=self.array, delim='_',
            version=['plasma', 'index'])
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

    def store(self, filename: str, path=None):
        """Store frame and subframe as groups within hdf file."""
        file = self.file(filename, path)
        self.frame.store(file, 'frame', mode='w')
        self.subframe.store(file, 'subframe', mode='a')

    def load(self, filename: str, path=None):
        """Load frameset from file."""
        file = self.file(filename, path)
        self.frame.load(file, 'frame')
        self.subframe.load(file, 'subframe')
        return self


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
