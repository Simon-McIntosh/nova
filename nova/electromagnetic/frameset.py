"""Extend pandas.DataFrame to manage coil and subcoil data."""
from dataclasses import dataclass, field

import netCDF4
import pandas

from nova.database.filepath import FilePath
from nova.database.netcdf import netCDF
from nova.electromagnetic.framedata import FrameData
from nova.electromagnetic.framesetloc import FrameSetLoc
from nova.electromagnetic.framespace import FrameSpace
from nova.electromagnetic.select import Select


@dataclass
class FrameSet(FilePath, FrameSetLoc):
    """Manage FrameSet instances."""

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
        'Ic', 'nturn', 'active', 'passive', 'plasma', 'coil',
        'fix', 'free', 'ferritic'])

    def __post_init__(self):
        """Create frame and subframe."""
        self.frame = FrameSpace(
            base=self.base, required=self.required, additional=self.additional,
            available=self.available, subspace=[],
            exclude=['frame', 'Ic', 'It',
                     'active', 'passive', 'plasma', 'coil',
                     'fix', 'free', 'ferritic'], array=[],
            version=['index'])
        self.subframe = FrameSpace(
            base=self.base, required=self.required, additional=self.additional,
            available=self.available, subspace=self.subspace,
            exclude=['turn', 'scale', 'nfilament', 'delta'],
            array=self.array, delim='_',
            version=['index', 'nturn'])
        self.subframe.frame_attr(Select, ['Ic'])
        super().__post_init__()

    def __str__(self):
        """Return string representation of coilset frame."""
        columns = [col for col in ['link', 'part', 'section', 'turn',
                                   'delta', 'nturn']
                   if col in self.frame]
        superframe = pandas.DataFrame(self.Loc[:, columns])
        superframe['Ic'] = self.sloc['Ic'][self.frame.subref]
        superframe['It'] = superframe['Ic'] * superframe['nturn']
        return superframe.__str__()

    def clear_frameset(self):
        """Clear all frameset instances."""
        delattrs = []
        for attr in self.__dict__:
            if isinstance(getattr(self, attr), FrameData):
                delattrs.append(attr)
        for attr in delattrs:
            delattr(self, attr)

    def subset(self, dataset):
        """Return dataset subgroup."""
        if self.group is not None:
            return dataset[self.group]
        return dataset

    def load_metadata(self, filename=None, path=None):
        """Return metadata from netCDF file."""
        file = self.file(filename, path)
        metadata = {}
        with netCDF4.Dataset(file) as dataset:
            dataset = self.subset(dataset)
            if not hasattr(dataset, 'metadata'):
                return {}
            for attr in dataset.metadata:
                metadata[attr] = getattr(dataset, attr)
        return metadata

    def store_metadata(self, filename=None, path=None, metadata=None):
        """Store metadata to netCDF file."""
        if metadata is None:
            return
        file = self.file(filename, path)
        with netCDF4.Dataset(file, 'a') as dataset:
            dataset = self.subset(dataset)
            dataset.metadata = list(metadata)
            for attr in metadata:
                setattr(dataset, attr, metadata[attr])

    @property
    def subgroup(self):
        """Return netcdf group."""
        if self.group is None:
            return ''
        return self.group

    def load(self, filename=None, path=None):
        """Load frameset from file."""
        file = self.file(filename, path)
        self.frame.load(file, self.netcdf_path('frame'))
        self.subframe.load(file, self.netcdf_path('subframe'))
        self.clear_frameset()
        with netCDF4.Dataset(file) as dataset:
            dataset = self.subset(dataset)
            for attr in dataset.groups:
                if attr in dir(self.__class__) and isinstance(
                        data := getattr(self, attr), netCDF):
                    data.group = self.group
                    data.load(file)
        return self

    def store(self, filename=None, path=None):
        """Store frame, subframe and biot attrs as groups within hdf file."""
        file = self.file(filename, path)
        self.frame.store(file, self.netcdf_path('frame'), mode=self.mode(file))
        self.subframe.store(file, self.netcdf_path('subframe'), mode='a')
        for attr in self.__dict__:
            if isinstance(data := getattr(self, attr), netCDF):
                data.group = self.group
                data.store(file)
        return self

    def plot(self, index=None, axes=None, **kwargs):
        """Plot coilset subframe via polyplot instance."""
        self.subframe.polyplot(index=index, axes=axes, **kwargs)


if __name__ == '__main__':

    frameset = FrameSet(required=['rms'], additional=['Ic'])
    frameset.subframe.insert([2, 4], It=6, link=True)
