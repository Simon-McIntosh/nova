"""Biot-Savart calculation base class."""
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import ClassVar

import numpy as np
import pandas
from tqdm import tqdm
import xarray

from nova.biot.biotcylinder import BiotCylinder
from nova.biot.biotpolygon import BiotPolygon
from nova.biot.biotring import BiotRing
from nova.biot.biotset import BiotSet


@dataclass
class BiotSolve(BiotSet):
    """Manage biot interaction between multiple filament types."""

    name: str = 'biot'
    attrs: list[str] = field(default_factory=lambda: [
        'Aphi', 'Psi', 'Br', 'Bz'])

    svd: bool = True
    source_segment: np.ndarray = field(init=False, repr=False)
    data: xarray.Dataset = field(init=False, default_factory=xarray.Dataset)

    generator: ClassVar[dict] = {'ring': BiotRing, 'cylinder': BiotCylinder,
                                 'polygon': BiotPolygon}

    def __post_init__(self):
        """Initialise dataset and compute biot interaction."""
        super().__post_init__()
        self.check_segments()
        self.initialize()
        self.compose()
        self.decompose()

    def check_segments(self):
        """Check for segment in self.generator."""
        self.source_segment = self.source.segment.copy()
        for segment in self.source_segment.unique():
            if segment not in self.generator:
                raise NotImplementedError(
                    f'segment <{segment}> not implemented '
                    f'in Biot.generator: {self.generator.keys()}')
            index = self.source.index[self.source_segment == segment]
            #for i, chunk in enumerate(
            #        self.group_segments(index, 150, index[-1])):
            #    print(self.source_segment)
            #    self.source_segment.loc[chunk, 'segment'] = f'{segment}_{i}'

    @staticmethod
    def group_segments(iterable, length, fillvalue):
        """Return grouped itterable."""
        length = min([length, len(iterable)])
        args = length * [iter(iterable)]
        return zip_longest(*args, fillvalue=fillvalue)

    def initialize(self):
        """Initialize dataset."""
        self.data = xarray.Dataset(
            coords=dict(source=self.get_index('source'),
                        plasma=self.source.index[self.source.plasma].to_list(),
                        target=self.get_index('target')))
        self.data.attrs['attributes'] = self.attrs

        for attr in self.attrs:
            self.data[attr] = xarray.DataArray(
                0., dims=['target', 'source'],
                coords=[self.data.target, self.data.source])
        self._initialize_svd('target', 'source')

        for attr in self.attrs:  # unit filaments
            self.data[f'_{attr}'] = xarray.DataArray(
                0., dims=['target', 'plasma'],
                coords=[self.data.target, self.data.plasma])

        self._initialize_svd('target', 'plasma', prefix='_')
        '''
        if self.data.dims['plasma'] < self.data.dims['target']:
            sigma = 'plasma'
        else:
            sigma = 'target'
        for attr in self.attrs:  # unit filament svd matricies
            self.data[f'_U{attr}'] = xarray.DataArray(
                0., dims=['target', sigma],
                coords=[self.data.target, self.data[sigma]])
            self.data[f'_s{attr}'] = xarray.DataArray(
                0., dims=[sigma], coords=[self.data[sigma]])
            self.data[f'_V{attr}'] = xarray.DataArray(
                0., dims=[sigma, 'plasma'],
                coords=[self.data[sigma], self.data.plasma])
        '''

    def _initialize_svd(self, row: str, column: str, prefix=''):
        """Initialize svd data structures."""
        if self.data.dims[column] < self.data.dims[row]:
            sigma = column
        else:
            sigma = row
        for attr in self.attrs:  # unit filament svd matricies
            self.data[f'{prefix}U{attr}'] = xarray.DataArray(
                0., dims=[row, sigma],
                coords=[self.data[row], self.data[sigma]])
            self.data[f'{prefix}s{attr}'] = xarray.DataArray(
                0., dims=[sigma], coords=[self.data[sigma]])
            self.data[f'{prefix}V{attr}'] = xarray.DataArray(
                0., dims=[sigma, column],
                coords=[self.data[sigma], self.data[column]])

    def get_index(self, frame: str) -> list[str]:
        """Return matrix coordinate, reduce if flag True."""
        biotframe = getattr(self, frame)
        if biotframe.reduce:
            return biotframe.biotreduce.index.to_list()
        return biotframe.index.to_list()

    def compose(self):
        """Calculate full ensemble biot interaction."""
        for segment in tqdm(self.source_segment.unique(), ncols=65,
                            desc=self.name):
            self.compute(segment)

    def source_index(self, segment):
        """Return source segment index."""
        frame = self.source.frame[self.source_segment == segment]
        return np.isin(self.get_index('source'), frame)

    def plasma_index(self, segment):
        """Return plasma segment index."""
        plasma = self.source_segment[self.source.index[self.source.plasma]]
        return np.array(plasma == segment)

    def compute(self, segment: str):
        """Compute segment and update dataset."""
        source_index = self.source_index(segment)
        plasma_index = self.plasma_index(segment)
        generator = self.generator[segment.split('_')[0]](
            pandas.DataFrame(
                self.source.loc[self.source_segment == segment, :]),
            self.target, turns=self.turns, reduce=self.reduce,
            chunks=self.chunks)
        for attr in self.attrs:
            matrix, plasma = generator.compute(attr)
            self.data[attr].loc[:, source_index] += matrix
            self.data[f'_{attr}'].loc[:, plasma_index] += plasma

    def decompose(self):
        """Compute plasma svd and update dataset."""
        if not self.svd:
            return
        if self.data.dims['plasma'] < self.data.dims['target']:
            sigma = 'plasma'
        else:
            sigma = 'target'
        for attr in self.attrs:
            UsV = np.linalg.svd(self.data[f'_{attr}'], full_matrices=False)
            self.data[f'_U{attr}'] = ('target', sigma), UsV[0]
            self.data[f'_s{attr}'] = sigma, UsV[1]
            self.data[f'_V{attr}'] = (sigma, 'plasma'), UsV[2]
