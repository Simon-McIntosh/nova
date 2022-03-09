"""Biot-Savart calculation base class."""
from dataclasses import dataclass
from typing import ClassVar

import dask.array as da
import numpy as np
import xarray

from nova.electromagnetic.biotring import BiotRing
from nova.electromagnetic.biotcalc import BiotMatrix


@dataclass
class BiotSolve(BiotMatrix):
    """Manage biot interaction between multiple filament types."""

    block: bool = False
    svd: bool = True
    dask: bool = False

    generator: ClassVar[dict] = {'ring': BiotRing}
    partition_size: ClassVar[int] = 2_000_000
    chunk_size: ClassVar[int] = 2000

    def __post_init__(self):
        """Solve biot interaction."""
        super().__post_init__()
        self.solve()

    def solve(self):
        """Calculate full ensemble biot interaction."""
        self.initialize_dataset()
        self.check_segments()
        for segment in self.source.segment.unique():
            if self.block:
                self._solve_block(segment)
                continue
            self._solve(segment)
        if self.svd:
            self.decompose()

    def check_segments(self):
        """Check for segment in self.generator."""
        for segment in self.source.segment.unique():
            if segment not in self.generator:
                raise NotImplementedError(
                    f'segment <{segment}> not implemented '
                    f'in Biot.generator: {self.generator.keys()}')

    def source_index(self, segment):
        """Return source segment index."""
        source = self.source.segment[self.get_index('source')]
        return np.array(source == segment)

    def plasma_index(self, segment):
        """Return plasma segment index."""
        plasma = self.source.segment[self.source.index[self.source.plasma]]
        return np.array(plasma == segment)

    def _store_segment(self, segment: str, data: xarray.Dataset):
        """Store segment solution data to dataframe."""
        source_index = self.source_index(segment)
        plasma_index = self.plasma_index(segment)
        for attr in self.attrs:
            self.data[attr].loc[:, source_index] = data[attr]
            self.data[f'_{attr}'].loc[:, plasma_index] = data[f'_{attr}']

    def _solve(self, segment: str):
        """Solve single block biot interaction."""
        data = self.generator[segment](
            self.source.loc[self.source.segment == segment, :],
            self.target,
            turns=self.turns, reduce=self.reduce,
            attrs=self.attrs).data
        self._store_segment(segment, data)

    def _solve_block(self, segment: str):
        """Solve multi-block biot interaction."""
        block_size = len(self.source) * len(self.target)
        partition_number = 1 + block_size // self.partition_size
        target_array = np.array_split(range(len(self.target)),
                                      partition_number)
        for i, target_index in enumerate(target_array):
            partition_data = self.generator[segment](
                        self.source.loc[self.source.segment == segment, :],
                        self.target.iloc[target_index, :],
                        turns=self.turns, reduce=self.reduce,
                        attrs=self.attrs).data
            if i == 0:
                data = partition_data
                continue
            data = xarray.concat([data, partition_data], 'target')
        self._store_segment(segment, data)

    def _store_svd(self, attr, UsV):
        """Store svd to xarray dataset."""
        if self.data.dims['plasma'] < self.data.dims['target']:
            sigma = 'plasma'
        else:
            sigma = 'target'
        self.data[f'_U{attr}'] = ('target', sigma), UsV[0]
        self.data[f'_s{attr}'] = sigma, UsV[1]
        self.data[f'_V{attr}'] = (sigma, 'plasma'), UsV[2]

    def decompose_numpy(self):
        """Compute plasma svd with numpy."""
        for attr in self.attrs:
            UsV = np.linalg.svd(self.data[f'_{attr}'], full_matrices=False)
            self._store_svd(attr, UsV)

    def decompose_dask(self):
        """Compute plasma svd with dask."""
        for attr in self.attrs:
            data = da.from_array(self.data[f'_{attr}'].data,
                                 chunks=(self.chunk_size, -1))
            UsV = [usv.compute() for usv in da.linalg.svd(data)]
            self._store_svd(attr, UsV)

    def decompose(self):
        """Apply singular value decomposition to plasma interaction matrix."""
        if self.dask:
            return self.decompose_dask()
        self.decompose_numpy()
