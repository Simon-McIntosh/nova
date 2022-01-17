"""Biot-Savart calculation base class."""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from nova.electromagnetic.biotring import BiotRing
from nova.electromagnetic.biotcalc import BiotMatrix


@dataclass
class BiotSolve(BiotMatrix):
    """Manage biot interaction between multiple filament types."""

    svd: bool = True
    generator: ClassVar[dict] = {'ring': BiotRing}

    def __post_init__(self):
        """Solve biot interaction."""
        super().__post_init__()
        self.calculate()

    def calculate(self):
        """Calculate full ensemble biot interaction."""
        self.initialize_dataset()
        for segment in self.source.segment.unique():
            self._calculate(segment)
        if self.svd:
            self.decompose()

    def source_index(self, segment):
        """Return source segment index."""
        source = self.source.segment[self.get_index('source')]
        return np.array(source == segment)

    def plasma_index(self, segment):
        """Return plasma segment index."""
        plasma = self.source.segment[self.source.index[self.source.plasma]]
        return np.array(plasma == segment)

    def _calculate(self, segment):
        """Calculate segment biot interaction."""
        index = np.array(self.source.segment == segment)
        if segment not in self.generator:
            raise NotImplementedError(
                f'segment <{segment}> not implemented '
                f'in Biot.generator: {self.generator.keys()}')
        data = self.generator[segment](
            self.source.loc[index, :], self.target,
            turns=self.turns, reduce=self.reduce,
            attrs=self.attrs).data
        source_index = self.source_index(segment)
        plasma_index = self.plasma_index(segment)
        for attr in self.attrs:
            self.data[attr].loc[:, source_index] = data[attr]
            self.data[f'_{attr}'].loc[:, plasma_index] = data[f'_{attr}']

    def decompose(self):
        """Apply singular value decomposition to plasma interaction matrix."""
        if self.data.dims['plasma'] < self.data.dims['target']:
            sigma = 'plasma'
        else:
            sigma = 'target'
        for attr in self.attrs:
            UsV = np.linalg.svd(self.data[f'_{attr}'], full_matrices=False)
            self.data[f'_U{attr}'] = ('target', sigma), UsV[0]
            self.data[f'_s{attr}'] = sigma, UsV[1]
            self.data[f'_V{attr}'] = (sigma, 'plasma'), UsV[2]
