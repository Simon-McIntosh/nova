"""Biot-Savart calculation base class."""
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray

from nova.electromagnetic.biotcylinder import BiotCylinder
from nova.electromagnetic.biotpolygon import BiotPolygon
from nova.electromagnetic.biotring import BiotRing
from nova.electromagnetic.biotset import BiotSet


@dataclass
class BiotSolve(BiotSet):
    """Manage biot interaction between multiple filament types."""

    attrs: list[str] = field(default_factory=lambda: [
        'Aphi', 'Psi', 'Br', 'Bz'])
    svd: bool = True
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
        for segment in self.source.segment.unique():
            if segment not in self.generator:
                raise NotImplementedError(
                    f'segment <{segment}> not implemented '
                    f'in Biot.generator: {self.generator.keys()}')

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
        for attr in self.attrs:  # unit filaments
            self.data[f'_{attr}'] = xarray.DataArray(
                0., dims=['target', 'plasma'],
                coords=[self.data.target, self.data.plasma])

    def get_index(self, frame: str) -> list[str]:
        """Return matrix coordinate, reduce if flag True."""
        biotframe = getattr(self, frame)
        if biotframe.reduce:
            return biotframe.biotreduce.index.to_list()
        return biotframe.index.to_list()

    def compose(self):
        """Calculate full ensemble biot interaction."""
        for segment in self.source.segment.unique():
            self.compute(segment)

    def source_index(self, segment):
        """Return source segment index."""
        source = self.source.segment[self.get_index('source')]
        return np.array(source == segment)

    def plasma_index(self, segment):
        """Return plasma segment index."""
        plasma = self.source.segment[self.source.index[self.source.plasma]]
        return np.array(plasma == segment)

    def compute(self, segment: str):
        """Compute segment and update dataset."""
        source_index = self.source_index(segment)
        plasma_index = self.plasma_index(segment)
        generator = self.generator[segment](
            self.source.loc[self.source.segment == segment, :],
            self.target, turns=self.turns, reduce=self.reduce)
        for attr in self.attrs:
            matrix, plasma = generator.compute(attr)
            self.data[attr].loc[:, source_index] = matrix
            self.data[f'_{attr}'].loc[:, plasma_index] = plasma

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
