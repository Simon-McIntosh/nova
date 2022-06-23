"""Biot-Savart calculation for complete circular filaments."""
from dataclasses import dataclass
from typing import ClassVar

import dask.array as da
import numpy as np
import xarray

from nova.electromagnetic.biotconstants import BiotConstants
from nova.electromagnetic.biotmatrix import BiotMatrix


@dataclass
class OffsetFilaments:
    """Offset source and target filaments."""

    data: dict[str, da.Array]

    fold_number: int = 0  # Number of e-foling lenghts within filament
    merge_number: int = 1  # Merge radius, multiple of filament widths
    rms_offset: bool = True  # Maintain rms offset for filament pairs

    def __post_init__(self):
        """Offset coincident filaments."""
        self.apply_offset()

    def __getitem__(self, attr):
        """Return attributes from dataset."""
        return self.data[attr]

    def __setitem__(self, attr, value):
        """Update dataset attribute."""
        self.data[attr] = value

    def effective_turn_radius(self):
        """Return effective source turn radius."""
        return da.max(da.stack([self['dx'], self['dz']]), axis=0) / 2

    def source_target_seperation(self):
        """Return source-target seperation vector."""
        return da.stack([self['r']-self['rs'], self['z']-self['zs']])

    def turnturn_seperation(self):
        """Return self seperation length."""
        return 0.5 * self['dx'] * self['turnturn']

    def blending_factor(self, span_length, turn_radius):
        """Return blending factor."""
        if self.fold_number == 0:
            # linear
            return 1 - span_length / (turn_radius * self.merge_number)
        # exponential
        return np.exp(-self.fold_number * (span_length / turn_radius)**2)

    def apply_rms_offset(self, merge_index, radial_offset):
        """Return effective rms offfset."""
        merge_index = merge_index.compute()
        source_radius = self['rs'][merge_index].compute_chunk_sizes()
        target_radius = self['r'][merge_index].compute_chunk_sizes()
        radial_offset = radial_offset[merge_index].compute_chunk_sizes()
        rms_delta = np.zeros(merge_index.shape)
        rms_delta[merge_index] = (np.sqrt(
            (target_radius + source_radius)**2 -
            8*radial_offset*(target_radius - source_radius + 2*radial_offset))
            - (target_radius + source_radius)) / 4
        self['rs'] += rms_delta
        self['r'] += rms_delta

    def apply_offset(self):
        """Apply radial and vertical offsets."""
        turn_radius = self.effective_turn_radius()
        span = self.source_target_seperation()
        span_length = da.linalg.norm(span, axis=0)
        # index
        merge_index = span_length <= turn_radius*self.merge_number
        if not merge_index.any().compute():
            return
        # interacton orientation
        turn_index = da.isclose(span_length, 0)
        pair_index = da.invert(turn_index)
        span_norm = da.zeros((2, *turn_index.shape))
        span_norm[0] = da.where(turn_index, 1, span_norm[0])  # radial offset
        for i in range(2):
            span_norm[i] = da.where(pair_index, span[i] / span_length, 0)
        turnturn_length = self.turnturn_seperation()
        # blend interaction
        blending_factor = self.blending_factor(span_length, turn_radius)
        radial_offset = blending_factor*turnturn_length*span_norm[0, :]
        if self.rms_offset:
            self.apply_rms_offset(merge_index, radial_offset)
        vertical_offset = blending_factor*turnturn_length*span_norm[1, :]
        # offset source filaments
        self['rs'] -= np.where(merge_index, radial_offset/2, 0)
        self['zs'] -= np.where(merge_index, vertical_offset/2, 0)
        # offset target filaments
        self['r'] += np.where(merge_index, radial_offset/2, 0)
        self['z'] += np.where(merge_index, vertical_offset/2, 0)


@dataclass
class BiotRing(BiotMatrix):
    """
    Extend Biot base class.

    Compute interaction for complete circular filaments.

    """

    name: ClassVar[str] = 'ring'  # element name
    attrs: ClassVar[list[str]] = dict(
        rs='rms', zs='z', dx='dx', dz='dz', turnturn='turnturn', r='x', z='z')

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()
        OffsetFilaments(self.data)
        self.const = BiotConstants(self['rs'], self['zs'],
                                   self['r'], self['z'])

    @property
    def Aphi(self):
        """Return Aphi dask array."""
        return 1 / (2*np.pi) * self.const['a']/self['r'] * \
            ((1 - self.const['k2']/2) * self.const['K'] - self.const['E'])

    @property
    def Psi(self):
        """Return Psi dask array."""
        return 2 * np.pi * self.mu_o * self['r'] * self.Aphi

    @property
    def Br(self):
        """Return radial field dask array."""
        return self.mu_o / (2*np.pi) * self.const['gamma'] * \
            (self.const['K'] - (2-self.const['k2']) / (2*self.const['ck2']) *
             self.const['E']) / (self.const['a'] * self['r'])

    @property
    def Bz(self):
        """Return vertical field dask array."""
        return self.mu_o / (2*np.pi) * \
            (self['r']*self.const['K'] -
             (2*self['r'] - self.const['b']*self.const['k2']) /
             (2*self.const['ck2']) * self.const['E']) / \
            (self.const['a']*self['r'])
