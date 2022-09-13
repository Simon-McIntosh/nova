"""Biot-Savart calculation for complete circular filaments."""
from dataclasses import dataclass
from typing import ClassVar

import dask.array as da
import numpy as np

from nova.biot.biotconstants import BiotConstants
from nova.biot.biotmatrix import BiotMatrix


@dataclass
class OffsetFilaments:
    """Offset source and target filaments."""

    data: dict[str, da.Array | np.ndarray]

    fold_number: int = 0  # Number of e-foling lenghts within filament
    merge_number: float = 1.5  # Merge radius, multiple of filament widths
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
        return np.max(np.stack([self['dx'], self['dz']]), axis=0) / 2

    def source_target_seperation(self):
        """Return source-target seperation vector."""
        return np.stack([self['r']-self['rs'], self['z']-self['zs']])

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
        merge_index = merge_index
        source_radius = self['rs'][merge_index]
        target_radius = self['r'][merge_index]
        radial_offset = radial_offset[merge_index]
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
        span_length = np.linalg.norm(span, axis=0)
        # index
        merge_index = span_length <= turn_radius*self.merge_number
        if not merge_index.any():
            return
        # interacton orientation
        turn_index = np.isclose(span_length, 0, atol=5e-2*np.max(turn_radius))
        span_norm = span / span_length
        span_norm[0] = np.where(turn_index, 1, span_norm[0])  # radial offset
        span_norm[1] = np.where(turn_index, 0, span_norm[1])
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
    attrs: ClassVar[dict[str, str]] = dict(
        rs='rms', zs='z', dx='dx', dz='dz', turnturn='turnturn', r='x', z='z')

    def __post_init__(self):
        """Load intergration constants."""
        super().__post_init__()
        OffsetFilaments(self.data)
        self.constant = BiotConstants(self['rs'], self['zs'],
                                      self['r'], self['z'])

    def __getattr__(self, attr):
        """Return coefficient evaluated at self.corner."""
        return self.constant[attr]

    @property
    def Aphi(self):
        """Return Aphi array."""
        return 1 / (2*np.pi) * self.a/self.r * \
            ((1 - self.k2/2) * self.K - self.E)

    @property
    def Psi(self):
        """Return Psi array."""
        return 2 * np.pi * self.mu_o * self.r * self.Aphi

    @property
    def Br(self):
        """Return radial field array."""
        return self.mu_o / (2*np.pi) * self.gamma * \
            (self.K - (2-self.k2) / (2*self.ck2) * self.E) / (self.a * self.r)

    @property
    def Bz(self):
        """Return vertical field array."""
        return self.mu_o / (2*np.pi) * \
            (self.r*self.K - (2*self.r - self.b*self.k2) /
             (2*self.ck2) * self.E) / (self.a*self.r)


if __name__ == '__main__':

    from nova.electromagnetic.coilset import CoilSet

    coilset = CoilSet(dcoil=-100, dplasma=-150, chunks=None)
    coilset.coil.insert(5, 0.5, 0.01, 0.8, section='r', turn='r',
                        nturn=300, segment='ring')
    coilset.coil.insert(5.1, 0.5+0.4, 0.2, 0.01, section='r', turn='r',
                        nturn=300, segment='ring')
    coilset.coil.insert(5.1, 0.5-0.4, 0.2, 0.01, section='r', turn='r',
                        nturn=300, segment='ring')
    coilset.coil.insert(5.2, 0.5, 0.01, 0.8, section='r', turn='r',
                        nturn=300, segment='ring')
    coilset.saloc['Ic'] = 5e3

    coilset.grid.solve(2000, 1)
    coilset.grid.plot('bz', colors='C1')
    coilset.plot()
