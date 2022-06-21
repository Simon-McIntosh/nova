"""Biot-Savart calculation for complete circular filaments."""
from dataclasses import dataclass, field
from typing import ClassVar

import dask.array as da
import numpy as np
import scipy.special

from nova.electromagnetic.biotframe import BiotFrame
from nova.electromagnetic.biotbase import BiotBase


# pylint: disable=no-member  # disable scipy.special module not found


@dataclass
class PolidalCoordinates:
    """Manage poloidal coordinates."""

    source: BiotFrame
    target: BiotFrame

    def __post_init__(self):
        """Extract source and target coordinates."""
        self.source_radius = self.source('rms')
        self.source_height = self.source('z')
        self.target_radius = self.target('x')
        self.target_height = self.target('z')


@dataclass
class PoloidalOffset(PolidalCoordinates):
    """Offset source and target filaments."""

    fold_number: int = 0  # Number of e-foling lenghts within filament
    merge_number: int = 1  # Merge radius, multiple of filament widths
    rms_offset: bool = True  # Maintain rms offset for filament pairs

    def __post_init__(self):
        """Apply radial and vertical offsets to source and target filaments."""
        super().__post_init__()
        self._apply_offsets()

    def effective_turn_radius(self):
        """Return effective source turn radius."""
        return np.max([self.source('dx'), self.source('dz')], axis=0) / 2

    def source_target_seperation(self):
        """Return source-target seperation vector."""
        return da.from_array([self.target_radius-self.source_radius,
                              self.target_height-self.source_height])

    def turnturn_seperation(self, merge_index):
        """Return self seperation length."""
        return 0.5 * self.source('dx').compute()[merge_index] * \
            self.source('turnturn').compute()[merge_index]

    def blending_factor(self, span_length, turn_radius):
        """Return blending factor."""
        if self.fold_number == 0:
            # linear
            return 1 - span_length / (turn_radius * self.merge_number)
        # exponential
        return np.exp(-self.fold_number * (span_length / turn_radius)**2)

    def apply_rms_offset(self, merge_index, radial_offset):
        """Return effective rms offfset."""
        source_radius = self.source_radius.compute()[merge_index]
        target_radius = self.target_radius.compute()[merge_index]
        rms_delta = (da.sqrt(
            (target_radius + source_radius)**2 -
            8*radial_offset*(target_radius - source_radius + 2*radial_offset))
            - (target_radius + source_radius)) / 4
        merge_index = merge_index.compute_chunk_sizes()
        print(rms_delta.compute())

        self.source_radius.map_blocks(
            )

        didx.map_blocks(lambda block, lut=None: lut[block], lut=lut)

        self.source_radius[merge_index] += rms_delta
        self.target_radius[merge_index] += rms_delta

    def _apply_offsets(self):
        """Apply radial and vertical offsets."""
        turn_radius = self.effective_turn_radius()
        span = self.source_target_seperation()
        span_length = da.linalg.norm(span, axis=0)
        # reduce
        merge_index = span_length <= turn_radius*self.merge_number
        # \np.where(span_length <= turn_radius*self.merge_number)[:2]
        turn_radius = turn_radius[merge_index]
        span = da.from_array([span[i][merge_index] for i in range(2)])
        span_length = span_length[merge_index].compute_chunk_sizes()
        # interacton orientation
        turn_index = da.isclose(span_length, 0)
        span_norm = da.zeros((2, *turn_index.shape))
        span_norm[0, turn_index] = 1  # radial offset
        span_norm[:, ~turn_index] = \
            span[:, ~turn_index].compute_chunk_sizes() / \
            span_length[~turn_index].compute_chunk_sizes()
        turnturn_length = self.turnturn_seperation(merge_index)
        # blend interaction
        blending_factor = self.blending_factor(span_length, turn_radius)
        radial_offset = blending_factor*turnturn_length*span_norm[0, :]
        if self.rms_offset:
            self.apply_rms_offset(merge_index, radial_offset)
        vertical_offset = blending_factor*turnturn_length*span_norm[1, :]
        # offset source filaments
        self.source_radius[merge_index] -= radial_offset/2
        self.source_height[merge_index] -= vertical_offset/2
        # offset target filaments
        self.target_radius[merge_index] += radial_offset/2
        self.target_height[merge_index] += vertical_offset/2


@dataclass
class BiotRing(BiotBase):
    """
    Extend Biot base class.

    Compute interaction for complete circular filaments.

    """

    name = 'ring'  # element name
    attrs: list[str] = field(default_factory=lambda: [
        'Aphi', 'Psi', 'Br', 'Bz'])
    _attrs: ClassVar[list[str]] = ['Aphi', 'Psi', 'Br', 'Bz']

    def calculate_coefficients(self):
        """Return interaction coefficients."""
        offset = PoloidalOffset(self.source, self.target)
        coeff = {'rs': offset.source_radius, 'zs': offset.source_height,
                 'r': offset.target_radius, 'z': offset.target_height}
        coeff['b'] = coeff['rs'] + coeff['r']
        coeff['gamma'] = coeff['zs'] - coeff['z']
        coeff['a2'] = coeff['gamma']**2 + (coeff['r'] + coeff['rs'])**2
        coeff['a'] = np.sqrt(coeff['a2'])
        coeff['k2'] = 4 * coeff['r'] * coeff['rs'] / coeff['a2']
        coeff['ck2'] = 1 - coeff['k2']  # complementary modulus
        coeff['K'] = scipy.special.ellipk(coeff['k2'])  # ellip integral - 1st
        coeff['E'] = scipy.special.ellipe(coeff['k2'])  # ellip integral - 2nd
        self.coef = coeff

    def calculate_vector_potential(self):
        """Calculate target vector potential (r, phi, z), Wb/Amp-turn-turn."""
        self.vector['Aphi'] = 1 / (2*np.pi) * self.coef['a']/self.coef['r'] * \
            ((1 - self.coef['k2']/2) * self.coef['K'] - self.coef['E'])

    def calculate_scalar_potential(self):
        """Calculate scalar potential."""
        self.vector['Psi'] = 2 * np.pi * self.mu_o * \
            self.coef['r'] * self.vector['Aphi']

    def calculate_magnetic_field(self):
        """Calculate magnetic field (r, phi, z), T/Amp-turn-turn."""
        self.vector['Br'] = \
            self.mu_o / (2*np.pi) * self.coef['gamma'] * \
            (self.coef['K'] - (2-self.coef['k2']) / (2*self.coef['ck2']) *
             self.coef['E']) / (self.coef['a'] * self.coef['r'])
        self.vector['Bz'] = \
            self.mu_o / (2*np.pi) * \
            (self.coef['r']*self.coef['K'] -
             (2*self.coef['r'] - self.coef['b']*self.coef['k2']) /
             (2*self.coef['ck2']) * self.coef['E']) / \
            (self.coef['a']*self.coef['r'])
