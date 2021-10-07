"""Biot-Savart calculation for complete circular filaments."""
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import scipy.special

from nova.electromagnetic.biotframe import BiotFrame
from nova.electromagnetic.biotdata import BiotMatrix, BiotSolve


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
        return np.array([(self.target_radius-self.source_radius),
                         (self.target_height-self.source_height)])

    def turnturn_seperation(self, merge_index):
        """Return self seperation length."""
        return 0.5 * self.source('dx')[merge_index] * \
            self.source('turnturn')[merge_index]

    def blending_factor(self, span_length, turn_radius):
        """Return blending factor."""
        if self.fold_number == 0:
            # linear
            return 1 - span_length / (turn_radius * self.merge_number)
        # exponential
        return np.exp(-self.fold_number * (span_length / turn_radius)**2)

    def apply_rms_offset(self, merge_index, radial_offset):
        """Return effective rms offfset."""
        source_radius = self.source_radius[merge_index]
        target_radius = self.target_radius[merge_index]
        rms_delta = (np.sqrt(
            (target_radius + source_radius)**2 -
            8*radial_offset*(target_radius - source_radius + 2*radial_offset))
            - (target_radius + source_radius)) / 4
        self.source_radius[merge_index] += rms_delta
        self.target_radius[merge_index] += rms_delta

    def _apply_offsets(self):
        """Apply radial and vertical offsets."""
        turn_radius = self.effective_turn_radius()
        span = self.source_target_seperation()
        span_length = np.linalg.norm(span, axis=0)
        # reduce
        merge_index = np.where(span_length <= turn_radius*self.merge_number)[0]
        turn_radius = turn_radius[merge_index]
        span = span[:, merge_index]
        span_length = span_length[merge_index]
        # interacton orientation
        turn_index = np.isclose(span_length, 0)
        span_norm = np.zeros((2, len(turn_index)))
        span_norm[0, turn_index] = 1  # radial offset
        span_norm[:, ~turn_index] = \
            span[:, ~turn_index] / span_length[~turn_index]
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
class BiotRing(BiotSolve):
    """
    Extend BiotMatrix base class.

    Compute interaction for complete circular filaments.

    """

    name = 'ring'  # element name

    columns: list[str] = field(default_factory=lambda: [
        'Aphi', 'Psi', 'Br', 'Bz'])

    def calculate_coefficients(self) -> dict[npt.ArrayLike]:
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
        return coeff

    def calculate_vector_potential(self, coeff):
        """Calculate target vector potential (r, phi, z), Wb/Amp-turn-turn."""
        self.vector['Aphi'] = 1 / (2*np.pi) * coeff['a']/coeff['r'] * \
            ((1 - coeff['k2']/2) * coeff['K'] - coeff['E'])

    def calculate_scalar_potential(self, coeff):
        """Calculate scalar potential."""
        self.vector['Psi'] = 2 * np.pi * self.mu_o * \
            coeff['r'] * self.vector['Aphi']

    def calculate_magnetic_field(self, coeff):
        """Calculate magnetic field (r, phi, z), T/Amp-turn-turn."""
        self.vector['Br'] = self.mu_o / (2*np.pi) * \
            coeff['gamma'] * (coeff['K'] - (2-coeff['k2']) / (2*coeff['ck2']) *
                              coeff['E']) / (coeff['a'] * coeff['r'])
        self.vector['Bz'] = self.mu_o / (2*np.pi) * \
            (coeff['r']*coeff['K'] - (2*coeff['r'] - coeff['b']*coeff['k2']) /
             (2*coeff['ck2']) * coeff['E']) / (coeff['a']*coeff['r'])


@dataclass
class Biot(BiotMatrix):
    """Manage biot interaction between multiple filament types."""

    generator = {'ring': BiotRing}

    def __post_init__(self):
        """Solve biot interaction."""
        super().__post_init__()
        self.calculate()

    def calculate(self):
        """Calculate full ensemble biot interaction."""
        self.initialize_dataset()
        for segment in self.source.segment.unique():
            self.update(segment)

    def source_index(self, segment):
        """Return source segment index."""
        source = self.source.segment[self.get_index('source')]
        return np.array(source == segment)

    def plasma_index(self, segment):
        """Return plasma segment index."""
        plasma = self.source.segment[self.source.index[self.source.plasma]]
        return np.array(plasma == segment)

    def update(self, segment):
        """Calculate segment biot interaction."""
        index = np.array(self.source.segment == segment)
        try:
            data = self.generator[segment](
                self.source.loc[index, :], self.target,
                turns=self.turns, reduce=self.reduce).data
        except KeyError:
            raise NotImplementedError(f'segment {segment} not implemented '
                                      f'in Biot.generator: '
                                      '{self.generator.keys()}')
        source_index = self.source_index(segment)
        plasma_index = self.plasma_index(segment)
        for var in self.columns:
            self.data[var].loc[:, source_index] = data[var]
            self.data[f'_{var}'].loc[:, plasma_index] = data[f'_{var}']
