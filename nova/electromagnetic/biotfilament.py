"""Biot-Savart calculation for complete circular filaments."""
from dataclasses import dataclass

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
class BiotCircle(BiotSolve):
    """
    Extend BiotMatrix base class.

    Compute interaction for complete circular filaments.

    """

    name = 'circle'  # element name

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
        self.vector['Ay'] = 1 / (2*np.pi) * coeff['a']/coeff['r'] * \
            ((1 - coeff['k2']/2) * coeff['K'] - coeff['E'])

    def calculate_scalar_potential(self, coeff):
        """Calculate scalar potential."""
        self.vector['Psi'] = 2 * np.pi * self.mu_o * \
            coeff['r'] * self.vector['Ay']

    def calculate_magnetic_field(self, coeff):
        """Calculate magnetic field (r, phi, z), T/Amp-turn-turn."""
        self.vector['Bx'] = self.mu_o / (2*np.pi) * \
            coeff['gamma'] * (coeff['K'] - (2-coeff['k2']) / (2*coeff['ck2']) *
                              coeff['E']) / (coeff['a'] * coeff['r'])
        self.vector['Bz'] = self.mu_o / (2*np.pi) * \
            (coeff['r']*coeff['K'] - (2*coeff['r'] - coeff['b']*coeff['k2']) /
             (2*coeff['ck2']) * coeff['E']) / (coeff['a']*coeff['r'])


@dataclass
class BiotFilament(BiotMatrix):

    def __post_init__(self):
        super().__post_init__()
        self.initialize()

        index = np.array(self.source.segment == 'circle')
        circle = BiotCircle(self.source.loc[index, :],
                            self.target, turns=self.turns, reduce=self.reduce)

        source_index = self.get_coord('source')
        index = self.source.segment[source_index] == 'circle'

        for var in self.data_vars:
            self.static[var].loc[:, index.to_numpy()] = circle.static[var]


if __name__ == '__main__':

    biotframe = BiotFrame(subspace=['Ic'])
    biotframe.insert([10, 10], [-0.5, 0.5], dl=0.95, dt=0.95, section='hex')
    biotframe.insert(11, 0, dl=0.95, dt=0.1, section='sk')
    biotframe.insert(12, 0, dl=0.6, dt=0.9, section='r', segment='circle')
    #biotframe.insert([1, 3], 2, dl=0.95, dt=0.95, section='sq', link=True)
    #biotframe.insert([1, 3], 3, dl=0.95, dt=0.6, section='sk', link=True)

    biotframe.multipoint.link(['Coil0', 'Coil1'], -1)

    biotframe.polyplot()

    x, z = np.linspace(9.5, 12.5, 100), np.linspace(-1, 1, 300)
    X, Z = np.meshgrid(x, z, indexing='ij')
    target = BiotFrame()
    target.insert(X.flatten(), Z.flatten())

    filament = BiotFilament(biotframe, target, reduce=[True, False])

    biotframe.subspace.Ic = [1, 0.7, 1.65]

    from nova.utilities.pyplot import plt

    Psi = np.dot(filament.static.Psi, biotframe.subspace.Ic)
    plt.contour(x, z, Psi.reshape(100, 300).T, 51)
