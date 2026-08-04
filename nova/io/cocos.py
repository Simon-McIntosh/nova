"""Coordinate conventions and the factor that carries a quantity between two.

A measured array is a number and a convention, and the convention is not written
on the number.  Sauter & Medvedev (CPC 184, 2013) reduce the choice to four
digits -- the sign of the poloidal flux, whether that flux carries the 2*pi, and
the handedness of the cylindrical and flux-surface coordinate triples -- and
enumerate the sixteen combinations.  Converting between two conventions is then
the composition of their digits, and every affected quantity picks up one of a
small set of factors depending on what kind of quantity it is.

The Data Dictionary publishes that kind per field as a transformation type, so a
map row names the DD's own tag and the factor follows from the two conventions
rather than from a hand-written table.  Getting the factor from the algebra is
what makes an error in either convention visible: a wrong digit moves several
factors at once and the round trip catches it.

Two things this module deliberately refuses to guess.

``e_bp`` -- whether an array carries the 2*pi -- **must be declared per source
array, never inferred from its magnitude.**  A small, low-field machine has a
small flux in Weber, so a magnitude threshold reads the digit off the machine
size instead of off the convention.  It also cannot be read off a units string,
because a units string is metadata and metadata is wrong often enough to have
cost real time.  Declaring it is what lets a measured total flux in Weber and a
reconstructed flux function in Weber per radian sit in the same map: they are the
same physics in two conventions differing in exactly that digit, and the algebra
then hands the first a factor of one and the second a factor of 2*pi without
either row being special-cased.

Poloidal-angle transformation is not offered.  The digit algebra has an answer
for it, but a sensitive-axis orientation is a statement about installed hardware,
and where a measured orientation exists it is authored from the measurement
rather than transformed from somebody else's convention.  A caller who needs an
angle carried between conventions has to say so explicitly somewhere a reader
will see it, which is the point.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import tau
from typing import Iterator

CONVENTION_DIGITS: dict[int, tuple[int, int, int, int]] = {
    1: (+1, 0, +1, +1),
    2: (+1, 0, -1, +1),
    3: (-1, 0, +1, -1),
    4: (-1, 0, -1, -1),
    5: (+1, 0, +1, -1),
    6: (+1, 0, -1, -1),
    7: (-1, 0, +1, +1),
    8: (-1, 0, -1, +1),
    11: (+1, 1, +1, +1),
    12: (+1, 1, -1, +1),
    13: (-1, 1, +1, -1),
    14: (-1, 1, -1, -1),
    15: (+1, 1, +1, -1),
    16: (+1, 1, -1, -1),
    17: (-1, 1, +1, +1),
    18: (-1, 1, -1, +1),
}
"""Sauter & Medvedev CPC 184 (2013) Table I: the sixteen conventions as digits.

Ordered ``(sigma_bp, e_bp, sigma_r_phi_z, sigma_rho_theta_phi)``.  The eleven-to-
eighteen block is the one-to-eight block with the 2*pi moved into the flux, which
is why a convention and the same convention with the flux scaled differ by ten.
"""

PSI_LIKE = "psi_like"
"""Poloidal flux and anything sharing its sign and its 2*pi."""

IP_LIKE = "ip_like"
"""A toroidal current: its sign follows the direction of the toroidal unit vector."""

B0_LIKE = "b0_like"
"""A toroidal field or ``R * B_phi``: same dependence as a toroidal current."""

Q_LIKE = "q_like"
"""The safety factor, which carries the flux-surface handedness as well."""

DODPSI_LIKE = "dodpsi_like"
"""A derivative with respect to the poloidal flux: the reciprocal of psi_like."""

ONE_LIKE = "one_like"
"""A quantity no convention touches -- a length, or a field a probe reads locally."""

TRANSFORMATIONS = frozenset({PSI_LIKE, IP_LIKE, B0_LIKE, Q_LIKE, DODPSI_LIKE, ONE_LIKE})
"""The transformation types this module derives a factor for."""


class ConventionError(ValueError):
    """Raised when a convention or a transformation type is not recognised."""


@dataclass(frozen=True)
class Convention:
    """One coordinate convention, as the four digits that define it."""

    identifier: int
    sigma_bp: int
    e_bp: int
    sigma_r_phi_z: int
    sigma_rho_theta_phi: int

    @property
    def digits(self) -> tuple[int, int, int, int]:
        """Return the digits in Table I order."""

        return (
            self.sigma_bp,
            self.e_bp,
            self.sigma_r_phi_z,
            self.sigma_rho_theta_phi,
        )


def convention(identifier: int) -> Convention:
    """Return the named convention's digits."""

    digits = CONVENTION_DIGITS.get(int(identifier))
    if digits is None:
        raise ConventionError(
            f"{identifier!r} is not one of the sixteen conventions "
            f"{sorted(CONVENTION_DIGITS)}"
        )
    return Convention(int(identifier), *digits)


def identify_convention(
    *,
    sigma_bp: int,
    e_bp: int,
    sigma_r_phi_z: int,
    sigma_rho_theta_phi: int,
) -> int:
    """Return the convention the four digits name."""

    want = (sigma_bp, e_bp, sigma_r_phi_z, sigma_rho_theta_phi)
    for identifier, digits in CONVENTION_DIGITS.items():
        if digits == want:
            return identifier
    raise ConventionError(f"no convention has digits {want}")


def conventions() -> Iterator[Convention]:
    """Iterate every convention, ascending."""

    for identifier in sorted(CONVENTION_DIGITS):
        yield convention(identifier)


@dataclass(frozen=True)
class ConventionTransform:
    """The composed digits carrying quantities from one convention to another.

    Each effective sign is the product of the two conventions' digits, and the
    effective flux exponent is their difference, so the transform is its own
    inverse when the two conventions are swapped.  Nothing here is a table of
    factors: :meth:`factor` composes them, and a mistake in a digit therefore
    shows up on several quantities rather than on one.
    """

    source: Convention
    target: Convention

    @property
    def sigma_bp(self) -> int:
        """Return the effective poloidal-flux sign."""

        return self.source.sigma_bp * self.target.sigma_bp

    @property
    def sigma_r_phi_z(self) -> int:
        """Return the effective cylindrical handedness."""

        return self.source.sigma_r_phi_z * self.target.sigma_r_phi_z

    @property
    def sigma_rho_theta_phi(self) -> int:
        """Return the effective flux-surface handedness."""

        return self.source.sigma_rho_theta_phi * self.target.sigma_rho_theta_phi

    @property
    def flux_exponent(self) -> int:
        """Return how many factors of 2*pi the poloidal flux gains."""

        return self.target.e_bp - self.source.e_bp

    def factor(self, transformation: str) -> float:
        """Return the factor one transformation type's quantities are scaled by."""

        if transformation not in TRANSFORMATIONS:
            raise ConventionError(
                f"no factor is derived for transformation type {transformation!r}; "
                f"supported types are {sorted(TRANSFORMATIONS)}"
            )
        if transformation == ONE_LIKE:
            return 1.0
        if transformation in (IP_LIKE, B0_LIKE):
            return float(self.sigma_r_phi_z)
        if transformation == Q_LIKE:
            return float(self.sigma_rho_theta_phi * self.sigma_r_phi_z)
        flux = self.sigma_bp * self.sigma_r_phi_z
        if transformation == PSI_LIKE:
            return float(flux) * tau**self.flux_exponent
        return float(flux) * tau**-self.flux_exponent

    def inverse(self) -> ConventionTransform:
        """Return the transform back to the source convention."""

        return ConventionTransform(source=self.target, target=self.source)


def convention_transform(*, source: int, target: int) -> ConventionTransform:
    """Return the transform between two named conventions."""

    return ConventionTransform(source=convention(source), target=convention(target))


def transform_factor(transformation: str, *, source: int, target: int) -> float:
    """Return the factor one transformation type takes between two conventions."""

    return convention_transform(source=source, target=target).factor(transformation)
