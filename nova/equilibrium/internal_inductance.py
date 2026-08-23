r"""Internal-inductance definitions with explicit geometric support.

Both definitions below normalise the same poloidal-field energy

.. math::
    E_p = \int_{V_C} B_p^2\,\mathrm{d}V

inside one declared closed contour :math:`C`, but they do not share a
normaliser.

The IMAS data-dictionary ``li_3`` convention is

.. math::
    l_i(3) = \frac{2 E_p}{\mu_0^2 I_p^2 R_{\mathrm{geo}}},

where :math:`I_p` is the toroidal current enclosed by the same contour and
:math:`R_{\mathrm{geo}}=(R_{\max}+R_{\min})/2` is that contour's geometric
major radius.  This is the definition documented for
``equilibrium/time_slice/global_quantities/li_3`` in IMAS DD 4.1.1:
https://imas-data-dictionary.readthedocs.io/en/4.1.1/generated/ids/summary.html

The boundary-average ``li_2`` convention is

.. math::
    l_i(2) = \frac{E_p}{V_C (G_C/l_p)^2}, \qquad
    G_C = \oint_C B_p\,\mathrm{d}l,

where :math:`V_C` is the toroidal volume inside :math:`C` and :math:`l_p` is
the poloidal perimeter of :math:`C`.  Conversion is therefore a ratio of
normalisers; it is exact only when the energy, current, volume, perimeter and
geometric radius all describe the same contour support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from scipy.constants import mu_0

__all__ = [
    "Li2Geometry",
    "Li3Geometry",
    "convert_li_2_to_li_3",
    "convert_li_3_to_li_2",
    "li_2_from_field_energy",
    "li_2_normaliser",
    "li_3_from_field_energy",
    "li_3_normaliser",
]


@dataclass(frozen=True)
class Li2Geometry:
    """Geometry that defines a boundary-average internal inductance.

    ``toroidal_volume`` is the volume enclosed by ``boundary_support``.
    ``boundary_circulation`` and ``boundary_perimeter`` are line integrals on
    that same contour.  The field energy supplied to :func:`li_2_from_field_energy`
    must be integrated over ``field_energy_support``.
    """

    toroidal_volume: Any
    boundary_circulation: Any
    boundary_perimeter: Any

    field_energy_support: ClassVar[str] = "toroidal volume enclosed by the LCFS"
    volume_support: ClassVar[str] = "toroidal volume enclosed by the LCFS"
    boundary_support: ClassVar[str] = "the same LCFS contour"


@dataclass(frozen=True)
class Li3Geometry:
    """Geometry that defines the IMAS DD ``li_3`` normaliser.

    ``plasma_current`` is the toroidal current enclosed by the LCFS.
    ``geometric_major_radius`` is half the sum of that LCFS's minimum and
    maximum major radii.  The field energy supplied to
    :func:`li_3_from_field_energy` must be integrated over
    ``field_energy_support``.
    """

    plasma_current: Any
    geometric_major_radius: Any

    field_energy_support: ClassVar[str] = "toroidal volume enclosed by the LCFS"
    current_support: ClassVar[str] = "toroidal current enclosed by the same LCFS"
    radius_definition: ClassVar[str] = "half-sum of the LCFS radial extrema"


def li_2_normaliser(geometry: Li2Geometry):
    """Return ``V * (integral_C(Bp dl) / lp)**2`` in T^2 m^3."""

    mean_boundary_field = geometry.boundary_circulation / geometry.boundary_perimeter
    return geometry.toroidal_volume * mean_boundary_field**2


def li_3_normaliser(geometry: Li3Geometry):
    """Return ``0.5 * mu0**2 * Ip**2 * Rgeo`` in T^2 m^3."""

    return 0.5 * mu_0**2 * geometry.plasma_current**2 * geometry.geometric_major_radius


def li_2_from_field_energy(field_energy, geometry: Li2Geometry):
    """Return boundary-average ``li_2`` from one LCFS-supported energy."""

    return field_energy / li_2_normaliser(geometry)


def li_3_from_field_energy(field_energy, geometry: Li3Geometry):
    """Return IMAS DD ``li_3`` from one LCFS-supported energy."""

    return field_energy / li_3_normaliser(geometry)


def convert_li_2_to_li_3(
    internal_inductance,
    li_2_geometry: Li2Geometry,
    li_3_geometry: Li3Geometry,
):
    """Convert ``li_2`` to ``li_3`` without changing field energy."""

    return (
        internal_inductance
        * li_2_normaliser(li_2_geometry)
        / li_3_normaliser(li_3_geometry)
    )


def convert_li_3_to_li_2(
    internal_inductance,
    li_3_geometry: Li3Geometry,
    li_2_geometry: Li2Geometry,
):
    """Convert ``li_3`` to ``li_2`` without changing field energy."""

    return (
        internal_inductance
        * li_3_normaliser(li_3_geometry)
        / li_2_normaliser(li_2_geometry)
    )
