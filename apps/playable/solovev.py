"""The playable Solov'ev machine.

The small Solov'ev free-boundary problem is the same one the reduced-Newton
contract uses (``tests/test_reduced_newton.py``): a ring of external
conductors fitted to hold an analytic seed, driven by an edge-vanishing
absolute source so the wall-limited branch attracts.  Here the external
conductors are carried as a prescribed-current field — the same response-
carrier shape the MAST frozen-six carrier uses — so a constrained keyframe
can compensate with actual circuit currents instead of waiting for a profile
amplitude route that is not exercised on this machine.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import mu_0

from nova.biot.greens import hybrid_greens
from nova.biot.null import Null1D, Null2D
from nova.equilibrium.conservation import FluxLattice
from nova.equilibrium.forward import (
    ForwardProfile,
    _cubic_cell_average_stencil,
    _lattice_cells,
)
from nova.equilibrium.forward_operator import (
    FluxTarget,
    ForwardFluxOperator,
    PrescribedCurrentField,
)
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh
from nova.geometry.hexstencil import hex_stencil
from nova.jax.config import configure_dtypes
import jax.numpy as jnp

from apps.playable.production import ForwardMachine

#: The same analytic seed the reduced-Newton fixture pins.
P_PRIME = -3.0e5
FF_PRIME = -0.25
AXIS_RADIUS = 1.0
SEED_SPAN = 0.35
DRIVE = 1.4
BOUNDARY_FIELD_FUNCTION = 5.0
CONDUCTORS = 16
CONDUCTOR_RADIUS = 0.62

#: Radius of each fitted conductor's outline ring in the poloidal view [m].
CONDUCTOR_OUTLINE_RADIUS = 0.05


def conductor_centres() -> np.ndarray:
    """Return the fitted external-conductor ring centres [m].

    The sixteen conductors sit on a circle of radius ``CONDUCTOR_RADIUS``
    about the axis; the free-boundary seed is held by currents on this ring.
    """
    angle = 2 * np.pi * np.arange(CONDUCTORS) / CONDUCTORS
    return np.c_[
        AXIS_RADIUS + CONDUCTOR_RADIUS * np.cos(angle),
        CONDUCTOR_RADIUS * np.sin(angle),
    ]


def coil_outlines(
    *, ring_radius: float = CONDUCTOR_OUTLINE_RADIUS, vertices: int = 25
) -> tuple[np.ndarray, ...]:
    """Return one small outline ring per fitted conductor centre.

    The rings are the Solov'ev machine's conductor outlines: each is a small
    polygon centred on one conductor, so the poloidal view's coil channel
    draws the coil set the seed is held by rather than a decorative ring.
    """
    centres = conductor_centres()
    angle = 2 * np.pi * np.arange(vertices) / vertices
    ring = ring_radius * np.c_[np.cos(angle), np.sin(angle)]
    return tuple(centre + ring for centre in centres)


def _terms() -> tuple[float, float, float]:
    """Return the Solov'ev quartic, offset and vertical coefficients."""
    alpha = np.pi**2 * mu_0 * P_PRIME / 2.0
    return alpha, -2.0 * alpha * AXIS_RADIUS**2, 2.0 * np.pi**2 * FF_PRIME


def _solovev(radius, height) -> np.ndarray:
    """Return the analytic seed flux [Wb] the conductors are fitted to."""
    alpha, offset, beta = _terms()
    return alpha * radius**4 + offset * radius**2 + beta * height**2


def _wall_loop(points: int = 61):
    """Return a material boundary lying on one seed flux surface."""
    alpha, offset, beta = _terms()
    wall_flux = _solovev(AXIS_RADIUS, 0.0) - SEED_SPAN
    inner, outer = np.sqrt(np.sort(np.roots([alpha, offset, -wall_flux])))
    centre, half = 0.5 * (inner + outer), 0.5 * (outer - inner)
    angle = 2 * np.pi * np.arange(points) / points
    radius = centre + half * np.cos(angle)
    argument = np.clip((wall_flux - _solovev(radius, 0.0)) / beta, 0.0, None)
    return np.c_[radius, np.sign(np.sin(angle)) * np.sqrt(argument)], wall_flux


def _green_block(target, source, section: float = 0.05) -> np.ndarray:
    """Return the total-flux coupling [Wb/A] of one source set on one target."""
    return np.stack(
        [
            hybrid_greens(target[:, 0], target[:, 1], a, z, section, section)[0]
            for a, z in source
        ],
        axis=1,
    )


def _flat_profile(amplitude):
    """Return a constant absolute gradient."""

    def gradient(psi_norm):
        """Return the constant value at every normalised flux."""
        return jnp.full_like(jnp.asarray(psi_norm, dtype=jnp.float64), amplitude)

    return gradient


def _edge_vanishing_profile(amplitude):
    """Return an absolute gradient that falls linearly to zero at the edge."""

    def gradient(psi_norm):
        """Return the tapered value at one normalised flux."""
        return amplitude * (1.0 - jnp.clip(jnp.asarray(psi_norm), 0.0, 1.0))

    return gradient


class _Fit:
    """The analytic seed and the conductor state fitted to hold it."""

    def __init__(self) -> None:
        """Fit the base conductor currents on the flat-source map."""
        self.lattice = FluxLattice(
            np.linspace(0.6, 1.42, 15), np.linspace(-0.42, 0.42, 15)
        )
        coordinate = self.lattice.coordinate
        self.wall, wall_flux = _wall_loop()
        seed_flux = _solovev(coordinate[:, 0], coordinate[:, 1])
        wall_seed = _solovev(self.wall[:, 0], self.wall[:, 1])
        self.inside = seed_flux >= wall_flux
        conductor = conductor_centres()
        self.coupling = {
            "plasma_to_grid": _green_block(coordinate, coordinate),
            "plasma_to_wall": _green_block(self.wall, coordinate),
            "source_to_grid": _green_block(coordinate, conductor),
            "source_to_wall": _green_block(self.wall, conductor),
        }
        seed = np.r_[seed_flux, wall_seed]
        flat = self._profile(
            DomainProfile(
                p_prime=_flat_profile(P_PRIME), ff_prime=_flat_profile(FF_PRIME)
            ),
            np.zeros(CONDUCTORS),
        )
        cell_current = np.asarray(flat.operator.cell_current(seed))
        target = np.r_[
            seed_flux - self.coupling["plasma_to_grid"] @ cell_current,
            wall_seed - self.coupling["plasma_to_wall"] @ cell_current,
        ]
        weight = np.r_[self.inside.astype(float), np.ones(len(self.wall))]
        matrix = np.r_[self.coupling["source_to_grid"], self.coupling["source_to_wall"]]
        self.base_current = np.linalg.lstsq(
            matrix * weight[:, None], target * weight, rcond=None
        )[0]
        self.seed = seed

    def _profile(self, core: DomainProfile, current: np.ndarray) -> ForwardProfile:
        """Return the solve for one declared source and conductor state."""
        return ForwardProfile.from_lattice(
            self.lattice,
            ForwardSource(core=core, boundary_field_function=BOUNDARY_FIELD_FUNCTION),
            external_current=current,
            wall_coordinate=self.wall,
            polarity=1,
            inside_material=self.inside,
            **self.coupling,
        )


def build_machine() -> "ForwardMachine":
    """Return the playable Solov'ev machine and its analytic seed."""
    configure_dtypes()
    fit = _Fit()
    lattice = fit.lattice
    coordinate = lattice.coordinate
    grid_null = Null2D.from_coordinates(
        coordinate, hex_stencil(lattice.shape), maxsize=5
    )
    wall_null = Null1D(jnp.asarray(fit.wall, dtype=jnp.float64))
    moment_mesh = StencilMesh(
        coordinate=coordinate,
        stencil=hex_stencil(lattice.shape),
        area=lattice.cell_area,
    )
    stencil, weight = _cubic_cell_average_stencil(lattice.shape)
    profile = ForwardProfile(
        operator=ForwardFluxOperator(
            grid=FluxTarget(
                source_target=jnp.asarray(fit.coupling["source_to_grid"]),
                plasma_target=jnp.asarray(fit.coupling["plasma_to_grid"]),
                null=grid_null,
            ),
            wall=FluxTarget(
                source_target=jnp.asarray(fit.coupling["source_to_wall"]),
                plasma_target=jnp.asarray(fit.coupling["plasma_to_wall"]),
                null=wall_null,
            ),
            source=ForwardSource(
                core=DomainProfile(
                    p_prime=_edge_vanishing_profile(2.0 * DRIVE * P_PRIME),
                    ff_prime=_edge_vanishing_profile(2.0 * DRIVE * FF_PRIME),
                ),
                boundary_field_function=BOUNDARY_FIELD_FUNCTION,
            ),
            external_current=jnp.zeros(CONDUCTORS),
            area=jnp.asarray(lattice.cell_area),
            cell_average_stencil=stencil,
            cell_average_weight=weight,
            polarity=1,
            inside_material=fit.inside,
            moment_geometry=MomentGeometry.from_cells(
                moment_mesh, _lattice_cells(lattice)
            ),
            use_linear_moments=False,
            prescribed_current_field=PrescribedCurrentField(
                response=jnp.asarray(
                    np.vstack(
                        [
                            fit.coupling["source_to_grid"],
                            fit.coupling["source_to_wall"],
                        ]
                    )
                ),
                current=jnp.asarray(fit.base_current),
            ),
        ),
        lattice=lattice,
    )
    return ForwardMachine(
        profile=profile,
        seed=jnp.asarray(fit.seed),
        wall=fit.wall,
        identity="solovev",
    )
