r"""Forward free-boundary equilibrium solve from prescribed source functions.

``ForwardProfile`` is the public problem: prescribed sources and flux
functions in, equilibrium flux out. It is one cell of Nova's
``ProblemRepresentation`` matrix and has exactly one implementation-level
map, :class:`~nova.equilibrium.forward_operator.ForwardFluxOperator`, which
carries the same solve as a traced function so a fixed-point ladder can
differentiate it and a batch of slices can be solved under ``vmap``.

The solve reads no magnetic measurement, forms no whitened residual and
updates no profile coefficient. A caller that needs a state inferred from
diagnostics reconstructs it first with
:class:`~nova.equilibrium.profile.ReconstructProfile`, or supplies the
posterior source state an upstream estimator produced.

Host and accelerated routes drive the same immutable state and return the
same typed result. ``host`` runs the relaxed fixed-point step under host
control flow with an early exit, ``host_krylov`` hands the residual to a
Jacobian-free Newton-Krylov root find, and the three accelerated routes hand
the map to the shared fixed-point ladder in
:mod:`nova.equilibrium.fixed_point`. All of them report their residual
history through one
:class:`~nova.equilibrium.fixed_point.FixedPointResult`, so a convergence
plot does not need to know which route produced it.

Which fixed point a route reaches is a property of the seed, not of the
accelerator. An absolute source drives current only where the topology read
finds an axis-connected core, so the map always has a second, trivial fixed
point at the vacuum field; a globalised Newton step can cross to it. The
receipt reports the branch it landed on — an empty core and a zero ledger —
rather than hiding the outcome behind a converged residual.

Which route can reach a fixed point at all is a property of the map. A
relaxed iteration converges only where the map contracts, and the
free-boundary map of an elongated column held at fixed conductor currents
does not: displacing the column vertically moves it into shaping field whose
decay index is negative there, so the force acts along the displacement
rather than against it, and the Jacobian of the write-then-read cycle carries
an eigenvalue outside the unit circle on that mode. ``picard`` and
``anderson`` mix successive images of that map, so they walk away from such
an equilibrium however they are damped or seeded, until the column meets the
wall and the source switches itself off — and the vacuum branch they land on
converges to a BETTER residual than the equilibrium they left, which is why a
prescribed-source solve cannot be qualified by its residual alone.
``newton_krylov`` solves ``(I - J) s = f`` rather than iterating ``g``, and a
root find is indifferent to the sign of that eigenvalue, so it holds the
equilibrium — which is why it is the default route. A relaxed route stays the
cheaper choice wherever the map does contract, which is the limited,
low-elongation case, and the residual history says which case a caller is in
without any extra diagnostic: a relaxed residual that GROWS from a good seed
is the non-contractive mode, not a bad start.

No route resolves flux differences below the quantum of the axis-flux read
the normalised flux is formed against. That read is fitted in the precision
the null search carries, so the floor scales with the flux itself rather than
being a fixed number, and a tolerance carried over from another case can be
unreachable on this one. A Newton budget that has already reached the floor
spends its remaining steps rattling inside it rather than descending.

The result is a receipt, not just a flux map: it carries the residual
history, the axis and separatrix state, the domain-labelled current ledger,
the integral observations, the conservation residuals, the finite checks,
the force-balance closure the source declared with the conventions it is
unreadable without, the continuation each open domain was driven under, and
every normalisation action the solve took — which, for an absolute source, is
none.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

from nova.biot.null import Null1D, Null2D
from nova.biot.target import FluxTarget
from nova.equilibrium import fixed_point
from nova.equilibrium.conservation import (
    ConservationLedger,
    FluxLattice,
    FluxMesh,
    conservation_ledger,
    poloidal_field,
)
from nova.equilibrium.domain import DomainMasks
from nova.equilibrium.forward_operator import ForwardFluxOperator
from nova.equilibrium.observation import (
    CurrentLedger,
    IntegralObservation,
    MomentTargets,
    current_ledger,
    moment_residual,
    observe_moments,
    reject_unsupported_enforcement,
)
from nova.equilibrium.source import (
    ContinuationLedger,
    ForwardSource,
    NormalisationRecord,
    RotationRecord,
    absolute_normalisation_record,
)
from nova.equilibrium.stencil_mesh import MomentGeometry, StencilMesh
from nova.equilibrium.topology import TopologyState
from nova.geometry.hexstencil import hex_stencil

__all__ = [
    "FiniteCheck",
    "ForwardEquilibrium",
    "ForwardProfile",
]

#: Routes that drive the same map to the same fixed point.
SolveRoute = Literal["host", "host_krylov", "picard", "anderson", "newton_krylov"]

_HOST: tuple[str, ...] = ("host", "host_krylov")
_ACCELERATED: tuple[str, ...] = ("picard", "anderson", "newton_krylov")


def _lattice_cells(lattice: FluxLattice) -> tuple[np.ndarray, ...]:
    """Return rectangular control polygons centred on a structured lattice."""
    half_radial = 0.5 * lattice.radial_step
    half_vertical = 0.5 * lattice.vertical_step
    offset = np.asarray(
        [
            [-half_radial, -half_vertical],
            [half_radial, -half_vertical],
            [half_radial, half_vertical],
            [-half_radial, half_vertical],
        ]
    )
    return tuple(coordinate + offset for coordinate in lattice.coordinate)


class FiniteCheck(NamedTuple):
    """Finiteness of every array the result publishes."""

    flux: jax.Array
    cell_current: jax.Array
    moments: jax.Array
    conservation: jax.Array

    @property
    def passed(self) -> jax.Array:
        """Return whether every published array is finite."""
        return self.flux & self.cell_current & self.moments & self.conservation


class ForwardEquilibrium(NamedTuple):
    """Converged equilibrium and the receipts that qualify it."""

    flux: jax.Array
    cell_current: jax.Array
    domains: DomainMasks
    topology: TopologyState
    fixed_point: fixed_point.FixedPointResult
    moments: IntegralObservation
    ledger: CurrentLedger
    conservation: ConservationLedger
    normalisation: NormalisationRecord
    rotation: RotationRecord
    continuation: ContinuationLedger
    finite: FiniteCheck


@dataclass
class ForwardProfile:
    """Solve the free-boundary equilibrium for prescribed sources and profiles.

    The flux functions supplied through
    :class:`~nova.equilibrium.source.ForwardSource` set the toroidal current
    density on the domains the topology read labels, and the equilibrium is
    the fixed point of the resulting write-then-read cycle. Fluxes are total
    poloidal fluxes, :math:`\\Phi = 2 \\pi R A_\\phi` in Wb, concatenated over
    the plasma grid nodes followed by the wall nodes.

    ``lattice`` is the mesh the plasma grid is carried on, meeting the
    :class:`~nova.equilibrium.conservation.FluxMesh` contract: a uniform
    :class:`~nova.equilibrium.conservation.FluxLattice` for a structured
    raster, or a :class:`~nova.equilibrium.stencil_mesh.StencilMesh` for the
    offset, wall-trimmed hexagonal tiling the package ships. It is required
    rather than optional because the conservation and integral receipts are
    differentiated on it; a solve that cannot produce its receipts is not the
    capability this class publishes. Nothing else in the solve reads it — the
    map, the ladder and the domain partition are already mesh-agnostic — so
    the mesh kind changes which stencil the receipts are formed on and
    nothing about the equilibrium that is found.
    """

    operator: ForwardFluxOperator
    lattice: FluxMesh
    evaluations: int = 60
    relaxation: float = 0.5
    newton_steps: int = 4

    def __post_init__(self):
        """Validate that the lattice indexes the operator's plasma grid."""
        if self.lattice.node_count != self.operator.grid.node_number:
            raise ValueError(
                "the flux lattice and the plasma grid must carry the same nodes"
            )
        if self.evaluations < 1:
            raise ValueError("evaluations must be at least one")

    @classmethod
    def from_lattice(
        cls,
        lattice: FluxLattice,
        source: ForwardSource,
        *,
        external_current,
        source_to_grid,
        plasma_to_grid,
        source_to_wall,
        plasma_to_wall,
        wall_coordinate,
        plasma_to_grid_r=None,
        plasma_to_grid_z=None,
        plasma_to_wall_r=None,
        plasma_to_wall_z=None,
        polarity: int = 1,
        inside_material=None,
        maxsize: int = 5,
        **kwargs,
    ) -> ForwardProfile:
        """Build the solve from a structured lattice and its coupling operators.

        The coupling operators carry total poloidal flux in Wb per ampere, the
        convention :func:`nova.biot.greens.hybrid_greens` assembles.
        """
        grid_null = Null2D.from_coordinates(
            lattice.coordinate, hex_stencil(lattice.shape), maxsize=maxsize
        )
        wall_null = Null1D(jnp.asarray(wall_coordinate, dtype=jnp.float64))
        moment_mesh = StencilMesh(
            coordinate=lattice.coordinate,
            stencil=hex_stencil(lattice.shape),
            area=lattice.cell_area,
        )
        operator = ForwardFluxOperator(
            grid=FluxTarget(
                source_target=jnp.asarray(source_to_grid),
                plasma_target=jnp.asarray(plasma_to_grid),
                null=grid_null,
                plasma_target_r=(
                    None if plasma_to_grid_r is None else jnp.asarray(plasma_to_grid_r)
                ),
                plasma_target_z=(
                    None if plasma_to_grid_z is None else jnp.asarray(plasma_to_grid_z)
                ),
            ),
            wall=FluxTarget(
                source_target=jnp.asarray(source_to_wall),
                plasma_target=jnp.asarray(plasma_to_wall),
                null=wall_null,
                plasma_target_r=(
                    None if plasma_to_wall_r is None else jnp.asarray(plasma_to_wall_r)
                ),
                plasma_target_z=(
                    None if plasma_to_wall_z is None else jnp.asarray(plasma_to_wall_z)
                ),
            ),
            source=source,
            external_current=jnp.asarray(external_current),
            area=jnp.asarray(lattice.cell_area),
            polarity=polarity,
            inside_material=inside_material,
            moment_geometry=MomentGeometry.from_cells(
                moment_mesh, _lattice_cells(lattice)
            ),
            use_linear_moments=all(
                block is not None
                for block in (
                    plasma_to_grid_r,
                    plasma_to_grid_z,
                    plasma_to_wall_r,
                    plasma_to_wall_z,
                )
            ),
        )
        return cls(operator=operator, lattice=lattice, **kwargs)

    @property
    def source(self) -> ForwardSource:
        """Return the immutable source state the solve consumes."""
        return self.operator.source

    def flux_map(self, current=None) -> Callable[[jax.Array], jax.Array]:
        """Return the traced free-boundary map at one conductor state."""
        return self.operator.flux_map(current)

    def observe(self, flux, current=None) -> ForwardEquilibrium:
        """Return the full receipt of one flux map without iterating it.

        ``fixed_point`` reports the residual of the supplied map alone, so a
        caller can qualify an externally produced flux map through the same
        contract a solve returns.
        """
        residual = jnp.max(jnp.abs(self.operator.residual(flux, current)))
        scale = jnp.maximum(jnp.max(jnp.abs(flux)), 1.0e-30)
        history = fixed_point.FixedPointResult(
            state=jnp.asarray(flux),
            residual=residual / scale,
            trace=jnp.atleast_1d(residual / scale),
        )
        return self._receipt(jnp.asarray(flux), history)

    def integral_observation(self, flux) -> IntegralObservation:
        """Return the integral observations of one flux map.

        This is the differentiable moment map: it reads the topology, applies
        the declared source and integrates, with no conservation differencing
        in the way, so ``jacfwd`` through it costs one observation.
        """
        masks, topology = self.operator.read(flux)
        current_masks = self.operator.current_domain_masks(flux)
        cell_current = self.operator.cell_current_moments(flux).cell_current
        radial, vertical = poloidal_field(self.lattice, flux[: self.lattice.node_count])
        return observe_moments(
            self.operator.source,
            current_masks,
            jnp.asarray(self.lattice.node_radius),
            self.operator.area,
            cell_current,
            radial**2 + vertical**2,
            topology.flux_span,
        )

    def _receipt(
        self, flux: jax.Array, history: fixed_point.FixedPointResult
    ) -> ForwardEquilibrium:
        """Return the typed result of one converged or supplied flux map."""
        masks, topology = self.operator.read(flux)
        current_masks = self.operator.current_domain_masks(flux)
        cell_current = self.operator.cell_current_moments(flux).cell_current
        grid_flux = flux[: self.lattice.node_count]
        radius = jnp.asarray(self.lattice.node_radius)
        radial, vertical = poloidal_field(self.lattice, grid_flux)
        moments = observe_moments(
            self.operator.source,
            current_masks,
            radius,
            self.operator.area,
            cell_current,
            radial**2 + vertical**2,
            topology.flux_span,
        )
        conservation = conservation_ledger(
            self.lattice,
            grid_flux,
            self.operator.source,
            masks,
            topology.flux_span,
        )
        return ForwardEquilibrium(
            flux=flux,
            cell_current=cell_current,
            domains=masks,
            topology=topology,
            fixed_point=history,
            moments=moments,
            ledger=current_ledger(cell_current, current_masks),
            conservation=conservation,
            normalisation=absolute_normalisation_record(flux.dtype),
            rotation=self.operator.source.rotation_record(radius, masks),
            continuation=self.operator.source.continuation_ledger(flux.dtype),
            finite=FiniteCheck(
                flux=jnp.all(jnp.isfinite(flux)),
                cell_current=jnp.all(jnp.isfinite(cell_current)),
                moments=jnp.all(jnp.isfinite(moments.stack())),
                conservation=jnp.all(jnp.isfinite(jnp.stack([*conservation[:-1]]))),
            ),
        )

    def _host_history(self, trace, flux, current) -> fixed_point.FixedPointResult:
        """Return the shared fixed-point result of a host solve."""
        mapped = self.operator(flux, current)
        scale = jnp.maximum(jnp.max(jnp.abs(mapped)), 1.0e-30)
        return fixed_point.FixedPointResult(
            state=flux,
            residual=jnp.max(jnp.abs(mapped - flux)) / scale,
            trace=jnp.asarray(trace),
        )

    def _solve_host(
        self,
        initial_flux,
        current,
        *,
        evaluations: int | None = None,
        relaxation: float | None = None,
        tolerance: float = 1.0e-10,
        **options,
    ) -> ForwardEquilibrium:
        """Drive the map with a host relaxed fixed-point iteration.

        The eager counterpart of :func:`nova.equilibrium.fixed_point.picard`:
        the same relaxed step under host control flow, with an early exit
        once the relative residual passes ``tolerance``. Run to the same
        evaluation count with an unreachable tolerance it reproduces the
        traced ladder step for step while the domain labels hold; once the
        residual reaches the flux one boundary cell's current carries, the
        relaxed step cycles between labellings instead of contracting and
        the two routes separate at that floor.
        """
        if options:
            raise TypeError(
                f"unexpected host solve argument(s) {', '.join(sorted(options))}"
            )
        budget = self.evaluations if evaluations is None else int(evaluations)
        step = self.relaxation if relaxation is None else float(relaxation)
        mapped = self.flux_map(current)
        trace = np.full(budget, np.nan)
        state = np.asarray(initial_flux, dtype=np.float64)
        for index in range(budget):
            image = np.asarray(mapped(jnp.asarray(state)))
            residual = np.max(np.abs(image - state)) / max(
                np.max(np.abs(image)), 1.0e-30
            )
            trace[index] = residual
            state = state + step * (image - state)
            if residual < tolerance:
                break
        flux = jnp.asarray(state)
        return self._receipt(flux, self._host_history(trace, flux, current))

    def _solve_host_krylov(
        self, initial_flux, current, **options
    ) -> ForwardEquilibrium:
        """Drive the map with a host Jacobian-free Newton-Krylov root find.

        The Krylov step is globalised only by its own line search, so it
        moves freely between the branches of the free-boundary map and needs
        a seed already on the intended one.
        """
        mapped = self.flux_map(current)
        trace = np.full(self.evaluations, np.nan)
        recorded = 0

        def residual(psi):
            """Return the host free-boundary residual of a trial flux."""
            return np.asarray(mapped(jnp.asarray(psi))) - psi

        def record(psi, value):
            """Record the relative residual of one accepted host step."""
            nonlocal recorded
            if recorded < trace.size:
                total = np.max(np.abs(psi + value))
                trace[recorded] = np.max(np.abs(value)) / max(total, 1.0e-30)
                recorded += 1

        solution = scipy.optimize.newton_krylov(
            residual,
            np.asarray(initial_flux, dtype=np.float64),
            callback=record,
            **options,
        )
        flux = jnp.asarray(solution)
        return self._receipt(flux, self._host_history(trace, flux, current))

    def _solve_accelerated(
        self, route: str, initial_flux, current, **options
    ) -> ForwardEquilibrium:
        """Drive the map with the shared fixed-point ladder."""
        mapped = self.flux_map(current)
        if route == "newton_krylov":
            history = fixed_point.newton_krylov(
                mapped,
                initial_flux,
                **{"newton_steps": self.newton_steps, **options},
            )
        else:
            scheme = fixed_point.picard if route == "picard" else fixed_point.anderson
            history = scheme(
                mapped,
                initial_flux,
                **{
                    "evaluations": self.evaluations,
                    "relaxation": self.relaxation,
                    **options,
                },
            )
        return self._receipt(history.state, history)

    def solve(
        self,
        initial_flux,
        *,
        route: SolveRoute = "newton_krylov",
        current=None,
        enforce: Sequence[str] = (),
        **options,
    ) -> ForwardEquilibrium:
        """Return the equilibrium the prescribed source supports.

        The default route is the root find because the map does not contract
        at the states this class is usually asked for. An elongated column
        held at fixed conductor currents is axisymmetrically unstable, and the
        write-then-read cycle carries that as a Jacobian eigenvalue outside
        the unit circle — 1.25 to 1.40 measured on a diverted reference case.
        A step relaxed by ``beta`` scales that mode by
        ``(1 - beta) + beta * lambda``, which exceeds one for every ``beta``
        in ``(0, 1]`` once ``lambda`` does, so no damping rescues a relaxed
        route; ``newton_krylov`` solves ``(I - J) s = f`` instead and is
        indifferent to the sign of that eigenvalue. ``picard`` and
        ``anderson`` stay available by name — they are the cheaper choice on a
        map that does contract, which is the limited, low-elongation case, and
        they are the only routes that expose the map iteration itself to a
        caller measuring how it behaves.

        ``enforce`` names integral observations the caller wants closed by
        the solve. The declared source has to carry one scalar degree of
        freedom per enforced moment; an absolute source carries none, so any
        request fails here, before a single profile value is read.
        """
        reject_unsupported_enforcement(enforce, self.source.closure_degrees)
        if route == "host":
            return self._solve_host(initial_flux, current, **options)
        if route == "host_krylov":
            return self._solve_host_krylov(initial_flux, current, **options)
        if route not in _ACCELERATED:
            raise ValueError(
                f"unknown solve route {route!r}; available: "
                f"{', '.join((*_HOST, *_ACCELERATED))}"
            )
        return self._solve_accelerated(route, initial_flux, current, **options)

    def solve_batch(
        self,
        initial_flux,
        *,
        route: SolveRoute = "newton_krylov",
        current=None,
        enforce: Sequence[str] = (),
        **options,
    ) -> ForwardEquilibrium:
        """Return :meth:`solve` mapped over a leading ensemble axis.

        The trial flux always carries the ensemble axis; the conductor
        currents carry it only when a batched array is supplied, so one
        machine state can be shared across an ensemble of seeds.
        """
        reject_unsupported_enforcement(enforce, self.source.closure_degrees)
        if route not in _ACCELERATED:
            raise ValueError(
                "a batched ensemble solve needs a fixed-shape route; "
                f"available: {', '.join(_ACCELERATED)}"
            )
        current_axis = None if current is None or jnp.ndim(current) == 1 else 0
        return jax.vmap(
            lambda flux, conductor: self._solve_accelerated(
                route, flux, conductor, **options
            ),
            in_axes=(0, current_axis),
        )(initial_flux, current)

    def moment_residual(self, flux, targets: MomentTargets) -> jax.Array:
        """Return the scale-normalised integral-observation residuals."""
        return moment_residual(self.integral_observation(flux), targets)

    def moment_jacobian(self, flux, targets: MomentTargets) -> jax.Array:
        """Return the derivative of the moment residuals with respect to flux.

        The map is differentiated at a converged flux, so the Jacobian is the
        observation operator a conditioning or reconstruction caller needs;
        this class never applies it to a profile itself.
        """
        return jax.jacfwd(lambda state: self.moment_residual(state, targets))(
            jnp.asarray(flux)
        )
