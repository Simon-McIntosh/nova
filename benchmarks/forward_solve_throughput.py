"""Latency and batched throughput of the free-boundary forward flux solve.

The solve measured here is the one the reference demonstration drives: an
absolute prescribed-source ``ForwardFluxOperator`` on the production hexagonal
plasma mesh of a stored reactor equilibrium, closed by the fixed-budget
Newton-Krylov ladder. Nothing about the numerics is changed to time it. The
same couplings, the same source, the same seed and the same step budget are
used, and every reported timing is cross-checked against the stored solve's
own converged flux so a fast run that has stopped solving the problem cannot
pass unnoticed.

Two numbers answer different questions and the driver separates them.

single-solve latency
    One equilibrium, start to finish, after compilation. The map evaluation is
    dominated by a coupling matrix-vector product over the plasma cells, whose
    arithmetic intensity is about one flop per byte read: it is bound by memory
    bandwidth, not by arithmetic, and a single solve therefore leaves most of a
    large accelerator idle no matter how the kernel is written.

batched throughput
    ``jit(vmap)`` over an ensemble of independent solves that differ in their
    conductor currents. Batching turns that matrix-vector product into a
    matrix-matrix product, which is the same memory traffic amortised over the
    whole ensemble, so throughput per member keeps improving until the product
    becomes arithmetic-bound. The ratio of the saturated ensemble rate to the
    single-solve rate is the headroom any parallel-in-time or coupled-waveform
    scheme can harvest without touching the solver, and it is reported as such.

The stages run in separate processes on purpose. A compile time is only
honest if the process has never seen the executable, so ``--compile-cache``
selects an explicit persistent-cache directory or turns the cache off, and a
cold measurement is a fresh process against an empty (or disabled) cache while
a warm one is a fresh process against a populated one. Nothing in the solve
path enables that cache by default, which is one of the reported findings
rather than an accident of the harness.

Usage::

    uv run python benchmarks/forward_solve_throughput.py prepare \\
      --cells -1500 --bundle /path/to/forward_solve_1587.npz

    uv run python benchmarks/forward_solve_throughput.py measure \\
      --bundle /path/to/forward_solve_1587.npz --platform gpu \\
      --compile-cache off --stage latency \\
      --output /path/to/forward_solve_1587_cold.json

    uv run python benchmarks/forward_solve_throughput.py figure \\
      --result /path/to/forward_solve_1587.json \\
      --output docs/figures/flux-function-forward-equilibrium/h200-throughput.png
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

#: Step budget of the reference ladder. Each Newton step linearises the map
#: once and solves ``(I - J) s = f`` in a fixed-shape Krylov space, costing
#: ``2 + gmres_iterations`` map evaluations, so the whole solve evaluates the
#: free-boundary map a fixed and known number of times. That count is what
#: turns a measured single-map cost into a budget the solve can be read
#: against, and it is why no timing here depends on a convergence test.
NEWTON_STEPS = 10
KRYLOV_ITERATIONS = 30
WARMUP_SWEEPS = 0

#: Conductor-current spread across an ensemble member. The batch is a proxy
#: for a coupled-waveform or parallel-in-time window, where neighbouring
#: solves differ by a small conductor increment rather than by a new machine,
#: so the members share every coupling matrix and differ only in their drive.
#: The spread is kept small because the absolute-source map has a vacuum twin
#: and a large excursion would change which branch a member converges to,
#: turning a throughput measurement into a basin measurement.
CURRENT_SPREAD = 2.0e-3

#: Repeats a reported latency is the median of, and untimed calls made first.
#: The median rather than the mean because a single host-side hiccup on a
#: shared login node is not a property of the kernel.
LATENCY_REPEATS = 7
LATENCY_WARMUP = 2
#: Repeats for a batched point. Fewer, because each already carries N solves.
BATCH_REPEATS = 3

#: How far the benchmark solve may sit from the flux map the bundle stored,
#: relative to the flux span. The two run the same ladder on the same seed, so
#: the difference should be round-off; this only has to catch a rebuild that
#: has quietly stopped solving the same problem.
PARITY_TOLERANCE = 1.0e-9

#: Square matrix orders the device's own arithmetic ceiling is read at, and
#: the batch widths the same product is timed at. The coupling product the
#: solve spends its time in has exactly this shape — a dense square operator
#: applied to one column per ensemble member — so measuring it directly gives
#: the roofline the solve is read against without modelling anything.
KERNEL_ORDERS = (1587, 4190)
KERNEL_WIDTHS = (1, 4, 16, 64, 256)


# --------------------------------------------------------------------------
# the bundle: one machine, its couplings, its source and its stored solve
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SolveBundle:
    """Everything one forward solve needs, detached from its construction.

    The couplings are assembled once by the reference construction and stored,
    so a measurement process rebuilds the operator through the public API
    alone. That keeps a timing run independent of the pulse store and of the
    assembly path, and it is what allows a cold-compile measurement to start a
    fresh process cheaply.
    """

    node: np.ndarray = field(repr=False)
    area: np.ndarray = field(repr=False)
    hexagon: np.ndarray = field(repr=False)
    stencil: np.ndarray = field(repr=False)
    wall_node: np.ndarray = field(repr=False)
    source_to_grid: np.ndarray = field(repr=False)
    plasma_to_grid: np.ndarray = field(repr=False)
    source_to_wall: np.ndarray = field(repr=False)
    plasma_to_wall: np.ndarray = field(repr=False)
    coil_current: np.ndarray = field(repr=False)
    psi_norm: np.ndarray = field(repr=False)
    p_prime: np.ndarray = field(repr=False)
    ff_prime: np.ndarray = field(repr=False)
    boundary_pressure: float
    boundary_field_function: float
    seed: np.ndarray = field(repr=False)
    reference_flux: np.ndarray = field(repr=False)
    reference_residual: float
    build_seconds: float
    cells: int

    @property
    def cell_number(self) -> int:
        """Return the number of plasma cells the mesh carries."""
        return len(self.node)

    @property
    def node_number(self) -> int:
        """Return the length of the concatenated grid and wall flux vector."""
        return len(self.seed)

    @property
    def flux_span(self) -> float:
        """Return the flux range the parity check is read against [Wb]."""
        return float(np.ptp(self.reference_flux))

    def save(self, path: Path) -> None:
        """Write the bundle to one compressed archive."""
        np.savez_compressed(
            path,
            node=self.node,
            area=self.area,
            hexagon=self.hexagon,
            stencil=self.stencil,
            wall_node=self.wall_node,
            source_to_grid=self.source_to_grid,
            plasma_to_grid=self.plasma_to_grid,
            source_to_wall=self.source_to_wall,
            plasma_to_wall=self.plasma_to_wall,
            coil_current=self.coil_current,
            psi_norm=self.psi_norm,
            p_prime=self.p_prime,
            ff_prime=self.ff_prime,
            boundary_pressure=self.boundary_pressure,
            boundary_field_function=self.boundary_field_function,
            seed=self.seed,
            reference_flux=self.reference_flux,
            reference_residual=self.reference_residual,
            build_seconds=self.build_seconds,
            cells=self.cells,
        )

    @classmethod
    def load(cls, path: Path) -> SolveBundle:
        """Return the bundle one archive holds."""
        stored = np.load(path, allow_pickle=False)
        return cls(
            node=stored["node"],
            area=stored["area"],
            hexagon=stored["hexagon"],
            stencil=stored["stencil"],
            wall_node=stored["wall_node"],
            source_to_grid=stored["source_to_grid"],
            plasma_to_grid=stored["plasma_to_grid"],
            source_to_wall=stored["source_to_wall"],
            plasma_to_wall=stored["plasma_to_wall"],
            coil_current=stored["coil_current"],
            psi_norm=stored["psi_norm"],
            p_prime=stored["p_prime"],
            ff_prime=stored["ff_prime"],
            boundary_pressure=float(stored["boundary_pressure"]),
            boundary_field_function=float(stored["boundary_field_function"]),
            seed=stored["seed"],
            reference_flux=stored["reference_flux"],
            reference_residual=float(stored["reference_residual"]),
            build_seconds=float(stored["build_seconds"]),
            cells=int(stored["cells"]),
        )


def prepare_bundle(cells: int) -> SolveBundle:
    """Assemble one machine at a cell count and solve it once.

    This is the only entry point that reaches the pulse store and the coupling
    assembly, and it is deliberately separated from every timing stage: the
    assembly is a host-side build whose cost is reported once, while the solve
    is the device-side quantity the rest of the driver measures.
    """
    from nova.jax.config import configure_dtypes

    configure_dtypes()

    from tests import test_equilibrium_forward_reference as demonstration

    case = demonstration.reference_case()
    if isinstance(case, str):
        raise RuntimeError(f"the stored reference is unreachable: {case}")

    start = time.perf_counter()
    machine = demonstration.build_machine(case, cells)
    build_seconds = time.perf_counter() - start

    solved = demonstration.solve(case, machine)
    return SolveBundle(
        node=machine.node,
        area=machine.area,
        hexagon=machine.hexagon,
        stencil=machine.stencil,
        wall_node=machine.wall_node,
        source_to_grid=machine.source_to_grid,
        plasma_to_grid=machine.plasma_to_grid,
        source_to_wall=machine.source_to_wall,
        plasma_to_wall=machine.plasma_to_wall,
        coil_current=np.asarray(case.coil_current, dtype=float),
        psi_norm=np.asarray(case.psi_norm, dtype=float),
        p_prime=np.asarray(case.p_prime, dtype=float),
        ff_prime=np.asarray(case.ff_prime, dtype=float),
        boundary_pressure=float(case.pressure[-1]),
        boundary_field_function=float(case.field_function[-1]),
        seed=np.asarray(demonstration.seed_flux(case, machine), dtype=float),
        reference_flux=np.asarray(solved.flux, dtype=float),
        reference_residual=float(solved.fixed_point.residual),
        build_seconds=build_seconds,
        cells=len(machine.node),
    )


# --------------------------------------------------------------------------
# the operator, rebuilt through the public API alone
# --------------------------------------------------------------------------
def build_operator(bundle: SolveBundle):
    """Return the free-boundary map the bundle's couplings carry.

    The polarity is negative because the stored flux sense puts the magnetic
    axis at a minimum of the total poloidal flux, and the null search is given
    only the rings whose centre and all six neighbours are uncut hexagons —
    both are properties of the reference construction, restated here rather
    than imported so a measurement process needs no test module.
    """
    import jax.numpy as jnp

    from nova.biot.null import Null1D, Null2D
    from nova.biot.target import FluxTarget
    from nova.equilibrium.forward_operator import ForwardFluxOperator
    from nova.equilibrium.source import DomainProfile, ForwardSource

    def flux_function(sample: np.ndarray) -> Callable:
        """Return a traceable absolute flux function of normalised flux."""
        grid = jnp.asarray(bundle.psi_norm)
        value = jnp.asarray(sample)

        def gradient(argument):
            """Return the tabulated gradient at one normalised flux."""
            return jnp.interp(jnp.asarray(argument), grid, value)

        return gradient

    stencil = np.asarray(bundle.stencil)
    interior = stencil[np.asarray(bundle.hexagon)[stencil].all(axis=1)]
    return ForwardFluxOperator(
        grid=FluxTarget(
            jnp.asarray(bundle.source_to_grid),
            jnp.asarray(bundle.plasma_to_grid),
            Null2D.from_coordinates(bundle.node, interior, maxsize=5),
        ),
        wall=FluxTarget(
            jnp.asarray(bundle.source_to_wall),
            jnp.asarray(bundle.plasma_to_wall),
            Null1D(jnp.asarray(bundle.wall_node, dtype=jnp.float64)),
        ),
        source=ForwardSource(
            core=DomainProfile(
                p_prime=flux_function(bundle.p_prime),
                ff_prime=flux_function(bundle.ff_prime),
            ),
            boundary_pressure=bundle.boundary_pressure,
            boundary_field_function=bundle.boundary_field_function,
        ),
        external_current=jnp.asarray(bundle.coil_current),
        area=jnp.asarray(bundle.area),
        polarity=-1,
    )


def build_receipt_mesh(bundle: SolveBundle):
    """Return the mesh the published receipts are differentiated on.

    Every ring the least-squares quadratic can carry is admitted, which is a
    weaker selection than the null search uses: a derivative needs only that
    the cluster determine a quadratic, while a stationary-point search on a
    clipped cell's displaced centroid can report an extremum the map does not
    have. The limit is the one the mesh class enforces on itself.
    """
    from nova.equilibrium.stencil_mesh import (
        RING_CONDITION_LIMIT,
        StencilMesh,
        ring_condition,
    )

    stencil = np.asarray(bundle.stencil)
    condition = ring_condition(bundle.node, stencil)
    return StencilMesh(
        coordinate=bundle.node,
        stencil=stencil[condition < RING_CONDITION_LIMIT],
        area=bundle.area,
    )


def solve_callable(operator):
    """Return the one-argument-pair solve the whole driver times.

    Conductor current is an argument rather than a closure so an ensemble maps
    over it, which is the batch axis a coupled-waveform window would carry.
    The seed is an argument for the same reason.
    """
    from nova.equilibrium import fixed_point

    def solve_one(current, seed):
        """Return the converged flux one conductor state supports."""
        return fixed_point.newton_krylov(
            operator.flux_map(current),
            seed,
            newton_steps=NEWTON_STEPS,
            gmres_iterations=KRYLOV_ITERATIONS,
            warmup=WARMUP_SWEEPS,
        )

    return solve_one


def map_evaluations() -> int:
    """Return how many times one solve evaluates the free-boundary map."""
    return WARMUP_SWEEPS + NEWTON_STEPS * (2 + KRYLOV_ITERATIONS)


def ensemble_current(bundle: SolveBundle, members: int) -> np.ndarray:
    """Return one conductor state per ensemble member.

    The members are spread symmetrically about the stored currents so the
    single-member case is the stored state exactly and a wider ensemble stays
    inside the confined branch.
    """
    if members == 1:
        return bundle.coil_current[None, :]
    offset = np.linspace(-CURRENT_SPREAD, CURRENT_SPREAD, members)
    return bundle.coil_current[None, :] * (1.0 + offset[:, None])


# --------------------------------------------------------------------------
# timing primitives
# --------------------------------------------------------------------------
def time_call(call: Callable, repeats: int, warmup: int) -> dict[str, float]:
    """Return the wall time of a device call, blocked to completion.

    JAX dispatch is asynchronous, so a timing that does not wait on the result
    measures the host's enqueue rather than the device's work. Every entry
    here blocks on the returned buffers.
    """
    import jax

    for _ in range(warmup):
        jax.block_until_ready(call())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(call())
        samples.append(time.perf_counter() - start)
    ordered = sorted(samples)
    return {
        "median": ordered[len(ordered) // 2],
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "repeats": repeats,
    }


def device_report() -> dict[str, Any]:
    """Return what the process is running on."""
    import jax
    import jaxlib

    device = jax.devices()[0]
    report = {
        "platform": device.platform,
        "kind": device.device_kind,
        "count": len(jax.devices()),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "host": platform.node(),
    }
    try:
        stats = device.memory_stats() or {}
        report["memory_limit_bytes"] = int(stats.get("bytes_limit", 0))
    except Exception:  # noqa: BLE001 - absent on the host backend
        report["memory_limit_bytes"] = 0
    return report


def peak_bytes() -> int:
    """Return the device's peak allocation, or zero where none is reported."""
    import jax

    try:
        stats = jax.devices()[0].memory_stats() or {}
    except Exception:  # noqa: BLE001 - absent on the host backend
        return 0
    return int(stats.get("peak_bytes_in_use", 0))


# --------------------------------------------------------------------------
# stage: single-solve compile and latency
# --------------------------------------------------------------------------
def measure_latency(bundle: SolveBundle, operator) -> dict[str, Any]:
    """Return compile time, per-solve latency and the parity that qualifies it.

    Compilation is separated from execution by lowering and compiling
    explicitly, so the reported compile time is the executable's and not a
    first call's compile plus its run.
    """
    import jax
    import jax.numpy as jnp

    solve_one = solve_callable(operator)
    current = jnp.asarray(bundle.coil_current)
    seed = jnp.asarray(bundle.seed)
    jitted = jax.jit(solve_one)

    lowered = jitted.lower(current, seed)
    start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - start

    latency = time_call(
        lambda: compiled(current, seed), LATENCY_REPEATS, LATENCY_WARMUP
    )
    result = compiled(current, seed)
    flux = np.asarray(jax.block_until_ready(result.state))
    deviation = float(np.max(np.abs(flux - bundle.reference_flux)))

    transfer_start = time.perf_counter()
    for _ in range(LATENCY_REPEATS):
        jax.block_until_ready(jax.device_put(bundle.seed))
    transfer_in = (time.perf_counter() - transfer_start) / LATENCY_REPEATS
    fetch_start = time.perf_counter()
    for _ in range(LATENCY_REPEATS):
        np.asarray(result.state)
    transfer_out = (time.perf_counter() - fetch_start) / LATENCY_REPEATS

    return {
        "compile_seconds": compile_seconds,
        "latency_seconds": latency,
        "solves_per_second": 1.0 / latency["median"],
        "map_evaluations": map_evaluations(),
        "seconds_per_map_evaluation": latency["median"] / map_evaluations(),
        "residual": float(result.residual),
        "reference_residual": bundle.reference_residual,
        "flux_deviation": deviation,
        "flux_deviation_relative": deviation / bundle.flux_span,
        "parity": deviation / bundle.flux_span < PARITY_TOLERANCE,
        "seed_transfer_seconds": transfer_in,
        "result_fetch_seconds": transfer_out,
        "peak_bytes": peak_bytes(),
    }


# --------------------------------------------------------------------------
# stage: batched throughput
# --------------------------------------------------------------------------
def measure_batch(bundle: SolveBundle, operator, widths: tuple[int, ...]):
    """Return solves per second against ensemble width.

    One outer ``jit`` around a ``vmap`` over conductor states: the couplings
    are captured constants shared by every member, so the batch axis costs
    only the state vectors and the Krylov space, and the coupling product
    widens from a matrix-vector to a matrix-matrix product.
    """
    import jax
    import jax.numpy as jnp

    solve_one = solve_callable(operator)
    batched = jax.jit(jax.vmap(solve_one))
    seed = jnp.asarray(bundle.seed)
    points = []
    for members in widths:
        current = jnp.asarray(ensemble_current(bundle, members))
        seeds = jnp.broadcast_to(seed, (members, seed.shape[0]))
        try:
            lowered = batched.lower(current, seeds)
            start = time.perf_counter()
            compiled = lowered.compile()
            compile_seconds = time.perf_counter() - start
            before = peak_bytes()
            timing = time_call(
                lambda: compiled(current, seeds),  # noqa: B023 - called at once
                BATCH_REPEATS,
                1,
            )
            after = peak_bytes()
            result = compiled(current, seeds)
            residual = float(jnp.max(jnp.abs(result.residual)))
            flux = np.asarray(jax.block_until_ready(result.state))
            spread = float(
                np.max(np.abs(flux[0] - bundle.reference_flux)) / bundle.flux_span
            )
        except Exception as error:  # noqa: BLE001 - an exhausted device is a datum
            points.append(
                {
                    "members": members,
                    "failed": f"{type(error).__name__}: {str(error)[:200]}",
                }
            )
            break
        points.append(
            {
                "members": members,
                "compile_seconds": compile_seconds,
                "wall_seconds": timing["median"],
                "solves_per_second": members / timing["median"],
                "seconds_per_member": timing["median"] / members,
                "worst_residual": residual,
                "first_member_deviation_relative": spread,
                "peak_bytes": after,
                "peak_bytes_increment": max(after - before, 0),
                "bytes_per_member": max(after - before, 0) / members,
            }
        )
    return points


# --------------------------------------------------------------------------
# stage: where the time goes
# --------------------------------------------------------------------------
#: Chain lengths a marginal per-application cost is differenced across. Timing
#: one call of a small kernel measures its launch, not its arithmetic, and on
#: an accelerator the launch can be the larger number by an order of magnitude.
#: Applying the same step a fixed number of times inside one compiled loop and
#: differencing two lengths cancels every fixed cost — the compile, the
#: transfer, the launch, the result fetch — and leaves the marginal cost of one
#: application, which is the quantity a per-evaluation budget needs.
CHAIN_SHORT = 32
CHAIN_LONG = 96
#: Increment scaling that holds a map chain on the equilibrium it starts from.
CHAIN_DAMPING = 1.0e-6
#: Runs each chain endpoint is timed over. More than a latency point needs,
#: because a difference of two measurements carries the error of both.
CHAIN_REPEATS = 9


def chained(step: Callable, count: int) -> Callable:
    """Return a compiled loop that applies one self-map ``count`` times."""
    import jax

    def run(state):
        """Return the state after repeated application."""
        return jax.lax.fori_loop(0, count, lambda _, carry: step(carry), state)

    return jax.jit(run)


def marginal_seconds(step: Callable, state) -> dict[str, float]:
    """Return the cost of one application of a self-map, overhead differenced.

    Both chain lengths pay exactly the same fixed cost, so their difference is
    the arithmetic of the extra applications and nothing else.

    The endpoints are the fastest observed run rather than the median. A
    difference of two noisy medians is far noisier than either — interference
    from anything else on the machine only ever makes a run slower, so it
    biases the endpoints in the same direction but by different amounts, and
    on a shared host that alone was enough to drive a difference negative.
    The fastest run of each is the one least contaminated, and differencing
    two of those is the stable estimator.

    A non-positive slope survives that and still means the measurement is not
    usable — the loop was folded away, or the interference was larger than the
    work — so it is flagged rather than propagated into an attribution that
    would silently turn into nonsense.
    """
    short = chained(step, CHAIN_SHORT)
    long_chain = chained(step, CHAIN_LONG)
    short_seconds = time_call(lambda: short(state), CHAIN_REPEATS, 2)["minimum"]
    long_seconds = time_call(lambda: long_chain(state), CHAIN_REPEATS, 2)["minimum"]
    slope = (long_seconds - short_seconds) / (CHAIN_LONG - CHAIN_SHORT)
    return {
        "marginal": slope,
        "usable": slope > 0.0,
        "short_seconds": short_seconds,
        "long_seconds": long_seconds,
    }


def measure_breakdown(bundle: SolveBundle, operator, saturation: int):
    """Return where one map evaluation and one solve spend their time.

    The attribution is built from nested self-maps rather than from isolated
    calls. Each chain applies one stage repeatedly inside a single compiled
    loop, so differencing two chain lengths gives the marginal cost of that
    stage with every fixed cost removed; nesting them — topology alone, then
    topology and source, then the whole map — makes each layer's cost a
    difference of two measurements rather than a separate timing that would
    carry its own launch.

    Every chain is a genuine self-map on the state it iterates, and each layer
    feeds a vanishing multiple of its own result back into the state so that
    nothing it computes can be eliminated as dead. The multiples are far below
    the last bit of the flux, so the chain neither changes the state it starts
    from nor leaves the converged equilibrium it is measured at.
    """
    import jax
    import jax.numpy as jnp

    flux = jnp.asarray(bundle.reference_flux)
    current = jnp.asarray(bundle.coil_current)
    radius = operator.radius
    area = operator.area
    masks = operator.read(flux)[0]
    cell_current = operator.source.cell_current(radius, area, masks)

    def topology_step(psi):
        """Read the topology and feed a vanishing trace of it back."""
        domain, state = operator.read(psi)
        live = state.axis_flux + jnp.sum(domain.core.astype(psi.dtype))
        return psi + 1.0e-30 * live

    def source_step(psi):
        """Read the topology, evaluate the source, keep both live."""
        domain, state = operator.read(psi)
        cell = operator.source.cell_current(radius, area, domain)
        return psi + 1.0e-30 * (state.axis_flux + jnp.sum(cell))

    def coupling_step(cell):
        """Apply both coupling products and renormalise to stay bounded."""
        grid_flux = operator.grid.internal(cell)
        wall_flux = operator.wall.internal(cell)
        scale = jnp.max(jnp.abs(grid_flux)) + 1.0e-300 * jnp.sum(jnp.abs(wall_flux))
        return grid_flux / scale

    def map_step(psi):
        """Apply the free-boundary map, held against its own fixed point.

        Iterating the bare map would be wrong here for a physical reason: this
        map does not contract — its dominant eigenvalue exceeds one — so a
        long chain would walk off the equilibrium and eventually onto the
        vacuum branch, and the arithmetic would stop being the arithmetic of
        the case under measurement. Scaling the increment by a small factor
        pins the chain to the converged state it starts from while still
        evaluating the map exactly once per application, which is the only
        property the timing depends on.
        """
        return flux + CHAIN_DAMPING * (operator(psi, current) - psi)

    marginal = {
        "topology_read": marginal_seconds(topology_step, flux),
        "topology_and_source": marginal_seconds(source_step, flux),
        "coupling_product": marginal_seconds(coupling_step, cell_current),
        "map_evaluation": marginal_seconds(map_step, flux),
    }
    single = {name: entry["marginal"] for name, entry in marginal.items()}
    # A layer's own cost is the difference between two nested chains, so it is
    # only meaningful when both chains measured something. Deriving from an
    # unusable endpoint would publish a difference of two artefacts.
    reliable = all(entry["usable"] for entry in marginal.values())
    if reliable:
        single["source_evaluation"] = (
            single["topology_and_source"] - single["topology_read"]
        )
        single["coupling_in_map"] = (
            single["map_evaluation"] - single["topology_and_source"]
        )

    published = measure_published_route(bundle, operator)

    batched = {}
    if saturation > 1:
        fluxes = jnp.broadcast_to(flux, (saturation, flux.shape[0]))
        cells = jnp.broadcast_to(cell_current, (saturation, cell_current.shape[0]))
        wide = {
            "topology_read": (jax.vmap(topology_step), fluxes),
            "topology_and_source": (jax.vmap(source_step), fluxes),
            "coupling_product": (jax.vmap(coupling_step), cells),
            "map_evaluation": (jax.vmap(map_step), fluxes),
        }
        for name, (step, state) in wide.items():
            entry = marginal_seconds(step, state)
            batched[name] = {
                "marginal": entry["marginal"],
                "marginal_per_member": entry["marginal"] / saturation,
            }

    return {
        "single": single,
        "chains": marginal,
        "reliable": reliable,
        "map_budget_seconds": single["map_evaluation"] * map_evaluations(),
        "published": published,
        "batched": batched,
        "batched_members": saturation if saturation > 1 else 0,
    }


def measure_published_route(bundle: SolveBundle, operator):
    """Return the cost of the receipt-carrying entry point, two ways.

    ``ForwardProfile.solve`` is not compiled as a unit: it drives the ladder
    and then evaluates every receipt through ordinary eager calls, so a caller
    invoking it in a loop pays to re-trace the ladder and to dispatch each
    receipt separately on every call. Timing it as it ships and again inside
    one ``jit`` separates the two costs — what the receipts compute, and what
    not compiling the entry point costs on top of that.
    """
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.forward import ForwardProfile

    seed = jnp.asarray(bundle.seed)
    result: dict[str, Any] = {}
    try:
        profile = ForwardProfile(
            operator=operator,
            lattice=build_receipt_mesh(bundle),
            newton_steps=NEWTON_STEPS,
        )
    except Exception as error:  # noqa: BLE001 - a construction failure is a datum
        return {"failed": f"{type(error).__name__}: {str(error)[:200]}"}

    def call(flux):
        """Return the published solve on one seed."""
        return profile.solve(
            flux,
            route="newton_krylov",
            gmres_iterations=KRYLOV_ITERATIONS,
            warmup=WARMUP_SWEEPS,
        )

    try:
        result["eager_seconds"] = time_call(lambda: call(seed), 3, 1)["median"]
    except Exception as error:  # noqa: BLE001
        result["eager_seconds"] = f"{type(error).__name__}: {str(error)[:200]}"
    try:
        compiled = jax.jit(call)
        start = time.perf_counter()
        compiled = compiled.lower(seed).compile()
        result["compile_seconds"] = time.perf_counter() - start
        result["compiled_seconds"] = time_call(
            lambda: compiled(seed), BATCH_REPEATS, 1
        )["median"]
    except Exception as error:  # noqa: BLE001 - a receipt that will not trace
        result["compiled_seconds"] = f"{type(error).__name__}: {str(error)[:200]}"
    return result


# --------------------------------------------------------------------------
# stage: the device's own ceiling
# --------------------------------------------------------------------------
def measure_kernel_ceiling(orders: tuple[int, ...], widths: tuple[int, ...]):
    """Return the achieved rate of the product the solve spends its time in.

    A dense square operator applied to one column per ensemble member is the
    coupling product itself, so timing it at both precisions and across the
    batch axis gives the arithmetic ceiling and the bandwidth floor the solve
    is read against without inferring either from a specification sheet.
    """
    import jax
    import jax.numpy as jnp

    rows = []
    for order in orders:
        for dtype, label in ((jnp.float64, "float64"), (jnp.float32, "float32")):
            operator_matrix = jnp.asarray(
                np.random.default_rng(0).standard_normal((order, order)), dtype=dtype
            )
            for width in widths:
                columns = jnp.asarray(
                    np.random.default_rng(1).standard_normal((order, width)),
                    dtype=dtype,
                )
                product = jax.jit(lambda a, b: a @ b)
                try:
                    timing = time_call(
                        lambda p=product, a=operator_matrix, b=columns: p(a, b),
                        LATENCY_REPEATS,
                        LATENCY_WARMUP,
                    )
                except Exception as error:  # noqa: BLE001
                    rows.append(
                        {
                            "order": order,
                            "dtype": label,
                            "width": width,
                            "failed": f"{type(error).__name__}",
                        }
                    )
                    continue
                seconds = timing["median"]
                flops = 2.0 * order * order * width
                itemsize = 8 if label == "float64" else 4
                bytes_moved = itemsize * (order * order + order * width * 2)
                rows.append(
                    {
                        "order": order,
                        "dtype": label,
                        "width": width,
                        "seconds": seconds,
                        "gflop_per_second": flops / seconds / 1.0e9,
                        "gbyte_per_second": bytes_moved / seconds / 1.0e9,
                    }
                )
    return rows


# --------------------------------------------------------------------------
# assembly of the measurement record
# --------------------------------------------------------------------------
def run_measurement(arguments) -> dict[str, Any]:
    """Return the record one measurement process produces."""
    from nova.jax.config import configure_dtypes

    configure_dtypes()

    cache = configure_compilation_cache(arguments.compile_cache)
    bundle = SolveBundle.load(Path(arguments.bundle))
    record: dict[str, Any] = {
        "bundle": str(Path(arguments.bundle).name),
        "cells": bundle.cells,
        "nodes": bundle.node_number,
        "coils": int(bundle.coil_current.shape[0]),
        "build_seconds": bundle.build_seconds,
        "newton_steps": NEWTON_STEPS,
        "krylov_iterations": KRYLOV_ITERATIONS,
        "map_evaluations": map_evaluations(),
        "label": arguments.label,
        "compile_cache": str(cache) if cache else "off",
        "device": device_report(),
        "stage": arguments.stage,
    }

    stages = (
        ("latency", "batch", "breakdown", "ceiling")
        if arguments.stage == "all"
        else (arguments.stage,)
    )
    operator = None
    if {"latency", "batch", "breakdown"} & set(stages):
        start = time.perf_counter()
        operator = build_operator(bundle)
        record["operator_seconds"] = time.perf_counter() - start

    if "latency" in stages:
        record["latency"] = measure_latency(bundle, operator)
        print(
            "latency %.4f s" % record["latency"]["latency_seconds"]["median"],
            flush=True,
        )
    if "batch" in stages:
        widths = tuple(int(item) for item in arguments.batch.split(","))
        record["batch"] = measure_batch(bundle, operator, widths)
        for point in record["batch"]:
            if "failed" in point:
                print(
                    "batch %d FAILED %s" % (point["members"], point["failed"]),
                    flush=True,
                )
            else:
                print(
                    "batch %d %.2f solves/s"
                    % (point["members"], point["solves_per_second"])
                )
    if "breakdown" in stages:
        saturation = arguments.saturation
        if saturation <= 0 and record.get("batch"):
            usable = [point for point in record["batch"] if "failed" not in point]
            saturation = max(
                usable, key=lambda point: point["solves_per_second"], default={}
            ).get("members", 0)
        record["breakdown"] = measure_breakdown(bundle, operator, saturation)
        print(
            "breakdown map %.6f s" % record["breakdown"]["single"]["map_evaluation"],
            flush=True,
        )
    if "ceiling" in stages:
        record["ceiling"] = measure_kernel_ceiling(KERNEL_ORDERS, KERNEL_WIDTHS)
        print("ceiling rows %d" % len(record["ceiling"]), flush=True)

    record["headroom"] = summarise_headroom(record)
    return record


def summarise_headroom(record: dict[str, Any]) -> dict[str, Any]:
    """Return the named gaps a single solve leaves on the device.

    The ratio the batch buys is stated against the single-solve rate measured
    in the same process on the same executable shape, so it is a like-for-like
    comparison rather than a comparison against a published figure.
    """
    summary: dict[str, Any] = {}
    latency = record.get("latency")
    batch = [point for point in record.get("batch", []) if "failed" not in point]
    if latency and batch:
        single = latency["solves_per_second"]
        best = max(batch, key=lambda point: point["solves_per_second"])
        summary["single_solve_rate"] = single
        summary["saturated_rate"] = best["solves_per_second"]
        summary["saturation_members"] = best["members"]
        summary["batch_headroom_ratio"] = best["solves_per_second"] / single
        summary["device_idle_fraction_single"] = (
            1.0 - single / (best["solves_per_second"])
        )
        summary["bytes_per_member"] = best["bytes_per_member"]
    if latency:
        total = latency["latency_seconds"]["median"]
        summary["transfer_fraction"] = (
            latency["seed_transfer_seconds"] + latency["result_fetch_seconds"]
        ) / total
        summary["build_over_solve"] = record["build_seconds"] / total
    ceiling = record.get("ceiling")
    if ceiling:
        usable = [row for row in ceiling if "failed" not in row]
        for width in (1, max(KERNEL_WIDTHS)):
            pair = {
                row["dtype"]: row
                for row in usable
                if row["width"] == width and row["order"] == KERNEL_ORDERS[0]
            }
            if {"float64", "float32"} <= set(pair):
                summary[f"single_over_double_width_{width}"] = (
                    pair["float64"]["seconds"] / pair["float32"]["seconds"]
                )
        wide = [
            row
            for row in usable
            if row["dtype"] == "float64" and row["order"] == KERNEL_ORDERS[0]
        ]
        if wide:
            narrow = [row for row in wide if row["width"] == 1]
            broad = max(wide, key=lambda row: row["gflop_per_second"])
            if narrow:
                summary["kernel_gflops_width_one"] = narrow[0]["gflop_per_second"]
                summary["kernel_gbytes_width_one"] = narrow[0]["gbyte_per_second"]
            summary["kernel_gflops_best"] = broad["gflop_per_second"]
            summary["kernel_best_width"] = broad["width"]
    breakdown = record.get("breakdown")
    if breakdown and latency:
        single = breakdown["single"]
        total = latency["latency_seconds"]["median"]
        one_map = single["map_evaluation"]
        # The ladder's own accounting says the solve is this many map
        # evaluations, so a fraction near one means the marginal costs and the
        # measured solve agree and everything else the Krylov step does is
        # small; a fraction below one is what that organisation costs on top.
        summary["map_budget_fraction"] = breakdown["map_budget_seconds"] / total
        if breakdown.get("reliable"):
            summary["coupling_fraction_of_map"] = single["coupling_in_map"] / one_map
            summary["topology_fraction_of_map"] = single["topology_read"] / one_map
            summary["source_fraction_of_map"] = single["source_evaluation"] / one_map
        published = breakdown.get("published", {})
        for name, key in (
            ("receipt_overhead_eager", "eager_seconds"),
            ("receipt_overhead_compiled", "compiled_seconds"),
        ):
            value = published.get(key)
            if isinstance(value, (int, float)):
                summary[name] = value / total - 1.0
    return summary


def configure_compilation_cache(request: str | None):
    """Point JAX's persistent cache where the run asked, and say where.

    Nothing in the forward solve path enables this cache, so a process that
    does not ask for it recompiles the whole ladder every time. The default
    here is therefore an explicit choice of the driver and not inherited.
    """
    from nova.biot.tiledassembly import compilation_cache

    if request is None:
        return None
    if request.strip().lower() in {"off", "none", ""}:
        os.environ["NOVA_COMPILATION_CACHE"] = "off"
        return None
    return compilation_cache(request, min_compile_seconds=0.0)


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------
def render_figure(
    records: list[dict[str, Any]],
    output: Path,
    baselines: list[dict[str, Any]] | None = None,
) -> None:
    """Write the throughput scaling and its time attribution.

    A baseline record contributes one horizontal rate per cell count. Drawing
    it against the same axis is what makes the comparison answerable: the
    accelerator only repays its own latency once enough members are in flight,
    and the crossing is where that happens.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), gridspec_kw={"width_ratios": (1.32, 1.0)}
    )
    palette = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")

    for index, record in enumerate(
        sorted(baselines or [], key=lambda item: item["cells"])
    ):
        latency = record.get("latency")
        if not latency:
            continue
        left.axhline(
            latency["solves_per_second"],
            color=palette[index % len(palette)],
            linestyle="--",
            linewidth=1.2,
            alpha=0.75,
        )
        left.annotate(
            "one host node, %d cells" % record["cells"],
            xy=(1.0, latency["solves_per_second"]),
            xytext=(3.0, 3.0),
            textcoords="offset points",
            color=palette[index % len(palette)],
            fontsize=7.5,
            alpha=0.9,
        )

    annotated = False
    for index, record in enumerate(sorted(records, key=lambda item: item["cells"])):
        batch = [point for point in record.get("batch", []) if "failed" not in point]
        if not batch:
            continue
        colour = palette[index % len(palette)]
        members = [point["members"] for point in batch]
        rate = [point["solves_per_second"] for point in batch]
        left.plot(
            members,
            rate,
            "o-",
            color=colour,
            markersize=4.5,
            linewidth=1.6,
            label="%d cells" % record["cells"],
        )
        latency = record.get("latency")
        if latency:
            single = latency["solves_per_second"]
            left.axhline(single, color=colour, linestyle=":", linewidth=1.1)
            left.annotate(
                "%.1f solves/s at N=1  (%.0f ms)"
                % (single, 1.0e3 * latency["latency_seconds"]["median"]),
                xy=(members[0], single),
                xytext=(2.0, -13.0),
                textcoords="offset points",
                color=colour,
                fontsize=8.0,
            )
            best = max(batch, key=lambda point: point["solves_per_second"])
            if not annotated:
                left.annotate(
                    "×%.0f at N=%d"
                    % (best["solves_per_second"] / single, best["members"]),
                    xy=(best["members"], best["solves_per_second"]),
                    xytext=(-16.0, 12.0),
                    textcoords="offset points",
                    color=colour,
                    fontsize=9.5,
                    fontweight="bold",
                    arrowprops={"arrowstyle": "-", "color": colour, "linewidth": 0.9},
                )
                annotated = True

    left.set_xscale("log", base=2)
    left.set_yscale("log")
    # The single-solve rate is annotated below its own line, so the axis needs
    # room under the slowest point for that label to sit inside the frame.
    floor = min(
        (
            point["solves_per_second"]
            for record in records
            for point in record.get("batch", [])
            if "failed" not in point
        ),
        default=None,
    )
    if floor:
        left.set_ylim(bottom=floor / 3.0)
    left.set_xlabel("ensemble members in flight, N")
    left.set_ylabel("solves per second")
    left.set_title("batched forward solve throughput", fontsize=10.5, loc="left")
    left.legend(frameon=False, fontsize=8.5, loc="upper left")
    for spine in ("top", "right"):
        left.spines[spine].set_visible(False)

    # Only the directly measured chains are drawn. A layer read as the
    # difference of two nested chains is meaningless once one layer dominates
    # by three decades — the difference is then all measurement error — and
    # that is exactly the regime this solve turns out to be in.
    reference = max(
        (item for item in records if item.get("breakdown", {}).get("chains")),
        key=lambda item: item["cells"],
        default=None,
    )
    if reference is not None:
        breakdown = reference["breakdown"]
        chains = breakdown["chains"]
        wide = breakdown.get("batched", {})
        members = breakdown.get("batched_members", 0)
        names = ("coupling_product", "topology_read", "map_evaluation")
        labels = ("coupling\nmatvec", "topology\nread", "whole map\nevaluation")
        alone = [1.0e6 * chains[name]["marginal"] for name in names]
        position = np.arange(len(names))
        right.bar(
            position - 0.19,
            alone,
            width=0.36,
            color="#c44e52",
            label="one solve in flight",
        )
        if wide and members > 1:
            batched_cost = [
                1.0e6 * wide[name]["marginal_per_member"]
                for name in names
                if name in wide
            ]
            if len(batched_cost) == len(names):
                right.bar(
                    position + 0.19,
                    batched_cost,
                    width=0.36,
                    color="#4c72b0",
                    label="per member, %d in flight" % members,
                )
        right.set_yscale("log")
        right.set_xticks(position)
        right.set_xticklabels(labels, fontsize=8.5)
        right.set_ylabel("microseconds per evaluation")
        right.set_title(
            "where one map evaluation goes, %d cells" % reference["cells"],
            fontsize=10.5,
            loc="left",
        )
        right.legend(frameon=False, fontsize=8.0)
        for spine in ("top", "right"):
            right.spines[spine].set_visible(False)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    print("wrote", output, flush=True)


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Run one driver command."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("prepare", help="assemble and store one machine")
    build.add_argument("--cells", type=int, required=True)
    build.add_argument("--bundle", required=True)

    run = commands.add_parser("measure", help="time one bundle on one device")
    run.add_argument("--bundle", required=True)
    run.add_argument("--platform", default=None, choices=("cpu", "gpu", "cuda"))
    run.add_argument("--compile-cache", default="off")
    run.add_argument("--stage", default="all")
    run.add_argument("--batch", default="1,4,16,64,256")
    run.add_argument("--saturation", type=int, default=0)
    run.add_argument("--label", default="")
    run.add_argument("--output", required=True)

    draw = commands.add_parser("figure", help="render the throughput figure")
    draw.add_argument("--result", nargs="+", required=True)
    draw.add_argument("--baseline", nargs="*", default=())
    draw.add_argument("--output", required=True)

    arguments = parser.parse_args(argv)

    if arguments.command == "prepare":
        bundle = prepare_bundle(arguments.cells)
        bundle.save(Path(arguments.bundle))
        print(
            "prepared %d cells, %d nodes, build %.1f s, residual %.3e -> %s"
            % (
                bundle.cell_number,
                bundle.node_number,
                bundle.build_seconds,
                bundle.reference_residual,
                arguments.bundle,
            )
        )
        return 0

    if arguments.command == "measure":
        if arguments.platform:
            # The backend registry names the accelerator by its vendor runtime,
            # so the colloquial request is translated rather than passed on.
            os.environ["JAX_PLATFORMS"] = (
                "cuda" if arguments.platform == "gpu" else arguments.platform
            )
        record = run_measurement(arguments)
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(record, indent=2, default=str))
        print("wrote", output, flush=True)
        return 0

    records = [json.loads(Path(item).read_text()) for item in arguments.result]
    baselines = [json.loads(Path(item).read_text()) for item in arguments.baseline]
    render_figure(records, Path(arguments.output), baselines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
