"""Where the free-boundary topology read spends its time, and what a plain
Newton root find on a lagged topology costs instead of the Krylov ladder.

The shipped forward solve closes the free-boundary map with a fixed-budget
Newton-Krylov ladder, and a companion driver
(``benchmarks/forward_solve_throughput.py``) established that the map is
dominated by one stage: reading the topology of a trial flux costs 99.6% of a
map evaluation on an accelerator, and is 145 times slower there than on a host
core. This driver asks the two questions that measurement leaves open, and it
reuses that driver's bundle, operator construction, timing discipline and
parity gate rather than restating them.

where the read goes
    The read is decomposed into its stages -- the stencil gather, the
    zero-crossing categorisation that separates extrema from saddles, the
    cluster selection, the sub-cell quadratic fits, the wall read, the
    axis/X-point selection, the connectivity cut and the label assembly --
    and each is timed as a chained self-map so its marginal cost carries no
    launch or transfer overhead. The compiled program's sequential loops are
    inventoried alongside, because a stage whose cost is a loop trip count
    behaves quite differently from one whose cost is arithmetic.

what a lagged topology buys
    The domain labels are frozen through a solve on the reference case, which
    makes the map a smooth function of the flux alone and admits a plain
    Newton step with a directly formed Jacobian. Two ways of forming it are
    measured against each other: differentiating the frozen map in forward
    mode over every direction, and writing the Jacobian down. With the labels
    frozen the cell current is local in normalised flux and couples globally
    only through the two scalars the normalisation is built from, so

        I - J = I - C (D + u a^T + v b^T),

    with ``C`` the plasma coupling, ``D`` diagonal, and ``a``, ``b`` the
    gradients of the fitted axis and boundary flux. That structure also
    reduces the solved system from one unknown per node to one per cell the
    source drives, plus two.

Every prototype lives in this file. Nothing in the package is changed by it,
and the converged flux is required to match the shipped route's within the
resolution floor the solve itself can express -- the coarser of the axis-flux
ladder and the coupling's accumulated round-off -- with identical domain
labels, so a fast route that has stopped solving the problem cannot pass.

Usage::

    uv run python benchmarks/forward_newton_ladder.py stages \\
      --bundle /path/to/forward_solve_1587.npz --platform cpu \\
      --output /path/to/stages_cpu_1587.json

    uv run python benchmarks/forward_newton_ladder.py newton \\
      --bundle /path/to/forward_solve_1587.npz --platform cpu \\
      --output /path/to/newton_cpu_1587.json

    uv run python benchmarks/forward_newton_ladder.py figure \\
      --stages /path/to/stages_*.json --newton /path/to/newton_*.json \\
      --output docs/figures/flux-function-forward-equilibrium/newton-ladder.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.forward_solve_throughput import (  # noqa: E402
    KRYLOV_ITERATIONS,
    LATENCY_REPEATS,
    LATENCY_WARMUP,
    NEWTON_STEPS,
    WARMUP_SWEEPS,
    SolveBundle,
    build_operator,
    device_report,
    ensemble_current,
    marginal_seconds,
    peak_bytes,
    solve_callable,
    time_call,
)

#: Outer topology reads and inner Newton steps the fixed-schedule route runs.
#: Both are budgets rather than tolerances, for the same reason the shipped
#: ladder carries a fixed step count: a data-dependent exit will not ``vmap``.
#: The adaptive measurement below reports what the case actually needs, and
#: these are set from it.
OUTER_READS = 4
INNER_STEPS = 4

#: Ceiling on the adaptive loop, which stops on its own convergence test long
#: before reaching either. A run that hits one has not converged and says so.
OUTER_LIMIT = 12
INNER_LIMIT = 12

#: The adaptive inner loop stops when a Newton step moves the flux by less
#: than this multiple of the resolution floor, and the outer loop stops when
#: the labels repeat and the flux has settled to the same measure. Eight steps
#: of the floor is the same margin the shipped solve's own root-find tolerance
#: is written against.
RESOLUTION_STEPS = 8.0

#: Ensemble widths the batched check is read at.
BATCH_WIDTHS = (1, 16, 64, 256)


# --------------------------------------------------------------------------
# the resolution floor the parity gate is read against
# --------------------------------------------------------------------------
def flux_resolution(operator, axis_flux: float, scale: float) -> float:
    """Return the finest flux difference [Wb] the whole solve can express.

    Two floors compete and the coarser binds: one step of the ladder the
    fitted axis flux lands on, and the round-off a dot product over every grid
    node accumulates on the flux scale. This is the measure the shipped
    solve's own convergence is qualified against, restated here so a
    comparison between two routes is read against the arithmetic rather than
    against a tolerance carried in from another case.
    """
    fit = np.dtype(operator.grid.null.fit_dtype)
    quantum = float(np.spacing(fit.type(abs(axis_flux))))
    accumulation = operator.grid.node_number * float(np.spacing(float(scale)))
    return max(quantum, accumulation)


# --------------------------------------------------------------------------
# the topology read, stage by stage
# --------------------------------------------------------------------------
def vectorised_crossing_count(psi_stencil):
    """Return the zero-crossing count of every stencil ring at once.

    The shipped categorisation walks the six ring vertices of every stencil in
    a sequential scan nested inside a second sequential scan over the stencils
    themselves, accumulating a sign and a counter. The counter is a property
    of each ring alone: with the ring's vertices compared against its centre,
    it counts the sign changes around the closed traversal, the initial sign
    being the last vertex's so the comparison is cyclic. Written that way the
    whole grid is one elementwise kernel, and the result is identical value
    for value rather than merely equivalent.

    Zero crossings separate the extrema (no change) from the saddles (four).
    """
    import jax.numpy as jnp

    sign = psi_stencil[:, 1:] > psi_stencil[:, :1]
    return jnp.sum(sign != jnp.roll(sign, 1, axis=1), axis=1)


def vectorised_categorise(null, psi_stencil):
    """Return the cluster selection the shipped categorisation returns.

    Same contract as ``Null2D.categorize`` -- the extremum and saddle counts,
    the selected clusters and their physical origin and scale -- with the
    sequential traversal replaced and the two null types selected in one pass
    instead of a scan over them.
    """
    import jax.numpy as jnp

    psi_stencil = jnp.asarray(psi_stencil, dtype=null.fit_dtype)
    count = vectorised_crossing_count(psi_stencil)
    number = jnp.array([jnp.sum(count == 0), jnp.sum(count == 4)])
    index = jnp.stack(
        [jnp.where(count == kind, size=null.maxsize)[0] for kind in (0, 4)]
    )
    cluster = jnp.concatenate(
        (
            null.local_coordinate_stencil[index],
            psi_stencil[index][..., jnp.newaxis],
        ),
        axis=-1,
    )
    return number, cluster, null.physical_origin[index], null.physical_scale[index]


def vectorised_interpolate(number, cluster, origin, scale):
    """Return the sub-cell fits of one null type, all clusters at once.

    The shipped fit walks the selected clusters in a scan whose only carried
    state is the position, used to mask the surplus clusters the fixed-size
    selection padded with. The position is known without carrying it, so every
    cluster's quadratic is fitted independently and the same mask applied.
    """
    import jax
    import jax.numpy as jnp

    from nova.geometry import select

    def fit(one, one_origin, one_scale):
        """Return one sub-cell null in physical coordinates."""
        local = select.traced_subnull(one[:, 0], one[:, 1], one[:, 2])
        physical = one_origin + local[:2].astype(jnp.float64) * one_scale
        return jnp.concatenate((physical, local[2:].astype(jnp.float64)))

    result = jax.vmap(fit)(cluster, origin, scale)
    position = jnp.arange(1, cluster.shape[0] + 1)
    return jnp.where((position <= number)[:, jnp.newaxis], result, jnp.full(4, jnp.nan))


def vectorised_nulls(null, psi_grid):
    """Return the extremum and saddle tables of one flux map."""
    import jax
    import jax.numpy as jnp

    psi_stencil = jnp.asarray(psi_grid, dtype=null.fit_dtype)[null.stencil]
    number, cluster, origin, scale = vectorised_categorise(null, psi_stencil)
    return jax.vmap(vectorised_interpolate)(number, cluster, origin, scale)


def vectorised_connectivity(topology, data_o, vmap_x):
    """Return the axis-connected cut, every X-point applied at once.

    The shipped cut folds the X-points in a scan, each narrowing a mask that
    can only lose entries: a cell already cut off stays cut off, because the
    scan's own guard keeps it. The fold is therefore a conjunction over the
    finite X-points of one half-plane test each, and reads as one.
    """
    import jax.numpy as jnp

    height = topology.grid.coordinate[:, 1]
    x_height = vmap_x[:, 1]
    below = (x_height < data_o[1])[:, jnp.newaxis]
    test = jnp.where(
        below,
        height[jnp.newaxis, :] > x_height[:, jnp.newaxis],
        height[jnp.newaxis, :] < x_height[:, jnp.newaxis],
    )
    finite = jnp.isfinite(vmap_x[:, 0])[:, jnp.newaxis]
    return jnp.all(jnp.where(finite, test, True), axis=0)


def vectorised_read(operator, psi):
    """Return the domain labels and topology state, sequential scans removed.

    Every selection, cut and fit the shipped read performs is performed here,
    in the same order and the same arithmetic; only the traversals that carried
    no state between their steps are written as kernels over the whole grid.
    The result is required to be identical to the shipped read value for value,
    which is what makes it a candidate implementation rather than an
    approximation.
    """
    from nova.equilibrium.domain import classify_domains

    topology = operator.topology
    polarity = operator.polarity
    psi_grid, psi_wall = topology.split_flux_map(psi)
    vmap_o, vmap_x = vectorised_nulls(topology.grid, psi_grid)
    data_o = topology.o_point_data(vmap_o, polarity)
    data_x = topology.x_point_data(vmap_x, polarity, data_o[2])
    data_w = topology.wall(psi_wall, polarity)
    data_b = topology.boundary(data_o, vmap_x, data_w, polarity)
    psi_norm = topology.normalize(data_o[2], data_b[2], psi_grid)
    psi_lcfs = topology.psi_lcfs(data_o[2], data_b[2])
    masks = classify_domains(
        psi_norm,
        topology.psi_mask(polarity, psi_grid, psi_lcfs),
        vectorised_connectivity(topology, data_o, vmap_x),
        operator.inside_material,
    )
    from nova.equilibrium.topology import TopologyState

    state = TopologyState(
        axis=data_o[:2],
        axis_flux=data_o[2],
        boundary=data_b[:2],
        boundary_flux=data_b[2],
        x_point=data_x[:2],
        x_point_flux=data_x[2],
        wall_point=data_w[:2],
        wall_point_flux=data_w[2],
        diverted=data_b[2] == data_x[2],
    )
    return masks, state


def read_stages(operator) -> dict[str, Callable]:
    """Return one self-map per cumulative stage of the topology read.

    Each entry evaluates the stages up to and including its own and folds a
    vanishing multiple of everything it computed back into the flux, so no
    stage can be eliminated as dead and none moves the state it is measured
    at. Differencing two neighbours gives one stage's own marginal cost.
    """
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.domain import classify_domains

    topology = operator.topology
    null = operator.grid.null
    polarity = operator.polarity
    grid_nodes = operator.grid.node_number

    def held(psi, value):
        """Return the flux with a vanishing trace of one stage folded in."""
        return psi + 1.0e-30 * jnp.asarray(value, dtype=psi.dtype)

    def stencil_flux(psi):
        """Return the flux gathered onto the stencil rings."""
        return jnp.asarray(psi[:grid_nodes], dtype=null.fit_dtype)[null.stencil]

    def gather(psi):
        """Gather the flux onto every ring and nothing else."""
        return held(psi, jnp.sum(stencil_flux(psi)))

    def crossing_scan(psi):
        """Add the shipped sequential zero-crossing categorisation."""
        number, count = jax.lax.scan(null.zero_cross_count, (0, 0), stencil_flux(psi))
        return held(psi, jnp.sum(count) + number[0] + number[1])

    def crossing_kernel(psi):
        """Add the same categorisation written as one elementwise kernel."""
        count = vectorised_crossing_count(stencil_flux(psi))
        return held(psi, jnp.sum(count) + jnp.sum(count == 0) + jnp.sum(count == 4))

    def selection_scan(psi):
        """Add the shipped cluster selection on top of the scan."""
        number, cluster, origin, scale = null.categorize(stencil_flux(psi))
        return held(psi, jnp.sum(number) + jnp.sum(cluster) + jnp.sum(origin * scale))

    def selection_kernel(psi):
        """Add the same cluster selection on top of the kernel."""
        number, cluster, origin, scale = vectorised_categorise(null, stencil_flux(psi))
        return held(psi, jnp.sum(number) + jnp.sum(cluster) + jnp.sum(origin * scale))

    def fit_scan(psi):
        """Add the shipped sub-cell quadratic fits."""
        vmap_o, vmap_x = null(psi[:grid_nodes])
        return held(psi, jnp.nansum(vmap_o) + jnp.nansum(vmap_x))

    def fit_kernel(psi):
        """Add the same sub-cell fits, mapped rather than scanned."""
        vmap_o, vmap_x = vectorised_nulls(null, psi[:grid_nodes])
        return held(psi, jnp.nansum(vmap_o) + jnp.nansum(vmap_x))

    def wall_read(psi):
        """Read the wall extremum alone."""
        return held(psi, jnp.nansum(topology.wall(psi[grid_nodes:], polarity)))

    def boundary_choice(psi):
        """Add the axis, X-point and wall-limit selection to the fits."""
        vmap_o, vmap_x = null(psi[:grid_nodes])
        data_o = topology.o_point_data(vmap_o, polarity)
        data_w = topology.wall(psi[grid_nodes:], polarity)
        data_b = topology.boundary(data_o, vmap_x, data_w, polarity)
        return held(psi, jnp.nansum(data_o) + jnp.nansum(data_b))

    def connectivity_scan(psi):
        """Add the shipped scanned axis-connectivity cut."""
        vmap_o, vmap_x = null(psi[:grid_nodes])
        data_o = topology.o_point_data(vmap_o, polarity)
        data_w = topology.wall(psi[grid_nodes:], polarity)
        data_b = topology.boundary(data_o, vmap_x, data_w, polarity)
        connected = topology.x_mask(data_o, vmap_x)
        return held(psi, jnp.nansum(data_b) + jnp.sum(connected))

    def connectivity_kernel(psi):
        """Add the same cut written as a conjunction of half-plane tests."""
        vmap_o, vmap_x = null(psi[:grid_nodes])
        data_o = topology.o_point_data(vmap_o, polarity)
        data_w = topology.wall(psi[grid_nodes:], polarity)
        data_b = topology.boundary(data_o, vmap_x, data_w, polarity)
        connected = vectorised_connectivity(topology, data_o, vmap_x)
        return held(psi, jnp.nansum(data_b) + jnp.sum(connected))

    def label_assembly(psi):
        """Add the normalisation and the label partition."""
        psi_grid = psi[:grid_nodes]
        vmap_o, vmap_x = null(psi_grid)
        data_o = topology.o_point_data(vmap_o, polarity)
        data_w = topology.wall(psi[grid_nodes:], polarity)
        data_b = topology.boundary(data_o, vmap_x, data_w, polarity)
        psi_norm = topology.normalize(data_o[2], data_b[2], psi_grid)
        psi_lcfs = topology.psi_lcfs(data_o[2], data_b[2])
        masks = classify_domains(
            psi_norm,
            topology.psi_mask(polarity, psi_grid, psi_lcfs),
            topology.x_mask(data_o, vmap_x),
            operator.inside_material,
        )
        return held(psi, jnp.sum(masks.label) + jnp.sum(masks.psi_norm))

    def shipped_read(psi):
        """Run the whole shipped read."""
        masks, state = operator.read(psi)
        return held(psi, jnp.sum(masks.label) + state.axis_flux + state.boundary_flux)

    def kernel_read(psi):
        """Run the whole read with every stateless traversal vectorised."""
        masks, state = vectorised_read(operator, psi)
        return held(psi, jnp.sum(masks.label) + state.axis_flux + state.boundary_flux)

    return {
        "stencil_gather": gather,
        "crossing_count_scan": crossing_scan,
        "crossing_count_kernel": crossing_kernel,
        "cluster_selection_scan": selection_scan,
        "cluster_selection_kernel": selection_kernel,
        "subcell_fit_scan": fit_scan,
        "subcell_fit_kernel": fit_kernel,
        "wall_read": wall_read,
        "boundary_choice": boundary_choice,
        "connectivity_scan": connectivity_scan,
        "connectivity_kernel": connectivity_kernel,
        "label_assembly": label_assembly,
        "shipped_read": shipped_read,
        "kernel_read": kernel_read,
    }


#: Neighbouring stages whose difference is one stage's own marginal cost, and
#: the pairs whose difference is what removing a sequential traversal saves.
STAGE_DIFFERENCES = (
    ("crossing_count", "crossing_count_scan", "stencil_gather"),
    ("cluster_selection", "cluster_selection_scan", "crossing_count_scan"),
    ("subcell_fit", "subcell_fit_scan", "cluster_selection_scan"),
    ("boundary_selection", "boundary_choice", "subcell_fit_scan"),
    ("connectivity_cut", "connectivity_scan", "boundary_choice"),
    ("label_partition", "label_assembly", "connectivity_scan"),
    ("crossing_count_saved", "crossing_count_scan", "crossing_count_kernel"),
    ("subcell_fit_saved", "subcell_fit_scan", "subcell_fit_kernel"),
    ("connectivity_saved", "connectivity_scan", "connectivity_kernel"),
    ("whole_read_saved", "shipped_read", "kernel_read"),
)


def loop_inventory(call, argument) -> dict[str, Any]:
    """Return the sequential loops one compiled program carries.

    A stage whose cost is a loop trip count and a stage whose cost is
    arithmetic respond to entirely different remedies, so the compiled
    program's own loop structure is read rather than inferred from the timing.
    """
    import jax

    try:
        text = jax.jit(call).lower(argument).compile().as_text()
    except Exception as error:  # noqa: BLE001 - a backend that will not lower
        return {"failed": f"{type(error).__name__}: {str(error)[:160]}"}
    loops = re.findall(r"=\s*\([^\n]*\)\s*while\(", text)
    trips = [
        int(item) for item in re.findall(r'"known_trip_count":\{"n":"(\d+)"\}', text)
    ]
    return {
        "while_loops": len(loops),
        "known_trip_counts": sorted(trips, reverse=True)[:8],
        "fusions": len(re.findall(r"fusion\(", text)),
        "text_bytes": len(text),
    }


def measure_stages(bundle: SolveBundle, operator) -> dict[str, Any]:
    """Return the marginal cost of every stage of the topology read."""
    import jax.numpy as jnp

    flux = jnp.asarray(bundle.reference_flux)
    stages = read_stages(operator)
    chains = {name: marginal_seconds(step, flux) for name, step in stages.items()}
    single = {name: entry["marginal"] for name, entry in chains.items()}
    difference = {}
    for name, upper, lower in STAGE_DIFFERENCES:
        if chains[upper]["usable"] and chains[lower]["usable"]:
            difference[name] = single[upper] - single[lower]
    inventory = {
        name: loop_inventory(stages[name], flux)
        for name in ("shipped_read", "kernel_read", "crossing_count_scan")
    }
    fraction = {}
    whole = single.get("shipped_read", 0.0)
    if whole > 0.0:
        for name, value in difference.items():
            if not name.endswith("_saved"):
                fraction[name] = value / whole
        fraction["kernel_read_over_shipped"] = single["kernel_read"] / whole
    return {
        "chains": chains,
        "marginal": single,
        "stage": difference,
        "fraction_of_read": fraction,
        "loops": inventory,
        "stencil_rings": int(operator.grid.null.stencil.shape[0]),
        "ring_vertices": int(operator.grid.null.stencil.shape[1]),
    }


def check_read_identity(operator, flux) -> dict[str, Any]:
    """Return whether the vectorised read reproduces the shipped one exactly."""
    import numpy as np

    shipped_masks, shipped_state = operator.read(flux)
    kernel_masks, kernel_state = vectorised_read(operator, flux)
    label_match = bool(
        np.array_equal(np.asarray(shipped_masks.label), np.asarray(kernel_masks.label))
    )
    norm_deviation = float(
        np.max(
            np.abs(
                np.asarray(shipped_masks.psi_norm) - np.asarray(kernel_masks.psi_norm)
            )
        )
    )
    return {
        "labels_identical": label_match,
        "psi_norm_deviation": norm_deviation,
        "axis_flux_deviation": float(
            abs(float(shipped_state.axis_flux) - float(kernel_state.axis_flux))
        ),
        "boundary_flux_deviation": float(
            abs(float(shipped_state.boundary_flux) - float(kernel_state.boundary_flux))
        ),
        "identical": label_match and norm_deviation == 0.0,
    }


# --------------------------------------------------------------------------
# the lagged-topology Newton prototype
# --------------------------------------------------------------------------
def coupling_matrix(operator):
    """Return the plasma coupling of every node, grid rows above wall rows."""
    import jax.numpy as jnp

    return jnp.concatenate(
        (operator.grid.plasma_target, operator.wall.plasma_target), axis=0
    )


def topology_identity(operator, psi, vectorised: bool = True):
    """Return the frozen topology one trial flux reads.

    The identity is everything the map's nonlinearity is lagged through: the
    domain label of every cell, which stencil ring carries the magnetic axis,
    which carries the primary X-point, and whether the boundary is the X-point
    or the wall contact. Freezing those makes the remaining map a smooth
    function of the flux, and re-reading them is the outer iteration.
    """
    import jax.numpy as jnp

    topology = operator.topology
    null = operator.grid.null
    polarity = operator.polarity
    psi_grid, psi_wall = topology.split_flux_map(psi)
    psi_stencil = jnp.asarray(psi_grid, dtype=null.fit_dtype)[null.stencil]
    count = vectorised_crossing_count(psi_stencil)
    ring = jnp.stack(
        [jnp.where(count == kind, size=null.maxsize)[0] for kind in (0, 4)]
    )
    if vectorised:
        vmap_o, vmap_x = vectorised_nulls(null, psi_grid)
    else:
        vmap_o, vmap_x = null(psi_grid)
    data_o = topology.o_point_data(vmap_o, polarity)
    order_o = topology.o_point_index(vmap_o, polarity)
    order_x = topology.x_point_index(vmap_x, polarity, data_o[2])
    data_w = topology.wall(psi_wall, polarity)
    data_b = topology.boundary(data_o, vmap_x, data_w, polarity)
    masks, state = vectorised_read(operator, psi) if vectorised else operator.read(psi)
    return {
        "label": masks.label,
        "axis_ring": ring[0][order_o],
        "x_ring": ring[1][order_x],
        "wall_limited": jnp.logical_not(state.diverted),
        "axis_flux": data_o[2],
        "boundary_flux": data_b[2],
    }


def frozen_route(operator, identity):
    """Return the frozen map, its Jacobian factors and one Newton step.

    With the labels frozen the cell current of a cell depends on the flux at
    that cell alone, through the normalised flux, and on the two scalars the
    normalisation is built from -- the fitted axis flux and the fitted
    boundary flux -- which every cell shares. Differentiating that gives

        dq/dpsi = D + u a^T + v b^T,

    diagonal plus rank two, with ``D`` the local derivative divided by the
    flux span, ``u`` and ``v`` its two normalisation partials, and ``a``, ``b``
    the gradients of the two fitted scalars. Those gradients are supported on
    the frozen rings alone, so the whole Jacobian of the map is the coupling
    applied to a diagonal and two outer products, and is written down rather
    than differentiated.
    """
    import jax
    import jax.numpy as jnp

    from nova.equilibrium.domain import DomainMasks
    from nova.geometry import select

    null = operator.grid.null
    topology = operator.topology
    polarity = operator.polarity
    grid_nodes = operator.grid.node_number
    coupling = coupling_matrix(operator)
    label = identity["label"]
    axis_ring = identity["axis_ring"]
    x_ring = identity["x_ring"]
    wall_limited = identity["wall_limited"]

    def ring_flux(psi_grid, ring):
        """Return the sub-cell fitted flux of one frozen stencil ring."""
        local = null.local_coordinate_stencil[ring]
        cluster = jnp.asarray(psi_grid, dtype=null.fit_dtype)[null.stencil[ring]]
        return select.traced_subnull(local[:, 0], local[:, 1], cluster)[2]

    def normalisation(psi):
        """Return the axis and boundary flux the frozen identity implies."""
        psi_grid, psi_wall = topology.split_flux_map(psi)
        axis = ring_flux(psi_grid, axis_ring)
        saddle = ring_flux(psi_grid, x_ring)
        contact = topology.wall(psi_wall, polarity)[2]
        boundary = jnp.where(wall_limited, contact, saddle)
        return (
            jnp.asarray(axis, dtype=psi.dtype),
            jnp.asarray(boundary, dtype=psi.dtype),
        )

    def cell_current(psi_norm):
        """Return the per-cell current the frozen labels drive."""
        return operator.source.cell_current(
            operator.radius, operator.area, DomainMasks(label=label, psi_norm=psi_norm)
        )

    def normalised(psi, axis, boundary):
        """Return the normalised flux of every cell."""
        return (psi[:grid_nodes] - axis) / (boundary - axis)

    def mapped(psi, external):
        """Return the frozen free-boundary map of one trial flux."""
        axis, boundary = normalisation(psi)
        return external + coupling @ cell_current(normalised(psi, axis, boundary))

    def factors(psi):
        """Return the diagonal, the two rank-one pairs and the flux span."""
        axis, boundary = normalisation(psi)
        span = boundary - axis
        psi_norm = normalised(psi, axis, boundary)
        # the source is elementwise in normalised flux under frozen labels, so
        # one forward pass along the all-ones direction carries every local
        # derivative at once rather than one column of a Jacobian
        local = jax.jvp(cell_current, (psi_norm,), (jnp.ones_like(psi_norm),))[1]
        diagonal = local / span
        gradient_a = jax.grad(lambda state: normalisation(state)[0])(psi)
        gradient_b = jax.grad(lambda state: normalisation(state)[1])(psi)
        return {
            "diagonal": diagonal,
            "axis_partial": diagonal * (psi_norm - 1.0),
            "boundary_partial": -diagonal * psi_norm,
            "axis_gradient": gradient_a,
            "boundary_gradient": gradient_b,
        }

    def dense_jacobian(psi):
        """Return the frozen map's Jacobian as a dense square operator."""
        part = factors(psi)
        nodes = coupling.shape[0]
        dense = jnp.zeros((nodes, nodes), dtype=psi.dtype)
        dense = dense.at[:, :grid_nodes].set(coupling * part["diagonal"])
        dense = dense + jnp.outer(
            coupling @ part["axis_partial"], part["axis_gradient"]
        )
        dense = dense + jnp.outer(
            coupling @ part["boundary_partial"], part["boundary_gradient"]
        )
        return dense

    def dense_step(psi, residual):
        """Return the Newton step from a dense factorisation of ``I - J``."""
        nodes = coupling.shape[0]
        operator_matrix = jnp.eye(nodes, dtype=psi.dtype) - dense_jacobian(psi)
        return jnp.linalg.solve(operator_matrix, residual)

    def reduced_step(psi, residual, support):
        """Return the same step from the system the driven cells alone span.

        Writing the step as ``s = f + C y`` leaves ``y`` supported on the cells
        the source drives, because a cell no closure reaches contributes no row
        to ``dq/dpsi``. The solved system is then one unknown per driven cell
        rather than one per node, and the two normalisation scalars ride along
        inside it as the pair of rank-one terms rather than as extra unknowns.
        """
        part = factors(psi)
        # a padded slot indexes a zero column of the coupling and a zero row of
        # the derivative, so it contributes an identity row and is harmless
        pad_diagonal = jnp.concatenate((part["diagonal"], jnp.zeros(1, psi.dtype)))
        pad_axis = jnp.concatenate((part["axis_partial"], jnp.zeros(1, psi.dtype)))
        pad_boundary = jnp.concatenate(
            (part["boundary_partial"], jnp.zeros(1, psi.dtype))
        )
        pad_coupling = jnp.concatenate(
            (coupling, jnp.zeros((coupling.shape[0], 1), psi.dtype)), axis=1
        )
        columns = pad_coupling[:, support]
        rows = jnp.where(support < grid_nodes, support, jnp.zeros_like(support))
        block = pad_coupling[rows][:, support]
        axis_row = part["axis_gradient"] @ pad_coupling[:, support]
        boundary_row = part["boundary_gradient"] @ pad_coupling[:, support]
        diagonal = pad_diagonal[support]
        axis_partial = pad_axis[support]
        boundary_partial = pad_boundary[support]
        system = (
            jnp.eye(support.shape[0], dtype=psi.dtype)
            - diagonal[:, jnp.newaxis] * block
            - jnp.outer(axis_partial, axis_row)
            - jnp.outer(boundary_partial, boundary_row)
        )
        forcing = (
            diagonal * residual[rows]
            + axis_partial * (part["axis_gradient"] @ residual)
            + boundary_partial * (part["boundary_gradient"] @ residual)
        )
        return residual + columns @ jnp.linalg.solve(system, forcing)

    def krylov_step(psi, residual, iterations=KRYLOV_ITERATIONS):
        """Return the step a Krylov solve on the same structure produces.

        The structured Jacobian applies in one coupling product and two inner
        products, without forming or factorising anything, so this is the route
        whose cost grows as the coupling rather than as the cube of the driven
        cell count. It is the shipped ladder's own inner solve with the exact
        tangent replaced by the written-down one and the topology lagged.
        """
        part = factors(psi)

        def apply(direction):
            """Return ``(I - J) w`` for one direction."""
            cell = (
                part["diagonal"] * direction[:grid_nodes]
                + part["axis_partial"] * (part["axis_gradient"] @ direction)
                + part["boundary_partial"] * (part["boundary_gradient"] @ direction)
            )
            return direction - coupling @ cell

        return jax.scipy.sparse.linalg.gmres(
            apply,
            residual,
            maxiter=iterations,
            restart=iterations,
            solve_method="batched",
        )[0]

    return {
        "map": mapped,
        "factors": factors,
        "dense_jacobian": dense_jacobian,
        "dense_step": dense_step,
        "reduced_step": reduced_step,
        "krylov_step": krylov_step,
        "normalisation": normalisation,
    }


def declared_support(operator, identity):
    """Return the mask of cells the frozen source drives current on."""
    import jax.numpy as jnp

    from nova.equilibrium.domain import DomainMasks

    return operator.source.declared_support(
        DomainMasks(
            label=identity["label"],
            psi_norm=jnp.zeros(operator.grid.node_number, dtype=jnp.float64),
        )
    )


def declared_support_size(operator, identity) -> int:
    """Return how many cells the frozen source drives, as a host integer."""
    return int(np.count_nonzero(np.asarray(declared_support(operator, identity))))


def driven_support(operator, identity, size: int):
    """Return the padded index of every cell the frozen source drives.

    The size is fixed so the reduced system keeps one shape across the whole
    solve and across an ensemble; a slot the case does not fill indexes the
    padded column, which contributes an identity row.
    """
    import jax.numpy as jnp

    return jnp.where(
        declared_support(operator, identity),
        size=size,
        fill_value=operator.grid.node_number,
    )[0]


def adaptive_newton(
    bundle: SolveBundle,
    operator,
    *,
    vectorised: bool = True,
    step: str = "reduced",
    resolution: float | None = None,
) -> dict[str, Any]:
    """Drive the lagged-topology Newton to convergence and report what it took.

    The loop is host driven and stops on its own tests, which is what makes it
    a measurement of how many topology reads and Newton steps the case needs
    rather than a restatement of a budget. The fixed-schedule route timed
    afterwards runs the counts this returns.
    """
    import jax
    import jax.numpy as jnp

    current = jnp.asarray(bundle.coil_current)
    external = operator.external(current)
    psi = jnp.asarray(bundle.seed)
    scale = float(np.max(np.abs(bundle.reference_flux)))
    support_size = None

    def true_residual(state):
        """Return the shipped map's own relative fixed-point residual."""
        mapped = operator(state, current)
        return float(
            jnp.max(jnp.abs(mapped - state))
            / jnp.maximum(jnp.max(jnp.abs(mapped)), 1e-30)
        )

    trace: list[float] = []
    outer_record: list[dict[str, Any]] = []
    labels_previous = None
    reads = 0
    for outer in range(OUTER_LIMIT):
        identity = topology_identity(operator, psi, vectorised=vectorised)
        reads += 1
        label = np.asarray(identity["label"])
        if resolution is None:
            resolution = flux_resolution(operator, float(identity["axis_flux"]), scale)
        if support_size is None:
            support_size = declared_support_size(operator, identity)
        route = frozen_route(operator, identity)
        stepper = jax.jit(
            step_callable(route, driven_support(operator, identity, support_size), step)
        )
        moves: list[float] = []
        for _inner in range(INNER_LIMIT):
            forcing = route["map"](psi, external) - psi
            increment = stepper(psi, forcing)
            psi = psi + increment
            move = float(jnp.max(jnp.abs(increment)))
            moves.append(move)
            trace.append(true_residual(psi))
            if move < RESOLUTION_STEPS * resolution:
                break
        settled = (
            labels_previous is not None
            and np.array_equal(labels_previous, label)
            and moves[0] < RESOLUTION_STEPS * resolution
        )
        outer_record.append(
            {
                "read": outer + 1,
                "inner_steps": len(moves),
                "step_magnitudes": moves,
                "label_changes": (
                    0
                    if labels_previous is None
                    else int(np.count_nonzero(labels_previous != label))
                ),
                "residual": trace[-1],
            }
        )
        labels_previous = label
        if settled:
            break

    flux = np.asarray(psi)
    deviation = float(np.max(np.abs(flux - bundle.reference_flux)))
    final_label = np.asarray(
        topology_identity(operator, psi, vectorised=vectorised)["label"]
    )
    reference_label = np.asarray(
        operator.read(jnp.asarray(bundle.reference_flux))[0].label
    )
    return {
        "outer_reads": reads,
        "converged": reads < OUTER_LIMIT,
        "outer": outer_record,
        "trace": trace,
        "final_residual": trace[-1] if trace else float("nan"),
        "reference_residual": bundle.reference_residual,
        "flux_deviation": deviation,
        "flux_resolution": resolution,
        "deviation_in_resolution_steps": deviation / resolution,
        "parity": deviation < RESOLUTION_STEPS * resolution,
        "labels_identical": bool(np.array_equal(final_label, reference_label)),
        "label_disagreements": int(np.count_nonzero(final_label != reference_label)),
        "support_size": support_size,
        "flux": flux,
    }


def step_callable(route, support, step: str):
    """Return the named linear step of one frozen route."""
    if step == "dense":
        return route["dense_step"]
    if step == "krylov":
        return route["krylov_step"]
    return lambda state, force: route["reduced_step"](state, force, support)


def newton_callable(
    operator,
    support_size: int,
    *,
    outer: int,
    inner: int,
    vectorised: bool = True,
    step: str = "reduced",
):
    """Return the fixed-schedule lagged-topology Newton solve.

    Fixed counts rather than convergence tests, for the reason the shipped
    ladder carries a fixed budget: a data-dependent exit does not ``vmap``, and
    a route that cannot be batched cannot be compared against one that can.
    """
    import jax
    import jax.numpy as jnp

    def solve_one(current, seed):
        """Return the converged flux one conductor state supports."""
        external = operator.external(current)

        def outer_body(_, psi):
            """Re-read the topology and take the inner Newton steps."""
            identity = topology_identity(operator, psi, vectorised=vectorised)
            route = frozen_route(operator, identity)
            stepper = step_callable(
                route, driven_support(operator, identity, support_size), step
            )

            def inner_body(_, state):
                """Take one Newton step on the frozen map."""
                return state + stepper(state, route["map"](state, external) - state)

            return jax.lax.fori_loop(0, inner, inner_body, psi)

        return jax.lax.fori_loop(0, outer, outer_body, jnp.asarray(seed))

    return solve_one


def kernel_read_ladder(operator):
    """Return the shipped ladder driven through the vectorised read.

    Everything but the read is the shipped route: the same map, the same step
    budget, the same Krylov space. Separating this from the shipped ladder is
    what attributes a speedup to removing the sequential traversal rather than
    to lagging the topology, and the two effects turn out to be independent.
    """
    import jax.numpy as jnp

    from nova.equilibrium import fixed_point

    def solve_one(current, seed):
        """Return the converged flux one conductor state supports."""
        external = operator.external(current)

        def mapped(psi):
            """Return the free-boundary map with the read vectorised."""
            masks = vectorised_read(operator, psi)[0]
            cell = operator.source.cell_current(operator.radius, operator.area, masks)
            return (
                external
                + jnp.r_[operator.grid.internal(cell), operator.wall.internal(cell)]
            )

        return fixed_point.newton_krylov(
            mapped,
            seed,
            newton_steps=NEWTON_STEPS,
            gmres_iterations=KRYLOV_ITERATIONS,
            warmup=WARMUP_SWEEPS,
        )

    return solve_one


def measure_jacobian_cost(bundle: SolveBundle, operator) -> dict[str, Any]:
    """Return what forming the frozen Jacobian costs, both ways.

    Forward-mode differentiation over every direction is the route that needs
    no derivation, and the written-down Jacobian is the route that exploits
    what freezing the labels did to the map's structure. The two are timed on
    the same state, against one evaluation of the frozen map, so the answer is
    in map evaluations rather than in seconds alone.
    """
    import jax
    import jax.numpy as jnp

    current = jnp.asarray(bundle.coil_current)
    external = operator.external(current)
    flux = jnp.asarray(bundle.reference_flux)
    identity = topology_identity(operator, flux)
    route = frozen_route(operator, identity)
    support_size = declared_support_size(operator, identity)
    support = driven_support(operator, identity, support_size)
    forcing = route["map"](flux, external) - flux

    entries: dict[str, Any] = {}

    def timed(name: str, call: Callable) -> None:
        """Compile one route and record its latency, or why it would not run."""
        try:
            compiled = jax.jit(call).lower(flux).compile()
            entries[name] = time_call(
                lambda: compiled(flux), LATENCY_REPEATS, LATENCY_WARMUP
            )["median"]
        except Exception as error:  # noqa: BLE001 - an exhausted device is a datum
            entries[name] = f"{type(error).__name__}: {str(error)[:160]}"

    timed("frozen_map", lambda state: route["map"](state, external))
    timed("shipped_map", lambda state: operator(state, current))
    timed("analytic_factors", route["factors"])
    timed("analytic_dense_jacobian", route["dense_jacobian"])
    timed(
        "forward_mode_jacobian", jax.jacfwd(lambda state: route["map"](state, external))
    )
    timed("dense_step", lambda state: route["dense_step"](state, forcing))
    timed("reduced_step", lambda state: route["reduced_step"](state, forcing, support))
    timed("krylov_step", lambda state: route["krylov_step"](state, forcing))

    # the two routes must agree; a structured Jacobian that is merely fast is
    # not a Jacobian
    try:
        analytic = np.asarray(jax.jit(route["dense_jacobian"])(flux))
        traced = np.asarray(
            jax.jit(jax.jacfwd(lambda state: route["map"](state, external)))(flux)
        )
        span = float(np.max(np.abs(traced)))
        entries["jacobian_agreement"] = float(
            np.max(np.abs(analytic - traced)) / max(span, 1e-300)
        )
    except Exception as error:  # noqa: BLE001
        entries["jacobian_agreement"] = f"{type(error).__name__}: {str(error)[:160]}"

    frozen = entries.get("frozen_map")
    if isinstance(frozen, float) and frozen > 0.0:
        entries["in_frozen_map_evaluations"] = {
            name: value / frozen
            for name, value in entries.items()
            if isinstance(value, float)
        }
    entries["support_size"] = support_size
    entries["node_number"] = int(bundle.node_number)
    return entries


def measure_end_to_end(bundle: SolveBundle, operator, adaptive) -> dict[str, Any]:
    """Return the wall time of both routes on the same machine and seed."""
    import jax
    import jax.numpy as jnp

    current = jnp.asarray(bundle.coil_current)
    seed = jnp.asarray(bundle.seed)
    outer = min(max(adaptive["outer_reads"], 2), OUTER_LIMIT)
    inner = max(
        (entry["inner_steps"] for entry in adaptive["outer"]), default=INNER_STEPS
    )
    support_size = adaptive["support_size"]
    record: dict[str, Any] = {"outer_reads": outer, "inner_steps": inner}

    def compile_and_time(name: str, call: Callable, *arguments) -> Any:
        """Compile one solve route and time it after compilation."""
        try:
            start = time.perf_counter()
            compiled = jax.jit(call).lower(*arguments).compile()
            record[f"{name}_compile_seconds"] = time.perf_counter() - start
            timing = time_call(
                lambda: compiled(*arguments), LATENCY_REPEATS, LATENCY_WARMUP
            )
            record[f"{name}_seconds"] = timing["median"]
            return compiled(*arguments)
        except Exception as error:  # noqa: BLE001
            record[f"{name}_seconds"] = f"{type(error).__name__}: {str(error)[:160]}"
            return None

    ladder = compile_and_time("ladder", solve_callable(operator), current, seed)
    ladder_kernel_read = compile_and_time(
        "ladder_kernel_read", kernel_read_ladder(operator), current, seed
    )
    newton = compile_and_time(
        "newton",
        newton_callable(operator, support_size, outer=outer, inner=inner),
        current,
        seed,
    )
    newton_shipped_read = compile_and_time(
        "newton_shipped_read",
        newton_callable(
            operator, support_size, outer=outer, inner=inner, vectorised=False
        ),
        current,
        seed,
    )
    newton_krylov_step = compile_and_time(
        "newton_krylov_step",
        newton_callable(
            operator, support_size, outer=outer, inner=inner, step="krylov"
        ),
        current,
        seed,
    )

    scale = float(np.max(np.abs(bundle.reference_flux)))
    for name, result in (
        ("ladder", None if ladder is None else np.asarray(ladder.state)),
        (
            "ladder_kernel_read",
            None
            if ladder_kernel_read is None
            else np.asarray(ladder_kernel_read.state),
        ),
        ("newton", None if newton is None else np.asarray(newton)),
        (
            "newton_shipped_read",
            None if newton_shipped_read is None else np.asarray(newton_shipped_read),
        ),
        (
            "newton_krylov_step",
            None if newton_krylov_step is None else np.asarray(newton_krylov_step),
        ),
    ):
        if result is None:
            continue
        record[f"{name}_deviation"] = float(
            np.max(np.abs(result - bundle.reference_flux))
        )
        record[f"{name}_deviation_in_resolution_steps"] = (
            record[f"{name}_deviation"] / adaptive["flux_resolution"]
        )
    if isinstance(record.get("ladder_seconds"), float) and isinstance(
        record.get("newton_seconds"), float
    ):
        record["speedup"] = record["ladder_seconds"] / record["newton_seconds"]
    # the two effects separate: what removing the sequential traversal buys on
    # each route, and what lagging the topology buys at each read
    for name, slow, fast in (
        ("read_vectorisation_on_newton", "newton_shipped_read", "newton"),
        ("read_vectorisation_on_ladder", "ladder", "ladder_kernel_read"),
        ("lagged_topology_on_shipped_read", "ladder", "newton_shipped_read"),
        ("lagged_topology_on_kernel_read", "ladder_kernel_read", "newton"),
    ):
        numerator = record.get(f"{slow}_seconds")
        denominator = record.get(f"{fast}_seconds")
        if isinstance(numerator, float) and isinstance(denominator, float):
            record[name] = numerator / denominator
    record["flux_scale"] = scale
    return record


def measure_batch(bundle: SolveBundle, operator, adaptive, widths) -> list[dict]:
    """Return batched throughput of both routes against ensemble width."""
    import jax
    import jax.numpy as jnp

    outer = min(max(adaptive["outer_reads"], 2), OUTER_LIMIT)
    inner = max(
        (entry["inner_steps"] for entry in adaptive["outer"]), default=INNER_STEPS
    )
    seed = jnp.asarray(bundle.seed)
    routes = {
        "ladder": solve_callable(operator),
        "newton": newton_callable(
            operator, adaptive["support_size"], outer=outer, inner=inner
        ),
    }
    points = []
    for members in widths:
        entry: dict[str, Any] = {"members": members}
        current = jnp.asarray(ensemble_current(bundle, members))
        seeds = jnp.broadcast_to(seed, (members, seed.shape[0]))
        for name, route in routes.items():
            try:
                compiled = jax.jit(jax.vmap(route)).lower(current, seeds).compile()
                timing = time_call(
                    lambda c=compiled: c(current, seeds),
                    3,
                    1,  # noqa: B023
                )
                entry[f"{name}_seconds"] = timing["median"]
                entry[f"{name}_solves_per_second"] = members / timing["median"]
            except Exception as error:  # noqa: BLE001 - an exhausted device is a datum
                entry[f"{name}_failed"] = f"{type(error).__name__}: {str(error)[:160]}"
        entry["peak_bytes"] = peak_bytes()
        points.append(entry)
        print(
            "batch %d ladder %s newton %s"
            % (
                members,
                entry.get("ladder_solves_per_second", entry.get("ladder_failed")),
                entry.get("newton_solves_per_second", entry.get("newton_failed")),
            ),
            flush=True,
        )
    return points


def ladder_trace(bundle: SolveBundle, operator) -> list[float]:
    """Return the shipped ladder's own residual trace on the same seed."""
    import jax.numpy as jnp

    result = solve_callable(operator)(
        jnp.asarray(bundle.coil_current), jnp.asarray(bundle.seed)
    )
    return [float(value) for value in np.asarray(result.trace)]


# --------------------------------------------------------------------------
# assembly of the measurement records
# --------------------------------------------------------------------------
def run_stages(arguments) -> dict[str, Any]:
    """Return the stage-resolved record of one topology read."""
    from nova.jax.config import configure_dtypes

    configure_dtypes()

    import jax.numpy as jnp

    bundle = SolveBundle.load(Path(arguments.bundle))
    operator = build_operator(bundle)
    record = {
        "command": "stages",
        "bundle": Path(arguments.bundle).name,
        "cells": bundle.cells,
        "nodes": bundle.node_number,
        "label": arguments.label,
        "device": device_report(),
    }
    record["identity"] = check_read_identity(
        operator, jnp.asarray(bundle.reference_flux)
    )
    print("read identity", record["identity"], flush=True)
    record["stages"] = measure_stages(bundle, operator)
    for name, value in sorted(
        record["stages"]["stage"].items(), key=lambda item: -item[1]
    ):
        print("  %-26s %10.3f us" % (name, 1.0e6 * value), flush=True)
    return record


def run_newton(arguments) -> dict[str, Any]:
    """Return the record of the lagged-topology Newton prototype."""
    from nova.jax.config import configure_dtypes

    configure_dtypes()

    bundle = SolveBundle.load(Path(arguments.bundle))
    operator = build_operator(bundle)
    record = {
        "command": "newton",
        "bundle": Path(arguments.bundle).name,
        "cells": bundle.cells,
        "nodes": bundle.node_number,
        "label": arguments.label,
        "device": device_report(),
        "newton_steps": NEWTON_STEPS,
        "krylov_iterations": KRYLOV_ITERATIONS,
        "warmup": WARMUP_SWEEPS,
    }
    adaptive = adaptive_newton(bundle, operator)
    flux = adaptive.pop("flux")
    record["adaptive"] = adaptive
    print(
        "adaptive: %d reads, inner %s, deviation %.3e (%.1f floor steps), parity %s"
        % (
            adaptive["outer_reads"],
            [entry["inner_steps"] for entry in adaptive["outer"]],
            adaptive["flux_deviation"],
            adaptive["deviation_in_resolution_steps"],
            adaptive["parity"],
        ),
        flush=True,
    )
    record["jacobian"] = measure_jacobian_cost(bundle, operator)
    print(
        "jacobian",
        {k: v for k, v in record["jacobian"].items() if isinstance(v, float)},
        flush=True,
    )
    record["end_to_end"] = measure_end_to_end(bundle, operator, adaptive)
    print("end to end", record["end_to_end"], flush=True)
    record["ladder_trace"] = ladder_trace(bundle, operator)
    if arguments.batch:
        widths = tuple(int(item) for item in arguments.batch.split(","))
        record["batch"] = measure_batch(bundle, operator, adaptive, widths)
    record["newton_flux_deviation"] = float(
        np.max(np.abs(flux - bundle.reference_flux))
    )
    return record


# --------------------------------------------------------------------------
# the figure
# --------------------------------------------------------------------------
def render_figure(stages: list[dict], newton: list[dict], output: Path) -> None:
    """Write the convergence ladder, the read attribution and the wall times."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    left, middle, right = axes
    ladder_colour = "#c44e52"
    newton_colour = "#4c72b0"

    # ---- convergence against the currency both routes spend --------------
    # The x-axis is topology reads rather than iterations, because that is the
    # stage the whole solve is priced in: the ladder spends one on each primal
    # map evaluation and none on a tangent pass, while every Newton step inside
    # one outer read spends none at all.
    reference = max(newton, key=lambda item: item["cells"], default=None)
    if reference is not None:
        trace = np.asarray(reference["ladder_trace"], dtype=float)
        measured = trace[np.isfinite(trace)]
        left.semilogy(
            np.arange(1, measured.size + 1),
            measured,
            "o-",
            color=ladder_colour,
            markersize=3.4,
            linewidth=1.3,
            label="Krylov ladder",
        )
        newton_trace = np.asarray(reference["adaptive"]["trace"], dtype=float)
        position = []
        taken = 0
        for read, entry in enumerate(reference["adaptive"]["outer"]):
            steps = entry["inner_steps"]
            position.extend(read + (step + 1) / steps for step in range(steps))
            taken += steps
        position = np.asarray(position[: newton_trace.size])
        left.semilogy(
            position,
            newton_trace,
            "s-",
            color=newton_colour,
            markersize=4.4,
            linewidth=1.4,
            label="lagged-topology Newton",
        )
        for read in range(1, reference["adaptive"]["outer_reads"]):
            left.axvline(read, color=newton_colour, linewidth=0.7, alpha=0.3)
        left.annotate(
            "%d reads,\n%d Newton steps"
            % (reference["adaptive"]["outer_reads"], taken),
            xy=(position[-1], newton_trace[-1]),
            xytext=(14.0, -4.0),
            textcoords="offset points",
            color=newton_colour,
            fontsize=8.5,
            arrowprops={"arrowstyle": "-", "color": newton_colour, "linewidth": 0.8},
        )
        left.annotate(
            "%d reads to the same floor" % measured.size,
            xy=(measured.size, measured[-1]),
            xytext=(-2.0, 16.0),
            textcoords="offset points",
            color=ladder_colour,
            fontsize=8.5,
            ha="right",
        )
        left.set_xlabel("topology reads consumed")
        left.set_ylabel("relative fixed-point residual")
        left.set_title(
            "convergence, %d cells" % reference["cells"], fontsize=10.5, loc="left"
        )
        left.legend(frameon=False, fontsize=8.5, loc="upper right")

    # ---- where the read goes, scan against kernel ------------------------
    # Read off the smaller accelerator mesh: once one stage dominates by two
    # decades the differences of the others are the measurement error of the
    # two large chains they come from, and only the smaller mesh separates
    # every stage cleanly.
    accelerated = [
        item for item in stages if item.get("device", {}).get("platform") == "gpu"
    ]
    stage_reference = min(
        accelerated or stages, key=lambda item: item["cells"], default=None
    )
    if stage_reference is not None:
        marginal = stage_reference["stages"]["marginal"]
        rows = (
            ("zero-crossing\ncount", "crossing_count", "stencil_gather"),
            ("cluster\nselection", "cluster_selection", "crossing_count"),
            ("sub-cell\nfits", "subcell_fit", "cluster_selection"),
        )
        labels, scanned, kernelled = [], [], []
        for label, upper, lower in rows:
            base = marginal.get(f"{lower}_scan", marginal.get(lower, 0.0))
            base_kernel = marginal.get(f"{lower}_kernel", marginal.get(lower, 0.0))
            labels.append(label)
            scanned.append(1.0e6 * (marginal[f"{upper}_scan"] - base))
            kernelled.append(1.0e6 * (marginal[f"{upper}_kernel"] - base_kernel))
        labels.append("wall\nread")
        scanned.append(1.0e6 * marginal["wall_read"])
        kernelled.append(1.0e6 * marginal["wall_read"])
        labels.append("WHOLE\nREAD")
        scanned.append(1.0e6 * marginal["shipped_read"])
        kernelled.append(1.0e6 * marginal["kernel_read"])

        position = np.arange(len(labels))
        middle.barh(
            position - 0.19,
            scanned,
            height=0.36,
            color=ladder_colour,
            label="as shipped (sequential scan)",
        )
        middle.barh(
            position + 0.19,
            kernelled,
            height=0.36,
            color=newton_colour,
            label="vectorised (one kernel)",
        )
        for index, (slow, fast) in enumerate(zip(scanned, kernelled)):
            if fast > 0.0 and slow / fast > 1.5:
                middle.annotate(
                    "x%.0f" % (slow / fast),
                    xy=(slow, index - 0.19),
                    xytext=(4.0, -2.0),
                    textcoords="offset points",
                    fontsize=8.0,
                    color=ladder_colour,
                    fontweight="bold" if index == 0 else "normal",
                )
        middle.set_yticks(position)
        middle.set_yticklabels(labels, fontsize=8.0)
        middle.invert_yaxis()
        middle.set_xscale("log")
        middle.set_xlim(right=max(scanned) * 12.0)
        middle.set_xlabel("microseconds per read")
        middle.set_title(
            "topology read on the H200, %d cells" % stage_reference["cells"],
            fontsize=10.5,
            loc="left",
        )
        middle.legend(frameon=False, fontsize=7.8, loc="center right")

    # ---- wall time: the decision panel -----------------------------------
    ordered = sorted(
        newton, key=lambda item: (item["device"]["platform"], item["cells"])
    )
    series = (
        ("ladder_seconds", "Krylov ladder, as shipped", ladder_colour),
        ("ladder_kernel_read_seconds", "ladder + vectorised read", "#dd8452"),
        ("newton_seconds", "lagged-topology Newton", newton_colour),
    )
    names, values = [], {key: [] for key, _, _ in series}
    for item in ordered:
        end = item["end_to_end"]
        if not all(isinstance(end.get(key), float) for key, _, _ in series):
            continue
        names.append(
            "%s\n%d cells"
            % ("H200" if item["device"]["platform"] == "gpu" else "host", item["cells"])
        )
        for key, _, _ in series:
            values[key].append(1.0e3 * end[key])
    if names:
        position = np.arange(len(names))
        width = 0.26
        for index, (key, label, colour) in enumerate(series):
            right.bar(
                position + (index - 1) * width,
                values[key],
                width=width,
                color=colour,
                label=label,
            )
        # the comparison that decides the verdict is the last two bars, not the
        # first and last: the read fix is available to either route
        for index in range(len(names)):
            fixed = values["ladder_kernel_read_seconds"][index]
            lagged = values["newton_seconds"][index]
            right.annotate(
                "x%.1f" % (fixed / lagged)
                if lagged < fixed
                else "x%.1f slower" % (lagged / fixed),
                xy=(position[index] + width, lagged),
                xytext=(0.0, 3.0),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=newton_colour,
                fontweight="bold",
            )
        right.set_xticks(position)
        right.set_xticklabels(names, fontsize=8.0)
        right.set_yscale("log")
        right.set_ylim(top=max(values["ladder_seconds"]) * 6.0)
        right.set_ylabel("milliseconds per solve")
        right.set_title("single solve, after compilation", fontsize=10.5, loc="left")
        right.legend(frameon=False, fontsize=7.6, loc="upper left")

    for axis in axes:
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
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

    for name in ("stages", "newton"):
        run = commands.add_parser(name)
        run.add_argument("--bundle", required=True)
        run.add_argument("--platform", default=None, choices=("cpu", "gpu", "cuda"))
        run.add_argument("--label", default="")
        run.add_argument("--output", required=True)
        if name == "newton":
            run.add_argument("--batch", default="")

    draw = commands.add_parser("figure")
    draw.add_argument("--stages", nargs="*", default=())
    draw.add_argument("--newton", nargs="+", required=True)
    draw.add_argument("--output", required=True)

    arguments = parser.parse_args(argv)

    if arguments.command == "figure":
        render_figure(
            [json.loads(Path(item).read_text()) for item in arguments.stages],
            [json.loads(Path(item).read_text()) for item in arguments.newton],
            Path(arguments.output),
        )
        return 0

    if arguments.platform:
        os.environ["JAX_PLATFORMS"] = (
            "cuda" if arguments.platform == "gpu" else arguments.platform
        )
    os.environ.setdefault("NOVA_COMPILATION_CACHE", "off")
    record = (
        run_stages(arguments)
        if arguments.command == "stages"
        else run_newton(arguments)
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, default=str))
    print("wrote", output, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
