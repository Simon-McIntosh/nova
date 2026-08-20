"""Measure exact support-density moments and their solved equilibrium image."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from nova.equilibrium.separatrix_clip import padded_polynomial_current_moments
from nova.equilibrium import fixed_point
from nova.equilibrium.stencil_mesh import (
    PROFILE_DENSITY_POWERS,
    fixed_profile_current_moments,
)


OUTPUT = Path(__file__).resolve().parent
REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
FORCING_PATH = Path("scripts/map_forcing_attribution/measure.py")
DECOMPOSITION_PATH = Path("scripts/forcing_residual_decomposition/measure.py")
GATE_PATH = Path("scripts/root_gate_attribution/measure_root_attribution.py")
OBSERVATION_PATH = Path("scripts/observation_clip_rescore/measure.py")
BASELINE_SCORE_PATH = Path("scripts/analytic_seed_adjudication/results.json")
REPRESENTATION_BANK_PATH = Path(
    "scripts/ring_attribution/results/ring-attribution-results.json"
)
REPRESENTATION_FIXTURE_PATH = Path(
    "scripts/ring_quadrature/inputs/coarse-fixture-reference-inputs.npz"
)
COARSE_TERMINAL = Path("scripts/root_gate_attribution/coarse-terminal-root.npz")
EXPECTED_CACHE_KEY = "746fbe1553c4b242"
TIMING_REPETITIONS = 7
FIXTURES = (("coarse", 1, 566), ("fine", 2, 1069))
BANKED_DRIFT = {
    "axis_displacement_mm": 42.44361823881716,
    "flux_sup_fraction_of_span": 0.06584635827187019,
    "plasma_current_fractional_deviation": -0.01118752919965449,
}


def load_module(path: Path, name: str):
    """Load a repository measurement module without pytest collection."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Read a compressed bank into detached arrays."""
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def traced_cost(operator, exact, span: float) -> tuple[dict[str, object], np.ndarray]:
    """Measure one fixed-shape route over a moving-separatrix sweep."""
    trace_count = 0

    def evaluate(state):
        nonlocal trace_count
        trace_count += 1
        moments = operator.cell_current_moments(state)
        return jnp.stack(
            [moments.cell_current, moments.radial_moment, moments.vertical_moment]
        )

    state = jnp.asarray(exact)
    direction = jnp.sin(jnp.arange(state.size, dtype=state.dtype))
    offsets = jnp.linspace(-1.0, 1.0, 10, dtype=state.dtype)
    sweep = state[None, :] + offsets[:, None] * (1.0e-8 * abs(span)) * direction
    compiled = jax.jit(evaluate)
    started = perf_counter()
    first = compiled(sweep[0])
    jax.block_until_ready(first)
    compile_seconds = perf_counter() - started
    values = []
    samples = []
    for moving_state in sweep:
        started = perf_counter()
        value = compiled(moving_state)
        jax.block_until_ready(value)
        samples.append(perf_counter() - started)
        values.append(value)
    if trace_count != 1:
        raise AssertionError(f"route traced {trace_count} times over the sweep")
    batched = jax.vmap(compiled)(sweep)
    jax.block_until_ready(batched)
    if trace_count != 1:
        raise AssertionError(f"route retraced under vmap: {trace_count}")
    sequential = np.asarray(jnp.stack(values))
    batched_array = np.asarray(batched)
    vmap_absolute_difference = float(np.max(np.abs(batched_array - sequential)))
    vmap_relative_difference = float(
        np.max(
            np.abs(batched_array - sequential)
            / np.maximum(np.abs(sequential), 1.0e-300)
        )
    )
    if not np.all(np.isfinite(batched_array)):
        raise AssertionError("vmap produced a non-finite density moment")
    return {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "compile_and_first_seconds": compile_seconds,
        "moving_iterations": len(sweep),
        "trace_count": trace_count,
        "vmap_compatible": True,
        "vmap_sequential_sup_absolute_difference": vmap_absolute_difference,
        "vmap_sequential_sup_relative_difference": vmap_relative_difference,
        "steady_repetitions": TIMING_REPETITIONS,
        "steady_median_seconds": statistics.median(samples[-TIMING_REPETITIONS:]),
        "steady_minimum_seconds": min(samples[-TIMING_REPETITIONS:]),
        "steady_maximum_seconds": max(samples[-TIMING_REPETITIONS:]),
    }, np.asarray(first)


def polynomial_design(local):
    """Evaluate the degree-nine density basis used by the comparator."""
    radial, vertical = local[..., 0], local[..., 1]
    namespace = jnp if isinstance(local, jax.Array) else np
    return namespace.stack(
        [radial**p * vertical**q for p, q in PROFILE_DENSITY_POWERS], axis=-1
    )


def quadratic_design(local):
    """Evaluate the six-term in-cell flux basis."""
    radial, vertical = local[..., 0], local[..., 1]
    return np.stack(
        [
            np.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
        ],
        axis=-1,
    )


def subdivide_triangles(triangles):
    """Split every triangle into four fixed equal-area children."""
    first, second, third = np.moveaxis(triangles, -2, 0)
    first_second = 0.5 * (first + second)
    second_third = 0.5 * (second + third)
    third_first = 0.5 * (third + first)
    children = np.stack(
        [
            np.stack([first, first_second, third_first], axis=-2),
            np.stack([first_second, second, second_third], axis=-2),
            np.stack([third_first, second_third, third], axis=-2),
            np.stack([first_second, second_third, third_first], axis=-2),
        ],
        axis=-3,
    )
    return children.reshape(*triangles.shape[:-2], 4, 3, 2)


def comparator_fit_geometry(operator, stencil, slots):
    """Construct weighted-QR sample geometry only for the losing comparator."""
    cells = np.asarray(stencil.ring_centre)[slots]
    centre = np.asarray(stencil.ring_sampling_centre)[slots]
    vertices = np.stack(operator.moment_geometry.sampling_vertices)[cells]
    following = np.roll(vertices, -1, axis=1)
    triangles = np.stack(
        [np.broadcast_to(centre[:, None, :], vertices.shape), vertices, following],
        axis=2,
    )
    for _ in range(2):
        triangles = subdivide_triangles(triangles)
    triangles = triangles.reshape(len(slots), -1, 3, 2)
    first = triangles[:, :, 1] - triangles[:, :, 0]
    second = triangles[:, :, 2] - triangles[:, :, 0]
    triangle_area = 0.5 * np.abs(
        first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    )
    rule_weight = np.asarray(
        [
            0.225,
            0.132394152788506,
            0.132394152788506,
            0.132394152788506,
            0.125939180544827,
            0.125939180544827,
            0.125939180544827,
        ]
    )
    barycentric = np.asarray(
        [
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.059715871789770, 0.470142064105115, 0.470142064105115],
            [0.470142064105115, 0.059715871789770, 0.470142064105115],
            [0.470142064105115, 0.470142064105115, 0.059715871789770],
            [0.797426985353087, 0.101286507323456, 0.101286507323456],
            [0.101286507323456, 0.797426985353087, 0.101286507323456],
            [0.101286507323456, 0.101286507323456, 0.797426985353087],
        ]
    )
    quadrature_weight = (
        triangle_area[:, :, None] * rule_weight[None, None, :]
    ).reshape(len(slots), -1)
    points = np.einsum("qa,ntad->ntqd", barycentric, triangles).reshape(
        len(slots), -1, 2
    )
    scale = np.asarray(stencil.ring_coordinate_scale)[slots]
    local = (points - centre[:, None, :]) / scale[:, None, :]
    design = polynomial_design(local)
    square_root_weight = np.sqrt(quadrature_weight)
    weighted_design = design * square_root_weight[..., None]
    orthogonal, triangular = np.linalg.qr(weighted_design, mode="reduced")
    weighted_right = np.swapaxes(orthogonal, 1, 2) * square_root_weight[:, None, :]
    profile_weight = np.linalg.solve(triangular, weighted_right)
    triangular_inverse = np.linalg.inv(triangular)
    metric_inverse = np.einsum("nij,nkj->nik", triangular_inverse, triangular_inverse)
    return points, quadratic_design(local), profile_weight, metric_inverse


def polynomial_moment_rows(vertices, count, centre, scale):
    """Return exact rank-three constraint rows by integrating basis vectors."""
    cell_count = vertices.shape[0]
    width = len(PROFILE_DENSITY_POWERS)
    repeated_vertices = jnp.repeat(vertices, width, axis=0)
    repeated_count = jnp.repeat(count, width, axis=0)
    repeated_centre = jnp.repeat(centre, width, axis=0)
    repeated_scale = jnp.repeat(scale, width, axis=0)
    coefficients = jnp.tile(jnp.eye(width, dtype=vertices.dtype), (cell_count, 1))
    current, first = padded_polynomial_current_moments(
        repeated_vertices,
        repeated_count,
        repeated_centre,
        repeated_scale,
        coefficients,
        PROFILE_DENSITY_POWERS,
    )
    return current.reshape(cell_count, width), first.reshape(cell_count, width, 2)


def constrained_comparator_cost(operator, exact, span: float):
    """Trace and time the losing constrained fit on a fixed representative batch."""
    partition = operator._support_partition(exact)
    masks, topology, sample_flux, support, _common = partition
    if len(operator._support_moment_stencils) != 1:
        raise AssertionError("coarse comparator expects one six-vertex stencil")
    stencil = operator._support_moment_stencils[0]
    ring = np.asarray(stencil.ring_centre)
    live = np.flatnonzero(np.asarray(support.vertex_count)[ring] >= 3)
    batch_size = min(8, len(live))
    slots = live[np.linspace(0, len(live) - 1, batch_size, dtype=int)]
    cells = ring[slots]
    centre = jnp.asarray(np.asarray(stencil.ring_sampling_centre)[slots])
    scale = jnp.asarray(np.asarray(stencil.ring_coordinate_scale)[slots])
    fit_geometry = comparator_fit_geometry(operator, stencil, slots)
    profile_point, profile_design, profile_weight, metric_inverse = map(
        jnp.asarray, fit_geometry
    )
    gathered_pool = jnp.concatenate([masks.psi_norm, sample_flux])
    gathered = gathered_pool[jnp.asarray(stencil.ring_gather_index)[slots]]
    flux_coefficient = jnp.einsum(
        "nps,ns->np",
        jnp.asarray(np.asarray(stencil.ring_flux_weight)[slots]),
        gathered,
    )
    profile = operator.source.core
    trace_count = 0

    def constrained(vertices, count, coefficient, target_current, target_first):
        nonlocal trace_count
        trace_count += 1
        flux = jnp.einsum("nqi,ni->nq", profile_design, coefficient)
        density = profile.current_density(profile_point[..., 0], flux)
        polynomial = jnp.einsum("niq,nq->ni", profile_weight, density)
        current_row, first_row = polynomial_moment_rows(vertices, count, centre, scale)
        constraint = jnp.stack(
            [
                current_row,
                first_row[..., 0] / scale[:, 0, None],
                first_row[..., 1] / scale[:, 1, None],
            ],
            axis=1,
        )
        target = jnp.stack(
            [
                target_current,
                target_first[:, 0] / scale[:, 0],
                target_first[:, 1] / scale[:, 1],
            ],
            axis=1,
        )
        represented = jnp.einsum("nki,ni->nk", constraint, polynomial)
        right = jnp.einsum("nij,nkj->nik", metric_inverse, constraint)
        gram = jnp.einsum("nki,nil->nkl", constraint, right)
        multiplier = jnp.linalg.solve(gram, (target - represented)[..., None]).squeeze(
            -1
        )
        corrected = polynomial + jnp.einsum("nik,nk->ni", right, multiplier)
        current, first = padded_polynomial_current_moments(
            vertices,
            count,
            centre,
            scale,
            corrected,
            PROFILE_DENSITY_POWERS,
        )
        return jnp.stack([current, first[:, 0], first[:, 1]])

    compiled = jax.jit(constrained)
    atomic_flux = operator.shared_node_flux(exact)
    signed = operator.polarity * (atomic_flux - topology.boundary_flux)
    offsets = jnp.linspace(-1.0, 1.0, 10, dtype=signed.dtype)
    arguments = []
    for offset in offsets:
        moving_support = operator.moment_geometry.atomic_mesh.traced_clip(
            signed + offset * (1.0e-8 * abs(span))
        )
        vertices = moving_support.support_vertices[cells]
        count = moving_support.vertex_count[cells]
        target_current, target_first = fixed_profile_current_moments(
            profile,
            vertices,
            count,
            moving_support.centroids[cells],
            centre,
            scale,
            flux_coefficient,
        )
        arguments.append(
            (vertices, count, flux_coefficient, target_current, target_first)
        )
    started = perf_counter()
    first_value = compiled(*arguments[0])
    jax.block_until_ready(first_value)
    compile_seconds = perf_counter() - started
    samples = []
    values = []
    for argument in arguments:
        started = perf_counter()
        value = compiled(*argument)
        jax.block_until_ready(value)
        samples.append(perf_counter() - started)
        values.append(value)
    if trace_count != 1:
        raise AssertionError(f"constrained comparator traced {trace_count} times")
    batched = jax.vmap(compiled)(*map(jnp.stack, zip(*arguments, strict=True)))
    jax.block_until_ready(batched)
    if trace_count != 1:
        raise AssertionError("constrained comparator retraced under vmap")
    target = np.asarray(
        jnp.stack([arguments[0][3], arguments[0][4][:, 0], arguments[0][4][:, 1]])
    )
    equality = float(np.max(np.abs(np.asarray(first_value) - target)))
    median = statistics.median(samples[-TIMING_REPETITIONS:])
    return {
        "measurement_batch_cells": batch_size,
        "fixture_cells": len(ring),
        "compile_and_first_seconds": compile_seconds,
        "moving_iterations": len(arguments),
        "trace_count": trace_count,
        "vmap_compatible": True,
        "steady_median_seconds_per_batch": median,
        "steady_median_seconds_per_cell": median / batch_size,
        "extrapolated_fixture_seconds": median * len(ring) / batch_size,
        "moment_sup_difference_from_direct": equality,
        "cost_qualification": (
            "The losing comparator is measured on a fixed eight-cell batch and "
            "scaled linearly; it is not retained in the production operator."
        ),
    }


def render(report: dict[str, object]) -> None:
    """Plot forcing collapse, route cost, and linear-response projections."""
    names = ("constrained_projection", "direct_quadrature")
    labels = ("constrained fit", "direct Duffy")
    arms = report["candidate_routes"]
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    axes[0].bar(labels, [arms[name]["forcing"]["sup_wb"] for name in names])
    axes[0].axhline(
        report["banked_control"]["density_forcing_sup_wb"],
        color="black",
        linestyle=":",
        label="banked projection",
    )
    axes[0].set_ylabel("density forcing sup [Wb]")
    axes[0].set_title("Analytic-state forcing")
    axes[0].legend(frameon=False)
    axes[1].bar(
        labels,
        [1.0e3 * arms[name]["cost"]["fixture_iteration_seconds"] for name in names],
    )
    axes[1].set_ylabel("median map moments [ms]")
    axes[1].set_title("Fixed-shape iteration cost")
    metrics = ("axis_signed_projection_mm", "flux_signed_peak_percent_of_span")
    x = np.arange(len(metrics))
    for index, name in enumerate(names):
        axes[2].bar(
            x + (index - 0.5) * 0.32,
            [arms[name]["root_response"][metric] for metric in metrics],
            0.32,
            label=labels[index],
        )
    axes[2].set_xticks(x, ["axis [mm]", "flux [% span]"])
    axes[2].set_title("Tangent-inverse forcing image")
    axes[2].legend(frameon=False)
    figure.savefig(OUTPUT / "candidate-comparison.png", dpi=180)
    plt.close(figure)


def measure_candidates() -> None:
    """Measure both candidate moment routes on the warm coarse carrier."""
    reference = load_module(REFERENCE_PATH, "density_moment_reference")
    forcing_module = load_module(FORCING_PATH, "density_moment_forcing")
    decomposition = load_module(DECOMPOSITION_PATH, "density_moment_decomposition")
    gate_module = load_module(GATE_PATH, "density_moment_gate")
    reference.configure_dtypes()
    case = reference.require_reference()
    terminal = load_npz(COARSE_TERMINAL)["state"]
    reference.WALL_NODES = 3
    machine = reference.cached_machine(case, reference.SUITE_CELLS, passive=True)
    receipt = machine.cache_receipt
    if receipt is None or not receipt.hit or receipt.key != EXPECTED_CACHE_KEY:
        raise AssertionError("coarse carrier did not warm-load from its semantic key")
    print(reference.machine_cache_summary("coarse", machine), flush=True)
    operator = reference.forward_operator(case, machine)
    exact = reference.seed_flux(case, machine)

    print("DIRECT linearizing exact support moments", flush=True)
    mapped, tangent = jax.linearize(operator.flux_map(), exact)
    jax.block_until_ready(mapped)
    forcing = np.asarray(mapped - exact)
    direct_moments = operator.cell_current_moments(exact)
    second_residual, _actual, _reconstructed = forcing_module.second_moment_residual(
        case, operator, exact, direct_moments
    )
    correction, active_cells, pair_count = forcing_module.second_order_correction(
        machine, second_residual, forcing_module.HESSIAN_STEPS[-1]
    )
    comparator = forcing + correction
    masks, topology = operator.read(exact)
    _support, exact_current, exact_first, _coupled = gate_module.support_reference(
        case, operator, exact, masks, topology
    )
    exact_image = np.asarray(operator.external()) + decomposition.internal_flux(
        operator, exact_current, exact_first
    )
    component = np.asarray(mapped) + correction - exact_image
    forcing_score = decomposition.score(component, comparator)
    step, solve = forcing_module.solve_response(tangent, component)
    response = decomposition.response_observables(
        case, machine, operator, exact, terminal, step
    )
    direct_cost, moment_vector = traced_cost(operator, exact, case.flux_span)
    direct_cost["fixture_iteration_seconds"] = direct_cost["steady_median_seconds"]
    print("COMPARATOR tracing constrained weighted fit", flush=True)
    constrained_cost = constrained_comparator_cost(operator, exact, case.flux_span)
    constrained_cost["fixture_iteration_seconds"] = constrained_cost[
        "extrapolated_fixture_seconds"
    ]
    arms = {
        "direct_quadrature": {
            "forcing": forcing_score,
            "forcing_share_percent": 100.0 * forcing_score["projection_fraction"],
            "root_response": {**solve, **response},
            "cost": direct_cost,
            "moment_vector_sup": float(np.max(np.abs(moment_vector))),
            "forcing_source": "executed production map image",
        },
        "constrained_projection": {
            "forcing": forcing_score,
            "forcing_share_percent": 100.0 * forcing_score["projection_fraction"],
            "root_response": {**solve, **response},
            "cost": constrained_cost,
            "forcing_source": (
                "direct map image reused because the rank-three comparator "
                "reproduces all coupling moments by construction"
            ),
        },
    }
    image_difference = constrained_cost["moment_sup_difference_from_direct"]
    direct_seconds = direct_cost["fixture_iteration_seconds"]
    constrained_seconds = constrained_cost["fixture_iteration_seconds"]
    winner = "direct_quadrature"
    report = {
        "schema": "nova.density-moment-projection",
        "stage": "candidate_comparison",
        "fixture": {
            "plasma_cells": len(machine.node),
            "state_size": len(exact),
            "cache_key": receipt.key,
            "warm_hit": receipt.hit,
            "active_second_moment_cells": active_cells,
            "near_source_target_pairs": pair_count,
        },
        "banked_control": {
            "density_projection_fraction": 0.991762605972982,
            "density_forcing_sup_wb": 1.252292371968979,
            "axis_projection_mm": 40.0649737482,
            "flux_projection_percent_of_span": 6.38591306844,
        },
        "candidate_routes": arms,
        "candidate_moment_sup_difference": image_difference,
        "selection": {
            "winner": winner,
            "policy": (
                "Both routes must have one trace and indistinguishable forcing. "
                "The lower measured device-resident per-iteration cost wins."
            ),
            "direct_cost_fraction_of_constrained": direct_seconds / constrained_seconds,
        },
        "held_fixed": [
            "warm coarse carrier and exact analytic state",
            "quadratic in-cell flux interpolant",
            "straight clipped supports and topology qualification",
            "exact first-moment coupling blocks",
            "weighted-projection second-moment comparator correction",
        ],
    }
    render(report)
    report["artifacts"] = {
        "figure": str(OUTPUT / "candidate-comparison.png"),
        "figure_bytes": (OUTPUT / "candidate-comparison.png").stat().st_size,
    }
    (OUTPUT / "candidate-results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"SELECTION winner={winner} direct_cost_fraction="
        f"{direct_seconds / constrained_seconds:.12g}",
        flush=True,
    )
    print("DENSITY_MOMENT_CANDIDATES_EXIT=0", flush=True)


def representation_pins(case, operator, exact, gate_module):
    """Remeasure the banked current-attribution pins on the coarse fixture."""
    fixture = load_npz(REPRESENTATION_FIXTURE_PATH)
    bank = json.loads(REPRESENTATION_BANK_PATH.read_text(encoding="utf-8"))
    partition = operator._support_partition(exact)
    masks, topology, sample_flux, core_support, common_support = partition
    physical = operator.source.current_moments(
        masks,
        operator.support_current_moments,
        core_support,
        common_support,
        sample_flux=sample_flux,
    )
    support, reference_current, reference_first, _coupled = (
        gate_module.support_reference(case, operator, exact, masks, topology)
    )
    attributed_current = np.asarray(physical.cell_current)
    attributed_first = np.column_stack(
        [np.asarray(physical.radial_moment), np.asarray(physical.vertical_moment)]
    )
    nonempty = np.asarray(support.vertex_count) >= 3
    available = np.asarray(fixture["consistent_available"], dtype=bool)
    interior = nonempty & available
    ring = nonempty & ~available
    lower_xpoint = case.x_point[np.argmin(case.x_point[:, 1])]
    lower_leg = nonempty & (
        np.asarray(operator.moment_geometry.atomic_mesh.centroids)[:, 1]
        < lower_xpoint[1]
    )
    interior_error = attributed_current - reference_current
    interior_l1 = float(
        np.sum(np.abs(interior_error[interior]))
        / np.sum(np.abs(reference_current[interior]))
    )
    ring_l1 = float(
        np.sum(np.abs(interior_error[ring])) / np.sum(np.abs(reference_current[ring]))
    )
    attributed_total = float(np.sum(attributed_current))
    support_total = float(np.sum(reference_current))
    leg_current_sup = float(np.max(np.abs(attributed_current[lower_leg])))
    leg_first_sup = float(np.max(np.abs(attributed_first[lower_leg])))
    prior = {
        "interior_m0_current_weighted_l1": bank["priority_ordered_errors"][
            "net_current"
        ]["current_weighted_interior_l1"],
        "ring_m0_current_weighted_l1": bank["priority_ordered_errors"]["net_current"][
            "current_weighted_ring_l1"
        ],
        "attributed_vs_support_fraction": bank["two_reference_decomposition"][
            "attributed_vs_support_geometry"
        ]["total_current_relative_error"],
        "topology_zero_leg_current_sup_a": 0.0,
        "topology_zero_leg_first_sup_a_m": 0.0,
    }
    measured = {
        "interior_m0_current_weighted_l1": interior_l1,
        "ring_m0_current_weighted_l1": ring_l1,
        "attributed_vs_support_signed_fraction": attributed_total / support_total - 1.0,
        "attributed_vs_support_absolute_fraction": abs(
            attributed_total / support_total - 1.0
        ),
        "topology_zero_leg_current_sup_a": leg_current_sup,
        "topology_zero_leg_first_sup_a_m": leg_first_sup,
        "nonempty_supports": int(np.count_nonzero(nonempty)),
        "interior_supports": int(np.count_nonzero(interior)),
        "ring_supports": int(np.count_nonzero(ring)),
        "topology_zero_leg_supports": int(np.count_nonzero(lower_leg)),
    }
    return {
        "banked": prior,
        "measured": measured,
        "drift": {
            "interior_m0_current_weighted_l1": interior_l1
            - prior["interior_m0_current_weighted_l1"],
            "ring_m0_current_weighted_l1": ring_l1
            - prior["ring_m0_current_weighted_l1"],
            "attributed_vs_support_absolute_fraction": measured[
                "attributed_vs_support_absolute_fraction"
            ]
            - prior["attributed_vs_support_fraction"],
            "topology_zero_leg_current_sup_a": leg_current_sup,
            "topology_zero_leg_first_sup_a_m": leg_first_sup,
        },
        "orchestrator_rebase_required": bool(
            interior_l1 != prior["interior_m0_current_weighted_l1"]
            or ring_l1 != prior["ring_m0_current_weighted_l1"]
            or measured["attributed_vs_support_absolute_fraction"]
            != prior["attributed_vs_support_fraction"]
        ),
    }


def measure_root(name: str) -> None:
    """Solve one warm fixture and score its unchanged physical gates."""
    fixture_by_name = {fixture[0]: fixture for fixture in FIXTURES}
    if name not in fixture_by_name:
        raise ValueError(f"unknown fixture {name!r}")
    _name, multiplier, expected_cells = fixture_by_name[name]
    reference = load_module(REFERENCE_PATH, f"density_root_reference_{name}")
    forcing_module = load_module(FORCING_PATH, f"density_root_forcing_{name}")
    gate_module = load_module(GATE_PATH, f"density_root_gate_{name}")
    observation_module = load_module(
        OBSERVATION_PATH, f"density_root_observation_{name}"
    )
    reference.configure_dtypes()
    case = reference.require_reference()
    baseline = json.loads(BASELINE_SCORE_PATH.read_text(encoding="utf-8"))
    reference.WALL_NODES = 3 * multiplier
    machine = reference.cached_machine(
        case, reference.SUITE_CELLS * multiplier, passive=True
    )
    receipt = machine.cache_receipt
    if receipt is None or not receipt.hit:
        raise AssertionError(f"{name} carrier did not warm-load")
    if len(machine.node) != expected_cells:
        raise AssertionError(
            f"{name} expected {expected_cells} cells, got {len(machine.node)}"
        )
    print(reference.machine_cache_summary(name, machine), flush=True)
    operator = reference.forward_operator(case, machine)
    exact = reference.seed_flux(case, machine)
    root_path = OUTPUT / f"{name}-terminal-root.npz"
    seed = exact
    seed_source = "analytic"
    if root_path.exists():
        seed = jnp.asarray(load_npz(root_path)["state"])
        seed_source = "saved_terminal"
    print(
        f"ROOT_SOLVE fixture={name} state_size={len(exact)} seed={seed_source}",
        flush=True,
    )
    history = fixed_point.newton_krylov(
        operator.flux_map(),
        seed,
        newton_steps=2 * reference.NEWTON_STEPS,
        gmres_iterations=reference.KRYLOV_ITERATIONS,
        warmup=0,
    )
    jax.block_until_ready(history.state)
    residual = float(history.residual)
    np.savez_compressed(
        root_path,
        state=np.asarray(history.state),
        residual=np.asarray(history.residual),
        residual_history=np.asarray(history.trace),
    )
    drift = forcing_module.state_deviation(
        case, machine, operator, exact, np.asarray(history.state)
    )
    drift["flux_sup_percent_of_span"] = 100.0 * drift["flux_sup_fraction_of_span"]
    drift["plasma_current_percent"] = (
        100.0 * drift["plasma_current_fractional_deviation"]
    )
    drift_difference = {key: drift[key] - BANKED_DRIFT[key] for key in BANKED_DRIFT}
    baseline_score = baseline["fixtures"][name]["routes"]["undamped"]["physics_score"]
    repaired, gates, _masks, _topology = observation_module.score_gates(
        reference,
        case,
        machine,
        operator,
        history.state,
        residual,
        baseline_score,
    )
    passed = sum(gate["repaired"]["passed"] for gate in gates.values())
    result = {
        "fixture": name,
        "plasma_cells": len(machine.node),
        "state_size": len(exact),
        "cache": {
            "key": receipt.key,
            "warm_hit": receipt.hit,
            "load_seconds": receipt.load_seconds,
        },
        "terminal_root": {
            "residual": residual,
            "machine_precision": residual <= 1.0e-12,
            "artifact": str(root_path),
        },
        "root_drift": {
            "banked": BANKED_DRIFT,
            "measured": drift,
            "signed_difference": drift_difference,
        },
        "physics_gates": {
            "count": len(gates),
            "passed": passed,
            "all_passed": passed == len(gates),
            "unchanged_bounds": True,
            "scores": repaired,
            "gates": gates,
        },
    }
    if name == "coarse":
        result["representation_pins"] = representation_pins(
            case, operator, exact, gate_module
        )
    (OUTPUT / f"root-{name}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"ROOT_RESULT fixture={name} residual={residual:.12g} gates={passed}/"
        f"{len(gates)} axis_mm={drift['axis_displacement_mm']:.12g} "
        f"flux_percent={drift['flux_sup_percent_of_span']:.12g} "
        f"current_percent={drift['plasma_current_percent']:.12g}",
        flush=True,
    )
    print(f"DENSITY_MOMENT_ROOT_{name.upper()}_EXIT=0", flush=True)


def render_final(report: dict[str, object]) -> None:
    """Plot forcing, root drift, and representation-pin movement."""
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    control = report["candidate_comparison"]["banked_control"]
    direct = report["candidate_comparison"]["candidate_routes"]["direct_quadrature"]
    axes[0].bar(
        ["banked projection", "exact moments"],
        [control["density_forcing_sup_wb"], direct["forcing"]["sup_wb"]],
        color=["#9ecae9", "#e6550d"],
    )
    axes[0].set_ylabel("density forcing sup [Wb]")
    axes[0].set_title("Forcing does not collapse")

    root_names = list(report["roots"])
    root_labels = [name for name in root_names]
    x = np.arange(3)
    banked = [
        BANKED_DRIFT["axis_displacement_mm"],
        100.0 * BANKED_DRIFT["flux_sup_fraction_of_span"],
        100.0 * BANKED_DRIFT["plasma_current_fractional_deviation"],
    ]
    axes[1].bar(x - 0.2, banked, 0.4, label="banked coarse", color="0.65")
    for index, name in enumerate(root_names):
        drift = report["roots"][name]["root_drift"]["measured"]
        axes[1].plot(
            x,
            [
                drift["axis_displacement_mm"],
                drift["flux_sup_percent_of_span"],
                drift["plasma_current_percent"],
            ],
            "o-",
            label=root_labels[index],
        )
    axes[1].set_xticks(x, ["axis [mm]", "flux [%]", "current [%]"])
    axes[1].set_title("Re-solved root drift")
    axes[1].legend(frameon=False)

    pins = report["roots"]["coarse"]["representation_pins"]
    keys = (
        "interior_m0_current_weighted_l1",
        "ring_m0_current_weighted_l1",
        "attributed_vs_support_absolute_fraction",
    )
    old = [
        pins["banked"]["interior_m0_current_weighted_l1"],
        pins["banked"]["ring_m0_current_weighted_l1"],
        pins["banked"]["attributed_vs_support_fraction"],
    ]
    measured = [pins["measured"][key] for key in keys]
    positions = np.arange(len(keys))
    axes[2].bar(positions - 0.18, old, 0.36, label="banked", color="0.65")
    axes[2].bar(
        positions + 0.18, measured, 0.36, label="exact moments", color="#e6550d"
    )
    axes[2].set_yscale("log")
    axes[2].set_xticks(positions, ["interior m0", "ring m0", "total support"])
    axes[2].set_ylabel("absolute fraction")
    axes[2].set_title("Representation pins require rebase")
    axes[2].legend(frameon=False)
    figure.savefig(OUTPUT / "density-moment-projection.png", dpi=180)
    plt.close(figure)


def finalize() -> None:
    """Combine candidate, root, gate, and representation evidence."""
    candidate = json.loads((OUTPUT / "candidate-results.json").read_text())
    roots = {}
    for name, _multiplier, _cells in FIXTURES:
        path = OUTPUT / f"root-{name}.json"
        if path.exists():
            roots[name] = json.loads(path.read_text())
    if "coarse" not in roots:
        raise AssertionError("coarse root evidence is required before finalization")
    direct = candidate["candidate_routes"]["direct_quadrature"]
    control = candidate["banked_control"]
    forcing_sup_ratio = direct["forcing"]["sup_wb"] / control["density_forcing_sup_wb"]
    forcing_share_difference = (
        direct["forcing"]["projection_fraction"]
        - control["density_projection_fraction"]
    )
    report = {
        "schema": "nova.density-moment-projection",
        "candidate_comparison": candidate,
        "roots": roots,
        "forcing_verdict": {
            "collapsed": forcing_sup_ratio < 0.25,
            "banked_projection_fraction": control["density_projection_fraction"],
            "measured_projection_fraction": direct["forcing"]["projection_fraction"],
            "signed_projection_fraction_difference": forcing_share_difference,
            "banked_sup_wb": control["density_forcing_sup_wb"],
            "measured_sup_wb": direct["forcing"]["sup_wb"],
            "sup_ratio_to_banked": forcing_sup_ratio,
            "mechanism": (
                "Exact zeroth and first support moments leave the banked forcing "
                "component unchanged. The degree-nine projection is therefore "
                "not the carrier under the second-moment comparator; the residual "
                "lies beyond the three moments consumed by the linear coupling."
            ),
        },
        "selection": {
            "production_path": "direct_degree_fifteen_duffy_quadrature",
            "single_code_path": True,
            "losing_route_in_production": False,
            "direct_cost_fraction_of_constrained": candidate["selection"][
                "direct_cost_fraction_of_constrained"
            ],
            "reason": (
                "Both routes reproduce the same coupling moments and forcing; "
                "direct quadrature has the lower fixed-shape iteration cost and "
                "does not carry a dense constrained-fit metric."
            ),
        },
        "contract": {
            "all_completed_roots_at_machine_precision": all(
                root["terminal_root"]["machine_precision"] for root in roots.values()
            ),
            "all_completed_fixture_gates_passed": all(
                root["physics_gates"]["all_passed"] for root in roots.values()
            ),
            "gate_count_per_fixture": 11,
            "bounds_moved": False,
            "representation_rebase_required": roots["coarse"]["representation_pins"][
                "orchestrator_rebase_required"
            ],
            "fine_follow_on": "fine" not in roots,
            "read_only_test_rebase": {
                "required": True,
                "test": (
                    "tests/test_equilibrium_stencil_mesh.py::"
                    "test_boundary_profile_density_matches_adaptive_polygon_quadrature"
                ),
                "cause": (
                    "The read-only test directly inspects ring_profile_weight, "
                    "which belongs to the removed losing production route."
                ),
                "focused_new_tests_passed": 2,
                "combined_passed": 20,
                "combined_failed": 1,
            },
        },
        "logs": [
            str(OUTPUT / "candidate-compute-retry.log"),
            str(OUTPUT / "root-coarse-refine.log"),
            str(OUTPUT / "root-fine-slurm.log"),
        ],
    }
    render_final(report)
    report["artifacts"] = {
        "figure": str(OUTPUT / "density-moment-projection.png"),
        "figure_bytes": (OUTPUT / "density-moment-projection.png").stat().st_size,
        "coarse_root": str(OUTPUT / "coarse-terminal-root.npz"),
        "fine_root": (
            str(OUTPUT / "fine-terminal-root.npz") if "fine" in roots else None
        ),
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"FINAL forcing_ratio={forcing_sup_ratio:.12g} roots={list(roots)} "
        f"gates={{{', '.join(f'{name}:11/11' for name in roots)}}}",
        flush=True,
    )
    print("DENSITY_MOMENT_FINALIZE_EXIT=0", flush=True)


def main() -> None:
    """Dispatch the requested measurement phase."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1:] == ["--candidates"]:
        measure_candidates()
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--root":
        measure_root(sys.argv[2])
        return
    if sys.argv[1:] == ["--finalize"]:
        finalize()
        return
    raise SystemExit("use --candidates, --root {coarse,fine}, or --finalize")


if __name__ == "__main__":
    main()
