"""Measure density forcing against in-cell flux sampling order."""

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
from nova.equilibrium.stencil_mesh import CellCurrentMoments, PROFILE_DENSITY_POWERS


OUTPUT = Path(__file__).resolve().parent
REFERENCE_PATH = Path("tests/test_equilibrium_forward_reference.py")
FORCING_PATH = Path("scripts/map_forcing_attribution/measure.py")
DECOMPOSITION_PATH = Path("scripts/forcing_residual_decomposition/measure.py")
GATE_PATH = Path("scripts/root_gate_attribution/measure_root_attribution.py")
TERMINAL_PATH = Path("scripts/root_gate_attribution/coarse-terminal-root.npz")
EXPECTED_CACHE_KEY = "746fbe1553c4b242"
EXPECTED_DENSITY_SHARE = 0.991762605972982
EXPECTED_DENSITY_SUP_WB = 1.252292371968979
TIMING_REPETITIONS = 10


def load_module(path: Path, name: str):
    """Load a repository measurement module without collecting its tests."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Read one compressed bank into detached arrays."""
    with np.load(path) as stored:
        return {name: stored[name] for name in stored.files}


def cubic_design(local):
    """Evaluate the complete total-degree-three flux basis."""
    radial, vertical = local[..., 0], local[..., 1]
    namespace = jnp if isinstance(local, jax.Array) else np
    return namespace.stack(
        [
            namespace.ones_like(radial),
            radial,
            vertical,
            radial**2,
            radial * vertical,
            vertical**2,
            radial**3,
            radial**2 * vertical,
            radial * vertical**2,
            vertical**3,
        ],
        axis=-1,
    )


def interpolation_diagnostics(profile_flux, exact_flux, profile, points):
    """Measure flux error and its nonlinear density image at profile points."""
    flux_error = np.asarray(profile_flux - exact_flux)
    radius = jnp.asarray(points[..., 0])
    exact_density = np.asarray(profile.current_density(radius, jnp.asarray(exact_flux)))
    represented_density = np.asarray(
        profile.current_density(radius, jnp.asarray(profile_flux))
    )
    density_error = represented_density - exact_density

    def bias(values):
        absolute_mean = float(np.mean(np.abs(values)))
        return {
            "signed_mean": float(np.mean(values)),
            "mean_absolute": absolute_mean,
            "absolute_mean_over_mean_absolute": float(
                abs(np.mean(values)) / max(absolute_mean, 1.0e-300)
            ),
            "positive_fraction": float(np.mean(values > 0.0)),
            "negative_fraction": float(np.mean(values < 0.0)),
            "rms": float(np.sqrt(np.mean(values**2))),
            "sup": float(np.max(np.abs(values))),
        }

    return {
        "flux_interpolation_error_wb": bias(flux_error),
        "density_error": bias(density_error),
    }


def build_image_evaluator(operator, stencil, support, selection, fit, design):
    """Return one fixed-shape sample-to-plasma-image contraction."""
    ring = jnp.asarray(stencil.ring_centre)
    profile = operator.source.core
    profile_point = jnp.asarray(stencil.ring_profile_point)
    profile_weight = jnp.asarray(stencil.ring_profile_weight)
    sampling_centre = jnp.asarray(stencil.ring_sampling_centre)
    coordinate_scale = jnp.asarray(stencil.ring_coordinate_scale)
    moment_centre = jnp.asarray(support.centroids)[ring]
    vertices = jnp.asarray(support.support_vertices)[ring]
    vertex_count = jnp.asarray(support.vertex_count)[ring]
    selected = jnp.asarray(selection)[ring]
    fit_weight = None if fit is None else jnp.asarray(fit)
    profile_design = None if design is None else jnp.asarray(design)

    def evaluate(sample_flux):
        if fit_weight is None:
            profile_flux = sample_flux
        else:
            coefficient = jnp.einsum("nps,ns->np", fit_weight, sample_flux)
            profile_flux = jnp.einsum("nqp,np->nq", profile_design, coefficient)
        density = profile.current_density(profile_point[..., 0], profile_flux)
        polynomial = jnp.einsum("niq,nq->ni", profile_weight, density)
        current, first_sampling = padded_polynomial_current_moments(
            vertices,
            vertex_count,
            sampling_centre,
            coordinate_scale,
            polynomial,
            PROFILE_DENSITY_POWERS,
        )
        first = first_sampling + current[:, None] * (sampling_centre - moment_centre)
        entries = jnp.stack([current, first[:, 0], first[:, 1]])
        entries = jnp.where(selected[None, :], entries, 0.0)
        physical = jnp.zeros((3, operator.grid.node_number), dtype=entries.dtype)
        physical = physical.at[:, ring].set(entries)
        coupled = operator.coupling_current_moments(CellCurrentMoments(*physical))
        return jnp.r_[
            operator.grid.internal(coupled),
            operator.wall.internal(coupled),
            operator.sample.internal(coupled),
        ]

    return evaluate


def benchmark(evaluate, sample_flux):
    """Compile and time one device-resident fixed-shape arm."""
    compiled = jax.jit(evaluate)
    argument = jnp.asarray(sample_flux)
    started = perf_counter()
    image = compiled(argument)
    jax.block_until_ready(image)
    compile_seconds = perf_counter() - started
    samples = []
    for _ in range(TIMING_REPETITIONS):
        started = perf_counter()
        image = compiled(argument)
        jax.block_until_ready(image)
        samples.append(perf_counter() - started)
    return np.asarray(image), {
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "compile_and_first_evaluation_seconds": compile_seconds,
        "steady_repetitions": TIMING_REPETITIONS,
        "steady_median_microseconds": 1.0e6 * statistics.median(samples),
        "steady_minimum_microseconds": 1.0e6 * min(samples),
        "steady_maximum_microseconds": 1.0e6 * max(samples),
        "host_to_device_transfer_included": False,
    }


def render(report: dict[str, object]) -> None:
    """Plot forcing, linear response, interpolation, and evaluation cost."""
    order = [
        "seven_sample_quadratic",
        "thirteen_sample_cubic",
        "exact_flux_at_density_samples",
    ]
    labels = ["7 quadratic", "13 cubic", "exact at 672"]
    arms = report["arms"]
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)

    forcing = [arms[name]["forcing"]["sup_wb"] for name in order]
    axes[0, 0].bar(labels, forcing, color=["#4c78a8", "#72b7b2", "#f58518"])
    axes[0, 0].set_ylabel("density forcing sup [Wb]")
    axes[0, 0].set_title("Density forcing ladder")

    axis = [arms[name]["root_response"]["axis_signed_projection_mm"] for name in order]
    flux = [
        arms[name]["root_response"]["flux_signed_peak_percent_of_span"]
        for name in order
    ]
    x = np.arange(len(order))
    axes[0, 1].bar(x - 0.18, axis, 0.36, label="axis [mm]")
    axes[0, 1].bar(x + 0.18, flux, 0.36, label="flux [% span]")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_title("Tangent-inverse response")
    axes[0, 1].legend()

    flux_error = [
        arms[name]["interpolation"]["flux_interpolation_error_wb"]["sup"]
        for name in order
    ]
    density_error = [
        arms[name]["interpolation"]["density_error"]["sup"] for name in order
    ]
    axes[1, 0].semilogy(labels, flux_error, "o-", label="flux [Wb]")
    twin = axes[1, 0].twinx()
    twin.semilogy(labels, density_error, "s--", color="#e45756", label="j [A/m²]")
    axes[1, 0].set_ylabel("flux interpolation sup [Wb]")
    twin.set_ylabel("density sample error sup [A/m²]")
    axes[1, 0].set_title("Interpolation and nonlinear image")
    lines = axes[1, 0].lines + twin.lines
    axes[1, 0].legend(lines, [line.get_label() for line in lines])

    cost = [arms[name]["cost"]["steady_median_microseconds"] for name in order]
    sample_count = [arms[name]["cost"]["flux_samples_per_cell"] for name in order]
    axes[1, 1].bar(labels, cost, color=["#4c78a8", "#72b7b2", "#f58518"])
    axes[1, 1].set_ylabel("device-resident median [µs]")
    axes[1, 1].set_title("Evaluation cost")
    for position, count in enumerate(sample_count):
        axes[1, 1].text(
            position,
            cost[position],
            f"{count} samples",
            ha="center",
            va="bottom",
        )

    figure.savefig(OUTPUT / "density-forcing-ladder.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """Run the coarse in-cell representation ladder and bank its evidence."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = load_module(REFERENCE_PATH, "density_ladder_reference")
    forcing_module = load_module(FORCING_PATH, "density_ladder_forcing")
    decomposition = load_module(DECOMPOSITION_PATH, "density_ladder_decomposition")
    gate_module = load_module(GATE_PATH, "density_ladder_gate")
    reference.configure_dtypes()
    case = reference.require_reference()
    terminal = load_npz(TERMINAL_PATH)["state"]

    reference.WALL_NODES = 3
    machine = reference.cached_machine(case, reference.SUITE_CELLS, passive=True)
    receipt = machine.cache_receipt
    if receipt is None or not receipt.hit or receipt.key != EXPECTED_CACHE_KEY:
        raise AssertionError(
            "coarse machine did not warm-load from the semantic carrier"
        )
    operator = reference.forward_operator(case, machine)
    exact = reference.seed_flux(case, machine)
    if len(exact) != len(terminal):
        raise AssertionError("coarse exact and terminal states differ in shape")
    print(reference.machine_cache_summary("coarse", machine), flush=True)

    print(f"LINEARIZE state_size={len(exact)}", flush=True)
    mapped, tangent = jax.linearize(operator.flux_map(), exact)
    jax.block_until_ready(mapped)
    forcing = np.asarray(mapped - exact)
    coupled = operator.cell_current_moments(exact)
    second_residual, _actual, _reconstructed = forcing_module.second_moment_residual(
        case, operator, exact, coupled
    )
    print("COMPARATOR contracting omitted second moments", flush=True)
    correction, active_cells, pair_count = forcing_module.second_order_correction(
        machine, second_residual, forcing_module.HESSIAN_STEPS[-1]
    )
    comparator = forcing + correction

    partition = operator._support_partition(exact)
    masks, topology, sample_norm, support, _common_support = partition
    selection = np.asarray(masks.core | masks.common_sol)
    if len(operator._support_moment_stencils) != 1:
        raise AssertionError("coarse carrier no longer has one six-vertex stencil")
    stencil = operator._support_moment_stencils[0]
    if stencil.ring_profile_point.shape[1] != 672:
        raise AssertionError("coarse density projection sample count changed")

    value_pool = np.concatenate([np.asarray(masks.psi_norm), np.asarray(sample_norm)])
    seven_flux = value_pool[stencil.ring_gather_index]
    seven_fit = np.asarray(stencil.ring_flux_weight)
    quadratic_design = np.asarray(stencil.ring_profile_flux_design)
    seven_profile_flux = np.einsum(
        "nqp,np->nq", quadratic_design, np.einsum("nps,ns->np", seven_fit, seven_flux)
    )

    centres = np.asarray(stencil.ring_sampling_centre)
    vertices = np.stack(operator.moment_geometry.sampling_vertices)[stencil.ring_centre]
    midpoints = 0.5 * (vertices + np.roll(vertices, -1, axis=1))
    thirteen_points = np.concatenate([centres[:, None, :], vertices, midpoints], axis=1)
    scale = np.asarray(stencil.ring_coordinate_scale)
    thirteen_local = (thirteen_points - centres[:, None, :]) / scale[:, None, :]
    thirteen_fit = np.linalg.pinv(cubic_design(thirteen_local))
    profile_local = (
        np.asarray(stencil.ring_profile_point) - centres[:, None, :]
    ) / scale[:, None, :]
    profile_cubic_design = cubic_design(profile_local)
    axis_flux = float(topology.axis_flux)
    flux_span = float(topology.flux_span)
    thirteen_flux = (
        case.flux(thirteen_points[..., 0], thirteen_points[..., 1]) - axis_flux
    ) / flux_span
    thirteen_profile_flux = np.einsum(
        "nqp,np->nq",
        profile_cubic_design,
        np.einsum("nps,ns->np", thirteen_fit, thirteen_flux),
    )
    profile_points = np.asarray(stencil.ring_profile_point)
    exact_profile_flux = (
        case.flux(profile_points[..., 0], profile_points[..., 1]) - axis_flux
    ) / flux_span

    definitions = {
        "seven_sample_quadratic": {
            "sample_flux": seven_flux,
            "fit": seven_fit,
            "design": quadratic_design,
            "profile_flux": seven_profile_flux,
            "samples_per_cell": 7,
            "degree": 2,
        },
        "thirteen_sample_cubic": {
            "sample_flux": thirteen_flux,
            "fit": thirteen_fit,
            "design": profile_cubic_design,
            "profile_flux": thirteen_profile_flux,
            "samples_per_cell": 13,
            "degree": 3,
        },
        "exact_flux_at_density_samples": {
            "sample_flux": exact_profile_flux,
            "fit": None,
            "design": None,
            "profile_flux": exact_profile_flux,
            "samples_per_cell": 672,
            "degree": None,
        },
    }

    _support, exact_current, exact_first, _coupled = gate_module.support_reference(
        case, operator, exact, masks, topology
    )
    exact_image = np.asarray(operator.external()) + decomposition.internal_flux(
        operator, exact_current, exact_first
    )
    external = np.asarray(operator.external())
    arms = {}
    components = {}
    for name, definition in definitions.items():
        print(f"ARM name={name}", flush=True)
        evaluator = build_image_evaluator(
            operator,
            stencil,
            support,
            selection,
            definition["fit"],
            definition["design"],
        )
        internal, timing = benchmark(evaluator, definition["sample_flux"])
        image = external + internal
        component = image + correction - exact_image
        components[name] = component
        forcing_score = decomposition.score(component, comparator)
        step, solve = forcing_module.solve_response(tangent, component)
        response = decomposition.response_observables(
            case, machine, operator, exact, terminal, step
        )
        sample_count = definition["samples_per_cell"]
        timing.update(
            {
                "flux_samples_per_cell": sample_count,
                "incremental_samples_per_cell_over_production": sample_count - 7,
                "total_flux_sample_values": sample_count * len(stencil.ring_centre),
                "cost_scope": (
                    "device-resident sample-to-density-moments-to-coupling image; "
                    "sample generation and host-device transfer excluded"
                ),
            }
        )
        arms[name] = {
            "flux_polynomial_total_degree": definition["degree"],
            "forcing": forcing_score,
            "forcing_share_percent": 100.0 * forcing_score["projection_fraction"],
            "root_response": {**solve, **response},
            "interpolation": interpolation_diagnostics(
                definition["profile_flux"],
                exact_profile_flux,
                operator.source.core,
                profile_points,
            ),
            "cost": timing,
        }

    production_cost = arms["seven_sample_quadratic"]["cost"][
        "steady_median_microseconds"
    ]
    for arm in arms.values():
        arm["cost"]["steady_cost_ratio_over_production"] = (
            arm["cost"]["steady_median_microseconds"] / production_cost
        )

    production = arms["seven_sample_quadratic"]
    production_component = components["seven_sample_quadratic"]
    reconstructed_internal = production_component - correction + exact_image - external
    bank_difference = {
        "density_projection_fraction": production["forcing"]["projection_fraction"]
        - EXPECTED_DENSITY_SHARE,
        "density_forcing_sup_wb": production["forcing"]["sup_wb"]
        - EXPECTED_DENSITY_SUP_WB,
        "production_image_sup_wb": float(
            np.max(np.abs((np.asarray(mapped) - external) - reconstructed_internal))
        ),
    }
    if max(abs(value) for value in bank_difference.values()) > 1.0e-11:
        raise AssertionError(
            f"production ladder arm did not reproduce: {bank_difference}"
        )

    exact_ratio = (
        arms["exact_flux_at_density_samples"]["forcing"]["sup_wb"]
        / production["forcing"]["sup_wb"]
    )
    cubic_ratio = (
        arms["thirteen_sample_cubic"]["forcing"]["sup_wb"]
        / production["forcing"]["sup_wb"]
    )
    production_flux_bias = production["interpolation"]["flux_interpolation_error_wb"][
        "absolute_mean_over_mean_absolute"
    ]
    production_density_bias = production["interpolation"]["density_error"][
        "absolute_mean_over_mean_absolute"
    ]
    interpolation_dominant = exact_ratio < 0.25
    rectification = production_density_bias > production_flux_bias + 0.1
    dominant_stage = (
        "in_cell_flux_interpolation"
        if interpolation_dominant
        else "density_fit_projection_after_exact_flux_sampling"
    )
    mechanism = {
        "dominant_stage": dominant_stage,
        "exact_flux_forcing_sup_fraction_of_production": exact_ratio,
        "cubic_forcing_sup_fraction_of_production": cubic_ratio,
        "production_flux_error_bias_fraction": production_flux_bias,
        "production_density_error_bias_fraction": production_density_bias,
        "nonlinear_rectification_detected": rectification,
        "classification_policy": (
            "Interpolation owns the flat term when exact flux at every density "
            "sample leaves less than 25 percent of production forcing. Nonlinear "
            "rectification is detected when density-error bias exceeds flux-error "
            "bias by more than 0.1."
        ),
        "statement": (
            "The exact-flux arm removes the in-cell interpolant while retaining "
            "the degree-nine density fit, exact clipped moments, topology, and "
            "coupling. Its residual fraction therefore assigns the flat term "
            "between interpolation and the downstream fit-and-projection chain."
        ),
    }

    report = {
        "schema": "nova.density-forcing-ladder",
        "fixture": {
            "plasma_cells": len(machine.node),
            "state_size": len(exact),
            "support_cells": int(
                np.count_nonzero(np.asarray(support.vertex_count) >= 3)
            ),
            "cache": {
                "store": receipt.store,
                "semantic_key": receipt.key,
                "warm_hit": receipt.hit,
                "load_seconds": receipt.load_seconds,
                "arrays_verified": receipt.arrays_verified,
                "bytes_verified": receipt.bytes_verified,
            },
            "second_moment_comparator": {
                "active_cells": active_cells,
                "near_source_target_pairs": pair_count,
                "correction_sup_wb": float(np.max(np.abs(correction))),
            },
        },
        "held_fixed": [
            "exact analytic state and production tangent inverse",
            "production straight clipped supports and topology qualification",
            "degree-nine weighted density projection",
            "exact clipped zeroth and first moments",
            "first-moment coupling blocks and second-moment comparator correction",
        ],
        "sample_sets": {
            "seven_sample_quadratic": "cell centroid and six pre-clip vertices",
            "thirteen_sample_cubic": (
                "cell centroid, six pre-clip vertices, and six edge midpoints"
            ),
            "exact_flux_at_density_samples": (
                "exact analytic normalised flux at all 672 fixed density-projection "
                "quadrature points per cell"
            ),
        },
        "cost_qualification": (
            "CPU medians price the common device-resident density projection, "
            "exact support moments, and coupling image. The wide min-to-max range "
            "shows shared-node scheduling noise, so the three approximately equal "
            "medians are not a resolved speed ordering. Additional flux-value "
            "generation is excluded from timing and priced explicitly by the "
            "per-cell sample counts."
        ),
        "arms": arms,
        "production_control_difference": bank_difference,
        "mechanism": mechanism,
        "verdict": {
            "density_forcing_falls_with_cubic_sampling": cubic_ratio < 1.0,
            "flat_term_carrier_stage": dominant_stage,
            "measurement_only": True,
        },
    }
    render(report)
    report["artifacts"] = {
        "figure": str(OUTPUT / "density-forcing-ladder.png"),
        "figure_bytes": (OUTPUT / "density-forcing-ladder.png").stat().st_size,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"RESULT production_sup={production['forcing']['sup_wb']:.9g} "
        f"cubic_ratio={cubic_ratio:.9g} exact_ratio={exact_ratio:.9g} "
        f"stage={dominant_stage}",
        flush=True,
    )
    print("DENSITY_FORCING_LADDER_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
