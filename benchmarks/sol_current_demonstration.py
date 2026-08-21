"""Bank a confined and common-SOL equilibrium at identical plasma current.

The carrier is the warm, cached ITER diverted fixture used by the forward
equilibrium reference lane.  Its banked confined root is the control.  The
comparison source continues the same core gradients with the Eich closure and
solves flux and one globally declared source amplitude together.  The extra
scalar equation is the net-current identity, so neither current image is
silently rescaled after the equilibrium solve.

The output is deliberately a measurement bundle: roots and domain images in
NPZ, machine-readable metrics in JSON, and two SVG figures showing the
topology-qualified current profile and matched normalised-flux contours.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
import matplotlib.tri as mtri
import numpy as np

from nova.equilibrium import fixed_point
from nova.equilibrium.observation import current_ledger
from nova.equilibrium.separatrix_clip import AtomicCellMesh
from nova.equilibrium.sol_closure import (
    EichSolClosure,
    SolDecayVariant,
    eich_width,
)
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.jax.config import configure_dtypes
from tests import test_equilibrium_forward_reference as reference


DEFAULT_OUTPUT = Path("docs/figures/sol-current-demonstration")
ROOT_BANK = Path("scripts/root_gate_attribution/coarse-terminal-root.npz")
INSIGNIFICANT_FRACTION = 1.0e-6
SPREADING_WIDTH_FACTOR = 4.0
SPREADING_FRACTION = 0.35
JOINT_NEWTON_STEPS = 10
JOINT_GMRES_ITERATIONS = 30
JOINT_RESIDUAL_LIMIT = 1.0e-10
CURRENT_IDENTITY_LIMIT = 1.0e-10
PRIVATE_CURRENT_LIMIT_A = 0.0
PROFILE_BIN_COUNT = 56
SADDLE_AREA_LIMIT_M2 = 2.0e-15
SADDLE_ADDITIVE_LIMIT = 2.0e-15


def _strict_json(value):
    """Return ordinary JSON-compatible scalars and reject non-finite values."""
    if isinstance(value, dict):
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, np.ndarray):
        return _strict_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("the evidence payload contains a non-finite scalar")
    return value


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(_strict_json(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _warm_machine(case):
    """Load the semantic fixture cache without permitting a miss rebuild."""
    store, identity = reference._machine_cache_store(
        case, reference.SUITE_CELLS, passive=True
    )
    with reference._machine_cache_lock(store) as lock_wait_seconds:
        started = perf_counter()
        machine = reference._load_cached_machine(store, identity)
        load_seconds = perf_counter() - started
    return machine, {
        "status": "warm",
        "store": str(store.filepath),
        "semantic_key": store.group,
        "lock_wait_seconds": lock_wait_seconds,
        "load_seconds": load_seconds,
        "plasma_cells": len(machine.node),
    }


def _scaled_profile(profile: DomainProfile, amplitude) -> DomainProfile:
    """Return the same physical gradient pair at one declared amplitude."""
    scale = jnp.asarray(amplitude, dtype=jnp.float64)
    return DomainProfile(
        p_prime=lambda psi_norm: scale * profile.p_prime(psi_norm),
        ff_prime=lambda psi_norm: scale * profile.ff_prime(psi_norm),
    )


def _edge_anchored_source(case) -> tuple[ForwardSource, dict[str, object]]:
    """Continue the last finite tabulated trend with one Hermite shoulder.

    The stored profile reaches exactly zero at its final sample.  Such a source
    cannot demonstrate a finite crossing.  Both comparison arms therefore use
    the same demonstration-only edge choice.  The cubic-Hermite representation
    matches the last finite value and its incoming slope, then carries that
    slope analytically to the separatrix.  Its endpoint value and derivative
    are explicit polynomial boundary data rather than a clipped interpolant.

    Keeping the incoming trend is the minimum-change C1 choice: the Hermite
    cubic degenerates to its exact linear member, so no curvature unsupported
    by the tabulated core is invented.  This modified source is not attributed
    to the stored reference, whose zero endpoint remains unchanged on disk.
    """
    psi_grid = np.asarray(case.psi_norm)
    pressure = np.asarray(case.p_prime)
    diamagnetic = np.asarray(case.ff_prime)
    nonzero = np.flatnonzero((np.abs(pressure) + np.abs(diamagnetic)) > 0.0)
    if nonzero.size == 0 or nonzero[-1] >= len(psi_grid) - 1:
        raise ValueError("the stored profile has no finite pre-separatrix anchor")
    anchor = int(nonzero[-1])
    anchor_psi = float(psi_grid[anchor])
    interval = 1.0 - anchor_psi

    def shoulder(values):
        grid = jnp.asarray(psi_grid)
        samples = jnp.asarray(values)
        start_value = float(values[anchor])
        start_slope = float(
            (values[anchor] - values[anchor - 1])
            / (psi_grid[anchor] - psi_grid[anchor - 1])
        )
        endpoint_value = start_value + interval * start_slope
        endpoint_slope = start_slope

        def gradient(psi_norm):
            argument = jnp.asarray(psi_norm)
            interpolated = jnp.interp(argument, grid, samples)
            coordinate = (argument - anchor_psi) / interval
            coordinate_squared = coordinate * coordinate
            coordinate_cubed = coordinate_squared * coordinate
            start_value_basis = 2.0 * coordinate_cubed - 3.0 * coordinate_squared + 1.0
            start_slope_basis = coordinate_cubed - 2.0 * coordinate_squared + coordinate
            endpoint_value_basis = -2.0 * coordinate_cubed + 3.0 * coordinate_squared
            endpoint_slope_basis = coordinate_cubed - coordinate_squared
            hermite = (
                start_value_basis * start_value
                + start_slope_basis * interval * start_slope
                + endpoint_value_basis * endpoint_value
                + endpoint_slope_basis * interval * endpoint_slope
            )
            return jnp.where(argument >= anchor_psi, hermite, interpolated)

        return gradient, {
            "start_value": start_value,
            "start_slope_per_psi_norm": start_slope,
            "endpoint_value": endpoint_value,
            "endpoint_slope_per_psi_norm": endpoint_slope,
        }

    pressure_gradient, pressure_receipt = shoulder(pressure)
    diamagnetic_gradient, diamagnetic_receipt = shoulder(diamagnetic)

    source = ForwardSource(
        core=DomainProfile(
            p_prime=pressure_gradient,
            ff_prime=diamagnetic_gradient,
        ),
        boundary_pressure=float(case.pressure[-1]),
        boundary_field_function=float(case.field_function[-1]),
    )
    return source, {
        "authority": "demonstration_choice_not_stored_reference",
        "family": "analytic_cubic_hermite",
        "minimum_change_member": "incoming_linear_trend",
        "interval_psi_norm": [anchor_psi, 1.0],
        "pressure_gradient_pa_per_wb": pressure_receipt,
        "diamagnetic_gradient_t2m2_per_wb": diamagnetic_receipt,
    }


def _ledger_receipt(ledger) -> dict[str, float]:
    """Return one domain-current ledger as ordinary signed amperes."""
    return {name: float(getattr(ledger, name)) for name in ledger._fields}


def _sol_source(
    case,
    closure: EichSolClosure,
    variant: SolDecayVariant,
    amplitude,
) -> ForwardSource:
    """Compose one globally scaled core-plus-common-SOL source."""
    base, _receipt = _edge_anchored_source(case)
    core = _scaled_profile(base.core, amplitude)
    return ForwardSource(
        core=core,
        common_sol=closure.domain_profile(core, variant),
        boundary_pressure=float(amplitude) * base.boundary_pressure,
        boundary_field_function=base.boundary_field_function,
    )


def _field_components(
    machine, conductor_current, moments
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the machine's analytic poloidal-field blocks on cell centres."""
    radial = np.asarray(machine.radial_field[0]) @ np.asarray(conductor_current)
    vertical = np.asarray(machine.vertical_field[0]) @ np.asarray(conductor_current)
    for block, coefficient in zip(machine.radial_field[1:], moments, strict=True):
        radial += np.asarray(block) @ np.asarray(coefficient)
    for block, coefficient in zip(machine.vertical_field[1:], moments, strict=True):
        vertical += np.asarray(block) @ np.asarray(coefficient)
    return radial, vertical


def _outboard_midplane_index(machine, masks, topology) -> int:
    """Select the closest core-side separatrix sample at the outboard midplane."""
    radius = np.asarray(machine.node[:, 0])
    height = np.asarray(machine.node[:, 1])
    psi_norm = np.asarray(masks.psi_norm)
    core = np.asarray(masks.core)
    pitch = math.sqrt(float(np.median(machine.area[machine.hexagon])))
    candidates = core & (radius > float(topology.axis[0]))
    candidates &= np.abs(height - float(topology.axis[1])) < 1.5 * pitch
    index = np.flatnonzero(candidates)
    if index.size == 0:
        raise AssertionError("no core-side outboard-midplane sample was found")
    score = (
        np.abs(psi_norm[index] - 1.0)
        + 0.05 * np.abs(height[index] - float(topology.axis[1])) / pitch
    )
    return int(index[np.argmin(score)])


def _joint_root(unit_operator, initial_flux, target_current, initial_amplitude):
    """Solve the free-boundary map and net-current equation as one root."""
    target_scale = max(abs(float(target_current)), 1.0)
    external = unit_operator.external()

    def mapped(state):
        flux = state[:-1]
        amplitude = state[-1]
        moments = unit_operator.cell_current_moments(flux)
        internal = unit_operator.internal(flux)
        current = amplitude * jnp.sum(moments.cell_current)
        amplitude_image = amplitude - (current - target_current) / target_scale
        return jnp.r_[external + amplitude * internal, amplitude_image]

    initial = jnp.r_[jnp.asarray(initial_flux), jnp.asarray(initial_amplitude)]
    history = fixed_point.newton_krylov(
        mapped,
        initial,
        newton_steps=JOINT_NEWTON_STEPS,
        gmres_iterations=JOINT_GMRES_ITERATIONS,
        warmup=0,
    )
    jax.block_until_ready(history.state)
    return history


def _strike_points(machine, flux, topology) -> np.ndarray:
    """Read the two material intersections nearest the separatrix level."""
    grid_count = len(machine.node)
    wall_count = len(machine.wall_node)
    wall_flux = np.asarray(flux)[grid_count : grid_count + wall_count]
    distance = np.abs(
        (wall_flux - float(topology.axis_flux)) / float(topology.flux_span) - 1.0
    )
    wall = np.asarray(machine.wall_node)
    x_radius = float(topology.x_point[0])
    lower = wall[:, 1] < float(topology.axis[1])
    result = []
    for radial_selection in (wall[:, 0] <= x_radius, wall[:, 0] > x_radius):
        candidates = np.flatnonzero(lower & radial_selection)
        if candidates.size == 0:
            raise AssertionError("the wall samples do not bracket both divertor legs")
        result.append(wall[candidates[np.argmin(distance[candidates])]])
    return np.asarray(result)


def _strike_cells(
    machine, masks, strike_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return common-SOL cells adjacent to the two material intersections."""
    common = np.flatnonzero(np.asarray(masks.common_sol))
    if common.size == 0:
        raise AssertionError("the solved state has no common-SOL cells")
    coordinate = np.asarray(machine.node)
    cell = []
    distance = []
    for point in strike_points:
        separation = np.linalg.norm(coordinate[common] - point, axis=1)
        selected = int(common[np.argmin(separation)])
        cell.append(selected)
        distance.append(float(np.min(separation)))
    return np.asarray(cell, dtype=np.intp), np.asarray(distance)


def _surface_average(
    psi_norm: np.ndarray,
    density: np.ndarray,
    radius: np.ndarray,
    area: np.ndarray,
    support: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Form volume-weighted discrete flux-surface averages in fixed bins."""
    weight = 2.0 * np.pi * radius * area
    selected_bin = np.digitize(psi_norm, edges) - 1
    centre = 0.5 * (edges[:-1] + edges[1:])
    average = np.full(len(centre), np.nan)
    for index in range(len(centre)):
        selected = support & (selected_bin == index)
        if np.any(selected):
            average[index] = np.average(density[selected], weights=weight[selected])
    return centre, average


def _saddle_branch_receipt(machine, topology) -> dict[str, object]:
    """Exercise additive branch integration in the machine's X-point cell."""
    x_point = np.asarray(topology.x_point, dtype=float)
    containing = [
        index
        for index, polygon in enumerate(machine.cell_polygons)
        if PlotPath(np.asarray(polygon)).contains_point(x_point, radius=1.0e-12)
    ]
    if not containing:
        containing = [
            int(
                np.argmin(
                    [
                        np.min(np.linalg.norm(np.asarray(polygon) - x_point, axis=1))
                        for polygon in machine.cell_polygons
                    ]
                )
            )
        ]
    cell = containing[0]
    polygon = np.asarray(machine.cell_polygons[cell])
    clearance = np.min(np.linalg.norm(polygon - x_point, axis=1))
    half_width = max(0.2 * clearance, 1.0e-3)
    local_cell = x_point + half_width * np.asarray(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
    )
    mesh = AtomicCellMesh.from_cells([local_cell], centroids=x_point[None, :])
    offset = mesh.node_coordinates - x_point
    signed_flux = offset[:, 0] * offset[:, 1]
    common = mesh.traced_clip(jnp.asarray(signed_flux), saddle_vertex=x_point)
    private = mesh.traced_clip(jnp.asarray(-signed_flux), saddle_vertex=x_point)
    branch_density = jnp.asarray([[1.0, 0.0]])
    branch_gradient = jnp.zeros((1, 2, 2), dtype=jnp.float64)
    common_current, _ = common.branch_linear_current_moments(
        branch_density, branch_gradient
    )
    private_current, _ = private.branch_linear_current_moments(
        jnp.zeros_like(branch_density), branch_gradient
    )
    area_closure = float(
        jnp.sum(common.branch_area) + jnp.sum(private.branch_area) - half_width**2 * 4.0
    )
    additive = float(jnp.sum(common_current) - common_current[0, 0])
    if not bool(common.saddle[0]) or not bool(private.saddle[0]):
        raise AssertionError("the X-point audit cell did not take the saddle path")
    if abs(area_closure) > SADDLE_AREA_LIMIT_M2:
        raise AssertionError(f"saddle branch area closure is {area_closure:.17g} m2")
    if (
        abs(additive) > SADDLE_ADDITIVE_LIMIT
        or abs(float(jnp.sum(private_current))) > SADDLE_ADDITIVE_LIMIT
    ):
        raise AssertionError(
            "saddle branches did not preserve additive domain selection"
        )
    return {
        "machine_cell": cell,
        "x_point_m": x_point,
        "audit_half_width_m": half_width,
        "common_branch_vertex_count": np.asarray(common.branch_vertex_count[0]),
        "private_branch_vertex_count": np.asarray(private.branch_vertex_count[0]),
        "common_branch_area_m2": np.asarray(common.branch_area[0]),
        "private_branch_area_m2": np.asarray(private.branch_area[0]),
        "area_closure_m2": area_closure,
        "selected_common_branch_integral": float(jnp.sum(common_current)),
        "selected_private_branch_integral": float(jnp.sum(private_current)),
        "additive_closure": additive,
    }


def _profile_figure(
    output: Path,
    centres: np.ndarray,
    control_average: np.ndarray,
    sol_average: np.ndarray,
    dense_psi: np.ndarray,
    single_reference: np.ndarray,
    dual_reference: np.ndarray,
    support_extents: dict[str, float],
) -> None:
    """Plot discrete surface averages and the two measured closure tails."""
    figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
    axis.axvspan(1.0, dense_psi[-1], color="#f2f2f2", zorder=0)
    axis.axvline(1.0, color="#555", linewidth=0.9)
    axis.plot(
        centres,
        control_average,
        "o",
        ms=3.1,
        color="#3b4cc0",
        label="confined control ⟨jφ⟩",
    )
    axis.plot(
        centres, sol_average, "o", ms=3.1, color="#b40426", label="SOL solution ⟨jφ⟩"
    )
    axis.plot(
        dense_psi,
        single_reference,
        color="#e08214",
        linewidth=1.4,
        label="single Eich length",
    )
    axis.plot(
        dense_psi,
        dual_reference,
        color="#8c510a",
        linewidth=1.4,
        linestyle="--",
        label="dual Eich + spreading",
    )
    for name, extent in support_extents.items():
        axis.axvline(
            extent,
            color="#888",
            linewidth=0.75,
            linestyle=":" if name == "single" else "-.",
        )
    axis.set_yscale("log")
    visible_average = np.r_[control_average, sol_average]
    visible_average = visible_average[
        np.isfinite(visible_average) & (visible_average > 0.0)
    ]
    upper = max(2.0, 1.5 * float(np.max(visible_average, initial=1.0)))
    axis.set_ylim(5.0e-8, upper)
    axis.set_xlim(float(np.nanmin(centres)), float(dense_psi[-1]))
    axis.set_xlabel("normalised total poloidal flux ψN")
    axis.set_ylabel("|⟨jφ⟩| / |jφ,separatrix|")
    axis.set_title("Finite current crosses ψN = 1 and decays over the common SOL")
    axis.legend(frameon=False, ncol=2, fontsize=8.5)
    axis.grid(axis="y", linewidth=0.4, color="#ddd")
    figure.savefig(output, format="svg")
    plt.close(figure)


def _contour_figure(
    output: Path,
    machine,
    control_masks,
    sol_masks,
    control_topology,
    sol_topology,
    control_strike: np.ndarray,
    sol_strike: np.ndarray,
) -> None:
    """Plot matched discrete normalised-flux levels on the shared carrier."""
    triangulation = mtri.Triangulation(machine.node[:, 0], machine.node[:, 1])
    levels = np.linspace(0.1, 1.1, 11)
    colours = plt.cm.viridis(np.linspace(0.08, 0.92, len(levels)))
    figure, axes = plt.subplots(
        1, 3, figsize=(14.6, 5.2), constrained_layout=True, sharex=True, sharey=True
    )
    states = (
        ("confined control", control_masks, control_topology, control_strike),
        ("SOL-carrying solution", sol_masks, sol_topology, sol_strike),
    )
    for axis, (title, masks, topology, strikes) in zip(axes[:2], states, strict=True):
        axis.tricontour(
            triangulation,
            np.asarray(masks.psi_norm),
            levels=levels,
            colors=colours,
            linewidths=1.0,
        )
        axis.plot(
            machine.wall_node[:, 0],
            machine.wall_node[:, 1],
            color="#222",
            linewidth=0.8,
        )
        axis.plot(*np.asarray(topology.axis), marker="x", color="#b2182b", ms=6)
        axis.plot(
            *np.asarray(topology.x_point),
            marker="D",
            markerfacecolor="none",
            color="#b2182b",
            ms=5,
        )
        axis.plot(
            strikes[:, 0],
            strikes[:, 1],
            marker="o",
            linestyle="none",
            color="#2166ac",
            ms=4,
        )
        axis.set_title(title)
        axis.set_aspect("equal")
        axis.set_xlabel("R [m]")
    overlay = axes[2]
    overlay.tricontour(
        triangulation,
        np.asarray(control_masks.psi_norm),
        levels=levels,
        colors=colours,
        linewidths=0.9,
        linestyles="--",
    )
    overlay.tricontour(
        triangulation,
        np.asarray(sol_masks.psi_norm),
        levels=levels,
        colors=colours,
        linewidths=1.25,
    )
    overlay.plot(
        machine.wall_node[:, 0], machine.wall_node[:, 1], color="#222", linewidth=0.8
    )
    overlay.plot(
        *np.asarray(control_topology.x_point),
        marker="D",
        markerfacecolor="none",
        color="#3b4cc0",
        ms=5,
    )
    overlay.plot(
        *np.asarray(sol_topology.x_point),
        marker="D",
        markerfacecolor="none",
        color="#b40426",
        ms=5,
    )
    overlay.plot(
        control_strike[:, 0],
        control_strike[:, 1],
        marker="o",
        linestyle="none",
        markerfacecolor="none",
        color="#3b4cc0",
        ms=5,
    )
    overlay.plot(
        sol_strike[:, 0],
        sol_strike[:, 1],
        marker="o",
        linestyle="none",
        color="#b40426",
        ms=4,
    )
    overlay.set_title("matched levels: dashed control, solid SOL")
    overlay.set_aspect("equal")
    overlay.set_xlabel("R [m]")
    axes[0].set_ylabel("Z [m]")
    figure.suptitle(
        "SOL current moves the boundary, X-point and strike geometry on one machine"
    )
    figure.savefig(output, format="svg")
    plt.close(figure)


def measure(
    output: Path = DEFAULT_OUTPUT, *, reuse_roots: bool = False
) -> dict[str, object]:
    """Run the banked two-solution comparison and validate every live claim."""
    configure_dtypes()
    output.mkdir(parents=True, exist_ok=True)
    case = reference.require_reference()
    machine, cache = _warm_machine(case)
    bank = np.load(ROOT_BANK)
    banked_seed = jnp.asarray(bank["state"])
    reference_operator = reference.forward_operator(case, machine)
    base, edge_anchor = _edge_anchored_source(case)
    control_operator = replace(reference_operator, source=base)
    existing = None
    if reuse_roots:
        existing = np.load(output / "solutions.npz")
        control_flux = jnp.asarray(existing["control_flux"])
        control_trace = np.asarray(existing["control_residual_history"])
        control_solve_seconds = 0.0
        control_root_residual = math.nan
    else:
        control_started = perf_counter()
        control_history = fixed_point.newton_krylov(
            control_operator.flux_map(),
            banked_seed,
            newton_steps=JOINT_NEWTON_STEPS,
            gmres_iterations=JOINT_GMRES_ITERATIONS,
            warmup=0,
        )
        jax.block_until_ready(control_history.state)
        control_solve_seconds = perf_counter() - control_started
        control_flux = control_history.state
        control_trace = np.asarray(control_history.trace)
        control_root_residual = float(control_history.residual)
    control_moments, _control_measure, control_masks, control_topology = (
        control_operator.current_moments_and_observation(control_flux)
    )
    control_map_residual = float(
        jnp.max(jnp.abs(control_operator.residual(control_flux)))
        / jnp.maximum(jnp.max(jnp.abs(control_flux)), 1.0e-30)
    )
    if reuse_roots:
        control_root_residual = control_map_residual
    control_ledger = current_ledger(
        control_moments.cell_current,
        control_operator.current_domain_masks(control_flux),
    )
    target_current = float(control_ledger.total)

    radial_field, vertical_field = _field_components(
        machine, machine.source_current, control_moments
    )
    outboard = _outboard_midplane_index(machine, control_masks, control_topology)
    width = eich_width(
        outboard_midplane_radius_m=machine.node[outboard, 0],
        radial_poloidal_field_t=radial_field[outboard],
        vertical_poloidal_field_t=vertical_field[outboard],
        flux_span_wb=float(control_topology.flux_span),
    )
    closure = EichSolClosure(
        width=width,
        spreading_length_m=SPREADING_WIDTH_FACTOR * width.heat_flux_width_m,
        spreading_fraction=SPREADING_FRACTION,
    )
    support_extents = {
        "single": closure.support_extent(
            base.core,
            SolDecayVariant.SINGLE_LENGTH,
            insignificant_fraction=INSIGNIFICANT_FRACTION,
        ),
        "dual": closure.support_extent(
            base.core,
            SolDecayVariant.DUAL_LENGTH,
            insignificant_fraction=INSIGNIFICANT_FRACTION,
        ),
    }

    unit_source = _sol_source(case, closure, SolDecayVariant.DUAL_LENGTH, amplitude=1.0)
    unit_operator = replace(control_operator, source=unit_source)
    if reuse_roots:
        sol_flux = jnp.asarray(existing["sol_flux"])
        amplitude = float(existing["sol_source_amplitude"])
        sol_trace = np.asarray(existing["sol_joint_residual_history"])
        solve_seconds = 0.0
        joint_root_residual = math.nan
    else:
        unit_current_at_control = float(
            jnp.sum(unit_operator.cell_current_moments(control_flux).cell_current)
        )
        initial_amplitude = target_current / unit_current_at_control
        started = perf_counter()
        history = _joint_root(
            unit_operator, control_flux, target_current, initial_amplitude
        )
        solve_seconds = perf_counter() - started
        sol_flux = history.state[:-1]
        amplitude = float(history.state[-1])
        sol_trace = np.asarray(history.trace)
        joint_root_residual = float(history.residual)
    sol_source = _sol_source(
        case, closure, SolDecayVariant.DUAL_LENGTH, amplitude=amplitude
    )
    sol_operator = replace(control_operator, source=sol_source)
    sol_moments, _sol_measure, sol_masks, sol_topology = (
        sol_operator.current_moments_and_observation(sol_flux)
    )
    sol_current = float(jnp.sum(sol_moments.cell_current))
    sol_ledger = current_ledger(
        sol_moments.cell_current,
        sol_operator.current_domain_masks(sol_flux),
    )
    current_relative_error = abs(sol_current - target_current) / abs(target_current)
    sol_map_residual = float(
        jnp.max(jnp.abs(sol_operator.residual(sol_flux)))
        / jnp.maximum(jnp.max(jnp.abs(sol_flux)), 1.0e-30)
    )
    if reuse_roots:
        joint_root_residual = max(sol_map_residual, current_relative_error)

    private_current = np.asarray(sol_moments.cell_current)[
        np.asarray(sol_masks.private_flux)
    ]
    maximum_private_current = float(np.max(np.abs(private_current), initial=0.0))
    control_private = np.asarray(control_moments.cell_current)[
        np.asarray(control_masks.private_flux)
    ]
    maximum_control_private_current = float(
        np.max(np.abs(control_private), initial=0.0)
    )
    control_strikes = _strike_points(machine, control_flux, control_topology)
    sol_strikes = _strike_points(machine, sol_flux, sol_topology)
    strike_cells, strike_distance = _strike_cells(machine, sol_masks, sol_strikes)
    strike_current = np.asarray(sol_moments.cell_current)[strike_cells]
    saddle = _saddle_branch_receipt(machine, sol_topology)

    separatrix_radius = width.outboard_midplane_radius_m
    core_sep = float(
        sol_source.core.current_density(separatrix_radius, jnp.asarray(1.0))
    )
    common_profile = sol_source.common_sol
    sol_sep = float(common_profile.current_density(separatrix_radius, jnp.asarray(1.0)))
    control_sep = float(base.core.current_density(separatrix_radius, jnp.asarray(1.0)))
    left_gradient = float(
        jax.grad(
            lambda psi_norm: sol_source.core.current_density(
                separatrix_radius, psi_norm
            )
        )(jnp.asarray(1.0))
    )
    right_gradient = float(
        jax.grad(
            lambda psi_norm: common_profile.current_density(separatrix_radius, psi_norm)
        )(jnp.asarray(1.0))
    )
    value_relative_jump = abs(sol_sep - core_sep) / max(abs(core_sep), 1.0)
    gradient_relative_jump = abs(right_gradient - left_gradient) / max(
        abs(left_gradient), abs(right_gradient), 1.0
    )

    psi = np.asarray(sol_masks.psi_norm)
    radius = np.asarray(machine.node[:, 0])
    area = np.asarray(machine.area)
    sol_density = np.asarray(sol_source.current_density(jnp.asarray(radius), sol_masks))
    control_density = np.asarray(
        control_operator.source.current_density(jnp.asarray(radius), control_masks)
    )
    maximum_profile_psi = max(
        float(np.max(psi[np.asarray(sol_masks.common_sol)])), support_extents["dual"]
    )
    minimum_profile_psi = max(0.0, float(np.min(psi[np.asarray(sol_masks.core)])))
    edges = np.linspace(minimum_profile_psi, maximum_profile_psi, PROFILE_BIN_COUNT + 1)
    centres, sol_average = _surface_average(
        psi,
        sol_density,
        radius,
        area,
        np.asarray(sol_masks.core | sol_masks.common_sol),
        edges,
    )
    _, control_average = _surface_average(
        np.asarray(control_masks.psi_norm),
        control_density,
        radius,
        area,
        np.asarray(control_masks.core),
        edges,
    )
    separatrix_scale = abs(sol_sep)
    sol_average = np.abs(sol_average) / separatrix_scale
    control_average = np.abs(control_average) / separatrix_scale
    dense_psi = np.linspace(1.0, maximum_profile_psi, 500)
    single_profile = closure.domain_profile(base.core, SolDecayVariant.SINGLE_LENGTH)
    dual_profile = closure.domain_profile(base.core, SolDecayVariant.DUAL_LENGTH)
    single_reference = np.abs(
        np.asarray(
            single_profile.current_density(separatrix_radius, jnp.asarray(dense_psi))
        )
    ) / abs(float(single_profile.current_density(separatrix_radius, jnp.asarray(1.0))))
    dual_reference = np.abs(
        np.asarray(
            dual_profile.current_density(separatrix_radius, jnp.asarray(dense_psi))
        )
    ) / abs(float(dual_profile.current_density(separatrix_radius, jnp.asarray(1.0))))

    axis_shift = float(
        np.linalg.norm(
            np.asarray(sol_topology.axis) - np.asarray(control_topology.axis)
        )
    )
    x_point_shift = float(
        np.linalg.norm(
            np.asarray(sol_topology.x_point) - np.asarray(control_topology.x_point)
        )
    )
    strike_shift = np.linalg.norm(sol_strikes - control_strikes, axis=1)
    current_profile_figure = output / "current-profiles.svg"
    contour_figure = output / "flux-map-contours.svg"
    _profile_figure(
        current_profile_figure,
        centres,
        control_average,
        sol_average,
        dense_psi,
        single_reference,
        dual_reference,
        support_extents,
    )
    _contour_figure(
        contour_figure,
        machine,
        control_masks,
        sol_masks,
        control_topology,
        sol_topology,
        control_strikes,
        sol_strikes,
    )

    np.savez_compressed(
        output / "solutions.npz",
        control_flux=np.asarray(control_flux),
        sol_flux=np.asarray(sol_flux),
        control_cell_current_a=np.asarray(control_moments.cell_current),
        sol_cell_current_a=np.asarray(sol_moments.cell_current),
        control_domain_label=np.asarray(control_masks.label),
        sol_domain_label=np.asarray(sol_masks.label),
        control_psi_norm=np.asarray(control_masks.psi_norm),
        sol_psi_norm=np.asarray(sol_masks.psi_norm),
        control_residual_history=control_trace,
        sol_joint_residual_history=sol_trace,
        sol_source_amplitude=np.asarray(amplitude),
        profile_psi_norm=centres,
        control_surface_average_normalized=control_average,
        sol_surface_average_normalized=sol_average,
        control_strike_points_m=control_strikes,
        sol_strike_points_m=sol_strikes,
    )

    assertions = {
        "control_converged": control_map_residual < JOINT_RESIDUAL_LIMIT,
        "sol_converged": sol_map_residual < JOINT_RESIDUAL_LIMIT,
        "net_current_identity": current_relative_error < CURRENT_IDENTITY_LIMIT,
        "control_private_flux_zero": maximum_control_private_current
        == PRIVATE_CURRENT_LIMIT_A,
        "sol_private_flux_zero": maximum_private_current == PRIVATE_CURRENT_LIMIT_A,
        "common_sol_current_at_both_strikes": bool(
            np.all(np.abs(strike_current) > 0.0)
        ),
        "saddle_branches_additive": abs(saddle["additive_closure"])
        <= SADDLE_ADDITIVE_LIMIT,
        "separatrix_value_continuous": value_relative_jump < 1.0e-12,
        "separatrix_gradient_continuous": gradient_relative_jump < 1.0e-5,
    }
    if not all(assertions.values()):
        failed = [name for name, passed in assertions.items() if not passed]
        raise AssertionError(f"live SOL demonstration assertions failed: {failed}")

    report = {
        "schema": "nova.sol-current-demonstration",
        "carrier": {
            **cache,
            "reference_pulse": reference.PULSE,
            "reference_run": reference.RUN,
            "reference_time_slice": reference.TIME_SLICE,
            "root_bank": str(ROOT_BANK),
        },
        "closure": {
            "shared_edge_anchor": edge_anchor,
            "eich_poloidal_field_t": width.outboard_midplane_poloidal_field_t,
            "outboard_midplane_radius_m": width.outboard_midplane_radius_m,
            "heat_flux_width_m": width.heat_flux_width_m,
            "normalized_flux_width": width.normalized_flux_width,
            "spreading_width_factor": SPREADING_WIDTH_FACTOR,
            "spreading_fraction": SPREADING_FRACTION,
            "insignificant_fraction": INSIGNIFICANT_FRACTION,
            "support_extent_psi_norm": support_extents,
            "separatrix_value_relative_jump": value_relative_jump,
            "separatrix_gradient_relative_jump": gradient_relative_jump,
            "finite_separatrix_current_density_a_per_m2": sol_sep,
            "control_separatrix_current_density_a_per_m2": control_sep,
        },
        "solutions": {
            "control": {
                "solve_seconds": control_solve_seconds,
                "root_relative_residual": control_root_residual,
                "reused_existing_root": reuse_roots,
                "map_relative_residual": control_map_residual,
                "net_plasma_current_a": target_current,
                "current_ledger_a": _ledger_receipt(control_ledger),
                "axis_m": np.asarray(control_topology.axis),
                "x_point_m": np.asarray(control_topology.x_point),
                "strike_points_m": control_strikes,
                "private_flux_maximum_cell_current_a": maximum_control_private_current,
            },
            "sol": {
                "joint_relative_residual": joint_root_residual,
                "reused_existing_root": reuse_roots,
                "map_relative_residual": sol_map_residual,
                "net_plasma_current_a": sol_current,
                "current_ledger_a": _ledger_receipt(sol_ledger),
                "source_amplitude": amplitude,
                "solve_seconds": solve_seconds,
                "axis_m": np.asarray(sol_topology.axis),
                "x_point_m": np.asarray(sol_topology.x_point),
                "strike_points_m": sol_strikes,
                "private_flux_maximum_cell_current_a": maximum_private_current,
                "strike_adjacent_cell": strike_cells,
                "strike_adjacent_distance_m": strike_distance,
                "strike_adjacent_cell_current_a": strike_current,
            },
            "net_current_relative_difference": current_relative_error,
            "axis_shift_m": axis_shift,
            "x_point_shift_m": x_point_shift,
            "strike_point_shift_m": strike_shift,
        },
        "saddle_partition": saddle,
        "assertions": assertions,
        "artifacts": {
            "solutions": str(output / "solutions.npz"),
            "current_profiles": {
                "path": str(current_profile_figure),
                "project_absolute_src": (
                    "/nova/figures/sol-current-demonstration/current-profiles.svg"
                ),
            },
            "flux_map_contours": {
                "path": str(contour_figure),
                "project_absolute_src": (
                    "/nova/figures/sol-current-demonstration/flux-map-contours.svg"
                ),
            },
        },
    }
    _write_json(output / "results.json", report)
    return report


def main() -> None:
    """Run the measurement and print one compact outcome receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-roots",
        action="store_true",
        help="finalize an existing solutions bank without running either root",
    )
    arguments = parser.parse_args()
    report = measure(arguments.output, reuse_roots=arguments.reuse_roots)
    print(json.dumps(_strict_json(report), sort_keys=True))
    print("SOL_CURRENT_DEMONSTRATION_EXIT=0")


if __name__ == "__main__":
    main()
