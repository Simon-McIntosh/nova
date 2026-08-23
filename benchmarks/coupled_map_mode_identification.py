"""Identify the near-marginal eigenmode of the passive-control flux map.

The benchmark linearises the exact coupled map at the qualified near-circular
root, computes several dominant Ritz eigenpairs, and localises the leading
right eigenvector against physical translation and algebraic flux-coordinate
subspaces.  It then follows the passive-drive tangent on the independent
elongated carrier so the mechanism is not inferred from one discretisation.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from time import perf_counter

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from benchmarks.measurement_provenance import measurement_stamp
from benchmarks.passive_closure_trace import (
    CONTROL_GMRES_ITERATIONS,
    CONTROL_NEWTON_STEPS,
    CONTROL_NEWTON_WARMUP,
    DENSE_CELL_REQUEST,
    ELONGATED_BOUNDARY_ELONGATION,
    ELONGATED_DIRECT_PEAK_POINTS,
    MEASURED_UNPINNED_SPECTRAL_RADIUS,
    RUNTIME_MITIGATION_DEFAULTS,
    _analytic_solovev,
    _build_analytic_control,
    _equilibrium_core,
    _require_control_branch,
    _solve_analytic_control,
)
from nova.jax.config import configure_dtypes
from tests import test_equilibrium_forward_reference as reference


DEFAULT_RECEIPT = Path(
    "docs/figures/forward-operator-refinement/coupled-map-mode-identification.json"
)
DEFAULT_FIGURE = Path(
    "docs/figures/forward-operator-refinement/coupled-map-mode-identification.png"
)
STABLE_LATE_GROWTH = 0.9883689804610637
STABLE_RESOLVENT_GAIN = 106.99077030932676
ELONGATED_LATE_GROWTH = 1.3372082983171414
EIGENPAIR_COUNT = 6
ARNOLDI_SUBSPACE = 36
TANGENT_ITERATIONS = 16


def _strict_json(value):
    """Return finite built-in values suitable for an attributable receipt."""
    if isinstance(value, dict):
        return {key: _strict_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_strict_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_json(value.tolist())
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("mode receipt contains a non-finite scalar")
    return value


def _complex(value: complex) -> dict[str, float]:
    """Serialise one complex scalar without losing its phase."""
    return {
        "real": float(np.real(value)),
        "imaginary": float(np.imag(value)),
        "magnitude": float(abs(value)),
        "phase_radians": float(np.angle(value)),
    }


def _orthonormal_basis(*vectors: np.ndarray) -> np.ndarray:
    """Return a numerically independent orthonormal column basis."""
    matrix = np.column_stack([np.asarray(vector).reshape(-1) for vector in vectors])
    basis, triangular = np.linalg.qr(matrix)
    scale = max(float(np.max(np.abs(np.diag(triangular)))), np.finfo(float).tiny)
    keep = np.abs(np.diag(triangular)) > 1.0e-10 * scale
    return basis[:, keep]


def _projection_fraction(vector: np.ndarray, basis: np.ndarray) -> float:
    """Return squared vector energy captured by an orthonormal subspace."""
    vector = np.asarray(vector).reshape(-1)
    denominator = float(np.vdot(vector, vector).real)
    if denominator == 0.0 or basis.size == 0:
        return 0.0
    projected = basis.conj().T @ vector
    return float(np.vdot(projected, projected).real / denominator)


def _phase_fixed_real(vector: np.ndarray) -> np.ndarray:
    """Choose the complex phase whose largest component is positive real."""
    vector = np.asarray(vector, dtype=np.complex128)
    pivot = int(np.argmax(np.abs(vector)))
    phase = np.angle(vector[pivot])
    real = np.real(vector * np.exp(-1j * phase))
    if real[pivot] < 0.0:
        real = -real
    return real


def _axis_motion(operator, root, mode, span: float) -> dict[str, object]:
    """Measure the axis displacement induced by a small eigenmode perturbation."""
    real_mode = _phase_fixed_real(mode)
    peak = float(np.max(np.abs(real_mode)))
    scale = 1.0e-5 * span / peak
    left = operator.read(jnp.asarray(root) - scale * jnp.asarray(real_mode))[1]
    right = operator.read(jnp.asarray(root) + scale * jnp.asarray(real_mode))[1]
    displacement = 0.5 * (
        np.asarray(right.axis, dtype=float) - np.asarray(left.axis, dtype=float)
    )
    norm = float(np.linalg.norm(displacement))
    direction = displacement / norm if norm else np.zeros(2)
    return {
        "probe_peak_wb": 1.0e-5 * span,
        "symmetric_axis_displacement_m": displacement,
        "axis_displacement_magnitude_m": norm,
        "axis_displacement_direction_r_z": direction,
        "absolute_radial_over_vertical_motion": float(
            abs(displacement[0]) / max(abs(displacement[1]), np.finfo(float).tiny)
        ),
    }


def _analytic_translation_templates(control) -> tuple[np.ndarray, np.ndarray]:
    """Return radial and vertical translations of the analytic control seed."""
    coordinate = np.r_[
        np.asarray(control.without_passive.lattice.coordinate),
        np.asarray(control.material_boundary),
    ]
    radius = coordinate[:, 0]
    height = coordinate[:, 1]
    step = 1.0e-5
    radial = -(
        _analytic_solovev(radius + step, height)
        - _analytic_solovev(radius - step, height)
    ) / (2.0 * step)
    vertical = -(
        _analytic_solovev(radius, height + step)
        - _analytic_solovev(radius, height - step)
    ) / (2.0 * step)
    return radial, vertical


def _reference_translation_templates(case, machine) -> tuple[np.ndarray, np.ndarray]:
    """Return stored-map translation templates on every elongated state row."""
    coordinate = np.r_[machine.node, machine.wall_node, machine.sample_coordinates]
    radius = coordinate[:, 0]
    height = coordinate[:, 1]
    radial = case.spline.ev(radius, height, dx=1)
    vertical = case.spline.ev(radius, height, dy=1)
    return radial, vertical


def _localisation(
    vector,
    root,
    drive,
    operator,
    grid_coordinate,
    translation_templates,
    span: float,
) -> dict[str, object]:
    """Localise a mode in state blocks and mechanism-defined subspaces."""
    vector = np.asarray(vector, dtype=np.complex128)
    root = np.asarray(root, dtype=float)
    drive = np.asarray(drive, dtype=float)
    grid_nodes = operator.grid.node_number
    wall_nodes = operator.wall.node_number
    physical_nodes = grid_nodes + wall_nodes
    total_energy = float(np.vdot(vector, vector).real)
    wall_energy = float(
        np.vdot(
            vector[grid_nodes:physical_nodes], vector[grid_nodes:physical_nodes]
        ).real
    )
    sample_energy = float(
        np.vdot(vector[physical_nodes:], vector[physical_nodes:]).real
    )
    topology = operator.read(jnp.asarray(root))[1]
    axis = np.asarray(topology.axis, dtype=float)
    coordinate = np.asarray(grid_coordinate, dtype=float)
    nearest = np.sort(np.linalg.norm(coordinate - axis, axis=1))
    axis_radius = float(nearest[min(12, nearest.size - 1)])
    axis_mask = np.linalg.norm(coordinate - axis, axis=1) <= axis_radius
    axis_energy = float(
        np.vdot(vector[:grid_nodes][axis_mask], vector[:grid_nodes][axis_mask]).real
    )
    translation_basis = _orthonormal_basis(*translation_templates)
    normalisation_basis = _orthonormal_basis(
        np.ones_like(root), root - float(np.mean(root[:physical_nodes]))
    )
    drive_basis = _orthonormal_basis(drive)
    source_policy = operator.source.normalisation
    return {
        "definitions": {
            "boundary_state_fraction": "squared eigenvector norm on wall rows",
            "axis_neighbourhood_fraction": (
                "squared eigenvector norm on the thirteen grid cells nearest "
                "the solved axis"
            ),
            "passive_current_state_fraction": (
                "fraction in dynamic passive-current coordinates; the flux "
                "map state has none"
            ),
            "passive_drive_overlap_fraction": (
                "squared projection onto the retained-minus-zeroed passive "
                "external field"
            ),
            "normalisation_content_fraction": (
                "squared projection onto affine gauge and flux-span "
                "coordinate directions"
            ),
            "axis_translation_content_fraction": (
                "squared projection onto radial and vertical translation derivatives"
            ),
        },
        "state_rows": int(vector.size),
        "grid_rows": int(grid_nodes),
        "wall_rows": int(wall_nodes),
        "direct_sample_rows": int(vector.size - physical_nodes),
        "boundary_state_fraction": wall_energy / total_energy,
        "axis_neighbourhood_fraction": axis_energy / total_energy,
        "direct_sample_state_fraction": sample_energy / total_energy,
        "passive_current_state_fraction": 0.0,
        "passive_drive_overlap_fraction": _projection_fraction(vector, drive_basis),
        "normalisation_content_fraction": _projection_fraction(
            vector, normalisation_basis
        ),
        "axis_translation_content_fraction": _projection_fraction(
            vector, translation_basis
        ),
        "normalisation_policy": source_policy.name.lower(),
        "normalisation_amplitude": 1.0,
        "normalisation_rescaled": False,
        "axis_neighbourhood_radius_m": axis_radius,
        "axis_motion": _axis_motion(operator, root, vector, span),
    }


def _dominant_eigenpairs(tangent, drive, count: int = EIGENPAIR_COUNT):
    """Compute dominant right Ritz pairs of one exact JAX tangent action."""
    drive = np.asarray(drive, dtype=float)
    size = drive.size
    tangent(jnp.asarray(drive)).block_until_ready()

    def action(vector):
        return np.asarray(tangent(jnp.asarray(vector)), dtype=float)

    operator = LinearOperator((size, size), matvec=action, dtype=np.float64)
    generator = np.random.default_rng(982451653)
    initial = drive + 1.0e-3 * np.linalg.norm(drive) * generator.normal(size=size)
    values, vectors = eigs(
        operator,
        k=count,
        which="LM",
        v0=initial,
        ncv=ARNOLDI_SUBSPACE,
        tol=1.0e-10,
        maxiter=800,
    )
    order = np.argsort(np.abs(values))[::-1]
    values = values[order]
    vectors = vectors[:, order]
    residuals = np.asarray(
        [
            np.linalg.norm(
                action(vectors[:, index].real)
                + 1j * action(vectors[:, index].imag)
                - values[index] * vectors[:, index]
            )
            / np.linalg.norm(vectors[:, index])
            for index in range(count)
        ]
    )
    return values, vectors, residuals


def _driven_mode(tangent, drive, grid_nodes: int) -> dict[str, object]:
    """Follow one passive drive until its tangent increment exposes a mode."""
    term = jnp.asarray(drive)
    peaks = []
    norms = []
    rows = []
    for iteration in range(1, TANGENT_ITERATIONS + 1):
        if iteration > 1:
            term = tangent(term)
        term.block_until_ready()
        array = np.asarray(term, dtype=float)
        peak = float(np.max(np.abs(array[:grid_nodes])))
        norm = float(np.linalg.norm(array))
        rows.append(
            {
                "iteration": iteration,
                "grid_peak_wb": peak,
                "state_l2_wb": norm,
                "grid_peak_growth_ratio": None if not peaks else peak / peaks[-1],
                "state_l2_growth_ratio": None if not norms else norm / norms[-1],
            }
        )
        peaks.append(peak)
        norms.append(norm)
    previous = np.asarray(tangent(term), dtype=float)
    rayleigh = float(
        np.dot(np.asarray(term), previous) / np.dot(np.asarray(term), np.asarray(term))
    )
    return {
        "iterations": rows,
        "late_grid_peak_growth_median": float(
            np.median([row["grid_peak_growth_ratio"] for row in rows[-5:]])
        ),
        "late_l2_growth_median": float(
            np.median([row["state_l2_growth_ratio"] for row in rows[-5:]])
        ),
        "terminal_rayleigh_quotient": rayleigh,
        "terminal_vector": np.asarray(term, dtype=float),
    }


def _stable_measurement() -> tuple[dict[str, object], object, object, np.ndarray]:
    """Solve, qualify and diagonalise the stable analytic control map."""
    started = perf_counter()
    control = _build_analytic_control()
    solved, solve_receipt = _solve_analytic_control(
        control.without_passive,
        control.seed_flux,
        "passive_currents_zeroed",
    )
    _require_control_branch(
        control.reference_axis,
        control.material_boundary,
        solve_receipt,
    )
    operator = control.without_passive.operator
    root = solved.flux
    mapped, tangent = jax.linearize(operator.flux_map(), root)
    drive = (
        control.with_passive.operator.external()
        - control.without_passive.operator.external()
    )
    values, vectors, residuals = _dominant_eigenpairs(tangent, drive)
    radial, vertical = _analytic_translation_templates(control)
    pairs = []
    for index, (value, residual) in enumerate(zip(values, residuals, strict=True)):
        pairs.append(
            {
                "rank": index + 1,
                "eigenvalue": _complex(value),
                "ritz_residual_l2": float(residual),
                "localisation": _localisation(
                    vectors[:, index],
                    root,
                    drive,
                    operator,
                    control.without_passive.lattice.coordinate,
                    (radial, vertical),
                    control.reference_flux_span_wb,
                ),
            }
        )
    driven = _driven_mode(tangent, drive, operator.grid.node_number)
    leading_alignment = _projection_fraction(
        driven["terminal_vector"], _orthonormal_basis(vectors[:, 0])
    )
    root_residual = float(
        np.max(np.abs(np.asarray(mapped - root)))
        / max(float(np.max(np.abs(np.asarray(root)))), np.finfo(float).tiny)
    )
    receipt = {
        "carrier": {
            "construction": "analytic structured free-boundary carrier",
            "lattice_shape": list(control.without_passive.lattice.shape),
            "plasma_cells": int(operator.grid.node_number),
            "wall_nodes": int(operator.wall.node_number),
            "state_rows": int(root.size),
            "boundary_elongation": control.construction_receipt[
                "control_boundary_elongation"
            ],
            "vertical_decay_index": solve_receipt["vertical_conditioning"][
                "decay_index"
            ],
            "vertical_decay_index_stable": solve_receipt["vertical_conditioning"][
                "stable"
            ],
        },
        "solve": solve_receipt,
        "root_relative_map_residual": root_residual,
        "dominant_eigenpairs": pairs,
        "passive_driven_tangent": {
            key: value for key, value in driven.items() if key != "terminal_vector"
        },
        "leading_eigenvector_alignment_with_passive_driven_late_vector": (
            leading_alignment
        ),
        "elapsed_seconds": perf_counter() - started,
    }
    return receipt, control, solved, vectors[:, 0]


def _elongated_measurement() -> dict[str, object]:
    """Measure whether the elongated passive drive approaches the same mode."""
    started = perf_counter()
    case = reference.require_reference()
    machine = reference.cached_machine(case, DENSE_CELL_REQUEST, passive=True)
    current = machine.source_current.copy()
    current[-machine.passive_columns :] = 0.0
    without_machine = replace(
        machine,
        source_current=current,
        passive_columns=0,
        cache_receipt=None,
    )
    with_operator = reference.forward_operator(case, machine)
    operator = reference.forward_operator(case, without_machine)
    solved = reference.solve(case, without_machine)
    root = solved.flux
    mapped, tangent = jax.linearize(operator.flux_map(), root)
    drive = with_operator.external() - operator.external()
    driven = _driven_mode(tangent, drive, operator.grid.node_number)
    radial, vertical = _reference_translation_templates(case, without_machine)
    localisation = _localisation(
        driven["terminal_vector"],
        root,
        drive,
        operator,
        without_machine.node,
        (radial, vertical),
        abs(float(case.flux_span)),
    )
    core = _equilibrium_core(solved)
    root_residual = float(
        np.max(np.abs(np.asarray(mapped - root)[: len(machine.node)][core]))
        / abs(float(case.flux_span))
    )
    return {
        "carrier": {
            "requested_cells": DENSE_CELL_REQUEST,
            "plasma_cells": int(operator.grid.node_number),
            "wall_nodes": int(operator.wall.node_number),
            "direct_sample_rows": int(
                operator.node_number - operator.physical_node_number
            ),
            "state_rows": int(root.size),
            "boundary_elongation": ELONGATED_BOUNDARY_ELONGATION,
        },
        "root_relative_core_map_residual": root_residual,
        "passive_driven_tangent": {
            key: value for key, value in driven.items() if key != "terminal_vector"
        },
        "late_vector_localisation": localisation,
        "elapsed_seconds": perf_counter() - started,
    }


def _mechanism(stable: dict[str, object], elongated: dict[str, object]):
    """Classify the leading mode from measured subspaces and carrier transport."""
    leading = stable["dominant_eigenpairs"][0]
    stable_content = leading["localisation"]
    elongated_content = elongated["late_vector_localisation"]
    stable_direction = np.asarray(
        stable_content["axis_motion"]["axis_displacement_direction_r_z"]
    )
    elongated_direction = np.asarray(
        elongated_content["axis_motion"]["axis_displacement_direction_r_z"]
    )
    direction_alignment = abs(float(np.dot(stable_direction, elongated_direction)))
    stable_translation = float(stable_content["axis_translation_content_fraction"])
    elongated_translation = float(
        elongated_content["axis_translation_content_fraction"]
    )
    normalisation = float(stable_content["normalisation_content_fraction"])
    leading_alignment = float(
        stable["leading_eigenvector_alignment_with_passive_driven_late_vector"]
    )
    elongated_growth = float(
        elongated["passive_driven_tangent"]["late_grid_peak_growth_median"]
    )
    same_mode = bool(
        leading_alignment >= 0.8
        and direction_alignment >= 0.8
        and abs(elongated_growth - ELONGATED_LATE_GROWTH) / ELONGATED_LATE_GROWTH
        <= 0.15
    )
    algebraic = bool(
        normalisation >= 0.5
        or stable_content["normalisation_policy"] != "absolute"
        or stable_content["passive_current_state_fraction"] != 0.0
    )
    physical = bool(
        same_mode
        and max(stable_translation, elongated_translation) >= 0.25
        and not algebraic
    )
    if algebraic:
        classification = "ALGEBRAIC_CONSTRAINT_OR_NORMALISATION_MODE"
        mechanism = "affine_flux_coordinate_or_scalar_current_constraint"
    elif physical:
        classification = "PHYSICAL_NEAR_MARGINAL_MODE"
        mechanism = "free_boundary_axis_displacement_mode"
    else:
        classification = "DISCRETISATION_ARTIFACT"
        mechanism = "carrier_specific_flux_map_mode"
    return {
        "classification": classification,
        "mechanism": mechanism,
        "same_mode_carries_elongated_growth": same_mode,
        "stable_passive_drive_to_leading_mode_fraction": leading_alignment,
        "stable_to_elongated_axis_motion_direction_alignment": direction_alignment,
        "stable_axis_translation_content_fraction": stable_translation,
        "elongated_axis_translation_content_fraction": elongated_translation,
        "stable_normalisation_content_fraction": normalisation,
        "absolute_source_policy_excludes_scalar_current_rescaling": bool(
            stable_content["normalisation_policy"] == "absolute"
            and stable_content["normalisation_amplitude"] == 1.0
            and not stable_content["normalisation_rescaled"]
        ),
        "passive_current_is_not_a_map_state_coordinate": bool(
            stable_content["passive_current_state_fraction"] == 0.0
        ),
        "discretisation_discriminator": (
            "The stable leading eigenvector is compared with the late passive-driven "
            "mode on the independent elongated unstructured carrier; agreement in "
            "axis-motion direction and the independently reproduced elongated growth "
            "excludes a mode confined to the 625-cell structured control."
        ),
        "repair_scope": (
            "A physical near-marginal mode requires route qualification or screening "
            "in nova/equilibrium/forward_operator.py and fixed_point.py. An algebraic "
            "mode would instead require nova/equilibrium/source.py or domain.py. A "
            "carrier-specific mode would require mesh/coupling work in nova/biot/."
        ),
    }


def run() -> dict[str, object]:
    """Run the stable-root eigensystem and elongated-carrier discriminator."""
    source_commit = measurement_stamp(Path.cwd())
    configure_dtypes()
    stable, _control, _solved, _leading = _stable_measurement()
    elongated = _elongated_measurement()
    mechanism = _mechanism(stable, elongated)
    return {
        "receipt": {
            "kind": "coupled_map_mode_identification",
            "status": "complete",
            "source_commit": source_commit,
            "checkout_porcelain_empty_before_measurement": True,
            "device_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "runtime_mitigations": {
                "defaults": RUNTIME_MITIGATION_DEFAULTS,
                "effective": {
                    name: os.environ.get(name) for name in RUNTIME_MITIGATION_DEFAULTS
                },
            },
        },
        "comparators": {
            "stable_measured_late_growth": STABLE_LATE_GROWTH,
            "stable_measured_resolvent_gain": STABLE_RESOLVENT_GAIN,
            "elongated_measured_late_growth": ELONGATED_LATE_GROWTH,
            "elongated_measured_unpinned_spectral_radius": (
                MEASURED_UNPINNED_SPECTRAL_RADIUS
            ),
            "elongated_direct_external_peak_points": ELONGATED_DIRECT_PEAK_POINTS,
        },
        "solver_budget": {
            "stable_warmup": CONTROL_NEWTON_WARMUP,
            "stable_newton_steps": CONTROL_NEWTON_STEPS,
            "stable_gmres_iterations": CONTROL_GMRES_ITERATIONS,
            "dominant_eigenpairs": EIGENPAIR_COUNT,
            "arnoldi_subspace": ARNOLDI_SUBSPACE,
            "passive_tangent_iterations": TANGENT_ITERATIONS,
        },
        "stable_control": stable,
        "elongated_cross_carrier": elongated,
        "mechanism_verdict": mechanism,
    }


def _plot(receipt: dict[str, object], path: Path) -> None:
    """Plot the eigenvalues, mode content and passive-driven growth."""
    pairs = receipt["stable_control"]["dominant_eigenpairs"]
    values = np.asarray(
        [
            complex(pair["eigenvalue"]["real"], pair["eigenvalue"]["imaginary"])
            for pair in pairs
        ]
    )
    stable_content = pairs[0]["localisation"]
    elongated_content = receipt["elongated_cross_carrier"][
        "late_vector_localisation"
    ]
    labels = [
        "boundary",
        "axis cells",
        "passive drive",
        "normalisation",
        "translation",
    ]
    keys = [
        "boundary_state_fraction",
        "axis_neighbourhood_fraction",
        "passive_drive_overlap_fraction",
        "normalisation_content_fraction",
        "axis_translation_content_fraction",
    ]
    stable_growth = receipt["stable_control"]["passive_driven_tangent"]["iterations"]
    elongated_growth = receipt["elongated_cross_carrier"]["passive_driven_tangent"][
        "iterations"
    ]

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    angle = np.linspace(0.0, 2.0 * np.pi, 300)
    axes[0].plot(np.cos(angle), np.sin(angle), color="0.75", linewidth=1.0)
    axes[0].scatter(
        values.real, values.imag, c=np.arange(len(values)), cmap="viridis", s=55
    )
    for rank, value in enumerate(values, start=1):
        axes[0].annotate(
            str(rank),
            (value.real, value.imag),
            xytext=(4, 4),
            textcoords="offset points",
        )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Re eigenvalue")
    axes[0].set_ylabel("Im eigenvalue")
    axes[0].set_title("stable-root dominant spectrum")

    position = np.arange(len(labels))
    width = 0.38
    axes[1].bar(
        position - width / 2,
        [stable_content[key] for key in keys],
        width,
        label="stable eigenmode",
    )
    axes[1].bar(
        position + width / 2,
        [elongated_content[key] for key in keys],
        width,
        label="elongated late mode",
    )
    axes[1].set_xticks(position, labels, rotation=35, ha="right")
    axes[1].set_ylabel("squared projection or state fraction")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("mode localisation")

    axes[2].plot(
        [row["iteration"] for row in stable_growth[1:]],
        [row["grid_peak_growth_ratio"] for row in stable_growth[1:]],
        marker="o",
        label="stable control",
    )
    axes[2].plot(
        [row["iteration"] for row in elongated_growth[1:]],
        [row["grid_peak_growth_ratio"] for row in elongated_growth[1:]],
        marker="s",
        label="elongated",
    )
    axes[2].axhline(STABLE_LATE_GROWTH, color="C0", linestyle="--", linewidth=1.0)
    axes[2].axhline(ELONGATED_LATE_GROWTH, color="C1", linestyle="--", linewidth=1.0)
    axes[2].set_xlabel("tangent iteration")
    axes[2].set_ylabel("grid-peak increment growth")
    axes[2].set_title(
        receipt["mechanism_verdict"]["classification"].replace("_", " ").title()
    )
    axes[2].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, transparent=True)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    arguments = parser.parse_args()
    receipt = _strict_json(run())
    arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
    arguments.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _plot(receipt, arguments.figure)
    leading = receipt["stable_control"]["dominant_eigenpairs"][0]
    print(
        json.dumps(
            {
                "classification": receipt["mechanism_verdict"]["classification"],
                "leading_eigenvalue": leading["eigenvalue"],
                "same_mode_carries_elongated_growth": receipt["mechanism_verdict"][
                    "same_mode_carries_elongated_growth"
                ],
                "receipt": str(arguments.receipt),
                "figure": str(arguments.figure),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
