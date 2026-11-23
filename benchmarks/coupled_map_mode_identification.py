"""Measure passive-shell representation sensitivity of a near-marginal map.

The benchmark refits the vertical passive drive on a qualified near-circular
control using several conductor representations.  It solves and linearises one
byte-identical zero-passive-current map, recomputes the dominant
Ritz pairs from each fitted drive, and localises the leading right eigenvector
against physical translation and algebraic flux-coordinate subspaces.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

from benchmarks.measurement_provenance import measurement_stamp
from benchmarks.passive_closure_trace import (
    PASSIVE_SHELL_EXPANSION,
    CONTROL_GMRES_ITERATIONS,
    CONTROL_NEWTON_STEPS,
    CONTROL_NEWTON_WARMUP,
    DENSE_CELL_REQUEST,
    ELONGATED_BOUNDARY_ELONGATION,
    ELONGATED_DIRECT_PEAK_POINTS,
    RUNTIME_MITIGATION_DEFAULTS,
    _analytic_material_loop,
    _analytic_solovev,
    _build_analytic_control,
    _equilibrium_core,
    _require_control_branch,
    _solve_analytic_control,
    _vertical_translation_mode,
)
from nova.jax.config import configure_dtypes
from tests import test_equilibrium_forward_solve as forward_solve_suite
from tests import test_equilibrium_forward_reference as reference


DEFAULT_RECEIPT = Path(
    "docs/figures/forward-operator-refinement/"
    "passive-representation-eigenvalue-response.json"
)
DEFAULT_FIGURE = Path(
    "docs/figures/forward-operator-refinement/"
    "passive-representation-eigenvalue-response.png"
)
BASELINE_LEADING_EIGENVALUE = 0.9874290999339198
STABLE_LATE_GROWTH = 0.9883689804610637
STABLE_RESOLVENT_GAIN = 106.99077030932676
ELONGATED_LATE_GROWTH = 1.3372082983171414
EIGENPAIR_COUNT = 6
ARNOLDI_SUBSPACE = 36
TANGENT_ITERATIONS = 16
REFINED_PASSIVE_CONDUCTORS = 32
CHANGED_SHELL_EXPANSION = 1.16
MATERIAL_MOVEMENT_FRACTION = 0.1


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
    leading_value = values[0]
    scalar_resolvent = 1.0 / abs(1.0 - leading_value)
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
        "leading_mode_comparison": {
            "leading_eigenvalue": _complex(leading_value),
            "measured_late_growth": STABLE_LATE_GROWTH,
            "absolute_eigenvalue_minus_late_growth": float(
                abs(abs(leading_value) - STABLE_LATE_GROWTH)
            ),
            "relative_eigenvalue_minus_late_growth": float(
                abs(abs(leading_value) - STABLE_LATE_GROWTH) / STABLE_LATE_GROWTH
            ),
            "scalar_modal_resolvent_factor": float(scalar_resolvent),
            "measured_peak_field_resolvent_gain": STABLE_RESOLVENT_GAIN,
            "measured_gain_over_scalar_modal_factor": float(
                STABLE_RESOLVENT_GAIN / scalar_resolvent
            ),
            "interpretation": (
                "The eigenvalue fixes the near-pole denominator. The measured "
                "peak-field gain is larger than the scalar modal factor because "
                "the drive, eigenvector and reported grid-peak norm are not a "
                "unit-normal orthogonal scalar channel."
            ),
        },
        "passive_driven_tangent": {
            key: value for key, value in driven.items() if key != "terminal_vector"
        },
        "leading_eigenvector_alignment_with_passive_driven_late_vector": (
            leading_alignment
        ),
        "elapsed_seconds": perf_counter() - started,
    }
    return receipt, control, solved, vectors[:, 0]


def _with_passive_representation(control, conductor_count: int, expansion: float):
    """Return the control with one refitted wall-following passive shell."""
    operator = control.with_passive.operator
    grid_coordinate = np.asarray(control.with_passive.lattice.coordinate)
    wall_coordinate = np.asarray(control.material_boundary)
    axis = np.asarray(control.reference_axis)
    boundary, _boundary_flux = _analytic_material_loop(conductor_count)
    passive_coordinate = axis + expansion * (boundary - axis)
    passive_to_grid = forward_solve_suite._green_block(
        grid_coordinate, passive_coordinate
    )
    passive_to_wall = forward_solve_suite._green_block(
        wall_coordinate, passive_coordinate
    )

    active_count = control.construction_receipt["active_fit"]["conductor_count"]
    active_to_grid = np.asarray(operator.grid.source_target)[:, :active_count]
    active_to_wall = np.asarray(operator.wall.source_target)[:, :active_count]
    active_current = np.asarray(operator.external_current)[:active_count]
    inside = np.asarray(operator.inside_material, dtype=bool)
    weight = np.r_[inside.astype(float), np.ones(len(wall_coordinate))]
    translation_target = np.r_[
        _vertical_translation_mode(grid_coordinate[:, 0], grid_coordinate[:, 1]),
        _vertical_translation_mode(wall_coordinate[:, 0], wall_coordinate[:, 1]),
    ]
    passive_matrix = np.r_[passive_to_grid, passive_to_wall]
    passive_current = np.linalg.lstsq(
        passive_matrix * weight[:, None],
        translation_target * weight,
        rcond=None,
    )[0]
    raw_fit = passive_matrix @ passive_current
    fit_error = float(
        np.linalg.norm((raw_fit - translation_target) * weight)
        / np.linalg.norm(translation_target * weight)
    )
    passive_scale = (
        ELONGATED_DIRECT_PEAK_POINTS
        / 100.0
        * control.reference_flux_span_wb
        / np.max(np.abs(passive_to_grid @ passive_current))
    )
    passive_current *= passive_scale

    source_to_grid = np.c_[active_to_grid, passive_to_grid]
    source_to_wall = np.c_[active_to_wall, passive_to_wall]
    with_current = np.r_[active_current, passive_current]

    def represented_profile(current):
        represented_operator = replace(
            operator,
            grid=replace(operator.grid, source_target=jnp.asarray(source_to_grid)),
            wall=replace(operator.wall, source_target=jnp.asarray(source_to_wall)),
            external_current=jnp.asarray(current),
        )
        return replace(control.with_passive, operator=represented_operator)

    with_passive = represented_profile(with_current)
    material_radius = np.linalg.norm(boundary - axis, axis=1)
    shell_standoff = (expansion - 1.0) * material_radius
    passive_image = passive_to_grid @ passive_current
    construction = deepcopy(control.construction_receipt)
    construction["passive_material_coupling"] = {
        "conductor_count": conductor_count,
        "boundary_expansion_factor": expansion,
        "shell_standoff_m": {
            "minimum": float(np.min(shell_standoff)),
            "mean": float(np.mean(shell_standoff)),
            "maximum": float(np.max(shell_standoff)),
        },
        "green_section_width_m": 0.05,
        "green_section_height_m": 0.05,
        "source_to_grid_shape": list(passive_to_grid.shape),
        "source_to_wall_shape": list(passive_to_wall.shape),
        "translation_mode_weighted_relative_l2_error": fit_error,
        "current_l1_a": float(np.linalg.norm(passive_current, ord=1)),
        "current_l2_a": float(np.linalg.norm(passive_current)),
        "maximum_absolute_current_a": float(np.max(np.abs(passive_current))),
        "control_direct_external_peak_points": float(
            100.0 * np.max(np.abs(passive_image)) / control.reference_flux_span_wb
        ),
    }
    return replace(
        control,
        with_passive=with_passive,
        without_passive=control.without_passive,
        construction_receipt=construction,
    )


def _build_qualified_control():
    """Build every fitted control input on the qualified centroid image."""
    profile_type = forward_solve_suite.ForwardProfile
    from_lattice = profile_type.from_lattice.__func__

    def centroid_from_lattice(profile_class, *args, **kwargs):
        kwargs["cubic_cell_average"] = False
        return from_lattice(profile_class, *args, **kwargs)

    with patch.object(
        profile_type,
        "from_lattice",
        classmethod(centroid_from_lattice),
    ):
        return _build_analytic_control()


def _representation_response() -> dict[str, object]:
    """Recompute the leading mode for distinct fitted passive shells."""
    started = perf_counter()
    base_control = _build_qualified_control()
    specifications = (
        ("baseline", 16, PASSIVE_SHELL_EXPANSION),
        (
            "refined_conductor_count",
            REFINED_PASSIVE_CONDUCTORS,
            PASSIVE_SHELL_EXPANSION,
        ),
        ("changed_shell_standoff", 16, CHANGED_SHELL_EXPANSION),
    )
    controls = [(specifications[0][0], base_control)]
    controls.extend(
        (name, _with_passive_representation(base_control, count, expansion))
        for name, count, expansion in specifications[1:]
    )
    baseline_control = controls[0][1]
    baseline_operator = baseline_control.without_passive.operator
    baseline_external = np.asarray(baseline_operator.external())
    if not all(
        control.without_passive is baseline_control.without_passive
        and control.without_passive.operator is baseline_operator
        for _name, control in controls
    ):
        raise RuntimeError("passive representations do not share one control map")
    solved, solve_receipt = _solve_analytic_control(
        baseline_control.without_passive,
        baseline_control.seed_flux,
        "passive_currents_zeroed",
    )
    _require_control_branch(
        baseline_control.reference_axis,
        baseline_control.material_boundary,
        solve_receipt,
    )
    root = solved.flux
    mapped, tangent = jax.linearize(baseline_operator.flux_map(), root)
    radial, vertical = _analytic_translation_templates(baseline_control)
    drives = []
    for _name, control in controls:
        drives.append(
            np.asarray(
                control.with_passive.operator.external()
                - control.without_passive.operator.external()
            )
        )
    baseline_drive_basis = _orthonormal_basis(drives[0])
    representations = []
    for (name, control), drive in zip(controls, drives, strict=True):
        operator = control.without_passive.operator
        zero_external_difference = float(
            np.max(np.abs(np.asarray(operator.external()) - baseline_external))
        )
        values, vectors, residuals = _dominant_eigenpairs(tangent, drive)
        leading = values[0]
        localisation = _localisation(
            vectors[:, 0],
            root,
            drive,
            operator,
            control.without_passive.lattice.coordinate,
            (radial, vertical),
            control.reference_flux_span_wb,
        )
        driven = _driven_mode(tangent, drive, operator.grid.node_number)
        passive = control.construction_receipt["passive_material_coupling"]
        representations.append(
            {
                "name": name,
                "passive_representation": passive,
                "zero_passive_external_field_max_difference_wb": (
                    zero_external_difference
                ),
                "root_relative_map_residual": float(
                    np.max(np.abs(np.asarray(mapped - root)))
                    / max(
                        float(np.max(np.abs(np.asarray(root)))),
                        np.finfo(float).tiny,
                    )
                ),
                "shared_without_passive_profile_object": True,
                "shared_without_passive_operator_object": True,
                "drive_overlap_with_baseline_fraction": _projection_fraction(
                    drive, baseline_drive_basis
                ),
                "leading_eigenvalue": _complex(leading),
                "leading_eigenvalue_absolute_movement_from_banked": float(
                    abs(leading - BASELINE_LEADING_EIGENVALUE)
                ),
                "leading_ritz_residual_l2": float(residuals[0]),
                "dominant_eigenpairs": [
                    {
                        "rank": rank,
                        "eigenvalue": _complex(value),
                        "ritz_residual_l2": float(residual),
                    }
                    for rank, (value, residual) in enumerate(
                        zip(values, residuals, strict=True), start=1
                    )
                ],
                "leading_eigenvector_localisation": localisation,
                "passive_driven_late_grid_peak_growth_median": driven[
                    "late_grid_peak_growth_median"
                ],
                "leading_eigenvector_alignment_with_driven_late_vector": (
                    _projection_fraction(
                        driven["terminal_vector"], _orthonormal_basis(vectors[:, 0])
                    )
                ),
            }
        )

    reproduced = representations[0]["leading_eigenvalue_absolute_movement_from_banked"]
    if reproduced > 1.0e-8:
        raise RuntimeError(
            f"baseline leading eigenvalue did not reproduce: movement={reproduced:.3e}"
        )
    contraction_margin = 1.0 - BASELINE_LEADING_EIGENVALUE
    material_threshold = MATERIAL_MOVEMENT_FRACTION * contraction_margin
    additional = representations[1:]
    maximum_movement = max(
        row["leading_eigenvalue_absolute_movement_from_banked"] for row in additional
    )
    material = maximum_movement >= material_threshold
    most_moved = max(
        additional,
        key=lambda row: row["leading_eigenvalue_absolute_movement_from_banked"],
    )
    movement = most_moved["leading_eigenvalue"]["real"] - BASELINE_LEADING_EIGENVALUE
    if material:
        direction = (
            "toward unity with weaker damping"
            if movement > 0.0
            else "away from unity with stronger damping"
        )
        classification = "REPRESENTATION_IMPROVABLE_NEAR_MARGINALITY"
        ruling = (
            "The vertical-channel eigenvalue moves by a material fraction of its "
            "baseline contraction margin when the passive shell representation "
            f"changes, {direction}."
        )
    else:
        direction = "static within the declared contraction-margin threshold"
        classification = "INTRINSIC_MAP_PROPERTY"
        ruling = (
            "The leading vertical-channel eigenvalue is static under both passive "
            "shell changes. The shell is a prescribed drive rather than a dynamic "
            "map coordinate, so representation work cannot move this pole."
        )

    incident_widened_residual = 7.726885632357845e-6
    incident_banked_residual = 1.3931414905413574e-16
    return {
        "carrier": {
            "construction": "qualified analytic structured free-boundary control",
            "cell_current_representation": (
                "centroid image pinned by the banked eigenvalue comparator"
            ),
            "lattice_shape": list(baseline_control.without_passive.lattice.shape),
            "plasma_cells": int(baseline_operator.grid.node_number),
            "wall_nodes": int(baseline_operator.wall.node_number),
            "state_rows": int(baseline_operator.node_number),
            "vertical_decay_index": solve_receipt["vertical_conditioning"][
                "decay_index"
            ],
            "vertical_decay_index_stable": solve_receipt["vertical_conditioning"][
                "stable"
            ],
        },
        "solve": solve_receipt,
        "banked_baseline_leading_eigenvalue": BASELINE_LEADING_EIGENVALUE,
        "representations": representations,
        "shape_sensitivity_incident": {
            "banked_byte_identical_map_terminal_relative_residual": (
                incident_banked_residual
            ),
            "reconstructed_semantically_zero_map_terminal_relative_residual": (
                incident_widened_residual
            ),
            "residual_amplification_factor": (
                incident_widened_residual / incident_banked_residual
            ),
            "orders_of_magnitude": float(
                np.log10(incident_widened_residual / incident_banked_residual)
            ),
            "cpu_reproduction": 7.726885632464709e-6,
            "gpu_reproduction": incident_widened_residual,
            "attribution_boundary": (
                "The rejected reconstruction also traversed the current "
                "construction-time cell-current representation. It corroborates "
                "amplification of representation-scale changes but is not used as "
                "a passive-only eigenvalue sensitivity estimate."
            ),
            "mechanism": (
                "Reconstructing a semantically zero passive-current suffix changed "
                "the fixed source matrix representation and floating reduction "
                "path. The near-marginal nonlinear solve amplified that nominally "
                "null representation difference by more than ten orders in its "
                "terminal residual. The accepted comparison therefore reuses the "
                "original profile, arrays, shapes and reduction order byte-for-byte."
            ),
        },
        "verdict": {
            "classification": classification,
            "ruling": ruling,
            "direction": direction,
            "baseline_contraction_margin": contraction_margin,
            "material_movement_rule": (
                "absolute eigenvalue movement at least ten percent of the "
                "banked contraction margin"
            ),
            "material_movement_threshold": material_threshold,
            "maximum_additional_representation_movement": maximum_movement,
            "movement_fraction_of_baseline_contraction_margin": (
                maximum_movement / contraction_margin
            ),
            "most_moved_representation": most_moved["name"],
            "zero_passive_map_identity": (
                "All representations retain exactly the same active external "
                "field and zero the refitted passive-current suffix."
            ),
        },
        "elapsed_seconds": perf_counter() - started,
    }


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
        stable_translation >= 0.5
        and leading_alignment >= 0.95
        and stable_content["boundary_state_fraction"] < 0.25
        and not algebraic
    )
    if algebraic:
        classification = "ALGEBRAIC_CONSTRAINT_OR_NORMALISATION_MODE"
        mechanism = "affine_flux_coordinate_or_scalar_current_constraint"
    elif physical:
        classification = "PHYSICAL_NEAR_MARGINAL_MODE"
        mechanism = "vertical_free_boundary_axis_displacement_mode"
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
            "A carrier-local mesh mode would concentrate on boundary rows or fail to "
            "project onto a smooth physical template. The leading vector instead "
            "projects predominantly onto the analytic axis-translation subspace, "
            "moves the fitted axis vertically, and has low wall-row and affine "
            "normalisation content. The independent elongated carrier carries a "
            "different radial-dominant mode, so it is not used as false same-mode "
            "evidence."
        ),
        "repair_scope": (
            "A physical near-marginal mode requires route qualification or screening "
            "in nova/equilibrium/forward_operator.py and fixed_point.py. An algebraic "
            "mode would instead require nova/equilibrium/source.py or domain.py. A "
            "carrier-specific mode would require mesh/coupling work in nova/biot/."
        ),
    }


def run() -> dict[str, object]:
    """Measure the vertical pole against passive shell representation."""
    source_commit = measurement_stamp(Path.cwd())
    configure_dtypes()
    return {
        "receipt": {
            "kind": "passive_representation_eigenvalue_response",
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
            "banked_leading_eigenvalue": BASELINE_LEADING_EIGENVALUE,
            "banked_stable_late_growth": STABLE_LATE_GROWTH,
            "banked_stable_resolvent_gain": STABLE_RESOLVENT_GAIN,
        },
        "solver_budget": {
            "stable_warmup": CONTROL_NEWTON_WARMUP,
            "stable_newton_steps": CONTROL_NEWTON_STEPS,
            "stable_gmres_iterations": CONTROL_GMRES_ITERATIONS,
            "dominant_eigenpairs": EIGENPAIR_COUNT,
            "arnoldi_subspace": ARNOLDI_SUBSPACE,
            "passive_tangent_iterations": TANGENT_ITERATIONS,
        },
        "response": _representation_response(),
    }


def _plot(receipt: dict[str, object], path: Path) -> None:
    """Plot eigenvalue movement, fit error and mode localisation."""
    response = receipt["response"]
    representations = response["representations"]
    names = [
        {
            "baseline": "16 at 1.08x",
            "refined_conductor_count": "32 at 1.08x",
            "changed_shell_standoff": "16 at 1.16x",
        }[row["name"]]
        for row in representations
    ]
    eigenvalues = np.asarray(
        [row["leading_eigenvalue"]["real"] for row in representations]
    )
    residuals = np.asarray([row["leading_ritz_residual_l2"] for row in representations])
    fit_errors = np.asarray(
        [
            row["passive_representation"]["translation_mode_weighted_relative_l2_error"]
            for row in representations
        ]
    )
    localisation_keys = [
        "boundary_state_fraction",
        "passive_drive_overlap_fraction",
        "normalisation_content_fraction",
        "axis_translation_content_fraction",
    ]
    localisation_labels = [
        "boundary",
        "passive drive",
        "normalisation",
        "translation",
    ]

    figure, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), constrained_layout=True)
    position = np.arange(len(names))
    movement = 1.0e15 * (eigenvalues - BASELINE_LEADING_EIGENVALUE)
    axes[0].plot(position, movement, marker="o", linewidth=1.5, color="C0")
    axes[0].axhline(0.0, color="0.35", linestyle="--", linewidth=1.0)
    axes[0].text(
        0.03,
        0.97,
        f"banked lambda = {BASELINE_LEADING_EIGENVALUE:.9f}",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
    )
    axes[0].set_xticks(position, names)
    axes[0].set_ylabel("lambda movement from banked [1e-15]")
    axes[0].set_title("vertical-channel pole movement")
    inset = axes[0].inset_axes([0.56, 0.1, 0.4, 0.32])
    inset.semilogy(position, residuals, marker=".", color="C3")
    inset.set_xticks([])
    inset.set_ylabel("Ritz residual", fontsize=7)
    inset.tick_params(labelsize=7)

    axes[1].bar(position, fit_errors, color="C1")
    axes[1].set_xticks(position, names)
    axes[1].set_ylabel("weighted relative L2 error")
    axes[1].set_title("vertical-drive fit")

    width = 0.2
    local_position = np.arange(len(localisation_keys))
    for index, (name, row) in enumerate(zip(names, representations, strict=True)):
        content = row["leading_eigenvector_localisation"]
        axes[2].bar(
            local_position + (index - 1) * width,
            [content[key] for key in localisation_keys],
            width,
            label=name.replace("\n", ", "),
        )
    axes[2].set_xticks(
        local_position,
        localisation_labels,
        rotation=28,
        ha="right",
    )
    axes[2].set_ylabel("squared projection or state fraction")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title("leading-mode localisation")
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
    response = receipt["response"]
    print(
        json.dumps(
            {
                "classification": response["verdict"]["classification"],
                "maximum_eigenvalue_movement": response["verdict"][
                    "maximum_additional_representation_movement"
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
