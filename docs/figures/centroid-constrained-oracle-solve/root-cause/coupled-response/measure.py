"""Read centroid moments and the analytic-state fixed-point susceptibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from benchmarks import centroid_constrained_fixture_receipt as fixture
from benchmarks.plasma_cell_map_fidelity import (
    current_centroid,
    recover_physical_first_moments,
)
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.observation import MomentIntegralSupport
from nova.jax.config import configure_dtypes


CASE = "weak-rotation-reactor-static"
REQUESTED_CELLS = -110
REQUESTED_CLASS = 0
STATE_PATH = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20261002T133233397975-cco-cap-factor-finite-tangent/"
    "h200-receipt/control-positive-state.npy"
)
OUTPUT = Path(__file__).resolve().parent


def _array(value):
    return np.asarray(jax.block_until_ready(value), dtype=np.float64)


def _digest(value):
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype="<f8").tobytes()
    ).hexdigest()


def _moments(context, state):
    profile = context["profile"]
    operator = profile.operator
    target = context["target_current"]
    moments, amplitude = operator.normalised_current_moments(
        jnp.asarray(state), target, REQUESTED_CLASS
    )
    current = _array(moments.cell_current)
    centres = _array(operator.moment_geometry.atomic_mesh.centroids)
    row = profile.current_moment_observation(
        jnp.asarray(state),
        support=MomentIntegralSupport.ALL_DOMAIN,
        requested_class=REQUESTED_CLASS,
        target_current=target,
    )
    production = np.array([float(row.centroid_r), float(row.centroid_z)])
    centre_only = np.sum(current[:, None] * centres, axis=0) / current.sum()
    second = _array(operator.moment_geometry.second_moment)
    radial, vertical = recover_physical_first_moments(
        second, _array(moments.radial_moment), _array(moments.vertical_moment)
    )
    first = np.array(current_centroid(centres, current, radial, vertical))
    target_centroid = np.asarray(context["centroid"], dtype=np.float64)
    if not np.allclose(production, centre_only, rtol=0, atol=2e-12):
        raise RuntimeError("production observation and cell-centre sum disagree")
    return {
        "production_centroid_m": production.tolist(),
        "first_moment_centroid_m": first.tolist(),
        "continuous_analytic_centroid_m": target_centroid.tolist(),
        "production_offset_m": (production - target_centroid).tolist(),
        "first_moment_offset_m": (first - target_centroid).tolist(),
        "within_cell_contribution_m": (first - production).tolist(),
        "cell_current_sum_a": float(current.sum()),
        "normalisation_amplitude": float(amplitude),
        "radial_first_moment_sum_a_m": float(radial.sum()),
        "vertical_first_moment_sum_a_m": float(vertical.sum()),
        "nonzero_current_cells": int(np.count_nonzero(current)),
        "observation_minus_cell_centre_m": (production - centre_only).tolist(),
    }


def centroids(context):
    analytic = np.asarray(context["analytic"], dtype=np.float64)
    terminal = np.load(STATE_PATH, allow_pickle=False)
    if analytic.shape != terminal.shape or analytic.shape != (772,):
        raise RuntimeError(
            "unexpected analytic/terminal state shapes "
            f"{analytic.shape}/{terminal.shape}"
        )
    if len(context["profile"].operator.grid.coordinate) != 135:
        raise RuntimeError("the case lost its 135-cell carrier")
    return {
        "case": CASE,
        "requested_cells": REQUESTED_CELLS,
        "realised_cells": 135,
        "state_length": len(analytic),
        "clip_mode": "exact",
        "analytic_state_digest": _digest(analytic),
        "terminal_state_path": str(STATE_PATH),
        "terminal_state_digest": _digest(terminal),
        "analytic": _moments(context, analytic),
        "terminal": _moments(context, terminal),
        "first_moment_source": (
            "benchmarks/plasma_cell_map_fidelity.py::"
            "recover_physical_first_moments/current_centroid"
        ),
    }


def response(context, centroid_receipt):
    profile = context["profile"]
    operator = profile.operator
    target = context["target_current"]
    state = jnp.asarray(context["analytic"], dtype=jnp.float64)
    external = operator.external()
    traced = operator.traced_flux_map(REQUESTED_CLASS, target)

    def mapped(psi):
        return traced(psi, external, operator, target)

    def observed(psi):
        row = profile.current_moment_observation(
            psi,
            support=MomentIntegralSupport.ALL_DOMAIN,
            target_current=target,
        )
        return jnp.stack((row.centroid_r, row.centroid_z))

    def observed_requested(psi):
        row = profile.current_moment_observation(
            psi,
            support=MomentIntegralSupport.ALL_DOMAIN,
            requested_class=REQUESTED_CLASS,
            target_current=target,
        )
        return jnp.stack((row.centroid_r, row.centroid_z))

    started = time.perf_counter()
    print("linearizing map and centroid observation", flush=True)
    base_map, tangent_map = jax.linearize(mapped, state)
    base_centroid, tangent_centroid = jax.linearize(observed, state)
    _, tangent_requested = jax.linearize(observed_requested, state)
    tangent_map = jax.jit(tangent_map)
    tangent_centroid = jax.jit(tangent_centroid)
    tangent_requested = jax.jit(tangent_requested)
    columns = _array(operator.prescribed_current_field.response)
    if columns.shape != (state.size, 3):
        raise RuntimeError(f"unexpected compensator columns {columns.shape}")
    n = state.size

    def jacobian_product(value):
        return _array(tangent_map(jnp.asarray(value, dtype=jnp.float64)))

    def centroid_product(value):
        return _array(tangent_centroid(jnp.asarray(value, dtype=jnp.float64)))

    def a_product(value):
        return value - jacobian_product(value)

    operator_a = LinearOperator((n, n), matvec=a_product, dtype=np.float64)
    diagnostics = []

    def solve(rhs, label):
        residuals = []
        solution, info = gmres(
            operator_a,
            rhs,
            rtol=1e-9,
            atol=1e-11,
            restart=80,
            maxiter=12,
            callback=lambda norm: residuals.append(float(norm)),
            callback_type="pr_norm",
        )
        relative = float(
            np.linalg.norm(a_product(solution) - rhs) / np.linalg.norm(rhs)
        )
        diagnostics.append(
            {
                "name": label,
                "info": int(info),
                "iterations": len(residuals),
                "relative_residual": relative,
                "last_preconditioned_residual": residuals[-1] if residuals else None,
            }
        )
        if info != 0 or relative > 2e-8:
            raise RuntimeError(
                f"linear response did not converge for {label}: {diagnostics[-1]}"
            )
        print(f"solved {label}: {diagnostics[-1]}", flush=True)
        return solution

    # The direct arm is the declared negative control for the inverse response.
    direct = np.column_stack([centroid_product(columns[:, i]) for i in range(3)])
    requested_direct = np.column_stack(
        [_array(tangent_requested(jnp.asarray(columns[:, i]))) for i in range(3)]
    )
    step_t = 1.0e-6
    finite = np.column_stack(
        [
            (
                _array(observed(state + step_t * columns[:, i]))
                - _array(observed(state - step_t * columns[:, i]))
            )
            / (2 * step_t)
            for i in range(3)
        ]
    )
    requested_finite = np.column_stack(
        [
            (
                _array(observed_requested(state + step_t * columns[:, i]))
                - _array(observed_requested(state - step_t * columns[:, i]))
            )
            / (2 * step_t)
            for i in range(3)
        ]
    )

    def support_read(psi, requested_class):
        moments, amplitude = operator.normalised_current_moments(
            psi, target, requested_class
        )
        mask = _array(moments.cell_current) != 0.0
        return {
            "active_cell_indices": np.flatnonzero(mask).tolist(),
            "active_cell_count": int(np.count_nonzero(mask)),
            "normalisation_amplitude": float(amplitude),
            "normalised_current_a": float(np.sum(_array(moments.cell_current))),
        }

    support = {}
    for name, requested_class in (
        ("row_native", None),
        ("requested_limited", REQUESTED_CLASS),
    ):
        support[name] = {"base": support_read(state, requested_class), "columns": []}
        for index in range(3):
            support[name]["columns"].append(
                {
                    "minus": support_read(
                        state - step_t * columns[:, index], requested_class
                    ),
                    "plus": support_read(
                        state + step_t * columns[:, index], requested_class
                    ),
                }
            )
    diagnostics_receipt = {
        "state_digest": _digest(np.asarray(state)),
        "column_order": ["vertical_t", "radial_t", "level_wb"],
        "field_column_units": (
            "Wb per tesla for vertical and radial; Wb per weber for level"
        ),
        "closed_form_column_sup_error_wb_per_unit": [
            float(
                np.max(
                    np.abs(
                        columns[:, 0]
                        - np.pi * (context["coordinates"][:, 0] ** 2 - 6.2**2)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        columns[:, 1]
                        + 2
                        * np.pi
                        * context["coordinates"][:, 0]
                        * context["coordinates"][:, 1]
                    )
                )
            ),
            float(np.max(np.abs(columns[:, 2] - 1.0))),
        ],
        "finite_difference_step": step_t,
        "row_native_centroid_m": _array(base_centroid).tolist(),
        "row_native_jvp_m_per_unit": direct.tolist(),
        "row_native_central_m_per_unit": finite.tolist(),
        "requested_limited_jvp_m_per_unit": requested_direct.tolist(),
        "requested_limited_central_m_per_unit": requested_finite.tolist(),
        "historical_radial_response_m_per_t": 1.62922526181,
        "active_support_and_normalisation": support,
        "row_source": (
            "nova/equilibrium/constraint.py::CurrentCentroidConstraint.observed"
        ),
    }
    diagnostics_path = OUTPUT / "derivatives.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics_receipt, indent=2, allow_nan=False) + "\n"
    )
    print(
        f"wrote {diagnostics_path}; {direct.tolist()} versus {finite.tolist()}",
        flush=True,
    )
    print("waiting for committed derivative receipt before inverse", flush=True)
    deadline = time.time() + 8 * 60
    while not (OUTPUT / "continue-response").exists():
        if (OUTPUT / "stop-response").exists() or time.time() > deadline:
            raise SystemExit(2)
        time.sleep(1)
    zero_states = np.linalg.solve(np.eye(n), columns).T
    zero_jacobian_control = np.column_stack(
        [centroid_product(value) for value in zero_states]
    )
    if not np.allclose(zero_jacobian_control, direct, rtol=1e-10, atol=1e-10):
        raise RuntimeError("zero-Jacobian control did not recover direct leverage")
    if abs(direct[0, 0] - 1.62922526181) > 0.01:
        raise RuntimeError("direct radial leverage missed its historical control")
    print(f"direct response {direct.tolist()}", flush=True)
    coupled_states = np.column_stack(
        [solve(columns[:, i], f"column_{i}") for i in range(3)]
    )
    coupled = np.column_stack(
        [centroid_product(coupled_states[:, i]) for i in range(3)]
    )
    fixed_point_residual = _array(base_map - state)
    residual_state = solve(fixed_point_residual, "analytic_map_residual")
    residual_centroid = centroid_product(residual_state)
    baseline = np.asarray(centroid_receipt["analytic"]["production_offset_m"])
    field_only = np.linalg.solve(coupled[:, :2], -baseline)
    with_residual = np.linalg.solve(coupled[:, :2], -baseline - residual_centroid)
    source_receipt = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "cap-factor-repair/control-positive.json"
        ).read_text()
    )
    observed_amplitudes = np.asarray(
        source_receipt["solve"]["compensating_amplitudes"], dtype=np.float64
    )
    terminal = np.load(STATE_PATH, allow_pickle=False)
    predicted = (
        np.asarray(state) + residual_state + coupled_states @ observed_amplitudes
    )
    observed_delta = terminal - np.asarray(state)
    predicted_delta = predicted - np.asarray(state)
    anchor = int(
        np.argmin(
            np.linalg.norm(
                np.asarray(context["coordinates"]) - np.asarray((6.2, 0.0)), axis=1
            )
        )
    )
    observed_shape = observed_delta - observed_delta[anchor]
    predicted_shape = predicted_delta - predicted_delta[anchor]
    shape_error = predicted_shape - observed_shape
    shape_rmse = float(np.sqrt(np.mean(shape_error**2)))
    observed_shape_rmse = float(np.sqrt(np.mean(observed_shape**2)))
    shape_correlation = float(np.corrcoef(predicted_shape, observed_shape)[0, 1])
    result = {
        "direct_centroid_response_m_per_unit": direct.tolist(),
        "zero_jacobian_control_m_per_unit": zero_jacobian_control.tolist(),
        "coupled_centroid_response_m_per_unit": coupled.tolist(),
        "analytic_map_residual_sup_wb": float(np.max(np.abs(fixed_point_residual))),
        "analytic_map_residual_centroid_effect_m": residual_centroid.tolist(),
        "field_only_cancellation_t": field_only.tolist(),
        "residual_corrected_cancellation_t": with_residual.tolist(),
        "observed_terminal_field_t": -0.021255810528996315,
        "observed_terminal_amplitudes": observed_amplitudes.tolist(),
        "direct_radial_field_prediction_t": float(-baseline[0] / direct[0, 0]),
        "linear_terminal_shape_anchor_index": anchor,
        "linear_terminal_shape_rmse_wb": shape_rmse,
        "observed_terminal_shape_rmse_wb": observed_shape_rmse,
        "linear_terminal_shape_error_over_observed": shape_rmse / observed_shape_rmse,
        "linear_terminal_shape_correlation": shape_correlation,
        "linear_terminal_shape_error_sup_wb": float(np.max(np.abs(shape_error))),
        "linear_terminal_shape_predicted_range_wb": [
            float(np.min(predicted_shape)),
            float(np.max(predicted_shape)),
        ],
        "linear_terminal_shape_observed_range_wb": [
            float(np.min(observed_shape)),
            float(np.max(observed_shape)),
        ],
        "linear_solve_diagnostics": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    partial = OUTPUT / "response-partial.json"
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        f"wrote {partial}; assembling Jacobian for condition and eigenvalues",
        flush=True,
    )
    basis = np.eye(n, dtype=np.float64)
    jacobian = np.empty((n, n), dtype=np.float64)
    for index in range(n):
        jacobian[:, index] = jacobian_product(basis[:, index])
        if (index + 1) % 64 == 0:
            print(f"Jacobian columns {index + 1}/{n}", flush=True)
    fixed_point_matrix = np.eye(n) - jacobian
    eigenvalues = np.linalg.eigvals(jacobian)
    nearest = min(eigenvalues, key=lambda value: abs(value - 1.0))
    result.update(
        {
            "fixed_point_matrix_condition_2": float(np.linalg.cond(fixed_point_matrix)),
            "fixed_point_matrix_smallest_singular_value": float(
                np.linalg.svd(fixed_point_matrix, compute_uv=False)[-1]
            ),
            "jacobian_spectral_radius": float(np.max(np.abs(eigenvalues))),
            "jacobian_eigenvalue_nearest_one": [
                float(nearest.real),
                float(nearest.imag),
            ],
            "jacobian_eigenvalue_nearest_one_distance": float(abs(nearest - 1.0)),
            "jacobian_dense_solve_residual_relative": [
                float(
                    np.linalg.norm(
                        fixed_point_matrix @ coupled_states[:, i] - columns[:, i]
                    )
                    / np.linalg.norm(columns[:, i])
                )
                for i in range(3)
            ],
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("centroids", "response"))
    args = parser.parse_args()
    configure_dtypes()
    if not jax.config.jax_enable_x64:
        raise RuntimeError("binary64 disabled")
    set_support_clip_mode("exact")
    context = fixture._context(CASE, REQUESTED_CELLS)
    if args.mode == "centroids":
        result = centroids(context)
    else:
        baseline = json.loads((OUTPUT / "centroids.json").read_text())
        result = response(context, baseline)
    path = OUTPUT / f"{args.mode}.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "summary": result}, allow_nan=False))


if __name__ == "__main__":
    main()
