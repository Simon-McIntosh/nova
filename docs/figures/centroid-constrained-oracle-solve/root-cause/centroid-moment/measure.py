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
from scipy.sparse.linalg import LinearOperator, eigs, gmres

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
            requested_class=REQUESTED_CLASS,
            target_current=target,
        )
        return jnp.stack((row.centroid_r, row.centroid_z))

    started = time.perf_counter()
    base_map, tangent_map = jax.linearize(mapped, state)
    base_centroid, tangent_centroid = jax.linearize(observed, state)
    tangent_map = jax.jit(tangent_map)
    tangent_centroid = jax.jit(tangent_centroid)
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
    operator_j = LinearOperator((n, n), matvec=jacobian_product, dtype=np.float64)
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
        return solution

    # The direct arm is the declared negative control for the inverse response.
    direct = np.column_stack([centroid_product(columns[:, i]) for i in range(3)])
    zero_jacobian_control = direct.copy()
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
    eigenvalues = eigs(
        operator_j, k=6, which="LM", return_eigenvectors=False, tol=1e-5, maxiter=500
    )
    near_one = min(eigenvalues, key=lambda value: abs(value - 1.0))
    return {
        "direct_centroid_response_m_per_unit": direct.tolist(),
        "zero_jacobian_control_m_per_unit": zero_jacobian_control.tolist(),
        "coupled_centroid_response_m_per_unit": coupled.tolist(),
        "analytic_map_residual_sup_wb": float(np.max(np.abs(fixed_point_residual))),
        "analytic_map_residual_centroid_effect_m": residual_centroid.tolist(),
        "field_only_cancellation_t": field_only.tolist(),
        "residual_corrected_cancellation_t": with_residual.tolist(),
        "observed_terminal_field_t": -0.021255810528996315,
        "direct_radial_field_prediction_t": float(-baseline[0] / direct[0, 0]),
        "jacobian_eigenvalues_largest_magnitude": [
            [float(v.real), float(v.imag)] for v in eigenvalues
        ],
        "eigenvalue_closest_to_one_among_computed": [
            float(near_one.real),
            float(near_one.imag),
        ],
        "linear_solve_diagnostics": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }


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
