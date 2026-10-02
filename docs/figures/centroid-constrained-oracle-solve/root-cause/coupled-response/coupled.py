"""Measure the formal fixed-point response of the analytic centroid state."""

from __future__ import annotations

import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from benchmarks import centroid_constrained_fixture_receipt as fixture
from measure import CASE, OUTPUT, REQUESTED_CELLS, REQUESTED_CLASS, _array
from nova.equilibrium.forward_operator import set_support_clip_mode
from nova.equilibrium.observation import MomentIntegralSupport
from nova.jax.config import configure_dtypes


def main():
    configure_dtypes()
    assert jax.config.jax_enable_x64
    set_support_clip_mode("exact")
    context = fixture._context(CASE, REQUESTED_CELLS)
    profile = context["profile"]
    operator = profile.operator
    target = context["target_current"]
    state = jnp.asarray(context["analytic"], dtype=jnp.float64)
    external = operator.external()
    traced = operator.traced_flux_map(REQUESTED_CLASS, target)
    columns = _array(operator.prescribed_current_field.response)
    assert columns.shape == (state.size, 3)
    saved = json.loads((OUTPUT / "derivatives.json").read_text())
    centroid_baseline = json.loads((OUTPUT / "centroids.json").read_text())
    started = time.perf_counter()

    def mapped(psi):
        return traced(psi, external, operator, target)

    def observed(psi):
        row = profile.current_moment_observation(
            psi, support=MomentIntegralSupport.ALL_DOMAIN, target_current=target
        )
        return jnp.stack((row.centroid_r, row.centroid_z))

    print("linearizing map and row-native centroid", flush=True)
    base_map, tangent_map = jax.linearize(mapped, state)
    _, tangent_centroid = jax.linearize(observed, state)
    tangent_map = jax.jit(tangent_map)
    tangent_centroid = jax.jit(tangent_centroid)

    def map_product(value):
        return _array(tangent_map(jnp.asarray(value, dtype=jnp.float64)))

    def centroid_product(value):
        return _array(tangent_centroid(jnp.asarray(value, dtype=jnp.float64)))

    size = state.size
    direct = np.column_stack([centroid_product(columns[:, i]) for i in range(3)])
    recorded = np.asarray(saved["row_native_jvp_m_per_unit"])
    if not np.allclose(direct, recorded, rtol=1e-9, atol=1e-9):
        raise RuntimeError("fresh centroid JVP disagrees with saved derivative receipt")
    zero_states = np.linalg.solve(np.eye(size), columns)
    zero_map_control = np.column_stack(
        [centroid_product(zero_states[:, i]) for i in range(3)]
    )
    if not np.allclose(zero_map_control, direct, rtol=1e-12, atol=1e-12):
        raise RuntimeError("zero-map control did not reduce to direct response")
    print(f"direct JVP confirmed: {direct.tolist()}", flush=True)

    def a_product(value):
        return value - map_product(value)

    a_operator = LinearOperator((size, size), matvec=a_product, dtype=np.float64)
    diagnostics = []

    def solve(rhs, label):
        residuals = []
        answer, info = gmres(
            a_operator,
            rhs,
            rtol=1e-9,
            atol=1e-11,
            restart=80,
            maxiter=12,
            callback=lambda norm: residuals.append(float(norm)),
            callback_type="pr_norm",
        )
        relative = float(np.linalg.norm(a_product(answer) - rhs) / np.linalg.norm(rhs))
        receipt = {
            "name": label,
            "info": int(info),
            "iterations": len(residuals),
            "relative_residual": relative,
        }
        diagnostics.append(receipt)
        print(f"linear solve: {receipt}", flush=True)
        if info or relative > 2e-8:
            raise RuntimeError(f"linear response did not converge: {receipt}")
        return answer

    coupled_states = np.column_stack(
        [solve(columns[:, i], f"column_{i}") for i in range(3)]
    )
    coupled = np.column_stack(
        [centroid_product(coupled_states[:, i]) for i in range(3)]
    )
    residual = _array(base_map - state)
    residual_state = solve(residual, "analytic_map_residual")
    residual_centroid = centroid_product(residual_state)
    offset = np.asarray(centroid_baseline["analytic"]["production_offset_m"])
    field_only = np.linalg.solve(coupled[:, :2], -offset)
    corrected = np.linalg.solve(coupled[:, :2], -offset - residual_centroid)
    result = {
        "interpretation": (
            "formal JVP linearization; central differences disagree with D_c"
        ),
        "direct_centroid_jvp_m_per_unit": direct.tolist(),
        "zero_map_control_m_per_unit": zero_map_control.tolist(),
        "coupled_centroid_jvp_m_per_unit": coupled.tolist(),
        "analytic_map_residual_sup_wb": float(np.max(np.abs(residual))),
        "analytic_map_residual_centroid_effect_m": residual_centroid.tolist(),
        "field_only_cancellation_t": field_only.tolist(),
        "residual_corrected_cancellation_t": corrected.tolist(),
        "observed_terminal_field_t": -0.021255810528996315,
        "linear_solve_diagnostics": diagnostics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    partial = OUTPUT / "response-partial.json"
    partial.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {partial}", flush=True)

    basis = np.eye(size, dtype=np.float64)
    jacobian = np.empty((size, size), dtype=np.float64)
    for index in range(size):
        jacobian[:, index] = map_product(basis[:, index])
        if (index + 1) % 64 == 0:
            print(f"Jacobian columns {index + 1}/{size}", flush=True)
    fixed_point_matrix = np.eye(size) - jacobian
    eigenvalues = np.linalg.eigvals(jacobian)
    nearest = sorted(eigenvalues, key=lambda value: abs(value - 1.0))[:5]
    result.update(
        {
            "fixed_point_matrix_condition_2": float(np.linalg.cond(fixed_point_matrix)),
            "fixed_point_matrix_smallest_singular_value": float(
                np.linalg.svd(fixed_point_matrix, compute_uv=False)[-1]
            ),
            "jacobian_spectral_radius": float(np.max(np.abs(eigenvalues))),
            "jacobian_eigenvalues_nearest_one": [
                [float(value.real), float(value.imag)] for value in nearest
            ],
            "dense_solve_residual_relative": [
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
    path = OUTPUT / "response.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"wrote {path}; condition={result['fixed_point_matrix_condition_2']}")


if __name__ == "__main__":
    main()
