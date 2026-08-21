"""Bank an exact Solov'ev-family field with an in-domain X-point."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.constants import mu_0

from nova.equilibrium.topology import boundary_mode
from nova.jax.config import configure_dtypes
from scripts.analytic_oracle_fixtures.measure import (
    FIXTURE_REQUESTS,
    WALL_POINT_COUNT,
    analytic_case,
    cached_machine,
    forward_operator,
)


OUTPUT = Path(__file__).resolve().parent
REPOSITORY_ROOT = OUTPUT.parents[1]
BANK_PATH = OUTPUT / "diverted-state.npz"
RECEIPT_PATH = OUTPUT / "diverted-receipt.json"
RESOLUTION = "fine"

# These two stationary points coincide with well-conditioned interior stencil
# centres on the immutable fine analytic carrier. The coefficients are solved
# from zero-gradient constraints rather than fitted to a topology label.
AXIS_M = np.array([1.7267405315443793, -0.12905304720048028])
X_POINT_M = np.array([1.1577641525044828, -0.50448009360187762])
R_QUARTIC = -0.1
Z_QUADRATIC = -0.8


def _digest(values: np.ndarray) -> str:
    packed = np.ascontiguousarray(values)
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _identity(values: np.ndarray) -> dict[str, object]:
    return {
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "sha256": _digest(values),
    }


def _strict_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _solve_coefficients() -> np.ndarray:
    """Solve the homogeneous Solov'ev terms from the two stationary points."""
    axis_r, axis_z = AXIS_M
    x_r, x_z = X_POINT_M
    system = np.array(
        [
            [
                2.0 * axis_r,
                0.0,
                2.0 * axis_r * axis_z,
                4.0 * axis_r**3 - 8.0 * axis_r * axis_z**2,
            ],
            [0.0, 1.0, axis_r**2, -8.0 * axis_r**2 * axis_z],
            [
                2.0 * x_r,
                0.0,
                2.0 * x_r * x_z,
                4.0 * x_r**3 - 8.0 * x_r * x_z**2,
            ],
            [0.0, 1.0, x_r**2, -8.0 * x_r**2 * x_z],
        ],
        dtype=np.float64,
    )
    forcing = -np.array(
        [
            4.0 * R_QUARTIC * axis_r**3,
            2.0 * Z_QUADRATIC * axis_z,
            4.0 * R_QUARTIC * x_r**3,
            2.0 * Z_QUADRATIC * x_z,
        ],
        dtype=np.float64,
    )
    r_squared, z_linear, r_squared_z, homogeneous_quartic = np.linalg.solve(
        system, forcing
    )
    x_r2 = x_r**2
    x_z2 = x_z**2
    gauge = -(
        R_QUARTIC * x_r2**2
        + Z_QUADRATIC * x_z2
        + r_squared * x_r2
        + z_linear * x_z
        + r_squared_z * x_r2 * x_z
        + homogeneous_quartic * (x_r2**2 - 4.0 * x_r2 * x_z2)
    )
    return np.array(
        [
            R_QUARTIC,
            Z_QUADRATIC,
            gauge,
            r_squared,
            z_linear,
            r_squared_z,
            homogeneous_quartic,
        ],
        dtype="<f8",
    )


def flux(coordinates: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate the exact total poloidal flux in Wb."""
    radius = np.asarray(coordinates, dtype=np.float64)[:, 0]
    height = np.asarray(coordinates, dtype=np.float64)[:, 1]
    alpha, beta, gauge, r2, z1, r2z, quartic = coefficients
    radius2 = radius**2
    height2 = height**2
    return np.asarray(
        alpha * radius2**2
        + beta * height2
        + gauge
        + r2 * radius2
        + z1 * height
        + r2z * radius2 * height
        + quartic * (radius2**2 - 4.0 * radius2 * height2),
        dtype="<f8",
    )


def gradient(point: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate the exact field gradient at one physical point."""
    radius, height = point
    alpha, beta, _gauge, r2, z1, r2z, quartic = coefficients
    return np.array(
        [
            4.0 * alpha * radius**3
            + 2.0 * r2 * radius
            + 2.0 * r2z * radius * height
            + quartic * (4.0 * radius**3 - 8.0 * radius * height**2),
            2.0 * beta * height
            + z1
            + r2z * radius**2
            - 8.0 * quartic * radius**2 * height,
        ],
        dtype=np.float64,
    )


def hessian(point: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Evaluate the exact symmetric field Hessian at one physical point."""
    radius, height = point
    alpha, beta, _gauge, r2, _z1, r2z, quartic = coefficients
    mixed = 2.0 * r2z * radius - 16.0 * quartic * radius * height
    return np.array(
        [
            [
                12.0 * alpha * radius**2
                + 2.0 * r2
                + 2.0 * r2z * height
                + quartic * (12.0 * radius**2 - 8.0 * height**2),
                mixed,
            ],
            [mixed, 2.0 * beta - 8.0 * quartic * radius**2],
        ],
        dtype=np.float64,
    )


def _inside_polygon(point: np.ndarray, vertices: np.ndarray) -> bool:
    """Return whether a point lies strictly inside a simple closed polygon."""
    x_coordinate, y_coordinate = point
    inside = False
    previous = vertices[-1]
    for current in vertices:
        x_current, y_current = current
        x_previous, y_previous = previous
        crosses = (y_current > y_coordinate) != (y_previous > y_coordinate)
        if crosses:
            crossing_x = x_current + (y_coordinate - y_current) * (
                x_previous - x_current
            ) / (y_previous - y_current)
            if x_coordinate < crossing_x:
                inside = not inside
        previous = current
    return inside


def _finite_rows(values) -> list[list[float]]:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array[:, 0])].tolist()


def _production_read(operator, state: np.ndarray) -> dict[str, object]:
    grid_flux, _wall_flux = operator.topology.split_flux_map(jnp.asarray(state))
    extrema, saddles = operator.topology.grid(grid_flux)
    masks, topology = operator.read(jnp.asarray(state))
    finite_extrema = _finite_rows(extrema)
    finite_saddles = _finite_rows(saddles)
    return {
        "class": boundary_mode(topology).value,
        "diverted": bool(topology.diverted),
        "axis_m": np.asarray(topology.axis, dtype=np.float64).tolist(),
        "axis_flux_wb": float(topology.axis_flux),
        "boundary_point_m": np.asarray(topology.boundary, dtype=np.float64).tolist(),
        "boundary_flux_wb": float(topology.boundary_flux),
        "wall_contact_point_m": np.asarray(
            topology.wall_point, dtype=np.float64
        ).tolist(),
        "wall_contact_flux_wb": float(topology.wall_point_flux),
        "finite_o_point_count": len(finite_extrema),
        "finite_o_points": finite_extrema,
        "finite_x_point_count": len(finite_saddles),
        "finite_x_points": finite_saddles,
        "selected_x_point_m": np.asarray(topology.x_point, dtype=np.float64).tolist(),
        "selected_x_point_flux_wb": float(topology.x_point_flux),
        "boundary_x_point_distance_m": float(
            np.linalg.norm(np.asarray(topology.boundary) - topology.x_point)
        ),
        "boundary_x_point_flux_difference_wb": float(
            topology.boundary_flux - topology.x_point_flux
        ),
        "boundary_wall_point_distance_m": float(
            np.linalg.norm(np.asarray(topology.boundary) - topology.wall_point)
        ),
        "boundary_wall_flux_difference_wb": float(
            topology.boundary_flux - topology.wall_point_flux
        ),
        "core_cell_count": int(np.asarray(masks.core).sum()),
        "state_precision": str(state.dtype),
        "null_fit_precision": str(operator.topology.grid.fit_dtype),
    }


def build() -> dict[str, object]:
    """Build the bank and return its strict machine-readable receipt."""
    configure_dtypes()
    case = analytic_case()
    machine = cached_machine(
        case, FIXTURE_REQUESTS[RESOLUTION], wall_nodes=WALL_POINT_COUNT
    )
    if not machine.cache["hit"]:
        raise RuntimeError("the fine semantic carrier was not a warm cache hit")
    coefficients = _solve_coefficients()
    stationary_points = np.vstack((AXIS_M, X_POINT_M)).astype("<f8")
    coordinates = np.vstack(
        (machine.node, machine.wall_node, machine.sample_coordinates)
    )
    state = flux(coordinates, coefficients)
    np.savez(
        BANK_PATH,
        state=state,
        coefficients=coefficients,
        stationary_points=stationary_points,
    )

    operator = forward_operator(case, machine)
    production = _production_read(operator, state)
    axis_hessian = hessian(AXIS_M, coefficients)
    x_hessian = hessian(X_POINT_M, coefficients)
    axis_eigenvalues = np.linalg.eigvalsh(axis_hessian)
    x_eigenvalues = np.linalg.eigvalsh(x_hessian)
    x_wall_distance = float(
        np.min(np.linalg.norm(machine.wall_node - X_POINT_M, axis=1))
    )
    x_locator_error = float(
        np.linalg.norm(np.asarray(production["selected_x_point_m"]) - X_POINT_M)
    )
    if production["class"] != "diverted":
        raise AssertionError("the stored state did not produce a diverted read")
    if production["finite_x_point_count"] != 1:
        raise AssertionError("the stored state did not produce exactly one X-point")
    if not (x_eigenvalues[0] < 0.0 < x_eigenvalues[1]):
        raise AssertionError("the intended X-point Hessian is not indefinite")
    if not _inside_polygon(X_POINT_M, machine.wall_node):
        raise AssertionError("the intended X-point is not inside the wall")
    if x_locator_error > 0.01:
        raise AssertionError("the production X-point read is too far from the oracle")

    arrays = {
        "state": state,
        "coefficients": coefficients,
        "stationary_points": stationary_points,
    }
    receipt = {
        "schema": "nova.diverted-analytic-oracle",
        "schema_version": 1,
        "family": "static Solov'ev polynomial with homogeneous free-boundary terms",
        "bank": str(BANK_PATH.relative_to(REPOSITORY_ROOT)),
        "arrays": {name: _identity(values) for name, values in arrays.items()},
        "carrier": {
            "resolution": RESOLUTION,
            "requested_cells": FIXTURE_REQUESTS[RESOLUTION],
            "realised_cells": len(machine.node),
            "cache_semantic_key": machine.cache["semantic_key"],
            "cache_warm_hit": bool(machine.cache["hit"]),
            "state_layout": [
                ["grid", len(machine.node)],
                ["wall", len(machine.wall_node)],
                ["direct_sample", len(machine.sample_coordinates)],
            ],
        },
        "closed_form": {
            "formula": ("Phi=a*R^4+b*Z^2+c0+c1*R^2+c2*Z+c3*R^2*Z+c4*(R^4-4*R^2*Z^2)"),
            "coefficient_order": ["a", "b", "c0", "c1", "c2", "c3", "c4"],
            "coefficients": coefficients.tolist(),
            "delta_star_phi_wb_per_m2": "8*a*R^2 + 2*b",
            "constant_flux_functions": {
                "p_prime_pa_per_wb": float(2.0 * coefficients[0] / (np.pi**2 * mu_0)),
                "ff_prime_t2_m2_per_wb": float(coefficients[1] / (2.0 * np.pi**2)),
            },
            "gauge": "Phi is exactly zero at the analytic X-point",
        },
        "analytic_stationary_points": {
            "axis": {
                "coordinate_m": AXIS_M.tolist(),
                "gradient_wb_per_m": gradient(AXIS_M, coefficients).tolist(),
                "flux_wb": float(flux(AXIS_M[None, :], coefficients)[0]),
                "hessian_wb_per_m2": axis_hessian.tolist(),
                "hessian_eigenvalues_wb_per_m2": axis_eigenvalues.tolist(),
                "hessian_determinant_wb2_per_m4": float(np.linalg.det(axis_hessian)),
                "kind": "maximum_magnetic_axis",
            },
            "x_point": {
                "coordinate_m": X_POINT_M.tolist(),
                "gradient_wb_per_m": gradient(X_POINT_M, coefficients).tolist(),
                "flux_wb": float(flux(X_POINT_M[None, :], coefficients)[0]),
                "hessian_wb_per_m2": x_hessian.tolist(),
                "hessian_eigenvalues_wb_per_m2": x_eigenvalues.tolist(),
                "hessian_determinant_wb2_per_m4": float(np.linalg.det(x_hessian)),
                "kind": "saddle",
                "inside_wall_polygon": True,
                "nearest_wall_vertex_distance_m": x_wall_distance,
            },
        },
        "stored_precision_production_read": production,
        "evidence": {
            "jax_backend": jax.default_backend(),
            "jax_x64_enabled": bool(jax.config.x64_enabled),
            "state_precision": "binary64",
            "topology_locator_precision": "float64",
            "receipt_label_used_as_classification_input": False,
            "production_x_point_localization_error_m": x_locator_error,
        },
    }
    _strict_json(RECEIPT_PATH, receipt)
    return receipt


def main() -> None:
    receipt = build()
    production = receipt["stored_precision_production_read"]
    saddle = receipt["analytic_stationary_points"]["x_point"]
    print(
        "BANKED "
        f"class={production['class']} "
        f"x_point={production['selected_x_point_m']} "
        f"hessian_eigenvalues={saddle['hessian_eigenvalues_wb_per_m2']} "
        f"receipt={RECEIPT_PATH}"
    )


if __name__ == "__main__":
    main()
