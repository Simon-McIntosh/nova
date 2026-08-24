"""Re-score the shipped global-surface clip on STEP and ITER equilibria.

The measurement composes clipped-to-contour, clipped-to-TORAX, and
contour-to-TORAX errors directly from the three readers.  Historical
straight-chord and local-bicubic scores are carried as immutable context; they
are never reconstructed by subtracting percentages from another pairing.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Callable
from unittest.mock import patch

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import flux_surface_extraction as extraction
from nova.jax.config import configure_dtypes
from nova.linalg.tensor_spline import TensorBSpline
from nova.transport import torax_geometry_from_fsa
from tests.test_transport_geometry_reference import (
    _contour_geometry,
    _nova_input,
    _relative_error,
    _torax_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/curved-clip-global-surface/global-surface-clip-rescore.json"
)
CASES = {
    "ITER": ("iterhybrid_cocos17.eqdsk", 17),
    "STEP": ("STEP_SPP_001_ECHD_ftop.eqdsk", 1),
}
FIELDS = {
    "ITER": (
        "g2g3_over_rhon_face",
        "g0_face",
        "g1_face",
        "g2_face",
    ),
    "STEP": ("vpr_face", "g1_face"),
}
STEP_GATES_PERCENT = {"vpr_face": 3.92, "g1_face": 6.20}
STRAIGHT_CHORD_PERCENT = {
    "ITER": {
        "g2g3_over_rhon_face": 2.63,
        "g0_face": 0.90,
        "g1_face": 2.08,
        "g2_face": 1.98,
    },
    "STEP": {"vpr_face": 16.46, "g1_face": 9.78},
}
LOCAL_BICUBIC_PERCENT = {
    "ITER": {
        "g2g3_over_rhon_face": 2.96,
        "g0_face": 1.03,
        "g1_face": 2.36,
        "g2_face": 2.31,
    },
    "STEP": {"vpr_face": 18.14, "g1_face": 22.31},
}
UNWIDENED_VERTEX_CAPACITY = 8


def _source_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict(value.tolist())
    if isinstance(value, np.generic):
        return _strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("receipt contains a non-finite measurement")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _contour_fields(contour: Any) -> dict[str, np.ndarray]:
    rho_face = np.asarray(contour.rho_tor_norm)
    g2 = (
        np.asarray(contour.gradient_rho_squared_over_radius_squared)
        * np.asarray(contour.volume_derivative) ** 2
    )
    g2g3 = np.zeros_like(rho_face)
    g2g3[1:] = g2[1:] * np.asarray(contour.inverse_square_radius)[1:] / rho_face[1:]
    return {
        "vpr_face": (
            np.asarray(contour.volume_derivative) * np.asarray(contour.boundary_rho_tor)
        ),
        "g0_face": (
            np.asarray(contour.gradient_rho) * np.asarray(contour.volume_derivative)
        ),
        "g1_face": (
            np.asarray(contour.gradient_rho_squared)
            * np.asarray(contour.volume_derivative) ** 2
        ),
        "g2_face": g2,
        "g2g3_over_rhon_face": g2g3,
    }


def _production_geometry(filename: str, cocos: int) -> tuple[Any, dict[str, Any]]:
    """Run the grid-in production extractor on the reference EQDSK."""
    configure_dtypes()
    data = _nova_input(filename, cocos)
    boundary_major_radius = 0.5 * (
        float(np.max(data["xbdry"])) + float(np.min(data["xbdry"]))
    )
    boundary_field = (
        float(data["bcentr"]) * float(data["xcentr"]) / boundary_major_radius
    )
    record = extraction.extract_flux_surface_geometry(
        jnp.asarray(np.asarray(data["psi"]).T),
        jnp.asarray(data["x"]),
        jnp.asarray(data["z"]),
        jnp.ones((int(data["nz"]), int(data["nx"])), dtype=bool),
        axis_psi=jnp.asarray(data["simagx"]),
        boundary_psi=jnp.asarray(data["sibdry"]),
        profile_coefficients=jnp.zeros(2),
        coefficient_scale=jnp.ones(2),
        ip_amperes=jnp.asarray(data["Ip"]),
        major_radius=jnp.asarray(boundary_major_radius),
        boundary_toroidal_field=jnp.asarray(boundary_field),
        field_function_psi_n=jnp.asarray(data["pnorm"]),
        field_function=jnp.asarray(data["fpol"]),
        n_pressure=1,
        n_diamagnetic=1,
        n_radial_cells=24,
        n_surface_bins=96,
        psi_n_min=jnp.asarray(0.01),
        psi_n_max=jnp.asarray(0.99),
    )
    if not bool(record["valid"]):
        raise RuntimeError(f"production extraction is invalid for {filename}")
    return torax_geometry_from_fsa(record), record


def _route_instrumentation() -> tuple[ExitStack, dict[str, int]]:
    calls = {
        "_surface_clips": 0,
        "fit_tensor_spline": 0,
        "TensorBSpline.cell_coefficients": 0,
        "_tensor_bicubic_coefficients": 0,
        "_clipped_surface_geometry": 0,
    }
    stack = ExitStack()

    original_surface_clips = extraction._surface_clips
    original_fit = extraction.fit_tensor_spline
    original_coefficients = TensorBSpline.cell_coefficients.fget
    if original_coefficients is None:
        raise RuntimeError("TensorBSpline.cell_coefficients has no getter")

    def counted_surface_clips(*args: Any, **kwargs: Any) -> Any:
        calls["_surface_clips"] += 1
        return original_surface_clips(*args, **kwargs)

    def counted_fit(*args: Any, **kwargs: Any) -> Any:
        calls["fit_tensor_spline"] += 1
        return original_fit(*args, **kwargs)

    def counted_coefficients(spline: TensorBSpline) -> Any:
        calls["TensorBSpline.cell_coefficients"] += 1
        return original_coefficients(spline)

    def reject_legacy(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def rejected(*args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            raise RuntimeError(f"measured route entered rejected producer {name}")

        rejected.__wrapped__ = original
        return rejected

    stack.enter_context(
        patch.object(extraction, "_surface_clips", counted_surface_clips)
    )
    stack.enter_context(patch.object(extraction, "fit_tensor_spline", counted_fit))
    stack.enter_context(
        patch.object(
            TensorBSpline,
            "cell_coefficients",
            property(counted_coefficients),
        )
    )
    for name in ("_tensor_bicubic_coefficients", "_clipped_surface_geometry"):
        original = getattr(extraction, name)
        stack.enter_context(
            patch.object(extraction, name, reject_legacy(name, original))
        )
    return stack, calls


def _measure_machine(machine: str, filename: str, cocos: int) -> dict[str, Any]:
    clipped, record = _production_geometry(filename, cocos)
    torax = _torax_geometry(filename, cocos)
    contour = _contour_geometry(filename, cocos, record["rho_face"])
    contour_fields = _contour_fields(contour)

    fields: dict[str, Any] = {}
    for field in FIELDS[machine]:
        clipped_values = np.asarray(getattr(clipped, field))
        contour_values = np.asarray(contour_fields[field])
        torax_values = np.asarray(getattr(torax, field))
        pairings = {
            "clipped_to_contour": 100.0
            * _relative_error(clipped_values, contour_values),
            "clipped_to_torax": 100.0 * _relative_error(clipped_values, torax_values),
            "contour_to_torax": 100.0 * _relative_error(contour_values, torax_values),
        }
        row: dict[str, Any] = {
            "pairings_percent": pairings,
            "banked_clipped_to_contour_percent": {
                "straight_chord": STRAIGHT_CHORD_PERCENT[machine][field],
                "local_bicubic": LOCAL_BICUBIC_PERCENT[machine][field],
            },
        }
        if machine == "STEP":
            gate = STEP_GATES_PERCENT[field]
            row["contour_to_torax_gate_percent"] = gate
            row["clipped_to_contour_passes_gate"] = (
                pairings["clipped_to_contour"] < gate
            )
        fields[field] = row

    required = int(record["clipped_vertex_count_required"])
    used = int(record["clipped_vertex_count_max"])
    capacity = int(record["clipped_vertex_capacity"])
    return {
        "source": {"filename": filename, "cocos": cocos},
        "fields": fields,
        "clipped_vertices": {
            "required": required,
            "used": used,
            "capacity": capacity,
            "unwidened_capacity": UNWIDENED_VERTEX_CAPACITY,
            "required_within_capacity": required <= capacity,
            "used_no_greater_than_required": used <= required,
            "capacity_is_unwidened": capacity == UNWIDENED_VERTEX_CAPACITY,
        },
    }


def measure(output: Path) -> dict[str, Any]:
    """Run the real-equilibrium score and write its complete receipt."""
    stack, calls = _route_instrumentation()
    with stack:
        machines = {
            machine: _measure_machine(machine, *case) for machine, case in CASES.items()
        }

    expected_surface_calls = len(CASES)
    expected_spline_calls = 2 * len(CASES)
    route_checks = {
        "surface_clips_reached": calls["_surface_clips"] == expected_surface_calls,
        "fit_tensor_spline_reached": (
            calls["fit_tensor_spline"] == expected_spline_calls
        ),
        "cell_coefficients_reached": (
            calls["TensorBSpline.cell_coefficients"] == expected_spline_calls
        ),
        "tensor_bicubic_coefficients_not_reached": (
            calls["_tensor_bicubic_coefficients"] == 0
        ),
        "clipped_surface_geometry_not_reached": (
            calls["_clipped_surface_geometry"] == 0
        ),
    }
    route_passes = all(route_checks.values())
    capacity_passes = all(
        all(
            row["clipped_vertices"][key]
            for key in (
                "required_within_capacity",
                "used_no_greater_than_required",
                "capacity_is_unwidened",
            )
        )
        for row in machines.values()
    )
    step_gap_closed = all(
        row["clipped_to_contour_passes_gate"]
        for row in machines["STEP"]["fields"].values()
    )
    step_values = {
        field: row["pairings_percent"]["clipped_to_contour"]
        for field, row in machines["STEP"]["fields"].items()
    }
    if step_gap_closed:
        statement = (
            "Yes. The shipped global-surface clip puts STEP vpr_face at "
            f"{step_values['vpr_face']:.6f}% against 3.92% and g1_face at "
            f"{step_values['g1_face']:.6f}% against 6.20%; both direct "
            "clipped-to-contour scores are below the contour-to-TORAX gates."
        )
    else:
        failed = [
            f"{field} {step_values[field]:.6f}% >= {STEP_GATES_PERCENT[field]:.2f}%"
            for field in FIELDS["STEP"]
            if step_values[field] >= STEP_GATES_PERCENT[field]
        ]
        statement = (
            "No. The shipped global-surface clip does not close the STEP gap "
            "opened by the straight-chord and local-bicubic routes: "
            + "; ".join(failed)
            + "."
        )

    benchmark_path = Path(__file__).resolve()
    payload = {
        "schema_version": 1,
        "measurement": "global_surface_clip_three_way_rescore",
        "method": {
            "pairings_composed_directly": True,
            "pairing_definition": (
                "sup(abs(actual[2:] - expected[2:])) / "
                "max(sup(abs(expected[2:])), 1e-12)"
            ),
            "percentage_subtraction_used": False,
            "current_route": "shipped global tensor-spline surface clip",
            "banked_routes": {
                "straight_chord": "47808904 first parent",
                "local_bicubic": "47808904",
            },
            "attribution": (
                "The historical banks are quoted independently. Causal "
                "attribution across 47808904..current is declined because "
                "more than the curve producer changed in that tree range."
            ),
        },
        "machines": machines,
        "route_verification": {
            "calls": calls,
            "expected_calls": {
                "_surface_clips": expected_surface_calls,
                "fit_tensor_spline": expected_spline_calls,
                "TensorBSpline.cell_coefficients": expected_spline_calls,
                "_tensor_bicubic_coefficients": 0,
                "_clipped_surface_geometry": 0,
            },
            "checks": route_checks,
            "passes": route_passes,
        },
        "capacity_receipt": {
            "unwidened_capacity": UNWIDENED_VERTEX_CAPACITY,
            "passes": capacity_passes,
        },
        "verdict": {
            "measurement_valid": route_passes and capacity_passes,
            "step_gap_closed": step_gap_closed,
            "statement": statement,
        },
        "provenance": {
            "measured_at_utc": datetime.now(UTC).isoformat(),
            "source_revision": _source_revision(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jax_backend": jax.default_backend(),
            "benchmark_sha256": _sha256(benchmark_path),
            "flux_surface_extraction_sha256": _sha256(
                ROOT / "nova/equilibrium/flux_surface_extraction.py"
            ),
            "tensor_spline_sha256": _sha256(ROOT / "nova/linalg/tensor_spline.py"),
            "reference_helpers_sha256": _sha256(
                ROOT / "tests/test_transport_geometry_reference.py"
            ),
        },
    }
    if not payload["verdict"]["measurement_valid"]:
        raise RuntimeError("route or fixed-capacity verification failed")
    _write_json(output, payload)
    return payload


def check(receipt: Path) -> dict[str, Any]:
    """Validate the persisted direct-pairing, route, capacity, and verdict receipt."""
    report = json.loads(receipt.read_text(encoding="utf-8"))
    if report["method"]["pairings_composed_directly"] is not True:
        raise ValueError("receipt does not assert direct pairing composition")
    if report["method"]["percentage_subtraction_used"] is not False:
        raise ValueError("receipt permits percentage subtraction")
    if not report["route_verification"]["passes"]:
        raise ValueError("receipt did not verify the production route")
    if not all(report["route_verification"]["checks"].values()):
        raise ValueError("one or more production-route checks failed")
    if not report["capacity_receipt"]["passes"]:
        raise ValueError("receipt did not verify fixed clipping capacity")
    if report["capacity_receipt"]["unwidened_capacity"] != 8:
        raise ValueError("receipt does not retain clipping capacity 8")

    for machine, fields in FIELDS.items():
        for field in fields:
            row = report["machines"][machine]["fields"][field]
            pairings = row["pairings_percent"]
            if set(pairings) != {
                "clipped_to_contour",
                "clipped_to_torax",
                "contour_to_torax",
            }:
                raise ValueError(f"{machine} {field} lacks a direct three-way score")
            if not all(np.isfinite(float(value)) for value in pairings.values()):
                raise ValueError(f"{machine} {field} has a non-finite score")
            banked = row["banked_clipped_to_contour_percent"]
            if banked["straight_chord"] != STRAIGHT_CHORD_PERCENT[machine][field]:
                raise ValueError(f"{machine} {field} straight-chord bank changed")
            if banked["local_bicubic"] != LOCAL_BICUBIC_PERCENT[machine][field]:
                raise ValueError(f"{machine} {field} local-bicubic bank changed")

    calculated_closed = all(
        report["machines"]["STEP"]["fields"][field]["pairings_percent"][
            "clipped_to_contour"
        ]
        < STEP_GATES_PERCENT[field]
        for field in FIELDS["STEP"]
    )
    if report["verdict"]["step_gap_closed"] is not calculated_closed:
        raise ValueError("STEP closure verdict does not match its direct scores")
    statement = report["verdict"]["statement"]
    if calculated_closed and not statement.startswith("Yes."):
        raise ValueError("positive STEP verdict lacks an explicit yes statement")
    if not calculated_closed and not statement.startswith("No."):
        raise ValueError("negative STEP verdict lacks an explicit no statement")
    if not report["verdict"]["measurement_valid"]:
        raise ValueError("receipt does not carry a valid measurement")
    return report


def _summary(report: dict[str, Any]) -> str:
    step = report["machines"]["STEP"]["fields"]
    iteration = report["machines"]["ITER"]["fields"]
    iteration_values = "/".join(
        f"{iteration[field]['pairings_percent']['clipped_to_torax']:.4f}"
        for field in FIELDS["ITER"]
    )
    return (
        "GLOBAL_SURFACE_CLIP_RESCORE "
        f"step_vpr={step['vpr_face']['pairings_percent']['clipped_to_contour']:.6f}% "
        f"step_g1={step['g1_face']['pairings_percent']['clipped_to_contour']:.6f}% "
        f"iter_coefficients={iteration_values}% "
        f"step_gap_closed={str(report['verdict']['step_gap_closed']).lower()} "
        "route=PASS capacity=PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("measure", "check"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    report = (
        measure(arguments.output)
        if arguments.mode == "measure"
        else check(arguments.output)
    )
    print(_summary(report))


if __name__ == "__main__":
    main()
