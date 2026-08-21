"""Measure flux-surface record assembly on one real ITER equilibrium map."""

from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from nova.io import geqdsk
from nova.jax.config import configure_dtypes


RESULT_COLUMNS = (
    "route",
    "device",
    "execution",
    "batch_size",
    "map_shape",
    "timing",
    "repetitions",
    "median_total_seconds",
    "median_per_map_seconds",
    "minimum_total_seconds",
    "maximum_total_seconds",
    "source_revision",
    "backend",
    "device_model",
    "slurm_job_id",
    "status",
    "detail",
)
ITER_FILENAME = "iterhybrid_cocos17.eqdsk"
RADIAL_CELLS = 24
SURFACE_BINS = 28


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--route", choices=("service", "gaussian_baseline", "host_contour")
    )
    parser.add_argument("--batch-size", type=int, choices=(1, 8), default=1)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--initialize", action="store_true", help="write only the TSV header"
    )
    return parser.parse_args()


def _initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, delimiter="\t").writeheader()


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, delimiter="\t").writerow(row)


def _cpu_model() -> str:
    model = platform.processor().strip()
    if model:
        return model
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.machine()


def _device_metadata() -> tuple[str, str, str]:
    backend = jax.default_backend()
    devices = jax.devices()
    model = devices[0].device_kind if devices else _cpu_model()
    label = "H200" if "H200" in model.upper() else "CPU"
    return label, backend, model


def _input() -> dict[str, Any]:
    import torax

    directory = Path(torax.__file__).parent / "data" / "third_party" / "geo"
    data = geqdsk.read(str(directory / ITER_FILENAME))
    shape = np.asarray(data["psi"]).shape
    if shape != (129, 129):
        raise ValueError(f"expected the 129x129 ITER map, got {shape}")
    boundary_major_radius = 0.5 * (
        float(np.max(data["xbdry"])) + float(np.min(data["xbdry"]))
    )
    boundary_field = (
        float(data["bcentr"]) * float(data["xcentr"]) / boundary_major_radius
    )
    return {
        "data": data,
        "psi": jnp.asarray(np.asarray(data["psi"]).T),
        "radius": jnp.asarray(data["x"]),
        "height": jnp.asarray(data["z"]),
        "inside_limiter": jnp.ones((int(data["nz"]), int(data["nx"])), dtype=bool),
        "axis_psi": jnp.asarray(data["simagx"]),
        "boundary_psi": jnp.asarray(data["sibdry"]),
        "profile_coefficients": jnp.zeros(2),
        "coefficient_scale": jnp.ones(2),
        "ip_amperes": jnp.asarray(data["Ip"]),
        "major_radius": jnp.asarray(boundary_major_radius),
        "boundary_toroidal_field": jnp.asarray(boundary_field),
        "field_function_psi_n": jnp.asarray(data["pnorm"]),
        "field_function": jnp.asarray(data["fpol"]),
    }


def _block(tree: Any) -> Any:
    return jax.tree.map(
        lambda value: (
            value.block_until_ready() if hasattr(value, "block_until_ready") else value
        ),
        tree,
    )


def _kernel_call(kernel: Callable, inputs: dict[str, Any]) -> Callable:
    def call(psi: jax.Array, axis_psi: jax.Array, boundary_psi: jax.Array):
        return kernel(
            psi,
            inputs["radius"],
            inputs["height"],
            inputs["inside_limiter"],
            axis_psi=axis_psi,
            boundary_psi=boundary_psi,
            profile_coefficients=inputs["profile_coefficients"],
            coefficient_scale=inputs["coefficient_scale"],
            ip_amperes=inputs["ip_amperes"],
            major_radius=inputs["major_radius"],
            boundary_toroidal_field=inputs["boundary_toroidal_field"],
            field_function_psi_n=inputs["field_function_psi_n"],
            field_function=inputs["field_function"],
            n_pressure=1,
            n_diamagnetic=1,
            n_radial_cells=RADIAL_CELLS,
            n_surface_bins=SURFACE_BINS,
            psi_n_min=jnp.asarray(0.01),
            psi_n_max=jnp.asarray(0.99),
        )

    return call


def _record_row(
    *,
    route: str,
    execution: str,
    batch_size: int,
    timing: str,
    samples: list[float],
    source_revision: str,
    status: str = "measured",
    detail: str = "",
) -> dict[str, Any]:
    device, backend, model = _device_metadata()
    median = statistics.median(samples)
    return {
        "route": route,
        "device": device,
        "execution": execution,
        "batch_size": batch_size,
        "map_shape": "129x129",
        "timing": timing,
        "repetitions": len(samples),
        "median_total_seconds": f"{median:.9f}",
        "median_per_map_seconds": f"{median / batch_size:.9f}",
        "minimum_total_seconds": f"{min(samples):.9f}",
        "maximum_total_seconds": f"{max(samples):.9f}",
        "source_revision": source_revision,
        "backend": backend,
        "device_model": model,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "status": status,
        "detail": detail,
    }


def _measure_jitted(arguments: argparse.Namespace, inputs: dict[str, Any]) -> None:
    if arguments.route == "service":
        from nova.equilibrium.flux_surface_extraction import (
            extract_flux_surface_geometry,
        )

        kernel = extract_flux_surface_geometry.__wrapped__
    else:
        from nova.transport.current_diffusion import traced_flux_surface_geometry

        kernel = traced_flux_surface_geometry.__wrapped__

    call = _kernel_call(kernel, inputs)
    if arguments.batch_size == 1:
        executable = jax.jit(call)
        values = (inputs["psi"], inputs["axis_psi"], inputs["boundary_psi"])
        execution = "jit"
    else:
        scales = jnp.asarray(1.0 + 1.0e-5 * np.arange(arguments.batch_size))
        values = (
            scales[:, None, None] * inputs["psi"],
            scales * inputs["axis_psi"],
            scales * inputs["boundary_psi"],
        )
        executable = jax.jit(jax.vmap(call, in_axes=(0, 0, 0)))
        execution = "jit_vmap"

    start = time.perf_counter()
    compiled = executable.lower(*values).compile()
    compile_seconds = time.perf_counter() - start
    _append(
        arguments.output,
        _record_row(
            route=arguments.route,
            execution=execution,
            batch_size=arguments.batch_size,
            timing="cold_compile",
            samples=[compile_seconds],
            source_revision=arguments.source_revision,
            detail="lowering plus compilation; execution excluded",
        ),
    )

    warmed = []
    for repetition in range(arguments.repetitions):
        start = time.perf_counter()
        result = _block(compiled(*values))
        warmed.append(time.perf_counter() - start)
        if repetition == 0:
            valid = np.asarray(result["valid"])
            if not bool(np.all(valid)):
                raise RuntimeError(f"geometry record was invalid: {valid}")
    _append(
        arguments.output,
        _record_row(
            route=arguments.route,
            execution=execution,
            batch_size=arguments.batch_size,
            timing="jit_warm",
            samples=warmed,
            source_revision=arguments.source_revision,
        ),
    )


def _host_record(inputs: dict[str, Any]):
    from nova.equilibrium.flux_surface_geometry import FluxSurfaceGeometry

    data = inputs["data"]
    lattice = SimpleNamespace(radius=data["x"], height=data["z"])
    boundary_major_radius = float(inputs["major_radius"])

    def field_function(psi_n):
        return np.interp(psi_n, data["pnorm"], data["fpol"])

    return FluxSurfaceGeometry.from_flux_map(
        lattice,
        data["psi"],
        field_function,
        axis=(float(data["xmagx"]), float(data["zmagx"])),
        boundary_flux=float(data["sibdry"]),
        reference_radius=boundary_major_radius,
        rho_tor_norm=np.linspace(0.0, 1.0, RADIAL_CELLS + 1),
        surfaces=129,
        angles=256,
        edge_psi_norm=0.99,
    )


def _measure_host(arguments: argparse.Namespace, inputs: dict[str, Any]) -> None:
    execution = "host" if arguments.batch_size == 1 else "host_python_loop"

    def run():
        records = [_host_record(inputs) for _ in range(arguments.batch_size)]
        if any(record.size != RADIAL_CELLS + 1 for record in records):
            raise RuntimeError("host contour record has the wrong radial size")

    start = time.perf_counter()
    run()
    first_seconds = time.perf_counter() - start
    _append(
        arguments.output,
        _record_row(
            route=arguments.route,
            execution=execution,
            batch_size=arguments.batch_size,
            timing="first_execution_no_jit",
            samples=[first_seconds],
            source_revision=arguments.source_revision,
            detail="host route has no JIT compilation phase",
        ),
    )
    warmed = []
    for _ in range(arguments.repetitions):
        start = time.perf_counter()
        run()
        warmed.append(time.perf_counter() - start)
    _append(
        arguments.output,
        _record_row(
            route=arguments.route,
            execution=execution,
            batch_size=arguments.batch_size,
            timing="warm",
            samples=warmed,
            source_revision=arguments.source_revision,
            detail="repeated host assembly; no executable cache",
        ),
    )


def main() -> None:
    arguments = _arguments()
    if arguments.initialize:
        _initialize(arguments.output)
        return
    if arguments.route is None:
        raise ValueError("--route is required unless --initialize is used")
    if arguments.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    configure_dtypes()
    inputs = _input()
    if arguments.route == "host_contour":
        _measure_host(arguments, inputs)
    else:
        _measure_jitted(arguments, inputs)


if __name__ == "__main__":
    main()
