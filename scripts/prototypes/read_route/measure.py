"""Compare sampled and booked point fields on certificate topology reads."""

# Precision precedes imports with array-valued defaults.
# ruff: noqa: E402
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import resource
import subprocess
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, NearestNDInterpolator

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import current, topology
from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.order2.read import batched_field
from tests.equilibrium import test_topology_read as fixture


assert jax.config.jax_enable_x64 is True
ROOT = Path(__file__).resolve().parents[3]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SampledField:
    """Tensor cubic through point samples at the carrier's median pitch."""

    samples: jax.Array
    origin: jax.Array
    step: jax.Array

    def value(self, point):
        coordinate = (point - self.origin) / self.step
        base = jnp.clip(
            jnp.floor(coordinate).astype(jnp.int32),
            1,
            jnp.asarray(self.samples.shape) - 3,
        )
        offset = coordinate - base
        nodes = jnp.arange(4)
        x = base[0] + nodes - 1
        y = base[1] + nodes - 1
        weights = []
        for k in range(4):
            weight = jnp.ones(2)
            for m in range(4):
                if m != k:
                    weight = weight * (offset - (m - 1)) / (k - m)
            weights.append(weight)
        weights = jnp.stack(weights)
        return jnp.einsum(
            "i,j,ij->",
            weights[:, 0],
            weights[:, 1],
            self.samples[x[:, None], y[None, :]],
        )

    def evaluate(self, point):
        return topology.FieldJet(
            self.value(point),
            jax.grad(self.value)(point),
            jax.hessian(self.value)(point),
        )


def sample_field(field, geometry):
    pitch = float(np.median(np.asarray(geometry.pitch)))
    vertices = np.asarray(geometry.vertices).reshape(-1, 2)
    lo = np.min(vertices, axis=0) - 4 * pitch
    hi = np.max(vertices, axis=0) + 4 * pitch
    counts = np.ceil((hi - lo) / pitch).astype(int) + 1
    grid = np.stack(
        np.meshgrid(
            *(lo[d] + pitch * np.arange(counts[d]) for d in range(2)), indexing="ij"
        ),
        axis=-1,
    )
    mesh_points = np.asarray(geometry.sample_points).reshape(-1, 2)
    mesh_values = np.asarray(
        topology._point_values(field, geometry.sample_points)
    ).reshape(-1)
    unique, indices = np.unique(
        np.round(mesh_points, decimals=12), axis=0, return_index=True
    )
    measured = mesh_values[indices]
    cubic = CloughTocher2DInterpolator(unique, measured)
    samples = cubic(grid.reshape(-1, 2))
    missing = ~np.isfinite(samples)
    if np.any(missing):
        samples[missing] = NearestNDInterpolator(unique, measured)(
            grid.reshape(-1, 2)[missing]
        )
    assert np.isfinite(samples).all() and np.ptp(measured) > 0
    return SampledField(
        jnp.asarray(samples.reshape(tuple(counts))), jnp.asarray(lo), jnp.asarray(pitch)
    )


def read_row(
    field, geometry, convention, policy, reference, uncertainty, axis, saddle, name
):
    started = perf_counter()
    executable = (
        jax.jit(topology.read).lower(field, geometry, convention, policy).compile()
    )
    cold = perf_counter() - started
    started = perf_counter()
    result = executable(field, geometry, convention, policy)
    jax.block_until_ready(result)
    warm = perf_counter() - started
    started = perf_counter()
    again = executable(field, geometry, convention, policy)
    jax.block_until_ready(again)
    warm = min(warm, perf_counter() - started)
    area = np.asarray(geometry.full_area)
    pitch = float(np.median(np.asarray(geometry.pitch)))
    error = np.abs(np.asarray(result.membership) - reference) * area / np.median(area)
    near = (
        np.zeros(len(area), dtype=bool)
        if saddle is None
        else np.linalg.norm(np.asarray(geometry.centre) - saddle, axis=1)
        <= max(0.054806712567833996, 1.5 * pitch)
    )
    x_points = np.asarray(result.x_points)[np.asarray(result.x_point_valid)]
    try:
        executable_bytes = len(executable.runtime_executable().serialize())
    except AttributeError, RuntimeError:
        executable_bytes = None
    row = dict(
        representation=name,
        cells=len(area),
        pitch_m=pitch,
        axis=np.asarray(result.axis).tolist(),
        x_points=x_points.tolist(),
        axis_error_pitches=float(
            np.linalg.norm(np.asarray(result.axis) - axis) / pitch
        ),
        x_error_pitches=(
            None
            if saddle is None or len(x_points) != 1
            else float(np.linalg.norm(x_points[0] - saddle) / pitch)
        ),
        smooth_membership_error=float(np.max(error[~near])),
        saddle_membership_error=(float(np.max(error[near])) if near.any() else None),
        reference_uncertainty=float(np.max(uncertainty)),
        valid=bool(result.valid),
        qualified=bool(result.qualified),
        reason=int(result.reason),
        cold_compile_seconds=cold,
        warm_seconds=warm,
        serialized_executable_bytes=executable_bytes,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    print("READ_ROW " + json.dumps(row), flush=True)
    return row, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--perturbed", action="store_true")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"REVISION={revision} TREE={ROOT} COMMAND={' '.join(os.sys.argv)}", flush=True
    )
    print(f"MODULE={topology.__file__} CWD={Path.cwd().resolve()}", flush=True)
    assert jax.default_backend() == "gpu"
    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs(args.kind)
    geometry = fixture._realised_hex_geometry(wall, args.cells)
    assert len(geometry.centre) == args.cells
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0.0
    )
    reference_fraction, uncertainty = fixture._reference_cell_fractions(
        args.kind, oracle, geometry
    )
    scoring = (
        geometry,
        convention,
        policy,
        reference_fraction,
        uncertainty,
        axis,
        saddle,
    )
    rows = []
    analytic_row, analytic_read = read_row(analytic, *scoring, "closed_form")
    rows.append(analytic_row)
    sampled = sample_field(analytic, geometry)
    sampled_row, sampled_read = read_row(sampled, *scoring, "sampled")
    rows.append(sampled_row)
    if args.perturbed:
        pitch = sampled_row["pitch_m"]
        displaced = replace(sampled, origin=sampled.origin + jnp.asarray((0.0, pitch)))
        shifted = jax.jit(topology.read)(displaced, geometry, convention, policy)
        jax.block_until_ready(shifted)
        movement = float(
            np.linalg.norm(np.asarray(shifted.axis) - np.asarray(sampled_read.axis))
            / pitch
        )
        print(f"SAMPLED_SHIFT_CONTROL pitches={movement:.12g}", flush=True)
        assert 0.95 <= movement <= 1.05
    reference = fixture._kernel_backed_field(args.kind, oracle, analytic, geometry)
    kernel = batched_field(reference)
    kernel_row, _ = read_row(kernel, *scoring, "kernel")
    rows.append(kernel_row)
    if not args.perturbed:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                dict(kind=args.kind, cells=args.cells, rows=rows, batched_kernel=True),
                indent=2,
            )
            + "\n"
        )
        print("ROW_COMPLETE " + str(args.out), flush=True)
        return
    _, source, _ = certificate._case(
        "weak-rotation-reactor-static"
        if args.kind == "limited"
        else "diverted-single-null",
        clip_mode="exact",
    )
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    booking = current.book(
        psi,
        analytic_read,
        geometry,
        source.core,
        current.Quadrature(sigma=convention.sigma, segments=64),
    )
    jax.block_until_ready(booking)
    print(
        f"BOOKING valid={bool(booking.valid)} raw={float(booking.raw_current)}",
        flush=True,
    )
    perturbed = topology.TotalField(
        ClippedCurrentMoments(*booking[:3]), reference.coupling, reference.exterior
    )
    perturbed_sampled = sample_field(perturbed, geometry)
    perturbed_kernel = batched_field(perturbed)
    for name, field in (
        ("picard_sampled", perturbed_sampled),
        ("picard_kernel", perturbed_kernel),
    ):
        row, _ = read_row(field, *scoring, name)
        rows.append(row)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            dict(
                kind=args.kind,
                cells=args.cells,
                rows=rows,
                batched_kernel=True,
                booking_valid=bool(booking.valid),
                sampled_shift_control_pitches=movement,
            ),
            indent=2,
        )
        + "\n"
    )
    print("ROW_COMPLETE " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
