"""Compare sampled and booked point fields on certificate topology reads."""

# Precision precedes imports with array-valued defaults.
# ruff: noqa: E402
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from itertools import combinations
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
from scipy.interpolate import CloughTocher2DInterpolator

from benchmarks import solovev_certificate as certificate
from scripts.analytic_oracle_fixtures.measure import analytic_profile
from nova.equilibrium import current, topology
from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.prototypes.order2.read import batched_field
from scripts.prototypes.read_route.jets import (
    HermiteField,
    build_jet_matrix,
    read_with_jets,
)
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
    step = pitch / 4
    vertices = np.asarray(geometry.vertices).reshape(-1, 2)
    lo = np.min(vertices, axis=0) - 4 * pitch
    hi = np.max(vertices, axis=0) + 4 * pitch
    counts = np.ceil((hi - lo) / step).astype(int) + 1
    grid = np.stack(
        np.meshgrid(
            *(lo[d] + step * np.arange(counts[d]) for d in range(2)), indexing="ij"
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
        # The sampled carrier has no exterior cells; continue with its declared
        # exterior evaluator instead of a nearest-value wall plateau.
        samples[missing] = np.asarray(
            topology._point_values(field, jnp.asarray(grid.reshape(-1, 2)[missing]))
        )
    assert np.isfinite(samples).all() and np.ptp(measured) > 0
    return SampledField(
        jnp.asarray(samples.reshape(tuple(counts))), jnp.asarray(lo), jnp.asarray(step)
    )


def jet_reader(field, geometry, convention, policy):
    return read_with_jets(*field, geometry, convention, policy)


def reused_rows(directory, kind, cells):
    """Read only measured prefix rows from the known profile-lookup failure."""
    log = directory / f"{kind}-{cells}.log"
    content = log.read_text()
    failed_access = (
        "AttributeError: 'RotatingEquilibrium' object has no attribute 'core'"
    )
    if failed_access not in content:
        raise RuntimeError(f"prefix log lacks its expected failure: {log}")
    measured = {
        row["representation"]: row
        for line in content.splitlines()
        if line.startswith("READ_ROW ")
        for row in (json.loads(line.removeprefix("READ_ROW ")),)
    }
    if not {"closed_form", "sampled"} <= measured.keys():
        raise RuntimeError(f"prefix log lacks a measured arm: {log}")
    for row in measured.values():
        if row["cells"] != cells:
            raise RuntimeError(f"prefix row cell count differs: {log}")
        row["reused_from"] = str(log)
    print(f"REUSED_PREFIX log={log} arms=closed_form,sampled", flush=True)
    return measured, content


def _prepared_field(field, geometry):
    if not isinstance(field, tuple):
        return field
    matrix, moments, exterior, nodes, sample_index = field
    jets = exterior + jnp.einsum("ncmq,cm->nq", matrix, jnp.stack(tuple(moments), -1))
    return HermiteField(jets, nodes, sample_index, geometry.centre, geometry.pitch)


def _sample_stage(field, geometry):
    prepared = _prepared_field(field, geometry)
    if not isinstance(field, tuple):
        return topology._sampled_flux(prepared, geometry)
    values = prepared.jets[prepared.sample_index, 0]
    return (
        jnp.einsum("nij,nj->ni", geometry.fit_inverse, values),
        prepared.jets[prepared.sample_index[:, 0], 0],
    )


def _wall_stage(field, geometry, convention, policy):
    return topology._wall_events(
        _prepared_field(field, geometry), geometry, policy, convention.sigma
    )


def _probe_stage(function, *args):
    started = perf_counter()
    executable = jax.jit(function).lower(*args).compile()
    compile_wall = perf_counter() - started
    walls = []
    for _ in range(2):
        started = perf_counter()
        jax.block_until_ready(executable(*args))
        walls.append(perf_counter() - started)
    return dict(cold_compile_seconds=compile_wall, warm_seconds=min(walls))


def read_row(
    field,
    geometry,
    convention,
    policy,
    reference,
    uncertainty,
    axis,
    saddle,
    name,
    reader=topology.read,
):
    started = perf_counter()
    executable = jax.jit(reader).lower(field, geometry, convention, policy).compile()
    cold = perf_counter() - started
    trials = []
    for _ in range(2):
        started = perf_counter()
        result = executable(field, geometry, convention, policy)
        dispatch = perf_counter() - started
        jax.block_until_ready(result)
        trials.append((perf_counter() - started, dispatch))
    warm, warm_dispatch = min(trials)
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
        warm_dispatch_seconds=warm_dispatch,
        serialized_executable_bytes=executable_bytes,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    if (
        os.getenv("READ_ROUTE_PROFILE") == "1"
        and len(area) == 2000
        and saddle is not None
        and name in ("kernel", "jet")
    ):
        probes = {}
        for stage, function, operands in (
            ("sample", _sample_stage, (field, geometry)),
            ("wall", _wall_stage, (field, geometry, convention, policy)),
            (
                "labels",
                topology._connected_fragment_labels,
                (
                    result.fragment_area,
                    result.edge_interval,
                    geometry,
                    policy.edge_tolerance,
                ),
            ),
        ):
            try:
                probes[stage] = _probe_stage(function, *operands)
            except Exception as error:
                probes[stage] = {"error": repr(error)}
        row["stage_probes"] = probes
        print("STAGE_PROBES " + json.dumps(probes), flush=True)
    print("READ_ROW " + json.dumps(row), flush=True)
    return row, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--perturbed", action="store_true")
    parser.add_argument("--reuse-prefix", type=Path)
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
    reused, prior_log = (
        reused_rows(args.reuse_prefix, args.kind, args.cells)
        if args.reuse_prefix is not None
        else ({}, "")
    )
    if reused:
        analytic_read = jax.jit(topology.read)(analytic, geometry, convention, policy)
        jax.block_until_ready(analytic_read)
        analytic_row = reused["closed_form"]
    else:
        analytic_row, analytic_read = read_row(analytic, *scoring, "closed_form")
    rows.append(analytic_row)
    if args.cells in (2000, 5000):
        golden = np.load(
            ROOT
            / "scripts/prototypes/ladder/compatibility"
            / f"{args.kind}-{args.cells}.npz"
        )
        delta = float(
            np.max(np.abs(np.asarray(analytic_read.membership) - golden["membership"]))
        )
        axis_delta = float(
            np.linalg.norm(np.asarray(analytic_read.axis) - golden["axis"])
        )
        print(
            f"LADDER_CONTROL membership_delta={delta:.12g} "
            f"axis_delta_m={axis_delta:.12g}",
            flush=True,
        )
        assert delta < 1e-9 and axis_delta < 1e-9
    sampled = sample_field(analytic, geometry) if not reused or args.perturbed else None
    if reused:
        sampled_row = reused["sampled"]
        sample_line = next(
            line
            for line in prior_log.splitlines()
            if line.startswith("MESH_SAMPLE_CONTROL peak_flux_error=")
        )
        mesh_reconstruction_error = float(sample_line.split("=")[-1])
        if args.perturbed:
            sampled_read = jax.jit(topology.read)(sampled, geometry, convention, policy)
            jax.block_until_ready(sampled_read)
    else:
        mesh_truth = np.asarray(
            topology._point_values(analytic, geometry.sample_points)
        )
        mesh_reconstructed = np.asarray(
            topology._point_values(sampled, geometry.sample_points)
        )
        mesh_reconstruction_error = float(
            np.max(np.abs(mesh_reconstructed - mesh_truth))
        )
        print(
            f"MESH_SAMPLE_CONTROL peak_flux_error={mesh_reconstruction_error:.12g}",
            flush=True,
        )
        sampled_row, sampled_read = read_row(sampled, *scoring, "sampled")
    rows.append(sampled_row)
    if args.perturbed:
        if reused:
            shift_line = next(
                line
                for line in prior_log.splitlines()
                if line.startswith("SAMPLED_SHIFT_CONTROL pitches=")
            )
            movement = float(shift_line.split("=")[-1])
        else:
            pitch = sampled_row["pitch_m"]
            displaced = replace(
                sampled, origin=sampled.origin + jnp.asarray((0.0, pitch))
            )
            shifted = jax.jit(topology.read)(displaced, geometry, convention, policy)
            jax.block_until_ready(shifted)
            movement = float(
                np.linalg.norm(np.asarray(shifted.axis) - np.asarray(sampled_read.axis))
                / pitch
            )
        print(f"SAMPLED_SHIFT_CONTROL pitches={movement:.12g}", flush=True)
        assert 0.95 <= movement <= 1.05
    _, source, _ = certificate._case(
        "weak-rotation-reactor-static"
        if args.kind == "limited"
        else "diverted-single-null",
        clip_mode="exact",
    )
    profile = analytic_profile(source)
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    booking = current.book(
        psi,
        analytic_read,
        geometry,
        profile,
        current.Quadrature(sigma=convention.sigma, segments=64),
    )
    jax.block_until_ready(booking)
    assert bool(booking.valid)
    print(
        f"BOOKING valid={bool(booking.valid)} raw={float(booking.raw_current)}",
        flush=True,
    )
    moments = ClippedCurrentMoments(*booking[:3])
    reference = fixture._kernel_backed_field(args.kind, oracle, analytic, geometry)
    coupling = reference.coupling
    exterior = reference.exterior
    reference_moments = reference.moments
    print("EXTERIOR_REFERENCE exact analytic density quadrature", flush=True)
    booked_values = np.stack(tuple(np.asarray(value) for value in moments))
    exact_values = np.stack(tuple(np.asarray(value) for value in reference_moments))
    moment_delta = float(
        np.max(np.abs(booked_values - exact_values)) / np.max(np.abs(exact_values))
    )
    print(f"BOOKED_REFERENCE_DELTA relative_sup={moment_delta:.12g}", flush=True)
    assert moment_delta > 0.0
    reference = topology.TotalField(moments, coupling, exterior)
    kernel = batched_field(reference)
    kernel_row, _ = read_row(kernel, *scoring, "kernel")
    rows.append(kernel_row)
    matrix, exterior_jets, nodes, sample_index, build_wall = build_jet_matrix(
        coupling, geometry, analytic, reference_moments
    )
    jet_payload = (matrix, moments, exterior_jets, nodes, sample_index)
    jet_row, _ = read_row(jet_payload, *scoring, "jet", reader=jet_reader)
    jet_row["jet_build_seconds"] = build_wall
    jet_row["jet_bytes"] = int(np.prod(matrix.shape) * matrix.dtype.itemsize)
    jet_row["jet_shape"] = list(matrix.shape)
    rows.append(jet_row)
    if args.perturbed:
        zero_hessian = matrix.at[..., 3:].set(0)
        control, _ = read_row(
            (zero_hessian, moments, exterior_jets, nodes, sample_index),
            *scoring,
            "jet_zero_hessian_control",
            reader=jet_reader,
        )
        budget = (0.274714 if saddle is None else 2.1805365680219544) * (
            jet_row["pitch_m"] / (6.2 if saddle is None else 1.7)
        ) ** 2
        control_fails = (
            not control["valid"]
            or not control["qualified"]
            or (control["saddle_membership_error"] or 0.0) > budget
        )
        print(
            f"JET_HESSIAN_CONTROL fails={control_fails} budget={budget:.12g}",
            flush=True,
        )
        rows.append(control)
        assert control_fails, "zeroing source Hessians did not fail saddle membership"
        trial_booking = current.book(
            psi,
            sampled_read,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma, segments=64),
        )
        jax.block_until_ready(trial_booking)
        assert bool(trial_booking.valid)
        trial_moments = ClippedCurrentMoments(*trial_booking[:3])
        perturbed = topology.TotalField(trial_moments, coupling, exterior)
        trial_payload = (matrix, trial_moments, exterior_jets, nodes, sample_index)
        picard_reads = {}
        for name, field, reader in (
            (
                "picard_sampled",
                sample_field(batched_field(perturbed), geometry),
                topology.read,
            ),
            ("picard_kernel", batched_field(perturbed), topology.read),
            ("picard_jet", trial_payload, jet_reader),
        ):
            row, result = read_row(field, *scoring, name, reader=reader)
            rows.append(row)
            picard_reads[name] = result
        disagreements = {}
        for left, right in combinations(picard_reads, 2):
            a, b = picard_reads[left], picard_reads[right]
            disagreements[f"{left}/{right}"] = dict(
                axis_pitches=float(
                    np.linalg.norm(np.asarray(a.axis) - np.asarray(b.axis))
                    / jet_row["pitch_m"]
                ),
                membership_median_cells=float(
                    np.max(
                        np.abs(np.asarray(a.membership) - np.asarray(b.membership))
                        * np.asarray(geometry.full_area)
                        / np.median(np.asarray(geometry.full_area))
                    )
                ),
            )
        print("PICARD_DISAGREEMENT " + json.dumps(disagreements), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            dict(
                kind=args.kind,
                cells=args.cells,
                rows=rows,
                batched_kernel=True,
                booking_valid=bool(booking.valid),
                booked_reference_moment_relative_sup=moment_delta,
                sampled_shift_control_pitches=movement if args.perturbed else None,
                mesh_sample_peak_flux_error=mesh_reconstruction_error,
                picard_disagreement=disagreements if args.perturbed else None,
            ),
            indent=2,
        )
        + "\n"
    )
    print("ROW_COMPLETE " + str(args.out), flush=True)


if __name__ == "__main__":
    main()
