"""Measure cached Bell membership and geometric saddle controls on device."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.biot import polygonanalytic
from nova.equilibrium import current, topology
from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
from nova.equilibrium.solve_request import TopologyPolicy
from scripts.analytic_oracle_fixtures.measure import analytic_profile
from scripts.prototypes.fixed_points.solve import (
    CachedField,
    select_cache_nodes,
    solve,
)
from scripts.prototypes.fixed_points.measure import membership as straight_membership
from scripts.prototypes.membership.hybrid import membership
from scripts.prototypes.read_route import jets as jet_module
from scripts.prototypes.read_route.bell import build_mesh, field_from_jets, read_bell
from scripts.prototypes.read_route.check_element import stack_jet
from scripts.prototypes.read_route.measure import sample_field
from scripts.prototypes.order2.read import batched_field
from tests.equilibrium import test_topology_read as fixture

assert jax.config.jax_enable_x64 is True
ROOT = Path(__file__).resolve().parents[3]


def persist(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")


def cached_matrix(nodes, coupling, cache_revision):
    digest = hashlib.sha256(nodes.tobytes())
    for operand in jax.tree_util.tree_leaves(coupling):
        digest.update(np.asarray(operand).tobytes())
    source = subprocess.check_output(
        ["git", "show", f"{cache_revision}:scripts/prototypes/read_route/jets.py"]
    )

    def kernel_ast(text):
        tree = ast.parse(text)
        function = next(
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "_source_jets"
        )
        return ast.dump(function)

    assert kernel_ast(source) == kernel_ast(Path(jet_module.__file__).read_bytes())
    digest.update(source)
    digest.update(Path(polygonanalytic.__file__).read_bytes())
    directory = (
        Path("/work/projects/imas_gpu/sophelio/evidence/conforming-jets")
        / digest.hexdigest()
    )
    progress = json.loads((directory / "progress.json").read_text())
    matrix = np.load(directory / "matrix.npy", mmap_mode="r")
    assert progress["nodes_complete"] == len(nodes)
    assert matrix.shape == (len(nodes), len(coupling.centre), 3, 6)
    print(
        "CACHE_IDENTITY",
        directory,
        "nodes",
        len(nodes),
        "bytes",
        matrix.nbytes,
        flush=True,
    )
    return matrix


def run(args):
    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs(args.kind)
    geometry = fixture._realised_hex_geometry(wall, args.cells)
    pitch = jnp.median(geometry.pitch)
    positions = jnp.asarray([axis] if saddle is None else [axis, saddle])
    samples = np.asarray(geometry.sample_points)
    nodes, inverse = np.unique(
        np.round(samples.reshape(-1, 2), 12), axis=0, return_inverse=True
    )
    indices = jnp.asarray(inverse.reshape(samples.shape[:-1]))
    mesh = build_mesh(nodes, indices, geometry)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(
        normal_form_radius=0.054806712567833996 if saddle is not None else 0
    )
    truth, uncertainty = fixture._reference_cell_fractions(args.kind, oracle, geometry)
    area = np.asarray(geometry.full_area)
    near = (
        np.linalg.norm(np.asarray(geometry.centre) - np.asarray(positions[-1]), axis=1)
        <= max(0.054806712567833996, 1.5 * float(pitch))
        if saddle is not None
        else np.zeros(len(area), bool)
    )
    receipt = dict(
        kind=args.kind,
        cells=args.cells,
        pitch_m=float(pitch),
        rows=[],
        complete=False,
        reference_uncertainty=float(np.max(uncertainty)),
    )
    output = args.out / f"{args.kind}-{args.cells}.json"
    args.out.mkdir(parents=True, exist_ok=True)
    persist(output, receipt)

    def record(name, fractions, valid, reference=truth, **extra):
        error = np.abs(np.asarray(fractions) - reference) * area / np.median(area)
        row = dict(
            arm=name,
            valid=bool(valid),
            smooth_error=float(np.max(error[~near])),
            saddle_error=float(np.max(error[near])) if near.any() else None,
            **extra,
        )
        receipt["rows"].append(row)
        persist(output, receipt)
        print("MEMBERSHIP_ROW", json.dumps(row), flush=True)
        return row

    print("CLOSED_FORM_START", flush=True)
    closed = jax.jit(topology.read)(analytic, geometry, convention, policy)
    jax.block_until_ready(closed)
    record("closed_form", closed.membership, closed.valid)
    golden = np.load(
        ROOT
        / "scripts/prototypes/ladder/compatibility"
        / f"{args.kind}-{args.cells}-cpu.npz"
    )
    delta = float(np.max(np.abs(np.asarray(closed.membership) - golden["membership"])))
    receipt["closed_form_delta"] = delta
    print("CLOSED_FORM_REPRODUCTION delta=", delta, flush=True)
    assert delta < 1e-9
    closed_fixed = jax.jit(solve)(analytic, positions, pitch)
    # The straight-leg control is rerun before the curved prototype is measured.
    values = jax.jit(jax.vmap(analytic.value))(jnp.asarray(nodes))
    straight = jax.jit(straight_membership, static_argnames=("diverted",))(
        values,
        indices,
        geometry,
        closed_fixed,
        convention.sigma,
        1.5 * pitch,
        saddle is not None,
    )
    straight_row = record("recorded_straight", *straight)
    if saddle is not None:
        target = 0.005919276 if args.cells == 2000 else 0.004341904
        assert abs(straight_row["saddle_error"] - target) < 1e-8, straight_row
    print("REFERENCE_COUPLING_START", flush=True)
    reference = fixture._kernel_backed_field(args.kind, oracle, analytic, geometry)
    matrix = cached_matrix(nodes, reference.coupling, args.cache_revision)
    _, source, _ = certificate._case(
        "weak-rotation-reactor-static" if saddle is None else "diverted-single-null",
        clip_mode="exact",
    )
    profile = analytic_profile(source)
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    book = jax.jit(
        lambda reading: current.book(
            psi,
            reading,
            geometry,
            profile,
            current.Quadrature(sigma=convention.sigma, segments=64),
        )
    )
    booking = book(closed)
    jax.block_until_ready(booking)
    assert bool(booking.valid)
    exact = jnp.stack(tuple(reference.moments), -1)
    analytic_jets = stack_jet(jax.jit(jax.vmap(analytic.evaluate))(jnp.asarray(nodes)))
    selected = select_cache_nodes(mesh, np.asarray(positions), float(pitch))
    lookup = np.full(len(nodes), -1, np.int32)
    lookup[selected] = np.arange(len(selected))
    rows = jnp.asarray(np.asarray(matrix[selected]))
    exterior = analytic_jets[selected] - jnp.einsum("ncmq,cm->nq", rows, exact)
    states = [("booked", booking)]
    if args.cells == 2000 and saddle is not None:
        sampled = sample_field(analytic, geometry)
        sampled_read = jax.jit(topology.read)(sampled, geometry, convention, policy)
        trial = book(sampled_read)
        jax.block_until_ready(trial)
        assert bool(trial.valid)
        states.append(("picard", trial))
    for state, booked in states:
        moments = jnp.stack(tuple(booked[:3]), -1)
        difference = moments - exact
        # Stream the read-only dense cache to the device in bounded nodal batches.
        chunks = []
        contraction = jax.jit(
            lambda block, delta: jnp.einsum("ncmq,cm->nq", block, delta)
        )
        for first in range(0, len(nodes), 128):
            block = jnp.asarray(np.asarray(matrix[first : first + 128]))
            chunks.append(np.asarray(contraction(block, difference)))
        nodal = analytic_jets + jnp.asarray(np.concatenate(chunks))
        field = field_from_jets(nodal, indices, mesh)
        cached = CachedField(
            mesh, jnp.asarray(lookup), rows, moments, exterior, jnp.asarray(1.0)
        )
        fixed = jax.jit(solve)(cached, positions, pitch)
        jax.block_until_ready(fixed)
        assert np.all(np.asarray(fixed.valid)), fixed
        bell = jax.jit(read_bell)(field, geometry, convention, policy)
        jax.block_until_ready(bell)
        record(state + "_bell", bell.membership, bell.valid)
        level = fixed.jet.value[-1] if saddle is not None else bell.boundary_flux
        operands = (
            nodal,
            indices,
            mesh,
            geometry,
            fixed,
            level,
            convention.sigma,
            saddle is not None,
        )
        start = perf_counter()
        executable = jax.jit(membership).lower(*operands, jnp.asarray(1.0)).compile()
        compile_wall = perf_counter() - start
        result = executable(*operands, jnp.asarray(1.0))
        jax.block_until_ready(result)
        walls = []
        for _ in range(20):
            start = perf_counter()
            jax.block_until_ready(executable(*operands, jnp.asarray(1.0)))
            walls.append(perf_counter() - start)
        record(
            state + "_curved",
            result[0],
            result[1],
            median_seconds=float(np.median(walls)),
            p99_seconds=float(np.percentile(walls, 99)),
            compile_seconds=compile_wall,
            represented_cells=int(np.sum(result[3])),
            fixed_positions=np.asarray(fixed.position).tolist(),
        )
        zero = executable(*operands, jnp.asarray(0.0))
        jax.block_until_ready(zero)
        record(state + "_zero_third", zero[0], zero[1])
        if state == "booked":
            np.savez(
                args.out / f"{args.kind}-{args.cells}-panel.npz",
                vertices=np.asarray(geometry.vertices),
                counts=np.asarray(geometry.vertex_count),
                centre=np.asarray(geometry.centre),
                error=(np.asarray(result[0]) - truth) * area / np.median(area),
                membership=np.asarray(result[0]),
                reference=truth,
                near=near,
                wall=wall,
                fixed=np.asarray(fixed.position),
                truth_nulls=np.asarray(positions),
                direction=np.asarray(result[2].direction),
                curvature=np.asarray(result[2].curvature),
                cubic=np.asarray(result[2].cubic),
                represented=np.asarray(result[3]),
            )
        if state == "picard":
            kernel = topology.TotalField(
                ClippedCurrentMoments(*moments.T),
                reference.coupling,
                reference.exterior,
            )
            live = jax.jit(topology.read)(
                batched_field(kernel), geometry, convention, policy
            )
            jax.block_until_ready(live)
            record("picard_kernel", live.membership, live.valid)
            record(
                "picard_curved_vs_kernel",
                result[0],
                result[1] & live.valid,
                np.asarray(live.membership),
            )
            record(
                "picard_bell_vs_kernel",
                bell.membership,
                bell.valid & live.valid,
                np.asarray(live.membership),
            )
        persist(output, receipt)
    receipt["complete"] = True
    persist(output, receipt)
    print("ROW_COMPLETE", output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-revision", required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"REVISION={revision} TREE={ROOT} COMMAND={sys.executable} "
        "-m scripts.prototypes.membership.measure " + " ".join(sys.argv[1:]),
        flush=True,
    )
    print(
        f"NOVA_MODULE={nova.__file__} MODULE={__file__} "
        f"CWD={Path.cwd().resolve()} DEVICES={jax.devices()}",
        flush=True,
    )
    assert jax.default_backend() == "gpu"
    run(args)


if __name__ == "__main__":
    main()
