"""Measure the per-machine operator bank at the diverted 550-cell certificate.

Builds the bank once, publishes and reloads it through the content store, and
records: the build and load walls, the operator's reproduction of the analytic
map at the certificate's analytic state, the compilations a same-bucket and a
different-bucket geometry cost, the device memory per shape bucket with the
headroom a batch of 16 slices leaves, and the negative control in which one
operator captured as a closure constant recompiles on a same-bucket change.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

from nova.equilibrium import current, support_read, topology
from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
from nova.equilibrium.solve_request import TopologyPolicy

assert jax.config.jax_enable_x64 is True
ROOT = Path(__file__).resolve().parents[3]


def emit(token, **row):
    print(token + " " + json.dumps(row, sort_keys=True), flush=True)


def cache_entries(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


def relative(actual, reference):
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    scale = np.maximum(np.abs(reference), 1e-12 * np.max(np.abs(reference)))
    return float(np.max(np.abs(actual - reference) / scale))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=550)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--compile-cache", type=Path, required=True)
    args = parser.parse_args()

    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    print(
        f"REVISION={revision} TREE={ROOT} MODULE={support_read.__file__} "
        f"CWD={Path.cwd().resolve()} KIND=diverted CELLS={args.cells}",
        flush=True,
    )
    print(
        f"DEVICES={[str(device) for device in jax.local_devices()]} "
        f"PLATFORM={jax.default_backend()}",
        flush=True,
    )

    import tests.equilibrium.test_topology_read as fixture
    from benchmarks import solovev_certificate as certificate
    from scripts.analytic_oracle_fixtures.measure import analytic_profile
    from scripts.prototypes.read_route.jets import build_jet_matrix
    from scripts.prototypes.read_route.measure import sample_field

    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs("diverted")
    geometry = fixture._realised_hex_geometry(wall, args.cells)
    convention = topology.TopologyConvention.from_cocos(17, 1.0)
    policy = TopologyPolicy(normal_form_radius=0.054806712567833996)
    analytic_read = jax.jit(topology.read)(analytic, geometry, convention, policy)
    jax.block_until_ready(analytic_read)
    sampled_read = jax.jit(topology.read)(
        sample_field(analytic, geometry), geometry, convention, policy
    )
    jax.block_until_ready(sampled_read)

    _, source, _ = certificate._case("diverted-single-null", clip_mode="exact")
    profile = analytic_profile(source)
    psi = jax.vmap(jax.vmap(analytic.value))(geometry.sample_points)
    reference = fixture._kernel_backed_field("diverted", oracle, analytic, geometry)
    coupling = reference.coupling

    def book(read):
        return ClippedCurrentMoments(
            *current.book(
                psi,
                read,
                geometry,
                profile,
                current.Quadrature(sigma=convention.sigma, segments=64),
            )[:3]
        )

    converged_moments = book(analytic_read)
    trial_moments = book(sampled_read)

    started = perf_counter()
    matrix, exterior, nodes, sample_index, build_seconds = build_jet_matrix(
        coupling, geometry, analytic, reference.moments, persist=False
    )
    bank = support_read.SupportOperands.from_rows(
        geometry, matrix, exterior, nodes, sample_index, bucket=len(nodes)
    )
    jax.block_until_ready(bank.rows)
    emit(
        "BANK_BUILD",
        nodes=len(nodes),
        cells=len(coupling.centre),
        bucket=bank.bucket,
        active_nodes=int(bank.active_nodes),
        build_seconds=build_seconds,
        assembles_seconds=perf_counter() - started,
        operator_bytes=int(bank.rows.size * bank.rows.dtype.itemsize)
        + int(bank.exterior.size * bank.exterior.dtype.itemsize),
    )

    digest = support_read.write_bank(bank, args.store)
    load_started = perf_counter()
    loaded = support_read.read_bank(args.store, digest, geometry, bank.mesh)
    jax.block_until_ready(loaded.rows)
    load_seconds = perf_counter() - load_started
    emit(
        "BANK_LOAD",
        digest=digest,
        load_seconds=load_seconds,
        within_one_second=bool(load_seconds <= 1.0),
        round_trip_identical=bool(
            np.array_equal(np.asarray(loaded.rows), np.asarray(bank.rows))
        ),
    )

    program = jax.jit(support_read.bank_program)
    bank_jets = np.asarray(jax.block_until_ready(program(bank, reference.moments)))
    from nova.equilibrium import fixed_points

    analytic_jets = np.asarray(
        fixed_points.stack_jet(jax.jit(jax.vmap(analytic.evaluate))(jnp.asarray(nodes)))
    )
    active = len(nodes)
    names = ("value", "dx", "dy", "dxx", "dxy", "dyy")
    per_entry = {
        name: relative(bank_jets[:active, index], analytic_jets[:, index])
        for index, name in enumerate(names)
    }
    emit(
        "BANK_REPRODUCTION",
        state="reference-moments",
        per_entry=per_entry,
        worst_relative=max(per_entry.values()),
        within_1e10=bool(max(per_entry.values()) <= 1e-10),
    )

    # Compilation counting reads the persistent compilation cache, fresh per run.
    args.compile_cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.compile_cache))
    for name, value in (
        ("jax_persistent_cache_min_compile_time_secs", 0.0),
        ("jax_persistent_cache_min_entry_size_bytes", 0),
    ):
        try:
            jax.config.update(name, value)
        except AttributeError:
            pass

    def counted(label, operands):
        before = program._cache_size()
        entries_before = cache_entries(args.compile_cache)
        jax.block_until_ready(program(operands, converged_moments))
        after = program._cache_size()
        row = dict(
            label=label,
            bucket=operands.bucket,
            jit_cache_size=after,
            compiled=after - before,
            cache_entries=cache_entries(args.compile_cache) - entries_before,
        )
        emit("COMPILE", **row)
        return after

    first = counted("bucket-first", bank)
    same_rows = np.asarray(bank.rows)[:active] * 1.0
    same_rows = same_rows + 1e-3 * same_rows.max()
    same_exterior = np.asarray(bank.exterior)[:active] + 1e-3
    same_bucket = support_read.SupportOperands.from_rows(
        geometry,
        same_rows,
        same_exterior,
        np.asarray(bank.nodes)[:active],
        sample_index,
        bucket=int(bank.active_nodes),
    )
    second = counted("same-bucket-second-geometry", same_bucket)

    wide = support_read.SupportOperands.from_rows(
        geometry,
        same_rows,
        same_exterior,
        np.asarray(bank.nodes)[:active],
        sample_index,
        bucket=support_read.SHAPE_BUCKETS[
            support_read.SHAPE_BUCKETS.index(bank.bucket) + 1
        ],
    )
    third = counted("different-bucket-geometry", wide)

    emit(
        "COMPILE_SUMMARY",
        after_first=first,
        after_same_bucket=second,
        after_different_bucket=third,
        same_bucket_delta=second - first,
        different_bucket_delta=third - second,
        same_bucket_compiled=second - first,
        different_bucket_bucket=wide.bucket,
    )

    stats = jax.local_devices()[0].memory_stats() or {}
    limit = int(stats.get("bytes_limit", 0))
    peak = int(stats.get("peak_bytes_in_use", 0))
    in_use = int(stats.get("bytes_in_use", 0))
    per_slice = int(bank.rows.size * bank.rows.dtype.itemsize) + int(
        bank.exterior.size * bank.exterior.dtype.itemsize
    )
    emit(
        "MEMORY",
        bucket=bank.bucket,
        active_nodes=active,
        bank_bytes=per_slice,
        bytes_limit=limit,
        peak_bytes_in_use=peak,
        bytes_in_use=in_use,
        batch=args.batch,
        batch_bytes=args.batch * per_slice,
        headroom_bytes=limit - max(peak, in_use),
        batch_fits=bool(limit - max(peak, in_use) >= args.batch * per_slice),
        stats_present=bool(stats),
    )

    # Negative control: one operator captured as a closure constant. A
    # same-bucket geometry change builds a new program and compiles again.
    def closure_program(operands):
        captured = np.asarray(operands.rows)
        constant = jnp.asarray(captured[:, 0, 0, 0])
        return jax.jit(
            lambda moments: jnp.sum(constant) * 0.0 + jnp.sum(moments) * 0.0
        )

    before = cache_entries(args.compile_cache)
    closure_program(bank)(converged_moments)
    middle = cache_entries(args.compile_cache)
    closure_program(same_bucket)(converged_moments)
    after = cache_entries(args.compile_cache)
    before_traced = program._cache_size()
    program(bank, converged_moments)
    jax.block_until_ready(program(same_bucket, converged_moments))
    traced_delta = program._cache_size() - before_traced
    emit(
        "NEGATIVE_CONTROL",
        captured_operator="rows",
        closure_same_bucket_new_compiles=after - middle,
        closure_first_new_compiles=middle - before,
        traced_same_bucket_new_compiles=traced_delta,
        reproduces=bool(after - middle >= 1),
    )


if __name__ == "__main__":
    main()
    print("EXIT=0", flush=True)
