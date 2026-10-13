"""Measure bounded stationary solves and maintenance on moving physical states."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import nova
from nova.equilibrium import fixed_points as fixed
from scripts.prototypes.fixed_points.solve import CachedField, solve_adaptive
from tests.equilibrium import test_topology_read as fixture


def measure(kind, cells, out, states, calls):
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    oracle, analytic, wall, axis, saddle = fixture._analytic_inputs(kind)
    geometry = fixture._realised_hex_geometry(wall, cells)
    pitch = jnp.median(geometry.pitch)
    roots = jnp.asarray([axis] if saddle is None else [axis, saddle])
    print("BUILD", kind, cells, "realised", len(geometry.centre), flush=True)
    reference = fixture._kernel_backed_field(kind, oracle, analytic, geometry)
    # A small, nonzero physical plasma contribution moves with its booked
    # current amplitude. The analytic exterior translates through eight pitches.
    exact = jnp.stack(tuple(reference.moments), -1)
    cache = None
    previous = roots + jnp.asarray((0.0, -4 * pitch))
    records = []
    before = jax.jit(solve_adaptive)
    all_times, all_before = [], []
    for index, shift in enumerate(np.linspace(-4, 4, states)):
        exterior = replace(analytic, shift=jnp.asarray((0.0, shift * pitch)))
        cache, maintenance = fixed.maintain_rows(
            geometry,
            reference.coupling,
            exterior,
            previous,
            float(1.6 * pitch),
            cache=cache,
        )
        moments = exact * (0.002 + 0.001 * np.sin(index))
        seed = previous
        jax.block_until_ready((cache, moments, seed, pitch))
        start = perf_counter()
        result = fixed.solve(cache, moments, seed, previous, pitch)
        jax.block_until_ready(result)
        warm_seconds = perf_counter() - start
        field = CachedField(
            cache.mesh,
            cache.lookup,
            cache.rows,
            moments,
            cache.exterior,
            jnp.asarray(1.0),
        )
        base = before(field, seed, pitch)
        jax.block_until_ready(base)
        np.testing.assert_allclose(result.position, base.position, atol=2e-9, rtol=0)
        eigen = np.linalg.eigvalsh(np.asarray(result.jet.hessian))
        assert np.all(np.asarray(result.valid)), (kind, cells, index, result)
        assert np.all(eigen[0] < 0), (kind, cells, index, eigen)
        assert len(eigen) == 1 or np.prod(eigen[1]) < 0, (kind, cells, index, eigen)
        timings, base_timings = [], []
        # Alternate the order so slow host intervals cannot all land in one arm.
        for repeat in range(calls):
            arms = (
                ("production", "prototype")
                if repeat % 2
                else ("prototype", "production")
            )
            for arm in arms:
                start = perf_counter()
                if arm == "production":
                    measured = fixed.solve(cache, moments, seed, previous, pitch)
                else:
                    measured = before(field, seed, pitch)
                jax.block_until_ready(measured)
                wall_seconds = perf_counter() - start
                (timings if arm == "production" else base_timings).append(wall_seconds)
        all_times.extend(timings)
        all_before.extend(base_timings)
        records.append(
            dict(
                state=index,
                imposed_shift_pitches=float(shift),
                maintenance=asdict(maintenance),
                warm_seconds=warm_seconds,
                positions=np.asarray(result.position).tolist(),
                iterations=np.asarray(result.iterations).tolist(),
                status=np.asarray(result.status).tolist(),
                eigenvalues=eigen.tolist(),
                seconds=timings,
                prototype_seconds=base_timings,
            )
        )
        previous = result.position
        receipt = dict(
            kind=kind,
            requested_cells=cells,
            cells=len(geometry.centre),
            states=records,
            complete=False,
        )
        (out / f"{kind}-{cells}.json").write_text(json.dumps(receipt, indent=2))
        print(
            "STATE",
            kind,
            cells,
            index,
            "maintenance",
            maintenance.seconds,
            "added",
            maintenance.nodes_added,
            "p99",
            np.percentile(timings, 99),
            flush=True,
        )
    travel = np.linalg.norm(
        np.asarray(records[-1]["positions"]) - np.asarray(records[0]["positions"]),
        axis=-1,
    ) / float(pitch)
    refreshes = sum(row["maintenance"]["nodes_added"] > 0 for row in records[1:])
    assert refreshes > 0 and np.min(travel) > 6, (refreshes, travel)
    p99 = float(np.percentile(all_times, 99))
    receipt.update(
        complete=True,
        p99_seconds=p99,
        median_seconds=float(np.median(all_times)),
        prototype_p99_seconds=float(np.percentile(all_before, 99)),
        prototype_median_seconds=float(np.median(all_before)),
        travel_pitches=travel.tolist(),
        refresh_count=refreshes,
        failures=0,
        calls=len(all_times),
        maintenance_p99_seconds=float(
            np.percentile([row["maintenance"]["seconds"] for row in records], 99)
        ),
    )
    (out / f"{kind}-{cells}.json").write_text(json.dumps(receipt, indent=2))
    print(
        "SUMMARY",
        json.dumps({key: value for key, value in receipt.items() if key != "states"}),
        flush=True,
    )
    assert p99 < 0.001, f"{kind} {cells} p99={p99:.9f}s exceeds 1ms"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--states", type=int, default=21)
    parser.add_argument("--calls", type=int, default=20)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"REVISION={revision} TREE={Path.cwd()} COMMAND={sys.executable} "
        "-m scripts.prototypes.fixed_points.latency_gate "
        + " ".join(sys.argv[1:]),
        flush=True,
    )
    print(
        f"NOVA_MODULE={nova.__file__} MODULE={fixed.__file__} "
        f"CWD={Path.cwd().resolve()} DEVICES={jax.devices()}",
        flush=True,
    )
    measure(args.kind, args.cells, args.out, args.states, args.calls)


if __name__ == "__main__":
    main()
