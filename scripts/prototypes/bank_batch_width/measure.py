"""Measure converged solves at nested widths from one saved seed bank."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.bank_booking_determinism.measure import CacheEvents
from scripts.prototypes.bank_newton_converge.fresh_solve import analytic_oracle
from scripts.prototypes.bank_solve_profile import fixture
from tests.equilibrium.test_topology_read import _analytic_inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("width", type=int)
    parser.add_argument("process", type=int)
    args = parser.parse_args()
    print(
        f"NOVA={nova.__file__} MODULE={support_read.__file__} "
        f"CWD={Path.cwd().resolve()} DEVICE={jax.devices()[0].device_kind}",
        flush=True,
    )
    print(
        f"PREALLOCATE={os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')} "
        f"MEMORY_FRACTION={os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}",
        flush=True,
    )
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    assert "H200" in jax.devices()[0].device_kind
    events = CacheEvents()
    logger = logging.getLogger("jax._src.compiler")
    logger.addHandler(events)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    batch = fixture.build(16)
    common = args.run / "common-seeds.npz"
    if not common.exists():
        scales = list(np.linspace(fixture.SCALE_LOW, fixture.SCALE_HIGH, 16))
        divisions = 15
        while len(scales) < 256:
            scales.extend(
                fixture.SCALE_LOW
                + (fixture.SCALE_HIGH - fixture.SCALE_LOW) * (index + 0.5) / divisions
                for index in range(divisions)
            )
            divisions *= 2
        scales = np.asarray(scales[:256])
        assert len(np.unique(scales)) == 256
        np.savez_compressed(
            common,
            scales=scales,
            seeds=np.asarray(batch.seeds),
            previous=np.asarray(batch.roots),
            **{
                f"state_{i}": np.asarray(value)[None] * (1 + scales[:, None])
                for i, value in enumerate(batch.moments)
            },
        )
    with np.load(common) as saved:
        states = tuple(jnp.asarray(saved[f"state_{i}"][: args.width]) for i in range(3))
        seeds = jnp.broadcast_to(jnp.asarray(saved["seeds"]), (args.width, 2, 2))
        previous = jnp.broadcast_to(jnp.asarray(saved["previous"]), (args.width, 2, 2))
    inputs = batch.operands, batch.cache, states, seeds, previous

    @jax.jit
    def solve(bank, cache, state, seed, warm):
        return support_read.solve_batch(
            bank,
            cache,
            state,
            seed,
            warm,
            batch.convention,
            batch.policy,
            batch.profile,
            fixture.gradient,
            batch.quadrature,
            max_steps=60,
            reuse_factor=False,
            contraction_limit=0.5,
        )

    started = time.perf_counter()
    program = solve.lower(*inputs).compile()
    compile_seconds = time.perf_counter() - started
    lookups = [row for row in events.rows if row["module"] == "jit_solve"]
    assert len(lookups) == 1
    assert lookups[0]["outcome"] == ("miss" if args.process == 0 else "hit")
    size = program.runtime_executable().size_of_generated_code_in_bytes()
    assert size > 0
    analysis = program.memory_analysis()
    required = dict(
        temporary_bytes=analysis.temp_size_in_bytes,
        argument_bytes=analysis.argument_size_in_bytes,
        output_bytes=analysis.output_size_in_bytes,
        alias_bytes=analysis.alias_size_in_bytes,
    )
    assert required["temporary_bytes"] > 0
    capacity = jax.devices()[0].memory_stats()["bytes_limit"]
    print(
        "MEMORY_REQUIREMENT " + json.dumps(required | dict(capacity_bytes=capacity)),
        flush=True,
    )
    if required["temporary_bytes"] > capacity:
        assert args.width == 256, "required width exceeds available device capacity"
        row = dict(
            width=args.width,
            fits=False,
            compile_seconds=compile_seconds,
            executable_bytes=size,
            memory_requirement=required,
            capacity_bytes=capacity,
            cache=lookups,
            physical_bytes_estimate=capacity / 0.95,
        )
        (args.run / f"width-{args.width}-unfit.json").write_text(
            json.dumps(row, indent=2)
        )
        print("UNFIT " + json.dumps(row), flush=True)
        return
    print(
        f"COMPILED width={args.width} seconds={compile_seconds} bytes={size}",
        flush=True,
    )
    jax.block_until_ready(program(*inputs))
    walls = []
    for _ in range(5):
        started = time.perf_counter()
        result = jax.block_until_ready(program(*inputs))
        walls.append(1000 * (time.perf_counter() - started))
    memory = jax.devices()[0].memory_stats()
    assert memory["peak_bytes_in_use"] > 0
    _, analytic, wall, _, _ = _analytic_inputs("diverted")
    reading = result.support.reading
    oracle, roots = analytic_oracle(
        analytic, batch.roots, jax.tree.map(lambda value: value[0], reading), wall
    )
    boundary_flux = float(analytic.value(jnp.asarray(roots[1])))
    precision = dict(
        axis_ratio=(
            np.linalg.norm(np.asarray(reading.axis) - roots[0], axis=-1)
            / oracle["minor_radius"]
        ).tolist(),
        saddle_ratio=(
            np.linalg.norm(np.asarray(reading.boundary) - roots[1], axis=-1)
            / oracle["minor_radius"]
        ).tolist(),
        boundary_ratio=(
            np.abs(np.asarray(reading.boundary_flux) - boundary_flux)
            / oracle["flux_span"]
        ).tolist(),
    )
    row = dict(
        width=args.width,
        process=args.process,
        compile_seconds=compile_seconds,
        executable_bytes=size,
        memory=memory,
        memory_requirement=required,
        cache=lookups,
        precision=precision,
        common_sha256=hashlib.sha256(common.read_bytes()).hexdigest(),
        wall_ms=float(np.median(walls)),
        samples_ms=walls,
        **{
            key: np.asarray(getattr(result, key)).tolist()
            for key in ("reads", "steps", "refactors", "residual", "converged")
        },
    )
    destination = args.run / f"width-{args.width}-process-{args.process}"
    destination.with_suffix(".json").write_text(json.dumps(row, indent=2))
    np.savez_compressed(
        destination.with_suffix(".npz"),
        **{f"moment_{i}": np.asarray(value) for i, value in enumerate(result.moments)},
    )
    assert all(row["converged"]) and max(row["residual"]) <= 1e-10
    for name, budget in (
        ("axis_ratio", 5e-4),
        ("saddle_ratio", 5e-4),
        ("boundary_ratio", 1e-4),
    ):
        assert np.all(np.isfinite(precision[name])) and max(precision[name]) <= budget
    print(
        "SUMMARY "
        + json.dumps(
            dict(
                width=args.width,
                process=args.process,
                reads=max(row["reads"]),
                updates=max(row["steps"]),
                wall_ms=row["wall_ms"],
                per_slice_ms=row["wall_ms"] / args.width,
                residual=max(row["residual"]),
                peak_bytes=memory["peak_bytes_in_use"],
                precision={name: max(values) for name, values in precision.items()},
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
