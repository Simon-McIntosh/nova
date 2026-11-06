"""Compare discrete read decisions at common trial and converged states."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import ast
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read, topology
from scripts.prototypes.bank_read_count_sensitivity.capture import masks, signatures
from scripts.prototypes.bank_solve_profile import fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            dict(
                nova=nova.__file__,
                module=topology.__file__,
                cwd=str(Path.cwd()),
                revision=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                base=args.base,
                x64=jax.config.jax_enable_x64,
            )
        ),
        flush=True,
    )
    assert jax.config.jax_enable_x64
    source = subprocess.check_output(
        ["git", "show", f"{args.base}:nova/equilibrium/topology.py"], text=True
    )
    definition = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "_curvature_derivatives"
    )
    namespace = dict(vars(topology))
    exec(
        compile(
            ast.Module(body=[definition], type_ignores=[]), "<base-curvature>", "exec"
        ),
        namespace,
    )
    original = namespace["_curvature_derivatives"]
    batch = fixture.build(16)
    common = np.load(args.run / "common-seeds.npz")
    converged = np.load(args.run / "reference-states.npz")
    states = [
        tuple(jnp.asarray(common[f"state_0_{i}"]) for i in range(3)),
        tuple(jnp.asarray(converged[f"moment_{i}"]) for i in range(3)),
    ]
    seeds, previous = jnp.asarray(common["seeds"]), jnp.asarray(common["previous"])
    results = {}
    for arm in ("base", "head"):

        def read(bank, cache, state, seeds, previous):
            return support_read.read_batch(
                bank, cache, state, seeds, previous, batch.convention, batch.policy
            )

        program = jax.jit(read)
        context = patch.object(
            topology,
            "_curvature_derivatives",
            original if arm == "base" else topology._curvature_derivatives,
        )
        with context:
            compiled = program.lower(
                batch.operands, batch.cache, states[0], seeds, previous
            ).compile()
        results[arm] = []
        for state in states:
            value = jax.block_until_ready(
                compiled(batch.operands, batch.cache, state, seeds, previous)
            )
            assert np.all(value.valid)
            selected = [np.asarray(v) for v in masks(value.reading)]
            for v in selected:
                changed = v.copy()
                changed.flat[0] = ~changed.flat[0]
                assert signatures(v) != signatures(changed)
            results[arm].append([v.tolist() for v in selected])
    changes = []
    for before, after in zip(results["base"], results["head"]):
        changes.append(
            [
                int(np.count_nonzero(np.asarray(a) != np.asarray(b)))
                for a, b in zip(before, after)
            ]
        )
    (args.run / "decisions.json").write_text(
        json.dumps(dict(base=args.base, changes=changes, control=True, masks=results))
    )
    print("DECISIONS " + json.dumps(changes), flush=True)
    assert all(not any(row) for row in changes)


if __name__ == "__main__":
    main()
