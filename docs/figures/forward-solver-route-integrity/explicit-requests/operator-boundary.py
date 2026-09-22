"""Measure nested calls and a request stream on the real weak-row operator."""

import json
import os
from pathlib import Path
import re
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.solve_program_size_gate import _certificate_operands
from nova.equilibrium.fixed_point import (
    OperatorRequestKind,
    operator_request,
    operator_request_body,
)
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64
assert os.environ.get("SLURM_JOB_ID")
assert jax.default_backend() == "cpu"
root = Path.cwd()
out = root / "docs/figures/forward-solver-route-integrity/explicit-requests"
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print(
    f"revision={revision} plus explicit request body; tree={root}; "
    "command=python operator-boundary.py",
    flush=True,
)
profile, seed, topology, target, request = _certificate_operands(
    "weak-rotation-reactor-static", -300
)
operator = profile.operator
external = operator.external(request.current, request.prescribed_current)
seed = jnp.asarray(seed, dtype=jnp.float64)
shadow = operator.residual_shadow_mask(seed, topology)
map_fn = operator.traced_flux_map_with_shadow(topology, target)
body = operator_request_body(map_fn)


@jax.jit
def evaluated(state, shadow, external, operator):
    response = body(
        operator_request(OperatorRequestKind.RESIDUAL, state, shadow),
        external,
        operator,
        target,
    )
    with jax.named_scope("live_operator_result"):
        return jax.lax.optimization_barrier(response.mapped)


@jax.jit
def direct(state, shadow, external, operator):
    for _ in range(3):
        state = evaluated(state, shadow, external, operator)
    return state


@jax.jit
def streamed(state, shadow, external, operator):
    def advance(carry, _):
        return evaluated(carry, shadow, external, operator), None

    return jax.lax.scan(advance, state, None, length=3)[0]


receipt = {
    "revision": revision,
    "requested_cells": 300,
    "job_id": os.environ["SLURM_JOB_ID"],
    "programs": {},
}
outputs = {}
for name, function in (("one", evaluated), ("nested", direct), ("streamed", streamed)):
    start = time.perf_counter()
    compiled = function.lower(seed, shadow, external, operator).compile()
    wall = time.perf_counter() - start
    hlo = compiled.as_text()
    barriers = [
        line
        for line in hlo.splitlines()
        if re.search(rf" = f64\[{seed.size}\].* select\(", line)
        and "jit(evaluated)/jit(evaluate)" in line
    ]
    outputs[name] = np.asarray(compiled(seed, shadow, external, operator))
    receipt["programs"][name] = {
        "operator_boundaries": len(barriers),
        "boundary_operation": "full-flux live residual-shadow select",
        "compile_seconds": wall,
        "instructions": len(
            re.findall(r"^\s*(?:ROOT\s+)?%[\w.-]+ = ", hlo, re.MULTILINE)
        ),
    }
    print(name, json.dumps(receipt["programs"][name]), flush=True)
    (out / f"real-operator-{name}.hlo.txt").write_text(hlo)
    (out / "operator-boundary.json").write_text(json.dumps(receipt, indent=2) + "\n")
assert receipt["programs"]["one"]["operator_boundaries"] == 1
assert receipt["programs"]["nested"]["operator_boundaries"] == 3
assert receipt["programs"]["streamed"]["operator_boundaries"] == 1
np.testing.assert_array_equal(outputs["nested"], outputs["streamed"])
print(
    "REAL_OPERATOR_BOUNDARY_PASS one=1 nested=3 streamed=1; terminal bit-identical",
    flush=True,
)
