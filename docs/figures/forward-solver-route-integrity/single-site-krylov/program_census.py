"""Compile the weak certificate program and write its optimised HLO.

Arguments: cells, receipt path, mode (``stream`` or ``per-site``), HLO path.
``per-site`` first rebinds the qualified Krylov step to the base body that
calls the operator at each application site.
"""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from nova.jax.config import configure_dtypes

configure_dtypes()
import jax
import jax.numpy as jnp

assert jax.config.jax_enable_x64 is True
assert jax.default_backend() == "cpu"
cells, output, mode, hlo_path = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
sys.path.insert(0, "docs/figures/forward-solver-route-integrity/single-site-krylov")
if mode == "per-site":
    import per_site

    print(per_site.MUTATION, flush=True)
    per_site.install()
from benchmarks.solve_program_size_gate import _certificate_operands

revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print(f"revision={revision} tree={Path.cwd()} command={sys.argv}", flush=True)
profile, seed, requested_class, target_current, request = _certificate_operands(
    "weak-rotation-reactor-static", -cells
)
external = profile.operator.external(request.current, request.prescribed_current)
program = profile._accelerated_history_program(
    request.route,
    requested_class=requested_class,
    target_current=target_current,
    **request.policy.kernel_options(),
)
lowered = program.lower(jnp.asarray(seed, dtype=jnp.float64), external, profile.operator)
start = time.perf_counter()
compiled = lowered.compile()
wall = time.perf_counter() - start
hlo = compiled.as_text()
hlo_path.write_text(hlo)
count = len(re.findall(r"^\s*(?:ROOT\s+)?%[\w.-]+ = ", hlo, re.MULTILINE))
receipt = dict(
    mode=mode,
    revision=revision,
    cells=cells,
    optimized_instructions=count,
    compile_seconds=wall,
    hlo_path=str(hlo_path),
    job_id=os.environ.get("SLURM_JOB_ID"),
)
output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
