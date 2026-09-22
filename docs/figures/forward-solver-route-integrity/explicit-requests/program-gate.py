"""Inspect an explicit-operand weak certificate program on a CPU allocation."""

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import jax
import jax.numpy as jnp

from benchmarks.solve_program_size_gate import _certificate_operands
from nova.jax.config import configure_dtypes

configure_dtypes()
assert jax.config.jax_enable_x64
assert os.environ.get("SLURM_JOB_ID")
assert jax.default_backend() == "cpu"
cells = int(sys.argv[1])
output = Path(sys.argv[2])
mode = sys.argv[3]
mutation = "revert the single body to per-request evaluation"
if mode == "per-request":
    print(mutation, flush=True)
    from nova.equilibrium import fixed_point

    def per_request_evaluation(body, requests, *arguments):
        count = requests.state.shape[0]
        print(f"MUTATION_APPLIED unrolled_request_count={count}", flush=True)
        responses = [
            body(jax.tree.map(lambda value: value[index], requests), *arguments)
            for index in range(count)
        ]
        return jax.tree.map(lambda *values: jnp.stack(values), *responses)

    fixed_point.run_operator_requests = per_request_evaluation
source_digest = hashlib.sha256(
    b"".join(
        Path(p).read_bytes()
        for p in (
            "nova/equilibrium/fixed_point.py",
            "nova/equilibrium/reduced_newton.py",
        )
    )
).hexdigest()
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
start = time.perf_counter()
lowered = program.lower(
    jnp.asarray(seed, dtype=jnp.float64), external, profile.operator
)
print(f"lower_seconds={time.perf_counter() - start}", flush=True)
start = time.perf_counter()
compiled = lowered.compile()
wall = time.perf_counter() - start
hlo = compiled.as_text()
count = len(re.findall(r"^\s*(?:ROOT\s+)?%[\w.-]+ = ", hlo, re.MULTILINE))
assert count > 0
hlo_root = Path(
    "/home/ITER/mcintos/.config/reckon/crew/runs/"
    "r-20260922T103934949557-fsri-one-traced-operator-state-machine/optimized-hlo"
)
hlo_root.mkdir(parents=True, exist_ok=True)
hlo_path = hlo_root / f"{mode}-{cells}.hlo.txt"
hlo_path.write_text(hlo)
receipt = dict(
    source_digest=source_digest,
    mode=mode,
    hlo_path=str(hlo_path),
    revision=revision,
    tree=str(Path.cwd()),
    cells=cells,
    optimized_instructions=count,
    compile_seconds=wall,
    job_id=os.environ["SLURM_JOB_ID"],
    platform=jax.default_backend(),
)
output.write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)

if mode == "per-request":
    candidate = json.loads((output.parent / f"candidate-{cells}.json").read_text())
    assert count > candidate["optimized_instructions"], (
        f"negative control failed: {count} instructions did not rise above "
        f"{candidate['optimized_instructions']}"
    )
    print(
        f"NEGATIVE_CONTROL_PASS instructions={count} "
        f"shared={candidate['optimized_instructions']}",
        flush=True,
    )
else:
    ceiling = {300: 203078, 1000: 204531}[cells]
    assert count <= ceiling, (
        f"operator state-machine ceiling refused: {cells} cells has "
        f"{count} instructions > {ceiling}"
    )
