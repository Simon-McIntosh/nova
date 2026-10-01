"""Measure a map tangent at a recorded stationary iteration."""

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import subprocess
import hashlib
from time import perf_counter

import numpy as np
from nova.jax.config import configure_dtypes

configure_dtypes()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

REFERENCE = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-codex-20260916/cca-why-the-certificate-rows-do-not-converge-after-the-clip"
)
sys.path.insert(0, str(REFERENCE))
from instrument_boundary_band import _build  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402

fixture._cache_lock = lambda store: nullcontext(0.0)
assert jax.config.jax_enable_x64
start = perf_counter()
machine, operator, profile, exact, seed, target, seed_receipt, coordinates, analytic = (
    _build("weak-rotation-reactor-static", -110)
)
state = jnp.asarray(
    json.loads((REFERENCE / "record/cells-110/receipt.json").read_text())["trips"][-1][
        "flux"
    ]
)
external = operator.external()
program = jax.jit(operator.traced_flux_map(target_current=target))


def mapped(value):
    return program(value, external, operator, jnp.asarray(target))


image = mapped(state)
direction = image - state
tangent = jax.jvp(mapped, (state,), (direction,))[1]
control = (
    jax.jvp(lambda value: mapped(value) + value, (state,), (direction,))[1] - tangent
)
base_current = np.asarray(operator.cell_current_moments(state).cell_current)
data = {
    "backend": jax.default_backend(),
    "devices": str(jax.devices()),
    "revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "quadrature_sha256": hashlib.sha256(
        Path("nova/equilibrium/clip_quadrature.py").read_bytes()
    ).hexdigest(),
    "map_tangent_max_wb": float(jnp.max(jnp.abs(tangent))),
    "control_error": float(jnp.max(jnp.abs(control - direction))),
    "steps": [],
}
cotangent = jnp.linspace(-1.0, 1.0, state.size)
adjoint = jax.vjp(mapped, state)[1](cotangent)[0]
data["adjoint_finite"] = bool(jnp.all(jnp.isfinite(adjoint)))
data["duality_relative_error"] = float(
    jnp.abs(jnp.vdot(tangent, cotangent) - jnp.vdot(direction, adjoint))
    / jnp.maximum(jnp.abs(jnp.vdot(tangent, cotangent)), 1e-30)
)
for step in (1e-6, 1e-8):
    plus, minus = mapped(state + step * direction), mapped(state - step * direction)
    fd = (plus - minus) / (2 * step)
    current = np.asarray(
        operator.cell_current_moments(state + step * direction).cell_current
    )
    data["steps"].append(
        {
            "step": step,
            "fd_max_wb": float(jnp.max(jnp.abs(fd))),
            "relative_error": float(
                jnp.linalg.norm(fd - tangent) / jnp.maximum(jnp.linalg.norm(fd), 1e-30)
            ),
            "map_jump_wb": float(jnp.max(jnp.abs(plus - image))),
            "cell_66_delta_a": float(current[66] - base_current[66]),
        }
    )
data["wall_seconds"] = perf_counter() - start
Path(sys.argv[1]).write_text(json.dumps(data, indent=2))
print(json.dumps(data, indent=2), flush=True)
