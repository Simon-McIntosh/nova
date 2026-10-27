"""Split one program's cold compile into trace, lowering and XLA compile.

Run inside a fresh process with the compilation cache disabled; the XLA compile
is dumped pass by pass so a companion analysis can attribute its wall to the
individual HLO passes.
"""

import os

os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
from time import perf_counter

import numpy as np

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import nova

assert jax.config.jax_enable_x64 is True
assert not jax.config.jax_enable_compilation_cache


def point_jet(size):
    from nova.biot import topology
    from nova.equilibrium.clip_quadrature import ClippedCurrentMoments
    from tests.equilibrium.test_topology_read import _hexagon

    points = jnp.tile(jnp.asarray((2.0, 0.1)), (size, 1))
    coupling = topology.BiotMomentCoupling.from_polygons(
        (_hexagon(0.08) + np.asarray((1.5, 0.0)),)
    )
    moments = ClippedCurrentMoments(
        jnp.asarray((1000.0,)), jnp.asarray((0.2,)), jnp.asarray((-0.3,))
    )
    return (
        lambda target, kernel, current: jax.vmap(
            lambda point: kernel.evaluate(point, current)
        )(target),
        (points, coupling, moments),
    )


def control(depth, size):
    """A chain of elementwise work: compiles in seconds, touches no project code."""

    def composed(value):
        scale = jnp.asarray(1.0001)
        bias = jnp.asarray(0.3)
        decay = jnp.asarray(0.999)
        for step in range(depth):
            weight = jnp.asarray(1e-3 + 1e-9 * step)
            value = jnp.tanh(value * scale + bias) * decay + jnp.sin(value) * weight
        return jnp.sum(value)

    return composed, (jnp.linspace(0.1, 0.2, size),)


def build(name, size):
    if name == "point_jet":
        return point_jet(size)
    if name == "control":
        return control(size, 64)
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        f"revision={revision} tree={Path.cwd().resolve()} "
        f"command={json.dumps(sys.argv)}"
    )
    print(f"nova={nova.__file__} module={__file__} device={jax.devices()[0]}")
    print(f"cache_enabled={bool(jax.config.jax_enable_compilation_cache)}")
    print(f"xla_flags={os.environ.get('XLA_FLAGS', '')}")
    receipt = {
        "program": args.program,
        "size": args.size,
        "revision": revision,
        "tree": str(Path.cwd().resolve()),
        "nova_file": nova.__file__,
        "device": getattr(jax.devices()[0], "device_kind", str(jax.devices()[0])),
        "cache_enabled": bool(jax.config.jax_enable_compilation_cache),
    }

    def checkpoint(phase):
        receipt["phase"] = phase
        receipt["peak_host_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, default=str) + "\n")
        print("PHASE " + phase, flush=True)

    function, operands = build(args.program, args.size)
    checkpoint("input-ready")
    started = perf_counter()
    graph = jax.make_jaxpr(function)(*operands)
    receipt["trace_seconds"] = perf_counter() - started
    receipt["jaxpr_text_bytes"] = len(str(graph).encode())
    checkpoint("traced")

    started = perf_counter()
    lowered = jax.jit(function).lower(*operands)
    receipt["lower_seconds"] = perf_counter() - started
    receipt["stablehlo_bytes"] = len(lowered.as_text().encode())
    checkpoint("lowered")

    started = perf_counter()
    compile_started = time.time()
    compiled = lowered.compile()
    receipt["compile_finished_epoch"] = time.time()
    receipt["compile_started_epoch"] = compile_started
    receipt["compile_seconds"] = perf_counter() - started
    checkpoint("compiled")

    runtime = compiled.runtime_executable()
    receipt["executable_bytes"] = len(runtime.serialize())
    receipt["compile_wall_seconds"] = (
        receipt["lower_seconds"] + receipt["compile_seconds"]
    )
    checkpoint("done")


if __name__ == "__main__":
    main()