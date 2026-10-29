"""Split one program's cold compile into trace, lowering and XLA compile.

The cold-process setup and the point jet program come from the kernel compile
census worker, so this profile measures the same program the census counted and
records the same compilation-cache evidence it does. The XLA compile is dumped
pass by pass for a companion analysis to attribute its wall.
"""

import argparse
import json
from pathlib import Path
import resource
import sys
import time
from time import perf_counter


from scripts.prototypes.kernel_compile_census import row

import jax
import jax.numpy as jnp
import nova
from nova.biot import polygonanalytic

assert jax.config.jax_enable_x64 is True
assert not jax.config.jax_enable_compilation_cache


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
    if name == "control":
        return control(size, 64)
    return row.program(name, size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    revision = args.revision
    print(
        f"revision={revision} tree={Path.cwd().resolve()} "
        f"command={json.dumps(sys.argv)}"
    )
    print(
        f"nova={nova.__file__} measurement_module={polygonanalytic.__file__} "
        f"cwd={Path.cwd().resolve()} profiler={__file__} device={jax.devices()[0]}"
    )
    print(f"cache_enabled={bool(jax.config.jax_enable_compilation_cache)}")
    print(f"xla_flags={__import__('os').environ.get('XLA_FLAGS', '')}")
    receipt = {
        "program": args.program,
        "size": args.size,
        "revision": revision,
        "tree": str(Path.cwd().resolve()),
        "nova_file": nova.__file__,
        "measurement_module": polygonanalytic.__file__,
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
    ledger = row.cache_guard.ledger()
    if not ledger.rows():
        ledger.record(
            f"uncached-{args.program}-{args.size}",
            args.program,
            receipt["compile_seconds"],
            "miss",
            persisted=False,
        )
    receipt["cache_guard_rows"] = ledger.rows()
    receipt["cache_guard_misses"] = ledger.miss_count()
    receipt["cache_guard_hits"] = ledger.hit_count()
    assert ledger.miss_count() > 0 and ledger.hit_count() == 0
    checkpoint("done")


if __name__ == "__main__":
    main()
