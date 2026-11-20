"""Time one warm bank Newton step on the device, with no read inside it.

The program compiled here is ``support_read.newton_step`` with the
fresh-geometry booking tangent.  Its arguments are runtime values, so the same
executable serves every state of the certificate's shape bucket.  A fresh
process with the persistent cache disabled compiles the whole-step program,
then warms it and times eight calls.  One profiled warm call supplies the
kernel-launch count.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax
import jax.numpy as jnp
import numpy as np

import nova
from nova.equilibrium import support_read
from scripts.prototypes.smfp_autodiff_step.fixture import Certificate, emit


def executable_bytes(executable):
    try:
        return len(executable.runtime_executable().serialize()), "serialized"
    except Exception as error:  # noqa: BLE001 - the refusal is recorded, not raised
        method = getattr(
            executable.runtime_executable(),
            "size_of_generated_code_in_bytes",
            None,
        )
        if callable(method):
            return int(method()), f"generated-code fallback: {error}"
        return None, f"unavailable: {error}"


def profiled_launches(trace_dir, call, arguments):
    trace_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(trace_dir))
    result = call(*arguments)
    jax.block_until_ready(result)
    jax.profiler.stop_trace()
    paths = sorted(trace_dir.rglob("*.json.gz"))
    if not paths:
        raise RuntimeError("the profiler wrote no trace")
    with gzip.open(paths[-1], "rt") as stream:
        events = json.load(stream)["traceEvents"]
    devices = {
        event["pid"]
        for event in events
        if event.get("ph") == "M"
        and "/device:GPU" in event.get("args", {}).get("name", "")
    }
    kernels = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("pid") in devices
        and not event.get("name", "").startswith(("Memcpy", "Memset"))
    ]
    copies = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("pid") in devices
        and event.get("name", "").startswith(("Memcpy", "Memset"))
    ]
    return len(kernels), len(copies), sum(item["dur"] for item in kernels) / 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--state", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    print(
        "REVISION=" + os.environ.get("SMFP_REVISION", "unknown"),
        flush=True,
    )
    emit(
        "PROCESS",
        nova=nova.__file__,
        module=support_read.__file__,
        cwd=str(Path.cwd().resolve()),
        backend=jax.default_backend(),
        devices=[str(device) for device in jax.local_devices()],
        cache_root=jax.config.jax_compilation_cache_dir,
        cache_min_compile=jax.config.jax_persistent_cache_min_compile_time_secs,
        x64=bool(jax.config.jax_enable_x64),
    )
    if jax.default_backend() != "gpu":
        raise SystemExit("the warm step gate needs a GPU")

    certificate = Certificate(args.store)
    emit(
        "BANK",
        bucket=int(certificate.bank.bucket),
        nodes=int(certificate.bank.active_nodes),
    )
    moments = certificate.seed_moments
    if args.state is not None:
        ratio = 1.0 + args.state
        moments = tuple(value * ratio for value in moments)
    support = certificate.read(moments)
    jax.block_until_ready(support)
    emit(
        "SUPPORT", valid=bool(support.valid), qualified=bool(support.reading.qualified)
    )

    step = support_read.newton_step
    program = jax.jit(
        lambda b, m, s: step(
            b,
            m,
            s,
            certificate.profile,
            certificate.gradient,
            certificate.quadrature,
            fresh_geometry=True,
            condition_diagnostics=False,
        )
    )
    arguments = (certificate.bank, moments, support)
    started = perf_counter()
    executable = program.lower(*arguments).compile()
    first = program(*arguments)
    jax.block_until_ready(first)
    compile_seconds = perf_counter() - started
    size, size_method = executable_bytes(executable)
    emit(
        "COMPILE",
        seconds=compile_seconds,
        executable_bytes=size,
        size_method=size_method,
        within_60s=bool(compile_seconds <= 60.0),
        under_50MB=bool(size is not None and size < 50_000_000),
    )

    program(*arguments)  # warm outside the profile
    jax.block_until_ready(program(*arguments))
    walls = []
    for _ in range(8):
        started = perf_counter()
        result = program(*arguments)
        jax.block_until_ready(result)
        walls.append(1e3 * (perf_counter() - started))
    median = float(np.median(walls))
    emit(
        "WARM_STEP",
        walls_ms=walls,
        median_ms=median,
        minimum_ms=float(np.min(walls)),
        within_0p5ms=bool(median <= 0.5),
        linear_residual=float(result.linear_residual),
        update_finite=bool(jnp.all(jnp.isfinite(jnp.concatenate(result.update)))),
    )

    kernels, copies, device_ms = profiled_launches(args.trace, program, arguments)
    emit(
        "LAUNCHES",
        kernels=kernels,
        copies=copies,
        device_ms=device_ms,
        within_100=bool(kernels <= 100),
    )
    args.out.write_text(
        json.dumps(
            {
                "compile_seconds": compile_seconds,
                "executable_bytes": size,
                "size_method": size_method,
                "median_ms": median,
                "walls_ms": walls,
                "kernels": kernels,
                "copies": copies,
                "device_ms": device_ms,
            },
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
