"""Build one diverted analytic carrier in a caller-owned store."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402

import nova  # noqa: E402
from benchmarks import solovev_certificate as certificate  # noqa: E402
from nova.database.zarrstore import ZarrStore  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402


assert jax.config.jax_enable_x64 is True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") != "false":
        raise RuntimeError("CUDA preallocation must be disabled")
    if fixture.resolve_builder(None) != fixture.BUILDER_DEVICE:
        raise RuntimeError("the device coupling builder was not selected")
    if jax.default_backend() != "gpu" or len(jax.devices("gpu")) != 1:
        raise RuntimeError("one GPU device is required")
    print(f"nova.__file__={nova.__file__}", flush=True)
    print(f"device={jax.devices('gpu')[0]}", flush=True)

    def scoped_store(*, filename, dirname, group=None):
        return ZarrStore(filename=filename, dirname=output / "cache", group=group)

    fixture.ZarrStore = scoped_store
    carrier, _source, exact = certificate._case(
        "diverted-single-null", clip_mode="exact"
    )
    started = perf_counter()
    machine = certificate._case_machine(
        "diverted-single-null", carrier, exact, -10000, clip_mode="exact"
    )
    wall = perf_counter() - started
    store_bytes = sum(
        path.stat().st_size for path in (output / "cache").rglob("*") if path.is_file()
    )
    receipt = {
        "status": "complete",
        "realised_cells": int(len(machine.node)),
        "ring_count": int(len(machine.stencil)),
        "build_wall_seconds": wall,
        "store_bytes": store_bytes,
        "builder": fixture.resolve_builder(None),
        "device": str(jax.devices("gpu")[0]),
    }
    (output / "build.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
