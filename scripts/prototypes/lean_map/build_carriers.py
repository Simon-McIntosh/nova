"""Build device certificate carriers and exact-support exteriors in one run cache."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import jax
import numpy as np
import nova

from benchmarks import solovev_certificate as certificate
from nova.database.zarrstore import ZarrStore
from scripts.analytic_oracle_fixtures import measure as fixture


def run(kind: str, cells: int, cache: Path, output: Path):
    assert jax.config.jax_enable_x64 is True and jax.default_backend() == "gpu"
    if os.environ.get("NOVA_ANALYTIC_ORACLE_BUILDER") != "device":
        raise RuntimeError("the carrier must use the device builder")
    print(
        f"nova={nova.__file__} builder={fixture.__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    cache.mkdir(parents=True, exist_ok=True)
    original_store = fixture.ZarrStore
    fixture.ZarrStore = lambda *, filename, dirname, group=None: ZarrStore(
        filename=filename, dirname=cache, group=group
    )
    name = (
        "weak-rotation-reactor-static" if kind == "limited" else "diverted-single-null"
    )
    carrier, source, exact = certificate._case(name, clip_mode="exact")
    started = perf_counter()
    try:
        machine = certificate._case_machine(
            name, carrier, exact, -cells, clip_mode="exact"
        )
        machine_seconds = perf_counter() - started
        coordinates = np.vstack(
            (machine.node, machine.wall_node, machine.sample_coordinates)
        )
        analytic = certificate._exact_state(name, exact, coordinates)
        assert np.ptp(analytic) > 1e-10
        operator = fixture.forward_operator(source, machine)
        _, exterior, receipt = fixture.cached_fixture_exterior(
            source, exact, machine, operator, analytic
        )
    finally:
        fixture.ZarrStore = original_store
    assert np.max(np.abs(exterior)) > 0
    result = {
        "kind": kind,
        "requested_cells": cells,
        "realised_cells": len(machine.node),
        "builder": "device",
        "machine_cache": machine.cache,
        "exterior_cache": receipt,
        "machine_seconds": machine_seconds,
        "total_seconds": perf_counter() - started,
        "analytic_flux_range": float(np.ptp(analytic)),
        "exterior_flux_range": float(np.ptp(exterior)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("RESULT " + json.dumps(result, default=str), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("limited", "diverted"), required=True)
    parser.add_argument("--cells", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.kind, args.cells, args.cache, args.output)


if __name__ == "__main__":
    main()
