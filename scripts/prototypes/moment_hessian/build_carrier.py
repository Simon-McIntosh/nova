"""Build the limited certificate carrier in the caller's fixture store."""

import json
import os
from pathlib import Path
from time import perf_counter

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import nova  # noqa: E402
from benchmarks import solovev_certificate as certificate  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402
from scripts.prototypes.support import certificate_fixtures  # noqa: E402


def main():
    assert jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
    root = Path(os.environ["NOVA_BOOK_FIXTURE_ROOT"]).resolve()
    print(
        f"nova={nova.__file__} module={fixture.__file__} cwd={Path.cwd().resolve()}",
        flush=True,
    )
    print(f"fixture_root={root}", flush=True)
    carrier, _, _ = certificate._case("weak-rotation-reactor-static", clip_mode="exact")
    identity = fixture.cache_identity(
        carrier, requested_cells=-500, wall_nodes=fixture.WALL_POINT_COUNT, wall=None
    )
    digest = certificate_fixtures.fixture_digest(identity)
    started = perf_counter()
    machine = certificate_fixtures.cached_carrier(carrier, -500, None)
    entry = certificate_fixtures.fixture_entry(
        certificate_fixtures.CARRIER_KIND, digest
    )
    assert entry is not None and entry.is_relative_to(root)
    receipt = {
        "digest": digest,
        "cells": len(machine.node),
        "build_wall_seconds": perf_counter() - started,
        "cache": machine.cache,
        "entry": str(entry),
    }
    print("CARRIER " + json.dumps(receipt, default=str), flush=True)


if __name__ == "__main__":
    main()
