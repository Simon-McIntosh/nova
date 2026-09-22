"""Measure the current weak-rotation certificate terminal identity."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

from nova.jax.config import configure_dtypes

configure_dtypes()

import jax  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks import solovev_certificate as certificate  # noqa: E402
from instrument_boundary_band import _build  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as fixture  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixture._cache_lock = lambda _store: nullcontext(0.0)

    def refuse_store(*_args, **_kwargs):
        raise RuntimeError("fixture cache miss requires an out-of-scope cache write")

    fixture.ZarrStore.store = refuse_store
    case = "weak-rotation-reactor-static"
    machine, _operator, profile, _exact, seed, target, *_rest = _build(case, -300)
    request = certificate._certificate_solve_request(
        profile,
        seed,
        target,
        carrier_identity=f"solovev:{case}:-300",
    )
    receipt = profile.solve(request)
    flux = np.asarray(jax.block_until_ready(receipt.equilibrium.flux), dtype=np.float64)
    fixed_point = receipt.equilibrium.fixed_point
    result = {
        "case": case,
        "requested_cells": 300,
        "realised_cells": len(machine.node),
        "state_digest": hashlib.sha256(flux.tobytes()).hexdigest(),
        "residual": float(fixed_point.residual),
        "converged": bool(fixed_point.converged),
        "active_set_iterations": int(fixed_point.active_set_iterations),
        "termination_reason": str(receipt.termination_reason),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
