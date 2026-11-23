"""Drive the public forward solve on one stored MAST case and record its receipt.

The arm shape follows the slice-profile driver: a fresh process, the tree on
``PYTHONPATH``, ``JAX_PLATFORMS`` pinned to cpu, one case, and a receipt naming
the revision and the resolved working directory.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    from nova.jax.config import configure_dtypes

    configure_dtypes()
    import jax
    import numpy as np
    from nova.equilibrium import forward as forward_module
    from nova.equilibrium.forward import ForwardProfile

    assert jax.config.jax_enable_x64 is True
    assert jax.default_backend() == "cpu"

    from tests.magnetics_fit_harness import _physical_problem

    profile, request, coordinate, wall, sections, known = _physical_problem("mast")
    assert isinstance(profile, ForwardProfile)

    args.output.mkdir(parents=True, exist_ok=True)
    record = dict(
        revision=args.revision,
        route="public-forward-solve",
        case="mast",
        driver=str(Path(__file__).resolve()),
        forward_entry="nova.equilibrium.forward.ForwardProfile.solve",
        forward_module=str(Path(forward_module.__file__).resolve()),
        cwd=str(Path.cwd().resolve()),
        backend=str(jax.default_backend()),
        x64=bool(jax.config.jax_enable_x64),
        complete=False,
    )

    def save():
        (args.output / "receipt.json").write_text(json.dumps(record, indent=2))

    print("HEADER", json.dumps(record), flush=True)
    save()

    # One call traces, compiles and executes the public entry; the line map
    # needs the entry to have run, not a repeated timing.
    started = perf_counter()
    result = profile.solve(request)
    record.update(
        residual=float(np.asarray(result.terminal_state.fixed_point.residual)),
        converged=bool(result.terminal_state.fixed_point.converged),
        flux_entries=int(np.asarray(result.terminal_state.flux).size),
        wall_ms=1000 * (perf_counter() - started),
    )
    save()
    assert record["converged"]
    record["complete"] = True
    save()
    print("SUMMARY", json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
