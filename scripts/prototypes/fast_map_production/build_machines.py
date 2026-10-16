"""Build the analytic oracle carriers and exteriors a measurement lane needs.

The certified store holds only the 5000-cell hex carrier, and a measurement
lane reads a carrier by ``(analytic_case, requested_cells, builder)`` and its
exterior by the carrier's semantic key. This writes both for each requested
cell count into the directory named by ``NOVA_ANALYTIC_ORACLE_CACHE`` so a
later job can open them read-only. Both stores are cached, so a job that runs
out of wall clock resumes from whatever it already wrote.
"""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

import argparse
from time import perf_counter

import jax
import numpy as np

from benchmarks import solovev_certificate as certificate
from scripts.analytic_oracle_fixtures import measure as fixture

CASES = {
    "limited": "weak-rotation-reactor-static",
    "diverted": "diverted-single-null",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=tuple(CASES), action="append", required=True)
    parser.add_argument("--cells", type=int, action="append", required=True)
    arguments = parser.parse_args()
    print(f"jax={jax.__version__} backend={jax.default_backend()}", flush=True)

    for kind in arguments.kind:
        name = CASES[kind]
        for cells in arguments.cells:
            started = perf_counter()
            carrier, source, exact = certificate._case(name, clip_mode="exact")
            machine = certificate._case_machine(name, carrier, exact, -cells)
            machine_seconds = perf_counter() - started
            started = perf_counter()
            coordinates = np.vstack(
                (machine.node, machine.wall_node, machine.sample_coordinates)
            )
            analytic = certificate._exact_state(name, exact, coordinates)
            empty = fixture.forward_operator(source, machine)
            _physical, _external, exterior_cache = fixture.cached_fixture_exterior(
                source, exact, machine, empty, analytic
            )
            exterior_seconds = perf_counter() - started
            print(
                f"built kind={kind} case={name} cells={cells} "
                f"nodes={len(machine.node)} "
                f"machine={machine.cache.get('store')} "
                f"machine_hit={machine.cache.get('hit')} "
                f"exterior={exterior_cache.get('store')} "
                f"exterior_hit={exterior_cache.get('hit')} "
                f"machine_seconds={machine_seconds:.1f} "
                f"exterior_seconds={exterior_seconds:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()