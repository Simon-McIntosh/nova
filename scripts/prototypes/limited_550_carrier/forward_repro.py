"""Reproduce a refusal while building the forward operator on the limited carrier."""

from __future__ import annotations

import sys
import traceback

from nova.jax.config import configure_dtypes

configure_dtypes()

import nova  # noqa: E402
from scripts.analytic_oracle_fixtures import measure as oracle_fixture  # noqa: E402
from tests.rotating_equilibrium_references import reference_cases  # noqa: E402

REQUESTED = -500


def main() -> int:
    print(f"nova={nova.__file__}")
    case = reference_cases()["weak-rotation-reactor"].static_limit()
    machine = oracle_fixture.cached_machine(
        case, REQUESTED, wall_nodes=oracle_fixture.WALL_POINT_COUNT
    )
    print(f"realised_cells={len(machine.node)} hit={machine.cache.get('hit')}")
    try:
        operator = oracle_fixture.forward_operator(case, machine)
    except Exception:
        print("FORWARD-OPERATOR-REFUSED")
        traceback.print_exc()
        return 1
    print(f"operator built moment_cells={int(operator.area.shape[0])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
