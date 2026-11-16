"""Check that one completion call widens every target the solve contracts on.

The solve contracts each target's conductor response against the resolved
current vector, so a target left at the shipped conductor order is a shorter
operand than the vector and the contraction refuses it. The current adapter is
the single place that appends the circuit-driven columns, so it is the place
that has to reach every target.

The check runs without IMAS data: it stubs the coil response and a profile
whose three targets all carry the shipped order, then reads the widths back
after one completion call.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from nova.imas import diiid_current
from nova.imas.diiid_current import complete_profile_current_adapter
from nova.imas.diiid_description import CIRCUIT_DRIVEN_CONDUCTORS, POLOIDAL_CONDUCTORS

SHIPPED = tuple(POLOIDAL_CONDUCTORS)
ROWS = {"grid": 7, "wall": 5, "sample": 6}


@dataclass(frozen=True)
class StubTarget:
    """One solve target, carrying a conductor response and its coordinates."""

    source_target: jnp.ndarray
    coordinate: jnp.ndarray

    @property
    def node_number(self) -> int:
        return int(self.coordinate.shape[0])


@dataclass(frozen=True)
class StubOperator:
    """The three targets the DIII-D builder attaches to a booked operator."""

    grid: StubTarget
    wall: StubTarget
    sample: StubTarget | None
    external_current: object = None


@dataclass(frozen=True)
class StubProfile:
    """A profile whose operator carries the three booked-route targets."""

    operator: StubOperator


def _stub_target(rows: int) -> StubTarget:
    coordinate = np.stack(
        (np.linspace(1.0, 2.0, rows), np.linspace(-1.0, 1.0, rows)), axis=1
    )
    response = np.arange(rows * len(SHIPPED), dtype=float).reshape(rows, len(SHIPPED))
    return StubTarget(
        source_target=jnp.asarray(response), coordinate=jnp.asarray(coordinate)
    )


def _stub_response(_entry, _version, names, radius, height):
    """Return one synthetic circuit response column per requested conductor."""

    rows = np.shape(radius)[0]
    assert np.shape(height)[0] == rows
    response = np.full((len(names), rows), 0.5)
    return tuple(names), response, {"stub": True, "rows": int(rows)}


def main() -> int:
    diiid_current.active_coil_response_from_imas = _stub_response
    profile = StubProfile(
        operator=StubOperator(
            grid=_stub_target(ROWS["grid"]),
            wall=_stub_target(ROWS["wall"]),
            sample=_stub_target(ROWS["sample"]),
        )
    )
    adapter = complete_profile_current_adapter(
        profile,
        shipped_names=SHIPPED,
        shipped_current_a={name: 1.0e5 for name in SHIPPED},
    )
    operator = adapter.profile.operator
    expected = len(SHIPPED) + len(CIRCUIT_DRIVEN_CONDUCTORS)
    print(f"SHIPPED {len(SHIPPED)} CONDUCTOR_VECTOR {len(adapter.resolution.names)}")
    failures = []
    for label, target in (
        ("grid", operator.grid),
        ("wall", operator.wall),
        ("sample", operator.sample),
    ):
        if target is None:
            print(f"TARGET {label} absent")
            failures.append(label)
            continue
        width = int(target.source_target.shape[1])
        rows = int(target.source_target.shape[0])
        print(f"TARGET {label} rows {rows} columns {width}")
        if width != expected:
            failures.append(label)
        if rows != ROWS[label]:
            failures.append(label)
    if failures:
        print(f"SEAM FAILED: {' '.join(sorted(set(failures)))}")
        return 1
    print("SEAM OK: every target carries the conductor vector's order")
    return 0


if __name__ == "__main__":
    sys.exit(main())
