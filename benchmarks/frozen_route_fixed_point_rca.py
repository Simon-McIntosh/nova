"""Measure why a warmed frozen Newton partition misses the live-map root."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from nova.equilibrium import fixed_point
from nova.jax.config import configure_dtypes


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "docs/figures/millisecond-converged-solve/newton-krylov-route/fixed-point-rca"
)
REPORT = OUTPUT / "report.md"
RECEIPT = OUTPUT / "receipt.json"


def _fixture_machine():
    """Construct the route fixture without duplicating its physical machine."""
    sys.path.insert(0, str(ROOT / "tests"))
    import test_equilibrium_forward_solve as route_fixture

    return route_fixture.machine.__wrapped__()


def _as_float(value: Any) -> float:
    return float(np.asarray(value))


def _relative(mapped: jax.Array, state: jax.Array) -> float:
    return _as_float(fixed_point._relative_residual(mapped, state))


def _partition_read(shadowed_map, state, previous, external, operator):
    return shadowed_map._read_frozen_partition(state, previous, external, operator)


def _partition_map(shadowed_map, state, partition, external, operator):
    return shadowed_map._map_frozen_partition(state, partition, external, operator)


def _changed(left: Any, right: Any) -> dict[str, Any]:
    """Summarise scalar and array changes in one frozen partition field."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    difference = right_array - left_array
    return {
        "shape": list(left_array.shape),
        "changed_entries": int(np.count_nonzero(left_array != right_array)),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "warm_value": float(left_array) if left_array.ndim == 0 else None,
        "terminal_value": float(right_array) if right_array.ndim == 0 else None,
    }


def _partition_fields(warmed, terminal) -> dict[str, dict[str, Any]]:
    """Expose all state-dependent values retained by the frozen map."""
    result: dict[str, dict[str, Any]] = {
        "label": _changed(warmed.label, terminal.label),
        "residual_shadow": _changed(warmed.residual_shadow, terminal.residual_shadow),
    }
    for name in warmed.topology._fields:
        result[f"topology.{name}"] = _changed(
            getattr(warmed.topology, name), getattr(terminal.topology, name)
        )
    warm_support = warmed.profile_support
    terminal_support = terminal.profile_support
    if warm_support is not None:
        names = warm_support._fields if hasattr(warm_support, "_fields") else ()
        if is_dataclass(warm_support):
            names = tuple(field.name for field in fields(warm_support))
        for name in names:
            result[f"profile_support.{name}"] = _changed(
                getattr(warm_support, name), getattr(terminal_support, name)
            )
    return result


def _residual_terms(live, frozen, state, coordinate, physical_count: int):
    """Name the largest map residual terms by physical cell or wall entry."""
    live_term = np.asarray(live - state)
    frozen_term = np.asarray(frozen - state)
    mismatch = live_term - frozen_term
    ranking = np.argsort(np.abs(live_term))[::-1][:12]
    terms = []
    for index in ranking:
        if index < physical_count:
            radius, height = np.asarray(coordinate[index])
            name = f"grid cell {index} at R={radius:.8f}, Z={height:.8f}"
        else:
            name = f"wall entry {index - physical_count}"
        terms.append(
            {
                "name": name,
                "live_minus_state": float(live_term[index]),
                "frozen_minus_state": float(frozen_term[index]),
                "live_minus_frozen": float(mismatch[index]),
            }
        )
    return {
        "grid": float(np.max(np.abs(live_term[:physical_count]))),
        "wall": float(np.max(np.abs(live_term[physical_count:]))),
        "terms": terms,
    }


def _markdown(receipt: dict[str, Any]) -> str:
    partition = receipt["partition_fields"]
    largest = receipt["residual_terms"]["terms"]
    supported = receipt["refreshed_live_residual"] < receipt["live_residual"]
    lines = [
        "# Frozen Newton fixed-point RCA",
        "",
        "## Terminal requalification",
        "",
        f"The route recorded {receipt['frozen_partition_reads']} partition reads and "
        f"{receipt['frozen_partition_refreezes']} re-freezes. The terminal read ran; "
        f"its label/support partition difference was "
        f"{receipt['partition_difference_cells']} cells.",
        "",
        "## Residual at the frozen terminal state",
        "",
        f"The frozen map residual is {receipt['frozen_residual']:.12e}; the live-map "
        f"residual is {receipt['live_residual']:.12e}. The live residual is carried "
        f"by grid entries up to {receipt['residual_terms']['grid']:.12e} and wall "
        f"entries up to {receipt['residual_terms']['wall']:.12e}.",
        "",
        "| entry | live map − state | frozen map − state | live − frozen |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        "| {name} | {live_minus_state:.12e} | {frozen_minus_state:.12e} | "
        "{live_minus_frozen:.12e} |".format(**entry)
        for entry in largest
    )
    lines.extend(
        [
            "",
            "## What remained frozen",
            "",
            "The terminal label comparison is not a complete map comparison. The "
            "frozen partition also retains topology coordinates and fluxes, residual "
            "domain masking, and clipped support geometry/moments. There is no "
            "net-current normalisation scalar in this absolute-current fixture.",
            "",
            "| retained quantity | changed entries | maximum absolute "
            "warm-to-terminal difference |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, value in partition.items():
        lines.append(
            f"| {name} | {value['changed_entries']} | "
            f"{value['maximum_absolute_difference']:.12e} |"
        )
    lines.extend(
        [
            "",
            "## Falsifiable mechanism check",
            "",
            f"Refreshing the partition at the terminal state and taking "
            f"{receipt['refresh_newton_steps']} more Newton steps changed the live "
            "residual "
            f"from {receipt['live_residual']:.12e} to "
            f"{receipt['refreshed_live_residual']:.12e}. The prediction that a "
            "refreshed continuous partition, even when labels do not change, moves "
            "the frozen fixed point is therefore "
            f"{'supported' if supported else 'not supported'}.",
            "",
            "## Recommended route",
            "",
            "Keep live reads during warm-up and one frozen partition per Newton "
            "pass, but make the terminal reconciliation refresh the complete "
            "partition and continue a bounded Newton pass whenever its live "
            "residual exceeds tolerance. This retains the one-read-per-pass cost "
            "while targeting the measured refreshed residual above.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def measure(refresh_newton_steps: int) -> dict[str, Any]:
    """Run the frozen route, inspect both maps, and refresh the terminal partition."""
    configure_dtypes()
    assert jax.config.jax_enable_x64 is True
    profile, seed, _vacuum = _fixture_machine()
    operator = profile.operator
    external = operator.external()
    started = time.monotonic()
    result = profile.solve(seed, route="newton_krylov", newton_steps=8, warmup=40)
    terminal = result.flux
    jax.block_until_ready(terminal)

    shadowed_map = operator.traced_flux_map_with_shadow()
    live_map = operator.traced_flux_map()
    warmed = fixed_point.picard(
        live_map,
        seed,
        evaluations=40,
        relaxation=profile.relaxation,
        map_arguments=(external, operator, None),
    ).state
    jax.block_until_ready(warmed)
    warmed_partition = _partition_read(shadowed_map, warmed, None, external, operator)
    warm_shadow = shadowed_map._frozen_partition_shadow(warmed_partition)
    terminal_partition = _partition_read(
        shadowed_map, terminal, warm_shadow, external, operator
    )
    frozen = _partition_map(
        shadowed_map, terminal, warmed_partition, external, operator
    )
    live = live_map(terminal, external, operator, None)
    jax.block_until_ready((frozen, live))

    refreshed = profile.solve(
        terminal,
        route="newton_krylov",
        newton_steps=refresh_newton_steps,
        warmup=0,
    )
    refreshed_live = live_map(refreshed.flux, external, operator, None)
    jax.block_until_ready(refreshed_live)
    partition_difference = int(
        np.count_nonzero(
            np.asarray(warmed_partition.label) != np.asarray(terminal_partition.label)
        )
    )
    return {
        "frozen_partition_reads": int(result.fixed_point.frozen_partition_reads),
        "frozen_partition_refreezes": int(
            result.fixed_point.frozen_partition_refreezes
        ),
        "partition_difference_cells": partition_difference,
        "trajectory_residual": _as_float(result.fixed_point.trajectory_residual),
        "frozen_residual": _relative(frozen, terminal),
        "live_residual": _relative(live, terminal),
        "refreshed_live_residual": _relative(refreshed_live, refreshed.flux),
        "refresh_newton_steps": refresh_newton_steps,
        "partition_fields": _partition_fields(warmed_partition, terminal_partition),
        "residual_terms": _residual_terms(
            live,
            frozen,
            terminal,
            operator.grid.coordinate,
            operator.physical_node_number,
        ),
        "wall_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-newton-steps", type=int, default=8)
    arguments = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    receipt = measure(arguments.refresh_newton_steps)
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(_markdown(receipt))
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
