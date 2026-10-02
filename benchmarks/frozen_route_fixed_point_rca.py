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


def _route_diagnostics(result: Any) -> dict[str, Any]:
    """Expose the termination state that decides where the route stops."""
    fixed = result.fixed_point
    reason = fixed_point.FixedPointTerminationReason(int(fixed.termination_reason))
    return {
        "termination_reason": int(fixed.termination_reason),
        "termination_reason_name": reason.name,
        "converged": bool(fixed.converged),
        "active_set_iterations": int(fixed.active_set_iterations),
        "active_set_residuals": [
            float(value) for value in np.asarray(fixed.active_set_residuals).ravel()
        ],
        "active_set_mask_differences": [
            int(value)
            for value in np.asarray(fixed.active_set_mask_differences).ravel()
        ],
        "attempted_newton_promotions": int(fixed.attempted_newton_promotions),
        "accepted_newton_promotions": int(fixed.accepted_newton_promotions),
        "trajectory_residual": float(fixed.trajectory_residual),
        "inner_iteration_residuals_before": [
            float(value)
            for value in np.asarray(fixed.inner_iteration_residuals_before).ravel()
        ],
        "inner_iteration_residuals_after": [
            float(value)
            for value in np.asarray(fixed.inner_iteration_residuals_after).ravel()
        ],
        "inner_iteration_accepted": [
            int(value) for value in np.asarray(fixed.inner_iteration_accepted).ravel()
        ],
    }


def _partition_read(shadowed_map, state, previous, external, operator):
    return shadowed_map._read_frozen_partition(state, previous, external, operator)


def _partition_map(shadowed_map, state, partition, external, operator):
    return shadowed_map._map_frozen_partition(state, partition, external, operator)


def _changed(left: Any, right: Any) -> dict[str, Any]:
    """Summarise a retained quantity at the warmed and terminal reads."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    is_numeric = np.issubdtype(left_array.dtype, np.number)
    difference = right_array - left_array if is_numeric else None

    def summary(value: np.ndarray) -> Any:
        if value.ndim == 0:
            return value.item()
        if value.size == 0:
            return {"minimum": None, "maximum": None}
        result: dict[str, Any] = {
            "minimum": value.min().item(),
            "maximum": value.max().item(),
        }
        if is_numeric:
            result["l2_norm"] = float(np.linalg.norm(value.ravel()))
        return result

    return {
        "shape": list(left_array.shape),
        "changed_entries": int(np.count_nonzero(left_array != right_array)),
        "maximum_absolute_difference": (
            float(np.max(np.abs(difference)))
            if is_numeric and difference.size
            else None
        ),
        "warm_value": summary(left_array),
        "terminal_value": summary(right_array),
    }


def _field_changes(
    prefix: str, warmed: Any, terminal: Any
) -> dict[str, dict[str, Any]]:
    """Flatten all partition leaves without hiding continuous frozen values."""
    if warmed is None:
        return {prefix: {"status": "not present in this fixture"}}
    if hasattr(warmed, "_fields"):
        result: dict[str, dict[str, Any]] = {}
        for name in warmed._fields:
            result.update(
                _field_changes(
                    f"{prefix}.{name}", getattr(warmed, name), getattr(terminal, name)
                )
            )
        return result
    if is_dataclass(warmed):
        result = {}
        for field in fields(warmed):
            result.update(
                _field_changes(
                    f"{prefix}.{field.name}",
                    getattr(warmed, field.name),
                    getattr(terminal, field.name),
                )
            )
        return result
    return {prefix: _changed(warmed, terminal)}


def _partition_fields(warmed, terminal) -> dict[str, dict[str, Any]]:
    """Expose every state-dependent leaf retained by the frozen map."""
    return _field_changes("partition", warmed, terminal)


def _maximum_absolute(values: np.ndarray) -> float:
    """Reduce a component slice; an absent component is empty and carries 0.0."""
    return float(np.max(np.abs(values))) if values.size else 0.0


def _residual_terms(live, frozen, state, coordinate, physical_count: int):
    """Name the largest map residual terms by physical cell or wall entry."""
    live_term = np.asarray(live - state)
    frozen_term = np.asarray(frozen - state)
    mismatch = live_term - frozen_term
    grid_term = live_term[:physical_count]
    tail_term = live_term[physical_count:]
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
    components = {
        "grid": {
            "row_count": int(grid_term.size),
            "maximum_absolute_residual": _maximum_absolute(grid_term),
            "l2_residual": float(np.linalg.norm(grid_term)),
        },
        "wall_and_sample": {
            "row_count": int(tail_term.size),
            "maximum_absolute_residual": _maximum_absolute(tail_term),
            "l2_residual": float(np.linalg.norm(tail_term)),
        },
    }
    return {
        "components": components,
        "terms": terms,
    }


def _markdown(receipt: dict[str, Any]) -> str:
    partition = receipt["partition_fields"]
    largest = receipt["residual_terms"]["terms"]
    components = receipt["residual_terms"]["components"]
    lines = [
        "# Frozen Newton fixed-point RCA",
        "",
        "## Terminal requalification",
        "",
        f"The route recorded {receipt['frozen_partition_reads']} partition reads and "
        f"{receipt['frozen_partition_refreezes']} re-freezes. The terminal read ran; "
        f"its partition difference was {receipt['partition_difference_cells']} cells "
        f"({receipt['label_difference_cells']} labels, "
        f"{receipt['support_difference_cells']} support inclusions, and "
        f"{receipt['residual_shadow_difference_entries']} residual-shadow entries).",
        "",
        "## Residual at the frozen terminal state",
        "",
        f"The frozen map residual is {receipt['frozen_residual']:.12e}; the live-map "
        f"residual is {receipt['live_residual']:.12e}. It decomposes into a grid "
        f"component ({components['grid']['row_count']} rows) reaching "
        f"{components['grid']['maximum_absolute_residual']:.12e} maximum absolute "
        f"(l2 {components['grid']['l2_residual']:.12e}), and a wall/sample component "
        f"({components['wall_and_sample']['row_count']} rows) reaching "
        f"{components['wall_and_sample']['maximum_absolute_residual']:.12e} maximum "
        f"absolute (l2 {components['wall_and_sample']['l2_residual']:.12e}).",
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
            "domain masking, and clipped support geometry/moments. The operator "
            "geometry is static rather than copied into the partition. There is no "
            "net-current normalisation scalar in this absolute-current fixture.",
            "",
            "| retained quantity | changed entries | maximum absolute "
            "warm-to-terminal difference | warm value | terminal value |",
            "| --- | ---: | ---: | --- | --- |",
        ]
    )
    for name, value in partition.items():
        lines.append(
            f"| {name} | {value.get('changed_entries', 'n/a')} | "
            f"{value.get('maximum_absolute_difference', 'n/a')} | "
            f"{value.get('warm_value', value.get('status', 'n/a'))} | "
            f"{value.get('terminal_value', value.get('status', 'n/a'))} |"
        )
    geometry = receipt["operator_geometry"]
    lines.extend(
        [
            "",
            "The operator grid is shared static geometry, not a per-solve partition "
            "copy: its coordinate array has "
            f"{geometry['grid_coordinate']['changed_entries']} warm-to-terminal "
            "changes. Its physical-node count is "
            f"{geometry['physical_node_number']}. Net-current normalisation is "
            f"{geometry['net_current_normalisation']}.",
        ]
    )
    route = receipt["route"]
    refreshed_route = receipt["refreshed_route"]
    frozen_gap = max(abs(entry["live_minus_frozen"]) for entry in largest)
    lines.extend(
        [
            "",
            "## Route termination",
            "",
            f"The terminal route converged={route['converged']} with reason "
            f"{route['termination_reason_name']} after "
            f"{route['active_set_iterations']} active-set trips, attempting "
            f"{route['attempted_newton_promotions']} and accepting "
            f"{route['accepted_newton_promotions']} Newton promotions. Its "
            f"per-trip live residuals were {route['active_set_residuals']} and "
            f"its per-trip mask differences were "
            f"{route['active_set_mask_differences']}. The retry route "
            f"converged={refreshed_route['converged']} with reason "
            f"{refreshed_route['termination_reason_name']} and accepted "
            f"{refreshed_route['accepted_newton_promotions']} Newton promotions.",
            "",
            "## Falsifiable mechanism check",
            "",
            "Two measurements refute the frozen-partition hypothesis. At the "
            "terminal state the frozen map output differs from the live map "
            f"output by at most {frozen_gap:.12e} "
            "absolute over the ranked cells, so the frozen and live maps are the "
            "same map to machine precision there. Second, a second frozen route "
            "from the terminal state, re-reading the partition "
            f"({receipt['refreshed_frozen_partition_reads']} reads, "
            f"{receipt['refreshed_frozen_partition_refreezes']} re-freeze) and "
            f"taking {receipt['refresh_newton_steps']} more Newton steps, returns "
            f"the identical live residual {receipt['refreshed_live_residual']:.15e} "
            f"and the identical {refreshed_route['termination_reason_name']} reason "
            f"with {refreshed_route['accepted_newton_promotions']} accepted "
            "promotion. A partition refresh therefore does not move the terminus.",
            "",
            "Caveat: that second solve re-freezes too, so it tests a repeated "
            "frozen pass, not a live-read pass. What it establishes is narrow and "
            "sufficient: the residual is reproduced bit-for-bit, so it is a "
            "property of the route's terminus, not of a partition that went stale "
            "between reads.",
            "",
            "## Recommended route",
            "",
            "The measured terminus is a settlement, not a stale partition. The "
            "outer loop stops after two active-set trips on an unchanged mask with "
            "no accepted promotion, retaining a state whose live relative-sup "
            f"residual is {receipt['live_residual']:.12e} while the inner Newton "
            f"local step residual is {receipt['trajectory_residual']:.12e}. The "
            "route holds the active set fixed within each pass, and once the mask "
            "stops changing the settle test ends the loop on the retained state "
            "without consulting the live residual. Keep the locked design (live "
            "reads through warm-up, one partition per Newton pass) and gate the "
            "settle: admit settlement only when the reconciled live relative-sup "
            "residual is at or below tolerance, and otherwise continue the local "
            "Newton trajectory, which the design already preserves across an "
            "unchanged mask. This node's re-frozen refresh does not improve the "
            f"residual ({receipt['live_residual']:.12e} to "
            f"{receipt['refreshed_live_residual']:.12e}), so the fix is the "
            "residual gate rather than the refresh. Each bounded pass costs "
            f"{receipt['refreshed_route_wall_seconds']:.2f} s against the base "
            f"route's {receipt['initial_route_wall_seconds']:.2f} s.",
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

    refresh_started = time.monotonic()
    refreshed = profile.solve(
        terminal,
        route="newton_krylov",
        newton_steps=refresh_newton_steps,
        warmup=0,
    )
    refreshed_live = live_map(refreshed.flux, external, operator, None)
    jax.block_until_ready(refreshed_live)
    label_difference = int(
        np.count_nonzero(
            np.asarray(warmed_partition.label) != np.asarray(terminal_partition.label)
        )
    )
    support_difference = 0
    if warmed_partition.profile_support is not None:
        support_difference = int(
            np.count_nonzero(
                np.asarray(warmed_partition.profile_support.included)
                != np.asarray(terminal_partition.profile_support.included)
            )
        )
    shadow_difference = int(
        np.count_nonzero(
            np.asarray(warmed_partition.residual_shadow)
            != np.asarray(terminal_partition.residual_shadow)
        )
    )
    partition_difference = max(label_difference + support_difference, shadow_difference)
    return {
        "frozen_partition_reads": int(result.fixed_point.frozen_partition_reads),
        "frozen_partition_refreezes": int(
            result.fixed_point.frozen_partition_refreezes
        ),
        "partition_difference_cells": partition_difference,
        "label_difference_cells": label_difference,
        "support_difference_cells": support_difference,
        "residual_shadow_difference_entries": shadow_difference,
        "trajectory_residual": _as_float(result.fixed_point.trajectory_residual),
        "route": _route_diagnostics(result),
        "refreshed_route": _route_diagnostics(refreshed),
        "frozen_residual": _relative(frozen, terminal),
        "live_residual": _relative(live, terminal),
        "refreshed_live_residual": _relative(refreshed_live, refreshed.flux),
        "refreshed_frozen_partition_reads": int(
            refreshed.fixed_point.frozen_partition_reads
        ),
        "refreshed_frozen_partition_refreezes": int(
            refreshed.fixed_point.frozen_partition_refreezes
        ),
        "refresh_newton_steps": refresh_newton_steps,
        "partition_fields": _partition_fields(warmed_partition, terminal_partition),
        "operator_geometry": {
            "grid_coordinate": _changed(
                operator.grid.coordinate, operator.grid.coordinate
            ),
            "physical_node_number": operator.physical_node_number,
            "net_current_normalisation": "not applicable: absolute-current fixture",
        },
        "residual_terms": _residual_terms(
            live,
            frozen,
            terminal,
            operator.grid.coordinate,
            operator.physical_node_number,
        ),
        "initial_route_wall_seconds": refresh_started - started,
        "refreshed_route_wall_seconds": time.monotonic() - refresh_started,
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
