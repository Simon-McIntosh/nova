"""Measure and attribute exact-clip solve memory across cell-count rungs.

The production solve is compiled without execution.  Each rung is persisted as
soon as its executable and optimized HLO census are available, so a later
compiler failure does not erase an earlier measurement.  Predicate signatures
whose shape carries the realised cell count more than once are reported
separately: those are the all-cell pairwise constructions that cannot satisfy
the linear-memory contract of a per-cell topology or clipping operation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp

from benchmarks import solovev_certificate as certificate
from nova.equilibrium import clip_quadrature
from nova.equilibrium.forward_operator import set_support_clip_mode, support_clip_mode
from nova.jax.config import configure_dtypes


SCHEMA = "nova.exact-clip-memory-scaling"


def scaling_exponent(
    first_bytes: int,
    second_bytes: int,
    first_cells: int,
    second_cells: int,
) -> float:
    """Return the power-law exponent between two positive measurements."""
    values = (first_bytes, second_bytes, first_cells, second_cells)
    if any(value <= 0 for value in values):
        raise ValueError("scaling measurements must be positive")
    return math.log(second_bytes / first_bytes) / math.log(second_cells / first_cells)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace one receipt only after its complete JSON is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _cell_axes(shape: list[int], realised_cells: int) -> list[int]:
    """Return axes whose dimension is the rung's realised cell count."""
    return [axis for axis, size in enumerate(shape) if size == realised_cells]


def _pairwise_predicates(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return predicate results carrying two independent all-cell axes."""
    realised = int(row["realised_cells"])
    candidates = []
    for signature in row.get("qualifying_array_signatures", []):
        if signature["dtype"] != "pred":
            continue
        axes = _cell_axes(signature["shape"], realised)
        if len(axes) < 2:
            continue
        candidates.append(signature | {"realised_cell_axes": axes})
    return sorted(
        candidates,
        key=lambda item: (
            item["logical_size_in_bytes"],
            item["instruction_count"],
        ),
        reverse=True,
    )


def _scaling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit temporary and pairwise-predicate growth over completed rungs."""
    ordered = sorted(rows, key=lambda row: row["realised_cells"])
    adjacent = []
    for first, second in zip(ordered, ordered[1:], strict=False):
        adjacent.append(
            {
                "first_requested_cells": abs(first["requested_cells"]),
                "second_requested_cells": abs(second["requested_cells"]),
                "temporary_bytes_exponent": scaling_exponent(
                    first["memory_analysis"]["temp_size_in_bytes"],
                    second["memory_analysis"]["temp_size_in_bytes"],
                    first["realised_cells"],
                    second["realised_cells"],
                ),
            }
        )
    return {
        "independent_variable": "realised atomic cell count",
        "adjacent_rungs": adjacent,
    }


def _compile_memory_only(
    case_name: str,
    requested_cells: int,
    *,
    arm: str,
) -> dict[str, Any]:
    """Compile one solve and read memory without serializing multi-GiB HLO.

    Accelerator programs at the failing resolution can exceed protobuf's
    two-GiB serialization limit even though the compiled executable exposes a
    valid memory analysis. Keeping that receipt independent from the optional
    text census prevents the diagnostic export from erasing the gate it was
    meant to measure.
    """
    started = perf_counter()
    profile, seed, request, dimensions = certificate._certificate_compile_problem(
        case_name, requested_cells
    )
    mapped = profile.flux_map(
        request.current,
        target_current=request.target_current,
        prescribed_current=request.prescribed_current,
    )
    shadowed_map = profile.operator.flux_map_with_shadow(
        request.current,
        target_current=request.target_current,
        prescribed_current=request.prescribed_current,
    )

    def shadow_mask(state):
        return profile.operator.residual_shadow_mask(state)

    def promoted_shadow_mask(state, previous):
        return profile.operator.residual_shadow_mask(state, previous_shadow=previous)

    def solve_program(initial):
        result = certificate.recovery.fixed_point.newton_krylov(
            mapped,
            initial,
            shadow_mask_fn=shadow_mask,
            promoted_shadow_mask_fn=promoted_shadow_mask,
            shadowed_map_fn=shadowed_map,
            **request.policy.kernel_options(),
        )
        return result.state, result.residual, result.converged

    compiled = (
        jax.jit(solve_program).lower(jnp.asarray(seed, dtype=jnp.float64)).compile()
    )
    analysis = certificate._compiled_memory_fields(compiled.memory_analysis())
    return {
        "case": case_name,
        "arm": arm,
        "requested_cells": requested_cells,
        "realised_cells": dimensions["realised_cells"],
        "dimensions": dimensions,
        "compile_wall_seconds": perf_counter() - started,
        "memory_analysis": analysis,
        "largest_array_intermediates": [],
        "largest_predicate_intermediates": [],
        "qualifying_array_signatures": [],
        "executed": False,
        "method": (
            "jax.jit(solve_program).lower(seed).compile().memory_analysis(); "
            "optimized HLO serialization deliberately skipped"
        ),
    }


def measure(
    output: Path,
    compiler_root: Path,
    requested_cells: list[int],
    *,
    part_root: Path | None = None,
    capture_hlo: bool = True,
) -> dict:
    """Compile exact-clip rungs and persist each result before continuing.

    Every rung receives an immutable, independently readable part receipt in
    addition to the cumulative receipt.  A later compilation failure therefore
    cannot obscure which earlier executable and memory analysis actually
    landed.
    """
    configure_dtypes()
    if not hasattr(certificate.observation, "_UNIT_NODE"):
        certificate.observation._UNIT_NODE = clip_quadrature._UNIT_NODE
    original_mode = support_clip_mode()
    rows: list[dict[str, Any]] = []
    source_revision = certificate._source_revision()
    lane = certificate._lane()
    if part_root is None:
        part_root = output.parent / f"{output.stem}-parts"
    try:
        set_support_clip_mode("exact")
        for requested in requested_cells:
            if capture_hlo:
                row = certificate._compile_solve_memory(
                    "weak-rotation-reactor-static",
                    -abs(requested),
                    arm=f"exact-{abs(requested)}",
                    compiler_artifact_root=compiler_root,
                )
            else:
                row = _compile_memory_only(
                    "weak-rotation-reactor-static",
                    -abs(requested),
                    arm=f"exact-{abs(requested)}",
                )
            row["pairwise_predicate_candidates"] = _pairwise_predicates(row)
            _atomic_json(
                part_root / f"requested-{abs(requested)}.json",
                {
                    "schema": SCHEMA,
                    "source_revision": source_revision,
                    "lane": lane,
                    "row": row,
                },
            )
            rows.append(row)
            receipt = {
                "schema": SCHEMA,
                "source_revision": source_revision,
                "lane": lane,
                "rows": rows,
                "scaling": _scaling(rows),
                "completed": len(rows) == len(requested_cells),
            }
            _atomic_json(output, receipt)
            temporary_gib = row["memory_analysis"]["temp_size_in_bytes"] / 2**30
            print(
                "EXACT_CLIP_MEMORY_RUNG "
                f"requested={abs(requested)} realised={row['realised_cells']} "
                f"temporary_gib={temporary_gib:.6f} "
                f"pairwise_predicates={len(row['pairwise_predicate_candidates'])}",
                flush=True,
            )
    finally:
        set_support_clip_mode(original_mode)
    return json.loads(output.read_text(encoding="utf-8"))


def solve_and_measure(
    output: Path,
    figure_root: Path,
    part_root: Path,
    requested_cells: int,
) -> dict[str, Any]:
    """Run one exact-clip production solve and persist its allocation peak.

    The certificate driver owns the production ``profile.solve`` call and its
    line-contour panel.  This wrapper only redirects those durable outputs and
    reads the accelerator allocator after the terminal state is ready.  A
    positive byte counter proves the allocator instrument saw the live solve;
    an empty or uniformly zero report is rejected rather than presented as a
    low-memory measurement.
    """
    configure_dtypes()
    if not hasattr(certificate.observation, "_UNIT_NODE"):
        certificate.observation._UNIT_NODE = clip_quadrature._UNIT_NODE
    device = jax.devices()[0]
    original_mode = support_clip_mode()
    original_figure_root = certificate.FIGURE_ROOT
    original_part_root = certificate.PART_ROOT
    certificate.FIGURE_ROOT = figure_root
    certificate.PART_ROOT = part_root
    requested_cells = -abs(requested_cells)
    try:
        set_support_clip_mode("exact")
        certificate._measure(
            "weak-rotation-reactor-static",
            requested_cells,
        )
        row_path = certificate._part_path(
            "weak-rotation-reactor-static", requested_cells
        )
        figure_path = certificate._figure_path(
            "weak-rotation-reactor-static", requested_cells
        )
        row = json.loads(row_path.read_text(encoding="utf-8"))
        raw_stats = device.memory_stats() or {}
    finally:
        certificate.FIGURE_ROOT = original_figure_root
        certificate.PART_ROOT = original_part_root
        set_support_clip_mode(original_mode)

    statistics = {
        str(name): int(value) if isinstance(value, int | float) else str(value)
        for name, value in raw_stats.items()
    }
    byte_counters = {
        name: value
        for name, value in statistics.items()
        if "bytes" in name and isinstance(value, int)
    }
    observed_bytes = max(byte_counters.values(), default=0)
    if observed_bytes <= 0:
        raise RuntimeError("accelerator allocation counters did not see the solve")
    peak_bytes = statistics.get("peak_bytes_in_use")
    if not isinstance(peak_bytes, int) or peak_bytes <= 0:
        raise RuntimeError("accelerator allocator did not report a positive peak")

    receipt = {
        "schema": "nova.exact-clip-production-solve",
        "source_revision": certificate._source_revision(),
        "lane": certificate._lane(),
        "requested_cells": requested_cells,
        "row_receipt": str(row_path),
        "figure": {
            "filesystem_path": str(figure_path),
            "project_absolute_src": row["figure"]["project_absolute_src"],
            "sha256": row["figure"]["sha256"],
        },
        "allocator": {
            "device": str(device),
            "statistics": statistics,
            "instrument_check": {
                "byte_counter_count": len(byte_counters),
                "largest_observed_byte_counter": observed_bytes,
                "positive_peak_bytes_in_use": True,
            },
            "peak_bytes_in_use": peak_bytes,
            "peak_gib": peak_bytes / 2**30,
        },
        "row": row,
    }
    _atomic_json(output, receipt)
    print(
        "EXACT_CLIP_SOLVE "
        f"requested={abs(requested_cells)} realised={row['realised_cells']} "
        f"peak_gib={peak_bytes / 2**30:.6f} "
        f"residual={row['solver']['terminal_fixed_point_residual']}",
        flush=True,
    )
    return receipt


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-root", type=Path)
    parser.add_argument("--part-root", type=Path)
    parser.add_argument("--figure-root", type=Path)
    parser.add_argument("--execute-solve", action="store_true")
    parser.add_argument("--skip-hlo", action="store_true")
    parser.add_argument("--cells", type=int, nargs="+", default=[110, 300, 500])
    return parser.parse_args()


def main() -> None:
    arguments = _parse()
    if arguments.execute_solve:
        if len(arguments.cells) != 1:
            raise ValueError("the production solve accepts exactly one cell count")
        solve_and_measure(
            arguments.output,
            arguments.figure_root or arguments.output.parent / "solve-panels",
            arguments.part_root or arguments.output.parent / "solve-parts",
            arguments.cells[0],
        )
        return
    if arguments.compiler_root is None:
        raise ValueError("--compiler-root is required for memory analysis")
    receipt = measure(
        arguments.output,
        arguments.compiler_root,
        arguments.cells,
        part_root=arguments.part_root,
        capture_hlo=not arguments.skip_hlo,
    )
    print(
        json.dumps(
            {
                "completed": receipt["completed"],
                "rungs": len(receipt["rows"]),
                "scaling": receipt["scaling"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
