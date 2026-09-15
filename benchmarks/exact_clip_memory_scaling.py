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
from typing import Any

from benchmarks import solovev_certificate as certificate
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


def measure(output: Path, compiler_root: Path, requested_cells: list[int]) -> dict:
    """Compile exact-clip rungs and persist each completed memory receipt."""
    configure_dtypes()
    original_mode = support_clip_mode()
    rows: list[dict[str, Any]] = []
    try:
        set_support_clip_mode("exact")
        for requested in requested_cells:
            row = certificate._compile_solve_memory(
                "weak-rotation-reactor-static",
                -abs(requested),
                arm=f"exact-{abs(requested)}",
                compiler_artifact_root=compiler_root,
            )
            row["pairwise_predicate_candidates"] = _pairwise_predicates(row)
            rows.append(row)
            receipt = {
                "schema": SCHEMA,
                "source_revision": certificate._source_revision(),
                "lane": certificate._lane(),
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


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler-root", type=Path, required=True)
    parser.add_argument("--cells", type=int, nargs="+", default=[110, 300, 500])
    return parser.parse_args()


def main() -> None:
    arguments = _parse()
    receipt = measure(arguments.output, arguments.compiler_root, arguments.cells)
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
