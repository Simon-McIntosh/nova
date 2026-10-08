"""Compare the emitted MAST description with Nova's packaged registry."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import shapely

from nova.catalog.mast_geometry import (
    MachineGeometryRegistry,
    active_component_geometry,
)
from nova.imas.mast_flux_loop_adjudication import join_accounting
from nova.imas.mast_solve_inputs import reconstruction_loop_positions

SHOT = 11766
GEOMETRY_BOUND_M = 1e-5  # Registry outlines are snapped to a 10 µm grid.
POSITION_BOUND_M = 1e-5
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/machine-description-retirement/mdr-equivalence.json"
)
DEFAULT_PRODUCER_ROOT = Path.home() / "Code/imas-ambix"

# This process runs with imas-ambix's interpreter and package tree. It emits only
# the description values used by the comparison, without reading Nova's registry.
_PRODUCER_CODE = """
import json
from imas_alambic.machine_map import load_packaged_machine_map
from imas_alambic.transform_engine import transform_machine_description
from imas_ambix.data.geometry_adapter import geometry_table_from_description
from imas_ambix.data.paths import LEVEL2_DIR
catalog = load_packaged_machine_map("mast")
description = transform_machine_description(catalog, 11766, "zarr", LEVEL2_DIR)
if description.status != "emitted":
    raise RuntimeError(f"producer status: {description.status}")
table = geometry_table_from_description(description, catalog)
arrays = {
    item.binding_name: item.values.tolist()
    for item in description.arrays
    if item.dd_path.startswith("pf_active/coil/element/geometry/")
}
print("DESCRIPTION_JSON=" + json.dumps({
    "status": description.status,
    "transition": description.machine_map.transition,
    "loop_positions_m": [[float(row.r), float(row.z)] for row in table.flux_loops],
    "active_arrays": arrays,
}, separators=(",", ":")))
"""


def producer_snapshot(producer_root: Path = DEFAULT_PRODUCER_ROOT) -> dict:
    """Run the actual map transform in its owner's environment without syncing."""
    interpreter = producer_root / ".venv/bin/python"
    if not interpreter.is_file():
        raise FileNotFoundError(f"producer interpreter missing: {interpreter}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(producer_root)
    env["PYTHONNOUSERSITE"] = "1"
    result = subprocess.run(
        [str(interpreter), "-c", _PRODUCER_CODE],
        cwd=producer_root,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=180,
    )
    marker = "DESCRIPTION_JSON="
    lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(f"producer returned {len(lines)} description records")
    return json.loads(lines[0][len(marker) :])


def _active_outlines(arrays: dict[str, list]) -> dict[str, str]:
    families: dict[str, dict[str, list]] = {}
    pattern = re.compile(r"^mast-pf-active-(.+)-(r|z|width|height)$")
    for binding, values in arrays.items():
        match = pattern.fullmatch(binding)
        if match:
            families.setdefault(match[1].replace("-", "_"), {})[match[2]] = values
    outlines = {}
    for name, fields in families.items():
        if set(fields) != {"r", "z", "width", "height"}:
            raise ValueError(f"incomplete active conductor geometry: {name}")
        outlines[name] = active_component_geometry(
            *(
                np.asarray(fields[key], dtype=float)
                for key in ("r", "z", "width", "height")
            )
        )
    if not outlines:
        raise ValueError("producer emitted no active conductor geometry")
    return outlines


def _loop_geometry(positions: list[list[float]], outlines: dict[str, str]) -> dict:
    return {
        "magnetics": {"flux_loops": [[r, z, 2 * np.pi] for r, z in positions]},
        "active_components": outlines,
    }


def _unique_positions(positions: list[list[float]]) -> list[tuple[float, float]]:
    return sorted({tuple(float(x) for x in np.round(row, 5)) for row in positions})


def _coil_differences(reference: dict[str, str], produced: dict[str, str]) -> dict:
    rows = {}
    for name in sorted(set(reference) | set(produced)):
        if name not in reference or name not in produced:
            rows[name] = {
                "missing_from": "registry" if name not in reference else "producer"
            }
            continue
        left = shapely.from_wkb(bytes.fromhex(reference[name]))
        right = shapely.from_wkb(bytes.fromhex(produced[name]))
        left_vertices = np.asarray(left.exterior.coords)
        right_vertices = np.asarray(right.exterior.coords)
        rows[name] = {
            "vertex_coordinate_abs_m": (
                float(np.max(np.abs(left_vertices - right_vertices)))
                if left_vertices.shape == right_vertices.shape
                else None
            ),
            "vertex_hausdorff_m": float(left.hausdorff_distance(right)),
            "symmetric_area_m2": float(left.symmetric_difference(right).area),
        }
    return rows


def compare_descriptions(
    producer: dict, registry_geometry: dict, reconstruction: np.ndarray
) -> dict:
    """Score counts, the one-to-one solve join, and each named coil outline."""
    produced_outlines = _active_outlines(producer["active_arrays"])
    registry_positions = [
        row[:2] for row in registry_geometry["magnetics"]["flux_loops"]
    ]
    produced_positions = producer["loop_positions_m"]
    registry_unique = _unique_positions(registry_positions)
    produced_unique = _unique_positions(produced_positions)
    registry_joins = join_accounting(registry_geometry, reconstruction)
    produced_joins = join_accounting(
        _loop_geometry(produced_positions, produced_outlines), reconstruction
    )
    coils = _coil_differences(registry_geometry["active_components"], produced_outlines)
    numeric_deltas = [
        row["vertex_coordinate_abs_m"]
        for row in coils.values()
        if row.get("vertex_coordinate_abs_m") is not None
    ]
    maximum = max(numeric_deltas) if numeric_deltas else None
    registry_set = set(registry_unique)
    produced_set = set(produced_unique)
    missing = sorted(registry_set - produced_set)
    additional = sorted(produced_set - registry_set)
    registry_joined = {row.channel for row in registry_joins if row.served}
    produced_joined = {row.channel for row in produced_joins if row.served}
    verdict = (
        len(registry_unique) == len(produced_unique) == 44
        and len(registry_joins) == len(produced_joins) == 46
        and sum(row.served for row in registry_joins)
        == sum(row.served for row in produced_joins)
        == 43
        and registry_joined == produced_joined
        and not missing
        and not additional
        and set(coils) == set(registry_geometry["active_components"])
        and len(numeric_deltas) == len(coils)
        and all(row["symmetric_area_m2"] == 0 for row in coils.values())
        and maximum is not None
        and maximum <= GEOMETRY_BOUND_M
    )
    return {
        "shot": SHOT,
        "bounds": {
            "coil_vertex_coordinate_abs_m": GEOMETRY_BOUND_M,
            "position_m": POSITION_BOUND_M,
        },
        "registry": {
            "loop_entries": len(registry_positions),
            "unique_loop_positions": len(registry_unique),
            "joined_reconstruction_loops": sum(row.served for row in registry_joins),
            "reconstruction_loops": len(registry_joins),
        },
        "producer": {
            "status": producer["status"],
            "transition": producer["transition"],
            "loop_entries": len(produced_positions),
            "unique_loop_positions": len(produced_unique),
            "joined_reconstruction_loops": sum(row.served for row in produced_joins),
            "reconstruction_loops": len(produced_joins),
        },
        "coil_elements": coils,
        "maximum_coil_vertex_coordinate_abs_m": maximum,
        "loop_positions_only_in_registry_m": missing,
        "loop_positions_only_in_producer_m": additional,
        "joined_channels_only_in_registry": sorted(registry_joined - produced_joined),
        "joined_channels_only_in_producer": sorted(produced_joined - registry_joined),
        "registry_loop_positions_m": registry_positions,
        "producer_loop_positions_m": produced_positions,
        "equivalent": verdict,
    }


def measurement(producer_root: Path = DEFAULT_PRODUCER_ROOT) -> dict:
    registry = MachineGeometryRegistry.default()
    selected = registry.select(SHOT)
    reconstruction = reconstruction_loop_positions(SHOT)
    if reconstruction.shape != (46, 2):
        raise ValueError(
            f"expected 46 reconstruction positions, got {reconstruction.shape}"
        )
    receipt = compare_descriptions(
        producer_snapshot(producer_root),
        dict(selected.configuration.geometry),
        reconstruction,
    )
    receipt["sources"] = {
        "registry_digest": registry.registry_digest,
        "physical_digest": selected.configuration.physical_digest,
        "producer_revision": subprocess.check_output(
            ["git", "-C", str(producer_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "producer_environment": str(producer_root / ".venv/bin/python"),
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--producer-root", type=Path, default=DEFAULT_PRODUCER_ROOT)
    args = parser.parse_args()
    receipt = measurement(args.producer_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        "EQUIVALENCE "
        f"registry_unique={receipt['registry']['unique_loop_positions']} "
        f"producer_unique={receipt['producer']['unique_loop_positions']} "
        f"registry_join={receipt['registry']['joined_reconstruction_loops']}/46 "
        f"producer_join={receipt['producer']['joined_reconstruction_loops']}/46 "
        f"max_coil_delta_m={receipt['maximum_coil_vertex_coordinate_abs_m']:.6g} "
        f"equivalent={receipt['equivalent']} receipt={args.output}"
    )
    if not receipt["equivalent"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
