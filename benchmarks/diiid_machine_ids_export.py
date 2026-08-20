"""Write and verify the static DIII-D machine-description netCDF entry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import imas

from nova.imas.diiid_machine_ids import (
    DD_VERSION,
    IDS_NAMES,
    SOURCE_PATH,
    DiiidMachineIds,
    build_diiid_machine_ids,
    machine_ids_snapshot,
    round_trip_leaf_receipt,
)


DEFAULT_OUTPUT = Path(
    "docs/figures/diiid-forward-onboarding/ids-set/diiid_machine_description.nc"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(bundle: DiiidMachineIds, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    database = imas.DBEntry(output.resolve(), "w", dd_version=bundle.dd_version)
    try:
        for name in IDS_NAMES:
            database.put(bundle.ids[name])
    finally:
        database.close()


def _reopen(output: Path) -> tuple[dict[str, Any], dict[str, list[str]]]:
    database = imas.DBEntry(output.resolve(), "r", dd_version=DD_VERSION)
    try:
        occurrences = {
            name: database.list_all_occurrences(name)
            for name in (*IDS_NAMES, "pf_passive", "tf", "equilibrium")
        }
        ids = {name: database.get(name, 0, autoconvert=False) for name in IDS_NAMES}
        filled_paths = {
            name: database.list_filled_paths(name, 0, autoconvert=False)
            for name in IDS_NAMES
        }
    finally:
        database.close()
    if occurrences != {
        "wall": [0],
        "pf_active": [0],
        "magnetics": [0],
        "pf_passive": [],
        "tf": [],
        "equilibrium": [],
    }:
        raise ValueError(f"published IDS occurrence firewall failed: {occurrences}")
    for name, value in ids.items():
        value.validate()
        written = str(value.ids_properties.version_put.data_dictionary)
        if written != DD_VERSION:
            raise ValueError(
                f"reopened {name} carries Data Dictionary {written}, "
                f"expected {DD_VERSION}"
            )
    signal_paths = [
        path
        for path in filled_paths["magnetics"]
        if path.startswith(
            (
                "b_field_pol_probe/field/",
                "flux_loop/flux/",
                "ip/",
                "diamagnetic_flux/",
            )
        )
    ]
    if signal_paths:
        raise ValueError(f"published magnetics signals are forbidden: {signal_paths}")
    return ids, filled_paths


def export_machine_ids(
    output: Path | str = DEFAULT_OUTPUT,
    *,
    source_path: Path | str = SOURCE_PATH,
) -> dict[str, Any]:
    """Write, reopen, compare and describe the DIII-D IDS set."""

    output = Path(output)
    bundle = build_diiid_machine_ids(source_path)
    expected = machine_ids_snapshot(bundle.ids)
    _write(bundle, output)
    reopened, filled_paths = _reopen(output)
    comparison = round_trip_leaf_receipt(expected, machine_ids_snapshot(reopened))
    active = reopened["pf_active"]
    magnetics = reopened["magnetics"]
    wall = reopened["wall"]
    wall_outline = wall.description_2d[0].limiter.unit[0].outline
    element_count = sum(len(coil.element) for coil in active.coil)
    vertex_leaf_differences = [
        leaf["maximum_absolute_difference"]
        for path, leaf in comparison["pf_active"].items()
        if path.endswith(("geometry/outline/r", "geometry/outline/z"))
    ]
    wall_differences = [
        leaf["maximum_absolute_difference"]
        for path, leaf in comparison["wall"].items()
        if path.endswith(("outline/r", "outline/z"))
    ]
    receipt = {
        "measurement": "DIII-D static machine-description IDS netCDF export",
        "source": {
            "entry": str(bundle.source_path),
            "data_dictionary": bundle.dd_version,
            "backend": "imas-python netCDF DBEntry",
        },
        "output": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "sha256": _sha256(output),
        },
        "content": {
            "ids": list(IDS_NAMES),
            "wall_limiter_vertices": len(wall_outline.r),
            "pf_active_coils": len(active.coil),
            "pf_active_elements": element_count,
            "b_field_pol_probe_positions": len(magnetics.b_field_pol_probe),
            "flux_loop_positions": len(magnetics.flux_loop),
            "magnetics_signal_arrays": 0,
            "equilibrium_occurrences": 0,
        },
        "ids_properties": {
            name: {
                "data_dictionary": str(
                    reopened[name].ids_properties.version_put.data_dictionary
                ),
                "source": str(reopened[name].ids_properties.source),
                "comment": str(reopened[name].ids_properties.comment),
                "provenance": [
                    {
                        "path": str(node.path),
                        "sources": [str(value) for value in node.sources],
                    }
                    for node in reopened[name].ids_properties.provenance.node
                ],
            }
            for name in IDS_NAMES
        },
        "declared_absent": [item.as_dict() for item in bundle.absent],
        "round_trip": {
            "verdict": "exact",
            "wall_outline_maximum_absolute_difference": max(wall_differences),
            "element_vertex_maximum_absolute_difference": max(vertex_leaf_differences),
            "leaf_counts": {name: len(leaves) for name, leaves in comparison.items()},
            "ids": comparison,
        },
        "content_firewall": {
            "written_occurrences": list(IDS_NAMES),
            "magnetics_signal_arrays_written": False,
            "equilibrium_written": False,
            "filled_path_counts": {
                name: len(paths) for name, paths in filled_paths.items()
            },
        },
    }
    if receipt["content"] != {
        "ids": ["wall", "pf_active", "magnetics"],
        "wall_limiter_vertices": 82,
        "pf_active_coils": 24,
        "pf_active_elements": 140,
        "b_field_pol_probe_positions": 76,
        "flux_loop_positions": 44,
        "magnetics_signal_arrays": 0,
        "equilibrium_occurrences": 0,
    }:
        raise ValueError(f"unexpected published content: {receipt['content']}")
    if (
        receipt["round_trip"]["wall_outline_maximum_absolute_difference"] != 0.0
        or receipt["round_trip"]["element_vertex_maximum_absolute_difference"] != 0.0
    ):
        raise ValueError("geometry changed during the netCDF round trip")
    return receipt


def write_receipt(receipt: Mapping[str, Any], path: Path | str) -> Path:
    """Write the JSON receipt beside the exported IDS set."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    receipt = export_machine_ids(args.output, source_path=args.source)
    receipt_path = args.receipt or args.output.with_suffix(".receipt.json")
    write_receipt(receipt, receipt_path)
    print(
        json.dumps(
            {
                "output": receipt["output"],
                "content": receipt["content"],
                "round_trip": {
                    key: value
                    for key, value in receipt["round_trip"].items()
                    if key != "ids"
                },
                "receipt": str(receipt_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
