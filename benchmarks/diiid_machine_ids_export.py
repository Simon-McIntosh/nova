"""Write and verify the static DIII-D machine-description netCDF entry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import imas
import numpy as np

from nova.imas.diiid_machine_ids import (
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
SUPERSEDED_DD_VERSION = "3.41.0"
SUPERSEDED_SIZE_BYTES = 262087
SUPERSEDED_SHA256 = "a6ef157f558592ee9318e0ad02539dfbbe5c812e55d5d5218c5692f759f6e5ef"
SUPERSEDED_LEAF_COUNT = 1833


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


def _read_ids(output: Path, dd_version: str) -> dict[str, Any]:
    database = imas.DBEntry(output.resolve(), "r", dd_version=dd_version)
    try:
        return {name: database.get(name, 0, autoconvert=False) for name in IDS_NAMES}
    finally:
        database.close()


def _written_dd_version(output: Path) -> str:
    """Read the version declared by an existing entry without conversion."""

    with imas.DBEntry(output.resolve(), "r") as database:
        wall = database.get("wall", 0, lazy=True, autoconvert=False)
        return str(wall.ids_properties.version_put.data_dictionary)


def _reopen(
    output: Path, dd_version: str
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    database = imas.DBEntry(output.resolve(), "r", dd_version=dd_version)
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
        if written != dd_version:
            raise ValueError(
                f"reopened {name} carries Data Dictionary {written}, "
                f"expected {dd_version}"
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


def _major_comparison(
    previous: Mapping[str, Mapping[str, Any]] | None,
    current: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Name every leaf-presence and shape change from the superseded artifact."""

    current_count = sum(len(leaves) for leaves in current.values())
    presence_differences = []
    shape_differences = []
    if previous is not None:
        previous_count = sum(len(leaves) for leaves in previous.values())
        for ids_name in IDS_NAMES:
            previous_paths = set(previous[ids_name])
            current_paths = set(current[ids_name])
            presence_differences.extend(
                {
                    "ids": ids_name,
                    "path": path,
                    "presence": "DD4 only",
                }
                for path in sorted(current_paths - previous_paths)
            )
            presence_differences.extend(
                {
                    "ids": ids_name,
                    "path": path,
                    "presence": "DD3 only",
                }
                for path in sorted(previous_paths - current_paths)
            )
            for path in sorted(previous_paths & current_paths):
                previous_shape = list(np.asarray(previous[ids_name][path]).shape)
                current_shape = list(np.asarray(current[ids_name][path]).shape)
                if previous_shape != current_shape:
                    shape_differences.append(
                        {
                            "ids": ids_name,
                            "path": path,
                            "dd3_shape": previous_shape,
                            "dd4_shape": current_shape,
                        }
                    )
    else:
        previous_count = SUPERSEDED_LEAF_COUNT
    return {
        "superseded_leaf_count": previous_count,
        "published_leaf_count": current_count,
        "leaf_count_difference": current_count - previous_count,
        "presence_differences": presence_differences,
        "shape_differences": shape_differences,
        "comparison_available": previous is not None,
    }


def export_machine_ids(
    output: Path | str = DEFAULT_OUTPUT,
    *,
    source_path: Path | str = SOURCE_PATH,
) -> dict[str, Any]:
    """Write, reopen, compare and describe the DIII-D IDS set."""

    output = Path(output)
    superseded_size = SUPERSEDED_SIZE_BYTES
    superseded_sha256 = SUPERSEDED_SHA256
    previous_snapshot = None
    preserved_major_comparison = None
    bundle = build_diiid_machine_ids(source_path)
    if output.exists():
        written_dd_version = _written_dd_version(output)
        if written_dd_version == SUPERSEDED_DD_VERSION:
            superseded_size = output.stat().st_size
            superseded_sha256 = _sha256(output)
            previous_ids = _read_ids(output, written_dd_version)
            previous_snapshot = machine_ids_snapshot(previous_ids)
        elif written_dd_version == bundle.dd_version:
            existing_receipt = output.with_suffix(".receipt.json")
            if existing_receipt.exists():
                prior_receipt = json.loads(existing_receipt.read_text())
                preserved_major_comparison = prior_receipt["round_trip"][
                    "major_comparison"
                ]
        else:
            raise ValueError(
                f"existing output carries unsupported Data Dictionary "
                f"{written_dd_version}"
            )
    expected = machine_ids_snapshot(bundle.ids)
    major_comparison = _major_comparison(previous_snapshot, expected)
    if preserved_major_comparison is not None:
        expected_leaf_count = sum(len(leaves) for leaves in expected.values())
        if preserved_major_comparison["published_leaf_count"] != expected_leaf_count:
            raise ValueError(
                "existing cross-major comparison does not describe authored leaves"
            )
        major_comparison = preserved_major_comparison
    _write(bundle, output)
    reopened, filled_paths = _reopen(output, bundle.dd_version)
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
            "data_dictionary": bundle.source_dd_version,
            "backend": "imas-python netCDF DBEntry",
            "autoconvert": False,
            "machine_dataclass_route": [
                "nova.imas.machine.StaticMachineDescription",
                "nova.imas.machine.MachineContour",
                "nova.imas.machine.MachineCoil",
                "nova.imas.machine.MachineSection",
                "nova.imas.machine.DiagnosticSightline",
            ],
        },
        "native_authoring": {
            "target_data_dictionary": bundle.dd_version,
            "resolved_from": "imas.dd_zip.dd_xml_versions",
            "cross_major_conversion_performed": False,
            "statement": (
                "The DD3 source is read with autoconvert=False into Nova machine "
                "dataclasses, then a latest-DD4 IDSFactory authors fresh IDSs; "
                "no conversion of these IDSs occurs."
            ),
        },
        "output": {
            "path": str(output),
            "size_bytes": output.stat().st_size,
            "sha256": _sha256(output),
            "superseded": {
                "data_dictionary": SUPERSEDED_DD_VERSION,
                "size_bytes": superseded_size,
                "sha256": superseded_sha256,
            },
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
                "source": str(reopened[name].ids_properties.provenance.node[0].path),
                "source_leaf_present": hasattr(reopened[name].ids_properties, "source"),
                "comment": str(reopened[name].ids_properties.comment),
                "provenance": [
                    {
                        "path": str(node.path),
                        "sources": (
                            [str(value) for value in node.sources]
                            if hasattr(node, "sources")
                            else [str(reference.name) for reference in node.reference]
                        ),
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
            "total_leaf_count": sum(len(leaves) for leaves in comparison.values()),
            "major_comparison": major_comparison,
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
    if int(bundle.dd_version.partition(".")[0]) != 4:
        raise ValueError(f"published Data Dictionary is not DD4: {bundle.dd_version}")
    if major_comparison["published_leaf_count"] != sum(
        len(leaves) for leaves in comparison.values()
    ):
        raise ValueError("authored and reopened leaf counts disagree")
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
