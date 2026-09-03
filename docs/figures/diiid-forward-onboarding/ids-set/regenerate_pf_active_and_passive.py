"""Regenerate the persisted DIII-D machine-description IDS set.

Extends the existing wall/pf_active/magnetics bundle with pf_active circuits
and supplies (the ECOILA-driven ohmic circuit plus one direct circuit per
F-coil) and adds pf_passive (the 47 wall-following loops), then writes all
four IDSs back to the checked-in netCDF artifact and banks its manifest,
receipt and round-trip evidence. The wall/pf_active-coil/magnetics content
and its writer are untouched; only the wiring and pf_passive additions are
authored here, directly onto the same bundle the existing writer produces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import imas
import numpy as np

from nova.imas.diiid_description import (
    ALL_PF_ACTIVE_CIRCUITS,
    ALL_PF_ACTIVE_SUPPLIES,
    author_pf_active_circuits,
)
from nova.imas.diiid_machine_ids import (
    IDS_NAMES,
    build_diiid_machine_ids,
    machine_ids_snapshot,
    round_trip_leaf_receipt,
    _primitive_leaves,
)
from nova.imas.diiid_passive import LOOP_COUNT, build_description as build_pf_passive
from nova.imas.machine_artifact import (
    ArtifactShotRange,
    create_machine_artifact_manifest,
)
from nova.scripts.diiid_machine_artifact import (
    _publication_floor_receipt,
    _write_publication_recipe,
)
from benchmarks.diiid_ids_machine_description import write_repaired_artifact_receipt

OUTPUT_DIRECTORY = Path("docs/figures/diiid-forward-onboarding/ids-set")
OUTPUT_IDS = OUTPUT_DIRECTORY / "diiid_machine_description.nc"
REPOSITORY = "ghcr.io/registry-account/diii-d-machine-description"
EXTENDED_IDS_NAMES = (*IDS_NAMES, "pf_passive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wiring_and_passive_snapshot(
    pf_active_ids: Any, pf_passive_ids: Any
) -> dict[str, dict[str, Any]]:
    """Detach every circuit, supply and pf_passive loop leaf we authored."""

    wiring: dict[str, Any] = {}
    for index, circuit in enumerate(pf_active_ids.circuit):
        prefix = f"circuit[{index}]"
        for leaf, value in _primitive_leaves(circuit, ("name", "description")):
            wiring[f"{prefix}/{leaf}"] = value
        wiring[f"{prefix}/connections"] = np.asarray(circuit.connections).copy()
    for index, supply in enumerate(pf_active_ids.supply):
        prefix = f"supply[{index}]"
        for leaf, value in _primitive_leaves(supply, ("name", "description")):
            wiring[f"{prefix}/{leaf}"] = value

    passive: dict[str, Any] = {}
    for index, loop in enumerate(pf_passive_ids.loop):
        prefix = f"loop[{index}]"
        for leaf, value in _primitive_leaves(
            loop,
            (
                "name",
                "description",
                "resistivity",
                "resistance",
                "resistance_error_lower",
                "resistance_error_upper",
            ),
        ):
            passive[f"{prefix}/{leaf}"] = value
        for element_index, element in enumerate(loop.element):
            element_prefix = f"{prefix}/element[{element_index}]"
            for leaf, value in _primitive_leaves(
                element, ("name", "description", "area", "turns_with_sign")
            ):
                passive[f"{element_prefix}/{leaf}"] = value
            passive[f"{element_prefix}/geometry/geometry_type"] = int(
                element.geometry.geometry_type
            )
            passive[f"{element_prefix}/geometry/outline/r"] = np.asarray(
                element.geometry.outline.r
            ).copy()
            passive[f"{element_prefix}/geometry/outline/z"] = np.asarray(
                element.geometry.outline.z
            ).copy()
    return {"pf_active_wiring": wiring, "pf_passive": passive}


def _compare_leaves(
    expected: dict[str, dict[str, Any]], actual: dict[str, dict[str, Any]]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compare arbitrary leaf-path groups, raising on any change or drift."""

    receipt: dict[str, dict[str, dict[str, Any]]] = {}
    for group, expected_leaves in expected.items():
        actual_leaves = actual[group]
        if set(expected_leaves) != set(actual_leaves):
            missing = sorted(set(expected_leaves) - set(actual_leaves))
            extra = sorted(set(actual_leaves) - set(expected_leaves))
            raise ValueError(
                f"{group} leaf structure changed; missing={missing}, extra={extra}"
            )
        group_receipt = {}
        for path, left in expected_leaves.items():
            right = actual_leaves[path]
            left_array = np.asarray(left)
            right_array = np.asarray(right)
            equal = bool(np.array_equal(left_array, right_array))
            if left_array.dtype.kind in "SUO" or right_array.dtype.kind in "SUO":
                item = {"exact_equal": equal}
            else:
                difference = (
                    0.0
                    if left_array.size == 0
                    else float(np.max(np.abs(left_array - right_array)))
                )
                item = {"exact_equal": equal, "maximum_absolute_difference": difference}
            if not equal:
                raise ValueError(f"{group}/{path} changed during round trip")
            group_receipt[path] = item
        receipt[group] = group_receipt
    return receipt


def regenerate(output: Path = OUTPUT_IDS) -> dict[str, Any]:
    """Author, write, verify and bank the extended DIII-D IDS set."""

    bundle = build_diiid_machine_ids()
    author_pf_active_circuits(
        bundle.ids["pf_active"],
        supplies=ALL_PF_ACTIVE_SUPPLIES,
        circuits=ALL_PF_ACTIVE_CIRCUITS,
    )
    pf_passive_description = build_pf_passive()
    pf_passive_description.validate()
    ids = {**bundle.ids, "pf_passive": pf_passive_description.pf_passive}

    expected_core = machine_ids_snapshot(bundle.ids)
    expected_extra = _wiring_and_passive_snapshot(
        bundle.ids["pf_active"], ids["pf_passive"]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    database = imas.DBEntry(output.resolve(), "w", dd_version=bundle.dd_version)
    try:
        for name in EXTENDED_IDS_NAMES:
            database.put(ids[name])
    finally:
        database.close()

    database = imas.DBEntry(output.resolve(), "r", dd_version=bundle.dd_version)
    try:
        reopened = {
            name: database.get(name, 0, autoconvert=False)
            for name in EXTENDED_IDS_NAMES
        }
    finally:
        database.close()

    core_comparison = round_trip_leaf_receipt(
        expected_core, machine_ids_snapshot(reopened)
    )
    extra_comparison = _compare_leaves(
        expected_extra,
        _wiring_and_passive_snapshot(reopened["pf_active"], reopened["pf_passive"]),
    )
    core_leaves = [
        result for values in core_comparison.values() for result in values.values()
    ]
    extra_leaves = [
        result for values in extra_comparison.values() for result in values.values()
    ]
    all_leaves = core_leaves + extra_leaves
    if not all(result["exact_equal"] for result in all_leaves):
        raise ValueError("DIII-D machine IDS leaves changed during netCDF round trip")
    round_trip = {
        "authored_leaf_count": len(all_leaves),
        "exact_equal": True,
        "maximum_absolute_difference": max(
            (
                float(result["maximum_absolute_difference"])
                for result in all_leaves
                if "maximum_absolute_difference" in result
            ),
            default=0.0,
        ),
    }

    poloidal_coil_count = len(reopened["pf_active"].coil)
    circuit_count = len(reopened["pf_active"].circuit)
    supply_count = len(reopened["pf_active"].supply)
    passive_loop_count = len(reopened["pf_passive"].loop)
    if poloidal_coil_count != 24:
        raise ValueError(f"expected 24 poloidal coils, got {poloidal_coil_count}")
    if passive_loop_count != LOOP_COUNT:
        raise ValueError(
            f"expected {LOOP_COUNT} pf_passive loops, got {passive_loop_count}"
        )

    manifest_path = output.with_name("diiid_machine_description.manifest.json")
    receipt_path = output.with_name("diiid_pf_active_circuits.receipt.json")
    recipe_path = output.with_name("PUBLISH.md")

    payload_sha256 = _sha256(output)
    unresolved_gaps = tuple(
        f"{absence.quantity}: {absence.reason}" for absence in bundle.absent
    )
    # Manifest inventory is directory-wide (nova.imas.machine_artifact scans
    # every file under the given directory), so it is built from a clean
    # single-file staging directory -- never straight from OUTPUT_DIRECTORY,
    # which also holds this script, its receipt and PUBLISH.md.
    with TemporaryDirectory(prefix=".diiid-ids-set-manifest-") as staging:
        staged_ids = Path(staging) / output.name
        shutil.copyfile(output, staged_ids)
        manifest = create_machine_artifact_manifest(
            staged_ids.parent,
            machine="DIII-D",
            dd_version=bundle.dd_version,
            registry_digest=payload_sha256,
            physical_digest=payload_sha256,
            shot_ranges=(
                ArtifactShotRange(
                    first_shot=200000,
                    last_shot=200000,
                    physical_digest=payload_sha256,
                    evidence="observed",
                ),
            ),
            complete=False,
            unresolved_gaps=unresolved_gaps,
        )
    if len(manifest.files) != 1 or manifest.files[0].name != output.name:
        raise ValueError("manifest must describe exactly the DIII-D netCDF payload")
    manifest_path.write_bytes(manifest.canonical_bytes())
    _write_publication_recipe(
        recipe_path,
        repository=REPOSITORY,
        output=output,
        manifest_path=manifest_path,
        manifest=manifest,
        payload_sha256=payload_sha256,
    )

    # The identity receipt (nova/scripts/diiid_machine_artifact.py's own
    # schema) is re-derived through its owning helper rather than duplicated
    # here, so its cross-file digest checks stay authoritative. A synthetic
    # artifact_receipt supplies this node's own round trip and floor probe in
    # the exact shape write_repaired_artifact_receipt requires.
    artifact_receipt_path = output.with_name("diiid_machine_artifact.receipt.json")
    identity_receipt_path = output.with_name("diiid_machine_description.receipt.json")
    with TemporaryDirectory(prefix=".diiid-ids-set-floor-") as floor_directory:
        floor_receipt = _publication_floor_receipt(
            Path(floor_directory),
            registry_digest=payload_sha256,
            physical_digest=payload_sha256,
            unresolved_gaps=unresolved_gaps,
        )
    artifact_receipt_path.write_text(
        json.dumps(
            {
                "output": {"sha256": payload_sha256},
                "round_trip": round_trip,
                "data_dictionary_floor": floor_receipt,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_repaired_artifact_receipt(
        source_path=bundle.source_path,
        ids_path=output,
        manifest_path=manifest_path,
        artifact_receipt_path=artifact_receipt_path,
        recipe_path=recipe_path,
        receipt_path=identity_receipt_path,
    )

    receipt = {
        "measurement": "DIII-D pf_active circuits and pf_passive loops regeneration",
        "source": {
            "path": str(bundle.source_path),
            "source_dd_version": bundle.source_dd_version,
            "target_dd_version": bundle.dd_version,
        },
        "output": {
            "path": str(output),
            "sha256": payload_sha256,
            "size_bytes": output.stat().st_size,
        },
        "manifest": {
            "path": str(manifest_path),
            "digest": manifest.digest,
            "dd_version": manifest.dd_version,
        },
        "counts": {
            "poloidal_coils": poloidal_coil_count,
            "pf_active_circuits": circuit_count,
            "pf_active_supplies": supply_count,
            "pf_passive_loops": passive_loop_count,
        },
        "wiring": {
            "ohmic_circuit_conductors": [
                "ECOILA",
                *[drive.conductor for drive in ALL_PF_ACTIVE_CIRCUITS[0].drives],
            ],
            "ohmic_circuit_effective_gains": [
                drive.gain for drive in ALL_PF_ACTIVE_CIRCUITS[0].drives
            ],
            "f_coil_circuit_count": len(ALL_PF_ACTIVE_CIRCUITS) - 1,
            "connections_matrix_shape_ohmic": list(
                np.asarray(reopened["pf_active"].circuit[0].connections).shape
            ),
        },
        "round_trip": round_trip,
        "identity": {
            "artifact_receipt_path": str(artifact_receipt_path),
            "identity_receipt_path": str(identity_receipt_path),
        },
        "vacuum_field_reproduction": {
            "measured_in_this_node": False,
            "reason": (
                "the current-constrained forward solve that reproduces the "
                "inclusion ladder's all-conductors rung on frame 89 is owned by "
                "nova.equilibrium modules assigned to concurrent nodes in this "
                "sprint (nia-connected-stationary-points, "
                "nia-line-of-sight-limiter); this node's write scope is limited "
                "to nova/imas/diiid_description.py, nova/imas/diiid_passive.py, "
                "tests/test_diiid_pf_active_circuits.py and this figures "
                "directory, none of which can run a forward Grad-Shafranov "
                "solve"
            ),
            "comparison_target": (
                "docs/figures/current-constrained-forward-solve/inclusion-ladder/"
                "solenoid_inclusion_receipt.json (frame 89, all-conductors rung)"
            ),
            "recommended_follow_up": (
                "a separately dispatched node should build the vacuum field from "
                "this regenerated pf_active/pf_passive description (reusing "
                "nova.imas.diiid_description.active_coil_response_from_imas "
                "against the persisted diiid_machine_description.nc) and compare "
                "frame 89's X-point separation against the banked receipt"
            ),
        },
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_IDS)
    args = parser.parse_args()
    receipt = regenerate(args.output)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
