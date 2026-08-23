"""Publish the static DIII-D geometry stored in an IMAS netCDF entry.

The entry is read only through IMAS-Python's netCDF ``DBEntry`` backend.  IDSs
are requested with ``autoconvert=False`` so the receipt reports and uses the
Data Dictionary version written into each IDS.  Magnetics signals and
equilibrium labels are outside this machine-description route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_PATH = Path("/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc")
COMPETITION_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/machine-description/"
    "machine_description_receipt.json"
)
SOURCE_COCOS = 11
TARGET_COCOS = 17
VERBATIM_ARTIFACT_CONTENT_SHA256 = (
    "1e560b2ad2f2f224eed064ed3ccbeedbd88d4f4d2daca4966a855e37961c05ab"
)
VERBATIM_ARTIFACT_PHYSICAL_DIGEST = (
    "0eaa06b66b8a27263599c2c9953f43e734d4769bba0ae2398922bec9ab8a62cd"
)
VERBATIM_ARTIFACT_SEMANTIC_IDENTITY = (
    "sha256:c6592e096c5b79a0c1d435f6e89fcea02dcf178f891e5f400f033fee34266570"
)
VERBATIM_ARTIFACT_OCI_TAG = (
    "dd-4.1.1-physical-0eaa06b66b8a27263599c2c9953f43e734d4769bba0ae2398922bec9ab8a62cd"
)


def _text(value: Any) -> str:
    """Return an IMAS string scalar as plain text."""
    return str(value).strip()


def _geometry_record(element: Any, fallback_name: str) -> dict[str, Any]:
    """Extract one poloidal element without changing stored coordinates."""
    geometry = element.geometry
    geometry_type = int(geometry.geometry_type)
    name = _text(element.identifier) or _text(element.name) or fallback_name
    if geometry_type == 1:
        return {
            "name": name,
            "geometry_type": geometry_type,
            "r": [float(value) for value in np.asarray(geometry.outline.r)],
            "z": [float(value) for value in np.asarray(geometry.outline.z)],
        }
    if geometry_type == 2:
        rectangle = geometry.rectangle
        return {
            "name": name,
            "geometry_type": geometry_type,
            "r": float(rectangle.r),
            "z": float(rectangle.z),
            "width": float(rectangle.width),
            "height": float(rectangle.height),
        }
    raise ValueError(f"unsupported pf_active geometry type {geometry_type}")


def read_entry(source_path: Path = SOURCE_PATH) -> dict[str, Any]:
    """Read static geometry and content fences through the IMAS netCDF backend."""
    import imas

    with imas.DBEntry(source_path, "r") as entry:
        occurrences = {
            name: entry.list_all_occurrences(name)
            for name in (
                "wall",
                "pf_active",
                "pf_passive",
                "tf",
                "magnetics",
                "equilibrium",
            )
        }
        for required in ("wall", "pf_active", "tf", "magnetics", "equilibrium"):
            if occurrences[required] != [0]:
                raise ValueError(
                    f"expected exactly occurrence 0 for {required}, "
                    f"found {occurrences[required]}"
                )

        wall = entry.get("wall", 0, autoconvert=False)
        active = entry.get("pf_active", 0, autoconvert=False)
        toroidal = entry.get("tf", 0, lazy=True, autoconvert=False)
        magnetics = entry.get("magnetics", 0, lazy=True, autoconvert=False)
        equilibrium = entry.get("equilibrium", 0, lazy=True, autoconvert=False)

        descriptions = wall.description_2d
        if len(descriptions) != 1 or len(descriptions[0].limiter.unit) != 1:
            raise ValueError("expected one wall description with one limiter unit")
        limiter = descriptions[0].limiter.unit[0].outline
        contour = {
            "kind": "limiter",
            "r": [float(value) for value in np.asarray(limiter.r)],
            "z": [float(value) for value in np.asarray(limiter.z)],
        }

        coils = []
        for coil_index, coil in enumerate(active.coil):
            name = _text(coil.identifier) or _text(coil.name) or f"coil_{coil_index}"
            elements = [
                _geometry_record(element, f"{name}_{element_index}")
                for element_index, element in enumerate(coil.element)
            ]
            coils.append(
                {
                    "name": name,
                    "identifier": _text(coil.identifier),
                    "elements": elements,
                }
            )

        ids_versions = {
            "wall": _text(wall.ids_properties.version_put.data_dictionary),
            "pf_active": _text(active.ids_properties.version_put.data_dictionary),
            "tf": _text(toroidal.ids_properties.version_put.data_dictionary),
            "magnetics": _text(magnetics.ids_properties.version_put.data_dictionary),
            "equilibrium": _text(
                equilibrium.ids_properties.version_put.data_dictionary
            ),
            "pf_passive": None,
        }
        tf_paths = entry.list_filled_paths("tf", 0, autoconvert=False)
        equilibrium_paths = entry.list_filled_paths("equilibrium", 0, autoconvert=False)
        return {
            "source_path": str(source_path),
            "backend": "imas-python netCDF DBEntry",
            "mode": "read-only",
            "occurrences": occurrences,
            "dd_versions": ids_versions,
            "contour": contour,
            "pf_active": coils,
            "pf_passive_loop_count": 0,
            "tf_coil_count": 0,
            "tf": {
                "occurrence_present": True,
                "filled_paths": tf_paths,
                "static_geometry_present": any(
                    path.startswith(("coil/", "r0")) for path in tf_paths
                ),
            },
            "doctrine_fence": {
                "magnetics_occurrence_present": True,
                "equilibrium_occurrence_present": True,
                "equilibrium_time_slice_count": len(equilibrium.time_slice),
                "equilibrium_constraints_present": any(
                    path.startswith("time_slice/constraints/")
                    for path in equilibrium_paths
                ),
            },
        }


def read_competition_grid(
    receipt_path: Path = COMPETITION_RECEIPT,
) -> dict[str, Any]:
    """Read the released competition grid extent from its existing receipt."""
    receipt = json.loads(receipt_path.read_text())
    grid = receipt["quantities"]["efit_grid"]
    return {
        "shape": list(grid["shape"]),
        "r_extent_m": list(grid["r_extent_m"]),
        "z_extent_m": list(grid["z_extent_m"]),
        "source_receipt": str(receipt_path),
        "provenance": grid["provenance"],
    }


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_repaired_artifact_receipt(
    *,
    source_path: Path,
    ids_path: Path,
    manifest_path: Path,
    artifact_receipt_path: Path,
    recipe_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Record the repaired ring and its distinct publication identities."""

    from nova.imas.diiid_machine_ids import build_diiid_machine_ids
    from nova.imas.machine_artifact import MachineArtifactManifest

    bundle = build_diiid_machine_ids(source_path)
    manifest = MachineArtifactManifest.from_bytes(manifest_path.read_bytes())
    artifact_receipt = json.loads(artifact_receipt_path.read_text())
    if len(manifest.files) != 1 or manifest.files[0].name != ids_path.name:
        raise ValueError("manifest must describe exactly the DIII-D netCDF payload")
    content_sha256 = _sha256(ids_path)
    if content_sha256 != manifest.files[0].sha256:
        raise ValueError("manifest content digest does not match the netCDF payload")
    if artifact_receipt["output"]["sha256"] != content_sha256:
        raise ValueError("artifact receipt content digest does not match the payload")
    semantic_identity = manifest.semantic_identity()
    comparison = {
        "content_digest_distinct": (content_sha256 != VERBATIM_ARTIFACT_CONTENT_SHA256),
        "physical_digest_distinct": (
            manifest.physical_digest != VERBATIM_ARTIFACT_PHYSICAL_DIGEST
        ),
        "semantic_identity_distinct": (
            semantic_identity != VERBATIM_ARTIFACT_SEMANTIC_IDENTITY
        ),
        "oci_tag_distinct": manifest.oci.tag != VERBATIM_ARTIFACT_OCI_TAG,
    }
    if not all(comparison.values()):
        raise ValueError("repaired artifact collides with the verbatim DD4 artifact")
    receipt = {
        "measurement": "DIII-D repaired-ring machine-artifact publication identity",
        "limiter": bundle.limiter_repair.as_dict(),
        "artifact": {
            "ids_path": str(ids_path),
            "manifest_path": str(manifest_path),
            "manifest_valid": True,
            "data_dictionary": manifest.dd_version,
            "content_sha256": content_sha256,
            "physical_digest": manifest.physical_digest,
            "semantic_identity": semantic_identity,
            "manifest_digest": manifest.digest,
            "oci_tag": manifest.oci.tag,
            "round_trip": artifact_receipt["round_trip"],
        },
        "native_authoring": {
            "source_data_dictionary": bundle.source_dd_version,
            "target_data_dictionary": bundle.dd_version,
            "target_resolver": "publication_dd_version()",
            "cross_major_conversion_performed": False,
        },
        "declared_absent": [absence.as_dict() for absence in bundle.absent],
        "data_dictionary_floor": artifact_receipt["data_dictionary_floor"],
        "superseded_verbatim_artifact": {
            "content_sha256": VERBATIM_ARTIFACT_CONTENT_SHA256,
            "physical_digest": VERBATIM_ARTIFACT_PHYSICAL_DIGEST,
            "semantic_identity": VERBATIM_ARTIFACT_SEMANTIC_IDENTITY,
            "oci_tag": VERBATIM_ARTIFACT_OCI_TAG,
        },
        "identity_comparison": comparison,
        "publication": {
            "recipe_path": str(recipe_path),
            "owner_run_command": True,
            "network_publication_attempted": False,
        },
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def _element_receipt(section: Any, source: dict[str, Any]) -> dict[str, Any]:
    """Describe one element using only values exposed by the dataclass seam."""
    source_geometry = {1: "outline", 2: "rectangle"}[source["geometry_type"]]
    if source_geometry == "outline":
        status = "read_unmodified"
        change = None
    else:
        status = "read_after_named_change"
        change = (
            "expand the stored centre, width and height through "
            "CrossSection.transform[2] into the canonical polygon outline"
        )
    return {
        "name": section.name,
        "status": status,
        "change": change,
        "source_geometry": source_geometry,
        "outline_vertex_count": len(section.outline),
        "outline_vertices_m": [list(vertex) for vertex in section.outline],
        "centre_m": list(section.centre),
        "width_m": section.width,
        "height_m": section.height,
        "skew_rad": section.skew,
        "skew_definition": "directed angle of the first non-zero outline edge",
    }


def build_receipt(
    source: dict[str, Any], competition_grid: dict[str, Any]
) -> tuple[dict[str, Any], Any]:
    """Route the static entry through Nova and classify every quantity."""
    from nova.imas.diiid_machine_ids import repair_limiter_ring
    from nova.imas.machine import StaticMachineDescription
    from nova.io.cocos import (
        B0_LIKE,
        DODPSI_LIKE,
        IP_LIKE,
        PSI_LIKE,
        Q_LIKE,
        convention_transform,
    )

    source_contour = np.column_stack((source["contour"]["r"], source["contour"]["z"]))
    repaired_contour, limiter_repair = repair_limiter_ring(source_contour)
    machine_record = dict(source)
    machine_record["contour"] = {
        "kind": source["contour"]["kind"],
        "r": repaired_contour[:, 0],
        "z": repaired_contour[:, 1],
    }
    machine = StaticMachineDescription.from_record(machine_record)
    if len(machine.active_coils) != len(source["pf_active"]):
        raise ValueError("machine dataclass route did not preserve pf_active coils")
    coil_receipts = []
    for coil, coil_source in zip(
        machine.active_coils, source["pf_active"], strict=True
    ):
        elements = [
            _element_receipt(element, element_source)
            for element, element_source in zip(
                coil.elements, coil_source["elements"], strict=True
            )
        ]
        coil_receipts.append(
            {
                "name": coil.name,
                "identifier": coil.identifier,
                "element_count": len(elements),
                "elements": elements,
            }
        )

    contour = machine.contour
    transform = convention_transform(source=SOURCE_COCOS, target=TARGET_COCOS)
    factors = {
        quantity: transform.factor(quantity)
        for quantity in (PSI_LIKE, IP_LIKE, B0_LIKE, Q_LIKE, DODPSI_LIKE)
    }
    element_count = sum(coil["element_count"] for coil in coil_receipts)
    non_rectangular_count = sum(
        element["source_geometry"] == "outline"
        for coil in coil_receipts
        for element in coil["elements"]
    )
    receipt = {
        "measurement": "DIII-D netCDF static machine-description read",
        "source": {
            "path": source["source_path"],
            "backend": source["backend"],
            "mode": source["mode"],
            "occurrences": source["occurrences"],
        },
        "dd_version_policy": (
            "each IDS is read with autoconvert=False and its version is copied "
            "from ids_properties.version_put.data_dictionary"
        ),
        "dd_versions": source["dd_versions"],
        "machine_dataclass_route": {
            "class": "nova.imas.machine.StaticMachineDescription",
            "coil_class": "nova.imas.machine.MachineCoil",
            "element_class": "nova.imas.machine.MachineSection",
            "section_dispatch": "nova.imas.machine.CrossSection.transform",
        },
        "cocos": {
            "ids": {
                "source_index": SOURCE_COCOS,
                "source": "IMAS Data Dictionary coordinate convention",
                "target_index": TARGET_COCOS,
                "transform_to_nova": factors,
            },
            "competition_corpus": {
                "source_index": 5,
                "source": "separate empirical corpus determination",
                "used_for_this_ids_read": False,
            },
        },
        "quantities": {
            "wall_limiter": {
                "status": (
                    "read_unmodified"
                    if limiter_repair.source_chain_valid
                    else "read_after_named_change"
                ),
                "change": (
                    None
                    if limiter_repair.source_chain_valid
                    else (
                        "apply the declared validity repair and retain only the "
                        "largest positive polygon as one canonical physical ring"
                    )
                ),
                "kind": contour.kind if contour else None,
                "vertex_count": len(contour.r) if contour else 0,
                "r_extent_m": [min(contour.r), max(contour.r)] if contour else None,
                "z_extent_m": [min(contour.z), max(contour.z)] if contour else None,
                "provenance": limiter_repair.as_dict(),
            },
            "pf_active": {
                "status": "read_after_named_change",
                "change": (
                    "retain IDS coil-to-element grouping and expand rectangle "
                    "records through the canonical CrossSection polygon route"
                ),
                "coil_count": len(coil_receipts),
                "element_count": element_count,
                "non_rectangular_element_count": non_rectangular_count,
                "coils": coil_receipts,
            },
            "tf": {
                "status": "cannot_reach",
                "occurrence_present": source["tf"]["occurrence_present"],
                "static_geometry_present": source["tf"]["static_geometry_present"],
                "toroidal_coil_count": machine.toroidal_coil_count,
                "reason": (
                    "the tf IDS contains a time-dependent vacuum-field signal but "
                    "no static toroidal-conductor geometry; the signal is excluded"
                ),
            },
            "pf_passive": {
                "status": "cannot_reach",
                "occurrence_present": bool(source["occurrences"]["pf_passive"]),
                "loop_count": machine.passive_loop_count,
                "reason": (
                    "the netCDF entry contains no pf_passive occurrence; no loop "
                    "or vessel conductor is fabricated"
                ),
            },
            "competition_efit_grid": competition_grid,
        },
        "doctrine_fence": {
            "additional_entry_content": {
                "magnetics_ids": source["doctrine_fence"][
                    "magnetics_occurrence_present"
                ],
                "constrained_equilibrium": source["doctrine_fence"][
                    "equilibrium_constraints_present"
                ],
                "equilibrium_time_slice_count": source["doctrine_fence"][
                    "equilibrium_time_slice_count"
                ],
            },
            "admissible_machine_description": (
                "diagnostic and conductor geometry only"
            ),
            "magnetics_signal_used": False,
            "equilibrium_label_used": False,
            "statement": (
                "no magnetics signal and no equilibrium label from this file is "
                "used anywhere in this node"
            ),
        },
    }
    return receipt, machine


def write_figures(machine: Any, receipt: dict[str, Any], output: Path) -> list[Path]:
    """Draw the limiter, every active element, outline detail and grid extent."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch
    from matplotlib.patches import Rectangle as RectanglePatch

    output.mkdir(parents=True, exist_ok=True)
    paths = []

    figure, axes = plt.subplots(figsize=(10, 9), constrained_layout=True)
    contour = machine.contour
    axes.plot(contour.r, contour.z, color="black", linewidth=1.8)
    for coil in machine.active_coils:
        for element in coil.elements:
            outline = np.asarray(element.outline)
            axes.add_patch(
                PolygonPatch(
                    outline,
                    fill=False,
                    edgecolor="tab:blue",
                    linewidth=0.65,
                )
            )
            centre_r, centre_z = element.centre
            axes.text(
                centre_r,
                centre_z,
                element.name,
                fontsize=3.2,
                ha="center",
                va="center",
            )
    axes.set_aspect("equal")
    axes.set_xlabel("R [m]")
    axes.set_ylabel("Z [m]")
    axes.set_title("netCDF limiter and every pf_active element")
    path = output / "limiter_pf_active_elements.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    paths.append(path)

    outlines = [
        element
        for coil in machine.active_coils
        for element in coil.elements
        if element.section.name == "outline"
    ]
    if not outlines:
        raise ValueError("no non-rectangular element outlines found")
    columns = 3
    rows = (len(outlines) + columns - 1) // columns
    figure, axes_array = plt.subplots(
        rows,
        columns,
        figsize=(9, 2.8 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for axes, element in zip(axes_array.flat, outlines, strict=False):
        outline = np.asarray(element.outline)
        closed = np.vstack([outline, outline[0]])
        axes.plot(closed[:, 0], closed[:, 1], "o-", color="tab:blue")
        for index, vertex in enumerate(outline):
            axes.text(vertex[0], vertex[1], str(index), fontsize=7)
        axes.set_title(f"{element.name}: skew {element.skew:.4f} rad")
        axes.set_aspect("equal")
        axes.set_xlabel("R [m]")
        axes.set_ylabel("Z [m]")
    for axes in axes_array.flat[len(outlines) :]:
        axes.remove()
    path = output / "non_rectangular_element_outlines.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    paths.append(path)

    grid = receipt["quantities"]["competition_efit_grid"]
    r_min, r_max = grid["r_extent_m"]
    z_min, z_max = grid["z_extent_m"]
    figure, axes = plt.subplots(figsize=(7, 7), constrained_layout=True)
    axes.plot(contour.r, contour.z, color="black", linewidth=1.8, label="limiter")
    axes.add_patch(
        RectanglePatch(
            (r_min, z_min),
            r_max - r_min,
            z_max - z_min,
            fill=False,
            edgecolor="tab:orange",
            linestyle="--",
            linewidth=1.8,
            label=f"competition efit_grid {grid['shape'][0]}x{grid['shape'][1]}",
        )
    )
    axes.set_aspect("equal")
    axes.set_xlabel("R [m]")
    axes.set_ylabel("Z [m]")
    axes.legend(frameon=False)
    axes.set_title("netCDF limiter against the released competition grid")
    path = output / "limiter_competition_grid_extent.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    paths.append(path)
    return paths


def main() -> None:
    """Read the source and publish its receipt and figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--competition-receipt", type=Path, default=COMPETITION_RECEIPT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/diiid-forward-onboarding/netcdf-description"),
    )
    args = parser.parse_args()
    source = read_entry(args.source)
    competition_grid = read_competition_grid(args.competition_receipt)
    receipt, machine = build_receipt(source, competition_grid)
    figures = write_figures(machine, receipt, args.output)
    receipt["figures"] = [str(path) for path in figures]
    receipt_path = args.output / "netcdf_description_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "figures": len(figures),
                "limiter_vertices": receipt["quantities"]["wall_limiter"][
                    "vertex_count"
                ],
                "pf_active_coils": receipt["quantities"]["pf_active"]["coil_count"],
                "pf_active_elements": receipt["quantities"]["pf_active"][
                    "element_count"
                ],
                "pf_passive_loops": receipt["quantities"]["pf_passive"]["loop_count"],
            }
        )
    )


if __name__ == "__main__":
    main()
