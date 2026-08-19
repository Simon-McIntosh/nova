"""Publish the competition-authoritative DIII-D machine description receipt."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from nova.imas.diiid_description import (
    DiiidDatasetMachineDescription,
    dataset_machine_description,
)


DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/machine-description")
DATASET_COLUMNS = (
    "coil_name",
    "coil_input_column",
    "coil_R",
    "coil_Z",
    "coil_width",
    "coil_height",
    "coil_angle1",
    "coil_angle2",
    "thomson_chord_name",
    "thomson_chord_R",
    "thomson_chord_Z",
    "efit_grid_R",
    "efit_grid_Z",
)


def read_row(path: Path) -> dict[str, Any]:
    """Read only the released machine-geometry columns from one corpus object."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError(
            "the machine-description receipt requires a pyarrow-enabled runner"
        ) from error
    table = parquet.read_table(path, columns=list(DATASET_COLUMNS))
    if table.num_rows != 1:
        raise ValueError(f"expected one row in {path}, found {table.num_rows}")
    return {name: table[name][0].as_py() for name in table.column_names}


def _routed_vertices(section: Any) -> np.ndarray:
    data = section.section.data
    return np.column_stack(
        [np.asarray(data["r"], dtype=float), np.asarray(data["z"], dtype=float)]
    )


def build_receipt(
    row: dict[str, Any], *, source_row: str
) -> tuple[dict[str, Any], DiiidDatasetMachineDescription]:
    """Build and quantify the one-construction machine dataclass route."""
    description = dataset_machine_description(row, source_row=source_row)
    sections = {
        section.name: section for section in description.machine.active_sections
    }
    source_index = {str(name): index for index, name in enumerate(row["coil_name"])}
    conductor_receipts = []
    maximum_difference = 0.0
    for conductor in description.physical.conductors:
        if conductor.vertices is None:
            continue
        index = source_index[conductor.name]
        routed = _routed_vertices(sections[conductor.name])
        difference = float(np.max(np.abs(routed - conductor.vertices)))
        maximum_difference = max(maximum_difference, difference)
        angle1 = float(row["coil_angle1"][index])
        angle2 = float(row["coil_angle2"][index])
        conductor_receipts.append(
            {
                "name": conductor.name,
                "centre_m": [
                    float(row["coil_R"][index]),
                    float(row["coil_Z"][index]),
                ],
                "width_m": float(row["coil_width"][index]),
                "height_m": float(row["coil_height"][index]),
                "skew": {
                    "angle1_deg": angle1,
                    "angle2_deg": angle2,
                    "effective_deg": angle1 if angle1 else angle2,
                },
                "section_vertices_m": conductor.vertices.tolist(),
                "routed_section_vertices_m": routed.tolist(),
                "maximum_route_difference_m": difference,
                "machine_seam": {
                    "acceptance": "accepted_after_named_change",
                    "change": (
                        "convert the shipped rectangle and shear parameters once "
                        "to an outline record consumed by CrossSection.transform[1]"
                    ),
                    "classes": [
                        "StaticMachineDescription",
                        "MachineSection",
                        "CrossSection.transform[1]",
                    ],
                },
                "provenance": [asdict(receipt) for receipt in conductor.receipts],
            }
        )

    bcoil = next(
        conductor
        for conductor in description.physical.conductors
        if conductor.name == "bcoil"
    )
    sightlines = [
        {
            "name": sightline.name,
            "coordinates_r_z_m": [sightline.position[0], sightline.position[1]],
            "representative_phi_rad": sightline.position[2],
            "start": sightline.start,
            "end": sightline.end,
        }
        for sightline in description.machine.sightlines
    ]
    receipt = {
        "measurement": "competition-authoritative DIII-D machine description",
        "source_row": source_row,
        "physical_geometry_digest": description.physical.physical_digest,
        "provenance_complete": description.provenance_complete,
        "machine_dataclass_route": {
            "class": "nova.imas.machine.StaticMachineDescription",
            "section_class": "nova.imas.machine.MachineSection",
            "cross_section_dispatch": "nova.imas.machine.CrossSection.transform[1]",
            "active_section_count": len(description.machine.active_sections),
            "maximum_vertex_route_difference_m": maximum_difference,
            "vertex_identity_tolerance_m": 1.0e-12,
            "vertex_identity_passed": maximum_difference <= 1.0e-12,
        },
        "quantities": {
            "poloidal_conductors": {
                "acceptance": "accepted_after_named_change",
                "count": len(conductor_receipts),
                "expected_count": 19,
                "skewed_count": sum(
                    item["skew"]["effective_deg"] != 0.0 for item in conductor_receipts
                ),
                "conductors": conductor_receipts,
            },
            "bcoil": {
                "acceptance": "accepted_unmodified",
                "input_column": bcoil.input_column,
                "axisymmetric_poloidal_section": None,
                "machine_section_route": (
                    "not_applicable: the dataset declares no axisymmetric poloidal "
                    "section"
                ),
                "provenance": [asdict(item) for item in bcoil.receipts],
            },
            "thomson_chords": {
                "acceptance": "accepted_after_named_change",
                "change": (
                    "embed released poloidal R,Z coordinates at representative phi "
                    "zero; retain line-of-sight endpoints as absent"
                ),
                "count": len(sightlines),
                "records": sightlines,
                "provenance": asdict(description.receipts[0]),
            },
            "efit_grid": {
                "acceptance": "accepted_unmodified",
                "shape": [len(description.grid_z), len(description.grid_r)],
                "r_extent_m": [min(description.grid_r), max(description.grid_r)],
                "z_extent_m": [min(description.grid_z), max(description.grid_z)],
                "provenance": asdict(description.receipts[1]),
            },
            "wall_contour": {
                "acceptance": "absent",
                "value": None,
                "external_source": None,
                "reason": (
                    "not shipped by the competition dataset and no external source "
                    "is currently authorised"
                ),
                "provenance": asdict(description.receipts[2]),
            },
            "passive_structure": {
                "acceptance": "absent",
                "value": None,
                "external_source": None,
                "reason": (
                    "not shipped by the competition dataset and no external source "
                    "is currently authorised"
                ),
                "provenance": asdict(description.receipts[3]),
            },
        },
    }
    if len(conductor_receipts) != 19 or maximum_difference > 1.0e-12:
        raise RuntimeError(
            "the complete conductor set did not survive the machine seam"
        )
    return receipt, description


def write_figures(
    description: DiiidDatasetMachineDescription, output: Path
) -> list[Path]:
    """Draw the routed conductors, diagnostic coordinates, and skew details."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch
    from matplotlib.patches import Rectangle

    output.mkdir(parents=True, exist_ok=True)
    sections = description.machine.active_sections
    skewed_names = {
        conductor.name
        for conductor in description.physical.conductors
        if conductor.vertices is not None
        and not np.allclose(
            conductor.vertices,
            np.asarray(
                [
                    [
                        np.min(conductor.vertices[:, 0]),
                        np.min(conductor.vertices[:, 1]),
                    ],
                    [
                        np.max(conductor.vertices[:, 0]),
                        np.min(conductor.vertices[:, 1]),
                    ],
                    [
                        np.max(conductor.vertices[:, 0]),
                        np.max(conductor.vertices[:, 1]),
                    ],
                    [
                        np.min(conductor.vertices[:, 0]),
                        np.max(conductor.vertices[:, 1]),
                    ],
                ]
            ),
        )
    }
    paths = []

    figure, axes = plt.subplots(figsize=(7.0, 7.2), constrained_layout=True)
    for section in sections:
        vertices = _routed_vertices(section)
        skewed = section.name in skewed_names
        axes.add_patch(
            PolygonPatch(
                vertices,
                facecolor="tab:red" if skewed else "tab:blue",
                edgecolor="black",
                alpha=0.42,
                linewidth=0.7,
            )
        )
        centre = np.mean(vertices, axis=0)
        axes.text(centre[0], centre[1], section.name, fontsize=7, ha="center")
    axes.set_aspect("equal")
    axes.autoscale_view()
    axes.set_xlabel("R [m]")
    axes.set_ylabel("Z [m]")
    axes.set_title("All 19 dataset-routed poloidal conductor sections")
    path = output / "conductor_sections.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(figsize=(7.0, 6.4), constrained_layout=True)
    r_min, r_max = min(description.grid_r), max(description.grid_r)
    z_min, z_max = min(description.grid_z), max(description.grid_z)
    axes.add_patch(
        Rectangle(
            (r_min, z_min),
            r_max - r_min,
            z_max - z_min,
            fill=False,
            edgecolor="black",
            linewidth=1.4,
        )
    )
    positions = np.asarray(
        [sightline.position[:2] for sightline in description.machine.sightlines]
    )
    axes.scatter(positions[:, 0], positions[:, 1], s=14, color="tab:orange")
    axes.text(r_min, z_max, "65×65 EFIT grid extent", va="bottom", fontsize=8)
    axes.set_xlim(r_min - 0.08, r_max + 0.08)
    axes.set_ylim(z_min - 0.12, z_max + 0.12)
    axes.set_aspect("equal")
    axes.set_xlabel("R [m]")
    axes.set_ylabel("Z [m]")
    axes.set_title(
        f"{len(description.machine.sightlines)} Thomson coordinates on the EFIT grid"
    )
    path = output / "thomson_grid_extent.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    paths.append(path)

    skewed = [section for section in sections if section.name in skewed_names]
    figure, axes_grid = plt.subplots(2, 3, figsize=(10.0, 6.4), constrained_layout=True)
    for axes, section in zip(axes_grid.flat, skewed, strict=True):
        vertices = _routed_vertices(section)
        axes.add_patch(
            PolygonPatch(
                vertices,
                facecolor="tab:red",
                edgecolor="black",
                alpha=0.55,
            )
        )
        margin = 0.12 * max(np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1]))
        axes.set_xlim(np.min(vertices[:, 0]) - margin, np.max(vertices[:, 0]) + margin)
        axes.set_ylim(np.min(vertices[:, 1]) - margin, np.max(vertices[:, 1]) + margin)
        axes.set_aspect("equal")
        axes.set_title(section.name)
        axes.set_xlabel("R [m]")
        axes.set_ylabel("Z [m]")
    figure.suptitle("Six shipped skewed F-coil parallelograms")
    path = output / "skewed_conductor_sections.png"
    figure.savefig(path, dpi=200)
    plt.close(figure)
    paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--shot", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    shot = args.shot
    if shot is None:
        try:
            shot = next(iter(sorted(args.data.glob("*.parquet"))))
        except StopIteration:
            parser.error(f"no parquet files found under {args.data}")
    row = read_row(shot)
    receipt, description = build_receipt(row, source_row=str(shot))
    figures = write_figures(description, args.output)
    receipt["figures"] = [str(path) for path in figures]
    args.output.mkdir(parents=True, exist_ok=True)
    receipt_path = args.output / "machine_description_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "conductors": receipt["quantities"]["poloidal_conductors"]["count"],
                "thomson_chords": receipt["quantities"]["thomson_chords"]["count"],
                "grid_shape": receipt["quantities"]["efit_grid"]["shape"],
                "figures": len(figures),
            }
        )
    )


if __name__ == "__main__":
    main()
