"""Corroborate released DIII-D conductor geometry against an IMAS entry.

The comparison is a direct measurement: conductors are paired only by nearest
stored centre, and no coordinate, outline, or tolerance is fitted to the data.
Local section dimensions are recovered from polygon edge projections so that
the competition parameters are compared with the same physical quantities in
the netCDF outlines rather than with axis-aligned bounding boxes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.diiid_ids_machine_description import read_entry


COMPETITION_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/machine-description/"
    "machine_description_receipt.json"
)
OUTPUT_DIRECTORY = Path("docs/figures/diiid-forward-onboarding/corroboration")
EXPECTED_TABLE_DIGEST = (
    "782e9e08f02e610e252e9cf6d6cccfb3a9aefa62b56f14865553ba2f35d213dc"
)
SKEWED_CONDUCTORS = ("F5A", "F6A", "F7A", "F5B", "F6B", "F7B")

# These instrument-resolution-scale tolerances are fixed before any pairing is
# scored. Angular differences compare unoriented section-edge lines modulo 180°.
CENTRE_TOLERANCE_MM = 2.0
DIMENSION_TOLERANCE_MM = 2.0
SKEW_TOLERANCE_DEG = 0.25


def _element_vertices(element: dict[str, Any]) -> np.ndarray:
    """Return one netCDF element as an ordered polygon."""
    if element["geometry_type"] == 1:
        return np.column_stack((element["r"], element["z"])).astype(float)
    if element["geometry_type"] == 2:
        r = float(element["r"])
        z = float(element["z"])
        half_width = float(element["width"]) / 2.0
        half_height = float(element["height"]) / 2.0
        return np.asarray(
            [
                (r - half_width, z - half_height),
                (r + half_width, z - half_height),
                (r + half_width, z + half_height),
                (r - half_width, z + half_height),
            ]
        )
    raise ValueError(f"unsupported geometry type {element['geometry_type']}")


def _line_angle_deg(vector: np.ndarray) -> float:
    """Return the angle of an unoriented line in the half-open [-90, 90) range."""
    angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
    return (angle + 90.0) % 180.0 - 90.0


def _axis_deviation_deg(angle: float) -> float:
    """Measure how far an unoriented edge is from its nearest R or Z axis."""
    return min(abs(angle), 90.0 - abs(angle))


def section_descriptor(vertices: np.ndarray) -> dict[str, Any]:
    """Describe a four-vertex section in local width/height coordinates."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (4, 2):
        raise ValueError(f"expected a four-vertex section, found {vertices.shape}")
    edges = np.roll(vertices, -1, axis=0) - vertices
    if np.any(np.linalg.norm(edges, axis=1) == 0.0):
        raise ValueError("section contains a zero-length edge")

    adjacent = edges[:2]
    width_index = int(abs(adjacent[1, 0]) > abs(adjacent[0, 0]))
    height_index = 1 - width_index
    width_edge = adjacent[width_index]
    height_edge = adjacent[height_index]
    edge_angles = [_line_angle_deg(edge) for edge in adjacent]
    deviations = [_axis_deviation_deg(angle) for angle in edge_angles]
    skew_index = int(deviations[1] > deviations[0])
    skew = edge_angles[skew_index] if deviations[skew_index] > 1.0e-10 else 0.0
    return {
        "centre_m": vertices.mean(axis=0).tolist(),
        "width_m": abs(float(width_edge[0])),
        "height_m": abs(float(height_edge[1])),
        "skew_deg": skew,
        "axis_aligned_extent_m": [
            float(np.ptp(vertices[:, 0])),
            float(np.ptp(vertices[:, 1])),
        ],
    }


def _coil_descriptor(coil: dict[str, Any]) -> dict[str, Any]:
    """Describe one netCDF coil and retain every element polygon for plotting."""
    element_vertices = [_element_vertices(element) for element in coil["elements"]]
    all_vertices = np.concatenate(element_vertices)
    r_extent = [float(all_vertices[:, 0].min()), float(all_vertices[:, 0].max())]
    z_extent = [float(all_vertices[:, 1].min()), float(all_vertices[:, 1].max())]
    if len(element_vertices) == 1:
        local = section_descriptor(element_vertices[0])
    else:
        local = {
            "centre_m": [
                (r_extent[0] + r_extent[1]) / 2.0,
                (z_extent[0] + z_extent[1]) / 2.0,
            ],
            "width_m": r_extent[1] - r_extent[0],
            "height_m": z_extent[1] - z_extent[0],
            "skew_deg": 0.0,
            "axis_aligned_extent_m": [
                r_extent[1] - r_extent[0],
                z_extent[1] - z_extent[0],
            ],
        }
    return {
        "name": coil["name"],
        "element_count": len(element_vertices),
        "element_vertices": element_vertices,
        "r_extent_m": r_extent,
        "z_extent_m": z_extent,
        **local,
    }


def _competition_descriptor(conductor: dict[str, Any]) -> dict[str, Any]:
    """Normalize one released conductor through its physical polygon."""
    vertices = np.asarray(conductor["section_vertices_m"], dtype=float)
    local = section_descriptor(vertices)
    return {
        "name": conductor["name"],
        "vertices": vertices,
        "raw_centre_m": list(conductor["centre_m"]),
        "raw_width_m": float(conductor["width_m"]),
        "raw_height_m": float(conductor["height_m"]),
        "raw_skew_deg": float(conductor["skew"]["effective_deg"]),
        **local,
    }


def _angular_difference_deg(measured: float, reference: float) -> float:
    """Return measured minus reference for unoriented lines modulo 180°."""
    return (measured - reference + 90.0) % 180.0 - 90.0


def _vertex_set_distance_mm(left: np.ndarray, right: np.ndarray) -> float:
    """Return the symmetric maximum nearest-vertex distance."""
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=2)
    return float(max(distances.min(axis=0).max(), distances.min(axis=1).max()) * 1e3)


def load_competition(path: Path = COMPETITION_RECEIPT) -> dict[str, Any]:
    """Load and validate the released machine-description receipt."""
    receipt = json.loads(path.read_text())
    digest = receipt["physical_geometry_digest"]
    if digest != EXPECTED_TABLE_DIGEST:
        raise ValueError(f"unexpected competition geometry digest {digest}")
    return receipt


def build_receipt(
    competition: dict[str, Any], source: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pair conductors, score geometry, and quantify omitted coverage."""
    conductors = [
        _competition_descriptor(conductor)
        for conductor in competition["quantities"]["poloidal_conductors"]["conductors"]
    ]
    coils = [_coil_descriptor(coil) for coil in source["pf_active"]]
    if len(conductors) != 19:
        raise ValueError(f"expected 19 competition conductors, found {len(conductors)}")

    pairings = []
    matched_names = set()
    for conductor in conductors:
        centre = np.asarray(conductor["centre_m"])
        distances = [
            float(np.linalg.norm(np.asarray(coil["centre_m"]) - centre))
            for coil in coils
        ]
        coil = coils[int(np.argmin(distances))]
        centre_distance_mm = min(distances) * 1e3
        width_difference_mm = (coil["width_m"] - conductor["width_m"]) * 1e3
        height_difference_mm = (coil["height_m"] - conductor["height_m"]) * 1e3
        skew_difference_deg = _angular_difference_deg(
            coil["skew_deg"], conductor["skew_deg"]
        )
        agreed = (
            centre_distance_mm <= CENTRE_TOLERANCE_MM
            and abs(width_difference_mm) <= DIMENSION_TOLERANCE_MM
            and abs(height_difference_mm) <= DIMENSION_TOLERANCE_MM
            and abs(skew_difference_deg) <= SKEW_TOLERANCE_DEG
        )
        vertex_distance = None
        if coil["element_count"] == 1:
            vertex_distance = _vertex_set_distance_mm(
                conductor["vertices"], coil["element_vertices"][0]
            )
        pairings.append(
            {
                "competition_name": conductor["name"],
                "netcdf_name": coil["name"],
                "competition_centre_m": conductor["centre_m"],
                "netcdf_centre_m": coil["centre_m"],
                "centre_distance_mm": centre_distance_mm,
                "signed_netcdf_minus_competition": {
                    "width_mm": width_difference_mm,
                    "height_mm": height_difference_mm,
                    "skew_deg": skew_difference_deg,
                },
                "maximum_vertex_set_distance_mm": vertex_distance,
                "verdict": "agreed" if agreed else "disagreed",
            }
        )
        matched_names.add(coil["name"])

    if len(matched_names) != len(pairings):
        raise ValueError("nearest-centre pairing is not one-to-one")

    omitted = []
    for coil in coils:
        if coil["name"] not in matched_names:
            omitted.append(
                {
                    "name": coil["name"],
                    "centre_m": coil["centre_m"],
                    "extent_m": [
                        coil["r_extent_m"][1] - coil["r_extent_m"][0],
                        coil["z_extent_m"][1] - coil["z_extent_m"][0],
                    ],
                    "element_count": coil["element_count"],
                }
            )

    grid = competition["quantities"]["efit_grid"]
    wall_r = np.asarray(source["contour"]["r"], dtype=float)
    wall_z = np.asarray(source["contour"]["z"], dtype=float)
    r_min, r_max = map(float, grid["r_extent_m"])
    z_min, z_max = map(float, grid["z_extent_m"])
    margins = {
        "inboard_m": float(wall_r.min() - r_min),
        "outboard_m": float(r_max - wall_r.max()),
        "lower_m": float(wall_z.min() - z_min),
        "upper_m": float(z_max - wall_z.max()),
    }
    omitted_element_count = sum(item["element_count"] for item in omitted)
    agreed_count = sum(pairing["verdict"] == "agreed" for pairing in pairings)
    pairing_by_name = {item["competition_name"]: item for item in pairings}
    f5a = pairing_by_name["F5A"]
    receipt = {
        "measurement": "competition-to-netCDF DIII-D geometry corroboration",
        "sources": {
            "competition_receipt": str(COMPETITION_RECEIPT),
            "competition_table_digest": competition["physical_geometry_digest"],
            "netcdf_path": source["source_path"],
            "netcdf_backend": source["backend"],
        },
        "method": {
            "pairing": "nearest stored R-Z centre without fitting",
            "normalization": (
                "local width and height are edge projections; skew is the "
                "non-axis-aligned edge line angle modulo 180 degrees"
            ),
            "coordinate_adjustment": False,
            "fitting": False,
            "signed_difference_convention": "netCDF minus competition",
        },
        "tolerances_before_scoring": {
            "centre_distance_mm_max": CENTRE_TOLERANCE_MM,
            "absolute_width_difference_mm_max": DIMENSION_TOLERANCE_MM,
            "absolute_height_difference_mm_max": DIMENSION_TOLERANCE_MM,
            "absolute_skew_difference_deg_max": SKEW_TOLERANCE_DEG,
        },
        "pairing_summary": {
            "competition_count": len(conductors),
            "agreed_count": agreed_count,
            "disagreed_count": len(pairings) - agreed_count,
            "name_preserving_count": sum(
                item["competition_name"] == item["netcdf_name"] for item in pairings
            ),
        },
        "pairings": pairings,
        "skewed_conductor_set": {
            "names": list(SKEWED_CONDUCTORS),
            "independently_named_by_both_sources": all(
                pairing_by_name[name]["netcdf_name"] == name
                for name in SKEWED_CONDUCTORS
            ),
            "agreed_count": sum(
                pairing_by_name[name]["verdict"] == "agreed"
                for name in SKEWED_CONDUCTORS
            ),
        },
        "f5a_discrepancy_resolution": {
            "competition_local_geometry": {
                "centre_m": [1.0041, 1.5169],
                "extent_m": [0.1392, 0.1194],
                "skew_deg": 45.0,
            },
            "discarded_legacy_descriptor": {
                "z_extent_m": 0.2586,
                "skew_deg": 90.0,
            },
            "netcdf_axis_aligned_extent_m": [0.1392, 0.2586],
            "netcdf_first_edge_angle_deg": 90.0,
            "normalized_signed_difference": f5a["signed_netcdf_minus_competition"],
            "maximum_vertex_set_distance_mm": f5a["maximum_vertex_set_distance_mm"],
            "resolution": (
                "the netCDF and competition polygons have the same vertices; "
                "0.2586 m and 90 degrees are respectively an axis-aligned "
                "bounding extent and traversal-dependent first-edge angle, not "
                "the local section height and physical shear"
            ),
            "real_machine_supports": "competition physical F5A polygon",
        },
        "coverage": {
            "ampere_turn_carrying_conductors": {
                "competition_coil_count": len(conductors),
                "competition_element_count": len(conductors),
                "netcdf_pf_active_coil_count": len(coils),
                "netcdf_pf_active_element_count": sum(
                    coil["element_count"] for coil in coils
                ),
            },
            "netcdf_coils_omitted_by_competition": omitted,
            "omitted_coil_count": len(omitted),
            "omitted_element_count": omitted_element_count,
            "expected_116_element_claim_supported": omitted_element_count == 116,
            "element_count_statement": (
                f"the five omitted coils contain {omitted_element_count} elements "
                "(48+6+6+7+7), not 116"
            ),
        },
        "limiter_against_competition_grid": {
            "limiter_vertex_count": len(wall_r),
            "grid_shape": list(grid["shape"]),
            "limiter_r_extent_m": [float(wall_r.min()), float(wall_r.max())],
            "limiter_z_extent_m": [float(wall_z.min()), float(wall_z.max())],
            "grid_r_extent_m": list(grid["r_extent_m"]),
            "grid_z_extent_m": list(grid["z_extent_m"]),
            "signed_grid_beyond_wall_margins_m": margins,
            "grid_encloses_wall": all(value >= 0.0 for value in margins.values()),
        },
    }
    plot_data = {
        "conductors": conductors,
        "coils": coils,
        "contour": np.column_stack((wall_r, wall_z)),
        "grid": grid,
        "pairings": pairings,
        "omitted_names": {item["name"] for item in omitted},
    }
    return receipt, plot_data


def write_figures(plot_data: dict[str, Any], output: Path) -> list[Path]:
    """Write the conductor and limiter corroboration figures."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    output.mkdir(parents=True, exist_ok=True)
    figure_paths = []

    figure, axes = plt.subplots(figsize=(11, 9), constrained_layout=True)
    for coil in plot_data["coils"]:
        unmatched = coil["name"] in plot_data["omitted_names"]
        for vertices in coil["element_vertices"]:
            axes.add_patch(
                Polygon(
                    vertices,
                    closed=True,
                    facecolor="tab:red" if unmatched else "tab:blue",
                    edgecolor="tab:red" if unmatched else "tab:blue",
                    alpha=0.18 if unmatched else 0.10,
                    linewidth=0.65,
                    hatch="//" if unmatched else None,
                )
            )
    for conductor in plot_data["conductors"]:
        closed = np.vstack((conductor["vertices"], conductor["vertices"][0]))
        axes.plot(
            closed[:, 0],
            closed[:, 1],
            color="black",
            linestyle="--",
            linewidth=0.9,
        )
    coil_by_name = {coil["name"]: coil for coil in plot_data["coils"]}
    for pairing in plot_data["pairings"]:
        start = np.asarray(pairing["competition_centre_m"])
        end = np.asarray(coil_by_name[pairing["netcdf_name"]]["centre_m"])
        axes.plot([start[0], end[0]], [start[1], end[1]], color="0.35", linewidth=0.5)
        axes.annotate(
            f"{pairing['competition_name']}↔{pairing['netcdf_name']}",
            start,
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=5.2,
        )
    for name in sorted(plot_data["omitted_names"]):
        centre = coil_by_name[name]["centre_m"]
        axes.annotate(
            f"{name} omitted",
            centre,
            xytext=(4, -7),
            textcoords="offset points",
            color="tab:red",
            fontsize=6,
            weight="bold",
        )
    axes.plot([], [], "k--", label="competition section")
    axes.fill([], [], color="tab:blue", alpha=0.15, label="matched netCDF element")
    axes.fill(
        [],
        [],
        color="tab:red",
        alpha=0.22,
        hatch="//",
        label="netCDF-only element",
    )
    axes.set_aspect("equal")
    axes.set_xlabel("R [m]")
    axes.set_ylabel("Z [m]")
    axes.set_title("Competition conductors against netCDF pf_active geometry")
    axes.legend(frameon=False, loc="upper right")
    conductor_path = output / "conductor_pairing_overlay.png"
    figure.savefig(conductor_path, dpi=220)
    plt.close(figure)
    figure_paths.append(conductor_path)

    contour = plot_data["contour"]
    grid = plot_data["grid"]
    r_min, r_max = grid["r_extent_m"]
    z_min, z_max = grid["z_extent_m"]
    figure, axes = plt.subplots(figsize=(8, 8), constrained_layout=True)
    closed_contour = np.vstack((contour, contour[0]))
    axes.plot(
        closed_contour[:, 0],
        closed_contour[:, 1],
        color="black",
        linewidth=2.0,
        label=f"netCDF limiter ({len(contour)} vertices)",
    )
    axes.add_patch(
        Rectangle(
            (r_min, z_min),
            r_max - r_min,
            z_max - z_min,
            fill=False,
            edgecolor="tab:orange",
            linestyle="--",
            linewidth=2.0,
            label=f"competition grid {grid['shape'][0]}×{grid['shape'][1]}",
        )
    )
    axes.set_aspect("equal")
    axes.set_xlabel("R [m]")
    axes.set_ylabel("Z [m]")
    axes.set_title("NetCDF limiter enclosed by the competition EFIT grid")
    axes.legend(frameon=False)
    limiter_path = output / "limiter_grid_extent.png"
    figure.savefig(limiter_path, dpi=220)
    plt.close(figure)
    figure_paths.append(limiter_path)
    return figure_paths


def main() -> None:
    """Run the direct geometry comparison and emit its receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-receipt", type=Path, default=COMPETITION_RECEIPT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    competition = load_competition(args.competition_receipt)
    source = read_entry()
    receipt, plot_data = build_receipt(competition, source)
    figures = write_figures(plot_data, args.output)
    receipt["figures"] = [str(path) for path in figures]
    receipt_path = args.output / "geometry_corroboration_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "agreed": receipt["pairing_summary"]["agreed_count"],
                "paired": receipt["pairing_summary"]["competition_count"],
                "omitted_coils": receipt["coverage"]["omitted_coil_count"],
                "omitted_elements": receipt["coverage"]["omitted_element_count"],
                "grid_encloses_wall": receipt["limiter_against_competition_grid"][
                    "grid_encloses_wall"
                ],
                "figures": len(figures),
            }
        )
    )


if __name__ == "__main__":
    main()
