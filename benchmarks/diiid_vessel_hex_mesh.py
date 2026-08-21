"""Bank a clipped hexagonal discretisation of the DIII-D limiter interior.

The interaction matrix is the reciprocal, section-averaged poloidal-flux
coupling between uniformly-current-filled mesh cells. Source integration uses
Nova's closed polygon kernel; positive quadrature averages over each target
cell. Boundary cells retain the exact clipped polygon rather than a bounding
hexagon or point filament.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from nova.biot.sectionaverage import averaged_greens


DEFAULT_INPUT = Path(
    os.environ.get(
        "NOVA_DIIID_MACHINE_DESCRIPTION",
        "docs/figures/diiid-forward-onboarding/ids-set/diiid_machine_description.nc",
    )
)
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/vessel-mesh")
DATA_DICTIONARY = "4.1.1"
HEX_CIRCUMRADIUS_M = 0.12
TARGET_QUADRATURE_ORDER = 2
AREA_RELATIVE_TOLERANCE = 1.0e-4
AREA_FLOOR_M2 = 1.0e-12


@dataclass(frozen=True)
class VesselMesh:
    """Clipped polygon cells and their measured geometry."""

    cells: tuple[np.ndarray, ...]
    centres_rz_m: np.ndarray
    areas_m2: np.ndarray
    raw_polygon_area_m2: float
    valid_material_area_m2: float
    topology_component_count: int

    @property
    def characteristic_cell_size_m(self) -> float:
        """Return the centre pitch, equal to a complete hexagon's flat width."""
        return float(np.sqrt(3.0) * HEX_CIRCUMRADIUS_M)


def read_limiter_contour(
    path: Path = DEFAULT_INPUT, *, data_dictionary: str = DATA_DICTIONARY
) -> np.ndarray:
    """Read the authoritative limiter outline through imas-python."""
    import imas

    with imas.DBEntry(path, "r", dd_version=data_dictionary) as database:
        wall = database.get("wall", 0, autoconvert=False)
        limiter = wall.description_2d[0].limiter.unit[0].outline
        contour = np.column_stack(
            [
                np.asarray(limiter.r, dtype=np.float64),
                np.asarray(limiter.z, dtype=np.float64),
            ]
        )
    if contour.ndim != 2 or contour.shape[1] != 2 or len(contour) < 4:
        raise ValueError("wall limiter must contain an R-Z polygon")
    if not np.all(np.isfinite(contour)):
        raise ValueError("wall limiter contains non-finite coordinates")
    return contour


def _shoelace_area(vertices: np.ndarray) -> float:
    """Return the absolute area of a possibly self-crossing closed chain."""
    radial = vertices[:, 0]
    vertical = vertices[:, 1]
    return float(
        0.5 * abs(radial @ np.roll(vertical, -1) - vertical @ np.roll(radial, -1))
    )


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    """Return all positive polygon members from a Shapely result."""
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if hasattr(geometry, "geoms"):
        return [part for member in geometry.geoms for part in _polygon_parts(member)]
    return []


def hex_mesh(
    contour: np.ndarray, *, circumradius_m: float = HEX_CIRCUMRADIUS_M
) -> VesselMesh:
    """Tile the validity-repaired contour with exact clipped hexagonal cells."""
    if not np.isfinite(circumradius_m) or circumradius_m <= 0.0:
        raise ValueError("hexagon circumradius must be finite and positive")
    vertices = np.asarray(contour, dtype=np.float64)
    if np.array_equal(vertices[0], vertices[-1]):
        vertices = vertices[:-1]
    raw_polygon_area = _shoelace_area(vertices)
    material = shapely.make_valid(Polygon(vertices))
    material_parts = _polygon_parts(material)
    if not material_parts:
        raise ValueError("wall limiter does not enclose positive material")

    radial_pitch = np.sqrt(3.0) * circumradius_m
    vertical_pitch = 1.5 * circumradius_m
    angle = np.pi / 6.0 + np.arange(6) * np.pi / 3.0
    unit_vertices = np.column_stack([np.cos(angle), np.sin(angle)])
    minimum_r, minimum_z, maximum_r, maximum_z = material.bounds
    first_row = int(np.floor((minimum_z - circumradius_m) / vertical_pitch))
    last_row = int(np.ceil((maximum_z + circumradius_m) / vertical_pitch))

    polygons: list[Polygon] = []
    for row in range(first_row, last_row + 1):
        offset = 0.5 * (row & 1)
        first_column = int(
            np.floor((minimum_r - circumradius_m) / radial_pitch - offset)
        )
        last_column = int(np.ceil((maximum_r + circumradius_m) / radial_pitch - offset))
        for column in range(first_column, last_column + 1):
            centre = np.array([radial_pitch * (column + offset), vertical_pitch * row])
            candidate = Polygon(centre + circumradius_m * unit_vertices)
            polygons.extend(
                part
                for part in _polygon_parts(material.intersection(candidate))
                if part.area > AREA_FLOOR_M2
            )

    cells = tuple(
        np.asarray(polygon.exterior.coords, dtype=np.float64)[:-1, :2]
        for polygon in polygons
    )
    areas = np.asarray([polygon.area for polygon in polygons], dtype=np.float64)
    centres = np.asarray(
        [[polygon.centroid.x, polygon.centroid.y] for polygon in polygons],
        dtype=np.float64,
    )
    return VesselMesh(
        cells=cells,
        centres_rz_m=centres,
        areas_m2=areas,
        raw_polygon_area_m2=raw_polygon_area,
        valid_material_area_m2=float(material.area),
        topology_component_count=len(material_parts),
    )


def assemble_self_interaction(
    cells: tuple[np.ndarray, ...], *, order: int = TARGET_QUADRATURE_ORDER
) -> tuple[np.ndarray, float]:
    """Return the reciprocal cell-to-cell flux operator and raw asymmetry."""
    raw = np.column_stack(
        [averaged_greens(list(cells), source, order=order)[0] for source in cells]
    )
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("self-interaction assembly produced non-finite values")
    maximum_raw_asymmetry = float(np.max(np.abs(raw - raw.T)))
    reciprocal = 0.5 * (raw + raw.T)
    return reciprocal, maximum_raw_asymmetry


def _padded_cell_vertices(
    cells: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Return numeric padded vertices and per-cell vertex counts."""
    count = np.asarray([len(cell) for cell in cells], dtype=np.int64)
    vertices = np.full((len(cells), int(count.max()), 2), np.nan, dtype=np.float64)
    for index, cell in enumerate(cells):
        vertices[index, : len(cell)] = cell
    return vertices, count


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_preregistration(output: Path) -> Path:
    """Write the acceptance rule before any mesh score is computed."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "vessel_hex_mesh_preregistration.json"
    record = {
        "measurement": "DIII-D limiter hex-mesh and self-interaction operator",
        "area_acceptance": {
            "reference": "absolute shoelace area of the authored limiter chain",
            "relative_tolerance": AREA_RELATIVE_TOLERANCE,
            "rule": "abs(meshed_area / contour_polygon_area - 1) <= tolerance",
        },
        "mesh": {
            "hexagon_circumradius_m": HEX_CIRCUMRADIUS_M,
            "characteristic_cell_size_m": np.sqrt(3.0) * HEX_CIRCUMRADIUS_M,
            "boundary_treatment": "exact polygon intersection after validity repair",
        },
        "operator": {
            "quantity": "section-averaged total poloidal flux per ampere",
            "source_route": "nova.biot.polygonanalytic.polygon_analytic_greens",
            "target_quadrature_order": TARGET_QUADRATURE_ORDER,
            "reciprocity": "arithmetic symmetrisation after raw-asymmetry measurement",
        },
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return path


def write_operator(
    path: Path,
    contour: np.ndarray,
    mesh: VesselMesh,
    operator: np.ndarray,
) -> str:
    """Persist the interaction operator and mesh coordinates, returning its digest."""
    padded_vertices, vertex_count = _padded_cell_vertices(mesh.cells)
    np.savez_compressed(
        path,
        self_interaction_wb_per_a=operator,
        cell_centres_rz_m=mesh.centres_rz_m,
        cell_areas_m2=mesh.areas_m2,
        cell_vertices_rz_m=padded_vertices,
        cell_vertex_count=vertex_count,
        limiter_contour_rz_m=contour,
    )
    return _sha256(path)


def write_figure(path: Path, contour: np.ndarray, mesh: VesselMesh) -> None:
    """Draw the clipped hex mesh over the authored limiter chain."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as PolygonPatch

    figure, axis = plt.subplots(figsize=(7.2, 8.0), constrained_layout=True)
    patches = [PolygonPatch(cell, closed=True) for cell in mesh.cells]
    collection = PatchCollection(
        patches,
        facecolor="tab:cyan",
        edgecolor="0.25",
        linewidth=0.35,
        alpha=0.56,
    )
    axis.add_collection(collection)
    axis.plot(contour[:, 0], contour[:, 1], color="black", linewidth=1.5, label="wall")
    axis.set(
        xlabel="R [m]",
        ylabel="Z [m]",
        title=(
            "DIII-D limiter interior: clipped hex mesh\n"
            f"{len(mesh.cells)} cells, characteristic size "
            f"{mesh.characteristic_cell_size_m:.4f} m"
        ),
        aspect="equal",
    )
    axis.autoscale_view()
    axis.legend(loc="upper right")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def build_receipt(
    input_path: Path,
    contour: np.ndarray,
    mesh: VesselMesh,
    operator: np.ndarray,
    raw_asymmetry: float,
    operator_path: Path,
    operator_sha256: str,
    figure_path: Path,
    preregistration_path: Path,
) -> dict[str, Any]:
    """Return the complete scored geometry and conditioning receipt."""
    meshed_area = float(mesh.areas_m2.sum())
    relative_area_error = float(abs(meshed_area / mesh.raw_polygon_area_m2 - 1.0))
    diagonal = np.abs(np.diag(operator))
    off_diagonal = np.sum(np.abs(operator), axis=1) - diagonal
    dominance = diagonal / off_diagonal
    maximum_asymmetry = float(np.max(np.abs(operator - operator.T)))
    receipt = {
        "measurement": "DIII-D vessel hex mesh self-interaction",
        "source": {
            "path": str(input_path),
            "data_dictionary": DATA_DICTIONARY,
            "autoconvert": False,
            "limiter_vertex_count": int(len(contour)),
            "unique_limiter_vertex_count": int(
                len(contour) - int(np.array_equal(contour[0], contour[-1]))
            ),
            "r_extent_m": [float(contour[:, 0].min()), float(contour[:, 0].max())],
            "z_extent_m": [float(contour[:, 1].min()), float(contour[:, 1].max())],
        },
        "preregistration": {
            "path": str(preregistration_path),
            "area_relative_tolerance": AREA_RELATIVE_TOLERANCE,
        },
        "mesh": {
            "cell_count": len(mesh.cells),
            "hexagon_circumradius_m": HEX_CIRCUMRADIUS_M,
            "characteristic_cell_size_m": mesh.characteristic_cell_size_m,
            "boundary_treatment": "exact clipped polygons",
            "topology_repair": {
                "required": mesh.topology_component_count > 1,
                "component_count": mesh.topology_component_count,
                "raw_polygon_area_m2": mesh.raw_polygon_area_m2,
                "valid_material_area_m2": mesh.valid_material_area_m2,
                "relative_area_change": abs(
                    mesh.valid_material_area_m2 / mesh.raw_polygon_area_m2 - 1.0
                ),
            },
            "area_score": {
                "contour_polygon_area_m2": mesh.raw_polygon_area_m2,
                "meshed_area_m2": meshed_area,
                "relative_error": relative_area_error,
                "relative_tolerance": AREA_RELATIVE_TOLERANCE,
                "passed": relative_area_error <= AREA_RELATIVE_TOLERANCE,
            },
        },
        "self_interaction": {
            "quantity": "section-averaged total poloidal flux per ampere",
            "units": "Wb/A",
            "matrix_shape": list(operator.shape),
            "maximum_asymmetry_wb_per_a": maximum_asymmetry,
            "raw_maximum_asymmetry_wb_per_a": raw_asymmetry,
            "condition_number_2": float(np.linalg.cond(operator)),
            "diagonal_dominance_ratio": float(np.min(dominance)),
            "diagonal_dominance_definition": (
                "minimum row abs(diagonal) / sum(abs(off-diagonal))"
            ),
            "target_quadrature_order": TARGET_QUADRATURE_ORDER,
            "reciprocity_enforced": True,
        },
        "artifacts": {
            "operator_npz": str(operator_path),
            "operator_sha256": operator_sha256,
            "mesh_figure": str(figure_path),
        },
    }
    if not receipt["mesh"]["area_score"]["passed"]:
        raise RuntimeError(
            "meshed area misses the preregistered contour-area tolerance"
        )
    if operator.shape != (len(mesh.cells), len(mesh.cells)):
        raise RuntimeError("self-interaction matrix does not cover every mesh cell")
    if maximum_asymmetry != 0.0:
        raise RuntimeError("persisted reciprocal operator is not exactly symmetric")
    return receipt


def run(
    input_path: Path = DEFAULT_INPUT,
    output: Path = DEFAULT_OUTPUT,
    *,
    data_dictionary: str = DATA_DICTIONARY,
) -> dict[str, Any]:
    """Build, persist, score, and illustrate the vessel operator."""
    preregistration_path = write_preregistration(output)
    contour = read_limiter_contour(input_path, data_dictionary=data_dictionary)
    mesh = hex_mesh(contour)
    operator, raw_asymmetry = assemble_self_interaction(mesh.cells)
    operator_path = output / "diiid_vessel_self_interaction.npz"
    figure_path = output / "diiid_vessel_hex_mesh.png"
    receipt_path = output / "diiid_vessel_hex_mesh_receipt.json"
    operator_sha256 = write_operator(operator_path, contour, mesh, operator)
    write_figure(figure_path, contour, mesh)
    receipt = build_receipt(
        input_path,
        contour,
        mesh,
        operator,
        raw_asymmetry,
        operator_path,
        operator_sha256,
        figure_path,
        preregistration_path,
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main() -> None:
    """Run the banked DIII-D vessel measurement."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dictionary", default=DATA_DICTIONARY)
    arguments = parser.parse_args()
    receipt = run(
        arguments.input,
        arguments.output,
        data_dictionary=arguments.data_dictionary,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
