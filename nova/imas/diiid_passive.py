"""Static DIII-D passive description on wall-following polygon loops.

The loops form a geometric basis for later eigenmode reduction.  Their currents
are not independently recoverable, and this module does not define a dynamic
vessel-current model.  The limiter authority is the repaired 82-point contour
banked by the exact-tare work.  Its largest repaired component is the physical
wall; the tiny secondary component created at the self-intersection is recorded
as excluded topology-repair residue.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from nova.biot.sectionaverage import averaged_greens


SOURCE_DIRECTORY = Path("docs/figures/diiid-forward-onboarding/vessel-mesh")
SOURCE_MESH_RECEIPT = SOURCE_DIRECTORY / "diiid_vessel_hex_mesh_receipt.json"
SOURCE_OPERATOR = SOURCE_DIRECTORY / "diiid_vessel_self_interaction.npz"
EXACT_TARE_RECEIPT = Path(
    "docs/figures/diiid-forward-onboarding/exact-tare/exact_clipped_tare_receipt.json"
)
OUTPUT_DIRECTORY = Path("docs/figures/coil-circuit-discovery")
DEFAULT_OUTPUT_RECEIPT = OUTPUT_DIRECTORY / "pf_passive_description_receipt.json"
DEFAULT_OUTPUT_FIGURE = OUTPUT_DIRECTORY / "pf_passive_vessel_mesh.png"
DEFAULT_OUTPUT_OPERATOR = OUTPUT_DIRECTORY / "pf_passive_slender_operator.npz"

DATA_DICTIONARY = "4.1.1"
INCONEL_625_RESISTIVITY_OHM_M = 1.32e-6
STRUCTURAL_THICKNESS_RANGE_M = (0.025, 0.038)
THINCURR_EFFECTIVE_THICKNESS_RANGE_M = (0.010, 0.025)
NOMINAL_STRUCTURAL_THICKNESS_M = math.sqrt(
    STRUCTURAL_THICKNESS_RANGE_M[0] * STRUCTURAL_THICKNESS_RANGE_M[1]
)
LOOP_COUNT = 48
MINIMUM_SLENDERNESS = 5.0
AREA_RELATIVE_TOLERANCE = 1.0e-4

THINCURR_SOURCE = "https://arxiv.org/abs/2309.15336"
VESSEL_DESIGN_SOURCE = (
    "https://repository.lib.ncsu.edu/bitstreams/"
    "9a7f276d-bf86-4139-94c2-12d5b1310638/download"
)


class DiiidPassiveError(ValueError):
    """Raised when a passive-description authority violates an invariant."""


@dataclass(frozen=True)
class PassiveLoop:
    """One wall-following toroidal loop with a slender poloidal section."""

    name: str
    outline_rz_m: np.ndarray
    area_m2: float
    centroid_rz_m: tuple[float, float]
    poloidal_length_m: float
    through_thickness_m: float
    aspect_ratio: float
    resistance_ohm: float
    resistance_lower_ohm: float
    resistance_upper_ohm: float
    thincurr_resistance_lower_ohm: float
    thincurr_resistance_upper_ohm: float

    def validate(self) -> None:
        outline = np.asarray(self.outline_rz_m, dtype=float)
        if outline.ndim != 2 or outline.shape[1] != 2 or len(outline) < 3:
            raise DiiidPassiveError(f"{self.name} has no polygon outline")
        if not np.all(np.isfinite(outline)):
            raise DiiidPassiveError(f"{self.name} outline is non-finite")
        if self.area_m2 <= 0.0 or self.poloidal_length_m <= 0.0:
            raise DiiidPassiveError(f"{self.name} has non-positive geometry")
        if self.through_thickness_m != NOMINAL_STRUCTURAL_THICKNESS_M:
            raise DiiidPassiveError(
                f"{self.name} does not use the structural wall depth"
            )
        if self.aspect_ratio < MINIMUM_SLENDERNESS:
            raise DiiidPassiveError(f"{self.name} is not slender")
        if not (
            0.0
            < self.resistance_lower_ohm
            <= self.resistance_ohm
            <= self.resistance_upper_ohm
        ):
            raise DiiidPassiveError(f"{self.name} has invalid structural resistance")
        if not (
            self.resistance_lower_ohm
            < self.thincurr_resistance_lower_ohm
            <= self.thincurr_resistance_upper_ohm
        ):
            raise DiiidPassiveError(f"{self.name} has invalid effective resistance")


@dataclass(frozen=True)
class DiiidPassiveDescription:
    """Authored ``pf_passive`` IDS and its reciprocal interaction authority."""

    pf_passive: Any
    loops: tuple[PassiveLoop, ...]
    self_inductance_operator_wb_per_a: np.ndarray
    limiter_contour_rz_m: np.ndarray
    wall_band_area_m2: float
    area_relative_error: float
    excluded_repair_area_m2: float
    raw_operator_asymmetry_wb_per_a: float
    source_receipt: dict[str, Any]
    operator_path: Path
    operator_sha256: str

    def validate(self) -> None:
        if len(self.loops) != LOOP_COUNT or len(self.pf_passive.loop) != LOOP_COUNT:
            raise DiiidPassiveError(f"the description must contain {LOOP_COUNT} loops")
        for loop in self.loops:
            loop.validate()
        operator = np.asarray(self.self_inductance_operator_wb_per_a, dtype=float)
        if operator.shape != (LOOP_COUNT, LOOP_COUNT):
            raise DiiidPassiveError("the interaction operator has the wrong shape")
        if not np.all(np.isfinite(operator)) or not np.array_equal(
            operator, operator.T
        ):
            raise DiiidPassiveError(
                "the interaction operator is not exactly reciprocal"
            )
        if self.area_relative_error > AREA_RELATIVE_TOLERANCE:
            raise DiiidPassiveError(
                "wall-band polygons do not close the reference area"
            )
        if np.asarray(self.limiter_contour_rz_m).shape != (82, 2):
            raise DiiidPassiveError("the limiter authority must contain 82 points")
        self.pf_passive.validate()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resistance(radius_m: float, length_m: float, thickness_m: float) -> float:
    """Return toroidal resistance for a poloidal-length by thickness section."""

    return float(
        INCONEL_625_RESISTIVITY_OHM_M
        * 2.0
        * np.pi
        * radius_m
        / (length_m * thickness_m)
    )


def _wall_polygons() -> tuple[tuple[PassiveLoop, ...], np.ndarray, float, float, float]:
    """Partition the physical limiter wall band by equal-arclength generators."""

    from shapely import make_valid, voronoi_polygons
    from shapely.geometry import MultiPoint, Polygon

    with np.load(SOURCE_OPERATOR, allow_pickle=False) as bank:
        limiter = np.asarray(bank["limiter_contour_rz_m"], dtype=float).copy()
    repaired = make_valid(Polygon(limiter[:-1]))
    parts = sorted(
        repaired.geoms if hasattr(repaired, "geoms") else (repaired,),
        key=lambda polygon: polygon.area,
        reverse=True,
    )
    physical = parts[0]
    excluded_area = float(sum(polygon.area for polygon in parts[1:]))
    inner = physical.buffer(-NOMINAL_STRUCTURAL_THICKNESS_M, join_style="mitre")
    band = physical.difference(inner)
    perimeter = float(physical.exterior.length)
    segment_length = perimeter / LOOP_COUNT
    if segment_length / NOMINAL_STRUCTURAL_THICKNESS_M < MINIMUM_SLENDERNESS:
        raise DiiidPassiveError("the segmentation rule violates slenderness")

    generators = [
        physical.exterior.interpolate((index + 0.5) * segment_length)
        for index in range(LOOP_COUNT)
    ]
    regions = voronoi_polygons(
        MultiPoint(generators), extend_to=physical.envelope, ordered=True
    )
    loops: list[PassiveLoop] = []
    meshed_area = 0.0
    for index, region in enumerate(regions.geoms):
        intersection = region.intersection(band)
        polygons = (
            list(intersection.geoms)
            if hasattr(intersection, "geoms")
            else [intersection]
        )
        polygons = [
            polygon
            for polygon in polygons
            if polygon.geom_type == "Polygon" and polygon.area > 1.0e-12
        ]
        polygon = max(polygons, key=lambda candidate: candidate.area)
        outline = np.asarray(polygon.exterior.coords, dtype=float)[:-1, :2]
        radius = float(polygon.centroid.x)
        structural_lower, structural_upper = STRUCTURAL_THICKNESS_RANGE_M
        effective_lower, effective_upper = THINCURR_EFFECTIVE_THICKNESS_RANGE_M
        loops.append(
            PassiveLoop(
                name=f"DIIID_VESSEL_WALL_{index:03d}",
                outline_rz_m=outline,
                area_m2=float(polygon.area),
                centroid_rz_m=(radius, float(polygon.centroid.y)),
                poloidal_length_m=segment_length,
                through_thickness_m=NOMINAL_STRUCTURAL_THICKNESS_M,
                aspect_ratio=segment_length / NOMINAL_STRUCTURAL_THICKNESS_M,
                resistance_ohm=_resistance(
                    radius, segment_length, NOMINAL_STRUCTURAL_THICKNESS_M
                ),
                resistance_lower_ohm=_resistance(
                    radius, segment_length, structural_upper
                ),
                resistance_upper_ohm=_resistance(
                    radius, segment_length, structural_lower
                ),
                thincurr_resistance_lower_ohm=_resistance(
                    radius, segment_length, effective_upper
                ),
                thincurr_resistance_upper_ohm=_resistance(
                    radius, segment_length, effective_lower
                ),
            )
        )
        meshed_area += float(polygon.area)
    relative_error = abs(meshed_area / float(band.area) - 1.0)
    return tuple(loops), limiter, float(band.area), relative_error, excluded_area


def _author_ids(loops: tuple[PassiveLoop, ...]) -> Any:
    import imas

    ids = imas.IDSFactory(DATA_DICTIONARY).new("pf_passive")
    ids.ids_properties.homogeneous_time = 0
    ids.loop.resize(len(loops))
    for target, source in zip(ids.loop, loops, strict=True):
        target.name = source.name
        target.description = (
            "toroidal passive basis loop with a wall-following slender polygon "
            "section; its current is not independently observable"
        )
        target.resistivity = INCONEL_625_RESISTIVITY_OHM_M
        target.resistance = source.resistance_ohm
        target.resistance_error_lower = (
            source.resistance_ohm - source.resistance_lower_ohm
        )
        target.resistance_error_upper = (
            source.resistance_upper_ohm - source.resistance_ohm
        )
        target.element.resize(1)
        element = target.element[0]
        element.name = f"{source.name}_section"
        element.description = "wall-band polygon from equal-arclength segmentation"
        element.area = source.area_m2
        element.turns_with_sign = 1.0
        element.geometry.geometry_type = 1
        element.geometry.outline.r = source.outline_rz_m[:, 0]
        element.geometry.outline.z = source.outline_rz_m[:, 1]
    return ids


def assemble_self_interaction(
    loops: tuple[PassiveLoop, ...], *, order: int = 2
) -> tuple[np.ndarray, float]:
    """Assemble and reciprocalise exact-polygon section couplings."""

    sections = [loop.outline_rz_m for loop in loops]
    raw = np.column_stack(
        [averaged_greens(sections, source, order=order)[0] for source in sections]
    )
    if not np.all(np.isfinite(raw)):
        raise DiiidPassiveError("self-interaction assembly is non-finite")
    asymmetry = float(np.max(np.abs(raw - raw.T)))
    reciprocal = 0.5 * (raw + raw.T)
    if not np.array_equal(reciprocal, reciprocal.T):
        raise DiiidPassiveError("reciprocalisation did not produce exact symmetry")
    return reciprocal, asymmetry


def _padded_vertices(loops: tuple[PassiveLoop, ...]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([len(loop.outline_rz_m) for loop in loops], dtype=np.int64)
    vertices = np.full((len(loops), int(counts.max()), 2), np.nan, dtype=float)
    for index, loop in enumerate(loops):
        vertices[index, : counts[index]] = loop.outline_rz_m
    return vertices, counts


def _write_operator(
    path: Path,
    loops: tuple[PassiveLoop, ...],
    limiter: np.ndarray,
    operator: np.ndarray,
    raw_asymmetry: float,
    wall_band_area: float,
    area_error: float,
    excluded_area: float,
) -> None:
    vertices, counts = _padded_vertices(loops)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        self_interaction_wb_per_a=operator,
        cell_vertices_rz_m=vertices,
        cell_vertex_count=counts,
        cell_centres_rz_m=np.asarray([loop.centroid_rz_m for loop in loops]),
        cell_areas_m2=np.asarray([loop.area_m2 for loop in loops]),
        poloidal_lengths_m=np.asarray([loop.poloidal_length_m for loop in loops]),
        through_thickness_m=np.asarray([loop.through_thickness_m for loop in loops]),
        aspect_ratios=np.asarray([loop.aspect_ratio for loop in loops]),
        limiter_contour_rz_m=limiter,
        wall_band_area_m2=np.asarray(wall_band_area),
        wall_band_area_relative_error=np.asarray(area_error),
        excluded_repair_area_m2=np.asarray(excluded_area),
        raw_maximum_asymmetry_wb_per_a=np.asarray(raw_asymmetry),
    )


def build_description(
    operator_path: Path = DEFAULT_OUTPUT_OPERATOR,
) -> DiiidPassiveDescription:
    """Build the static description from the banked slender-loop operator."""

    loops, limiter, wall_area, area_error, excluded_area = _wall_polygons()
    source_receipt = json.loads(SOURCE_MESH_RECEIPT.read_text())
    with np.load(operator_path, allow_pickle=False) as bank:
        operator = np.asarray(bank["self_interaction_wb_per_a"], dtype=float).copy()
        raw_asymmetry = float(bank["raw_maximum_asymmetry_wb_per_a"])
    description = DiiidPassiveDescription(
        pf_passive=_author_ids(loops),
        loops=loops,
        self_inductance_operator_wb_per_a=operator,
        limiter_contour_rz_m=limiter,
        wall_band_area_m2=wall_area,
        area_relative_error=area_error,
        excluded_repair_area_m2=excluded_area,
        raw_operator_asymmetry_wb_per_a=raw_asymmetry,
        source_receipt=source_receipt,
        operator_path=operator_path,
        operator_sha256=_sha256(operator_path),
    )
    description.validate()
    return description


def _distribution(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def build_receipt(description: DiiidPassiveDescription) -> dict[str, Any]:
    """Return geometry, material, operator, and qualification evidence."""

    operator = description.self_inductance_operator_wb_per_a
    diagonal = np.abs(np.diag(operator))
    off_diagonal = np.sum(np.abs(operator), axis=1) - diagonal
    resistance = [loop.resistance_ohm for loop in description.loops]
    lower = [loop.resistance_lower_ohm for loop in description.loops]
    upper = [loop.resistance_upper_ohm for loop in description.loops]
    effective_lower = [loop.thincurr_resistance_lower_ohm for loop in description.loops]
    effective_upper = [loop.thincurr_resistance_upper_ohm for loop in description.loops]
    return {
        "measurement": "DIII-D static pf_passive wall-loop description",
        "scope": {
            "description_only": True,
            "dynamic_vessel_current_model_claimed": False,
            "basis_interpretation": (
                f"{LOOP_COUNT} geometric wall loops require eigenmode reduction; "
                "individual loop currents are not independently recoverable"
            ),
        },
        "provenance": {
            "banked_vessel_mesh_receipt": str(SOURCE_MESH_RECEIPT),
            "banked_vessel_mesh_receipt_sha256": _sha256(SOURCE_MESH_RECEIPT),
            "banked_limiter_and_hex_operator": str(SOURCE_OPERATOR),
            "banked_limiter_and_hex_operator_sha256": _sha256(SOURCE_OPERATOR),
            "exact_tare_vessel_anchor": str(EXACT_TARE_RECEIPT),
            "exact_tare_vessel_anchor_sha256": _sha256(EXACT_TARE_RECEIPT),
            "topology_repair": description.source_receipt["mesh"]["topology_repair"],
            "excluded_repair_component_area_m2": description.excluded_repair_area_m2,
            "excluded_repair_component_rule": (
                "retain the largest validity-repaired component as physical wall; "
                "bank but do not mesh the tiny self-intersection residue"
            ),
            "published_sources": {
                "vessel_design": {
                    "url": VESSEL_DESIGN_SOURCE,
                    "statement": (
                        "Inconel 625 sandwich wall with 25-38 mm structural depth"
                    ),
                },
                "thincurr": {
                    "url": THINCURR_SOURCE,
                    "statement": (
                        "1.32e-6 ohm.m resistivity and a 10-25 mm effective "
                        "electromagnetic thickness envelope"
                    ),
                },
            },
        },
        "mesh": {
            "loop_count": len(description.loops),
            "limiter_vertex_count": len(description.limiter_contour_rz_m),
            "segmentation_rule": (
                "the nominal structural wall band is assigned to 48 nearest "
                "equal-arclength exterior generators"
            ),
            "segmentation_justification": (
                "48 segments retain wall-following resolution while keeping every "
                "poloidal length at least five times the through-thickness width"
            ),
            "wall_band_reference_area_m2": description.wall_band_area_m2,
            "polygon_area_sum_m2": float(
                sum(loop.area_m2 for loop in description.loops)
            ),
            "wall_band_area_relative_error": description.area_relative_error,
            "wall_band_area_relative_tolerance": AREA_RELATIVE_TOLERANCE,
            "minimum_slenderness": MINIMUM_SLENDERNESS,
            "aspect_ratio": _distribution(
                [loop.aspect_ratio for loop in description.loops]
            ),
            "per_loop_geometry_banked": [
                "outline",
                "area",
                "centroid",
                "poloidal_length",
                "through_thickness",
                "aspect_ratio",
            ],
        },
        "material_and_resistance": {
            "material": "Inconel 625",
            "resistivity_ohm_m": INCONEL_625_RESISTIVITY_OHM_M,
            "structural_thickness_range_m": list(STRUCTURAL_THICKNESS_RANGE_M),
            "nominal_structural_thickness_m": NOMINAL_STRUCTURAL_THICKNESS_M,
            "thincurr_effective_thickness_range_m": list(
                THINCURR_EFFECTIVE_THICKNESS_RANGE_M
            ),
            "derivation": "R=rho*2*pi*centroid_R/(poloidal_length*thickness)",
            "structural_nominal_resistance_ohm": _distribution(resistance),
            "structural_lower_resistance_ohm": _distribution(lower),
            "structural_upper_resistance_ohm": _distribution(upper),
            "thincurr_effective_lower_resistance_ohm": _distribution(effective_lower),
            "thincurr_effective_upper_resistance_ohm": _distribution(effective_upper),
            "uncertainty_statement": (
                "the 25-38 mm structural depth and 10-25 mm electromagnetic "
                "envelope are distinct and are never silently equated"
            ),
        },
        "self_inductance": {
            "operator_reference": str(description.operator_path),
            "operator_sha256": description.operator_sha256,
            "kernel": "exact polygon section-averaged coupling, quadrature order 2",
            "quantity": "section-averaged total poloidal flux per ampere",
            "units": "Wb/A",
            "shape": list(operator.shape),
            "maximum_asymmetry_wb_per_a": float(np.max(np.abs(operator - operator.T))),
            "raw_maximum_asymmetry_wb_per_a": (
                description.raw_operator_asymmetry_wb_per_a
            ),
            "condition_number_2": float(np.linalg.cond(operator)),
            "diagonal_dominance_ratio": float(np.min(diagonal / off_diagonal)),
            "hex_operator_comparison": {
                "condition_number_2": 372.65522630279304,
                "diagonal_dominance_ratio": 0.029841680213628683,
            },
            "eigenmode_reduction_caveat": (
                "operator conditioning and off-diagonal coupling prohibit treating "
                "individual loop currents as independently recoverable; any dynamic "
                "model must reduce to identifiable eigenmodes"
            ),
        },
        "loops": [
            {
                "name": loop.name,
                "outline_rz_m": loop.outline_rz_m.tolist(),
                "area_m2": loop.area_m2,
                "centroid_rz_m": list(loop.centroid_rz_m),
                "poloidal_length_m": loop.poloidal_length_m,
                "through_thickness_m": loop.through_thickness_m,
                "aspect_ratio": loop.aspect_ratio,
                "resistance_ohm": loop.resistance_ohm,
                "resistance_lower_ohm": loop.resistance_lower_ohm,
                "resistance_upper_ohm": loop.resistance_upper_ohm,
                "thincurr_resistance_lower_ohm": loop.thincurr_resistance_lower_ohm,
                "thincurr_resistance_upper_ohm": loop.thincurr_resistance_upper_ohm,
            }
            for loop in description.loops
        ],
        "artifacts": {
            "receipt": str(DEFAULT_OUTPUT_RECEIPT),
            "mesh_figure": str(DEFAULT_OUTPUT_FIGURE),
            "slender_operator": str(DEFAULT_OUTPUT_OPERATOR),
        },
    }


def write_figure(path: Path, description: DiiidPassiveDescription) -> None:
    """Plot wall-following loops coloured by nominal structural resistance."""

    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Polygon as PolygonPatch

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 8.0), constrained_layout=True)
    collection = PatchCollection(
        [PolygonPatch(loop.outline_rz_m, closed=True) for loop in description.loops],
        cmap="viridis",
        norm=LogNorm(),
        edgecolor="0.2",
        linewidth=0.55,
    )
    collection.set_array(
        np.asarray([loop.resistance_ohm for loop in description.loops])
    )
    axis.add_collection(collection)
    limiter = description.limiter_contour_rz_m
    axis.plot(limiter[:, 0], limiter[:, 1], color="black", linewidth=1.1)
    axis.set(
        xlabel="R [m]",
        ylabel="Z [m]",
        title=(
            "DIII-D static passive-loop basis\n"
            f"{LOOP_COUNT} wall-following polygons; colour is resistance"
        ),
        aspect="equal",
    )
    axis.autoscale_view()
    colour = figure.colorbar(collection, ax=axis)
    colour.set_label("loop resistance [ohm]")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(
    output_receipt: Path = DEFAULT_OUTPUT_RECEIPT,
    output_figure: Path = DEFAULT_OUTPUT_FIGURE,
    output_operator: Path = DEFAULT_OUTPUT_OPERATOR,
) -> dict[str, Any]:
    """Assemble and bank the static ``pf_passive`` description evidence."""

    loops, limiter, wall_area, area_error, excluded_area = _wall_polygons()
    operator, raw_asymmetry = assemble_self_interaction(loops)
    _write_operator(
        output_operator,
        loops,
        limiter,
        operator,
        raw_asymmetry,
        wall_area,
        area_error,
        excluded_area,
    )
    description = build_description(output_operator)
    receipt = build_receipt(description)
    output_receipt.parent.mkdir(parents=True, exist_ok=True)
    write_figure(output_figure, description)
    output_receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
