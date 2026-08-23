"""Author the static DIII-D machine description as IMAS IDSs.

Only static geometry is sourced from the entry. The invalid limiter chain is
repaired into one declared physical ring; dynamic actuator and diagnostic
arrays, reconstructed equilibrium content, and unsupported geometry are kept
out of the published description by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import imas
import numpy as np
from imas.dd_zip import dd_xml_versions

from nova.imas.machine import DiagnosticSightline, StaticMachineDescription


SOURCE_PATH = Path("/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc")
IDS_NAMES = ("wall", "pf_active", "magnetics")
PUBLISHED_DD_MAJOR = 4
LIMITER_CHAIN_DIGEST_ALGORITHM = (
    "sha256 of closed R-Z chain as little-endian float64 pairs"
)
LIMITER_REPAIR_METHOD = (
    "Shapely make_valid, retain the largest positive polygon, canonicalize its "
    "exterior counter-clockwise from the lexicographically smallest vertex"
)


def latest_published_dd_version(versions: Iterable[str] | None = None) -> str:
    """Return the latest installed Data Dictionary in the publication major."""

    available = dd_xml_versions() if versions is None else list(versions)
    candidates = [
        version
        for version in available
        if int(version.partition(".")[0]) == PUBLISHED_DD_MAJOR
    ]
    if not candidates:
        raise RuntimeError("no DD4 Data Dictionary is available for publication")
    return max(candidates, key=lambda value: tuple(map(int, value.split("."))))


@dataclass(frozen=True)
class AbsentQuantity:
    """A machine-description quantity that its authoritative sources lack."""

    quantity: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-ready absence declaration."""

        return {
            "quantity": self.quantity,
            "status": "declared_absent",
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LimiterRepair:
    """Provenance for the deterministic repair of the source limiter chain."""

    source_chain_sha256: str
    source_vertex_count: int
    source_chain_valid: bool
    source_chain_area_m2: float
    valid_material_area_m2: float
    valid_material_relative_area_difference: float
    validity_component_count: int
    published_ring_sha256: str
    published_vertex_count: int
    published_ring_area_m2: float
    published_ring_relative_area_difference: float
    excluded_component_area_m2: float

    def as_dict(self) -> dict[str, Any]:
        """Return the declared repair as stable, JSON-ready metadata."""

        return {
            "authority": "repaired-ring-only",
            "source_chain_digest": {
                "algorithm": LIMITER_CHAIN_DIGEST_ALGORITHM,
                "sha256": self.source_chain_sha256,
            },
            "source_vertex_count": self.source_vertex_count,
            "source_chain_valid": self.source_chain_valid,
            "source_chain_area_m2": self.source_chain_area_m2,
            "repair": {
                "method": LIMITER_REPAIR_METHOD,
                "valid_material_area_m2": self.valid_material_area_m2,
                "valid_material_relative_area_difference": (
                    self.valid_material_relative_area_difference
                ),
                "validity_component_count": self.validity_component_count,
                "published_component_rule": "largest positive polygon only",
                "published_ring_sha256": self.published_ring_sha256,
                "published_vertex_count": self.published_vertex_count,
                "published_ring_area_m2": self.published_ring_area_m2,
                "published_ring_relative_area_difference": (
                    self.published_ring_relative_area_difference
                ),
                "excluded_component_area_m2": self.excluded_component_area_m2,
            },
        }

    def provenance_statement(self) -> str:
        """Return a compact IDS-provenance statement for the wall repair."""

        return (
            "limiter validity repair; authority=repaired-ring-only; "
            f"source_chain_sha256={self.source_chain_sha256}; "
            f"source_vertices={self.source_vertex_count}; "
            "valid_material_relative_area_difference="
            f"{self.valid_material_relative_area_difference:.16e}; "
            f"published_ring_sha256={self.published_ring_sha256}; "
            f"published_vertices={self.published_vertex_count}; "
            f"excluded_component_area_m2={self.excluded_component_area_m2:.16e}; "
            f"method={LIMITER_REPAIR_METHOD}"
        )


@dataclass(frozen=True)
class DiiidMachineIds:
    """The publishable IDS set and its explicit source gaps."""

    ids: Mapping[str, Any]
    source_path: Path
    source_dd_version: str
    dd_version: str
    absent: tuple[AbsentQuantity, ...]
    limiter_repair: LimiterRepair

    def validate(self) -> None:
        """Validate the static IDS set and its content firewall."""

        from shapely.geometry import Polygon

        if tuple(self.ids) != IDS_NAMES:
            raise ValueError(f"machine IDS set must contain exactly {IDS_NAMES!r}")
        for name, ids in self.ids.items():
            ids.validate()
            if int(ids.ids_properties.homogeneous_time) != 0:
                raise ValueError(f"{name} must describe time-independent geometry")
        magnetics = self.ids["magnetics"]
        if any(probe.field.data.has_value for probe in magnetics.b_field_pol_probe):
            raise ValueError("magnetics probe signals are forbidden in the description")
        if any(loop.flux.data.has_value for loop in magnetics.flux_loop):
            raise ValueError("magnetics flux signals are forbidden in the description")
        if len(magnetics.ip):
            raise ValueError("plasma-current signals are forbidden in the description")
        outline = self.ids["wall"].description_2d[0].limiter.unit[0].outline
        ring = np.column_stack(
            (np.asarray(outline.r, dtype=float), np.asarray(outline.z, dtype=float))
        )
        polygon = Polygon(ring[:-1] if np.array_equal(ring[0], ring[-1]) else ring)
        if not polygon.is_valid or not polygon.exterior.is_simple:
            raise ValueError("published limiter must be a valid simple ring")
        if _chain_sha256(ring) != self.limiter_repair.published_ring_sha256:
            raise ValueError("published limiter differs from its repair provenance")
        if len(ring) != self.limiter_repair.published_vertex_count:
            raise ValueError("published limiter vertex count differs from provenance")


@dataclass(frozen=True)
class SourceMachineDescription:
    """Version-pinned source values after routing through Nova's machine seam."""

    machine: StaticMachineDescription
    wall: Mapping[str, Any]
    active_coils: tuple[Mapping[str, Any], ...]
    probes: tuple[Mapping[str, Any], ...]
    flux_loops: tuple[Mapping[str, Any], ...]
    dd_version: str
    limiter_repair: LimiterRepair


def _chain_sha256(vertices: np.ndarray) -> str:
    """Return the portable byte identity of an ordered R-Z chain."""

    little_endian = np.ascontiguousarray(vertices, dtype="<f8")
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def _shoelace_area(vertices: np.ndarray) -> float:
    """Return the absolute shoelace area of a closed or open vertex chain."""

    radial = vertices[:, 0]
    vertical = vertices[:, 1]
    return float(
        0.5 * abs(radial @ np.roll(vertical, -1) - vertical @ np.roll(radial, -1))
    )


def _polygon_parts(geometry: Any) -> list[Any]:
    """Return all positive polygon members from a Shapely repair result."""

    from shapely.geometry import Polygon

    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry] if geometry.area > 0.0 else []
    if hasattr(geometry, "geoms"):
        return [part for member in geometry.geoms for part in _polygon_parts(member)]
    return []


def _canonical_exterior(polygon: Any) -> np.ndarray:
    """Return a closed, counter-clockwise ring with a stable first vertex."""

    vertices = np.asarray(polygon.exterior.coords, dtype=np.float64)[:-1, :2]
    signed_twice_area = float(
        vertices[:, 0] @ np.roll(vertices[:, 1], -1)
        - vertices[:, 1] @ np.roll(vertices[:, 0], -1)
    )
    if signed_twice_area < 0.0:
        vertices = vertices[::-1]
    start = min(range(len(vertices)), key=lambda index: tuple(vertices[index]))
    vertices = np.roll(vertices, -start, axis=0)
    return np.vstack((vertices, vertices[0]))


def repair_limiter_ring(contour: np.ndarray) -> tuple[np.ndarray, LimiterRepair]:
    """Repair a source chain and retain one deterministic physical wall ring."""

    from shapely import make_valid
    from shapely.geometry import Polygon

    source = np.asarray(contour, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or len(source) < 4:
        raise ValueError("limiter source must be an R-Z chain with at least 4 points")
    if not np.all(np.isfinite(source)):
        raise ValueError("limiter source contains non-finite coordinates")
    if not np.array_equal(source[0], source[-1]):
        source = np.vstack((source, source[0]))
    source_area = _shoelace_area(source[:-1])
    if source_area <= 0.0:
        raise ValueError("limiter source chain has no positive shoelace area")

    source_polygon = Polygon(source[:-1])
    repaired = make_valid(source_polygon)
    parts = sorted(_polygon_parts(repaired), key=lambda part: part.area, reverse=True)
    if not parts:
        raise ValueError("limiter validity repair produced no positive polygon")
    physical = parts[0]
    if len(physical.interiors):
        raise ValueError("published limiter ring must not contain interior holes")
    ring = _canonical_exterior(physical)
    published = Polygon(ring[:-1])
    if not published.is_valid or not published.exterior.is_simple:
        raise ValueError("published limiter ring is not a valid simple polygon")

    valid_material_area = float(sum(part.area for part in parts))
    published_area = float(published.area)
    repair = LimiterRepair(
        source_chain_sha256=_chain_sha256(source),
        source_vertex_count=len(source),
        source_chain_valid=bool(source_polygon.is_valid),
        source_chain_area_m2=source_area,
        valid_material_area_m2=valid_material_area,
        valid_material_relative_area_difference=abs(
            valid_material_area / source_area - 1.0
        ),
        validity_component_count=len(parts),
        published_ring_sha256=_chain_sha256(ring),
        published_vertex_count=len(ring),
        published_ring_area_m2=published_area,
        published_ring_relative_area_difference=abs(published_area / source_area - 1.0),
        excluded_component_area_m2=float(sum(part.area for part in parts[1:])),
    )
    return ring, repair


def _new_ids(
    factory: imas.IDSFactory,
    name: str,
    source_path: Path,
    *,
    source_dd_version: str,
    target_dd_version: str,
    provenance_sources: tuple[str, ...] = (),
    comment_suffix: str = "",
) -> Any:
    ids = factory.new(name)
    properties = ids.ids_properties
    properties.homogeneous_time = 0
    if hasattr(properties, "source"):
        properties.source = str(source_path)
    properties.comment = (
        f"Static DIII-D machine geometry sourced from {source_path} at IMAS "
        f"Data Dictionary {source_dd_version} and authored natively at Data "
        f"Dictionary {target_dd_version}; dynamic signals and reconstruction "
        "content are excluded, and no IDS conversion is performed."
        f"{comment_suffix}"
    )
    properties.provenance.node.resize(1)
    provenance = properties.provenance.node[0]
    provenance.path = str(source_path)
    sources = [
        f"IMAS netCDF source; Data Dictionary {source_dd_version}",
        f"native IDSFactory authoring; Data Dictionary {target_dd_version}",
        *provenance_sources,
    ]
    if hasattr(provenance, "sources"):
        provenance.sources = sources
    else:
        provenance.reference.resize(len(sources))
        for reference, source in zip(provenance.reference, sources, strict=True):
            reference.name = source
    return ids


def _primitive_record(source: Any, names: tuple[str, ...]) -> dict[str, Any]:
    """Detach populated primitive values from a source IDS structure."""

    record = {}
    for name in names:
        leaf = getattr(source, name)
        if not leaf.has_value:
            continue
        value = leaf.value
        record[name] = value.copy() if isinstance(value, np.ndarray) else value
    return record


def _write_record(target: Any, record: Mapping[str, Any]) -> None:
    """Write a detached record into a natively authored IDS structure."""

    for name, value in record.items():
        if hasattr(target, name):
            setattr(target, name, value)


def _element_record(element: Any, fallback_name: str) -> dict[str, Any]:
    """Detach one source element in the record shape the machine seam accepts."""

    geometry = element.geometry
    geometry_type = int(geometry.geometry_type)
    name = str(element.name).strip() or str(element.identifier).strip() or fallback_name
    if geometry_type == 1:
        geometry_record = {
            "geometry_type": geometry_type,
            "r": np.asarray(geometry.outline.r, dtype=float).copy(),
            "z": np.asarray(geometry.outline.z, dtype=float).copy(),
        }
    if geometry_type == 2:
        rectangle = geometry.rectangle
        geometry_record = {
            "geometry_type": geometry_type,
            "r": float(rectangle.r),
            "z": float(rectangle.z),
            "width": float(rectangle.width),
            "height": float(rectangle.height),
        }
    if geometry_type not in (1, 2):
        raise ValueError(f"unsupported pf_active geometry type {geometry_type}")
    return {
        "name": name,
        "identifier": str(element.identifier).strip(),
        "turns_with_sign": float(element.turns_with_sign),
        **geometry_record,
    }


def _source_dd_version(source_path: Path) -> str:
    """Discover the version written into the source without converting it."""

    with imas.DBEntry(source_path, "r") as database:
        wall = database.get("wall", 0, lazy=True, autoconvert=False)
        return str(wall.ids_properties.version_put.data_dictionary)


def _read_source_description(source_path: Path) -> SourceMachineDescription:
    """Read at the declared DD and detach values through Nova dataclasses."""

    source_dd_version = _source_dd_version(source_path)
    with imas.DBEntry(source_path, "r", dd_version=source_dd_version) as source_entry:
        source_ids = {
            name: source_entry.get(name, 0, autoconvert=False) for name in IDS_NAMES
        }
        for name, ids in source_ids.items():
            written = str(ids.ids_properties.version_put.data_dictionary)
            if written != source_dd_version:
                raise ValueError(
                    f"source {name} carries Data Dictionary {written}, "
                    f"expected declared version {source_dd_version}"
                )
        if source_entry.list_all_occurrences("pf_passive"):
            raise ValueError("source unexpectedly contains pf_passive geometry")

    wall_ids = source_ids["wall"]
    if len(wall_ids.description_2d) != 1:
        raise ValueError("expected exactly one wall description")
    source_wall = wall_ids.description_2d[0]
    if len(source_wall.limiter.unit) != 1:
        raise ValueError("expected exactly one limiter unit")
    source_limiter = source_wall.limiter.unit[0]
    source_contour = np.column_stack(
        (
            np.asarray(source_limiter.outline.r, dtype=float),
            np.asarray(source_limiter.outline.z, dtype=float),
        )
    )
    repaired_contour, limiter_repair = repair_limiter_ring(source_contour)
    contour = {
        "kind": "limiter",
        "r": repaired_contour[:, 0],
        "z": repaired_contour[:, 1],
    }

    active_records = []
    for coil_index, coil in enumerate(source_ids["pf_active"].coil):
        name = (
            str(coil.name).strip()
            or str(coil.identifier).strip()
            or f"coil_{coil_index}"
        )
        elements = tuple(
            _element_record(element, f"{name}_{element_index}")
            for element_index, element in enumerate(coil.element)
        )
        active_records.append(
            {
                "name": name,
                "identifier": str(coil.identifier).strip(),
                "functions": tuple(
                    _primitive_record(function, ("index", "name", "description"))
                    for function in coil.function
                ),
                "elements": elements,
            }
        )

    probes = []
    for index, probe in enumerate(source_ids["magnetics"].b_field_pol_probe):
        position = _primitive_record(probe.position, ("r", "z", "phi"))
        location = DiagnosticSightline.from_record(
            {
                "name": str(probe.name).strip() or f"probe_{index}",
                "position": tuple(
                    position.get(axis, 0.0) for axis in ("r", "z", "phi")
                ),
                "start": None,
                "end": None,
            }
        )
        probes.append(
            {
                "location": location,
                "position_leaves": tuple(position),
                "leaves": _primitive_record(
                    probe,
                    (
                        "name",
                        "identifier",
                        "length",
                        "turns",
                        "poloidal_angle",
                        "toroidal_angle",
                    ),
                ),
                "type": _primitive_record(probe.type, ("index", "name", "description")),
            }
        )

    flux_loops = []
    for loop_index, loop in enumerate(source_ids["magnetics"].flux_loop):
        positions = []
        for position_index, position in enumerate(loop.position):
            position_record = _primitive_record(position, ("r", "z", "phi"))
            locations = DiagnosticSightline.from_record(
                {
                    "name": f"flux_loop_{loop_index}_{position_index}",
                    "position": tuple(
                        position_record.get(axis, 0.0) for axis in ("r", "z", "phi")
                    ),
                    "start": None,
                    "end": None,
                }
            )
            positions.append(
                {
                    "location": locations,
                    "position_leaves": tuple(position_record),
                }
            )
        flux_loops.append(
            {
                "leaves": _primitive_record(loop, ("name", "identifier")),
                "type": _primitive_record(loop.type, ("index", "name", "description")),
                "positions": tuple(positions),
            }
        )

    machine = StaticMachineDescription.from_record(
        {
            "contour": contour,
            "pf_active": active_records,
            "pf_passive_loop_count": 0,
            "tf_coil_count": 0,
        }
    )
    return SourceMachineDescription(
        machine=machine,
        wall={
            "type": _primitive_record(
                source_wall.type, ("index", "name", "description")
            ),
            "limiter": _primitive_record(source_limiter, ("name", "identifier")),
        },
        active_coils=tuple(active_records),
        probes=tuple(probes),
        flux_loops=tuple(flux_loops),
        dd_version=source_dd_version,
        limiter_repair=limiter_repair,
    )


def _author_wall(
    factory: imas.IDSFactory,
    source: SourceMachineDescription,
    source_path: Path,
    *,
    target_dd_version: str,
) -> Any:
    target = _new_ids(
        factory,
        "wall",
        source_path,
        source_dd_version=source.dd_version,
        target_dd_version=target_dd_version,
        provenance_sources=(source.limiter_repair.provenance_statement(),),
        comment_suffix=(
            " The source limiter chain is self-intersecting; this wall IDS carries "
            "only its declared, validity-repaired physical ring."
        ),
    )
    target.description_2d.resize(1)
    description = target.description_2d[0]
    _write_record(description.type, source.wall["type"])
    description.limiter.unit.resize(1)
    limiter = description.limiter.unit[0]
    _write_record(limiter, source.wall["limiter"])
    contour = source.machine.contour
    if contour is None:
        raise ValueError("source machine has no limiter contour")
    limiter.outline.r = contour.r
    limiter.outline.z = contour.z
    return target


def _author_pf_active(
    factory: imas.IDSFactory,
    source: SourceMachineDescription,
    source_path: Path,
    *,
    target_dd_version: str,
) -> Any:
    target = _new_ids(
        factory,
        "pf_active",
        source_path,
        source_dd_version=source.dd_version,
        target_dd_version=target_dd_version,
    )
    target.coil.resize(len(source.machine.active_coils))
    for machine_coil, coil_record, coil in zip(
        source.machine.active_coils, source.active_coils, target.coil, strict=True
    ):
        coil.name = machine_coil.name
        if hasattr(coil, "identifier"):
            coil.identifier = machine_coil.identifier
        functions = coil_record["functions"]
        coil.function.resize(len(functions))
        for function_record, function in zip(functions, coil.function, strict=True):
            _write_record(function, function_record)
        coil.element.resize(len(machine_coil.elements))
        for machine_element, element_record, element in zip(
            machine_coil.elements, coil_record["elements"], coil.element, strict=True
        ):
            element.name = machine_element.name
            if element_record["identifier"] and hasattr(element, "identifier"):
                element.identifier = element_record["identifier"]
            element.turns_with_sign = element_record["turns_with_sign"]
            vertices = np.asarray(machine_element.outline, dtype=float)
            element.geometry.geometry_type = 1
            element.geometry.outline.r = vertices[:, 0]
            element.geometry.outline.z = vertices[:, 1]
    return target


def _write_position(target: Any, record: Mapping[str, Any]) -> None:
    location = record["location"].position
    for name, value in zip(("r", "z", "phi"), location, strict=True):
        if name in record["position_leaves"]:
            setattr(target, name, value)


def _author_magnetics(
    factory: imas.IDSFactory,
    source: SourceMachineDescription,
    source_path: Path,
    *,
    target_dd_version: str,
) -> Any:
    target = _new_ids(
        factory,
        "magnetics",
        source_path,
        source_dd_version=source.dd_version,
        target_dd_version=target_dd_version,
    )
    target.b_field_pol_probe.resize(len(source.probes))
    for probe_record, probe in zip(
        source.probes, target.b_field_pol_probe, strict=True
    ):
        _write_record(probe, probe_record["leaves"])
        _write_record(probe.type, probe_record["type"])
        _write_position(probe.position, probe_record)

    target.flux_loop.resize(len(source.flux_loops))
    for loop_record, loop in zip(source.flux_loops, target.flux_loop, strict=True):
        _write_record(loop, loop_record["leaves"])
        _write_record(loop.type, loop_record["type"])
        positions = loop_record["positions"]
        loop.position.resize(len(positions))
        for position_record, position in zip(positions, loop.position, strict=True):
            _write_position(position, position_record)
    return target


def build_diiid_machine_ids(source_path: Path | str = SOURCE_PATH) -> DiiidMachineIds:
    """Read at the source DD and author a native IDS set at the latest target."""

    source_path = Path(source_path)
    source = _read_source_description(source_path)
    target_dd_version = latest_published_dd_version()
    factory = imas.IDSFactory(version=target_dd_version)
    bundle = DiiidMachineIds(
        ids={
            "wall": _author_wall(
                factory,
                source,
                source_path,
                target_dd_version=target_dd_version,
            ),
            "pf_active": _author_pf_active(
                factory,
                source,
                source_path,
                target_dd_version=target_dd_version,
            ),
            "magnetics": _author_magnetics(
                factory,
                source,
                source_path,
                target_dd_version=target_dd_version,
            ),
        },
        source_path=source_path,
        source_dd_version=source.dd_version,
        dd_version=target_dd_version,
        absent=(
            AbsentQuantity(
                "pf_passive",
                "the source entry has no pf_passive occurrence; no passive "
                "loop or vessel conductor is fabricated",
            ),
            AbsentQuantity(
                "tf static conductor geometry",
                "the source tf IDS supplies a time-dependent vacuum-field "
                "signal but no static conductor geometry; the signal is excluded",
            ),
            AbsentQuantity(
                "Thomson scattering line-of-sight endpoints",
                "neither the source entry nor the competition dataset supplies "
                "endpoint pairs; representative positions are not fabricated "
                "into sightlines",
            ),
        ),
        limiter_repair=source.limiter_repair,
    )
    bundle.validate()
    return bundle


def _primitive_leaves(parent: Any, names: tuple[str, ...]) -> Iterator[tuple[str, Any]]:
    for name in names:
        if not hasattr(parent, name):
            continue
        leaf = getattr(parent, name)
        if leaf.has_value:
            value = leaf.value
            if isinstance(value, np.ndarray):
                value = value.copy()
            yield name, value


def machine_ids_snapshot(ids: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return every authored static leaf under stable, indexed paths."""

    result: dict[str, dict[str, Any]] = {name: {} for name in IDS_NAMES}
    for name in IDS_NAMES:
        properties = ids[name].ids_properties
        for leaf, value in _primitive_leaves(
            properties, ("homogeneous_time", "source", "comment")
        ):
            result[name][f"ids_properties/{leaf}"] = value
        for index, node in enumerate(properties.provenance.node):
            for leaf, value in _primitive_leaves(node, ("path", "sources")):
                result[name][f"ids_properties/provenance/node[{index}]/{leaf}"] = value
            if hasattr(node, "reference"):
                for reference_index, reference in enumerate(node.reference):
                    for leaf, value in _primitive_leaves(
                        reference, ("name", "timestamp")
                    ):
                        result[name][
                            "ids_properties/provenance/"
                            f"node[{index}]/reference[{reference_index}]/{leaf}"
                        ] = value

    wall = ids["wall"]
    for description_index, description in enumerate(wall.description_2d):
        for leaf, value in _primitive_leaves(
            description.type, ("index", "name", "description")
        ):
            result["wall"][f"description_2d[{description_index}]/type/{leaf}"] = value
        for unit_index, unit in enumerate(description.limiter.unit):
            prefix = f"description_2d[{description_index}]/limiter/unit[{unit_index}]"
            for leaf, value in _primitive_leaves(unit, ("name", "identifier")):
                result["wall"][f"{prefix}/{leaf}"] = value
            result["wall"][f"{prefix}/outline/r"] = np.asarray(unit.outline.r).copy()
            result["wall"][f"{prefix}/outline/z"] = np.asarray(unit.outline.z).copy()

    active = ids["pf_active"]
    for coil_index, coil in enumerate(active.coil):
        coil_prefix = f"coil[{coil_index}]"
        for leaf, value in _primitive_leaves(coil, ("name", "identifier")):
            result["pf_active"][f"{coil_prefix}/{leaf}"] = value
        for function_index, function in enumerate(coil.function):
            for leaf, value in _primitive_leaves(
                function, ("index", "name", "description")
            ):
                result["pf_active"][
                    f"{coil_prefix}/function[{function_index}]/{leaf}"
                ] = value
        for element_index, element in enumerate(coil.element):
            prefix = f"{coil_prefix}/element[{element_index}]"
            for leaf, value in _primitive_leaves(
                element, ("name", "identifier", "turns_with_sign")
            ):
                result["pf_active"][f"{prefix}/{leaf}"] = value
            result["pf_active"][f"{prefix}/geometry/geometry_type"] = int(
                element.geometry.geometry_type
            )
            result["pf_active"][f"{prefix}/geometry/outline/r"] = np.asarray(
                element.geometry.outline.r
            ).copy()
            result["pf_active"][f"{prefix}/geometry/outline/z"] = np.asarray(
                element.geometry.outline.z
            ).copy()

    magnetics = ids["magnetics"]
    for probe_index, probe in enumerate(magnetics.b_field_pol_probe):
        prefix = f"b_field_pol_probe[{probe_index}]"
        for leaf, value in _primitive_leaves(
            probe,
            (
                "name",
                "identifier",
                "length",
                "turns",
                "poloidal_angle",
                "toroidal_angle",
            ),
        ):
            result["magnetics"][f"{prefix}/{leaf}"] = value
        for leaf, value in _primitive_leaves(
            probe.type, ("index", "name", "description")
        ):
            result["magnetics"][f"{prefix}/type/{leaf}"] = value
        for leaf, value in _primitive_leaves(probe.position, ("r", "z", "phi")):
            result["magnetics"][f"{prefix}/position/{leaf}"] = value

    for loop_index, loop in enumerate(magnetics.flux_loop):
        prefix = f"flux_loop[{loop_index}]"
        for leaf, value in _primitive_leaves(loop, ("name", "identifier")):
            result["magnetics"][f"{prefix}/{leaf}"] = value
        for leaf, value in _primitive_leaves(
            loop.type, ("index", "name", "description")
        ):
            result["magnetics"][f"{prefix}/type/{leaf}"] = value
        for position_index, position in enumerate(loop.position):
            for leaf, value in _primitive_leaves(position, ("r", "z", "phi")):
                result["magnetics"][f"{prefix}/position[{position_index}]/{leaf}"] = (
                    value
                )
    return result


def round_trip_leaf_receipt(
    expected: Mapping[str, Mapping[str, Any]],
    actual: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compare every static leaf, requiring identical structure and values."""

    receipt: dict[str, dict[str, dict[str, Any]]] = {}
    for ids_name in IDS_NAMES:
        expected_leaves = expected[ids_name]
        actual_leaves = actual[ids_name]
        if set(expected_leaves) != set(actual_leaves):
            missing = sorted(set(expected_leaves) - set(actual_leaves))
            extra = sorted(set(actual_leaves) - set(expected_leaves))
            raise ValueError(
                f"{ids_name} leaf structure changed; missing={missing}, extra={extra}"
            )
        ids_receipt = {}
        for path, left in expected_leaves.items():
            right = actual_leaves[path]
            left_array = np.asarray(left)
            right_array = np.asarray(right)
            if left_array.dtype.kind in "SUO" or right_array.dtype.kind in "SUO":
                equal = bool(np.array_equal(left_array, right_array))
                item = {"exact_equal": equal}
            else:
                equal = bool(np.array_equal(left_array, right_array))
                difference = (
                    0.0
                    if left_array.size == 0
                    else float(np.max(np.abs(left_array - right_array)))
                )
                item = {
                    "exact_equal": equal,
                    "maximum_absolute_difference": difference,
                }
            if not equal:
                raise ValueError(f"{ids_name}/{path} changed during round trip")
            ids_receipt[path] = item
        receipt[ids_name] = ids_receipt
    return receipt
