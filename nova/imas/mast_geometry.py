"""Author and consume catalog-backed MAST geometry with IMAS DD 4.1.1.

Two authoring entry points sit side by side.  The catalog set writes only what
the shot catalogs measure.  The provisional set adds the seeds that public
sources license — documented circuit grouping, nominal passive material and the
resistance those imply — and carries an evidence record for every field so a
seed is never mistaken for a measurement.  Fields no source fixes stay unset in
both.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import imas
import numpy as np
import shapely

from nova.catalog.mast_geometry import (
    EvidenceState,
    GeometrySelection,
    MachineGeometryRegistry,
)
from nova.imas.machine import GeomData, MachineGeometryReader
from nova.imas.machine_drive import DriveMap
from nova.imas.machine_evidence import EvidenceLedger, FieldEvidence
from nova.imas.mast_artifact import (
    ArtifactShotRange,
    VerifiedMachineArtifact,
    create_machine_artifact_manifest,
    materialize_machine_artifact,
)
from nova.imas.mast_channel_drive import (
    channel_drives,
    circuit_connections,
    electrical_records,
)
from nova.imas.mast_fitted_parameters import (
    RADIAL_PROBE_FAMILY,
    authored_turns,
    refined_evidence,
)
from nova.imas.mast_seed_parameters import (
    CIRCUIT_RELATIONS,
    loop_sections,
    passive_material,
    seed_evidence,
)

DD_VERSION = "4.1.1"
REPRESENTATIVE_SHOT = 11766


def _outline_records(wkb_hex: str) -> tuple[dict[str, Any], ...]:
    geometry = shapely.from_wkb(bytes.fromhex(wkb_hex))
    polygons = (
        tuple(geometry.geoms) if geometry.geom_type == "MultiPolygon" else (geometry,)
    )
    records = []
    for polygon in polygons:
        coordinates = np.asarray(polygon.exterior.coords, dtype=float)
        records.append(
            {
                "geometry_type": 1,
                "r": coordinates[:, 0].tolist(),
                "z": coordinates[:, 1].tolist(),
            }
        )
    return tuple(records)


@dataclass
class RegistryGeometryReader(MachineGeometryReader):
    """Present one registry polygon through Nova's geometry-reader seam."""

    physical_digest: str
    evidence: EvidenceState

    @classmethod
    def for_component(
        cls,
        shot: int,
        family: str,
        *,
        passive: bool = False,
        registry: MachineGeometryRegistry | None = None,
    ) -> tuple[RegistryGeometryReader, ...]:
        """Select a shot and return readers for a named physical component."""

        registry = registry or MachineGeometryRegistry.default()
        selection = registry.select(shot)
        key = "passive_components" if passive else "active_components"
        try:
            wkb_hex = selection.configuration.geometry[key][family]
        except KeyError as error:
            raise KeyError(f"unknown {key} family {family!r}") from error
        return tuple(
            cls(
                source=record,
                physical_digest=selection.configuration.physical_digest,
                evidence=selection.evidence,
            )
            for record in _outline_records(wkb_hex)
        )

    @property
    def geometry_type(self) -> int:
        """Return the outline geometry type used by registry polygons."""

        return int(self.source["geometry_type"])

    def section(self, geometry: type[GeomData]) -> GeomData:
        """Build a Nova geometry section from the selected registry polygon."""

        try:
            data = {attr: self.source[attr] for attr in geometry.attrs}
        except KeyError as error:
            raise KeyError(
                f"registry geometry missing attribute {error.args[0]!r} "
                f"required by section {geometry.name!r}"
            ) from None
        return geometry(None, data)


@dataclass(frozen=True)
class CatalogIdsBundle:
    """Catalog-backed machine-description IDSs and unsupported semantics."""

    selection: GeometrySelection
    ids: Mapping[str, Any]
    authoring_gaps: tuple[str, ...]

    def validate(self) -> None:
        """Validate every IDS against its pinned dictionary."""

        for ids in self.ids.values():
            ids.validate()


def _new_ids(factory: imas.IDSFactory, name: str) -> Any:
    ids = factory.new(name)
    ids.ids_properties.homogeneous_time = 0
    return ids


def _author_pf_active(factory: imas.IDSFactory, geometry: Mapping[str, Any]) -> Any:
    ids = _new_ids(factory, "pf_active")
    components = geometry["active_components"]
    ids.coil.resize(len(components))
    for coil, (name, wkb_hex) in zip(ids.coil, sorted(components.items()), strict=True):
        coil.name = name
        outlines = _outline_records(wkb_hex)
        coil.element.resize(len(outlines))
        for index, (element, outline) in enumerate(
            zip(coil.element, outlines, strict=True)
        ):
            element.name = f"{name}_{index}"
            element.geometry.geometry_type = 1
            element.geometry.outline.r = outline["r"]
            element.geometry.outline.z = outline["z"]
    return ids


def _author_pf_passive(factory: imas.IDSFactory, geometry: Mapping[str, Any]) -> Any:
    ids = _new_ids(factory, "pf_passive")
    components = geometry["passive_components"]
    ids.loop.resize(len(components))
    for loop, (name, wkb_hex) in zip(ids.loop, sorted(components.items()), strict=True):
        loop.name = name
        outlines = _outline_records(wkb_hex)
        loop.element.resize(len(outlines))
        for index, (element, outline) in enumerate(
            zip(loop.element, outlines, strict=True)
        ):
            element.name = f"{name}_{index}"
            element.geometry.geometry_type = 1
            element.geometry.outline.r = outline["r"]
            element.geometry.outline.z = outline["z"]
    return ids


def _author_wall(factory: imas.IDSFactory, geometry: Mapping[str, Any]) -> Any:
    ids = _new_ids(factory, "wall")
    ids.description_2d.resize(1)
    description = ids.description_2d[0]
    description.type.index = 1
    description.limiter.unit.resize(1)
    limiter = description.limiter.unit[0]
    limiter.name = "MAST limiter"
    points = np.asarray(geometry["limiter"], dtype=float)
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    limiter.outline.r = points[:, 0].tolist()
    limiter.outline.z = points[:, 1].tolist()
    return ids


def _append_flux_loop(
    ids: Any,
    index: int,
    name: str,
    points: np.ndarray,
    *,
    loop_type: int,
) -> None:
    loop = ids.flux_loop[index]
    loop.name = name
    loop.type.index = loop_type
    loop.position.resize(len(points))
    for target, (r, z, phi) in zip(loop.position, points, strict=True):
        target.r = float(r)
        target.z = float(z)
        target.phi = float(phi)


def _author_magnetics(
    factory: imas.IDSFactory,
    geometry: Mapping[str, Any],
    *,
    radial_families: frozenset[str] = frozenset(),
) -> Any:
    """Author the diagnostic set in the DDv4 directed-angle convention.

    ``radial_families`` names the probe families the vacuum response placed along
    the major radius.  The MAST catalog angle increases counter-clockwise from
    increasing major radius in the poloidal plane, whereas DDv4
    ``poloidal_angle`` increases clockwise.  Negating the source angle preserves
    the installed directed sensitive axis without rewriting the catalog.
    """

    ids = _new_ids(factory, "magnetics")
    magnetics = geometry["magnetics"]
    loops = magnetics["flux_loops"]
    saddle_paths = [
        (family, index, path)
        for family, paths in sorted(magnetics["saddle_paths"].items())
        for index, path in enumerate(paths)
    ]
    ids.flux_loop.resize(len(loops) + len(saddle_paths))
    for index, (r, z, span) in enumerate(loops):
        points = np.asarray([[r, z, 0.0], [r, z, span]], dtype=float)
        _append_flux_loop(ids, index, f"flux_loop_{index}", points, loop_type=1)
    offset = len(loops)
    for index, (family, family_index, path) in enumerate(saddle_paths, start=offset):
        points = np.asarray(path, dtype=float)
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])
        _append_flux_loop(
            ids,
            index,
            f"saddle_{family}_{family_index}",
            points,
            loop_type=2,
        )

    probes = magnetics["poloidal_probes"]
    poloidal_points = [
        (family, index, point)
        for family, points in sorted(magnetics["additional_points"].items())
        if family.startswith("poloidal_")
        for index, point in enumerate(points)
    ]
    ids.b_field_pol_probe.resize(len(probes) + len(poloidal_points))
    for index, row in enumerate(probes):
        probe = ids.b_field_pol_probe[index]
        r, z, poloidal_angle, length = row["pose"]
        probe.name = f"{row['family']}_{index}"
        probe.position.r = float(r)
        probe.position.z = float(z)
        source_angle = 0.0 if row["family"] in radial_families else poloidal_angle
        probe.poloidal_angle = -float(source_angle)
        probe.length = float(length)
    for index, (family, family_index, point) in enumerate(
        poloidal_points,
        start=len(probes),
    ):
        probe = ids.b_field_pol_probe[index]
        probe.name = f"{family}_{family_index}"
        probe.position.r = float(point[0])
        probe.position.z = float(point[1])
        probe.position.phi = float(point[2])

    toroidal_points = [
        (family, index, point)
        for family, points in sorted(magnetics["additional_points"].items())
        if family.startswith("toroidal_")
        for index, point in enumerate(points)
    ]
    ids.b_field_phi_probe.resize(len(toroidal_points))
    for probe, (family, family_index, point) in zip(
        ids.b_field_phi_probe,
        toroidal_points,
        strict=True,
    ):
        probe.name = f"{family}_{family_index}"
        probe.position.r = float(point[0])
        probe.position.z = float(point[1])
        probe.position.phi = float(point[2])
    return ids


def _author_circuits(ids: Any) -> None:
    """Name each documented poloidal-field circuit and how it is connected."""

    ids.circuit.resize(len(CIRCUIT_RELATIONS))
    for circuit, relation in zip(ids.circuit, CIRCUIT_RELATIONS, strict=True):
        circuit.name = relation.name
        circuit.type = relation.connection


def _author_circuit_connections(ids: Any, geometry: Mapping[str, Any]) -> None:
    """Write the node matrix of every circuit whose whole relation is sourced.

    A circuit the sources leave partly open keeps an unset matrix rather than a
    partial one, because a node list is read as complete: a junction left out of
    a written matrix says the terminals are apart, which is a stronger claim than
    saying nothing.
    """

    matrices = circuit_connections(geometry)
    for circuit in ids.circuit:
        matrix = matrices.get(str(circuit.name))
        if matrix is None:
            continue
        circuit.connections = matrix


def _author_passive_seeds(ids: Any, geometry: Mapping[str, Any]) -> None:
    """Seed passive resistivity, single-loop resistance and section turns."""

    sections = loop_sections(geometry)
    for loop in ids.loop:
        section = sections[str(loop.name)]
        material = passive_material(section.family)
        for element in loop.element:
            element.turns_with_sign = 1.0
        if material is None:
            continue
        loop.resistivity = material.resistivity
        if section.is_single_loop:
            loop.resistance = material.loop_resistance(
                section.area,
                section.major_radius,
            )


def _author_toroidal_field(factory: imas.IDSFactory) -> Any:
    """Author the toroidal-field slot without inventing a winding or a constant."""

    return _new_ids(factory, "tf")


def author_catalog_ids(selection: GeometrySelection) -> CatalogIdsBundle:
    """Author the supported registry geometry without filling source gaps."""

    factory = imas.IDSFactory(version=DD_VERSION)
    geometry = selection.configuration.geometry
    bundle = CatalogIdsBundle(
        selection=selection,
        ids={
            "pf_active": _author_pf_active(factory, geometry),
            "pf_passive": _author_pf_passive(factory, geometry),
            "wall": _author_wall(factory, geometry),
            "magnetics": _author_magnetics(factory, geometry),
        },
        authoring_gaps=selection.configuration.authoring_gaps,
    )
    bundle.validate()
    return bundle


@dataclass(frozen=True)
class ProvisionalIdsBundle:
    """Seeded machine-description IDSs and the provenance of every field."""

    selection: GeometrySelection
    ids: Mapping[str, Any]
    evidence: EvidenceLedger
    authoring_gaps: tuple[str, ...]

    def validate(self) -> None:
        """Validate every IDS against its pinned dictionary and the ledger."""

        for ids in self.ids.values():
            ids.validate()
        self.evidence.validate()


def author_provisional_ids(
    selection: GeometrySelection,
    *,
    first_shot: int,
    last_shot: int,
) -> ProvisionalIdsBundle:
    """Author the catalog geometry plus every seed public sources license."""

    factory = imas.IDSFactory(version=DD_VERSION)
    geometry = selection.configuration.geometry
    pf_active = _author_pf_active(factory, geometry)
    _author_circuits(pf_active)
    pf_passive = _author_pf_passive(factory, geometry)
    _author_passive_seeds(pf_passive, geometry)
    bundle = ProvisionalIdsBundle(
        selection=selection,
        ids={
            "pf_active": pf_active,
            "pf_passive": pf_passive,
            "wall": _author_wall(factory, geometry),
            "magnetics": _author_magnetics(factory, geometry),
            "tf": _author_toroidal_field(factory),
        },
        evidence=seed_evidence(
            geometry,
            first_shot=first_shot,
            last_shot=last_shot,
        ),
        authoring_gaps=selection.configuration.authoring_gaps,
    )
    bundle.validate()
    return bundle


def _author_fitted_turns(ids: Any, turns: Mapping[str, float]) -> tuple[str, ...]:
    """Write each measured coil's signed turn count, leaving the rest unset.

    A coil the cohort could not see keeps an unset turn count rather than a
    plausible one, because a forward model that reads a fabricated turn count
    produces a field with no way to tell that it is wrong.  Which coils were left
    unset is returned so the caller can carry it into the artifact's own gaps.
    """

    unset: list[str] = []
    for coil in ids.coil:
        name = str(coil.name)
        value = turns.get(name)
        if value is None:
            unset.append(name)
            continue
        for element in coil.element:
            element.turns_with_sign = float(value)
    return tuple(sorted(unset))


@dataclass(frozen=True)
class RefinedIdsBundle:
    """Machine-description IDSs carrying every value the vacuum cohort measured."""

    selection: GeometrySelection
    ids: Mapping[str, Any]
    evidence: EvidenceLedger
    drives: DriveMap
    authoring_gaps: tuple[str, ...]
    unset_turns: tuple[str, ...]

    def validate(self) -> None:
        """Validate every IDS against its pinned dictionary, the ledger and the map."""

        for ids in self.ids.values():
            ids.validate()
        self.evidence.validate()
        self.drives.validate()


def author_refined_ids(
    selection: GeometrySelection,
    *,
    first_shot: int,
    last_shot: int,
) -> RefinedIdsBundle:
    """Author the seeded description plus everything the vacuum cohort fixed."""

    factory = imas.IDSFactory(version=DD_VERSION)
    geometry = selection.configuration.geometry
    pf_active = _author_pf_active(factory, geometry)
    _author_circuits(pf_active)
    _author_circuit_connections(pf_active, geometry)
    unset = _author_fitted_turns(pf_active, authored_turns())
    pf_passive = _author_pf_passive(factory, geometry)
    _author_passive_seeds(pf_passive, geometry)
    seed = seed_evidence(geometry, first_shot=first_shot, last_shot=last_shot)
    gaps = tuple(
        gap
        for gap in selection.configuration.authoring_gaps
        if not gap.startswith("active turns")
    )
    if unset:
        gaps = (*gaps, f"turns are not sourced for {', '.join(unset)}")
    bundle = RefinedIdsBundle(
        selection=selection,
        ids={
            "pf_active": pf_active,
            "pf_passive": pf_passive,
            "wall": _author_wall(factory, geometry),
            "magnetics": _author_magnetics(
                factory,
                geometry,
                radial_families=frozenset({RADIAL_PROBE_FAMILY}),
            ),
            "tf": _author_toroidal_field(factory),
        },
        evidence=EvidenceLedger.create(
            (
                *refined_evidence(
                    seed.records,
                    first_shot=first_shot,
                    last_shot=last_shot,
                ),
                *electrical_records(
                    geometry,
                    first_shot=first_shot,
                    last_shot=last_shot,
                ),
            )
        ),
        drives=channel_drives(geometry),
        authoring_gaps=tuple(sorted(gaps)),
        unset_turns=unset,
    )
    bundle.validate()
    return bundle


def publish_refined_artifact(
    cache_directory: Path | str,
    *,
    registry: MachineGeometryRegistry | None = None,
    shot: int = REPRESENTATIVE_SHOT,
) -> VerifiedMachineArtifact:
    """Author, round-trip and publish the refined content-addressed revision.

    The revision differs from the seeded one in what the fields say, never in the
    conductor geometry underneath, so the registry and physical digests it carries
    are the same ones the seeded revision carried.  The manifest's semantic
    identity is what moves, and that is the point: a consumer can tell the two
    apart by identity without either of them claiming a different machine.
    """

    registry = registry or MachineGeometryRegistry.default()
    shot_ranges = artifact_shot_ranges(registry)
    selection = registry.select(shot)
    bundle = author_refined_ids(
        selection,
        first_shot=min(row.first_shot for row in shot_ranges),
        last_shot=max(row.last_shot for row in shot_ranges),
    )
    unresolved = bundle.evidence.paths_with_state(FieldEvidence.UNRESOLVED)
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "machine_description"
        write_and_reopen(bundle, source)
        manifest = create_machine_artifact_manifest(
            source,
            dd_version=DD_VERSION,
            registry_digest=registry.registry_digest,
            physical_digest=selection.configuration.physical_digest,
            shot_ranges=shot_ranges,
            complete=not unresolved,
            unresolved_gaps=bundle.authoring_gaps,
            field_evidence=bundle.evidence.records,
            channel_drive=bundle.drives.drives,
        )
        return materialize_machine_artifact(source, cache_directory, manifest)


def artifact_shot_ranges(
    registry: MachineGeometryRegistry,
) -> tuple[ArtifactShotRange, ...]:
    """Carry the registry's evidence-typed shot ranges into artifact identity."""

    return tuple(
        sorted(
            ArtifactShotRange(
                first_shot=shot_range.first_shot,
                last_shot=shot_range.last_shot,
                physical_digest=shot_range.physical_digest,
                evidence=str(shot_range.evidence),
            )
            for shot_range in registry.ranges
        )
    )


def publish_provisional_artifact(
    cache_directory: Path | str,
    *,
    registry: MachineGeometryRegistry | None = None,
    shot: int = REPRESENTATIVE_SHOT,
) -> VerifiedMachineArtifact:
    """Author, round-trip and publish one content-addressed local revision."""

    registry = registry or MachineGeometryRegistry.default()
    shot_ranges = artifact_shot_ranges(registry)
    selection = registry.select(shot)
    bundle = author_provisional_ids(
        selection,
        first_shot=min(row.first_shot for row in shot_ranges),
        last_shot=max(row.last_shot for row in shot_ranges),
    )
    unresolved = bundle.evidence.paths_with_state(FieldEvidence.UNRESOLVED)
    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "machine_description"
        write_and_reopen(bundle, source)
        manifest = create_machine_artifact_manifest(
            source,
            dd_version=DD_VERSION,
            registry_digest=registry.registry_digest,
            physical_digest=selection.configuration.physical_digest,
            shot_ranges=shot_ranges,
            complete=not unresolved,
            unresolved_gaps=bundle.authoring_gaps,
            field_evidence=bundle.evidence.records,
        )
        return materialize_machine_artifact(source, cache_directory, manifest)


def write_and_reopen(
    bundle: CatalogIdsBundle | ProvisionalIdsBundle | RefinedIdsBundle,
    path: Path | str,
) -> dict[str, Any]:
    """Write and reopen a bundle with the same pinned DD version."""

    uri = f"imas:hdf5?path={Path(path)}"
    database = imas.DBEntry(uri, "x", dd_version=DD_VERSION)
    try:
        for ids in bundle.ids.values():
            database.put(ids)
    finally:
        database.close()

    reopened = imas.DBEntry(uri, "r", dd_version=DD_VERSION)
    try:
        result = {name: reopened.get(name) for name in bundle.ids}
    finally:
        reopened.close()
    for ids in result.values():
        ids.validate()
        written_version = str(ids.ids_properties.version_put.data_dictionary)
        if written_version != DD_VERSION:
            raise ValueError(
                f"reopened IDS carries DD version {written_version}, "
                f"expected {DD_VERSION}"
            )
    return result
