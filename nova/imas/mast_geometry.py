"""Author and consume catalog-backed MAST geometry with IMAS DD 4.1.1."""

from __future__ import annotations

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

DD_VERSION = "4.1.1"


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


def _author_magnetics(factory: imas.IDSFactory, geometry: Mapping[str, Any]) -> Any:
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
        probe.poloidal_angle = float(poloidal_angle)
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


def write_and_reopen(
    bundle: CatalogIdsBundle,
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
