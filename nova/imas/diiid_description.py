"""Dataset-authoritative DIII-D conductors and their vacuum-flux response.

The challenge rows are the geometry registry input: every selection is keyed by
the digest of the shipped conductor table, and every field in that table has a
receipt.  The Green-function route consumes the current channels exactly in the
units declared by the dataset.  In particular, F-coil currents are already
ampere-turns, ECOILA remains a plain current with unresolved physical turns, and
the toroidal bcoil has no axisymmetric poloidal-flux section.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nova.biot.polygon import polygon_greens

STARTER_KIT_VACUUM_R2_BAR = 0.94
STARTER_KIT_VACUUM_BAR_SOURCE = (
    "https://github.com/Sophelio/fusion-equilibrium-challenge-starter#"
    "new-machine-geometry--where-the-coils-and-chords-actually-are"
)
DATASET_SOURCE = "https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge"

F_COILS = tuple(f"F{number}{side}" for side in ("A", "B") for number in range(1, 10))
POLOIDAL_CONDUCTORS = F_COILS + ("ECOILA",)
ALL_CONDUCTORS = POLOIDAL_CONDUCTORS + ("bcoil",)

_GEOMETRY_COLUMNS = (
    "coil_name",
    "coil_input_column",
    "coil_R",
    "coil_Z",
    "coil_width",
    "coil_height",
    "coil_angle1",
    "coil_angle2",
)


class DiiidDescriptionError(ValueError):
    """Raised when a challenge row cannot author a complete description."""


@dataclass(frozen=True)
class GeometryReceipt:
    """Trace one described quantity to a challenge-dataset field."""

    fields: tuple[str, ...]
    locator: str
    source: str = DATASET_SOURCE
    unit: str = ""
    statement: str = ""

    def validate(self) -> None:
        if not self.fields or any(not value for value in self.fields):
            raise DiiidDescriptionError("a geometry receipt must name its fields")
        if not self.locator or not self.statement:
            raise DiiidDescriptionError("a geometry receipt must be followable")
        if not self.source.startswith("https://"):
            raise DiiidDescriptionError("a geometry receipt must use an https source")


@dataclass(frozen=True)
class TurnConvention:
    """Current-to-ampere-turn convention, including unresolved physical turns."""

    applied_multiplier: float
    lower_physical_turns: float | None
    upper_physical_turns: float | None
    resolved: bool
    affects_axisymmetric_poloidal_flux: bool
    statement: str

    def validate(self) -> None:
        if not math.isfinite(self.applied_multiplier):
            raise DiiidDescriptionError("turn multiplier must be finite")
        if (
            self.lower_physical_turns is not None
            and self.upper_physical_turns is not None
        ):
            if (
                self.lower_physical_turns <= 0
                or self.upper_physical_turns < self.lower_physical_turns
            ):
                raise DiiidDescriptionError("physical-turn interval is invalid")
        if not self.statement:
            raise DiiidDescriptionError(
                "turn convention needs an explanatory statement"
            )


@dataclass(frozen=True)
class DiiidConductor:
    """One named challenge actuator and its axisymmetric section, when applicable."""

    name: str
    input_column: str
    vertices: np.ndarray | None
    current_unit: str
    turns: TurnConvention
    receipts: tuple[GeometryReceipt, ...]

    def validate(self) -> None:
        if self.name not in ALL_CONDUCTORS:
            raise DiiidDescriptionError(f"unknown DIII-D conductor {self.name!r}")
        if self.input_column != f"magnetics_{self.name}":
            raise DiiidDescriptionError(
                f"{self.name} is joined to unexpected input {self.input_column!r}"
            )
        self.turns.validate()
        for receipt in self.receipts:
            receipt.validate()
        covered = {field for receipt in self.receipts for field in receipt.fields}
        required = {"input_column", "turn_convention"}
        if self.vertices is None:
            required.add("poloidal_section")
        else:
            required.update({"R", "Z", "width", "height", "angle1", "angle2"})
            vertices = np.asarray(self.vertices, dtype=float)
            if vertices.shape != (4, 2) or not np.isfinite(vertices).all():
                raise DiiidDescriptionError(
                    f"{self.name} section must have four finite vertices"
                )
            area = 0.5 * abs(
                np.sum(
                    vertices[:, 0] * np.roll(vertices[:, 1], -1)
                    - np.roll(vertices[:, 0], -1) * vertices[:, 1]
                )
            )
            if area <= 0:
                raise DiiidDescriptionError(f"{self.name} section has zero area")
        if not required <= covered:
            raise DiiidDescriptionError(
                f"{self.name} provenance is incomplete: {sorted(required - covered)}"
            )


@dataclass(frozen=True)
class DiiidDescription:
    """One registry-selected physical description."""

    physical_digest: str
    conductors: tuple[DiiidConductor, ...]
    source_rows: tuple[str, ...]

    def validate(self) -> None:
        names = tuple(conductor.name for conductor in self.conductors)
        if set(names) != set(ALL_CONDUCTORS) or len(names) != len(ALL_CONDUCTORS):
            raise DiiidDescriptionError(
                "description must contain every conductor exactly once"
            )
        for conductor in self.conductors:
            conductor.validate()

    @property
    def provenance_complete(self) -> bool:
        try:
            self.validate()
        except DiiidDescriptionError:
            return False
        return True


@dataclass
class DiiidDescriptionRegistry:
    """Digest-indexed registry authored from challenge rows."""

    configurations: dict[str, DiiidDescription] = field(default_factory=dict)

    def ingest(self, row: Mapping[str, Any], *, source_row: str) -> DiiidDescription:
        digest = geometry_digest(row)
        described = _description_from_row(row, digest=digest, source_row=source_row)
        existing = self.configurations.get(digest)
        if existing is None:
            self.configurations[digest] = described
            return described
        sources = tuple(sorted(set(existing.source_rows) | {source_row}))
        selected = DiiidDescription(digest, existing.conductors, sources)
        self.configurations[digest] = selected
        return selected

    def select(self, row: Mapping[str, Any]) -> DiiidDescription:
        digest = geometry_digest(row)
        try:
            return self.configurations[digest]
        except KeyError as error:
            raise KeyError(f"DIII-D geometry {digest} is not registered") from error


def section_vertices(
    radius: float,
    height: float,
    width: float,
    vertical_extent: float,
    angle1_deg: float,
    angle2_deg: float,
) -> np.ndarray:
    """Return EFIT rectangle/shear corners in the poloidal plane."""

    if width <= 0 or vertical_extent <= 0:
        raise DiiidDescriptionError("conductor extents must be positive")
    angle1_tangent = math.tan(math.radians(angle1_deg)) if angle1_deg else 0.0
    angle2_cotangent = 1.0 / math.tan(math.radians(angle2_deg)) if angle2_deg else 0.0
    half_width = width / 2.0
    half_height = vertical_extent / 2.0
    radial_shear = half_height * angle2_cotangent
    vertical_shear = half_width * angle1_tangent
    return np.asarray(
        [
            [radius - half_width - radial_shear, height - half_height - vertical_shear],
            [radius + half_width - radial_shear, height - half_height + vertical_shear],
            [radius + half_width + radial_shear, height + half_height + vertical_shear],
            [radius - half_width + radial_shear, height + half_height - vertical_shear],
        ],
        dtype=float,
    )


def geometry_digest(row: Mapping[str, Any]) -> str:
    """Hash the shipped conductor table without depending on shot labels."""

    payload = {column: _json_values(row, column) for column in _GEOMETRY_COLUMNS}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def vacuum_response(
    description: DiiidDescription,
    grid_r: Sequence[float],
    grid_z: Sequence[float],
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return Nova total flux on the shipped grid per ampere-turn."""

    description.validate()
    radius = np.asarray(grid_r, dtype=float)
    height = np.asarray(grid_z, dtype=float)
    if radius.ndim != 1 or height.ndim != 1 or np.any(radius <= 0):
        raise DiiidDescriptionError(
            "the EFIT grid must be one-dimensional at positive radius"
        )
    target_r, target_z = np.meshgrid(radius, height)
    names: list[str] = []
    responses: list[np.ndarray] = []
    for conductor in description.conductors:
        if (
            conductor.vertices is None
            or not conductor.turns.affects_axisymmetric_poloidal_flux
        ):
            continue
        total_flux = polygon_greens(
            target_r.ravel(), target_z.ravel(), conductor.vertices
        )[0]
        names.append(conductor.name)
        responses.append(total_flux.reshape(target_r.shape))
    return tuple(names), np.stack(responses)


def vacuum_psi(
    row: Mapping[str, Any],
    description: DiiidDescription,
    response: tuple[tuple[str, ...], np.ndarray] | None = None,
) -> np.ndarray:
    """Compute Nova total vacuum flux [Wb] from recorded currents alone."""

    names, matrix = response or vacuum_response(
        description, row["efit_grid_R"], row["efit_grid_Z"]
    )
    target_time = np.asarray(row["efit_times"], dtype=float)
    source_time = np.asarray(row["magnetics_time"], dtype=float)
    currents = []
    by_name = {conductor.name: conductor for conductor in description.conductors}
    for name in names:
        conductor = by_name[name]
        values = np.asarray(row[conductor.input_column], dtype=float)
        valid = np.isfinite(source_time) & np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            raise DiiidDescriptionError(
                f"{conductor.input_column} has fewer than two samples"
            )
        amperes = 1000.0 * np.interp(target_time, source_time[valid], values[valid])
        currents.append(amperes * conductor.turns.applied_multiplier)
    return np.einsum("tc,czr->tzr", np.column_stack(currents), matrix, optimize=True)


def _description_from_row(
    row: Mapping[str, Any], *, digest: str, source_row: str
) -> DiiidDescription:
    for column in _GEOMETRY_COLUMNS:
        if column not in row:
            raise DiiidDescriptionError(f"challenge row is missing {column}")
    columns = [_json_values(row, column) for column in _GEOMETRY_COLUMNS]
    if len({len(values) for values in columns}) != 1:
        raise DiiidDescriptionError("conductor geometry columns have different lengths")
    conductors = []
    for values in zip(*columns, strict=True):
        name, input_column, radius, height, width, extent, angle1, angle2 = values
        if name not in POLOIDAL_CONDUCTORS:
            raise DiiidDescriptionError(f"unexpected shipped geometry row {name!r}")
        turns = _turn_convention(str(name))
        conductors.append(
            DiiidConductor(
                name=str(name),
                input_column=str(input_column),
                vertices=section_vertices(
                    float(radius),
                    float(height),
                    float(width),
                    float(extent),
                    float(angle1),
                    float(angle2),
                ),
                current_unit="kA.turn" if name in F_COILS else "kA",
                turns=turns,
                receipts=(
                    GeometryReceipt(
                        fields=("R", "Z", "width", "height", "angle1", "angle2"),
                        locator=f"{source_row}:coil_name={name}; coil_* columns",
                        unit="m,m,m,m,degree,degree",
                        statement=(
                            "section copied from the dataset-shipped conductor row"
                        ),
                    ),
                    GeometryReceipt(
                        fields=("input_column",),
                        locator=f"{source_row}:coil_name={name}; coil_input_column",
                        statement="current join copied from the shipped geometry table",
                    ),
                    GeometryReceipt(
                        fields=("turn_convention",),
                        locator="dataset card machine-geometry current-unit contract",
                        statement=turns.statement,
                    ),
                ),
            )
        )
    if {conductor.name for conductor in conductors} != set(POLOIDAL_CONDUCTORS):
        missing = sorted(set(POLOIDAL_CONDUCTORS) - {c.name for c in conductors})
        raise DiiidDescriptionError(f"shipped conductor table is incomplete: {missing}")
    bcoil_turns = _turn_convention("bcoil")
    conductors.append(
        DiiidConductor(
            name="bcoil",
            input_column="magnetics_bcoil",
            vertices=None,
            current_unit="kA",
            turns=bcoil_turns,
            receipts=(
                GeometryReceipt(
                    fields=("poloidal_section",),
                    locator="dataset card machine-geometry exclusions",
                    statement="toroidal bcoil has no poloidal-plane rectangle",
                ),
                GeometryReceipt(
                    fields=("input_column",),
                    locator=f"{source_row}:magnetics_bcoil",
                    statement="recorded toroidal-field current channel",
                ),
                GeometryReceipt(
                    fields=("turn_convention",),
                    locator="dataset card main-coil unit contract",
                    statement=bcoil_turns.statement,
                ),
            ),
        )
    )
    result = DiiidDescription(digest, tuple(conductors), (source_row,))
    result.validate()
    return result


def _turn_convention(name: str) -> TurnConvention:
    if name in F_COILS:
        return TurnConvention(
            applied_multiplier=1.0,
            lower_physical_turns=1.0,
            upper_physical_turns=1.0,
            resolved=True,
            affects_axisymmetric_poloidal_flux=True,
            statement="recorded F-coil channel already contains total ampere-turns",
        )
    if name == "ECOILA":
        return TurnConvention(
            applied_multiplier=1.0,
            lower_physical_turns=1.0,
            upper_physical_turns=96.0,
            resolved=False,
            affects_axisymmetric_poloidal_flux=True,
            statement=(
                "recorded ECOILA is plain kA and is used without a fabricated turn "
                "multiplier; the published 48-element group and omitted co-located "
                "group bound the unresolved physical convention"
            ),
        )
    return TurnConvention(
        applied_multiplier=0.0,
        lower_physical_turns=None,
        upper_physical_turns=None,
        resolved=False,
        affects_axisymmetric_poloidal_flux=False,
        statement=(
            "toroidal bcoil turns are unresolved and do not enter axisymmetric "
            "poloidal flux"
        ),
    )


def _json_values(row: Mapping[str, Any], key: str) -> list[Any]:
    value = row[key]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list | tuple):
        raise DiiidDescriptionError(f"{key} must be a sequence")
    return list(value)


__all__ = [
    "ALL_CONDUCTORS",
    "DATASET_SOURCE",
    "DiiidConductor",
    "DiiidDescription",
    "DiiidDescriptionError",
    "DiiidDescriptionRegistry",
    "F_COILS",
    "GeometryReceipt",
    "STARTER_KIT_VACUUM_BAR_SOURCE",
    "STARTER_KIT_VACUUM_R2_BAR",
    "TurnConvention",
    "geometry_digest",
    "section_vertices",
    "vacuum_psi",
    "vacuum_response",
]
