"""Dataset-authoritative DIII-D conductors and their vacuum-flux response.

The challenge rows are the geometry registry input: every selection is keyed by
the digest of the shipped conductor table, and every field in that table has a
receipt.  The Green-function route consumes every fusion-coil channel on the
dataset-wide kA.turn contract.  A fit-once ``pf_active`` calibration record
routes the shipped ECOILA ampere-turns through the unshipped ohmic conductors;
the toroidal bcoil has no axisymmetric poloidal-flux section.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from nova.biot.polygon import polygon_greens

if TYPE_CHECKING:
    from nova.imas.machine import StaticMachineDescription

STARTER_KIT_VACUUM_R2_BAR = 0.94
STARTER_KIT_VACUUM_BAR_SOURCE = (
    "https://github.com/Sophelio/fusion-equilibrium-challenge-starter#"
    "new-machine-geometry--where-the-coils-and-chords-actually-are"
)
DATASET_SOURCE = "https://huggingface.co/datasets/Sophelio/fusion-equilibrium-challenge"

F_COILS = tuple(f"F{number}{side}" for side in ("A", "B") for number in range(1, 10))
POLOIDAL_CONDUCTORS = F_COILS + ("ECOILA",)
ALL_CONDUCTORS = POLOIDAL_CONDUCTORS + ("bcoil",)
CIRCUIT_DRIVEN_CONDUCTORS = ("ECOILB", "E567UP", "E567DN", "E89UP", "E89DN")

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


@dataclass(frozen=True)
class CircuitFitUncertainty:
    """Ensemble uncertainty retained beside one fitted circuit gain."""

    leave_one_shot_out_r_squared: float
    leave_one_shot_out_rmse_a_turn: float
    leave_one_shot_out_median_absolute_error_a_turn: float
    residual_rms_a_turn: float
    residual_sample_standard_deviation_a_turn: float
    residual_rms_fraction: float

    def validate(self) -> None:
        values = tuple(self.as_record().values())
        if not np.all(np.isfinite(values)):
            raise DiiidDescriptionError("circuit uncertainty must be finite")
        if not 0.0 <= self.leave_one_shot_out_r_squared <= 1.0:
            raise DiiidDescriptionError("circuit predictive score is outside [0, 1]")
        if any(value < 0.0 for value in values[1:]):
            raise DiiidDescriptionError("circuit residual spreads must be nonnegative")

    def as_record(self) -> dict[str, float]:
        """Return the uncertainty fields as a JSON-compatible record."""

        return {
            "leave_one_shot_out_r_squared": self.leave_one_shot_out_r_squared,
            "leave_one_shot_out_rmse_a_turn": self.leave_one_shot_out_rmse_a_turn,
            "leave_one_shot_out_median_absolute_error_a_turn": (
                self.leave_one_shot_out_median_absolute_error_a_turn
            ),
            "residual_rms_a_turn": self.residual_rms_a_turn,
            "residual_sample_standard_deviation_a_turn": (
                self.residual_sample_standard_deviation_a_turn
            ),
            "residual_rms_fraction": self.residual_rms_fraction,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> CircuitFitUncertainty:
        """Rebuild one uncertainty record without changing its values."""

        return cls(
            **{field: float(record[field]) for field in cls.__dataclass_fields__}
        )


@dataclass(frozen=True)
class PfActiveSupplyRecord:
    """One logical supply driven by a shipped competition current channel."""

    name: str
    identifier: str
    input_column: str
    input_unit: str
    output_unit: str
    scale_to_output: float
    statement: str

    def validate(self) -> None:
        if self.input_column != "magnetics_ECOILA":
            raise DiiidDescriptionError("ohmic supply must use shipped ECOILA")
        if self.input_unit != "kA.turn" or self.output_unit != "A.turn":
            raise DiiidDescriptionError("ohmic supply units must preserve ampere-turns")
        if self.scale_to_output != 1000.0:
            raise DiiidDescriptionError(
                "kA.turn supply conversion must be exactly 1000"
            )
        if not self.name or not self.identifier or not self.statement:
            raise DiiidDescriptionError("ohmic supply record is incomplete")

    def as_record(self) -> dict[str, Any]:
        """Return the supply as a JSON-compatible record."""

        return {
            "name": self.name,
            "identifier": self.identifier,
            "input_column": self.input_column,
            "input_unit": self.input_unit,
            "output_unit": self.output_unit,
            "scale_to_output": self.scale_to_output,
            "statement": self.statement,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> PfActiveSupplyRecord:
        """Rebuild one supply record."""

        result = cls(
            name=str(record["name"]),
            identifier=str(record["identifier"]),
            input_column=str(record["input_column"]),
            input_unit=str(record["input_unit"]),
            output_unit=str(record["output_unit"]),
            scale_to_output=float(record["scale_to_output"]),
            statement=str(record["statement"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class PfActiveCircuitDriveRecord:
    """Effective ampere-turn drive from one supply to one conductor."""

    conductor: str
    gain: float
    uncertainty: CircuitFitUncertainty

    def validate(self) -> None:
        if self.conductor not in CIRCUIT_DRIVEN_CONDUCTORS:
            raise DiiidDescriptionError(
                f"unexpected circuit conductor {self.conductor}"
            )
        if not math.isfinite(self.gain) or self.gain <= 0.0:
            raise DiiidDescriptionError("circuit gain must be finite and positive")
        self.uncertainty.validate()

    def as_record(self) -> dict[str, Any]:
        """Return the drive as a JSON-compatible record."""

        return {
            "conductor": self.conductor,
            "gain": self.gain,
            "uncertainty": self.uncertainty.as_record(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> PfActiveCircuitDriveRecord:
        """Rebuild one calibrated drive record."""

        result = cls(
            conductor=str(record["conductor"]),
            gain=float(record["gain"]),
            uncertainty=CircuitFitUncertainty.from_record(record["uncertainty"]),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class PfActiveCircuitRecord:
    """One-supply ohmic topology and its effective conductor drives."""

    name: str
    identifier: str
    supply_identifier: str
    source_conductor: str
    current_unit: str
    component_order: tuple[str, ...]
    connections: tuple[tuple[int, ...], ...]
    drives: tuple[PfActiveCircuitDriveRecord, ...]
    provenance: str
    caveats: tuple[str, ...]

    def validate(self) -> None:
        if self.source_conductor != "ECOILA" or self.current_unit != "A.turn":
            raise DiiidDescriptionError(
                "ohmic circuit source must be ECOILA ampere-turns"
            )
        drive_names = tuple(drive.conductor for drive in self.drives)
        if drive_names != CIRCUIT_DRIVEN_CONDUCTORS:
            raise DiiidDescriptionError(
                "ohmic circuit drives are incomplete or unordered"
            )
        expected_components = (
            self.supply_identifier,
            self.source_conductor,
            *CIRCUIT_DRIVEN_CONDUCTORS,
        )
        if self.component_order != expected_components:
            raise DiiidDescriptionError("ohmic circuit component order is invalid")
        matrix = np.asarray(self.connections, dtype=int)
        component_count = len(self.component_order)
        if matrix.shape != (component_count, 2 * component_count):
            raise DiiidDescriptionError(
                "ohmic circuit connection matrix has wrong shape"
            )
        if not np.all((matrix == 0) | (matrix == 1)):
            raise DiiidDescriptionError("ohmic circuit connections must be binary")
        if not np.all(matrix.sum(axis=0) == 1) or not np.all(matrix.sum(axis=1) == 2):
            raise DiiidDescriptionError(
                "ohmic circuit must form one closed series loop"
            )
        if not self.name or not self.identifier or len(self.caveats) < 4:
            raise DiiidDescriptionError("ohmic circuit provenance is incomplete")
        for drive in self.drives:
            drive.validate()

    def currents(self, source_current_a_turn: float) -> dict[str, float]:
        """Map one shipped ECOILA value to every unshipped conductor."""

        if not math.isfinite(source_current_a_turn):
            raise DiiidDescriptionError("ohmic circuit source current must be finite")
        return {
            drive.conductor: drive.gain * source_current_a_turn for drive in self.drives
        }

    def as_record(self) -> dict[str, Any]:
        """Return the complete circuit as a JSON-compatible record."""

        return {
            "name": self.name,
            "identifier": self.identifier,
            "supply_identifier": self.supply_identifier,
            "source_conductor": self.source_conductor,
            "current_unit": self.current_unit,
            "component_order": list(self.component_order),
            "connections": [list(row) for row in self.connections],
            "drives": [drive.as_record() for drive in self.drives],
            "provenance": self.provenance,
            "caveats": list(self.caveats),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> PfActiveCircuitRecord:
        """Rebuild a complete circuit record and revalidate its topology."""

        result = cls(
            name=str(record["name"]),
            identifier=str(record["identifier"]),
            supply_identifier=str(record["supply_identifier"]),
            source_conductor=str(record["source_conductor"]),
            current_unit=str(record["current_unit"]),
            component_order=tuple(str(value) for value in record["component_order"]),
            connections=tuple(
                tuple(int(value) for value in row) for row in record["connections"]
            ),
            drives=tuple(
                PfActiveCircuitDriveRecord.from_record(drive)
                for drive in record["drives"]
            ),
            provenance=str(record["provenance"]),
            caveats=tuple(str(value) for value in record["caveats"]),
        )
        result.validate()
        return result


def _closed_series_connections(component_count: int) -> tuple[tuple[int, ...], ...]:
    """Return an IMAS-style node-by-component-side connection matrix."""

    matrix = np.zeros((component_count, 2 * component_count), dtype=int)
    for component in range(component_count):
        matrix[component, 2 * component] = 1
        matrix[(component + 1) % component_count, 2 * component + 1] = 1
    return tuple(tuple(int(value) for value in row) for row in matrix)


PF_ACTIVE_SUPPLY = PfActiveSupplyRecord(
    name="DIII-D ohmic-heating supply",
    identifier="diiid_ohmic_supply",
    input_column="magnetics_ECOILA",
    input_unit="kA.turn",
    output_unit="A.turn",
    scale_to_output=1000.0,
    statement=(
        "logical description supply: the competition channel already carries "
        "total ampere-turns and is converted from kiloampere-turns exactly once"
    ),
)

_CIRCUIT_UNCERTAINTY = {
    "ECOILB": CircuitFitUncertainty(
        0.9933483532119266,
        4026.405742311334,
        1567.41255627204,
        3886.882865339553,
        3906.7678929572608,
        0.07748542028108989,
    ),
    "E567UP": CircuitFitUncertainty(
        0.9556677196797801,
        5491.813833823248,
        2924.3729372139474,
        5329.089678101602,
        4471.399220840082,
        0.20400807944058003,
    ),
    "E567DN": CircuitFitUncertainty(
        0.9420419098721444,
        6197.232415115648,
        3896.911882108876,
        6032.300818544522,
        4912.7339875007765,
        0.23424237119539848,
    ),
    "E89UP": CircuitFitUncertainty(
        0.9877483760442158,
        2817.419073607406,
        2093.257434373751,
        2793.614101796187,
        1882.3322191385087,
        0.10627969377421093,
    ),
    "E89DN": CircuitFitUncertainty(
        0.9851155287064202,
        3099.582545883437,
        2543.855213386234,
        3067.5679718120577,
        1959.9950004145705,
        0.11657434222566553,
    ),
}

PF_ACTIVE_CIRCUIT = PfActiveCircuitRecord(
    name="DIII-D effective ohmic circuit",
    identifier="diiid_effective_ohmic_circuit",
    supply_identifier=PF_ACTIVE_SUPPLY.identifier,
    source_conductor="ECOILA",
    current_unit="A.turn",
    component_order=(
        PF_ACTIVE_SUPPLY.identifier,
        "ECOILA",
        *CIRCUIT_DRIVEN_CONDUCTORS,
    ),
    connections=_closed_series_connections(2 + len(CIRCUIT_DRIVEN_CONDUCTORS)),
    drives=tuple(
        PfActiveCircuitDriveRecord(name, gain, _CIRCUIT_UNCERTAINTY[name])
        for name, gain in (
            ("ECOILB", 2.000918),
            ("E567UP", 1.023129),
            ("E567DN", 1.001657),
            ("E89UP", 1.045695),
            ("E89DN", 1.045624),
        )
    ),
    provenance=(
        "fit once from full-grid Tikhonov label-flux regressions over 60 frames "
        "and 20 shots; applied deterministically at inference"
    ),
    caveats=(
        "ECOILB and the shipped ECOILA response are nearly degenerate; the "
        "factor-two ECOILB gain is an effective calibration, not a turn count",
        "the fitted targets use EFIT label flux during calibration and are not "
        "an inference-time label read",
        "post-fit flux closure passes 1 of 60 frames against the 54 of 60 rule; "
        "the circuit does not explain all labelled flux",
        "competition fusion-coil channels are kA.turn; multiply by 1000 once to "
        "obtain the A.turn values consumed by the response operator",
    ),
)


@dataclass(frozen=True)
class DiiidDatasetMachineDescription:
    """Competition geometry routed through Nova's static machine seam.

    The shipped conductor parameters are converted to vertices exactly once by
    :func:`section_vertices`.  Those same vertices author the outline records
    consumed by ``MachineSection``; this wrapper adds the dataset quantities
    that are not poloidal sections while retaining explicit source receipts.
    """

    physical: DiiidDescription
    machine: StaticMachineDescription
    grid_r: tuple[float, ...]
    grid_z: tuple[float, ...]
    active_supplies: tuple[PfActiveSupplyRecord, ...]
    active_circuits: tuple[PfActiveCircuitRecord, ...]
    receipts: tuple[GeometryReceipt, ...]

    def validate(self) -> None:
        """Require the complete released machine-geometry contract."""
        self.physical.validate()
        if self.machine.contour is not None:
            raise DiiidDescriptionError(
                "the competition dataset does not ship a wall contour"
            )
        if self.machine.passive_loop_count != 0:
            raise DiiidDescriptionError(
                "the competition dataset does not ship passive structures"
            )
        expected = set(POLOIDAL_CONDUCTORS)
        section_names = {section.name for section in self.machine.active_sections}
        if section_names != expected or len(self.machine.active_sections) != len(
            expected
        ):
            raise DiiidDescriptionError(
                "the static machine route must contain all nineteen poloidal sections"
            )
        if len(self.grid_r) != 65 or len(self.grid_z) != 65:
            raise DiiidDescriptionError("the released EFIT grid must be 65 by 65")
        if not self.machine.sightlines:
            raise DiiidDescriptionError("the released Thomson geometry is empty")
        if self.active_supplies != (PF_ACTIVE_SUPPLY,):
            raise DiiidDescriptionError("description must carry the ECOILA supply")
        if self.active_circuits != (PF_ACTIVE_CIRCUIT,):
            raise DiiidDescriptionError("description must carry the effective circuit")
        self.active_supplies[0].validate()
        self.active_circuits[0].validate()
        if (
            self.active_circuits[0].supply_identifier
            != self.active_supplies[0].identifier
        ):
            raise DiiidDescriptionError("active circuit references an absent supply")
        for receipt in self.receipts:
            receipt.validate()
        covered = {field for receipt in self.receipts for field in receipt.fields}
        required = {
            "thomson_chord_name",
            "thomson_chord_R",
            "thomson_chord_Z",
            "efit_grid_R",
            "efit_grid_Z",
            "wall_contour",
            "passive_structure",
        }
        if not required <= covered:
            raise DiiidDescriptionError(
                "dataset machine provenance is incomplete: "
                f"{sorted(required - covered)}"
            )

    @property
    def provenance_complete(self) -> bool:
        """Return whether every present and explicitly absent quantity is traced."""
        try:
            self.validate()
        except DiiidDescriptionError:
            return False
        return True

    def pf_active_record(self) -> dict[str, Any]:
        """Return the supply and circuit payload for machine-description storage."""

        self.validate()
        return {
            "supply": [supply.as_record() for supply in self.active_supplies],
            "circuit": [circuit.as_record() for circuit in self.active_circuits],
            "response_order": [
                *POLOIDAL_CONDUCTORS,
                *CIRCUIT_DRIVEN_CONDUCTORS,
            ],
        }


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


def dataset_machine_description(
    row: Mapping[str, Any], *, source_row: str
) -> DiiidDatasetMachineDescription:
    """Build the released DIII-D geometry through ``StaticMachineDescription``.

    Dataset section parameters first author the physical registry description.
    The resulting vertices, rather than a second interpretation of the source
    angles, then pass through ``CrossSection.transform`` as polygon outlines.
    Thomson coordinates are poloidal points, so the named adapter embeds them
    in the representative phi-zero plane without inventing sightline endpoints.
    """
    from nova.imas.machine import StaticMachineDescription

    physical = _description_from_row(
        row,
        digest=geometry_digest(row),
        source_row=source_row,
    )
    sections = []
    for conductor in physical.conductors:
        if conductor.vertices is None:
            continue
        sections.append(
            {
                "geometry_type": 1,
                "name": conductor.name,
                "r": conductor.vertices[:, 0].tolist(),
                "z": conductor.vertices[:, 1].tolist(),
            }
        )

    chord_names = _json_values(row, "thomson_chord_name")
    chord_r = _json_values(row, "thomson_chord_R")
    chord_z = _json_values(row, "thomson_chord_Z")
    if len({len(chord_names), len(chord_r), len(chord_z)}) != 1:
        raise DiiidDescriptionError(
            "Thomson chord geometry columns have different lengths"
        )
    sightlines = [
        {
            "name": str(name),
            "position": [float(radius), float(height), 0.0],
            "start": None,
            "end": None,
        }
        for name, radius, height in zip(chord_names, chord_r, chord_z, strict=True)
    ]
    machine = StaticMachineDescription.from_record(
        {
            "contour": None,
            "pf_active": sections,
            "pf_passive_loop_count": 0,
            "tf_coil_count": 0,
            "thomson_scattering": sightlines,
        }
    )
    grid_r = tuple(float(value) for value in _json_values(row, "efit_grid_R"))
    grid_z = tuple(float(value) for value in _json_values(row, "efit_grid_Z"))
    receipts = (
        GeometryReceipt(
            fields=(
                "thomson_chord_name",
                "thomson_chord_R",
                "thomson_chord_Z",
            ),
            locator=f"{source_row}:thomson_chord_*",
            unit="name,m,m",
            statement=(
                "poloidal chord coordinates embedded at phi zero; no line-of-sight "
                "endpoint was fabricated"
            ),
        ),
        GeometryReceipt(
            fields=("efit_grid_R", "efit_grid_Z"),
            locator=f"{source_row}:efit_grid_R,efit_grid_Z",
            unit="m,m",
            statement="released 65 by 65 EFIT coordinate axes copied unmodified",
        ),
        GeometryReceipt(
            fields=("wall_contour",),
            locator="competition dataset machine-geometry exclusions",
            statement="wall contour explicitly absent and no external source selected",
        ),
        GeometryReceipt(
            fields=("passive_structure",),
            locator="competition dataset machine-geometry exclusions",
            statement=(
                "passive structure explicitly absent and no external source selected"
            ),
        ),
    )
    result = DiiidDatasetMachineDescription(
        physical=physical,
        machine=machine,
        grid_r=grid_r,
        grid_z=grid_z,
        active_supplies=(PF_ACTIVE_SUPPLY,),
        active_circuits=(PF_ACTIVE_CIRCUIT,),
        receipts=receipts,
    )
    result.validate()
    return result


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


def _imas_element_vertices(geometry: Any, coil_name: str) -> np.ndarray:
    """Return one IMAS active-coil element as an exact polygon."""

    geometry_type = int(geometry.geometry_type)
    if geometry_type == 1:
        vertices = np.c_[
            np.asarray(geometry.outline.r, dtype=float),
            np.asarray(geometry.outline.z, dtype=float),
        ]
    elif geometry_type == 2:
        rectangle = geometry.rectangle
        half_width = 0.5 * float(rectangle.width)
        half_height = 0.5 * float(rectangle.height)
        vertices = np.asarray(
            [
                (
                    float(rectangle.r) - half_width,
                    float(rectangle.z) - half_height,
                ),
                (
                    float(rectangle.r) + half_width,
                    float(rectangle.z) - half_height,
                ),
                (
                    float(rectangle.r) + half_width,
                    float(rectangle.z) + half_height,
                ),
                (
                    float(rectangle.r) - half_width,
                    float(rectangle.z) + half_height,
                ),
            ]
        )
    else:
        raise DiiidDescriptionError(
            f"unsupported geometry type {geometry_type} for {coil_name}"
        )
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise DiiidDescriptionError(f"{coil_name} element has an invalid polygon")
    if not np.all(np.isfinite(vertices)):
        raise DiiidDescriptionError(f"{coil_name} element has non-finite vertices")
    return vertices


def active_coil_response_from_imas(
    entry_path: str | Path,
    dd_version: str,
    coil_names: Sequence[str],
    target_r: np.ndarray,
    target_z: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray, dict[str, Any]]:
    """Build exact active-coil flux columns on arbitrary target coordinates.

    The returned response is Nova total poloidal flux in Wb per ampere.  Every
    element is evaluated with its written IMAS geometry and signed turns; no
    filament-centre approximation or data-dictionary conversion is allowed.
    """

    import imas

    names = tuple(str(name) for name in coil_names)
    if not names or len(set(names)) != len(names):
        raise DiiidDescriptionError("active-coil names must be unique and nonempty")
    radius, height = np.broadcast_arrays(
        np.asarray(target_r, dtype=float), np.asarray(target_z, dtype=float)
    )
    if radius.size == 0 or np.any(radius <= 0.0):
        raise DiiidDescriptionError("target radii must be nonempty and positive")
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(height)):
        raise DiiidDescriptionError("target coordinates must be finite")

    responses = []
    records = []
    with imas.DBEntry(Path(entry_path), "r", dd_version=dd_version) as entry:
        active = entry.get("pf_active", autoconvert=False)
        written_dd = str(active.ids_properties.version_put.data_dictionary)
        if written_dd != dd_version:
            raise DiiidDescriptionError(f"expected DD {dd_version}, read {written_dd}")
        coils = {str(coil.name): coil for coil in active.coil}
        missing = [name for name in names if name not in coils]
        if missing:
            raise DiiidDescriptionError(f"active coils are missing: {missing}")
        for name in names:
            coil = coils[name]
            response = np.zeros(radius.shape, dtype=float)
            turn_sum = 0.0
            for element in coil.element:
                vertices = _imas_element_vertices(element.geometry, name)
                turns = float(element.turns_with_sign)
                if not math.isfinite(turns):
                    raise DiiidDescriptionError(f"{name} has non-finite turns")
                turn_sum += turns
                response += turns * polygon_greens(
                    radius.ravel(), height.ravel(), vertices
                )[0].reshape(radius.shape)
            responses.append(response)
            records.append(
                {
                    "coil": name,
                    "elements": len(coil.element),
                    "signed_turn_sum": turn_sum,
                }
            )
    return (
        names,
        np.stack(responses),
        {
            "entry": str(entry_path),
            "dd_version": dd_version,
            "coils": records,
            "target_shape": list(radius.shape),
            "target_points": int(radius.size),
            "kernel": "nova.biot.polygon.polygon_greens",
            "flux_unit": "Wb per A",
        },
    )


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
                current_unit="kA.turn",
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
            lower_physical_turns=48.0,
            upper_physical_turns=96.0,
            resolved=True,
            affects_axisymmetric_poloidal_flux=True,
            statement=(
                "recorded ECOILA already contains total ampere-turns and receives no "
                "additional turn multiplier; the 48-to-96 physical grouping range is "
                "topology context rather than an input-unit correction"
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
    "DiiidDatasetMachineDescription",
    "F_COILS",
    "CIRCUIT_DRIVEN_CONDUCTORS",
    "CircuitFitUncertainty",
    "GeometryReceipt",
    "PF_ACTIVE_CIRCUIT",
    "PF_ACTIVE_SUPPLY",
    "PfActiveCircuitDriveRecord",
    "PfActiveCircuitRecord",
    "PfActiveSupplyRecord",
    "STARTER_KIT_VACUUM_BAR_SOURCE",
    "STARTER_KIT_VACUUM_R2_BAR",
    "TurnConvention",
    "active_coil_response_from_imas",
    "dataset_machine_description",
    "geometry_digest",
    "section_vertices",
    "vacuum_psi",
    "vacuum_response",
]
