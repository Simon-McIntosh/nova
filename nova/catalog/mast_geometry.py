"""Recover and select subdivision-independent MAST catalog geometry.

The level-2 catalog contains named physical components and diagnostic paths.
The level-1 catalog supplies magnetic-probe sensitive-axis angles and, on
later shots, flux-loop toroidal spans.  Its per-shot probe positions and EFIT
filament cells are retained only as setup-representation evidence: the named
level-2 geometry is the physical placement authority.  This prevents setup
calibration drift or cell-count changes from masquerading as hardware changes.

The command scans static arrays only.  It first hashes the source snapshots,
then canonicalizes one representative of each distinct snapshot.  This avoids
rebuilding polygon unions for every shot while retaining a complete catalog
census.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import shapely
import zarr
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

DEFAULT_LEVEL1_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")
DEFAULT_LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")
DEFAULT_REGISTRY_PATH = Path(__file__).with_name("mast_geometry.json")
GEOMETRY_PRECISION_M = 1e-5
ANGLE_PRECISION_DEG = 1e-4
ANGLE_PRECISION_RAD = float(np.deg2rad(ANGLE_PRECISION_DEG))
SOURCE_ANGLE_UNIT = "degree"

_ACTIVE_SUFFIXES = ("_r", "_z", "_width", "_height")
_ACTIVE_COMPONENTS = (
    "p2_inner_lower",
    "p2_inner_upper",
    "p2_outer_lower",
    "p2_outer_upper",
    "p3_lower",
    "p3_upper",
    "p4_lower",
    "p4_upper",
    "p5_lower",
    "p5_upper",
    "p6_lower",
    "p6_upper",
    "sol",
)
_PASSIVE_SUFFIXES = (
    "_r",
    "_z",
    "_width",
    "_height",
    "_shapeAngle1",
    "_shapeAngle2",
)
_PASSIVE_COMPONENTS = (
    "botcol",
    "coil_cases",
    "endcrown_l",
    "endcrown_u",
    "incon",
    "lhorw",
    "mid",
    "p2larm",
    "p2ldivpl",
    "p2uarm",
    "p2udivpl",
    "ring",
    "rodgr",
    "topcol",
    "uhorw",
    "vertw",
)
SUPERSEDED_PHYSICAL_DIGESTS = {
    "76cf833561e602a7": (
        "recovered before two source-reading corrections: every coil-case plate "
        "exchanged its radial and vertical extents about its own centre, and the "
        "outboard radial probe family carried the poloidal angle of the axial probe "
        "it shares a position with"
    ),
}
"""Canonical identities a corrected reading of the same sources has replaced.

A consumer holding one of these has a description of this machine, not of another
one, so it needs the mapping rather than a lookup failure.
"""

_RADIAL_PROBE_FAMILIES = ("obr",)
"""Poloidal probe families the level-2 store names for the radial component.

Outboard probes are installed as pairs reading the two poloidal components at one
position, so position alone cannot say which member of a pair a named family is.
The family name carries that, and the level-1 angles confirm it: the store gives
exactly as many radial angles as this family has members, all at its positions.
"""

_TRANSPOSED_EXTENT_FAMILIES = ("coil_cases",)
"""Passive families whose source width and height name the opposite axis.

Every other family measures width across the major radius.  Read that way a case
plate turns broadside and sweeps through the winding pack it should sit outside;
read across, each group of plates closes into an enclosure whose interior holds
its coil, and the plate faces meet edge to edge.
"""

_MAGNETICS_SUFFIXES = (
    "_r",
    "_z",
    "_phi",
    "_phi_1",
    "_phi_2",
    "_length",
    "_geometry_channel",
)
_XRAY_SUFFIXES = (
    "_origin_r",
    "_origin_z",
    "_endpoint_r",
    "_endpoint_z",
    "_phi",
    "_geometry_channel",
)
_L1_SETUP_KEYS = (
    "magpr_r",
    "magpr_z",
    "magpr_ang",
    "silop_r",
    "silop_z",
    "fcoil_r",
    "fcoil_z",
    "fcoil_turns",
    "limiterr",
    "limiterz",
)

_LOOP_NAME_PATTERN = re.compile(r"^FL_(?:CC0?(\d+)|(P\d)([UL])_(\d+))$")

_COMPONENT_MOUNT_PATTERN = re.compile(r"^(p\d)_(?:.*_)?(upper|lower)$")

CENTRE_COLUMN_MOUNT = "sol"
"""Active component the centre-column loops encircle.

They are named for the column rather than for a coil, and the only conductor
inside them is the solenoid, so that is the coil they are mounted on.
"""


@dataclass(frozen=True)
class SourceFingerprint:
    """Static catalog hashes for one shot."""

    shot: int
    source_digest: str | None
    representation_digest: str | None
    complete: bool
    missing: tuple[str, ...]


def _write_fingerprint_checkpoint(
    path: Path,
    rows: list[SourceFingerprint],
    level1_root: Path,
    level2_root: Path,
) -> None:
    """Atomically persist completed source fingerprints for scan resumption."""

    payload = {
        "schema": "mast-catalog-source-fingerprints",
        "level1_root": str(level1_root),
        "level2_root": str(level2_root),
        "rows": [
            {
                "shot": row.shot,
                "source_digest": row.source_digest,
                "representation_digest": row.representation_digest,
                "complete": row.complete,
                "missing": list(row.missing),
            }
            for row in sorted(rows, key=lambda row: row.shot)
        ],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_fingerprint_checkpoint(
    path: Path,
    level1_root: Path,
    level2_root: Path,
) -> list[SourceFingerprint]:
    """Load a compatible scan checkpoint or reject mismatched roots."""

    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if payload.get("schema") != "mast-catalog-source-fingerprints":
        raise ValueError(f"unsupported fingerprint checkpoint at {path}")
    roots = (payload.get("level1_root"), payload.get("level2_root"))
    expected = (str(level1_root), str(level2_root))
    if roots != expected:
        raise ValueError(
            f"checkpoint roots {roots!r} do not match requested roots {expected!r}"
        )
    return [
        SourceFingerprint(
            shot=int(row["shot"]),
            source_digest=row["source_digest"],
            representation_digest=row["representation_digest"],
            complete=bool(row["complete"]),
            missing=tuple(row["missing"]),
        )
        for row in payload["rows"]
    ]


class EvidenceState(StrEnum):
    """Strength of catalog evidence attached to a shot selection."""

    OBSERVED = "observed"
    INHERITED = "inherited"
    MISSING = "missing"


@dataclass(frozen=True)
class GeometryUnits:
    """Units used by the canonical physical payload."""

    length: str
    angle: str


@dataclass(frozen=True)
class GeometryTolerances:
    """Quantization tolerances used before physical identity hashing."""

    length_m: float
    angle_rad: float


@dataclass(frozen=True)
class ShotRange:
    """A configuration assignment over a closed shot interval."""

    first_shot: int
    last_shot: int
    physical_digest: str
    evidence: EvidenceState

    def contains(self, shot: int) -> bool:
        """Return whether ``shot`` lies in this closed range."""

        return self.first_shot <= shot <= self.last_shot


@dataclass(frozen=True)
class SourceGap:
    """Incomplete source evidence for one otherwise assigned shot."""

    shot: int
    missing: tuple[str, ...]


@dataclass(frozen=True)
class PhysicalConfiguration:
    """One canonical physical machine and diagnostic-pose record."""

    physical_digest: str
    geometry: Mapping[str, Any]
    authoring_gaps: tuple[str, ...]


@dataclass(frozen=True)
class GeometrySelection:
    """Shot-resolved physical configuration plus evidence quality."""

    shot: int
    configuration: PhysicalConfiguration
    evidence: EvidenceState
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class MachineGeometryRegistry:
    """Validated immutable MAST physical-geometry registry."""

    schema: str
    dd_version: str
    units: GeometryUnits
    tolerances: GeometryTolerances
    configurations: Mapping[str, PhysicalConfiguration]
    representation_aliases: Mapping[str, str]
    ranges: tuple[ShotRange, ...]
    incomplete_evidence: Mapping[int, SourceGap]
    provenance: Mapping[str, Any]
    registry_digest: str

    @classmethod
    def default(cls) -> MachineGeometryRegistry:
        """Load the packaged canonical MAST registry."""

        return cls.load(DEFAULT_REGISTRY_PATH)

    @classmethod
    def load(cls, path: Path | str) -> MachineGeometryRegistry:
        """Load and validate a registry JSON payload."""

        raw = json.loads(Path(path).read_text())
        configurations = {
            digest: PhysicalConfiguration(
                physical_digest=row["physical_digest"],
                geometry=row["geometry"],
                authoring_gaps=tuple(row.get("authoring_gaps", ())),
            )
            for digest, row in raw["configurations"].items()
        }
        ranges = tuple(
            ShotRange(
                first_shot=row["first_shot"],
                last_shot=row["last_shot"],
                physical_digest=row["physical_digest"],
                evidence=EvidenceState(row["evidence"]),
            )
            for row in raw["ranges"]
        )
        gaps = {
            int(row["shot"]): SourceGap(
                shot=int(row["shot"]),
                missing=tuple(row["missing"]),
            )
            for row in raw["incomplete_evidence"]
        }
        registry = cls(
            schema=raw["schema"],
            dd_version=raw["dd_version"],
            units=GeometryUnits(**raw["units"]),
            tolerances=GeometryTolerances(**raw["tolerances"]),
            configurations=configurations,
            representation_aliases=raw["representation_aliases"],
            ranges=ranges,
            incomplete_evidence=gaps,
            provenance=raw["provenance"],
            registry_digest=raw["registry_digest"],
        )
        registry.validate(raw)
        return registry

    def validate(self, raw: Mapping[str, Any] | None = None) -> None:
        """Reject ambiguous ranges, invalid aliases, and altered payloads."""

        if self.schema != "nova-machine-geometry-registry":
            raise ValueError(f"unsupported registry schema {self.schema!r}")
        if self.dd_version != "4.1.1":
            raise ValueError(
                f"registry DD version must be 4.1.1, got {self.dd_version}"
            )
        if self.units != GeometryUnits(length="m", angle="rad"):
            raise ValueError(f"registry units must be SI, got {self.units}")
        if self.tolerances.length_m <= 0 or self.tolerances.angle_rad <= 0:
            raise ValueError("registry tolerances must be positive")
        for digest, configuration in self.configurations.items():
            if digest != configuration.physical_digest:
                raise ValueError(f"configuration key {digest} disagrees with payload")
            actual = physical_digest(dict(configuration.geometry))
            if actual != digest:
                raise ValueError(
                    f"configuration {digest} has altered physical digest {actual}"
                )
            _validate_component_families(configuration.geometry)
        for alias, digest in self.representation_aliases.items():
            if not alias or digest not in self.configurations:
                raise ValueError(f"invalid representation alias {alias!r}: {digest!r}")
        previous_last: int | None = None
        for shot_range in self.ranges:
            if shot_range.physical_digest not in self.configurations:
                raise ValueError(
                    f"range references unknown digest {shot_range.physical_digest}"
                )
            if shot_range.first_shot > shot_range.last_shot:
                raise ValueError(f"empty shot range {shot_range}")
            if previous_last is not None and shot_range.first_shot <= previous_last:
                raise ValueError(f"overlapping shot range {shot_range}")
            previous_last = shot_range.last_shot
        for shot, gap in self.incomplete_evidence.items():
            if shot != gap.shot or not gap.missing:
                raise ValueError(f"invalid source gap for shot {shot}")
            if not any(shot_range.contains(shot) for shot_range in self.ranges):
                raise ValueError(f"source gap shot {shot} lies outside registry ranges")
        if raw is not None:
            digest_payload = dict(raw)
            digest_payload.pop("registry_digest", None)
            actual = stable_digest(digest_payload)
            if actual != self.registry_digest:
                raise ValueError(
                    f"registry digest {self.registry_digest} does not match {actual}"
                )

    def select(self, shot: int) -> GeometrySelection:
        """Select exactly one configuration and evidence state for ``shot``."""

        matches = [
            shot_range for shot_range in self.ranges if shot_range.contains(shot)
        ]
        if len(matches) != 1:
            raise KeyError(f"shot {shot} is outside the MAST geometry registry")
        shot_range = matches[0]
        gap = self.incomplete_evidence.get(shot)
        evidence = EvidenceState.MISSING if gap else shot_range.evidence
        return GeometrySelection(
            shot=shot,
            configuration=self.configurations[shot_range.physical_digest],
            evidence=evidence,
            missing=gap.missing if gap else (),
        )

    def resolve_representation(self, digest: str) -> PhysicalConfiguration:
        """Resolve a historical setup digest without making it identity."""

        try:
            physical_digest_value = self.representation_aliases[digest]
        except KeyError as error:
            raise KeyError(f"unknown MAST setup representation {digest!r}") from error
        return self.configurations[physical_digest_value]


def stable_digest(payload: Mapping[str, Any], *, length: int = 64) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible data."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def _validate_component_families(geometry: Mapping[str, Any]) -> None:
    """Validate the named catalog families required for a physical snapshot."""

    active = set(geometry.get("active_components", ()))
    passive = set(geometry.get("passive_components", ()))
    if active != set(_ACTIVE_COMPONENTS):
        missing = sorted(set(_ACTIVE_COMPONENTS) - active)
        extra = sorted(active - set(_ACTIVE_COMPONENTS))
        raise ValueError(
            f"active component families differ: missing={missing}, extra={extra}"
        )
    if passive != set(_PASSIVE_COMPONENTS):
        missing = sorted(set(_PASSIVE_COMPONENTS) - passive)
        extra = sorted(passive - set(_PASSIVE_COMPONENTS))
        raise ValueError(
            f"passive component families differ: missing={missing}, extra={extra}"
        )
    if not geometry.get("limiter"):
        raise ValueError("limiter cycle is empty")
    magnetics = geometry.get("magnetics", {})
    for key in ("poloidal_probes", "flux_loops", "saddle_paths"):
        if not magnetics.get(key):
            raise ValueError(f"magnetics {key} is empty")


def _array(group: zarr.Group, key: str) -> np.ndarray:
    return np.asarray(group[key][...])


def _finite_rows(*values: np.ndarray) -> tuple[np.ndarray, ...]:
    arrays = tuple(np.asarray(value).reshape(-1) for value in values)
    keep = np.ones(arrays[0].shape, dtype=bool)
    for value in arrays:
        if np.issubdtype(value.dtype, np.number):
            keep &= np.isfinite(value)
    return tuple(value[keep] for value in arrays)


def _update_hash(hasher: Any, label: str, value: np.ndarray) -> None:
    array = np.asarray(value)
    hasher.update(label.encode())
    hasher.update(str(array.shape).encode())
    if np.issubdtype(array.dtype, np.number):
        hasher.update(np.round(array.astype(np.float64), 7).tobytes())
    else:
        hasher.update("\0".join(str(item) for item in array.reshape(-1)).encode())


def _hash_group(
    hasher: Any,
    group: zarr.Group,
    *,
    suffixes: tuple[str, ...] = (),
    keys: tuple[str, ...] = (),
    prefixes: tuple[str, ...] = (),
) -> None:
    available = set(group.array_keys())
    selected = set(keys) & available
    selected.update(
        key
        for key in available
        if (not prefixes or key.startswith(prefixes))
        and (not suffixes or key.endswith(suffixes))
    )
    for key in sorted(selected):
        _update_hash(hasher, key, _array(group, key))


def _setup_representation_digest(group: zarr.Group) -> str:
    """Reproduce Ambix's rounded EFIT setup signature without dynamic arrays."""

    hasher = hashlib.sha256()
    for key in _L1_SETUP_KEYS:
        if key not in group:
            continue
        array = np.asarray(_array(group, key), dtype=np.float64).reshape(-1)
        finite = np.round(array[np.isfinite(array)], 4)
        hasher.update(np.ascontiguousarray(finite).tobytes())
    return hasher.hexdigest()[:16]


def source_fingerprint(
    shot: int,
    level1_root: Path = DEFAULT_LEVEL1_ROOT,
    level2_root: Path = DEFAULT_LEVEL2_ROOT,
) -> SourceFingerprint:
    """Hash all static source geometry needed to recover one machine snapshot."""

    l1_path = level1_root / f"{shot}.zarr"
    l2_path = level2_root / f"{shot}.zarr"
    missing: list[str] = []
    source = hashlib.sha256()

    if not l1_path.exists():
        representation_digest = None
    else:
        try:
            level1 = zarr.open_group(str(l1_path), mode="r")
            efm = level1["efm"]
            representation_digest = _setup_representation_digest(efm)
        except IndexError, KeyError, OSError, RuntimeError, ValueError:
            representation_digest = None

    required_groups = ("pf_active", "pf_passive", "wall", "magnetics")
    if not l2_path.exists():
        missing.append("level2")
    else:
        try:
            level2 = zarr.open_group(str(l2_path), mode="r")
            for group_name in required_groups:
                if group_name not in level2:
                    missing.append(f"level2-{group_name}")
                    continue
                group = level2[group_name]
                if group_name == "pf_active":
                    for component in _ACTIVE_COMPONENTS:
                        if f"{component}_r" not in group:
                            missing.append(f"level2-pf_active-{component}")
                    _hash_group(source, group, suffixes=_ACTIVE_SUFFIXES)
                elif group_name == "pf_passive":
                    for component in _PASSIVE_COMPONENTS:
                        if f"{component}_r" not in group:
                            missing.append(f"level2-pf_passive-{component}")
                    _hash_group(source, group, suffixes=_PASSIVE_SUFFIXES)
                elif group_name == "wall":
                    _hash_group(source, group, keys=("limiter_r", "limiter_z"))
                else:
                    _hash_group(source, group, suffixes=_MAGNETICS_SUFFIXES)
            if "soft_x_rays" in level2:
                _hash_group(source, level2["soft_x_rays"], suffixes=_XRAY_SUFFIXES)
        except IndexError, KeyError, OSError, RuntimeError, ValueError:
            missing.append("level2-geometry")

    complete = not missing
    return SourceFingerprint(
        shot=shot,
        source_digest=source.hexdigest()[:16] if complete else None,
        representation_digest=representation_digest,
        complete=complete,
        missing=tuple(sorted(set(missing))),
    )


def shaped_section_vertices(
    r: float,
    z: float,
    width: float,
    height: float,
    angle1_deg: float,
    angle2_deg: float,
) -> np.ndarray:
    """Return MAST catalog passive-section vertices in the poloidal plane."""

    width = abs(width)
    height = abs(height)
    angle1_tan = np.tan(np.deg2rad(angle1_deg)) if angle1_deg > 0 else 0.0
    angle2_cot = 1.0 / np.tan(np.deg2rad(angle2_deg)) if angle2_deg > 0 else 0.0
    radial = np.array(
        [
            r - width / 2 - height / 2 * angle2_cot,
            r + width / 2 - height / 2 * angle2_cot,
            r + width / 2 + height / 2 * angle2_cot,
            r - width / 2 + height / 2 * angle2_cot,
        ]
    )
    vertical = np.array(
        [
            z - height / 2 - width / 2 * angle1_tan,
            z - height / 2 + width / 2 * angle1_tan,
            z + height / 2 + width / 2 * angle1_tan,
            z + height / 2 - width / 2 * angle1_tan,
        ]
    )
    return np.column_stack([radial, vertical])


def _geometry_hex(geometry: Any) -> str:
    snapped = shapely.set_precision(geometry, GEOMETRY_PRECISION_M)
    simplified = shapely.simplify(
        snapped,
        GEOMETRY_PRECISION_M / 2,
        preserve_topology=True,
    )
    normalized = shapely.normalize(simplified)
    return shapely.to_wkb(normalized, byte_order=1, output_dimension=2).hex()


def active_component_geometry(
    r: np.ndarray,
    z: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
) -> str:
    """Canonical active-coil pack outline, independent of cell subdivision."""

    r, z, width, height = _finite_rows(r, z, width, height)
    cells = [
        box(
            float(rr - abs(ww) / 2),
            float(zz - abs(hh) / 2),
            float(rr + abs(ww) / 2),
            float(zz + abs(hh) / 2),
        )
        for rr, zz, ww, hh in zip(r, z, width, height, strict=True)
    ]
    return _geometry_hex(unary_union(cells).convex_hull)


def passive_component_geometry(
    r: np.ndarray,
    z: np.ndarray,
    width: np.ndarray,
    height: np.ndarray,
    angle1: np.ndarray,
    angle2: np.ndarray,
) -> str:
    """Canonical passive-section union, independent of element subdivision."""

    r, z, width, height, angle1, angle2 = _finite_rows(
        r,
        z,
        width,
        height,
        angle1,
        angle2,
    )
    sections = [
        Polygon(shaped_section_vertices(*map(float, values)))
        for values in zip(r, z, width, height, angle1, angle2, strict=True)
    ]
    return _geometry_hex(unary_union(sections))


def canonical_cycle(points: np.ndarray) -> list[list[float]]:
    """Canonicalize a closed path against start-index and direction changes."""

    rounded = np.round(np.asarray(points, dtype=np.float64), 5)
    if len(rounded) > 1 and np.array_equal(rounded[0], rounded[-1]):
        rounded = rounded[:-1]
    tuples = [tuple(float(value) for value in row) for row in rounded]
    if not tuples:
        return []
    candidates: list[tuple[tuple[float, ...], ...]] = []
    for direction in (tuples, list(reversed(tuples))):
        candidates.extend(
            tuple(direction[index:] + direction[:index])
            for index in range(len(direction))
        )
    return [list(row) for row in min(candidates)]


def _active_payload(group: zarr.Group) -> dict[str, str]:
    stems = sorted(
        key[: -len("_r")]
        for key in group.array_keys()
        if key.endswith("_r")
        and all(f"{key[:-2]}{suffix}" in group for suffix in _ACTIVE_SUFFIXES)
    )
    return {
        stem: active_component_geometry(
            _array(group, f"{stem}_r"),
            _array(group, f"{stem}_z"),
            _array(group, f"{stem}_width"),
            _array(group, f"{stem}_height"),
        )
        for stem in stems
    }


def _passive_payload(group: zarr.Group) -> dict[str, str]:
    stems = sorted(
        key[: -len("_r")]
        for key in group.array_keys()
        if key.endswith("_r")
        and all(
            f"{key[:-2]}{suffix}" in group for suffix in ("_z", "_width", "_height")
        )
    )
    payload: dict[str, str] = {}
    for stem in stems:
        r = _array(group, f"{stem}_r")
        zeros = np.zeros_like(r, dtype=float)
        across_r, across_z = f"{stem}_width", f"{stem}_height"
        if stem in _TRANSPOSED_EXTENT_FAMILIES:
            across_r, across_z = across_z, across_r
        payload[stem] = passive_component_geometry(
            r,
            _array(group, f"{stem}_z"),
            _array(group, across_r),
            _array(group, across_z),
            (
                _array(group, f"{stem}_shapeAngle1")
                if f"{stem}_shapeAngle1" in group
                else zeros
            ),
            (
                _array(group, f"{stem}_shapeAngle2")
                if f"{stem}_shapeAngle2" in group
                else zeros
            ),
        )
    return payload


def _nearest_values(
    target_r: np.ndarray,
    target_z: np.ndarray,
    source_r: np.ndarray,
    source_z: np.ndarray,
    values: np.ndarray,
    *,
    maximum_distance: float = 0.03,
) -> np.ndarray:
    result = np.full(np.asarray(target_r).shape, np.nan, dtype=float)
    source_r, source_z, values = _finite_rows(source_r, source_z, values)
    for index, (r, z) in enumerate(zip(target_r, target_z, strict=True)):
        distance = np.hypot(source_r - r, source_z - z)
        nearest = int(np.argmin(distance))
        if distance[nearest] <= maximum_distance:
            result[index] = float(values[nearest])
    return result


def _axis_offset_deg(angles_deg: np.ndarray, wanted_deg: float) -> np.ndarray:
    """Return how far each angle's axis lies from ``wanted_deg``, ignoring sense.

    A sensitive axis is the same line whichever way the probe is wired, so the
    comparison is modulo half a turn.
    """

    shifted = (np.asarray(angles_deg, dtype=float) - wanted_deg + 90.0) % 180.0
    return np.abs(shifted - 90.0)


def _probe_axis_angles(
    target_r: np.ndarray,
    target_z: np.ndarray,
    source_r: np.ndarray,
    source_z: np.ndarray,
    angles_deg: np.ndarray,
    *,
    radial: bool,
    maximum_distance: float = 0.03,
) -> np.ndarray:
    """Take each named probe's angle from the source probe on its own axis.

    Where two source probes share a position the placement match is a tie, and
    resolving it by array order hands both named families the same angle -- so one
    of them reports a component its probe never reads.  The requested axis breaks
    the tie; a position carrying a single source probe is unaffected.
    """

    result = np.full(np.asarray(target_r).shape, np.nan, dtype=float)
    source_r, source_z, angles_deg = _finite_rows(source_r, source_z, angles_deg)
    wanted = 0.0 if radial else 90.0
    for index, (r, z) in enumerate(zip(target_r, target_z, strict=True)):
        distance = np.hypot(source_r - r, source_z - z)
        within = np.flatnonzero(distance <= maximum_distance)
        if within.size == 0:
            continue
        offset = _axis_offset_deg(angles_deg[within], wanted)
        nearest = within[np.lexsort((distance[within], offset))[0]]
        result[index] = float(angles_deg[nearest])
    return result


def _source_angles_to_radians(values: Any) -> np.ndarray:
    """Convert catalog angles to the registry's canonical SI unit."""

    if SOURCE_ANGLE_UNIT != "degree":
        raise ValueError(f"unsupported source angle unit {SOURCE_ANGLE_UNIT!r}")
    return np.deg2rad(np.asarray(values, dtype=float))


def _parse_loop_name(name: str) -> tuple[str, int]:
    """Split a catalog flux-loop name into the coil it is mounted on and its number."""

    match = _LOOP_NAME_PATTERN.match(str(name))
    if match is None:
        raise ValueError(f"unrecognised flux-loop name {name!r}")
    if match[1] is not None:
        return CENTRE_COLUMN_MOUNT, int(match[1])
    side = "upper" if match[3] == "U" else "lower"
    return f"{match[2].lower()}_{side}", int(match[4])


def loop_mount(name: str) -> str:
    """Return the coil a flux loop's own name says it is mounted on."""

    return _parse_loop_name(name)[0]


def component_mount(component: str) -> str:
    """Return the coil set an active component belongs to.

    A loop encircles a coil set rather than one winding pack, so the two packs of
    a P2 half share a mounting: a loop around P2 upper links both of them and
    cannot be said to sit on either alone.
    """

    match = _COMPONENT_MOUNT_PATTERN.match(component)
    return component if match is None else f"{match[1]}_{match[2]}"


@dataclass(frozen=True)
class LoopPlacement:
    """Where one named flux loop is served, and where the catalog published it.

    ``mount`` is the coil the loop's name identifies and ``published_mount`` the
    coil the published coordinates actually sit nearest.  The two differing is the
    whole signal: a fixture is on the coil it is named for, so a loop beside a
    different coil is carrying a position that is not its own.  Both travel with
    the row because a refusal has to be able to name what was wrong.
    """

    name: str
    mount: str
    published_mount: str
    published_r: float
    published_z: float
    r: float
    z: float
    restored: bool

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation."""

        return {
            "mount": self.mount,
            "name": self.name,
            "published_mount": self.published_mount,
            "published_r": self.published_r,
            "published_z": self.published_z,
            "r": self.r,
            "restored": self.restored,
            "z": self.z,
        }


def _mount_outlines(outlines: Mapping[str, str]) -> dict[str, list[Any]]:
    """Group the active outlines by the coil set a loop would be mounted on."""

    grouped: dict[str, list[Any]] = {}
    for component, wkb_hex in sorted(outlines.items()):
        geometry = shapely.from_wkb(bytes.fromhex(wkb_hex))
        grouped.setdefault(component_mount(component), []).append(geometry)
    return grouped


def _nearest_mount(r: float, z: float, grouped: Mapping[str, list[Any]]) -> str:
    """Return the coil set a point sits nearest, or nothing when none is described."""

    point = shapely.Point(float(r), float(z))
    ranked = sorted(
        (min(geometry.distance(point) for geometry in group), mount)
        for mount, group in grouped.items()
    )
    return ranked[0][1] if ranked else ""


def position_mounts(
    positions: np.ndarray,
    outlines: Mapping[str, str],
) -> tuple[str, ...]:
    """Return the coil set each position in the poloidal plane sits nearest."""

    grouped = _mount_outlines(outlines)
    points = np.asarray(positions, dtype=float).reshape(-1, 2)
    return tuple(_nearest_mount(r, z, grouped) for r, z in points)


def placed_loop_positions(
    names: Sequence[str],
    radius: np.ndarray,
    height: np.ndarray,
    outlines: Mapping[str, str],
    reconstruction: np.ndarray,
) -> tuple[LoopPlacement, ...]:
    """Serve each named loop from the coil it is named for, restoring copied blocks.

    A flux loop is a fixture wound onto a coil, so the coil it sits nearest is the
    coil its name identifies.  That relation holds for every loop in both of the
    archive's tables but one block, and it needs no tolerance to apply: a block
    whose coordinates were transcribed from another block is not displaced by
    millimetres, it is beside a different coil entirely.

    A block that fails the relation is served from the reconstruction's own loop
    table instead, which carries the same loops at positions of its own.  The rows
    it may be served from are the ones sitting nearest the coil the block is named
    for and not already occupied by a loop the relation accepts; the count has to
    match the block exactly.  Anything else -- no free row, too many, a coil the
    catalog does not describe -- leaves the block exactly as published, because a
    replacement that is not identified is a sensor pose invented rather than
    recovered, and the position join refusing a channel is the better failure.
    """

    grouped = _mount_outlines(outlines)
    rows = [_parse_loop_name(name) for name in names]
    published = [
        _nearest_mount(radius[index], height[index], grouped)
        for index in range(len(names))
    ]
    settled = [mount == published[index] for index, (mount, _) in enumerate(rows)]

    table = np.asarray(reconstruction, dtype=float).reshape(-1, 2)
    occupied = set()
    for index, ok in enumerate(settled):
        if ok and table.size:
            distance = np.hypot(
                table[:, 0] - radius[index], table[:, 1] - height[index]
            )
            occupied.add(int(np.argmin(distance)))
    free: dict[str, list[int]] = {}
    for row in range(table.shape[0]):
        if row in occupied:
            continue
        mount = _nearest_mount(table[row, 0], table[row, 1], grouped)
        free.setdefault(mount, []).append(row)

    replacement: dict[int, int] = {}
    unsettled: dict[str, list[int]] = {}
    for index, ok in enumerate(settled):
        if not ok:
            unsettled.setdefault(rows[index][0], []).append(index)
    for mount, members in unsettled.items():
        candidates = free.get(mount, [])
        if len(candidates) != len(members):
            continue
        order = sorted(members, key=lambda index: rows[index][1])
        replacement.update(zip(order, candidates, strict=True))

    placements = []
    for index, name in enumerate(names):
        row = replacement.get(index)
        served = (
            (float(radius[index]), float(height[index]))
            if row is None
            else (float(table[row, 0]), float(table[row, 1]))
        )
        placements.append(
            LoopPlacement(
                name=str(name),
                mount=rows[index][0],
                published_mount=published[index],
                published_r=float(radius[index]),
                published_z=float(height[index]),
                r=served[0],
                z=served[1],
                restored=row is not None,
            )
        )
    return tuple(placements)


def shot_loop_placements(
    shot: int,
    level1_root: Path = DEFAULT_LEVEL1_ROOT,
    level2_root: Path = DEFAULT_LEVEL2_ROOT,
) -> tuple[LoopPlacement, ...]:
    """Return one shot's loop placements, both tables read from the catalogs."""

    level1 = zarr.open_group(str(level1_root / f"{shot}.zarr"), mode="r")
    level2 = zarr.open_group(str(level2_root / f"{shot}.zarr"), mode="r")
    magnetics = level2["magnetics"]
    return placed_loop_positions(
        [str(name) for name in _array(magnetics, "flux_loop_geometry_channel")],
        _array(magnetics, "flux_loop_r").astype(float),
        _array(magnetics, "flux_loop_z").astype(float),
        _active_payload(level2["pf_active"]),
        _reconstruction_loops(level1["efm"]),
    )


def _reconstruction_loops(efm: zarr.Group) -> np.ndarray:
    """Return the reconstruction's own loop table, or nothing where it is absent."""

    if "silop_r" not in efm or "silop_z" not in efm:
        return np.zeros((0, 2), dtype=float)
    return np.column_stack(
        [_array(efm, "silop_r").astype(float), _array(efm, "silop_z").astype(float)]
    )


def _magnetics_payload(
    level1: zarr.Group,
    level2: zarr.Group,
    outlines: Mapping[str, str] | None,
) -> dict[str, Any]:
    efm = level1["efm"]
    probes: list[dict[str, Any]] = []
    for family in ("ccbv", "obr", "obv"):
        stem = f"b_field_pol_probe_{family}"
        r = _array(level2, f"{stem}_r").astype(float)
        z = _array(level2, f"{stem}_z").astype(float)
        angles = _probe_axis_angles(
            r,
            z,
            _array(efm, "magpr_r"),
            _array(efm, "magpr_z"),
            _array(efm, "magpr_ang"),
            radial=family in _RADIAL_PROBE_FAMILIES,
        )
        lengths = _array(level2, f"{stem}_length").astype(float)
        phi_1 = _array(level2, f"{stem}_phi_1").astype(float)
        phi_2 = _array(level2, f"{stem}_phi_2").astype(float)
        for values in zip(r, z, angles, lengths, phi_1, phi_2, strict=True):
            rr, zz, angle_deg, length, start_deg, end_deg = values
            angle_rad, first_phi_rad, second_phi_rad = _source_angles_to_radians(
                [angle_deg, start_deg, end_deg]
            )
            probes.append(
                {
                    "family": family,
                    "pose": [
                        round(float(rr), 5),
                        round(float(zz), 5),
                        round(float(angle_rad), 5),
                        round(float(length), 5),
                    ],
                    "position_phi_candidates": [
                        round(float(first_phi_rad), 5),
                        round(float(second_phi_rad), 5),
                    ],
                }
            )

    loop_r = _array(level2, "flux_loop_r").astype(float)
    loop_z = _array(level2, "flux_loop_z").astype(float)
    if outlines is not None:
        placements = placed_loop_positions(
            [str(name) for name in _array(level2, "flux_loop_geometry_channel")],
            loop_r,
            loop_z,
            outlines,
            _reconstruction_loops(efm),
        )
        loop_r = np.asarray([row.r for row in placements], dtype=float)
        loop_z = np.asarray([row.z for row in placements], dtype=float)
    if "silop_dphi" in efm:
        loop_span = _nearest_values(
            loop_r,
            loop_z,
            _array(efm, "silop_r"),
            _array(efm, "silop_z"),
            _array(efm, "silop_dphi"),
        )
    else:
        # Later L1 stores publish 2π for every named L2 loop.  That catalog-wide
        # reference completes earlier stores where the span field is absent.
        loop_span = np.full(loop_r.shape, 2 * np.pi)
    loops = sorted(np.round(np.column_stack([loop_r, loop_z, loop_span]), 5).tolist())

    paths: dict[str, list[list[list[float]]]] = {}
    for family in ("l", "m", "u"):
        stem = f"b_field_tor_probe_saddle_{family}"
        r = _array(level2, f"{stem}_r")
        z = _array(level2, f"{stem}_z")
        phi = _source_angles_to_radians(_array(level2, f"{stem}_phi"))
        paths[family] = [
            canonical_cycle(np.column_stack([rr, zz, pp]))
            for rr, zz, pp in zip(r, z, phi, strict=True)
        ]

    points: dict[str, list[list[float]]] = {}
    for family in ("cc", "omv"):
        stem = f"b_field_pol_probe_{family}"
        if f"{stem}_r" in level2:
            columns = [
                _array(level2, f"{stem}_r"),
                _array(level2, f"{stem}_z"),
                _source_angles_to_radians(_array(level2, f"{stem}_phi")),
            ]
            points[f"poloidal_{family}"] = sorted(
                np.round(np.column_stack(columns), 5).tolist()
            )
    stem = "b_field_tor_probe_cc"
    if f"{stem}_r" in level2:
        points["toroidal_cc"] = sorted(
            np.round(
                np.column_stack(
                    [
                        _array(level2, f"{stem}_r"),
                        _array(level2, f"{stem}_z"),
                        _source_angles_to_radians(_array(level2, f"{stem}_phi")),
                    ]
                ),
                5,
            ).tolist()
        )
    return {
        "poloidal_probes": probes,
        "flux_loops": loops,
        "saddle_paths": paths,
        "additional_points": points,
    }


def _xray_payload(level2: zarr.Group) -> dict[str, list[list[float]]]:
    if "soft_x_rays" not in level2:
        return {}
    group = level2["soft_x_rays"]
    stems = sorted(
        key[: -len("_origin_r")]
        for key in group.array_keys()
        if key.endswith("_origin_r")
    )
    payload: dict[str, list[list[float]]] = {}
    for stem in stems:
        keys = (
            f"{stem}_origin_r",
            f"{stem}_origin_z",
            f"{stem}_endpoint_r",
            f"{stem}_endpoint_z",
            f"{stem}_phi",
        )
        if all(key in group for key in keys):
            columns = [_array(group, key) for key in keys[:-1]]
            columns.append(_source_angles_to_radians(_array(group, keys[-1])))
            payload[stem] = sorted(
                np.round(
                    np.column_stack(columns),
                    5,
                ).tolist()
            )
    return payload


def physical_snapshot(
    shot: int,
    level1_root: Path = DEFAULT_LEVEL1_ROOT,
    level2_root: Path = DEFAULT_LEVEL2_ROOT,
    *,
    place_loops: bool = False,
) -> dict[str, Any]:
    """Build the canonical physical geometry payload for a complete shot.

    ``place_loops`` serves each flux loop from the coil its own name identifies
    rather than from the coordinates the level-2 catalog published, which repairs
    the block whose coordinates were transcribed from another block -- see
    :func:`placed_loop_positions`.

    It is off by default, and the default is the whole point.  This payload's hash
    is the identity consumers select a machine by and pin by value, so the reader
    and the packaged file have to remain one statement about the machine: a reader
    that quietly built a different payload would make the next census report a
    hardware reconfiguration that never happened.  Moving the loops therefore
    moves the identity, which is a republication rather than a bug fix, and it
    happens when the packaged file is regenerated with it.  Until then the
    correction is available to whatever measures against it and absent from what
    is published.
    """

    level1 = zarr.open_group(str(level1_root / f"{shot}.zarr"), mode="r")
    level2 = zarr.open_group(str(level2_root / f"{shot}.zarr"), mode="r")
    wall = level2["wall"]
    active = _active_payload(level2["pf_active"])
    payload: dict[str, Any] = {
        "active_components": active,
        "passive_components": _passive_payload(level2["pf_passive"]),
        "limiter": canonical_cycle(
            np.column_stack(
                [
                    _array(wall, "limiter_r"),
                    _array(wall, "limiter_z"),
                ]
            )
        ),
        "magnetics": _magnetics_payload(
            level1, level2["magnetics"], active if place_loops else None
        ),
        "soft_x_ray_chords": _xray_payload(level2),
    }
    return payload


def physical_digest(payload: dict[str, Any]) -> str:
    """Return the identity hash of a canonical physical snapshot."""

    return stable_digest(payload, length=16)


def registry_payload(
    geometry: Mapping[str, Any],
    incomplete_evidence: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the immutable registry payload from census evidence."""

    digest = physical_digest(dict(geometry))
    payload: dict[str, Any] = {
        "schema": "nova-machine-geometry-registry",
        "dd_version": "4.1.1",
        "units": {"length": "m", "angle": "rad"},
        "tolerances": {
            "length_m": GEOMETRY_PRECISION_M,
            "angle_rad": ANGLE_PRECISION_RAD,
        },
        "configurations": {
            digest: {
                "physical_digest": digest,
                "geometry": geometry,
                "authoring_gaps": [
                    "active turns, polarity, and circuit topology are not sourced",
                    (
                        "passive material, resistance, and electrical topology "
                        "are not sourced"
                    ),
                    "detailed toroidal-field winding geometry is not sourced",
                    "independent toroidal probe orientation is not sourced",
                    (
                        "poloidal probe channel-to-bank toroidal position "
                        "assignment is not sourced"
                    ),
                    "saddle traversal sign is not sourced",
                ],
            }
        },
        "representation_aliases": {
            "9425ae4a8bf3bc15": digest,
            "edd753d282903679": digest,
            "1cb6f2ee742c4ee4": digest,
        },
        "ranges": [
            {
                "first_shot": 11695,
                "last_shot": 11765,
                "physical_digest": digest,
                "evidence": "inherited",
            },
            {
                "first_shot": 11766,
                "last_shot": 30471,
                "physical_digest": digest,
                "evidence": "observed",
            },
            {
                "first_shot": 30472,
                "last_shot": 30473,
                "physical_digest": digest,
                "evidence": "inherited",
            },
        ],
        "incomplete_evidence": sorted(
            (dict(row) for row in incomplete_evidence),
            key=lambda row: int(row["shot"]),
        ),
        "provenance": {
            "catalogs": {
                "level1": {
                    "root": str(DEFAULT_LEVEL1_ROOT),
                    "stores": 17111,
                    "first_shot": 11695,
                    "last_shot": 30473,
                },
                "level2": {
                    "root": str(DEFAULT_LEVEL2_ROOT),
                    "stores": 11573,
                    "first_shot": 11766,
                    "last_shot": 30471,
                },
            },
            "representative_shot": 11766,
            "complete_geometry_shots": 11556,
            "incomplete_geometry_shots": 17,
            "out_of_scope_source_gaps": [
                (
                    "installed poses beyond magnetics and soft-X-ray chords "
                    "are not sourced"
                )
            ],
            "source_census_physical_digest": "67f789d3d8b40135",
            "source_census_angle_unit": (
                "degree for diagnostic positions and orientations; "
                "radian for flux-loop span"
            ),
            "canonicalization_note": (
                "The registry normalizes degree-sourced diagnostic angles to radians "
                "at ingestion; flux-loop spans are already radians, and all length "
                "coordinates retain their source values."
            ),
            "representation_counts": {
                "9425ae4a8bf3bc15": 392,
                "edd753d282903679": 582,
                "1cb6f2ee742c4ee4": 10042,
            },
            "superseded_physical_digests": SUPERSEDED_PHYSICAL_DIGESTS,
        },
    }
    payload["registry_digest"] = stable_digest(payload)
    return payload


def observed_ranges(
    records: list[tuple[int, str | None]],
) -> list[dict[str, int | str]]:
    """Collapse observed shots into signature runs without treating holes as changes."""

    complete = [(shot, signature) for shot, signature in records if signature]
    if not complete:
        return []
    ranges: list[dict[str, int | str]] = []
    start = complete[0][0]
    previous_shot, previous_signature = complete[0]
    count = 1
    for shot, signature in complete[1:]:
        if signature != previous_signature:
            ranges.append(
                {
                    "first_observed_shot": start,
                    "last_observed_shot": previous_shot,
                    "physical_digest": str(previous_signature),
                    "observed_shots": count,
                }
            )
            start = shot
            count = 0
        previous_shot = shot
        previous_signature = signature
        count += 1
    ranges.append(
        {
            "first_observed_shot": start,
            "last_observed_shot": previous_shot,
            "physical_digest": str(previous_signature),
            "observed_shots": count,
        }
    )
    return ranges


def scan_catalog(
    level1_root: Path = DEFAULT_LEVEL1_ROOT,
    level2_root: Path = DEFAULT_LEVEL2_ROOT,
    *,
    workers: int = 12,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 100,
) -> dict[str, Any]:
    """Scan every L2 catalog shot and recover physical configuration runs."""

    if workers < 1:
        raise ValueError("workers must be positive")
    if checkpoint_every < 1:
        raise ValueError("checkpoint interval must be positive")
    shots = sorted(int(path.stem) for path in level2_root.glob("*.zarr"))
    source_rows = (
        _read_fingerprint_checkpoint(checkpoint_path, level1_root, level2_root)
        if checkpoint_path
        else []
    )
    completed = {row.shot for row in source_rows}
    pending = [shot for shot in shots if shot not in completed]

    def fingerprint(shot: int) -> SourceFingerprint:
        return source_fingerprint(shot, level1_root, level2_root)

    def pending_fingerprints() -> Iterator[SourceFingerprint]:
        if workers == 1:
            yield from map(fingerprint, pending)
            return
        with ThreadPoolExecutor(max_workers=workers) as executor:
            yield from executor.map(fingerprint, pending)

    for row in pending_fingerprints():
        source_rows.append(row)
        if checkpoint_path and len(source_rows) % checkpoint_every == 0:
            _write_fingerprint_checkpoint(
                checkpoint_path,
                source_rows,
                level1_root,
                level2_root,
            )
    source_rows.sort(key=lambda row: row.shot)
    if checkpoint_path:
        _write_fingerprint_checkpoint(
            checkpoint_path,
            source_rows,
            level1_root,
            level2_root,
        )

    representatives: dict[str, int] = {}
    for row in source_rows:
        if row.complete and row.source_digest is not None:
            representatives.setdefault(row.source_digest, row.shot)

    canonical_by_source: dict[str, str] = {}
    payload_by_physical: dict[str, dict[str, Any]] = {}
    for source_digest, shot in representatives.items():
        payload = physical_snapshot(shot, level1_root, level2_root)
        digest = physical_digest(payload)
        canonical_by_source[source_digest] = digest
        payload_by_physical.setdefault(digest, payload)

    records = [
        (
            row.shot,
            canonical_by_source.get(row.source_digest or "") if row.complete else None,
        )
        for row in source_rows
    ]
    representation_counts: dict[str, int] = {}
    missing_counts: dict[str, int] = {}
    for row in source_rows:
        if row.representation_digest:
            representation_counts[row.representation_digest] = (
                representation_counts.get(row.representation_digest, 0) + 1
            )
        for missing in row.missing:
            missing_counts[missing] = missing_counts.get(missing, 0) + 1

    ranges = observed_ranges(records)
    physical_counts: dict[str, int] = {}
    for _, digest in records:
        if digest:
            physical_counts[digest] = physical_counts.get(digest, 0) + 1
    return {
        "schema": "mast-catalog-physical-geometry",
        "level1_root": str(level1_root),
        "level2_root": str(level2_root),
        "catalog_shots": len(shots),
        "complete_geometry_shots": sum(row.complete for row in source_rows),
        "missing": missing_counts,
        "incomplete_evidence": [
            {"shot": row.shot, "missing": list(row.missing)}
            for row in source_rows
            if not row.complete
        ],
        "source_snapshot_count": len(representatives),
        "representation_signatures": representation_counts,
        "physical_configuration_counts": physical_counts,
        "observed_ranges": ranges,
        "representative_shots": {
            digest: next(
                shot
                for source, shot in representatives.items()
                if canonical_by_source[source] == digest
            )
            for digest in payload_by_physical
        },
        "payload_summary": {
            digest: {
                "active_components": len(payload["active_components"]),
                "passive_components": len(payload["passive_components"]),
                "poloidal_probes": len(payload["magnetics"]["poloidal_probes"]),
                "flux_loops": len(payload["magnetics"]["flux_loops"]),
                "saddle_paths": sum(
                    len(paths)
                    for paths in payload["magnetics"]["saddle_paths"].values()
                ),
                "soft_x_ray_chords": sum(
                    len(chords) for chords in payload["soft_x_ray_chords"].values()
                ),
            }
            for digest, payload in payload_by_physical.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover physical MAST geometry from the L1 and L2 catalogs."
    )
    parser.add_argument("--level1-root", type=Path, default=DEFAULT_LEVEL1_ROOT)
    parser.add_argument("--level2-root", type=Path, default=DEFAULT_LEVEL2_ROOT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="resume source fingerprinting from an atomic checkpoint",
    )
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--registry-output",
        type=Path,
        help="also write the canonical shot registry from the census",
    )
    args = parser.parse_args()
    report = scan_catalog(
        args.level1_root,
        args.level2_root,
        workers=args.workers,
        checkpoint_path=args.checkpoint,
        checkpoint_every=args.checkpoint_every,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    if args.registry_output:
        representative_shots = report["representative_shots"]
        if len(representative_shots) != 1:
            raise ValueError(
                "canonical registry requires exactly one physical configuration"
            )
        representative_shot = next(iter(representative_shots.values()))
        geometry = physical_snapshot(
            representative_shot,
            args.level1_root,
            args.level2_root,
        )
        payload = registry_payload(geometry, report["incomplete_evidence"])
        args.registry_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
