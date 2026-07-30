"""Recover subdivision-independent MAST geometry from the training catalogs.

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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
import zarr
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

DEFAULT_LEVEL1_ROOT = Path("/work/projects/imas_gpu/mast/level1/shots")
DEFAULT_LEVEL2_ROOT = Path("/work/projects/imas_gpu/mast/level2/shots")
GEOMETRY_PRECISION_M = 1e-5
ANGLE_PRECISION_DEG = 1e-4

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


@dataclass(frozen=True)
class SourceFingerprint:
    """Static catalog hashes for one shot."""

    shot: int
    source_digest: str | None
    representation_digest: str | None
    complete: bool
    missing: tuple[str, ...]


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
        payload[stem] = passive_component_geometry(
            r,
            _array(group, f"{stem}_z"),
            _array(group, f"{stem}_width"),
            _array(group, f"{stem}_height"),
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


def _magnetics_payload(level1: zarr.Group, level2: zarr.Group) -> dict[str, Any]:
    efm = level1["efm"]
    probes: list[dict[str, Any]] = []
    for family in ("ccbv", "obr", "obv"):
        stem = f"b_field_pol_probe_{family}"
        r = _array(level2, f"{stem}_r").astype(float)
        z = _array(level2, f"{stem}_z").astype(float)
        angles = _nearest_values(
            r,
            z,
            _array(efm, "magpr_r"),
            _array(efm, "magpr_z"),
            _array(efm, "magpr_ang"),
        )
        lengths = _array(level2, f"{stem}_length").astype(float)
        phi_1 = _array(level2, f"{stem}_phi_1").astype(float)
        phi_2 = _array(level2, f"{stem}_phi_2").astype(float)
        for values in zip(r, z, angles, lengths, phi_1, phi_2, strict=True):
            probes.append(
                {
                    "family": family,
                    "pose": np.round(values, 5).tolist(),
                }
            )

    loop_r = _array(level2, "flux_loop_r").astype(float)
    loop_z = _array(level2, "flux_loop_z").astype(float)
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
        phi = _array(level2, f"{stem}_phi")
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
                _array(level2, f"{stem}_phi"),
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
                        _array(level2, f"{stem}_phi"),
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
            payload[stem] = sorted(
                np.round(
                    np.column_stack([_array(group, key) for key in keys]),
                    5,
                ).tolist()
            )
    return payload


def physical_snapshot(
    shot: int,
    level1_root: Path = DEFAULT_LEVEL1_ROOT,
    level2_root: Path = DEFAULT_LEVEL2_ROOT,
) -> dict[str, Any]:
    """Build the canonical physical geometry payload for a complete shot."""

    level1 = zarr.open_group(str(level1_root / f"{shot}.zarr"), mode="r")
    level2 = zarr.open_group(str(level2_root / f"{shot}.zarr"), mode="r")
    wall = level2["wall"]
    payload: dict[str, Any] = {
        "active_components": _active_payload(level2["pf_active"]),
        "passive_components": _passive_payload(level2["pf_passive"]),
        "limiter": canonical_cycle(
            np.column_stack(
                [
                    _array(wall, "limiter_r"),
                    _array(wall, "limiter_z"),
                ]
            )
        ),
        "magnetics": _magnetics_payload(level1, level2["magnetics"]),
        "soft_x_ray_chords": _xray_payload(level2),
    }
    return payload


def physical_digest(payload: dict[str, Any]) -> str:
    """Return the identity hash of a canonical physical snapshot."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


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
) -> dict[str, Any]:
    """Scan every L2 catalog shot and recover physical configuration runs."""

    shots = sorted(int(path.stem) for path in level2_root.glob("*.zarr"))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        source_rows = list(
            executor.map(
                lambda shot: source_fingerprint(shot, level1_root, level2_root),
                shots,
            )
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = scan_catalog(args.level1_root, args.level2_root, workers=args.workers)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
