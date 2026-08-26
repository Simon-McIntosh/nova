"""Build and validate real MAST inputs for the single-device throughput run.

The benchmark widths are nested prefixes of one deterministic population.  Each
row is a distinct equilibrium time inside the FAIR-MAST catalog's published
flat-top interval and resolves to the nearest row returned by Nova's corrected
solve-input reader.  The artifact stores identities and provenance rather than
copying the arrays: the benchmark loader can recover every row from its shot and
input index, while the recorded digests make substitution or repetition visible.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from nova.imas.mast_solve_inputs import read_corrected_solve_inputs


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/mast-catalog-gpu-solve/mast-catalog-throughput-inputs.json"
)
DEFAULT_CATALOG_INDEX = "https://mastapp.site/parquet/level2/shots"
DEFAULT_INPUT_STORE = Path("/work/projects/imas_gpu/mast/level1/shots")
DEFAULT_EQUILIBRIUM_STORE = Path("/work/projects/imas_gpu/mast/level2/shots")
DEFAULT_ARTIFACT_CACHE = Path.home() / ".cache/mast-artifact-ef"
DEFAULT_ARTIFACT_DIGEST = (
    "sha256:b41c076e1fb7e16dabe3bada2f5d890125a857c400ce7599dfa488e8ebef90e4"
)
BENCHMARK_WIDTHS = (256, 512, 1024, 2048, 4096)
CATALOG_COLUMNS = (
    "shot_id",
    "campaign",
    "plasma_flat_top_start_time",
    "plasma_flat_top_end_time",
    "divertor_config",
    "plasma_shape",
)
ARRAY_NAMES = ("coil_currents_a", "sensor_signals", "plasma_current_a")
SCHEMA = "nova-mast-catalog-throughput-inputs"
SCHEMA_REVISION = 1
MAX_CLOCK_SEPARATION_S = 1.0e-3


class ManifestError(ValueError):
    """Raised when an input manifest is incomplete or internally inconsistent."""


def _canonical_bytes(value: object) -> bytes:
    """Return stable JSON bytes for a digest boundary."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _digest(value: object) -> str:
    """Return a tagged SHA-256 digest of a JSON-compatible value."""

    return f"sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _file_digest(path: Path) -> str:
    """Return a tagged SHA-256 digest of one file."""

    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _array_digest(value: np.ndarray | np.generic | float) -> str:
    """Digest one numeric input including its shape and dtype."""

    array = np.ascontiguousarray(np.asarray(value))
    header = _canonical_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
    checksum = hashlib.sha256()
    checksum.update(header)
    checksum.update(array.tobytes(order="C"))
    return f"sha256:{checksum.hexdigest()}"


def _json_value(value: Any) -> Any:
    """Convert pandas and NumPy scalar values into strict JSON values."""

    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _state_digest(arrays: Mapping[str, Mapping[str, Any]]) -> str:
    """Identify a numerical solve state without using its shot or time labels."""

    return _digest({name: arrays[name]["sha256"] for name in ARRAY_NAMES})


def _batch_digest(slices: list[dict[str, Any]], width: int) -> str:
    """Identify one nested benchmark population by its ordered state identities."""

    return _digest([row["state_digest"] for row in slices[:width]])


def _artifact_manifest(cache: Path, digest: str) -> tuple[Path, dict[str, Any]]:
    """Resolve and independently verify a content-addressed artifact manifest."""

    raw_digest = digest.removeprefix("sha256:")
    path = cache / "sha256" / raw_digest / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"machine-artifact manifest is absent: {path}")
    if _file_digest(path) != f"sha256:{raw_digest}":
        raise ManifestError("machine-artifact directory does not match manifest bytes")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "nova-mast-machine-artifact":
        raise ManifestError("resolved artifact is not a MAST machine description")
    return path, manifest


def _covered_by_artifact(shot: int, ranges: Iterable[Mapping[str, Any]]) -> bool:
    """Return whether one shot is covered by an artifact's declared ranges."""

    return any(
        int(row["first_shot"]) <= shot <= int(row["last_shot"]) for row in ranges
    )


def _nearest_indices(clock: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return nearest monotonically indexed clock rows for target times."""

    if clock.ndim != 1 or clock.size == 0 or np.any(np.diff(clock) <= 0.0):
        raise ManifestError("corrected input clock is not strictly increasing")
    right = np.searchsorted(clock, targets, side="left")
    right = np.clip(right, 0, clock.size - 1)
    left = np.clip(right - 1, 0, clock.size - 1)
    use_left = np.abs(clock[left] - targets) <= np.abs(clock[right] - targets)
    return np.where(use_left, left, right)


def _input_configuration(inputs: Any) -> dict[str, Any]:
    """Describe the fixed-shape row schema consumed by one compiled batch."""

    configuration = {
        "coil_channels": list(inputs.coil_channels),
        "sensor_channels": list(inputs.sensor_channels),
        "sensor_units": list(inputs.sensor_units),
        "arrays": {
            "coil_currents_a": {
                "dtype": np.asarray(inputs.coil_currents_a).dtype.str,
                "shape": [int(np.asarray(inputs.coil_currents_a).shape[1])],
                "unit": "A",
            },
            "sensor_signals": {
                "dtype": np.asarray(inputs.sensor_signals).dtype.str,
                "shape": [int(np.asarray(inputs.sensor_signals).shape[1])],
                "unit_by_column": "sensor_units",
            },
            "plasma_current_a": {
                "dtype": np.asarray(inputs.plasma_current_a).dtype.str,
                "shape": [],
                "unit": "A",
            },
        },
    }
    configuration["digest"] = _digest(configuration)
    return configuration


def _read_candidate_shot(arguments: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
    """Read and identify all finite flat-top inputs from one real shot."""

    import zarr

    catalog_row, input_store_text, equilibrium_store_text = arguments
    shot = int(catalog_row["shot_id"])
    input_store = Path(input_store_text)
    equilibrium_store = Path(equilibrium_store_text)
    equilibrium_path = equilibrium_store / f"{shot}.zarr"
    input_path = input_store / f"{shot}.zarr"
    if not equilibrium_path.is_dir() or not input_path.is_dir():
        return {"shot": shot, "skip": "required shot store is absent"}

    try:
        root = zarr.open_group(equilibrium_path, mode="r")
        equilibrium_times = np.asarray(root["equilibrium/time"], dtype=float)
    except (KeyError, OSError, ValueError) as error:
        return {"shot": shot, "skip": f"equilibrium time is unreadable: {error}"}

    start = float(catalog_row["plasma_flat_top_start_time"])
    end = float(catalog_row["plasma_flat_top_end_time"])
    equilibrium_indices = np.flatnonzero(
        np.isfinite(equilibrium_times)
        & (equilibrium_times >= start)
        & (equilibrium_times <= end)
    )
    if not equilibrium_indices.size:
        return {"shot": shot, "skip": "no equilibrium time lies in the flat top"}

    try:
        inputs = read_corrected_solve_inputs(shot, store=input_store)
    except Exception as error:  # the skip receipt retains the concrete read failure
        return {"shot": shot, "skip": f"corrected input read failed: {error}"}

    clock = np.asarray(inputs.time_s, dtype=float)
    target_times = equilibrium_times[equilibrium_indices]
    input_indices = _nearest_indices(clock, target_times)
    configuration = _input_configuration(inputs)
    slices = []
    for equilibrium_index, input_index, target_time in zip(
        equilibrium_indices, input_indices, target_times, strict=True
    ):
        input_time = float(clock[input_index])
        if abs(input_time - float(target_time)) > MAX_CLOCK_SEPARATION_S:
            continue
        values = {
            "coil_currents_a": np.asarray(inputs.coil_currents_a[input_index]),
            "sensor_signals": np.asarray(inputs.sensor_signals[input_index]),
            "plasma_current_a": np.asarray(inputs.plasma_current_a[input_index]),
        }
        if not all(np.all(np.isfinite(value)) for value in values.values()):
            continue
        arrays = {
            name: {
                "source": f"read_corrected_solve_inputs.{name}",
                "row_index": int(input_index),
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": _array_digest(value),
            }
            for name, value in values.items()
        }
        slices.append(
            {
                "shot": shot,
                "equilibrium_index": int(equilibrium_index),
                "input_index": int(input_index),
                "equilibrium_time_s": float(target_time),
                "input_time_s": input_time,
                "clock_separation_s": abs(input_time - float(target_time)),
                "arrays": arrays,
                "state_digest": _state_digest(arrays),
            }
        )

    provenance = [asdict(row) for row in inputs.provenance]
    return {
        "shot": shot,
        "campaign": str(catalog_row["campaign"]),
        "configuration": {
            "divertor_config": str(catalog_row["divertor_config"]),
            "plasma_shape": _json_value(catalog_row.get("plasma_shape")),
        },
        "flat_top": {"start_s": start, "end_s": end},
        "input_configuration": configuration,
        "input_store": str(input_store),
        "equilibrium_store": str(equilibrium_store),
        "source_groups": provenance,
        "source_groups_digest": _digest(provenance),
        "corrections_digest": _digest(
            [
                {
                    "channel": row.channel,
                    "shot": row.shot,
                    "scale": row.scale,
                    "disposition": row.disposition,
                }
                for row in inputs.corrections
            ]
        ),
        "slices": slices,
    }


def _catalog_rows(
    catalog_index: str, ranges: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], str]:
    """Read deterministic catalog metadata rows eligible for real-input staging."""

    import pandas as pd

    frame = pd.read_parquet(catalog_index, columns=list(CATALOG_COLUMNS))
    records = []
    identity_rows = []
    for raw in frame.to_dict(orient="records"):
        row = {key: _json_value(value) for key, value in raw.items()}
        identity_rows.append(row)
        shot = int(row["shot_id"])
        if not _covered_by_artifact(shot, ranges):
            continue
        if row["campaign"] in (None, "", "Unknown"):
            continue
        start = row["plasma_flat_top_start_time"]
        end = row["plasma_flat_top_end_time"]
        if start is None or end is None or float(end) <= float(start):
            continue
        if row["divertor_config"] in (None, ""):
            continue
        records.append(row)
    records.sort(key=lambda row: (str(row["campaign"]), int(row["shot_id"])))
    identity_rows.sort(key=lambda row: int(row["shot_id"]))
    return records, _digest(identity_rows)


def _choose_population(
    results: list[dict[str, Any]], required: int
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    """Choose the largest fixed-shape real-input cohort and its first rows."""

    by_configuration: dict[str, list[dict[str, Any]]] = {}
    shots: dict[int, dict[str, Any]] = {}
    configurations: dict[str, dict[str, Any]] = {}
    for result in results:
        if result.get("skip") or not result.get("slices"):
            continue
        configuration = result["input_configuration"]
        digest = configuration["digest"]
        configurations[digest] = configuration
        by_configuration.setdefault(digest, []).extend(result["slices"])
        shots[int(result["shot"])] = result
    if not by_configuration:
        raise ManifestError("no real flat-top input rows were readable")
    digest, candidates = max(
        by_configuration.items(), key=lambda item: (len(item[1]), item[0])
    )
    candidates.sort(
        key=lambda row: (
            int(row["shot"]),
            int(row["equilibrium_index"]),
            int(row["input_index"]),
        )
    )
    if len(candidates) < required:
        raise ManifestError(
            f"largest fixed-shape cohort has {len(candidates)} rows; "
            f"{required} required"
        )
    selected = candidates[:required]
    selected_shots = {int(row["shot"]) for row in selected}
    return (
        configurations[digest],
        selected,
        {shot: result for shot, result in shots.items() if shot in selected_shots},
    )


def build_manifest(
    *,
    catalog_index: str = DEFAULT_CATALOG_INDEX,
    input_store: Path = DEFAULT_INPUT_STORE,
    equilibrium_store: Path = DEFAULT_EQUILIBRIUM_STORE,
    artifact_cache: Path = DEFAULT_ARTIFACT_CACHE,
    artifact_digest: str = DEFAULT_ARTIFACT_DIGEST,
    required: int = max(BENCHMARK_WIDTHS),
    candidate_shots: int = 192,
    workers: int = 8,
) -> dict[str, Any]:
    """Build a strict manifest from real archive rows without replication."""

    artifact_path, artifact = _artifact_manifest(artifact_cache, artifact_digest)
    catalog_rows, catalog_digest = _catalog_rows(catalog_index, artifact["shot_ranges"])
    available = [
        row
        for row in catalog_rows
        if (input_store / f"{int(row['shot_id'])}.zarr").is_dir()
        and (equilibrium_store / f"{int(row['shot_id'])}.zarr").is_dir()
    ][:candidate_shots]
    arguments = [(row, str(input_store), str(equilibrium_store)) for row in available]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_read_candidate_shot, arguments, chunksize=1))
    configuration, slices, shots = _choose_population(results, required)

    for ordinal, row in enumerate(slices):
        shot = shots[int(row["shot"])]
        row.update(
            {
                "ordinal": ordinal,
                "campaign": shot["campaign"],
                "configuration": shot["configuration"],
                "machine_artifact_digest": artifact_digest,
                "input_configuration_digest": configuration["digest"],
                "source_provenance_ref": f"shots.{row['shot']}",
                "origin": "real_archive_row",
                "synthetic": False,
                "broadcast": False,
                "tiled": False,
            }
        )

    shot_records = {
        str(shot): {
            "campaign": result["campaign"],
            "configuration": result["configuration"],
            "flat_top": result["flat_top"],
            "input_store": result["input_store"],
            "input_shot_path": f"{result['input_store']}/{shot}.zarr",
            "equilibrium_store": result["equilibrium_store"],
            "equilibrium_time_path": (
                f"{result['equilibrium_store']}/{shot}.zarr/equilibrium/time"
            ),
            "source_groups": result["source_groups"],
            "source_groups_digest": result["source_groups_digest"],
            "corrections_digest": result["corrections_digest"],
        }
        for shot, result in sorted(shots.items())
    }
    batches = [
        {
            "width": width,
            "selection": [0, width],
            "unique_states": len({row["state_digest"] for row in slices[:width]}),
            "population_digest": _batch_digest(slices, width),
            "replication": "none",
        }
        for width in BENCHMARK_WIDTHS
        if width <= len(slices)
    ]
    skipped = [
        {"shot": int(row["shot"]), "reason": row["skip"]}
        for row in results
        if row.get("skip")
    ]
    manifest = {
        "schema": SCHEMA,
        "schema_revision": SCHEMA_REVISION,
        "generated_at": datetime.now(UTC).isoformat(),
        "selection": {
            "catalog_index": catalog_index,
            "catalog_columns": list(CATALOG_COLUMNS),
            "catalog_identity_sha256": catalog_digest,
            "flat_top_rule": (
                "level-2 equilibrium times inside the catalog's closed "
                "plasma_flat_top_start_time to plasma_flat_top_end_time interval"
            ),
            "input_alignment": (
                "nearest corrected level-1 field-clock row within 1 ms"
            ),
            "ordering": "shot, equilibrium index, corrected input index",
            "eligible_origin": "FAIR-MAST catalog metadata and local archive arrays",
            "candidate_shots_read": len(results),
            "skipped_candidate_shots": skipped,
        },
        "machine_artifact": {
            "digest": artifact_digest,
            "cache": str(artifact_cache),
            "manifest_path": str(artifact_path),
            "manifest_sha256": _file_digest(artifact_path),
            "schema": artifact["schema"],
            "dd_version": artifact["dd_version"],
            "physical_digest": artifact["physical_digest"],
            "registry_digest": artifact["registry_digest"],
            "complete": bool(artifact["complete"]),
            "shot_ranges": artifact["shot_ranges"],
        },
        "input_configuration": configuration,
        "shots": shot_records,
        "slices": slices,
        "benchmark_batches": batches,
        "anti_replication": {
            "policy": "every benchmark row is a separately read archive row",
            "broadcast_state": False,
            "tiled_state": False,
            "state_digest_excludes_labels": True,
        },
    }
    manifest["validation"] = validate_manifest(manifest, minimum_slices=required)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any], *, minimum_slices: int = max(BENCHMARK_WIDTHS)
) -> dict[str, Any]:
    """Validate completeness, provenance, uniqueness and anti-replication rules."""

    if manifest.get("schema") != SCHEMA:
        raise ManifestError("unexpected manifest schema")
    if manifest.get("schema_revision") != SCHEMA_REVISION:
        raise ManifestError("unexpected manifest schema revision")
    artifact = manifest.get("machine_artifact")
    artifact_keys = {
        "digest",
        "cache",
        "manifest_path",
        "manifest_sha256",
        "schema",
        "dd_version",
        "physical_digest",
        "registry_digest",
        "complete",
        "shot_ranges",
    }
    if not isinstance(artifact, Mapping) or not artifact_keys <= artifact.keys():
        raise ManifestError("machine-artifact provenance is incomplete")
    configuration = manifest.get("input_configuration")
    if (
        not isinstance(configuration, Mapping)
        or not {
            "digest",
            "coil_channels",
            "sensor_channels",
            "sensor_units",
            "arrays",
        }
        <= configuration.keys()
    ):
        raise ManifestError("input configuration is incomplete")
    if len(configuration["sensor_channels"]) != len(configuration["sensor_units"]):
        raise ManifestError("sensor channels and units disagree")
    undigested_configuration = dict(configuration)
    configuration_digest = undigested_configuration.pop("digest")
    if configuration_digest != _digest(undigested_configuration):
        raise ManifestError("input configuration digest does not match its content")

    selection = manifest.get("selection")
    if (
        not isinstance(selection, Mapping)
        or not {
            "catalog_index",
            "catalog_columns",
            "catalog_identity_sha256",
            "flat_top_rule",
            "input_alignment",
        }
        <= selection.keys()
    ):
        raise ManifestError("catalog and flat-top selection provenance is incomplete")
    shots = manifest.get("shots")
    if not isinstance(shots, Mapping) or not shots:
        raise ManifestError("shot-level source provenance is absent")
    for shot, source in shots.items():
        if (
            not isinstance(source, Mapping)
            or not {
                "campaign",
                "configuration",
                "flat_top",
                "input_shot_path",
                "equilibrium_time_path",
                "source_groups",
                "source_groups_digest",
                "corrections_digest",
            }
            <= source.keys()
        ):
            raise ManifestError(f"shot {shot} source provenance is incomplete")
        if not source["source_groups"]:
            raise ManifestError(f"shot {shot} has no channel provenance")
        if source["source_groups_digest"] != _digest(source["source_groups"]):
            raise ManifestError(f"shot {shot} channel provenance digest disagrees")
        for group in source["source_groups"]:
            required_group = {
                "store",
                "shot",
                "group",
                "channel",
                "group_identity",
            }
            if not required_group <= group.keys():
                raise ManifestError(f"shot {shot} has incomplete channel provenance")

    slices = manifest.get("slices")
    if not isinstance(slices, list) or len(slices) < minimum_slices:
        raise ManifestError(
            f"manifest has {0 if not isinstance(slices, list) else len(slices)} "
            f"slices; at least {minimum_slices} required"
        )
    coordinates = set()
    state_digests = []
    campaigns = set()
    configurations = set()
    for expected_ordinal, row in enumerate(slices):
        required_row = {
            "ordinal",
            "shot",
            "equilibrium_index",
            "input_index",
            "equilibrium_time_s",
            "input_time_s",
            "campaign",
            "configuration",
            "machine_artifact_digest",
            "input_configuration_digest",
            "source_provenance_ref",
            "arrays",
            "state_digest",
            "origin",
            "synthetic",
            "broadcast",
            "tiled",
        }
        if not required_row <= row.keys():
            raise ManifestError(f"slice {expected_ordinal} provenance is incomplete")
        if int(row["ordinal"]) != expected_ordinal:
            raise ManifestError("slice ordinals are not contiguous")
        if row["origin"] != "real_archive_row" or any(
            bool(row[name]) for name in ("synthetic", "broadcast", "tiled")
        ):
            raise ManifestError(f"slice {expected_ordinal} is replicated or synthetic")
        if row["machine_artifact_digest"] != artifact["digest"]:
            raise ManifestError("slice machine-artifact identity disagrees")
        if row["input_configuration_digest"] != configuration_digest:
            raise ManifestError("slice input configuration identity disagrees")
        shot_key = str(int(row["shot"]))
        if row["source_provenance_ref"] != f"shots.{shot_key}" or shot_key not in shots:
            raise ManifestError("slice source provenance reference is unresolved")
        source = shots[shot_key]
        if row["campaign"] != source["campaign"]:
            raise ManifestError("slice campaign disagrees with its shot provenance")
        if row["configuration"] != source["configuration"]:
            raise ManifestError(
                "slice configuration disagrees with its shot provenance"
            )
        if row["configuration"].get("divertor_config") in (None, ""):
            raise ManifestError("slice has no catalog configuration")
        time = float(row["equilibrium_time_s"])
        if not (
            float(source["flat_top"]["start_s"])
            <= time
            <= float(source["flat_top"]["end_s"])
        ):
            raise ManifestError("slice lies outside its catalog flat-top interval")
        if not math.isfinite(time) or not math.isfinite(float(row["input_time_s"])):
            raise ManifestError("slice time is not finite")
        coordinate = (
            int(row["shot"]),
            int(row["equilibrium_index"]),
            int(row["input_index"]),
        )
        if coordinate in coordinates:
            raise ManifestError("duplicate shot/index input coordinate")
        coordinates.add(coordinate)
        arrays = row["arrays"]
        if not isinstance(arrays, Mapping) or set(arrays) != set(ARRAY_NAMES):
            raise ManifestError("slice input-array provenance is incomplete")
        for name in ARRAY_NAMES:
            array = arrays[name]
            if not {"source", "row_index", "dtype", "shape", "sha256"} <= array.keys():
                raise ManifestError(f"slice array {name} provenance is incomplete")
            if int(array["row_index"]) != int(row["input_index"]):
                raise ManifestError(f"slice array {name} points at a different row")
            if not str(array["sha256"]).startswith("sha256:"):
                raise ManifestError(f"slice array {name} has no digest")
        if row["state_digest"] != _state_digest(arrays):
            raise ManifestError("slice state digest disagrees with its input arrays")
        state_digests.append(row["state_digest"])
        campaigns.add(str(row["campaign"]))
        configurations.add(
            _digest(
                {
                    "campaign": row["campaign"],
                    "configuration": row["configuration"],
                }
            )
        )
    duplicate_states = len(state_digests) - len(set(state_digests))
    if duplicate_states:
        raise ManifestError(f"manifest contains {duplicate_states} duplicate states")

    batches = manifest.get("benchmark_batches")
    expected_widths = [width for width in BENCHMARK_WIDTHS if width <= len(slices)]
    recorded_widths = (
        [row.get("width") for row in batches] if isinstance(batches, list) else []
    )
    if recorded_widths != expected_widths:
        raise ManifestError("benchmark width ladder is incomplete")
    for batch in batches:
        width = int(batch["width"])
        if batch.get("selection") != [0, width]:
            raise ManifestError("benchmark batch is not a nested real-input prefix")
        if batch.get("replication") != "none" or batch.get("unique_states") != width:
            raise ManifestError("benchmark batch contains replicated state")
        if batch.get("population_digest") != _batch_digest(slices, width):
            raise ManifestError("benchmark batch population digest disagrees")
    anti_replication = manifest.get("anti_replication")
    if not isinstance(anti_replication, Mapping) or any(
        bool(anti_replication.get(name)) for name in ("broadcast_state", "tiled_state")
    ):
        raise ManifestError("anti-replication declaration is absent or false")

    return {
        "eligible_real_flat_top_slices": len(slices),
        "unique_shot_index_input_coordinates": len(coordinates),
        "unique_state_digests": len(set(state_digests)),
        "duplicate_state_digests": duplicate_states,
        "shots": len(shots),
        "campaigns": sorted(campaigns),
        "campaign_configuration_strata": len(configurations),
        "shot_index_time_campaign_configuration_provenance_complete": True,
        "machine_artifact_provenance_complete": True,
        "input_array_provenance_complete": True,
        "broadcast_state": False,
        "tiled_state": False,
        "benchmark_widths": expected_widths,
    }


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write strict, human-readable JSON atomically within the output directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    """Build a manifest or validate an existing one."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--catalog-index", default=DEFAULT_CATALOG_INDEX)
    build.add_argument("--input-store", type=Path, default=DEFAULT_INPUT_STORE)
    build.add_argument(
        "--equilibrium-store", type=Path, default=DEFAULT_EQUILIBRIUM_STORE
    )
    build.add_argument("--artifact-cache", type=Path, default=DEFAULT_ARTIFACT_CACHE)
    build.add_argument("--artifact-digest", default=DEFAULT_ARTIFACT_DIGEST)
    build.add_argument("--candidate-shots", type=int, default=192)
    build.add_argument("--workers", type=int, default=8)
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    validate = subparsers.add_parser("validate")
    validate.add_argument("path", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    validate.add_argument("--minimum-slices", type=int, default=max(BENCHMARK_WIDTHS))
    arguments = parser.parse_args()

    if arguments.command == "build":
        manifest = build_manifest(
            catalog_index=arguments.catalog_index,
            input_store=arguments.input_store,
            equilibrium_store=arguments.equilibrium_store,
            artifact_cache=arguments.artifact_cache,
            artifact_digest=arguments.artifact_digest,
            candidate_shots=arguments.candidate_shots,
            workers=arguments.workers,
        )
        _write_manifest(arguments.output, manifest)
        print(json.dumps(manifest["validation"], indent=2))
        return
    manifest = json.loads(arguments.path.read_text())
    print(
        json.dumps(
            validate_manifest(manifest, minimum_slices=arguments.minimum_slices),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
