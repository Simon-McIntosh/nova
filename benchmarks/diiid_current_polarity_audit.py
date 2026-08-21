"""Locate the DIII-D coil-current polarity inversion boundary.

The raw side is represented twice: by hashes of the compressed Parquet page
bytes and by Arrow's direct decoding of those pages.  The consumer side is the
unchanged row reader used by the plasma-subtraction benchmark followed by the
registry current join.  Full-array byte equality and per-sample sign equality
therefore distinguish a producer-side stored polarity from a read-time
transformation.

The corpus census classifies one labelled frame per shot.  It chooses the
lowest all-coil native-rate derivative and records the sign of the centred
inner product between the labelled flux and the recorded-current coil map on
the rectangular grid boundary, which is outside every shipped LCFS.  The
analytic polygon-section response is built once, persisted with its complete
geometry and grid identity, then validated and reloaded before the census.
A negative inner product is the affected-shot predicate.  The normalised inner
product is diagnostic only; no coefficient is applied to a prediction or
written back to the corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from benchmarks import diiid_negative_tail_attribution as attribution
from benchmarks import diiid_plasma_subtraction_gate as gate
from benchmarks import diiid_vacuum_quiescent_gate as quiescent
from benchmarks.diiid_corpus_conventions import nova_total_flux_to_corpus
from nova.biot.polygon import polygon_greens
from nova.imas.diiid_description import (
    ALL_CONDUCTORS,
    POLOIDAL_CONDUCTORS,
    DiiidDescription,
    DiiidDescriptionRegistry,
    geometry_digest,
)

DEFAULT_DATA = Path("/work/projects/imas_gpu/sophelio/raw/data/diii_d_train")
DEFAULT_BANK = Path(
    "/work/projects/imas_gpu/sophelio/vacuum-gate/diiid_plasma_subtraction_gate.json"
)
DEFAULT_PREREGISTRATION = Path(
    "/work/projects/imas_gpu/sophelio/vacuum-gate/"
    "diiid_plasma_subtraction_preregistration.json"
)
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/current-polarity")
RESPONSE_ARTIFACT_NAME = "exact_polygon_boundary_response.npz"
TAIL_SHOTS = (
    "d3d_shot_009875548f.parquet",
    "d3d_shot_001bc5a4ae.parquet",
)
EXPECTED_BANK_SHA256 = (
    "5b53ffc30bbe823fa50c43bb945c184c8033e81b4083aaa338661278d1f6adea"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "7e60861de8c104a8d736bd5300993071da35fce93e206ad5bfb3010213f972fc"
)
SELECTED_DERIVATIVE_THRESHOLD = 100.0
CURRENT_COLUMNS = tuple(f"magnetics_{name}" for name in POLOIDAL_CONDUCTORS)
DERIVATIVE_COLUMNS = tuple(f"magnetics_{name}" for name in ALL_CONDUCTORS)
GEOMETRY_COLUMNS = (
    "coil_name",
    "coil_input_column",
    "coil_R",
    "coil_Z",
    "coil_width",
    "coil_height",
    "coil_angle1",
    "coil_angle2",
)
CENSUS_COLUMNS = (
    "efit_times",
    "efit_psirz",
    "efit_grid_R",
    "efit_grid_Z",
    "magnetics_time",
    *DERIVATIVE_COLUMNS,
    *GEOMETRY_COLUMNS,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _parquet_page_receipt(path: Path, column: str) -> dict[str, Any]:
    """Hash the compressed page bytes carrying one one-row list column."""

    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("run with `uv run --with pyarrow python ...`") from error
    metadata = parquet.ParquetFile(path).metadata
    if metadata.num_row_groups != 1:
        raise ValueError(f"{path.name} must have exactly one row group")
    matches = []
    group = metadata.row_group(0)
    for index in range(group.num_columns):
        chunk = group.column(index)
        if chunk.path_in_schema == f"{column}.list.element":
            matches.append(chunk)
    if len(matches) != 1:
        raise ValueError(f"found {len(matches)} Parquet chunks for {column}")
    chunk = matches[0]
    offsets = [chunk.data_page_offset]
    if chunk.dictionary_page_offset is not None and chunk.dictionary_page_offset >= 0:
        offsets.append(chunk.dictionary_page_offset)
    start = min(offsets)
    with path.open("rb") as stream:
        stream.seek(start)
        encoded = stream.read(chunk.total_compressed_size)
    if len(encoded) != chunk.total_compressed_size:
        raise IOError(f"short raw page read for {path.name}:{column}")
    return {
        "path_in_schema": chunk.path_in_schema,
        "file_offset": start,
        "compressed_size_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "compression": chunk.compression,
        "encodings": list(chunk.encodings),
    }


def _direct_arrow_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise RuntimeError("run with `uv run --with pyarrow python ...`") from error
    table = parquet.ParquetFile(path).read(columns=list(CURRENT_COLUMNS))
    return {
        column: table[column][0].values.to_numpy(zero_copy_only=False)
        for column in CURRENT_COLUMNS
    }


def _sign_counts(values: np.ndarray) -> dict[str, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "negative": int(np.count_nonzero(finite < 0.0)),
        "zero": int(np.count_nonzero(finite == 0.0)),
        "positive": int(np.count_nonzero(finite > 0.0)),
    }


def compare_read_boundary(path: Path, description: DiiidDescription) -> dict[str, Any]:
    """Compare compressed source pages, direct Arrow values, and Nova inputs."""

    direct = _direct_arrow_arrays(path)
    consumer = gate._read(path, columns=list(CURRENT_COLUMNS))
    conductors = {item.input_column: item for item in description.conductors}
    channels: dict[str, Any] = {}
    total_sign_mismatches = 0
    for column in CURRENT_COLUMNS:
        raw = np.asarray(direct[column])
        read = np.asarray(consumer[column], dtype=raw.dtype)
        if raw.shape != read.shape:
            raise ValueError(f"{path.name}:{column} changed shape at read boundary")
        finite = np.isfinite(raw) & np.isfinite(read)
        sign_mismatches = int(
            np.count_nonzero(np.signbit(raw[finite]) != np.signbit(read[finite]))
        )
        total_sign_mismatches += sign_mismatches
        conductor = conductors[column]
        nova_amperes = 1000.0 * read * conductor.turns.applied_multiplier
        unit_sign_mismatches = int(
            np.count_nonzero(
                np.signbit(read[finite]) != np.signbit(nova_amperes[finite])
            )
        )
        channels[column] = {
            "raw_page": _parquet_page_receipt(path, column),
            "native_samples": int(raw.size),
            "direct_arrow_sha256": _array_sha256(raw),
            "nova_read_sha256": _array_sha256(read),
            "bitwise_equal": bool(np.array_equal(raw, read, equal_nan=True)),
            "maximum_absolute_difference": float(
                np.max(np.abs(raw[finite] - read[finite])) if np.any(finite) else 0.0
            ),
            "raw_sign_counts": _sign_counts(raw),
            "nova_read_sign_counts": _sign_counts(read),
            "raw_to_read_sign_mismatches": sign_mismatches,
            "read_to_registry_ampere_sign_mismatches": unit_sign_mismatches,
            "registry_scale_to_amperes": (1000.0 * conductor.turns.applied_multiplier),
        }
    return {
        "shot": path.name,
        "raw_parquet_file_sha256": _sha256(path),
        "raw_parquet_size_bytes": path.stat().st_size,
        "channels": channels,
        "channel_count": len(channels),
        "all_channels_bitwise_equal": all(
            item["bitwise_equal"] for item in channels.values()
        ),
        "total_raw_to_read_sign_mismatches": total_sign_mismatches,
        "total_read_to_registry_ampere_sign_mismatches": sum(
            item["read_to_registry_ampere_sign_mismatches"]
            for item in channels.values()
        ),
    }


def orientation_cosine(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return the gauge-free shape orientation without fitting an amplitude."""

    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()
    finite = np.isfinite(actual + predicted)
    actual = actual[finite] - np.mean(actual[finite])
    predicted = predicted[finite] - np.mean(predicted[finite])
    denominator = float(np.linalg.norm(actual) * np.linalg.norm(predicted))
    if denominator <= np.finfo(float).tiny:
        raise ValueError("orientation vectors have no centred shape energy")
    return float((actual @ predicted) / denominator)


def affected_from_orientation(value: float) -> bool:
    """Apply the declared negative centred-inner-product predicate."""

    return bool(np.isfinite(value) and value < 0.0)


def _boundary_mask(radius: np.ndarray, height: np.ndarray) -> np.ndarray:
    mask = np.zeros((height.size, radius.size), dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    return mask


def _exact_polygon_response(
    description: DiiidDescription,
    radius: np.ndarray,
    height: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Evaluate the unchanged analytic polygon-section discriminator."""

    target_r, target_z = np.meshgrid(radius, height)
    names = []
    responses = []
    for conductor in description.conductors:
        if (
            conductor.vertices is None
            or not conductor.turns.affects_axisymmetric_poloidal_flux
        ):
            continue
        names.append(conductor.name)
        responses.append(
            polygon_greens(
                target_r[target_mask], target_z[target_mask], conductor.vertices
            )[0]
        )
    return tuple(names), np.stack(responses)


def _write_response_artifact(
    path: Path,
    description: DiiidDescription,
    radius: np.ndarray,
    height: np.ndarray,
    target_mask: np.ndarray,
) -> None:
    names, matrix = _exact_polygon_response(description, radius, height, target_mask)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".building.npz")
    np.savez_compressed(
        temporary,
        conductor_names=np.asarray(names, dtype=np.str_),
        response=np.asarray(matrix, dtype=np.float64),
        radius=np.asarray(radius, dtype=np.float64),
        height=np.asarray(height, dtype=np.float64),
        target_indices=np.flatnonzero(target_mask).astype(np.int64),
        geometry_digest=np.asarray(description.physical_digest, dtype=np.str_),
        kernel_route=np.asarray("nova.biot.polygon.polygon_greens", dtype=np.str_),
    )
    temporary.replace(path)


def _load_response_artifact(
    path: Path,
    description: DiiidDescription,
    radius: np.ndarray,
    height: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[tuple[str, ...], np.ndarray]:
    expected_names = tuple(
        conductor.name
        for conductor in description.conductors
        if conductor.vertices is not None
        and conductor.turns.affects_axisymmetric_poloidal_flux
    )
    with np.load(path, allow_pickle=False) as artifact:
        names = tuple(str(value) for value in artifact["conductor_names"])
        matrix = np.asarray(artifact["response"], dtype=float)
        stored_radius = np.asarray(artifact["radius"], dtype=float)
        stored_height = np.asarray(artifact["height"], dtype=float)
        target_indices = np.asarray(artifact["target_indices"], dtype=np.int64)
        stored_digest = str(artifact["geometry_digest"].item())
        kernel_route = str(artifact["kernel_route"].item())
    expected_indices = np.flatnonzero(target_mask)
    if names != expected_names:
        raise ValueError("persisted response conductor order does not match geometry")
    if stored_digest != description.physical_digest:
        raise ValueError("persisted response geometry digest does not match corpus")
    if kernel_route != "nova.biot.polygon.polygon_greens":
        raise ValueError(
            "persisted response does not declare the analytic polygon route"
        )
    if not np.array_equal(stored_radius, radius):
        raise ValueError("persisted response radial grid does not match corpus")
    if not np.array_equal(stored_height, height):
        raise ValueError("persisted response vertical grid does not match corpus")
    if not np.array_equal(target_indices, expected_indices):
        raise ValueError(
            "persisted response target boundary does not match discriminator"
        )
    if matrix.shape != (len(names), expected_indices.size):
        raise ValueError("persisted response has an invalid matrix shape")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("persisted response contains non-finite values")
    return names, matrix


def persisted_exact_polygon_response(
    path: Path,
    description: DiiidDescription,
    radius: np.ndarray,
    height: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[tuple[tuple[str, ...], np.ndarray], dict[str, Any]]:
    """Build once, then validate and reload the exact response from disk."""

    built = not path.exists()
    if built:
        _write_response_artifact(path, description, radius, height, target_mask)
    response = _load_response_artifact(path, description, radius, height, target_mask)
    names, matrix = response
    return response, {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "built_in_this_run": built,
        "persisted_before_census": True,
        "loaded_from_persisted_artifact": True,
        "kernel_route": "nova.biot.polygon.polygon_greens",
        "geometry_digest": description.physical_digest,
        "source_section_count": len(names),
        "target_boundary_point_count": int(matrix.shape[1]),
        "response_shape": list(matrix.shape),
        "filament_centre_proxy_used": False,
    }


def _limit_arrow_threads() -> None:
    """Give each census process one Arrow CPU and I/O worker."""

    import pyarrow

    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)


def _census_shot(
    path: Path,
    description: DiiidDescription,
    response: tuple[tuple[str, ...], np.ndarray],
    target_mask: np.ndarray,
    expected_geometry_digest: str,
) -> dict[str, Any]:
    row = gate._read(path, columns=list(CENSUS_COLUMNS))
    digest = geometry_digest(row)
    if digest != expected_geometry_digest:
        raise ValueError(f"{path.name} carries an unexpected conductor table")
    times = np.asarray(row["efit_times"], dtype=float)
    native_time = np.asarray(row["magnetics_time"], dtype=float)
    currents = np.column_stack(
        [np.asarray(row[column], dtype=float) for column in DERIVATIVE_COLUMNS]
    )
    derivative = quiescent.smoothed_native_derivative(native_time, currents, times)
    maximum_derivative = np.nanmax(np.abs(derivative), axis=1)
    frame = int(np.nanargmin(maximum_derivative))
    truth = np.asarray(row["efit_psirz"][frame], dtype=float)
    names, matrix = response
    current = attribution._current_vector(row, description, names, float(times[frame]))
    predicted = nova_total_flux_to_corpus(np.einsum("c,cp->p", current, matrix))
    actual = truth[target_mask]
    finite = np.isfinite(actual + predicted)
    cosine = orientation_cosine(actual[finite], predicted[finite])
    return {
        "shot": path.name,
        "frame": frame,
        "time_ms": float(times[frame]),
        "maximum_derivative": float(maximum_derivative[frame]),
        "meets_selected_quiescent_threshold": bool(
            maximum_derivative[frame] <= SELECTED_DERIVATIVE_THRESHOLD
        ),
        "exterior_points": int(np.count_nonzero(finite)),
        "orientation_cosine": cosine,
        "affected": affected_from_orientation(cosine),
    }


def census_polarity(
    paths: list[Path],
    first_row: dict[str, Any],
    response_path: Path,
    *,
    workers: int,
) -> dict[str, Any]:
    """Evaluate the polarity predicate on one lowest-derivative frame per shot."""

    registry = DiiidDescriptionRegistry()
    description = registry.ingest(first_row, source_row=paths[0].name)
    expected_digest = description.physical_digest
    radius = np.asarray(first_row["efit_grid_R"], dtype=float)
    height = np.asarray(first_row["efit_grid_Z"], dtype=float)
    target_mask = _boundary_mask(radius, height)
    response, response_receipt = persisted_exact_polygon_response(
        response_path, description, radius, height, target_mask
    )

    evaluate = partial(
        _census_shot,
        description=description,
        response=response,
        target_mask=target_mask,
        expected_geometry_digest=expected_digest,
    )
    records = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_limit_arrow_threads
    ) as executor:
        for number, record in enumerate(
            executor.map(evaluate, paths, chunksize=4), start=1
        ):
            records.append(record)
            if number % 250 == 0:
                print(f"CENSUS {number}/{len(paths)}", flush=True)
    affected = [item["shot"] for item in records if item["affected"]]
    orientations = np.asarray(
        [item["orientation_cosine"] for item in records], dtype=float
    )
    return {
        "predicate": "centred grid-boundary label/coil-map inner product < 0",
        "coefficients_fitted": 0,
        "grid_boundary_is_outside_all_shipped_lcfs": True,
        "frame_selection": "minimum all-coil smoothed native-rate derivative",
        "selected_derivative_threshold_ka_turn_per_s": (SELECTED_DERIVATIVE_THRESHOLD),
        "shot_count": len(records),
        "classified_shot_count": len(records),
        "quiescent_at_selected_threshold_count": sum(
            item["meets_selected_quiescent_threshold"] for item in records
        ),
        "affected_shot_count": len(affected),
        "affected_shots": sorted(affected),
        "exact_polygon_response": response_receipt,
        "orientation_cosine": {
            "minimum": float(np.min(orientations)),
            "q25": float(np.quantile(orientations, 0.25)),
            "median": float(np.median(orientations)),
            "q75": float(np.quantile(orientations, 0.75)),
            "maximum": float(np.max(orientations)),
        },
        "records": records,
    }


def audit(
    data_root: Path = DEFAULT_DATA,
    bank_path: Path = DEFAULT_BANK,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
    response_path: Path = DEFAULT_OUTPUT / RESPONSE_ARTIFACT_NAME,
    *,
    workers: int = 8,
) -> dict[str, Any]:
    """Run the immutable-boundary comparison and full-corpus census."""

    immutable_before = {
        str(bank_path): _sha256(bank_path),
        str(preregistration_path): _sha256(preregistration_path),
    }
    if immutable_before[str(bank_path)] != EXPECTED_BANK_SHA256:
        raise RuntimeError("the differencing gate bank is not the passed artifact")
    if immutable_before[str(preregistration_path)] != EXPECTED_PREREGISTRATION_SHA256:
        raise RuntimeError(
            "the differencing preregistration is not the passed artifact"
        )
    paths = sorted(data_root.glob("*.parquet"))
    if len(paths) != 7041:
        raise RuntimeError(f"expected 7041 corpus shots, found {len(paths)}")
    first_row = gate._read(paths[0])
    description = DiiidDescriptionRegistry().ingest(first_row, source_row=paths[0].name)
    boundary = {
        shot: compare_read_boundary(data_root / shot, description)
        for shot in TAIL_SHOTS
    }
    if not all(item["all_channels_bitwise_equal"] for item in boundary.values()):
        raise RuntimeError("a tail current array changes across the read boundary")
    census = census_polarity(paths, first_row, response_path, workers=workers)
    for shot in TAIL_SHOTS:
        if shot not in census["affected_shots"]:
            raise RuntimeError(f"known polarity tail {shot} was not reproduced")
    immutable_after = {
        str(bank_path): _sha256(bank_path),
        str(preregistration_path): _sha256(preregistration_path),
    }
    if immutable_after != immutable_before:
        raise RuntimeError("a passed gate artifact changed during the audit")
    return {
        "verdict": {
            "inverting_boundary": "corpus producer or storage before Nova read",
            "nova_read_path_inverts_current": False,
            "reason": (
                "all nineteen poloidal-current arrays are bitwise and sign "
                "identical before and after the Nova read boundary on both "
                "tail shots; the registry applies only a positive unit scale"
            ),
        },
        "tail_boundary_comparison": boundary,
        "full_corpus_census": census,
        "immutable_gate_artifacts": {
            "before": immutable_before,
            "after": immutable_after,
            "unchanged": True,
        },
    }


def _figure(receipt: dict[str, Any], path: Path) -> None:
    records = receipt["full_corpus_census"]["records"]
    tail = {item["shot"]: item for item in records if item["shot"] in TAIL_SHOTS}
    values = np.asarray([item["orientation_cosine"] for item in records])
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    axes[0].hist(values, bins=80, color="0.65", edgecolor="0.25", linewidth=0.4)
    axes[0].axvline(0.0, color="#cc0000", linewidth=1.0)
    axes[0].set_xlabel("vacuum remainder / coil orientation cosine")
    axes[0].set_ylabel("shots")
    axes[0].set_title("Full-corpus polarity census")
    labels = [shot.removesuffix(".parquet") for shot in TAIL_SHOTS]
    axes[1].bar(
        labels,
        [tail[shot]["orientation_cosine"] for shot in TAIL_SHOTS],
        color="#cc0000",
    )
    axes[1].axhline(0.0, color="0.2", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylabel("orientation cosine")
    axes[1].set_title("Named tail shots")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREGISTRATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    response_path = args.response or args.output / RESPONSE_ARTIFACT_NAME
    receipt = audit(
        args.data,
        args.bank,
        args.preregistration,
        response_path,
        workers=args.workers,
    )
    receipt_path = args.output / "current_polarity_audit_receipt.json"
    figure_path = args.output / "current_polarity_census.png"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _figure(receipt, figure_path)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "figure": str(figure_path),
                "inverting_boundary": receipt["verdict"]["inverting_boundary"],
                "affected_shot_count": receipt["full_corpus_census"][
                    "affected_shot_count"
                ],
                "gate_artifacts_unchanged": receipt["immutable_gate_artifacts"][
                    "unchanged"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
