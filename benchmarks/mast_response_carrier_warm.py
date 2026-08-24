"""Persist and verify the shared MAST frozen-reference response carrier.

The cold path resolves the six fixed reference rows, builds the prescribed
current response once, and publishes it under its complete semantic input
identity.  The cache-only path needs neither the shot store nor a Green
operator: it rejects the carrier before returning the response unless every
stored identity, target and shape assertion matches the requested contract.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from time import perf_counter
from typing import Any, Iterator

import numpy as np


SEMANTIC_RESPONSE_IDENTITY = (
    "1d2c4a2b2f448ab8f1ae981031bbaf85fe4ee87f8ed9606fe6847d0fc9f1e994"
)
RESOLVED_TARGET_DIGEST = (
    "5623983f54f144edd70f113bdf66ed60fd4de6b751bb8312a31aa422d158b4a9"
)
RESPONSE_SHAPE = (1126, 101)
STORED_CIRCUIT_COUNT = 101
DEFAULT_CARRIER = (
    Path("/work/projects/imas_gpu/sophelio/mast_frozen_six_response_carriers")
    / f"{SEMANTIC_RESPONSE_IDENTITY}.npz"
)
DEFAULT_RECEIPT = Path(
    "docs/figures/plateau-input-attribution/mast-response-carrier.json"
)
DIRECT_BUILDER_MODULES = frozenset(
    {
        "nova.biot.greens",
        "nova.biot.polygon",
        "nova.imas.mast_vacuum_response",
    }
)


def _array_digest(values: np.ndarray) -> str:
    """Return the input-contract digest for one typed, shaped array."""
    packed = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(packed.dtype.str.encode())
    digest.update(b"\0")
    digest.update(np.asarray(packed.shape, dtype=np.int64).tobytes())
    digest.update(packed.tobytes())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(archive: Any, name: str) -> str:
    values = np.asarray(archive[name])
    if values.shape != ():
        raise ValueError(f"persisted {name} must be scalar")
    return str(values.item())


class _DirectBuilderImportGuard(importlib.abc.MetaPathFinder):
    """Refuse imports that could construct a direct Green response."""

    entered: list[str]

    def __init__(self) -> None:
        self.entered = []

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> None:
        del path, target
        if any(
            fullname == module or fullname.startswith(f"{module}.")
            for module in DIRECT_BUILDER_MODULES
        ):
            self.entered.append(fullname)
            raise RuntimeError(
                f"cache-only response resolution entered direct builder {fullname}"
            )
        return None


@contextmanager
def _guard_direct_builders() -> Iterator[_DirectBuilderImportGuard]:
    already_loaded = sorted(DIRECT_BUILDER_MODULES.intersection(sys.modules))
    if already_loaded:
        raise RuntimeError(
            "cache-only process already imported direct builder modules: "
            + ", ".join(already_loaded)
        )
    guard = _DirectBuilderImportGuard()
    sys.meta_path.insert(0, guard)
    try:
        yield guard
    finally:
        sys.meta_path.remove(guard)


def load_carrier(
    path: Path,
    *,
    semantic_identity: str = SEMANTIC_RESPONSE_IDENTITY,
    resolved_target_digest: str = RESOLVED_TARGET_DIGEST,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a response only after its complete persisted contract matches."""
    started = perf_counter()
    with np.load(path, allow_pickle=False) as archive:
        stored_identity = _scalar(archive, "semantic_response_identity")
        if stored_identity != semantic_identity:
            raise ValueError(
                "persisted semantic response identity does not match request: "
                f"expected {semantic_identity}, got {stored_identity}"
            )
        stored_target_digest = _scalar(archive, "resolved_target_digest")
        if stored_target_digest != resolved_target_digest:
            raise ValueError(
                "persisted resolved-target digest does not match request: "
                f"expected {resolved_target_digest}, got {stored_target_digest}"
            )
        targets = np.asarray(archive["resolved_targets"], dtype=np.float64)
        if _array_digest(targets) != stored_target_digest:
            raise ValueError("persisted resolved targets do not match their digest")
        stored_circuits = int(np.asarray(archive["stored_circuit_count"]).item())
        if stored_circuits != STORED_CIRCUIT_COUNT:
            raise ValueError(
                "persisted circuit inventory does not match frozen contract: "
                f"expected {STORED_CIRCUIT_COUNT}, got {stored_circuits}"
            )
        response = np.asarray(archive["response"], dtype=np.float64)
        if response.shape != RESPONSE_SHAPE:
            raise ValueError(
                "persisted response shape does not match frozen contract: "
                f"expected {RESPONSE_SHAPE}, got {response.shape}"
            )
        if targets.shape != (RESPONSE_SHAPE[0], 2):
            raise ValueError("persisted targets do not span every response row")
        if not np.all(np.isfinite(response)):
            raise ValueError("persisted response contains non-finite values")
        response_digest = _scalar(archive, "response_sha256")
        if _array_digest(response) != response_digest:
            raise ValueError("persisted response does not match its digest")
        selected = json.loads(_scalar(archive, "frozen_references_json"))
        if len(selected) != 6:
            raise ValueError("persisted carrier does not name six frozen references")
        input_digests = json.loads(_scalar(archive, "input_digests_json"))
        if input_digests.get("combined_sha256") != stored_identity:
            raise ValueError("persisted input ledger does not match semantic identity")
    elapsed = perf_counter() - started
    return response, {
        "path": str(path.resolve()),
        "semantic_response_identity": stored_identity,
        "resolved_target_digest": stored_target_digest,
        "response_sha256": response_digest,
        "response_shape": list(response.shape),
        "stored_circuit_count": stored_circuits,
        "frozen_reference_count": len(selected),
        "warm_load_seconds": elapsed,
        "file_sha256": _file_digest(path),
        "size_bytes": path.stat().st_size,
    }


def _cold_response() -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact shared response through the frozen scoring seam."""
    from benchmarks.efit_forward_parity_slice import (
        DECOMPOSITION_BANK,
        REFERENCE_NATIVE_GRID_POINTS,
        _mast_case_from_selection,
        _passive_inclusive_case,
        select_slices_by_shot,
    )
    from nova.imas.mast_solve_inputs import SHOT_STORE
    from nova.jax.config import configure_dtypes

    configure_dtypes()
    selected = select_slices_by_shot(DECOMPOSITION_BANK)
    references = [
        {"shot": int(row["shot"]), "slice_index": int(row["slice_index"])}
        for row, _qualification in selected
    ]
    machine_started = perf_counter()
    first_row, qualification = selected[0]
    mast_case, context = _mast_case_from_selection(
        SHOT_STORE,
        first_row,
        qualification,
        grid_points=REFERENCE_NATIVE_GRID_POINTS,
    )
    machine_seconds = perf_counter() - machine_started
    targets = np.vstack((mast_case["grid_coordinate"], mast_case["wall_coordinate"]))
    response_started = perf_counter()
    _passive_case, profile, policy = _passive_inclusive_case(mast_case, context, None)
    response_seconds = perf_counter() - response_started
    prescribed = profile.operator.prescribed_current_field
    if prescribed is None:
        raise RuntimeError("cold builder did not attach a prescribed response")
    response = np.asarray(prescribed.response, dtype=np.float64)
    input_digests = policy["response_input_digests"]
    identity = input_digests["combined_sha256"]
    target_digest = input_digests["inputs"]["resolved_response_targets"]["sha256"]
    if identity != SEMANTIC_RESPONSE_IDENTITY:
        raise RuntimeError(
            "resolved response identity changed: "
            f"expected {SEMANTIC_RESPONSE_IDENTITY}, got {identity}"
        )
    if target_digest != RESOLVED_TARGET_DIGEST:
        raise RuntimeError(
            "resolved-target digest changed: "
            f"expected {RESOLVED_TARGET_DIGEST}, got {target_digest}"
        )
    if response.shape != RESPONSE_SHAPE:
        raise RuntimeError(
            f"cold response has shape {response.shape}, expected {RESPONSE_SHAPE}"
        )
    if int(policy["stored_circuit_count"]) != STORED_CIRCUIT_COUNT:
        raise RuntimeError("cold response does not carry every stored circuit")
    arrays = {
        "semantic_response_identity": np.asarray(identity, dtype=np.str_),
        "resolved_target_digest": np.asarray(target_digest, dtype=np.str_),
        "resolved_targets": np.asarray(targets, dtype=np.float64),
        "response": response,
        "response_sha256": np.asarray(_array_digest(response), dtype=np.str_),
        "stored_circuit_count": np.asarray(STORED_CIRCUIT_COUNT, dtype=np.int64),
        "frozen_references_json": np.asarray(
            json.dumps(references, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
        "input_digests_json": np.asarray(
            json.dumps(input_digests, sort_keys=True, separators=(",", ":")),
            dtype=np.str_,
        ),
        "audit_json": np.asarray(
            json.dumps(
                {
                    name: policy[name]
                    for name in (
                        "active_circuit_count",
                        "passive_or_vessel_circuit_count",
                        "section_kernel_evaluations",
                        "passive_registry_minimum_overlap_fraction",
                        "passive_registry_maximum_separation_m",
                    )
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            dtype=np.str_,
        ),
    }
    return arrays, {
        "machine_resolution_seconds": machine_seconds,
        "direct_response_build_seconds": response_seconds,
        "total_before_publication_seconds": machine_seconds + response_seconds,
        "section_kernel_evaluations": int(policy["section_kernel_evaluations"]),
        "frozen_references": references,
    }


def _cache_only_subprocess(carrier: Path) -> dict[str, Any]:
    """Load the carrier in a fresh interpreter with direct imports refused."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "probe",
        "--carrier",
        str(carrier),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ,
    )
    return json.loads(completed.stdout)


def build(carrier: Path, receipt: Path) -> dict[str, Any]:
    """Build, publish and immediately verify one content-addressed carrier."""
    if carrier.exists():
        raise FileExistsError(
            f"cold publication refuses to replace existing carrier {carrier}"
        )
    arrays, cold = _cold_response()
    carrier.parent.mkdir(parents=True, exist_ok=True)
    temporary = carrier.with_name(f".{carrier.name}.{os.getpid()}.building.npz")
    publication_started = perf_counter()
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(carrier)
    finally:
        if temporary.exists():
            temporary.unlink()
    publication_seconds = perf_counter() - publication_started
    warm = _cache_only_subprocess(carrier)
    entered = warm.pop("direct_builder_modules_entered")
    report = {
        "receipt": "persisted MAST frozen-reference response carrier",
        "verdict": {
            "carrier_persisted": True,
            "semantic_identity_matches": (
                warm["semantic_response_identity"] == SEMANTIC_RESPONSE_IDENTITY
            ),
            "resolved_targets_match": (
                warm["resolved_target_digest"] == RESOLVED_TARGET_DIGEST
            ),
            "response_shape_matches": warm["response_shape"] == list(RESPONSE_SHAPE),
            "all_stored_circuits_carried": (
                warm["stored_circuit_count"] == STORED_CIRCUIT_COUNT
            ),
            "cache_only_reload_passes": True,
            "direct_green_builder_entered_during_reload": bool(entered),
            "passes": not entered,
        },
        "carrier": warm,
        "cold_build": cold
        | {
            "publication_seconds": publication_seconds,
            "total_seconds": (
                cold["total_before_publication_seconds"] + publication_seconds
            ),
        },
        "cache_only_reload": {
            "warm_load_seconds": warm["warm_load_seconds"],
            "direct_builder_import_guard": sorted(DIRECT_BUILDER_MODULES),
            "direct_builder_modules_entered": entered,
            "source_store_opened": False,
            "reached_persisted_carrier": True,
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
    }
    if not report["verdict"]["passes"]:
        raise RuntimeError("cache-only reload entered a direct Green builder")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def check(carrier: Path, receipt: Path) -> dict[str, Any]:
    """Run the named cache-only check and cross-check the committed receipt."""
    with _guard_direct_builders() as guard:
        _response, warm = load_carrier(carrier)
    if guard.entered:
        raise RuntimeError("cache-only check entered a direct Green builder")
    report = json.loads(receipt.read_text(encoding="utf-8"))
    banked = report["carrier"]
    for key in (
        "path",
        "semantic_response_identity",
        "resolved_target_digest",
        "response_sha256",
        "response_shape",
        "stored_circuit_count",
        "file_sha256",
        "size_bytes",
    ):
        if warm[key] != banked[key]:
            raise ValueError(f"receipt field {key} does not match persisted carrier")
    if not report["verdict"]["passes"]:
        raise ValueError("carrier receipt does not carry a passing verdict")
    return warm | {
        "direct_builder_modules_entered": guard.entered,
        "receipt": str(receipt),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("build", "check", "probe"))
    parser.add_argument("--carrier", type=Path, default=DEFAULT_CARRIER)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    arguments = parser.parse_args()
    if arguments.mode == "probe":
        with _guard_direct_builders() as guard:
            _response, report = load_carrier(arguments.carrier)
        print(json.dumps(report | {"direct_builder_modules_entered": guard.entered}))
        return
    if arguments.mode == "build":
        report = build(arguments.carrier, arguments.receipt)
        print(
            "MAST_RESPONSE_CARRIER "
            f"shape={report['carrier']['response_shape']} "
            f"circuits={report['carrier']['stored_circuit_count']} "
            f"cold_seconds={report['cold_build']['total_seconds']:.6f} "
            f"warm_seconds={report['cache_only_reload']['warm_load_seconds']:.6f} "
            f"slurm_job_id={report['runtime']['slurm_job_id']} "
            "verdict=PASS"
        )
        return
    report = check(arguments.carrier, arguments.receipt)
    print(
        "MAST_RESPONSE_CARRIER_CACHE_ONLY "
        f"shape={report['response_shape']} "
        f"circuits={report['stored_circuit_count']} "
        f"warm_seconds={report['warm_load_seconds']:.6f} "
        "direct_builders=0 verdict=PASS"
    )


if __name__ == "__main__":
    main()
