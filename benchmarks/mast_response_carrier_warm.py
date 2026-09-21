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
from dataclasses import dataclass
import fcntl
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


CARRIER_STORE = Path(
    "/work/projects/imas_gpu/sophelio/mast_frozen_six_response_carriers"
)
STORED_CIRCUIT_COUNT = 101


@dataclass(frozen=True)
class CarrierGrid:
    """One response grid and the contract a build on it must satisfy.

    ``axis_points`` is the uniform per-axis node count the case is built on.
    ``None`` takes the stored-axis stride the parity module declares, which is
    the coarse grid this carrier was first frozen on.  At 65 it is the stored
    EFIT axis count itself, so the reference map is carried without any
    interpolation rather than resampled onto a different lattice.

    A grid whose contract values are ``None`` is not yet pinned: its build runs
    in discovery, publishes under whatever identity it computes -- the store
    names every file by its own identity, so a discovery build can never
    overwrite a pinned one -- and prints the contract to record here.  Nothing
    loads an unpinned grid, so the fail-closed property the pins exist for is
    unchanged.
    """

    name: str
    axis_points: int | None
    semantic_identity: str | None
    resolved_target_digest: str | None
    response_shape: tuple[int, int] | None

    @property
    def pinned(self) -> bool:
        """Return whether every contract value of this grid is known."""
        return None not in (
            self.semantic_identity,
            self.resolved_target_digest,
            self.response_shape,
        )

    def path(self, identity: str | None = None) -> Path:
        """Return the store path of this grid, or of a discovered identity."""
        resolved = identity or self.semantic_identity
        if resolved is None:
            raise ValueError(f"grid {self.name} has no identity to resolve a path")
        return CARRIER_STORE / f"{resolved}.npz"


CARRIER_GRIDS: dict[str, CarrierGrid] = {
    "stored-axis-stride": CarrierGrid(
        name="stored-axis-stride",
        axis_points=None,
        semantic_identity=(
            "1d2c4a2b2f448ab8f1ae981031bbaf85fe4ee87f8ed9606fe6847d0fc9f1e994"
        ),
        resolved_target_digest=(
            "5623983f54f144edd70f113bdf66ed60fd4de6b751bb8312a31aa422d158b4a9"
        ),
        response_shape=(1126, 101),
    ),
    "stored-axis-full": CarrierGrid(
        name="stored-axis-full",
        axis_points=65,
        semantic_identity=(
            "75029bb0f932cd4f6f57aa145eb49a938e2b973393ce8ae90b27634b1f6c526d"
        ),
        resolved_target_digest=(
            "eaa48962037c010f204885d16c5482b03721c6c7ea57ace779ada3621cc31783"
        ),
        response_shape=(4262, 101),
    ),
}
#: The grid a new build takes when none is named.  The full stored axes are the
#: declared default; the coarse grid stays pinned and loadable by name.
DEFAULT_CARRIER_GRID = "stored-axis-full"
#: The grid every current consumer resolves through.  It moves to the default
#: once the full-axis grid is pinned AND its consumers are verified against the
#: wider response, because each of them asserts the coarse row count today.
CONSUMER_CARRIER_GRID = "stored-axis-stride"

_CONSUMER = CARRIER_GRIDS[CONSUMER_CARRIER_GRID]
SEMANTIC_RESPONSE_IDENTITY = _CONSUMER.semantic_identity
RESOLVED_TARGET_DIGEST = _CONSUMER.resolved_target_digest
RESPONSE_SHAPE = _CONSUMER.response_shape
DEFAULT_CARRIER = _CONSUMER.path()
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


@contextmanager
def _carrier_build_lock(path: Path) -> Iterator[Path]:
    """Serialize cache misses through a persistent advisory lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(f"{path.suffix}.lock")
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield lock_path


def grid_for_carrier(path: Path) -> CarrierGrid:
    """Return the pinned grid whose identity this carrier file is named by.

    The store names every file by its own semantic identity, so the filename is
    the lookup key and no read is needed to choose the contract.  An identity
    that is pinned nowhere raises rather than defaulting to a grid, because
    defaulting is exactly how a carrier gets verified against the wrong
    contract and passes.
    """
    identity = path.stem
    for grid in CARRIER_GRIDS.values():
        if grid.pinned and grid.semantic_identity == identity:
            return grid
    known = ", ".join(sorted(CARRIER_GRIDS))
    raise ValueError(
        f"carrier identity {identity} is pinned by no grid; known grids: {known}"
    )


def load_carrier(
    path: Path,
    *,
    semantic_identity: str = SEMANTIC_RESPONSE_IDENTITY,
    resolved_target_digest: str = RESOLVED_TARGET_DIGEST,
    response_shape: tuple[int, int] = RESPONSE_SHAPE,
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
        if response.shape != response_shape:
            raise ValueError(
                "persisted response shape does not match frozen contract: "
                f"expected {response_shape}, got {response.shape}"
            )
        if targets.shape != (response_shape[0], 2):
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


def _cold_response(grid: CarrierGrid) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the exact shared response for one grid through the scoring seam.

    A pinned grid must reproduce its recorded identity, digest and shape; an
    unpinned one reports whatever it computed so the contract can be recorded
    before anything is allowed to load it.
    """
    from benchmarks.efit_forward_parity_slice import (
        DECOMPOSITION_BANK,
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
        grid_points=grid.axis_points,
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
    if grid.pinned and identity != grid.semantic_identity:
        raise RuntimeError(
            "resolved response identity changed: "
            f"expected {grid.semantic_identity}, got {identity}"
        )
    if grid.pinned and target_digest != grid.resolved_target_digest:
        raise RuntimeError(
            "resolved-target digest changed: "
            f"expected {grid.resolved_target_digest}, got {target_digest}"
        )
    if grid.pinned and response.shape != grid.response_shape:
        raise RuntimeError(
            f"cold response has shape {response.shape}, expected {grid.response_shape}"
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
        "grid": {
            "name": grid.name,
            "axis_points": grid.axis_points,
            "pinned_before_build": grid.pinned,
            "semantic_response_identity": identity,
            "resolved_target_digest": target_digest,
            "response_shape": list(response.shape),
        },
        "machine_resolution_seconds": machine_seconds,
        "direct_response_build_seconds": response_seconds,
        "total_before_publication_seconds": machine_seconds + response_seconds,
        "section_kernel_evaluations": int(policy["section_kernel_evaluations"]),
        "frozen_references": references,
    }


def _cache_only_subprocess(
    carrier: Path, contract: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load the carrier in a fresh interpreter with direct imports refused."""
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "probe",
        "--carrier",
        str(carrier),
    ]
    if contract is not None:
        command += [
            "--expect-identity",
            str(contract["semantic_response_identity"]),
            "--expect-target-digest",
            str(contract["resolved_target_digest"]),
            "--expect-rows",
            str(int(contract["response_shape"][0])),
        ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=os.environ,
    )
    return json.loads(completed.stdout)


def build(
    carrier: Path | None, receipt: Path, grid: CarrierGrid | None = None
) -> dict[str, Any]:
    """Build, publish and immediately verify one content-addressed carrier.

    A discovery build has no path until its identity is computed, so the store
    path is resolved from the built identity rather than supplied.  The store
    names every file by its own identity, so this can only ever create a new
    file; it cannot land on a pinned one.
    """
    grid = grid or CARRIER_GRIDS[CONSUMER_CARRIER_GRID]
    arrays, cold = _cold_response(grid)
    identity = str(arrays["semantic_response_identity"])
    contract = {
        "semantic_response_identity": identity,
        "resolved_target_digest": str(arrays["resolved_target_digest"]),
        "response_shape": list(arrays["response"].shape),
    }
    carrier = carrier or grid.path(identity)
    with _carrier_build_lock(carrier) as lock_path:
        if carrier.exists():
            raise FileExistsError(
                f"cold publication refuses to replace existing carrier {carrier}"
            )
        temporary = carrier.with_name(f".{carrier.name}.{os.getpid()}.building.npz")
        publication_started = perf_counter()
        try:
            np.savez_compressed(temporary, **arrays)
            temporary.replace(carrier)
        finally:
            if temporary.exists():
                temporary.unlink()
        publication_seconds = perf_counter() - publication_started
    warm = _cache_only_subprocess(carrier, contract)
    entered = warm.pop("direct_builder_modules_entered")
    report = {
        "receipt": "persisted MAST frozen-reference response carrier",
        "verdict": {
            "carrier_persisted": True,
            "semantic_identity_matches": (
                warm["semantic_response_identity"]
                == contract["semantic_response_identity"]
            ),
            "resolved_targets_match": (
                warm["resolved_target_digest"] == contract["resolved_target_digest"]
            ),
            "response_shape_matches": (
                warm["response_shape"] == contract["response_shape"]
            ),
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
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_reservation": os.environ.get("SLURM_JOB_RESERVATION"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "advisory_lock": str(lock_path),
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
    parser.add_argument("--carrier", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument(
        "--grid",
        choices=tuple(CARRIER_GRIDS),
        default=DEFAULT_CARRIER_GRID,
        help="response grid to build; check and probe resolve the consumer grid",
    )
    parser.add_argument("--expect-identity", default=None)
    parser.add_argument("--expect-target-digest", default=None)
    parser.add_argument("--expect-rows", type=int, default=None)
    arguments = parser.parse_args()
    grid = CARRIER_GRIDS[arguments.grid]
    if arguments.mode == "probe":
        carrier = arguments.carrier or DEFAULT_CARRIER
        expected = {}
        if arguments.expect_identity is not None:
            expected["semantic_identity"] = arguments.expect_identity
        if arguments.expect_target_digest is not None:
            expected["resolved_target_digest"] = arguments.expect_target_digest
        if arguments.expect_rows is not None:
            expected["response_shape"] = (arguments.expect_rows, STORED_CIRCUIT_COUNT)
        with _guard_direct_builders() as guard:
            _response, report = load_carrier(carrier, **expected)
        print(json.dumps(report | {"direct_builder_modules_entered": guard.entered}))
        return
    if arguments.mode == "build":
        report = build(arguments.carrier, arguments.receipt, grid)
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
    report = check(arguments.carrier or DEFAULT_CARRIER, arguments.receipt)
    print(
        "MAST_RESPONSE_CARRIER_CACHE_ONLY "
        f"shape={report['response_shape']} "
        f"circuits={report['stored_circuit_count']} "
        f"warm_seconds={report['warm_load_seconds']:.6f} "
        "direct_builders=0 verdict=PASS"
    )


if __name__ == "__main__":
    main()
