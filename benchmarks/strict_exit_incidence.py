"""Measure strict settled-exit incidence on the committed machine banks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import gc
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
import os
from pathlib import Path
import re
import resource
import socket
import subprocess
import threading
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RectBivariateSpline

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.bank_revision_reproduction import (
    CARRIER_FILE_SHA256,
    CARRIER_SEMANTIC_IDENTITY,
)
from benchmarks.diiid_forward_gs_match import (
    DEFAULT_DATA as DIIID_DATA,
    DEFAULT_MACHINE_ARTIFACT_DIGEST,
    GATE_RESIDUAL_TOLERANCE,
    POLOIDAL_CONDUCTORS,
    REGISTERED_ACCELERATED_GMRES_ITERATIONS,
    REGISTERED_ACCELERATED_NEWTON_STEPS,
    REGISTERED_ACCELERATED_RELAXATION,
    REGISTERED_ACCELERATED_STEP_CAP,
    REGISTERED_ACCELERATED_WARMUP,
    _build_profile,
    _target_current,
    _wall_topology_row,
    complete_profile_current_adapter,
    dataset_machine_description,
    shipped_current_at,
)
from benchmarks.efit_forward_parity_slice import (
    DECOMPOSITION_BANK,
    FIXED_POINT_CRITERION,
    GMRES_ITERATIONS,
    NEWTON_STEPS,
    RELAXATION,
    STEP_CAP,
    TOTAL_FLUX_FACTOR,
    WARMUP_SWEEPS,
    _mast_case_from_selection,
    _passive_inclusive_case,
    select_slices_by_shot,
)
from benchmarks.label_seed_residual_field import _persisted_response_cache
from nova.equilibrium.fixed_point import FixedPointTerminationReason
from nova.equilibrium.forward import SaddleSeedGeometry
from nova.equilibrium.forward import ForwardProfile
from nova.equilibrium.forward_operator import stack_forward_operators
from nova.equilibrium.solve_request import ForwardSolveMemberData, SampledFluxFunction
from nova.equilibrium.source import DomainProfile, ForwardSource
from nova.equilibrium.topology import TopologyClass
from nova.imas.mast_vacuum_cohort import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = (
    ROOT / "docs/figures/millisecond-converged-solve/strict-exit-incidence.json"
)
DEFAULT_PNG = DEFAULT_JSON.with_suffix(".png")
DEFAULT_BATCHED_JSON = (
    ROOT / "docs/figures/batched-operator-boundary/exit-incidence/"
    "batched-exit-incidence.json"
)
MAST_BANK = (
    ROOT / "docs/figures/primary-xpoint-evidence/efit-topology-corroboration.json"
)
DIIID_BANK = (
    ROOT / "docs/figures/diiid-forward-onboarding/forward-gs/forward_gs_receipt.json"
)
SETTLEMENT_CENSUS = (
    ROOT / "docs/figures/solver-trip-orchestration/settlement-histogram.json"
)
BANK_REVISION_ROUTE = ROOT / "benchmarks/bank_revision_reproduction.py"
DEFAULT_MAST_STATE_CACHE = ROOT / "logs/exact-operand-cache.npz"
DEFAULT_DIIID_MACHINE_CACHE = Path(
    "/run/user/39486/reckon-artifact-repaired-ring-cache"
)
REQUIRED_ANCESTOR = "588578a0"
TRIP_LIMIT = 16
CENSUS_PROJECTION_MS_PER_SLICE = 1.163
FULL_TRIP_BASELINE_MS_PER_MEMBER = 415.60787488197093
SETTLED_REASON = int(FixedPointTerminationReason.ACTIVE_SET_SETTLED)


@dataclass(frozen=True)
class Member:
    """One bank member and the profile-specific operands used to solve it."""

    identity: str
    profile: Any
    state: jax.Array
    current: jax.Array | None
    target_current: float
    tolerance: float
    options: dict[str, Any]
    state_authority: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _strict(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict(item) for item in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _termination_name(value: int) -> str:
    try:
        return FixedPointTerminationReason(value).name.lower()
    except ValueError:
        return f"unknown_{value}"


def _require_revision() -> str:
    revision = _git("rev-parse", "HEAD")
    accepted = subprocess.run(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, revision],
        cwd=ROOT,
        check=False,
    )
    if accepted.returncode != 0:
        raise RuntimeError(
            f"measurement revision {revision} does not contain {REQUIRED_ANCESTOR}"
        )
    return revision


def _require_gpu_allocation(expected_cpu_count: int = 1) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("measurement requires a SLURM allocation")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("measurement requires the betelgeuse partition")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("measurement requires reservation gpu_0003_grpA")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != expected_cpu_count:
        raise RuntimeError(
            f"measurement requires exactly {expected_cpu_count} allocated CPU(s)"
        )
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("measurement requires TMPDIR=/tmp")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("measurement requires JAX_PLATFORMS=cuda,cpu")
    requested_memory_mib = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    if requested_memory_mib != 128 * 1024:
        raise RuntimeError(
            "measurement requires exactly 128 GiB of node memory, received "
            f"{requested_memory_mib} MiB"
        )
    devices = jax.devices("gpu")
    if len(devices) != 1 or "H200" not in devices[0].device_kind:
        raise RuntimeError(f"measurement requires one H200, received {devices}")
    return {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "host": socket.gethostname(),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "cpu_count": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "gpu_count": int(os.environ.get("SLURM_GPUS_ON_NODE", "1")),
        "device": devices[0].device_kind,
        "jax_platforms": os.environ["JAX_PLATFORMS"].split(","),
        "tmpdir": os.environ["TMPDIR"],
        "requested_time_limit": os.environ.get("SLURM_TIMELIMIT"),
        "requested_memory_mib": requested_memory_mib,
        "attempt": int(os.environ.get("STRICT_EXIT_ATTEMPT", "1")),
        "prior_job_id": (
            int(os.environ["STRICT_EXIT_PRIOR_JOB_ID"])
            if os.environ.get("STRICT_EXIT_PRIOR_JOB_ID")
            else None
        ),
    }


def _require_cpu_self_check() -> dict[str, Any]:
    """Validate the CPU-allocation proof before a GPU measurement is submitted."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("the CPU self-check requires a SLURM allocation")
    if os.environ.get("SLURM_JOB_PARTITION") != "all_debug":
        raise RuntimeError("the CPU self-check requires the all_debug partition")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != 4:
        raise RuntimeError("the CPU self-check requires exactly four allocated CPUs")
    requested_memory_mib = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    if requested_memory_mib != 64 * 1024:
        raise RuntimeError(
            "the CPU self-check requires exactly 64 GiB of node memory, received "
            f"{requested_memory_mib} MiB"
        )
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("the CPU self-check requires TMPDIR=/tmp")
    if os.environ.get("JAX_PLATFORMS") != "cpu":
        raise RuntimeError("the CPU self-check requires JAX_PLATFORMS=cpu")
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "cpu":
        raise RuntimeError(
            f"the CPU self-check requires one CPU device, received {devices}"
        )
    return {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": socket.gethostname(),
        "host": socket.gethostname(),
        "partition": os.environ["SLURM_JOB_PARTITION"],
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "cpu_count": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "gpu_count": 0,
        "device": devices[0].device_kind,
        "jax_platforms": ["cpu"],
        "tmpdir": os.environ["TMPDIR"],
        "requested_time_limit": os.environ.get("SLURM_TIMELIMIT"),
        "requested_memory_mib": requested_memory_mib,
        "attempt": 0,
        "prior_job_id": None,
        "execution_mode": "slurm_cpu_self_check",
    }


def _mast_cache_rows(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], dict]:
    with np.load(path, allow_pickle=False) as stored:
        metadata = json.loads(str(stored["metadata"].item()))
        if int(metadata.get("arm_count", -1)) != 12:
            raise RuntimeError("the MAST operand cache must carry twelve arms")
        rows = {}
        for index, record in enumerate(metadata["rows"]):
            prefix = f"arm_{index:02d}"
            rows[(str(record["identity"]), str(record["arm"]))] = {
                "metadata": record,
                "radius": np.array(stored[f"{prefix}_radius"], copy=True),
                "height": np.array(stored[f"{prefix}_height"], copy=True),
                "flux": np.array(stored[f"{prefix}_flux"], copy=True),
            }
    if len(rows) != 12:
        raise RuntimeError("the MAST operand cache contains duplicate arm identities")
    return rows, metadata


def _compare_mast_cache_to_bank(
    cached: dict[tuple[str, str], dict[str, Any]], bank_rows: list[dict[str, Any]]
) -> list[str]:
    """Return exact JSON-scalar mismatches in cached and bank solve identities."""

    bank = {(str(row["identity"]), str(row["arm"])): row for row in bank_rows}
    mismatches = []
    for key in sorted(set(cached) - set(bank)):
        mismatches.append(f"cache-only member {key[0]} {key[1]}")
    for key in sorted(set(bank) - set(cached)):
        mismatches.append(f"bank-only member {key[0]} {key[1]}")
    for key in sorted(set(cached) & set(bank)):
        cached_row = cached[key]["metadata"]
        bank_row = bank[key]
        for field in ("shot", "slice_index", "arm", "terminal_residual"):
            cached_value = cached_row.get(field)
            bank_value = bank_row.get(field)
            if cached_value != bank_value:
                mismatches.append(
                    f"{key[0]} {key[1]} {field}: "
                    f"cache={cached_value!r}, bank={bank_value!r}"
                )
    return mismatches


def regenerate_mast_state_cache(state_cache: Path) -> dict[str, Any]:
    """Regenerate exact MAST operands with the established cache producer."""

    producer_path = (
        ROOT / "docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py"
    )
    specification = spec_from_file_location(
        "mast_operand_cache_producer", producer_path
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(
            f"could not load MAST operand cache producer at {producer_path}"
        )
    producer = module_from_spec(specification)
    specification.loader.exec_module(producer)
    original_cache_path = producer.CACHE_PATH
    producer.CACHE_PATH = state_cache
    try:
        configure_dtypes()
        configure_persistent_compilation_cache(
            default_persistent_compilation_cache_root()
        )
        response, carrier = _persisted_response_cache(
            response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
        )
        producer._build_operand_cache(response, carrier)
    finally:
        producer.CACHE_PATH = original_cache_path

    cached, metadata = _mast_cache_rows(state_cache)
    bank_rows = _read_json(MAST_BANK).get("rows", [])
    mismatches = _compare_mast_cache_to_bank(cached, bank_rows)
    if mismatches:
        raise RuntimeError(
            "regenerated MAST operand cache does not match the current bank: "
            + "; ".join(mismatches)
        )
    result = {
        "status": "mast_operand_cache_regenerated",
        "bank": {
            "path": str(MAST_BANK.relative_to(ROOT)),
            "sha256": _sha256(MAST_BANK),
        },
        "cache": {
            "path": str(state_cache),
            "sha256": _sha256(state_cache),
            "arm_count": len(cached),
            "authority": {
                key: metadata.get(key)
                for key in (
                    "schema_revision",
                    "response_carrier_semantic_identity",
                    "selection_source_commit",
                )
            },
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _state_from_cached_grid(
    profile, seed: Any, cached: dict[str, Any]
) -> tuple[Any, str]:
    grid_flux = cached["flux"]
    if grid_flux.size == 0:
        return (
            jnp.asarray(seed),
            "production seed retained for unavailable cached geometry",
        )
    radius = cached["radius"]
    height = cached["height"]
    lattice = profile.lattice
    if grid_flux.shape != (height.size, radius.size):
        raise RuntimeError("cached MAST grid flux does not match its coordinate axes")
    if tuple(lattice.shape) != (radius.size, height.size):
        raise RuntimeError(
            "cached MAST grid does not match the rebuilt profile lattice"
        )
    state = np.asarray(seed, dtype=np.float64).copy()
    grid_count = lattice.node_count
    wall = np.asarray(profile.operator.wall.coordinate, dtype=np.float64)
    physical_count = int(profile.operator.physical_node_number)
    if physical_count != grid_count + len(wall):
        raise RuntimeError("MAST state has an unsupported direct-sampling tail")
    spline = RectBivariateSpline(radius, height, grid_flux.T, kx=3, ky=3, s=0)
    wall_flux = spline.ev(wall[:, 0], wall[:, 1])
    state[:grid_count] = grid_flux.T.reshape(-1)
    state[grid_count:physical_count] = wall_flux
    return (
        jnp.asarray(state),
        "bank regeneration exact grid plus cubic wall interpolation",
    )


def _mast_source_with_sampled_profiles(
    source: ForwardSource, group: Any, row: int
) -> ForwardSource:
    """Carry one extracted MAST profile entirely as dynamic array leaves."""
    psi_norm = np.asarray(group["psi_norm"], dtype=np.float64)
    if psi_norm.shape != (65,) or not np.array_equal(
        psi_norm, np.linspace(0.0, 1.0, 65)
    ):
        raise ValueError("efm/psi_norm is not the declared uniform 65-point base")
    p_prime = -np.asarray(group["pprime"][row], dtype=np.float64) / TOTAL_FLUX_FACTOR
    ff_prime = -np.asarray(group["ffprime"][row], dtype=np.float64) / TOTAL_FLUX_FACTOR
    return replace(
        source,
        core=DomainProfile(
            p_prime=SampledFluxFunction(psi_norm, p_prime),
            ff_prime=SampledFluxFunction(psi_norm, ff_prime),
        ),
    )


def _build_mast_members(
    state_cache: Path, *, member_count: int = 12
) -> tuple[list[Member], dict[str, Any]]:
    bank = _read_json(MAST_BANK)
    rows = bank.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise RuntimeError("the MAST bank must carry twelve arms")
    cached, cache_metadata = _mast_cache_rows(state_cache)
    mismatches = _compare_mast_cache_to_bank(cached, rows)
    if mismatches:
        raise RuntimeError("MAST cache disagrees with bank: " + "; ".join(mismatches))
    response, carrier = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    carrier_record = carrier.get("carrier", carrier)
    if carrier_record["file_sha256"] != CARRIER_FILE_SHA256:
        raise RuntimeError("persisted response carrier file does not match the bank")
    if carrier_record["semantic_response_identity"] != CARRIER_SEMANTIC_IDENTITY:
        raise RuntimeError("persisted response carrier semantics do not match the bank")
    selected = {
        f"{int(row['shot'])}/{int(row['slice_index'])}": (row, qualification)
        for row, qualification in select_slices_by_shot(DECOMPOSITION_BANK)
    }
    profiles: dict[str, tuple[Any, Any, float]] = {}
    members = []
    if not 1 <= member_count <= len(rows):
        raise ValueError(f"MAST member count must be in [1, {len(rows)}]")
    for number, bank_row in enumerate(rows[:member_count], start=1):
        identity = str(bank_row["identity"])
        arm = str(bank_row["arm"])
        key = (identity, arm)
        if identity not in profiles:
            selected_row, qualification = selected[identity]
            case, context = _mast_case_from_selection(
                SHOT_STORE, selected_row, qualification
            )
            passive, profile, policy = _passive_inclusive_case(case, context, response)
            if int(policy["section_kernel_evaluations_this_shot"]) != 0:
                raise RuntimeError("MAST profile entered a direct response builder")
            source = _mast_source_with_sampled_profiles(
                profile.operator.source, context["group"], context["row"]
            )
            operator = replace(
                profile.operator,
                source=source,
                prescribed_current_field=profile.operator.prescribed_field,
            )
            profile = replace(profile, operator=operator)
            target = abs(float(passive["reference"]["plasma_current_a"]))
            profiles[identity] = (profile, passive["state"], target)
            print(
                f"STAGE MAST_PROFILE_READY identity={identity} "
                f"profile={len(profiles)}/6 "
                f"rss_mib={_PeakRssSampler._current_mib():.3f}",
                flush=True,
            )
        profile, seed, target = profiles[identity]
        state, authority = _state_from_cached_grid(profile, seed, cached[key])
        members.append(
            Member(
                identity=f"{identity} {arm}",
                profile=profile,
                state=state,
                current=None,
                target_current=target,
                tolerance=FIXED_POINT_CRITERION,
                options={
                    "newton_steps": NEWTON_STEPS,
                    "gmres_iterations": GMRES_ITERATIONS,
                    "warmup": WARMUP_SWEEPS,
                    "relaxation": RELAXATION,
                    "step_cap": STEP_CAP,
                    "active_set_steps": TRIP_LIMIT,
                },
                state_authority=authority,
            )
        )
        print(
            f"STAGE MAST_MEMBER_READY member={number}/12 "
            f"rss_mib={_PeakRssSampler._current_mib():.3f}",
            flush=True,
        )
    return members, {
        "bank": {
            "path": str(MAST_BANK.relative_to(ROOT)),
            "sha256": _sha256(MAST_BANK),
        },
        "state_cache": {"path": str(state_cache), "sha256": _sha256(state_cache)},
        "state_cache_authority": {
            key: cache_metadata[key]
            for key in (
                "schema_revision",
                "response_carrier_semantic_identity",
                "selection_source_commit",
            )
        },
        "response_carrier": carrier,
        "bank_route": {
            "path": str(BANK_REVISION_ROUTE.relative_to(ROOT)),
            "sha256": _sha256(BANK_REVISION_ROUTE),
            "profile_construction": (
                "passive-inclusive case on the current revision with the exact "
                "persisted response carrier"
            ),
            "current_pin": (
                "absolute bank reference plasma current passed as target_current"
            ),
        },
        "rebuilt_profile_count": len(profiles),
        "profile_representation": (
            "six distinct 65-point MAST profile pairs carried as dynamic array leaves"
        ),
        "member_count": len(members),
    }


def _diiid_machine_artifact_evidence(
    machine_cache: Path, surface_receipts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Record the governed artifact behind cached or transported coordinates."""
    artifact_digest = DEFAULT_MACHINE_ARTIFACT_DIGEST.removeprefix("sha256:")
    manifest = machine_cache / "sha256" / artifact_digest / "manifest.json"
    evidence = {
        "cache": str(machine_cache),
        "digest": DEFAULT_MACHINE_ARTIFACT_DIGEST,
    }
    if manifest.is_file():
        evidence["manifest_sha256"] = _sha256(manifest)
        return evidence
    if surface_receipts and all(
        receipt.get("coordinate_transport") for receipt in surface_receipts
    ):
        coordinate_digests = {
            str(receipt["wall_coordinate_sha256"]) for receipt in surface_receipts
        }
        if len(coordinate_digests) != 1:
            raise RuntimeError("DIII-D members consumed different transported walls")
        evidence.update(
            {
                "manifest_sha256": artifact_digest,
                "coordinate_transport": (
                    "digest-qualified wall coordinates supplied to the compute node"
                ),
                "wall_coordinate_sha256": coordinate_digests.pop(),
            }
        )
        return evidence
    raise FileNotFoundError(
        "DIII-D machine artifact manifest is unavailable and the members did not "
        "consume the verified coordinate transport"
    )


def _build_diiid_members(
    machine_cache: Path, *, member_count: int = 5
) -> tuple[list[Member], dict[str, Any]]:
    bank = _read_json(DIIID_BANK)
    rows = bank.get("result", {}).get("frame_records")
    if not isinstance(rows, list) or len(rows) != 5:
        raise RuntimeError("the DIII-D bank must carry five frame records")
    members = []
    surface_receipts = []
    if not 1 <= member_count <= len(rows):
        raise ValueError(f"DIII-D member count must be in [1, {len(rows)}]")
    for number, bank_row in enumerate(rows[:member_count], start=1):
        shot = str(bank_row["shot"])
        frame = int(bank_row["frame"])
        row = _wall_topology_row(DIIID_DATA / shot)
        built = _build_profile(
            row,
            frame,
            None,
            machine_artifact_cache=machine_cache,
            machine_artifact_digest=DEFAULT_MACHINE_ARTIFACT_DIGEST,
        )
        surface_receipts.append(built.surface_receipt)
        time_ms = float(row["efit_times"][frame])
        machine = dataset_machine_description(row, source_row=str(row["_source_path"]))
        shipped = shipped_current_at(
            row, machine.physical, POLOIDAL_CONDUCTORS, time_ms
        )
        adapter = complete_profile_current_adapter(
            built.profile,
            shipped_names=POLOIDAL_CONDUCTORS,
            shipped_current_a=shipped,
            use_circuit=True,
        )
        profile = adapter.profile
        current = jnp.asarray(adapter.resolution.current(()), dtype=jnp.float64)
        target = float(_target_current(row, time_ms))
        count = int(row["efit_lcfs_n"][frame])
        contour = np.c_[
            np.asarray(row["efit_lcfs_r"][frame][:count], dtype=float),
            np.asarray(row["efit_lcfs_z"][frame][:count], dtype=float),
        ]
        axis = np.asarray(
            (row["efit_r_axis"][frame], row["efit_z_axis"][frame]), dtype=float
        )
        saddle = contour[int(np.argmin(contour[:, 1]))]
        cold = profile.cold_seed_portfolio(
            target,
            axis,
            current=current,
            diverted_geometry=SaddleSeedGeometry(tuple(axis), tuple(saddle)),
        )
        state = cold.branches.flux[int(TopologyClass.DIVERTED)]
        members.append(
            Member(
                identity=f"{shot} frame {frame}",
                profile=profile,
                state=state,
                current=current,
                target_current=target,
                tolerance=GATE_RESIDUAL_TOLERANCE,
                options={
                    "newton_steps": REGISTERED_ACCELERATED_NEWTON_STEPS,
                    "gmres_iterations": REGISTERED_ACCELERATED_GMRES_ITERATIONS,
                    "warmup": REGISTERED_ACCELERATED_WARMUP,
                    "relaxation": REGISTERED_ACCELERATED_RELAXATION,
                    "step_cap": REGISTERED_ACCELERATED_STEP_CAP,
                    "active_set_steps": TRIP_LIMIT,
                },
                state_authority="production cold diverted seed for committed frame",
            )
        )
        print(
            f"STAGE DIIID_MEMBER_READY member={number}/5 "
            f"rss_mib={_PeakRssSampler._current_mib():.3f}",
            flush=True,
        )
    return members, {
        "bank": {
            "path": str(DIIID_BANK.relative_to(ROOT)),
            "sha256": _sha256(DIIID_BANK),
        },
        "machine_artifact": _diiid_machine_artifact_evidence(
            machine_cache, surface_receipts
        ),
        "corpus_root": str(DIIID_DATA),
        "member_count": len(members),
    }


class _PeakRssSampler:
    """Sample this process's resident host memory during one solve pass."""

    def __init__(self, interval_seconds: float = 0.05) -> None:
        self.interval_seconds = interval_seconds
        self.samples_mib: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _current_mib() -> float:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
        raise RuntimeError("/proc/self/status does not expose VmRSS")

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.samples_mib.append(self._current_mib())

    def __enter__(self) -> _PeakRssSampler:
        self.samples_mib.append(self._current_mib())
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.samples_mib.append(self._current_mib())

    def receipt(self) -> dict[str, Any]:
        return {
            "sampling_interval_seconds": self.interval_seconds,
            "initial_rss_mib": self.samples_mib[0],
            "max_rss_mib": max(self.samples_mib),
            "final_rss_mib": self.samples_mib[-1],
            "sample_count": len(self.samples_mib),
        }


def _resource_stage(name: str) -> dict[str, Any]:
    """Record the process RSS and its lifetime high-water mark at one stage."""
    stage = {
        "stage": name,
        "current_rss_mib": _PeakRssSampler._current_mib(),
        "resource_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024.0,
        "resource_peak_scope": "process lifetime through this stage",
    }
    print(
        "STAGE "
        f"{name} current_rss_mib={stage['current_rss_mib']:.3f} "
        f"resource_peak_rss_mib={stage['resource_peak_rss_mib']:.3f}",
        flush=True,
    )
    return stage


def _stack_member_data(members: list[Member]) -> ForwardSolveMemberData:
    """Stack every traced member operand along the program's leading axis."""
    return jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        *[
            ForwardSolveMemberData(
                seed_state=member.state,
                target_current=jnp.asarray(member.target_current),
                current=member.current,
            )
            for member in members
        ],
    )


def _batch_member_solver(lattice, options: dict[str, Any], tolerance: float):
    """Return the one-member solve that receives only traced batch leaves."""

    def solve(operator, data: ForwardSolveMemberData, settlement_enabled):
        profile = ForwardProfile(operator=operator, lattice=lattice)
        return profile.solve(
            data.seed_state,
            route="newton_krylov",
            current=data.current,
            target_current=data.target_current,
            prescribed_current=data.prescribed_current,
            convergence_tolerance=tolerance,
            stop_on_active_set_settlement=settlement_enabled,
            **options,
        )

    return solve


def _select_member(result: Any, index: int) -> Any:
    """Take one member from a vectorized equilibrium receipt."""
    return jax.tree_util.tree_map(
        lambda value: value[index] if getattr(value, "ndim", 0) > 0 else value,
        result,
    )


def _batch_member_result(result: Any, index: int, *, stacked: bool) -> Any:
    """Select a member from either the stacked or sequential execution route."""
    return _select_member(result, index) if stacked else result[index]


def _batch_stage_count(stages: list[dict[str, Any]], name: str) -> None:
    stages.append(_resource_stage(name))


def _compiled_member(member: Member) -> tuple[Callable, jax.Array, float]:
    state = jnp.asarray(member.state)

    def solve(state_value, settlement_enabled):
        return member.profile.solve(
            state_value,
            route="newton_krylov",
            current=member.current,
            target_current=member.target_current,
            convergence_tolerance=member.tolerance,
            stop_on_active_set_settlement=settlement_enabled,
            **member.options,
        )

    started = time.perf_counter()
    compiled = jax.jit(solve).lower(state, jnp.asarray(False)).compile()
    compile_seconds = time.perf_counter() - started
    return compiled, state, compile_seconds


def _block(result: Any) -> None:
    jax.block_until_ready(result.flux)


def _arm_row(result: Any) -> dict[str, Any]:
    fixed = result.fixed_point
    iterations = int(np.asarray(fixed.active_set_iterations))
    reason = int(np.asarray(fixed.termination_reason))
    return {
        "executed_trips": iterations,
        "no_op_trips": TRIP_LIMIT - iterations,
        "terminal_residual": float(np.asarray(fixed.residual, dtype=np.float64)),
        "termination": _termination_name(reason),
        "terminal_state_sha256": _array_sha256(result.flux),
    }


def _time_arm(
    compiled: Callable,
    state: jax.Array,
    *,
    settlement: bool,
    repeats: int,
    machine: str,
    member_number: int,
    arm_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    flag = jnp.asarray(settlement)
    started = time.perf_counter_ns()
    result = compiled(state, flag)
    _block(result)
    first_solve_ms = (time.perf_counter_ns() - started) / 1.0e6
    print(
        f"STAGE {machine}_MEMBER_{member_number}_{arm_name}_COMPILE_WARM "
        f"solve_ms={first_solve_ms:.6f} "
        f"rss_mib={_PeakRssSampler._current_mib():.3f}",
        flush=True,
    )
    samples = [first_solve_ms]
    for repetition in range(repeats):
        started = time.perf_counter_ns()
        result = compiled(state, flag)
        _block(result)
        elapsed = (time.perf_counter_ns() - started) / 1.0e6
        samples.append(elapsed)
        print(
            f"STAGE {machine}_MEMBER_{member_number}_{arm_name}_SAMPLE "
            f"additional_repetition={repetition + 1}/{repeats} "
            f"solve_ms={elapsed:.6f} "
            f"rss_mib={_PeakRssSampler._current_mib():.3f}",
            flush=True,
        )
    return _arm_row(result), {
        "first_compiled_solve_ms": first_solve_ms,
        "samples_compile_warm_ms": samples,
        "compile_warm_solve_ms": float(np.mean(samples)),
        "compile_warm_p95_ms": float(np.percentile(samples, 95)),
    }


def _distribution(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_count": len(array),
        "values": array.tolist(),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _job_receipt(job_id: int) -> dict[str, Any]:
    fields = "JobID,State,Elapsed,ExitCode,NodeList,AllocCPUS,ReqMem,MaxRSS"
    output = subprocess.check_output(
        ["sacct", "-j", str(job_id), "-n", "-P", f"--format={fields}"],
        text=True,
    )
    records = {}
    for line in output.splitlines():
        values = line.split("|")
        if len(values) != 8:
            continue
        records[values[0]] = dict(zip(fields.split(","), values, strict=True))
    allocation = records.get(str(job_id))
    batch = records.get(f"{job_id}.batch")
    if allocation is None or batch is None:
        raise RuntimeError(f"sacct omitted allocation or batch record for {job_id}")
    max_rss = batch["MaxRSS"]
    max_rss_mib = float(max_rss.removesuffix("K")) / 1024.0
    hours, minutes, seconds = (int(value) for value in allocation["Elapsed"].split(":"))
    return {
        "job_id": job_id,
        "scheduler_state": allocation["State"],
        "node": allocation["NodeList"],
        "cpu_count": int(allocation["AllocCPUS"]),
        "requested_memory": allocation["ReqMem"],
        "elapsed": allocation["Elapsed"],
        "elapsed_seconds": hours * 3600 + minutes * 60 + seconds,
        "allocation_exit_code": allocation["ExitCode"],
        "batch_state": batch["State"],
        "batch_exit_code": batch["ExitCode"],
        "batch_max_rss": max_rss,
        "batch_max_rss_mib": max_rss_mib,
        "exit_marker": None,
    }


def _partial_member(
    identity: str,
    compile_seconds: float,
    compile_rss_mib: float,
    without_exit_ms: float,
    without_exit_rss_mib: float,
    with_exit_ms: float,
    with_exit_rss_mib: float,
    member_peak_rss_mib: float,
    released_rss_mib: float,
) -> dict[str, Any]:
    saving_fraction = (without_exit_ms - with_exit_ms) / without_exit_ms
    unavailable = {
        "executed_trips": None,
        "no_op_trips": None,
        "terminal_residual": None,
        "termination": None,
        "terminal_state_sha256": None,
        "result_availability": "not persisted before scheduler timeout",
    }
    return {
        "identity": identity,
        "measurement_state": "paired_timing_complete_semantics_not_persisted",
        "compile_seconds": compile_seconds,
        "strict_qualification_firing_trip": None,
        "strict_qualification": "not_persisted",
        "with_exit": {
            **unavailable,
            "timing": {
                "first_compiled_solve_ms": with_exit_ms,
                "samples_compile_warm_ms": [with_exit_ms],
                "compile_warm_solve_ms": with_exit_ms,
                "compile_warm_p95_ms": with_exit_ms,
            },
            "rss_after_synchronized_solve_mib": with_exit_rss_mib,
        },
        "without_exit": {
            **unavailable,
            "timing": {
                "first_compiled_solve_ms": without_exit_ms,
                "samples_compile_warm_ms": [without_exit_ms],
                "compile_warm_solve_ms": without_exit_ms,
                "compile_warm_p95_ms": without_exit_ms,
            },
            "rss_after_synchronized_solve_mib": without_exit_rss_mib,
        },
        "exit_saving_ms": without_exit_ms - with_exit_ms,
        "exit_saving_fraction": saving_fraction,
        "terminal_state_bit_identical_where_exit_fired": None,
        "host_memory": {
            "rss_after_compile_mib": compile_rss_mib,
            "member_peak_rss_mib": member_peak_rss_mib,
            "released_rss_mib": released_rss_mib,
            "qualification": (
                "the stage log persisted RSS after each pass and the sampler's "
                "whole-member peak; it did not persist an independent peak for "
                "each solve pass"
            ),
        },
    }


def _partial_machine(rows: list[dict[str, Any]], declared: int) -> dict[str, Any]:
    without = [row["without_exit"]["timing"]["compile_warm_solve_ms"] for row in rows]
    with_exit = [row["with_exit"]["timing"]["compile_warm_solve_ms"] for row in rows]
    fractions = [row["exit_saving_fraction"] for row in rows]
    return {
        "execution_contract": {
            "width": 1,
            "declared_member_count": declared,
            "harvested_member_count": len(rows),
            "one_compiled_program_per_member": True,
            "same_program_reused_for_both_arms": True,
            "settlement_flag_is_runtime_argument": True,
            "arm_order_per_member": ["without_exit", "with_exit"],
            "additional_repetitions_after_first_compiled_solve": 0,
        },
        "members": rows,
        "summary": {
            "declared_members": declared,
            "harvested_paired_timing_members": len(rows),
            "semantic_result_members": 0,
            "strict_exit_fired_members": None,
            "strict_exit_never_members": None,
            "bit_identical_fired_members": None,
            "timing": {
                "without_exit_ms": _distribution(without),
                "with_exit_ms": _distribution(with_exit),
                "compile_seconds": _distribution(
                    [row["compile_seconds"] for row in rows]
                ),
                "exit_saving_fraction": _distribution(fractions),
            },
        },
    }


def harvest_partial(
    log_path: Path, output_json: Path, output_png: Path
) -> dict[str, Any]:
    """Recover synchronized paired timings from a time-limited stage log."""
    log_text = log_path.read_text(encoding="utf-8")
    starts = re.finditer(
        r"^STAGE (MAST|DIIID)_MEMBER_(\d+)_COMPILE_START width=1 "
        r"identity='([^']+)' rss_mib=([0-9.]+)$",
        log_text,
        flags=re.MULTILINE,
    )
    staged: dict[tuple[str, int], dict[str, Any]] = {}
    for match in starts:
        key = (match.group(1), int(match.group(2)))
        staged[key] = {
            "identity": match.group(3),
            "compile_start_rss_mib": float(match.group(4)),
        }
    patterns = {
        "compile": (
            r"^STAGE (MAST|DIIID)_MEMBER_(\d+)_COMPILE_DONE "
            r"seconds=([0-9.]+) rss_mib=([0-9.]+)$"
        ),
        "without_exit": (
            r"^STAGE (MAST|DIIID)_MEMBER_(\d+)_WITHOUT_EXIT_COMPILE_WARM "
            r"solve_ms=([0-9.]+) rss_mib=([0-9.]+)$"
        ),
        "with_exit": (
            r"^STAGE (MAST|DIIID)_MEMBER_(\d+)_WITH_EXIT_COMPILE_WARM "
            r"solve_ms=([0-9.]+) rss_mib=([0-9.]+)$"
        ),
        "released": (
            r"^STAGE (MAST|DIIID)_MEMBER_(\d+)_RELEASED "
            r"max_rss_mib=([0-9.]+) rss_mib=([0-9.]+)$"
        ),
    }
    for stage, pattern in patterns.items():
        for match in re.finditer(pattern, log_text, flags=re.MULTILINE):
            key = (match.group(1), int(match.group(2)))
            if key not in staged:
                raise RuntimeError(
                    f"{stage} stage appeared before compile start: {key}"
                )
            staged[key][stage] = (float(match.group(3)), float(match.group(4)))

    completed: dict[str, list[dict[str, Any]]] = {"MAST": [], "DIII-D": []}
    incomplete = []
    for (machine_token, member_number), record in staged.items():
        missing_stages = [stage for stage in patterns if stage not in record]
        machine = "DIII-D" if machine_token == "DIIID" else machine_token
        if missing_stages:
            incomplete.append(
                {
                    "machine": machine,
                    "member_number": member_number,
                    "identity": record["identity"],
                    "compile_start_rss_mib": record["compile_start_rss_mib"],
                    "missing_stages": missing_stages,
                }
            )
            continue
        compile_seconds, compile_rss = record["compile"]
        without_ms, without_rss = record["without_exit"]
        with_ms, with_rss = record["with_exit"]
        peak_rss, released_rss = record["released"]
        completed[machine].append(
            _partial_member(
                record["identity"],
                compile_seconds,
                compile_rss,
                without_ms,
                without_rss,
                with_ms,
                with_rss,
                peak_rss,
                released_rss,
            )
        )

    if len(completed["MAST"]) != 12 or len(completed["DIII-D"]) != 2:
        raise RuntimeError(
            "partial harvest expected twelve MAST and two DIII-D paired rows, "
            f"received {len(completed['MAST'])} and {len(completed['DIII-D'])}"
        )
    diiid_bank = _read_json(DIIID_BANK)["result"]["frame_records"]
    missing_diiid = [
        {
            "member_number": index,
            "identity": f"{row['shot']} frame {row['frame']}",
            "reason": (
                "compile interrupted by scheduler timeout"
                if index == 3
                else "not reached before scheduler timeout"
            ),
        }
        for index, row in enumerate(diiid_bank, start=1)
        if index >= 3
    ]
    job_match = re.search(r"HEARTBEAT .* job=(\d+)", log_text)
    if job_match is None:
        raise RuntimeError("stage log does not identify its SLURM job")
    job = _job_receipt(int(job_match.group(1)))
    revision = _git("rev-parse", "HEAD")
    driver_path = Path(__file__).relative_to(ROOT)
    measured_driver = subprocess.check_output(
        ["git", "show", f"{revision}:{driver_path}"], cwd=ROOT
    )
    machines = {
        "MAST": _partial_machine(completed["MAST"], 12),
        "DIII-D": _partial_machine(completed["DIII-D"], 5),
    }
    machines["DIII-D"]["missing_members"] = missing_diiid
    mast_values = [
        *machines["MAST"]["summary"]["timing"]["with_exit_ms"]["values"],
        *machines["MAST"]["summary"]["timing"]["without_exit_ms"]["values"],
    ]
    diiid_values = [
        *machines["DIII-D"]["summary"]["timing"]["with_exit_ms"]["values"],
        *machines["DIII-D"]["summary"]["timing"]["without_exit_ms"]["values"],
    ]
    diiid_compiles = machines["DIII-D"]["summary"]["timing"]["compile_seconds"]
    diiid_stage_peak_rss = max(
        row["host_memory"]["member_peak_rss_mib"] for row in completed["DIII-D"]
    )
    payload = {
        "schema": "nova.strict-exit-incidence/1",
        "measurement_state": "partial_scheduler_timeout",
        "recorded_at": datetime.now(UTC).isoformat(),
        "completion": {
            "declared_member_count": 17,
            "paired_timing_complete_member_count": 14,
            "semantic_result_member_count": 0,
            "missing_member_count": 3,
            "missing_members": missing_diiid,
            "incomplete_attempts": incomplete,
            "semantic_result_limitation": (
                "the driver assembled trip counts, terminal residuals, termination, "
                "state hashes, and bit-identity in memory but wrote them only after "
                "all members; the scheduler timeout prevented the atomic receipt "
                "write, and those values are absent from the flushed stage log"
            ),
        },
        "source": {
            "measurement_revision": revision,
            "required_ancestor": REQUIRED_ANCESTOR,
            "driver": str(driver_path),
            "measurement_driver_sha256": hashlib.sha256(measured_driver).hexdigest(),
            "harvest_driver_sha256": _sha256(Path(__file__)),
            "stage_log": str(log_path),
            "stage_log_sha256": _sha256(log_path),
            "solver_source_modified": False,
            "nova_diff_stat": subprocess.check_output(
                ["git", "diff", "--stat", "--", "nova"], cwd=ROOT, text=True
            ).splitlines(),
        },
        "execution": {
            **job,
            "partition": "betelgeuse",
            "reservation": "gpu_0003_grpA",
            "gpu_count": 1,
            "device": "NVIDIA H200",
            "jax_platforms": ["cuda", "cpu"],
            "tmpdir": "/tmp",
            "requested_time_limit": "03:00:00",
            "persistent_compilation_cache": True,
            "heartbeats_flushed": True,
            "stage_timings_flushed": True,
        },
        "configuration": {
            "trip_limit": TRIP_LIMIT,
            "execution_width": 1,
            "member_counts": {"MAST": 12, "DIII-D": 5},
            "paired_control": (
                "one width-1 compiled program per member received "
                "stop_on_active_set_settlement as a runtime boolean and was reused "
                "first with the exit disabled and then with the exit enabled"
            ),
            "memory_lifecycle": (
                "members were compiled and measured sequentially; each member's "
                "compiled host and device buffers were released before the next"
            ),
            "solve_ms_scope": (
                "wall time from invoking the compiled top-level solve through "
                "jax.block_until_ready(result.flux); it includes the complete trip "
                "loop, per-trip host reconciliation, any retrace or compilation "
                "triggered beneath the top-level boundary, and final device sync"
            ),
            "latency_qualification": (
                "width-1 compile-warm solve_ms is per-member latency, not batched "
                "milliseconds per slice; batched throughput awaits the boundary "
                "that separates shared geometry from member-varying operator data"
            ),
        },
        "evidence_inputs": {
            "MAST": {
                "bank": {
                    "path": str(MAST_BANK.relative_to(ROOT)),
                    "sha256": _sha256(MAST_BANK),
                },
                "bank_route": {
                    "path": str(BANK_REVISION_ROUTE.relative_to(ROOT)),
                    "sha256": _sha256(BANK_REVISION_ROUTE),
                    "current_pin": "current revision production bank route",
                },
            },
            "DIII-D": {
                "bank": {
                    "path": str(DIIID_BANK.relative_to(ROOT)),
                    "sha256": _sha256(DIIID_BANK),
                },
                "machine_artifact_digest": DEFAULT_MACHINE_ARTIFACT_DIGEST,
            },
            "settlement_census": {
                "path": str(SETTLEMENT_CENSUS.relative_to(ROOT)),
                "sha256": _sha256(SETTLEMENT_CENSUS),
            },
        },
        "machines": machines,
        "observations": {
            "width_one_latency": (
                f"MAST paired warm solves span {min(mast_values) / 1000:.1f} to "
                f"{max(mast_values) / 1000:.1f} s per member; the two completed "
                f"DIII-D members span {min(diiid_values) / 1000:.1f} to "
                f"{max(diiid_values) / 1000:.1f} s. These are seconds-scale "
                "width-1 latencies, not millisecond-scale batched throughput."
            ),
            "exit_saving": (
                "the paired exit effect is reported per member as "
                "(without_exit_ms - with_exit_ms) / without_exit_ms; it is not "
                "converted into a per-trip quantum because trip counts were not "
                "persisted"
            ),
            "heterogeneous_frame_compile": (
                "the two completed DIII-D members compiled in "
                f"{diiid_compiles['values'][0]:.1f} s and "
                f"{diiid_compiles['values'][1]:.1f} s; stage RSS reached "
                f"{diiid_stage_peak_rss:.1f} "
                "MiB and the batch MaxRSS reached "
                f"{job['batch_max_rss_mib']:.1f} MiB. This is direct evidence of "
                "the heterogeneous-frame recompile cost named by the batched "
                "operator-boundary work."
            ),
        },
        "head_per_trip_quantum_ms": {
            "value": None,
            "reason": "executed trip counts were not persisted before timeout",
            "supersedes_banked_ms": 25.0,
        },
        "comparison_baselines": {
            "strict_census_projection_ms_per_slice": CENSUS_PROJECTION_MS_PER_SLICE,
            "full_trip_baseline_ms_per_member": FULL_TRIP_BASELINE_MS_PER_MEMBER,
            "comparison_qualification": (
                "the 1.163 ms/slice census projection is batched throughput and is "
                "not directly comparable to width-1 latency"
            ),
        },
        "project_absolute_figure_src": (
            "/nova/figures/millisecond-converged-solve/strict-exit-incidence.png"
        ),
    }
    _draw(payload, output_png)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"PARTIAL_RECEIPT_WRITTEN={output_json}", flush=True)
    print(f"PARTIAL_FIGURE_WRITTEN={output_png}", flush=True)
    return payload


def _measure_batched_machine(
    members: list[Member],
    *,
    name: str,
) -> dict[str, Any]:
    """Compile one checked member program and run its paired exit passes."""
    if not members:
        raise ValueError("a batched machine measurement needs at least one member")
    options = members[0].options
    tolerance = members[0].tolerance
    if any(member.options != options for member in members):
        raise RuntimeError(f"{name} members do not share one solve option set")
    if any(member.tolerance != tolerance for member in members):
        raise RuntimeError(f"{name} members do not share one solve tolerance")

    batch = stack_forward_operators(member.profile.operator for member in members)
    member_data = _stack_member_data(members)
    stages: list[dict[str, Any]] = []
    _batch_stage_count(stages, f"{name}_BATCH_READY")
    geometry_digests = [operator.geometry_identity for operator in batch.operators]
    solver = _batch_member_solver(members[0].profile.lattice, options, tolerance)
    flags = {
        "without_exit": jnp.zeros(batch.width, dtype=bool),
        "with_exit": jnp.ones(batch.width, dtype=bool),
    }

    compile_count = 0
    if batch.stacked is not None:

        def batched_solve(operator, data, settlement):
            return jax.vmap(solver)(operator, data, settlement)

        _batch_stage_count(stages, f"{name}_BATCH_COMPILE_START")
        started = time.perf_counter()
        compiled = (
            jax.jit(batched_solve)
            .lower(batch.stacked, member_data, flags["without_exit"])
            .compile()
        )
        compile_seconds = time.perf_counter() - started
        compile_count = 1
        _batch_stage_count(stages, f"{name}_BATCH_COMPILE_DONE")

        arm_results = {}
        arm_timings = {}
        for arm_name, settlement in flags.items():
            _batch_stage_count(stages, f"{name}_{arm_name.upper()}_START")
            started = time.perf_counter_ns()
            result = compiled(batch.stacked, member_data, settlement)
            _block(result)
            elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
            arm_results[arm_name] = result
            arm_timings[arm_name] = elapsed_ms
            _batch_stage_count(stages, f"{name}_{arm_name.upper()}_DONE")
    else:
        compile_seconds = None
        arm_results = {}
        arm_timings = {}
        for arm_name, settlement in flags.items():
            _batch_stage_count(stages, f"{name}_{arm_name.upper()}_START")
            started = time.perf_counter_ns()
            result = batch.map(solver, member_data, settlement)
            jax.block_until_ready(result)
            arm_results[arm_name] = result
            arm_timings[arm_name] = (time.perf_counter_ns() - started) / 1.0e6
            compile_count += len(batch.geometry_groups)
            _batch_stage_count(stages, f"{name}_{arm_name.upper()}_DONE")

    rows = []
    for index, member in enumerate(members):
        without_exit = _batch_member_result(
            arm_results["without_exit"], index, stacked=batch.stacked is not None
        )
        with_exit = _batch_member_result(
            arm_results["with_exit"], index, stacked=batch.stacked is not None
        )
        control = _arm_row(without_exit)
        exited = _arm_row(with_exit)
        fired = exited["termination"] == _termination_name(SETTLED_REASON)
        bit_identical = (
            control["terminal_state_sha256"] == exited["terminal_state_sha256"]
            if fired
            else None
        )
        if fired and not bit_identical:
            raise RuntimeError(
                f"strict exit changed terminal state bits for {member.identity}"
            )
        rows.append(
            {
                "identity": member.identity,
                "state_authority": member.state_authority,
                "initial_state_sha256": _array_sha256(member.state),
                "strict_qualification_firing_trip": (
                    exited["executed_trips"] if fired else None
                ),
                "strict_qualification": "fired" if fired else "never",
                "without_exit": {
                    **control,
                    "batch_elapsed_ms": arm_timings["without_exit"],
                    "batched_ms_per_member": arm_timings["without_exit"] / batch.width,
                },
                "with_exit": {
                    **exited,
                    "batch_elapsed_ms": arm_timings["with_exit"],
                    "batched_ms_per_member": arm_timings["with_exit"] / batch.width,
                },
                "terminal_state_bit_identical_where_exit_fired": bit_identical,
            }
        )

    _batch_stage_count(stages, f"{name}_BATCH_COMPLETE")
    del member_data, arm_results
    if batch.stacked is not None:
        del compiled
    gc.collect()
    jax.clear_caches()
    gc.collect()
    _batch_stage_count(stages, f"{name}_BATCH_RELEASED")
    return {
        "execution_contract": {
            "width": batch.width,
            "declared_member_count": len(members),
            "same_arguments_for_exit_passes": True,
            "settlement_flag_is_member_runtime_argument": True,
            "arm_order": ["without_exit", "with_exit"],
            "geometry_identical": batch.geometry_identical,
            "geometry_groups": [list(group) for group in batch.geometry_groups],
            "geometry_digests": geometry_digests,
            "execution_route": (
                "one_stacked_vmap_program"
                if batch.stacked is not None
                else "sequential_jitted_geometry_fallback"
            ),
            "compile_count_per_pass": compile_count,
            "compile_once": compile_count == 1,
        },
        "compile_seconds": compile_seconds,
        "stage_host_memory": stages,
        "members": rows,
        "summary": {
            "declared_members": len(rows),
            "strict_exit_fired_members": sum(
                row["strict_qualification"] == "fired" for row in rows
            ),
            "strict_exit_never_members": sum(
                row["strict_qualification"] == "never" for row in rows
            ),
            "bit_identical_fired_members": sum(
                row["terminal_state_bit_identical_where_exit_fired"] is True
                for row in rows
            ),
            "batched_ms_per_member": {
                arm: arm_timings[arm] / batch.width for arm in arm_timings
            },
            "peak_host_rss_mib": max(
                stage["resource_peak_rss_mib"] for stage in stages
            ),
        },
    }


def _measure_machine(members: list[Member], repeats: int, name: str) -> dict[str, Any]:
    rows = []
    for member_number, member in enumerate(members, start=1):
        print(
            f"STAGE {name}_MEMBER_{member_number}_COMPILE_START width=1 "
            f"identity={member.identity!r} "
            f"rss_mib={_PeakRssSampler._current_mib():.3f}",
            flush=True,
        )
        sampler = _PeakRssSampler()
        with sampler:
            compiled, state, compile_seconds = _compiled_member(member)
            print(
                f"STAGE {name}_MEMBER_{member_number}_COMPILE_DONE "
                f"seconds={compile_seconds:.6f} "
                f"rss_mib={_PeakRssSampler._current_mib():.3f}",
                flush=True,
            )
            control, control_timing = _time_arm(
                compiled,
                state,
                settlement=False,
                repeats=repeats,
                machine=name,
                member_number=member_number,
                arm_name="WITHOUT_EXIT",
            )
            exited, exited_timing = _time_arm(
                compiled,
                state,
                settlement=True,
                repeats=repeats,
                machine=name,
                member_number=member_number,
                arm_name="WITH_EXIT",
            )
        memory = sampler.receipt()
        fired = exited["termination"] == _termination_name(SETTLED_REASON)
        bit_identical = (
            control["terminal_state_sha256"] == exited["terminal_state_sha256"]
            if fired
            else None
        )
        if fired and not bit_identical:
            raise RuntimeError(
                f"strict exit changed terminal state bits for {member.identity}"
            )
        del compiled, state
        gc.collect()
        jax.clear_caches()
        gc.collect()
        released_rss_mib = _PeakRssSampler._current_mib()
        print(
            f"STAGE {name}_MEMBER_{member_number}_RELEASED "
            f"max_rss_mib={memory['max_rss_mib']:.3f} "
            f"rss_mib={released_rss_mib:.3f}",
            flush=True,
        )
        rows.append(
            {
                "identity": member.identity,
                "state_authority": member.state_authority,
                "initial_state_sha256": _array_sha256(member.state),
                "compile_seconds": compile_seconds,
                "strict_qualification_firing_trip": (
                    exited["executed_trips"] if fired else None
                ),
                "strict_qualification": "fired" if fired else "never",
                "with_exit": {**exited, "timing": exited_timing},
                "without_exit": {**control, "timing": control_timing},
                "terminal_state_bit_identical_where_exit_fired": bit_identical,
                "host_memory": {
                    **memory,
                    "released_rss_mib": released_rss_mib,
                },
            }
        )

    timing = {
        "without_exit_ms": _distribution(
            [row["without_exit"]["timing"]["compile_warm_solve_ms"] for row in rows]
        ),
        "with_exit_ms": _distribution(
            [row["with_exit"]["timing"]["compile_warm_solve_ms"] for row in rows]
        ),
        "compile_seconds": _distribution([row["compile_seconds"] for row in rows]),
    }
    saved_trips = sum(
        row["without_exit"]["executed_trips"] - row["with_exit"]["executed_trips"]
        for row in rows
    )
    total_saved_ms = sum(
        row["without_exit"]["timing"]["compile_warm_solve_ms"]
        - row["with_exit"]["timing"]["compile_warm_solve_ms"]
        for row in rows
    )
    paired_quantum = total_saved_ms / saved_trips if saved_trips > 0 else None
    control_trip_count = sum(row["without_exit"]["executed_trips"] for row in rows)
    control_total_ms = sum(
        row["without_exit"]["timing"]["compile_warm_solve_ms"] for row in rows
    )
    return {
        "execution_contract": {
            "width": 1,
            "member_count": len(members),
            "unique_member_inputs": len(
                {_array_sha256(member.state) for member in members}
            ),
            "one_compiled_program_per_member": True,
            "same_program_reused_for_both_arms": True,
            "settlement_flag_is_runtime_argument": True,
            "arm_order_per_member": ["without_exit", "with_exit"],
            "additional_repetitions_after_first_compiled_solve": repeats,
        },
        "members": rows,
        "summary": {
            "declared_members": len(rows),
            "strict_exit_fired_members": sum(
                row["strict_qualification"] == "fired" for row in rows
            ),
            "strict_exit_never_members": sum(
                row["strict_qualification"] == "never" for row in rows
            ),
            "bit_identical_fired_members": sum(
                row["terminal_state_bit_identical_where_exit_fired"] is True
                for row in rows
            ),
            "saved_executed_trips": saved_trips,
            "timing": timing,
            "paired_saved_trip_quantum_ms": paired_quantum,
            "control_executed_trip_quantum_ms": (
                control_total_ms / control_trip_count if control_trip_count else None
            ),
        },
    }


def _draw(payload: dict[str, Any], output: Path) -> None:
    figure = plt.figure(figsize=(15.2, 10.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.15))
    colors = {"MAST": "#512b81", "DIII-D": "#1596a7"}
    for column, machine in enumerate(("MAST", "DIII-D")):
        rows = payload["machines"][machine]["members"]
        labels = [row["identity"].replace(" frame ", "\nframe ") for row in rows]
        positions = np.arange(len(rows))

        axis = figure.add_subplot(grid[0, column])
        firing = [row["strict_qualification_firing_trip"] for row in rows]
        displayed = [TRIP_LIMIT + 1 if value is None else value for value in firing]
        markers = [
            "x" if row["strict_qualification"] == "not_persisted" else "o"
            for row in rows
        ]
        for position, value, marker in zip(positions, displayed, markers, strict=True):
            axis.scatter(position, value, s=54, color=colors[machine], marker=marker)
        axis.axhline(TRIP_LIMIT, color="#999999", linewidth=1.0, linestyle="--")
        axis.set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
        axis.set_yticks(
            [1, 4, 8, 12, 16, 17],
            ["1", "4", "8", "12", "16", "not persisted"],
        )
        axis.set_ylim(0.25, 17.8)
        axis.set_ylabel("strict qualification firing trip")
        axis.set_title(f"{machine}: member-specific strict exit incidence")
        axis.spines[["top", "right"]].set_visible(False)

        timing_axis = figure.add_subplot(grid[1, column])
        control = [
            row["without_exit"]["timing"]["compile_warm_solve_ms"] for row in rows
        ]
        exited = [row["with_exit"]["timing"]["compile_warm_solve_ms"] for row in rows]
        timing_axis.plot(
            positions,
            control,
            marker="o",
            linewidth=1.3,
            color="#747474",
            label="strict exit disabled",
        )
        timing_axis.plot(
            positions,
            exited,
            marker="o",
            linewidth=1.3,
            color=colors[machine],
            label="strict exit enabled",
        )
        timing_axis.set_xticks(positions, labels, rotation=55, ha="right", fontsize=8)
        timing_axis.set_ylabel("compile-warm solve ms at width 1")
        timing_axis.set_title(f"{machine}: paired per-member latency")
        timing_axis.legend(frameon=False)
        timing_axis.spines[["top", "right"]].set_visible(False)
    state = payload.get("measurement_state", "complete").replace("_", " ")
    missing = payload.get("completion", {}).get("missing_members", [])
    missing_note = ""
    if missing:
        missing_note = "\nmissing DIII-D: " + ", ".join(
            row["identity"].replace("d3d_shot_", "").replace(".parquet frame ", "/")
            for row in missing
        )
    figure.suptitle(
        "Strict settled exit on real bank members — sequential width-1 solves\n"
        f"measurement state: {state}{missing_note}",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def preflight(mast_state_cache: Path, diiid_machine_cache: Path) -> dict[str, Any]:
    revision = _require_revision()
    mast = _read_json(MAST_BANK)
    diiid = _read_json(DIIID_BANK)
    mast_rows = mast.get("rows", [])
    diiid_rows = diiid.get("result", {}).get("frame_records", [])
    cached, metadata = _mast_cache_rows(mast_state_cache)
    cache_mismatches = _compare_mast_cache_to_bank(cached, mast_rows)
    cache_matches_current_bank = not cache_mismatches
    manifest = (
        diiid_machine_cache
        / "sha256"
        / DEFAULT_MACHINE_ARTIFACT_DIGEST.removeprefix("sha256:")
        / "manifest.json"
    )
    result = {
        "status": "preflight_complete",
        "source_revision": revision,
        "mast_members": len(mast_rows),
        "mast_cache_members": len(cached),
        "mast_cache_matches_current_bank": cache_matches_current_bank,
        "mast_cache_bank_rule": (
            "exact equality of JSON-decoded shot, slice_index, arm, and "
            "terminal_residual scalars for an identical member-key set"
        ),
        "mast_cache_bank_mismatches": cache_mismatches,
        "mast_cache_authority": {
            key: metadata.get(key)
            for key in (
                "schema_revision",
                "response_carrier_semantic_identity",
                "selection_source_commit",
            )
        },
        "diiid_members": len(diiid_rows),
        "diiid_machine_manifest": str(manifest),
        "diiid_machine_manifest_exists": manifest.is_file(),
        "census_projection_ms_per_slice": CENSUS_PROJECTION_MS_PER_SLICE,
        "census_sha256": _sha256(SETTLEMENT_CENSUS),
    }
    if (
        len(mast_rows) != 12
        or len(cached) != 12
        or len(diiid_rows) != 5
        or not cache_matches_current_bank
    ):
        raise RuntimeError(f"bank cardinality preflight failed: {result}")
    if not manifest.is_file():
        raise RuntimeError(f"DIII-D machine artifact is unavailable at {manifest}")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def _banked_sequential_comparison() -> dict[str, Any]:
    """Read, without rewriting, the prior width-one banked timing receipt."""
    receipt = _read_json(DEFAULT_JSON)
    machines = receipt["machines"]
    comparison = {}
    for name in ("MAST", "DIII-D"):
        summary = machines[name]["summary"]
        timing = summary["timing"]
        comparison[name] = {
            "receipt_path": str(DEFAULT_JSON.relative_to(ROOT)),
            "measurement_state": receipt["measurement_state"],
            "paired_timing_member_count": summary["harvested_paired_timing_members"],
            "without_exit_mean_ms": timing["without_exit_ms"]["mean"],
            "with_exit_mean_ms": timing["with_exit_ms"]["mean"],
        }
    return comparison


def _write_batched_report(
    report_path: Path,
    payload: dict[str, Any],
) -> None:
    """Write the compact machine comparison required by the batch receipt."""
    lines = [
        "# Strict-exit batch incidence",
        "",
        "| Machine | Width | Route | Compile count/pass | Compile seconds | "
        "Batched disabled ms/member | Banked width-1 disabled ms/member | "
        "Batched enabled ms/member | Banked width-1 enabled ms/member | "
        "Peak host RSS MiB | Under 16 GiB |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name in ("MAST", "DIII-D"):
        machine = payload["machines"][name]
        contract = machine["execution_contract"]
        summary = machine["summary"]
        baseline = payload["sequential_width_one_comparison"][name]
        lines.append(
            "| {name} | {width} | {route} | {compiles} | {compile_seconds} | "
            "{without:.6f} | {sequential_without:.6f} | {with_exit:.6f} | "
            "{sequential_with:.6f} | {peak:.3f} | {under_memory} |".format(
                name=name,
                width=contract["width"],
                route=contract["execution_route"],
                compiles=contract["compile_count_per_pass"],
                compile_seconds=(
                    "not one program"
                    if machine["compile_seconds"] is None
                    else f"{machine['compile_seconds']:.6f}"
                ),
                without=summary["batched_ms_per_member"]["without_exit"],
                sequential_without=baseline["without_exit_mean_ms"],
                with_exit=summary["batched_ms_per_member"]["with_exit"],
                sequential_with=baseline["with_exit_mean_ms"],
                peak=summary["peak_host_rss_mib"],
                under_memory=summary["peak_host_rss_mib"] < 16 * 1024,
            )
        )
    lines.extend(
        [
            "",
            "The sequential values are the banked width-1 receipt means. "
            "DIII-D's banked receipt was partial, so its paired timing count is "
            "retained in the JSON receipt and should not be interpreted as all five "
            "members.",
            "",
            "The JSON receipt records every member's strict firing trip, each stage's "
            "`resource.getrusage` host-RSS high-water mark, and geometry digests when "
            "the batch cannot take the one-program route.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _persist_batched_receipt(output_json: Path, payload: dict[str, Any]) -> None:
    """Atomically checkpoint each completed machine into the shared receipt."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_json.with_name(f".{output_json.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_json)


def run_batched(
    output_json: Path,
    mast_state_cache: Path,
    diiid_machine_cache: Path,
    report_path: Path | None,
    *,
    self_check: bool = False,
) -> dict[str, Any]:
    """Measure the real banks, or their allocation-free two-member CPU proof."""
    total_started = time.perf_counter()
    revision = _require_revision()
    configure_dtypes()
    allocation = (
        _require_cpu_self_check()
        if self_check
        else _require_gpu_allocation(expected_cpu_count=1)
    )
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    input_stages: list[dict[str, Any]] = []
    _batch_stage_count(input_stages, "INPUT_BUILD_START")
    mast_started = time.perf_counter()
    mast_members, mast_inputs = _build_mast_members(
        mast_state_cache, member_count=2 if self_check else 12
    )
    mast_build_seconds = time.perf_counter() - mast_started
    _batch_stage_count(input_stages, "MAST_INPUT_BUILD_DONE")
    mast_result = _measure_batched_machine(mast_members, name="MAST")
    del mast_members
    gc.collect()
    _batch_stage_count(input_stages, "MAST_INPUT_RELEASED")
    sequential_comparison = _banked_sequential_comparison()
    mast_peak_rss_mib = max(
        *[stage["resource_peak_rss_mib"] for stage in input_stages],
        mast_result["summary"]["peak_host_rss_mib"],
    )
    mast_checkpoint = {
        "schema": "nova.strict-exit-incidence/2",
        "measurement_state": "mast_complete_diiid_pending",
        "recorded_at": datetime.now(UTC).isoformat(),
        "checkpoint": {
            "completed_machines": ["MAST"],
            "pending_machines": ["DIII-D"],
            "persistence": "atomically replaced after each completed machine",
        },
        "source": {
            "revision": revision,
            "required_ancestor": REQUIRED_ANCESTOR,
            "driver": str(Path(__file__).relative_to(ROOT)),
            "driver_sha256": _sha256(Path(__file__)),
            "solver_source_modified": False,
        },
        "execution": {
            **allocation,
            "elapsed_seconds": time.perf_counter() - total_started,
            "exit_marker": None,
            "persistent_compilation_cache": cache.receipt(),
            "input_build_seconds": {"MAST": mast_build_seconds},
        },
        "configuration": {
            "trip_limit": TRIP_LIMIT,
            "program_widths": {"MAST": len(mast_result["members"]), "DIII-D": 5},
            "real_bank_widths": {"MAST": 12, "DIII-D": 5},
            "self_check": self_check,
            "paired_control": (
                "each machine program receives the same stacked member arguments "
                "for one exit-disabled and one exit-enabled runtime-flag pass"
            ),
            "strict_exit_definition": (
                "zero mask difference, own-mask acceptance, zero accepted Newton "
                "promotions, and bit-identical retained incoming state"
            ),
            "host_memory_limit_mib": 16 * 1024,
            "host_memory_peak_mib": mast_peak_rss_mib,
            "host_memory_under_limit": mast_peak_rss_mib < 16 * 1024,
            "host_memory_stages": input_stages,
        },
        "evidence_inputs": {
            "MAST": mast_inputs,
            "sequential_width_one_receipt": {
                "path": str(DEFAULT_JSON.relative_to(ROOT)),
                "sha256": _sha256(DEFAULT_JSON),
            },
        },
        "machines": {"MAST": mast_result},
        "sequential_width_one_comparison": sequential_comparison,
        "observations": {
            "mast_compile_once": mast_result["execution_contract"]["compile_once"],
            "batch_compile_count_per_pass": {
                "MAST": mast_result["execution_contract"]["compile_count_per_pass"]
            },
            "geometry_fallbacks": {
                "MAST": {
                    "fallback": not mast_result["execution_contract"][
                        "geometry_identical"
                    ],
                    "geometry_digests": mast_result["execution_contract"][
                        "geometry_digests"
                    ],
                }
            },
        },
    }
    _persist_batched_receipt(output_json, mast_checkpoint)
    print(f"BATCHED_MACHINE_RECEIPT_WRITTEN=MAST:{output_json}", flush=True)

    diiid_started = time.perf_counter()
    diiid_members, diiid_inputs = _build_diiid_members(
        diiid_machine_cache, member_count=2 if self_check else 5
    )
    diiid_build_seconds = time.perf_counter() - diiid_started
    _batch_stage_count(input_stages, "DIIID_INPUT_BUILD_DONE")
    diiid_result = _measure_batched_machine(diiid_members, name="DIIID")
    del diiid_members
    gc.collect()
    _batch_stage_count(input_stages, "DIIID_INPUT_RELEASED")
    machines = {"MAST": mast_result, "DIII-D": diiid_result}
    peak_rss_mib = max(
        *[stage["resource_peak_rss_mib"] for stage in input_stages],
        *[machine["summary"]["peak_host_rss_mib"] for machine in machines.values()],
    )
    payload = {
        "schema": "nova.strict-exit-incidence/2",
        "measurement_state": "complete",
        "recorded_at": datetime.now(UTC).isoformat(),
        "source": {
            "revision": revision,
            "required_ancestor": REQUIRED_ANCESTOR,
            "driver": str(Path(__file__).relative_to(ROOT)),
            "driver_sha256": _sha256(Path(__file__)),
            "solver_source_modified": False,
        },
        "execution": {
            **allocation,
            "elapsed_seconds": time.perf_counter() - total_started,
            "exit_marker": 0,
            "persistent_compilation_cache": cache.receipt(),
            "input_build_seconds": {
                "MAST": mast_build_seconds,
                "DIII-D": diiid_build_seconds,
            },
        },
        "configuration": {
            "trip_limit": TRIP_LIMIT,
            "program_widths": {
                "MAST": len(mast_result["members"]),
                "DIII-D": len(diiid_result["members"]),
            },
            "real_bank_widths": {"MAST": 12, "DIII-D": 5},
            "self_check": self_check,
            "paired_control": (
                "each machine program receives the same stacked member arguments "
                "for one exit-disabled and one exit-enabled runtime-flag pass"
            ),
            "strict_exit_definition": (
                "zero mask difference, own-mask acceptance, zero accepted Newton "
                "promotions, and bit-identical retained incoming state"
            ),
            "host_memory_limit_mib": 16 * 1024,
            "host_memory_peak_mib": peak_rss_mib,
            "host_memory_under_limit": peak_rss_mib < 16 * 1024,
            "host_memory_stages": input_stages,
        },
        "evidence_inputs": {
            "MAST": mast_inputs,
            "DIII-D": diiid_inputs,
            "sequential_width_one_receipt": {
                "path": str(DEFAULT_JSON.relative_to(ROOT)),
                "sha256": _sha256(DEFAULT_JSON),
            },
        },
        "machines": machines,
        "sequential_width_one_comparison": sequential_comparison,
        "checkpoint": {
            "completed_machines": ["MAST", "DIII-D"],
            "pending_machines": [],
            "persistence": "atomically replaced after each completed machine",
        },
        "observations": {
            "mast_compile_once": machines["MAST"]["execution_contract"]["compile_once"],
            "batch_compile_count_per_pass": {
                name: machine["execution_contract"]["compile_count_per_pass"]
                for name, machine in machines.items()
            },
            "geometry_fallbacks": {
                name: {
                    "fallback": not machine["execution_contract"]["geometry_identical"],
                    "geometry_digests": machine["execution_contract"][
                        "geometry_digests"
                    ],
                }
                for name, machine in machines.items()
            },
        },
    }
    payload["execution"]["elapsed_seconds"] = time.perf_counter() - total_started
    _persist_batched_receipt(output_json, payload)
    if report_path is not None:
        _write_batched_report(report_path, payload)
    print(f"BATCHED_RECEIPT_WRITTEN={output_json}", flush=True)
    print("EXIT_MARKER=0", flush=True)
    return payload


def run(
    output_json: Path,
    output_png: Path,
    mast_state_cache: Path,
    diiid_machine_cache: Path,
    repeats: int,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    revision = _require_revision()
    configure_dtypes()
    allocation = _require_gpu_allocation()
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    print(
        f"STAGE INPUT_BUILD_START rss_mib={_PeakRssSampler._current_mib():.3f}",
        flush=True,
    )
    build_started = time.perf_counter()
    mast_members, mast_inputs = _build_mast_members(mast_state_cache)
    mast_build_seconds = time.perf_counter() - build_started
    print(
        f"STAGE MAST_INPUT_BUILD_DONE seconds={mast_build_seconds:.6f} "
        f"rss_mib={_PeakRssSampler._current_mib():.3f}",
        flush=True,
    )
    mast_result = _measure_machine(mast_members, repeats, "MAST")
    del mast_members
    gc.collect()
    jax.clear_caches()
    gc.collect()
    print(
        f"STAGE MAST_MACHINE_RELEASED rss_mib={_PeakRssSampler._current_mib():.3f}",
        flush=True,
    )

    build_started = time.perf_counter()
    diiid_members, diiid_inputs = _build_diiid_members(diiid_machine_cache)
    diiid_build_seconds = time.perf_counter() - build_started
    print(
        f"STAGE DIIID_INPUT_BUILD_DONE seconds={diiid_build_seconds:.6f} "
        f"rss_mib={_PeakRssSampler._current_mib():.3f}",
        flush=True,
    )
    diiid_result = _measure_machine(diiid_members, repeats, "DIIID")
    del diiid_members
    gc.collect()
    jax.clear_caches()
    gc.collect()
    print(
        f"STAGE DIIID_MACHINE_RELEASED rss_mib={_PeakRssSampler._current_mib():.3f}",
        flush=True,
    )
    machines = {
        "MAST": mast_result,
        "DIII-D": diiid_result,
    }
    paired_quanta = [
        machine["summary"]["paired_saved_trip_quantum_ms"]
        for machine in machines.values()
        if machine["summary"]["paired_saved_trip_quantum_ms"] is not None
    ]
    control_trip_quanta = [
        machine["summary"]["control_executed_trip_quantum_ms"]
        for machine in machines.values()
    ]
    payload = {
        "schema": "nova.strict-exit-incidence/1",
        "measurement_state": "complete",
        "recorded_at": datetime.now(UTC).isoformat(),
        "source": {
            "revision": revision,
            "required_ancestor": REQUIRED_ANCESTOR,
            "driver": str(Path(__file__).relative_to(ROOT)),
            "driver_sha256": _sha256(Path(__file__)),
            "solver_source_modified": False,
            "nova_diff_stat": subprocess.check_output(
                ["git", "diff", "--stat", "--", "nova"], cwd=ROOT, text=True
            ).splitlines(),
        },
        "execution": {
            **allocation,
            "elapsed_seconds": time.perf_counter() - total_started,
            "exit_marker": 0,
            "persistent_compilation_cache": cache.receipt(),
            "input_build_seconds": {
                "MAST": mast_build_seconds,
                "DIII-D": diiid_build_seconds,
            },
        },
        "configuration": {
            "trip_limit": TRIP_LIMIT,
            "paired_control": (
                "one width-1 compiled program per member receives "
                "stop_on_active_set_settlement as a runtime boolean and is reused "
                "for the exit-disabled and exit-enabled solves"
            ),
            "memory_lifecycle": (
                "each member is compiled and measured independently; compiled host "
                "and device buffers are released before the next member"
            ),
            "strict_exit_definition": (
                "zero mask difference, own-mask acceptance, zero accepted Newton "
                "promotions, and bit-identical retained incoming state"
            ),
            "execution_width": 1,
            "member_counts": {"MAST": 12, "DIII-D": 5},
            "additional_repetitions_after_first_compiled_solve": repeats,
            "latency_qualification": (
                "width-1 compile-warm solve milliseconds are per-member latency, "
                "not batched milliseconds per slice; batched throughput awaits an "
                "external API boundary separating host geometry from member data"
            ),
        },
        "evidence_inputs": {
            "MAST": mast_inputs,
            "DIII-D": diiid_inputs,
            "settlement_census": {
                "path": str(SETTLEMENT_CENSUS.relative_to(ROOT)),
                "sha256": _sha256(SETTLEMENT_CENSUS),
            },
        },
        "machines": machines,
        "head_per_trip_quantum_ms": {
            "paired_saved_trip_mean": (
                float(np.mean(paired_quanta)) if paired_quanta else None
            ),
            "control_executed_trip_weighted_mean": float(np.mean(control_trip_quanta)),
            "per_machine": {
                machine: {
                    "paired_saved_trip": value["summary"][
                        "paired_saved_trip_quantum_ms"
                    ],
                    "control_executed_trip": value["summary"][
                        "control_executed_trip_quantum_ms"
                    ],
                }
                for machine, value in machines.items()
            },
            "supersedes_banked_ms": 25.0,
        },
        "comparison_baselines": {
            "strict_census_projection_ms_per_slice": CENSUS_PROJECTION_MS_PER_SLICE,
            "full_trip_baseline_ms_per_member": FULL_TRIP_BASELINE_MS_PER_MEMBER,
            "comparison_qualification": (
                "the 1.163 ms/slice census projection is a batched throughput "
                "projection and is not directly comparable to width-1 latency"
            ),
        },
        "project_absolute_figure_src": (
            "/nova/figures/millisecond-converged-solve/strict-exit-incidence.png"
        ),
    }
    _draw(payload, output_png)
    payload["execution"]["elapsed_seconds"] = time.perf_counter() - total_started
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(_strict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"RECEIPT_WRITTEN={output_json}", flush=True)
    print(f"FIGURE_WRITTEN={output_png}", flush=True)
    print("EXIT_MARKER=0", flush=True)
    return payload


def _validate_arguments(*, batched: bool, self_check: bool) -> None:
    if self_check and not batched:
        raise ValueError("--self-check requires --batched")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument(
        "--mast-state-cache", type=Path, default=DEFAULT_MAST_STATE_CACHE
    )
    parser.add_argument(
        "--diiid-machine-cache", type=Path, default=DEFAULT_DIIID_MACHINE_CACHE
    )
    parser.add_argument("--repeats", type=int, default=0)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--regenerate-mast-state-cache", action="store_true")
    parser.add_argument("--harvest-log", type=Path)
    parser.add_argument("--batched", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 0:
        raise ValueError("additional timing repetitions cannot be negative")
    _validate_arguments(batched=arguments.batched, self_check=arguments.self_check)
    if arguments.regenerate_mast_state_cache:
        regenerate_mast_state_cache(arguments.mast_state_cache.resolve())
        return
    if arguments.preflight:
        preflight(
            arguments.mast_state_cache.resolve(),
            arguments.diiid_machine_cache.resolve(),
        )
        return
    if arguments.harvest_log is not None:
        harvest_partial(
            arguments.harvest_log.resolve(),
            arguments.json.resolve(),
            arguments.png.resolve(),
        )
        return
    if arguments.batched:
        output_json = (
            DEFAULT_BATCHED_JSON if arguments.json == DEFAULT_JSON else arguments.json
        )
        run_batched(
            output_json.resolve(),
            arguments.mast_state_cache.resolve(),
            arguments.diiid_machine_cache.resolve(),
            None if arguments.report is None else arguments.report.resolve(),
            self_check=arguments.self_check,
        )
        return
    run(
        arguments.json.resolve(),
        arguments.png.resolve(),
        arguments.mast_state_cache.resolve(),
        arguments.diiid_machine_cache.resolve(),
        arguments.repeats,
    )


if __name__ == "__main__":
    main()
