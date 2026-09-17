"""Compare every DIII-D gate frame's terminal state against the committed batch.

The five frames of the DIII-D forward-GS gate bank each supply their own stored
boundary and their own cold diverted seed.  This driver solves the five frames
in one reserved H200 job, one compiled program per frame, and runs each frame's
paired strict-exit passes on its own program.  It then compares each frame's
terminal state and terminal residual against the committed batched artifact
left by the earlier width-five measurement, which was taken on the stacked vmap
route: this revision's accelerated history program builds its argument layout
with numpy and so refuses the stacked route as well as the batched solve.  One
receipt is written per frame as that frame's comparison is read out, so a
scheduler expiry loses at most one frame.

The frames are measured one at a time, each in a forked process that exits
before the next frame starts: a frame's compiled program is host memory that
process exit returns, and five frames in one process held 131 GiB resident
after the first frame.  The machine description is built before the first fork,
so that build is paid once.

Every frame's comparison states its identity row count through the certificate
identity comparator's refusal, so a comparison over no state values cannot read
as a machine-precision match.

The committed artifact carries terminal-state hashes and terminal residuals
rather than state arrays.  A frame's absolute terminal-state difference is
therefore taken between its two strict-exit passes at the driver's device, and
the artifact is compared by terminal-state hash and by absolute
terminal-residual difference.  A difference against the artifact is
device-qualified: the artifact was produced under a different H200 allocation.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import gc
import hashlib
import json
import numpy as np
import os
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# The identity comparator's refusal is shared with the certificate identity gate
# rather than restated here: an identity set with no rows must not read as a
# machine-precision match.
from benchmarks.solve_program_size_gate import (  # noqa: E402
    EmptyIdentitySetError,
    _require_identity_rows,
)

DEFAULT_ARTIFACT = (
    ROOT / "docs/figures/batched-operator-boundary/exit-incidence/diiid-width5.json"
)
DEFAULT_RECEIPT_DIR = (
    ROOT / "docs/figures/multi-unit-limiter-wall/diiid-gate-frame-identity"
)
DEFAULT_MACHINE_CACHE = Path(
    "/home/ITER/mcintos/.cache/nova/reckon-artifact-repaired-ring-cache"
)
ARTIFACT_ARMS = ("base", "head")
PRIMARY_ARTIFACT_ARM = "head"
SCHEDULED_CORE_COUNT = 8
# One frame's compile held 131 GiB resident, so the reservation is sized for one
# frame's footprint beside the machine description rather than for all five.
MINIMUM_NODE_MEMORY_MIB = 320 * 1024
DEFAULT_CACHE_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/jax-cache/trip-quantum-profile"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one JSON object")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments], text=True
    ).strip()


def _slug(identity: str) -> str:
    return re.sub(r"[^0-9a-z]+", "-", identity.lower()).strip("-") or "frame"


def _converged(termination: Any) -> bool | None:
    if termination is None:
        return None
    return str(termination) == "converged"


def _write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact_index(artifact: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index the committed artifact's per-frame rows by frame identity."""
    rows = artifact.get("members")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("the committed batched artifact carries no member rows")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row["identity"])
        reference: dict[str, Any] = {}
        for arm in ARTIFACT_ARMS:
            arm_row = row.get(arm)
            if not isinstance(arm_row, dict):
                raise RuntimeError(f"artifact row {identity!r} carries no {arm} arm")
            passes: dict[str, Any] = {}
            for pass_name in ("without_exit", "with_exit"):
                passed = arm_row.get(pass_name)
                if not isinstance(passed, dict):
                    raise RuntimeError(
                        f"artifact row {identity!r} arm {arm} carries no "
                        f"{pass_name} pass"
                    )
                passes[pass_name] = {
                    "terminal_residual": passed.get("terminal_residual"),
                    "terminal_state_sha256": passed.get("terminal_state_sha256"),
                    "termination": passed.get("termination"),
                    "converged": _converged(passed.get("termination")),
                }
            reference[arm] = passes
        indexed[identity] = reference
    return indexed


def _main_arm(passed: dict[str, Any]) -> dict[str, Any]:
    return {
        "executed_trips": passed.get("executed_trips"),
        "terminal_residual": passed.get("terminal_residual"),
        "terminal_state_sha256": passed.get("terminal_state_sha256"),
        "termination": passed.get("termination"),
        "converged": _converged(passed.get("termination")),
        "batch_elapsed_ms": passed.get("batch_elapsed_ms"),
        "batched_ms_per_member": passed.get("batched_ms_per_member"),
    }


def _difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return abs(float(left) - float(right))


def _against_artifact(
    main_arm: dict[str, Any], artifact_arm: dict[str, Any]
) -> dict[str, Any]:
    """Compare one pass at the driver's device against one artifact pass."""
    return {
        "main_terminal_residual": main_arm["terminal_residual"],
        "artifact_terminal_residual": artifact_arm["terminal_residual"],
        "absolute_terminal_residual_difference": _difference(
            main_arm["terminal_residual"], artifact_arm["terminal_residual"]
        ),
        "terminal_state_sha256_identical": (
            main_arm["terminal_state_sha256"] == artifact_arm["terminal_state_sha256"]
        ),
        "main_converged": main_arm["converged"],
        "artifact_converged": artifact_arm["converged"],
    }


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _bit_identity(comparisons: dict[str, Any] | None) -> bool | None:
    """Whether the primary arm's two passes match the committed artifact."""
    primary = (comparisons or {}).get(PRIMARY_ARTIFACT_ARM)
    if primary is None:
        return None
    return all(
        comparison["terminal_state_sha256_identical"] for comparison in primary.values()
    )


def _comparator_self_check() -> str:
    """Prove the shared refusal fires before trusting any comparison."""
    try:
        _require_identity_rows(0, "comparator self-check")
    except EmptyIdentitySetError as error:
        return str(error)
    raise RuntimeError("the identity comparator accepted an empty identity set")


def _flush_output() -> None:
    sys.stdout.flush()
    sys.stderr.flush()


def _spread(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _summarize(rows: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    """Summarize the frames that landed, one forked process each."""
    return {
        "landed_frames": len(rows),
        "strict_exit_fired_members": sum(
            row["strict_qualification"] == "fired" for row in rows
        ),
        "strict_exit_never_members": sum(
            row["strict_qualification"] == "never" for row in rows
        ),
        "bit_identical_converged_members": sum(
            row["terminal_state_bit_identical_where_both_arms_converged"] is True
            for row in rows
        ),
        "saved_executed_trips": sum(
            row["without_exit"]["executed_trips"] - row["with_exit"]["executed_trips"]
            for row in rows
        ),
        "per_frame_compile_seconds": {
            str(row["identity"]): row.get("compile_seconds") for row in rows
        },
        "compile_seconds": _spread(
            [
                row["compile_seconds"]
                for row in rows
                if row.get("compile_seconds") is not None
            ]
        ),
        "without_exit_solve_ms": _spread(
            [row["without_exit"]["timing"]["compile_warm_solve_ms"] for row in rows]
        ),
        "with_exit_solve_ms": _spread(
            [row["with_exit"]["timing"]["compile_warm_solve_ms"] for row in rows]
        ),
        "additional_repetitions_after_first_compiled_solve": repeats,
    }


def _require_h200_allocation(
    *, expected_cpu_count: int, minimum_memory_mib: int
) -> dict[str, Any]:
    """Prove this process holds the reserved H200 the fence names.

    The proof mirrors the instrument's own, with a memory floor where the
    instrument requires one fixed request: a single frame's compile holds about
    131 GiB resident, which does not fit the instrument's stacked-program
    reservation, and the frames are sized for one frame beside the machine
    description instead.  The instrument's proof is in
    benchmarks/strict_exit_incidence.py, outside this node's write scope.
    """
    import jax

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("the measurement requires a SLURM allocation")
    if os.environ.get("SLURM_JOB_PARTITION") != "betelgeuse":
        raise RuntimeError("the measurement requires the betelgeuse partition")
    if os.environ.get("SLURM_JOB_RESERVATION") != "gpu_0003_grpA":
        raise RuntimeError("the measurement requires reservation gpu_0003_grpA")
    if int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != expected_cpu_count:
        raise RuntimeError(
            f"the measurement requires exactly {expected_cpu_count} allocated CPU(s)"
        )
    if os.environ.get("TMPDIR") != "/tmp":
        raise RuntimeError("the measurement requires TMPDIR=/tmp")
    if os.environ.get("JAX_PLATFORMS") != "cuda,cpu":
        raise RuntimeError("the measurement requires JAX_PLATFORMS=cuda,cpu")
    requested_memory_mib = int(os.environ.get("SLURM_MEM_PER_NODE", "0"))
    if requested_memory_mib < minimum_memory_mib:
        raise RuntimeError(
            f"the measurement requires at least {minimum_memory_mib} MiB of node "
            f"memory, received {requested_memory_mib} MiB"
        )
    devices = jax.devices("gpu")
    if len(devices) != 1 or "H200" not in devices[0].device_kind:
        raise RuntimeError(f"the measurement requires one H200, received {devices}")
    return {
        "job_id": int(job_id),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node": os.environ.get("SLURMD_NODENAME"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "reservation": os.environ.get("SLURM_JOB_RESERVATION"),
        "cpu_count": int(os.environ["SLURM_CPUS_PER_TASK"]),
        "device": devices[0].device_kind,
        "jax_platforms": os.environ["JAX_PLATFORMS"].split(","),
        "tmpdir": os.environ["TMPDIR"],
        "requested_time_limit": os.environ.get("SLURM_TIMELIMIT"),
        "requested_memory_mib": requested_memory_mib,
    }


def _fork_device_probe() -> str:
    """Prove a forked child runs a compiled program before the machine is built."""
    import jax
    import jax.numpy as jnp

    def total(value):
        return (value * value).sum()

    program = jax.jit(total).lower(jnp.ones(16)).compile()
    parent_total = float(np.asarray(program(jnp.ones(16))))
    result_path = Path(tempfile.mkdtemp(prefix="nova-fork-probe2-")) / "child.json"
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            child_total = float(np.asarray(program(jnp.ones(16))))
            _write_json(result_path, {"total": child_total})
            code = 0
        except BaseException:
            traceback.print_exc()
        finally:
            _flush_output()
            os._exit(code)
    status = os.waitpid(pid, 0)[1]
    exit_code = os.waitstatus_to_exitcode(status)
    child_total = None
    if result_path.exists():
        child_total = _read_json(result_path).get("total")
    if exit_code != 0 or child_total != parent_total:
        raise RuntimeError(
            "a forked child could not run a compiled program: child exit "
            f"{exit_code}, parent {parent_total!r}, child {child_total!r}"
        )
    return (
        f"child_exit={exit_code} parent_total={parent_total} child_total={child_total}"
    )


def _frame_receipt(
    *,
    header: dict[str, Any],
    row: dict[str, Any],
    index: int,
    member_state_count: int,
    artifact_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    """One frame's identity comparison, written by the frame's own process."""
    identity = str(row["identity"])
    identity_row_count = _require_identity_rows(
        member_state_count, f"DIII-D gate frame {identity!r} terminal state"
    )
    main_arms = {
        "without_exit": _main_arm(row["without_exit"]),
        "with_exit": _main_arm(row["with_exit"]),
    }
    comparisons = None
    if artifact_reference is not None:
        comparisons = {
            arm: {
                pass_name: _against_artifact(
                    main_arms[pass_name], artifact_reference[arm][pass_name]
                )
                for pass_name in ("without_exit", "with_exit")
            }
            for arm in ARTIFACT_ARMS
        }
    return {
        "schema": "nova.diiid-gate-frame-identity/1",
        "header": header,
        "frame": {"identity": identity, "index": index},
        "identity_row_count": identity_row_count,
        "comparison_kind": (
            "one DIII-D gate frame's terminal state at the driver's device "
            "against the committed batched artifact"
        ),
        "maximum_absolute_terminal_state_difference": row["terminal_state_difference"][
            "max_absolute_flux_difference"
        ],
        "main": main_arms,
        "artifact_reference": artifact_reference,
        "maximum_absolute_terminal_state_difference_scope": (
            "between this frame's two strict-exit passes at the driver's device; "
            "the committed artifact carries terminal-state hashes rather than "
            "state arrays, so a state-array difference against it cannot be formed"
        ),
        "strict_qualification": row["strict_qualification"],
        "main_against_artifact": comparisons,
        "primary_artifact_arm": PRIMARY_ARTIFACT_ARM,
        "bit_identical_to_artifact": _bit_identity(comparisons),
    }


def _frame_narrative(receipt: dict[str, Any]) -> str:
    identity = receipt["frame"]["identity"]
    difference = receipt["maximum_absolute_terminal_state_difference"]
    comparisons = receipt["main_against_artifact"]
    if comparisons is None:
        return (
            f"{identity}: no committed artifact row carries this frame; paired-pass "
            f"terminal-state difference {difference:.3e}"
        )
    if receipt["bit_identical_to_artifact"]:
        return (
            f"{identity}: bit-identical terminal state to the committed batched "
            "artifact on both strict-exit passes"
        )
    magnitudes = [
        comparison["absolute_terminal_residual_difference"]
        if comparison["absolute_terminal_residual_difference"] is not None
        else 0.0
        for comparison in comparisons[PRIMARY_ARTIFACT_ARM].values()
    ]
    return (
        f"{identity}: differs from the committed batched artifact; largest absolute "
        f"terminal-residual difference {max(magnitudes):.3e}; paired-pass "
        f"terminal-state difference {difference:.3e}"
    )


def run(arguments: argparse.Namespace) -> int:
    from nova.jax.config import (
        configure_dtypes,
        configure_persistent_compilation_cache,
    )

    from benchmarks import strict_exit_incidence as instrument

    artifact = _read_json(arguments.artifact)
    artifact_index = _artifact_index(artifact)
    header: dict[str, Any] = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "artifact_path": str(arguments.artifact),
        "artifact_sha256": _file_sha256(arguments.artifact),
        "artifact_schema": artifact.get("schema"),
        "artifact_primary_arm": PRIMARY_ARTIFACT_ARM,
        "artifact_secondary_arm": "base",
        "artifact_revisions": artifact.get("source"),
        "device_sha": _git("rev-parse", "HEAD"),
        "device_tree": str(ROOT),
        "driver": str(Path(__file__).relative_to(ROOT)),
        "driver_sha256": _file_sha256(Path(__file__)),
        "machine_cache": str(arguments.machine_cache),
        "machine_artifact_digest": instrument.DEFAULT_MACHINE_ARTIFACT_DIGEST,
        "identity_comparator": (
            "benchmarks.solve_program_size_gate._require_identity_rows"
        ),
        "scheduled_core_count": arguments.cpu_count,
        "declared_member_count": arguments.member_count,
        "requested_repeats": arguments.repeats,
        "cache_root": str(arguments.cache_root),
        "execution_route": (
            "one_compiled_program_per_frame, both strict-exit passes on the same "
            "program: the stacked vmap program the committed artifact was measured "
            "with is refused by this revision's accelerated history program, whose "
            "argument layout converts the vmapped target current with numpy"
        ),
        "route_qualification": (
            "a difference measured on the per-frame route against an artifact "
            "measured on the stacked vmap route is device- and route-qualified"
        ),
        "process_isolation": (
            "one forked child per frame, forked after the machine description is in "
            "memory so the build is paid once, with each frame's receipt written by "
            "its own child: five frames in one process held 131 GiB resident after "
            "the first frame, and process exit is what returns it"
        ),
        "minimum_node_memory_mib": MINIMUM_NODE_MEMORY_MIB,
        "allocation_proof": (
            "driver-local proof of the reserved H200, mirroring the instrument's with "
            "a memory floor: the instrument's proof requires exactly 128 GiB, which "
            "one frame's 131 GiB compile footprint does not fit"
        ),
    }

    configure_dtypes()
    import jax

    if not jax.config.jax_enable_x64:
        raise RuntimeError("extended precision was not enabled before array build")
    cache = configure_persistent_compilation_cache(arguments.cache_root)
    allocation = _require_h200_allocation(
        expected_cpu_count=arguments.cpu_count,
        minimum_memory_mib=MINIMUM_NODE_MEMORY_MIB,
    )
    header.update(
        {
            "job_id": allocation["job_id"],
            "node": allocation["node"],
            "device": allocation["device"],
            "partition": allocation["partition"],
            "reservation": allocation["reservation"],
            "persistent_compilation_cache": cache.receipt(),
            "allocation": allocation,
        }
    )
    probe = _fork_device_probe()
    header["fork_device_probe"] = probe
    print(f"FORK_DEVICE_PROBE={probe}", flush=True)
    print("IDENTITY_HEADER=" + json.dumps(header, sort_keys=True), flush=True)
    for key, value in header.items():
        print(f"HEADER_FIELD {key}={value}", flush=True)
    return _measure(arguments, header, instrument, artifact_index)


def _measure(
    arguments: argparse.Namespace,
    header: dict[str, Any],
    instrument: Any,
    artifact_index: dict[str, Any],
) -> int:
    """Measure one frame per forked process, each writing its own receipt.

    The machine description is built once, in this process, and each fork
    happens after it, so that build is paid once.  The frames are then measured
    one at a time in children that exit, because a frame's compiled program is
    host memory that only process exit returns: five frames in one process held
    131 GiB resident after the first frame.
    """
    members, evidence_inputs = instrument._build_diiid_members(
        arguments.machine_cache, member_count=arguments.member_count
    )
    state_counts = [int(np.asarray(member.state).size) for member in members]
    header["member_identities"] = [member.identity for member in members]
    header["evidence_inputs"] = evidence_inputs
    header["comparator_self_check"] = _comparator_self_check()
    print(f"COMPARATOR_SELF_CHECK=PASS {header['comparator_self_check']}", flush=True)
    exchange = Path(tempfile.mkdtemp(prefix="nova-diiid-frames-"))
    header["frame_exchange_directory"] = str(exchange)
    frames = []
    for index, member in enumerate(members):
        frame = _frame_in_child(
            arguments=arguments,
            instrument=instrument,
            member=member,
            index=index,
            member_state_count=state_counts[index],
            header=header,
            artifact_index=artifact_index,
            exchange=exchange,
        )
        frames.append(frame)
        print(
            f"FRAME_OUTCOME {frame['identity']} status={frame['status']} "
            f"child_pid={frame['child_pid']} "
            f"child_exit_code={frame['child_exit_code']}",
            flush=True,
        )
    del members
    gc.collect()
    return _persist(arguments, header, frames, artifact_index, arguments.repeats)


def _frame_in_child(
    *,
    arguments: argparse.Namespace,
    instrument: Any,
    member: Any,
    index: int,
    member_state_count: int,
    header: dict[str, Any],
    artifact_index: dict[str, Any],
    exchange: Path,
) -> dict[str, Any]:
    """Measure one frame in a child process, and let that process exit.

    The child writes the frame's own receipt before it exits, and hands the
    measurement back through a JSON record in the exchange directory: the
    parent cannot read the child's memory, and the record is what lets the run
    receipt summarize frames that no longer share a process.
    """
    identity = str(member.identity)
    slug = _slug(identity)
    receipt_path = arguments.receipt_dir / f"{slug}.json"
    outcome_path = exchange / f"{slug}.json"
    _flush_output()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            result = instrument._measure_machine(
                [member], arguments.repeats, name="DIIID"
            )
            row = result["members"][0]
            try:
                receipt = _frame_receipt(
                    header=header,
                    row=row,
                    index=index,
                    member_state_count=member_state_count,
                    artifact_reference=artifact_index.get(identity),
                )
            except EmptyIdentitySetError as error:
                _write_json(
                    receipt_path,
                    {
                        "schema": "nova.diiid-gate-frame-identity/1",
                        "header": header,
                        "frame": {"identity": identity, "index": index},
                        "identity_row_count": 0,
                        "identity_refused": str(error),
                        "identity_refused_note": (
                            "the frame's terminal state carried no identity rows, "
                            "so no comparison is formed for it"
                        ),
                    },
                )
                outcome = {
                    "status": "refused",
                    "identity": identity,
                    "index": index,
                    "receipt": _relative(receipt_path),
                    "refusal": str(error),
                }
            else:
                _write_json(receipt_path, receipt)
                outcome = {
                    "status": "measured",
                    "identity": identity,
                    "index": index,
                    "receipt": _relative(receipt_path),
                    "row": row,
                    "execution_contract": result["execution_contract"],
                }
            _write_json(outcome_path, outcome)
            print(f"FRAME_LANDED {identity} {receipt_path}", flush=True)
            code = 0
        except BaseException:
            traceback.print_exc()
            try:
                _write_json(
                    outcome_path,
                    {
                        "status": "failed",
                        "identity": identity,
                        "index": index,
                        "error": traceback.format_exc(),
                    },
                )
            except BaseException:
                traceback.print_exc()
        finally:
            _flush_output()
            os._exit(code)

    status = os.waitpid(pid, 0)[1]
    exit_code = os.waitstatus_to_exitcode(status)
    outcome: dict[str, Any]
    if outcome_path.exists():
        try:
            outcome = _read_json(outcome_path)
        except (OSError, ValueError) as error:
            outcome = {
                "status": "failed",
                "identity": identity,
                "index": index,
                "error": f"the child's outcome record is unreadable: {error}",
            }
    else:
        outcome = {
            "status": "failed",
            "identity": identity,
            "index": index,
            "error": "the child wrote no outcome record",
        }
    outcome.update(
        {
            "child_pid": pid,
            "child_exit_code": exit_code,
            "receipt_path": str(receipt_path),
        }
    )
    return outcome


def _frame_failures(failed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "identity": str(frame["identity"]),
            "child_pid": frame["child_pid"],
            "child_exit_code": frame["child_exit_code"],
            "error": frame.get("error"),
        }
        for frame in failed
    ]


def _persist(
    arguments: argparse.Namespace,
    header: dict[str, Any],
    frames: list[dict[str, Any]],
    artifact_index: dict[str, Any],
    repeats: int,
) -> int:
    """Write the run's own record over the frames whose children landed.

    Each frame's receipt was written by the child that measured it, so this
    reads those receipts back rather than rebuilding them: the run record then
    states what is on disk rather than what the parent believes; a frame whose
    child failed or refused is listed as such and does not abort the others.
    """
    measured = [frame for frame in frames if frame["status"] == "measured"]
    refused = [frame for frame in frames if frame["status"] == "refused"]
    failed = [frame for frame in frames if frame["status"] == "failed"]
    receipts = []
    for frame in measured:
        path = Path(frame["receipt_path"])
        receipt = _read_json(path)
        receipts.append((path, receipt))
        print(f"FRAME_RECEIPT={path} {_frame_narrative(receipt)}", flush=True)
    rows = [frame["row"] for frame in measured]
    contracts = [
        frame["execution_contract"]
        for frame in measured
        if frame.get("execution_contract") is not None
    ]
    execution_contract = None
    if contracts:
        execution_contract = dict(contracts[0])
        execution_contract["member_count"] = len(contracts)
        execution_contract["one_forked_process_per_frame"] = True
        execution_contract["forked_after_member_state_build"] = True
        execution_contract["forked_process_count"] = len(frames)
    host_memory = {
        str(frame["identity"]): (frame.get("row") or {}).get("host_memory")
        for frame in frames
    }
    exit_codes = {str(frame["identity"]): frame["child_exit_code"] for frame in frames}
    run_receipt = {
        "schema": "nova.diiid-gate-frame-identity-run/1",
        "header": header,
        "frame_receipts": [_relative(path) for path, _ in receipts],
        "frames": {
            str(receipt["frame"]["identity"]): {
                "identity_row_count": receipt["identity_row_count"],
                "bit_identical_to_artifact": receipt["bit_identical_to_artifact"],
                "maximum_absolute_terminal_state_difference": (
                    receipt["maximum_absolute_terminal_state_difference"]
                ),
                "child_exit_code": exit_codes[str(receipt["frame"]["identity"])],
                "child_host_memory": host_memory[str(receipt["frame"]["identity"])],
            }
            for _, receipt in receipts
        },
        "artifact_frames_without_a_measured_member": sorted(
            set(artifact_index) - {str(row["identity"]) for row in rows}
        ),
        "execution_contract": execution_contract,
        "summary": _summarize(rows, repeats),
        "frame_failures": _frame_failures(failed),
        "narrative": [_frame_narrative(receipt) for _, receipt in receipts],
        "identity_refusals": [str(frame["refusal"]) for frame in refused],
    }
    _write_json(arguments.receipt_dir / "run.json", run_receipt)
    print(f"RUN_RECEIPT={arguments.receipt_dir / 'run.json'}", flush=True)
    if failed:
        raise RuntimeError(
            "the run record is written and these frames did not land: "
            + ", ".join(str(frame["identity"]) for frame in failed)
        )
    if refused:
        refusals = "; ".join(str(frame["refusal"]) for frame in refused)
        raise EmptyIdentitySetError(refusals)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--machine-cache", type=Path, default=DEFAULT_MACHINE_CACHE)
    parser.add_argument("--member-count", type=int, default=5)
    parser.add_argument("--cpu-count", type=int, default=SCHEDULED_CORE_COUNT)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--probe", action="store_true")
    arguments = parser.parse_args()

    if arguments.probe:
        from benchmarks import strict_exit_incidence as instrument

        print(
            "IMPORT_PROBE=PASS "
            f"comparator={_require_identity_rows.__module__} "
            f"instrument={instrument.__name__} "
            f"receipt_dir={arguments.receipt_dir} artifact={arguments.artifact}",
            flush=True,
        )
        return 0

    return run(arguments)


if __name__ == "__main__":
    sys.exit(main())
