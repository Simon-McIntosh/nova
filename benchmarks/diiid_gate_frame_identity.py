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
import re
import subprocess
import sys
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


def _frame_receipt(
    *,
    header: dict[str, Any],
    row: dict[str, Any],
    index: int,
    member_state_count: int,
    artifact_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    """One frame's identity comparison, ready to persist on its own."""
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
    }

    configure_dtypes()
    import jax

    if not jax.config.jax_enable_x64:
        raise RuntimeError("extended precision was not enabled before array build")
    cache = configure_persistent_compilation_cache(arguments.cache_root)
    allocation = instrument._require_gpu_allocation(
        expected_cpu_count=arguments.cpu_count
    )
    header.update(
        {
            "job_id": allocation["job_id"],
            "node": allocation["node"] or allocation["host"],
            "device": allocation["device"],
            "partition": allocation["partition"],
            "reservation": allocation["reservation"],
            "persistent_compilation_cache": cache.receipt(),
        }
    )
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
    members, evidence_inputs = instrument._build_diiid_members(
        arguments.machine_cache, member_count=arguments.member_count
    )
    state_counts = [int(np.asarray(member.state).size) for member in members]
    header["member_identities"] = [member.identity for member in members]
    header["evidence_inputs"] = evidence_inputs
    result = instrument._measure_machine(members, arguments.repeats, name="DIIID")
    del members
    gc.collect()
    return _guarded_persist(arguments, header, result, state_counts, artifact_index)


def _guarded_persist(
    arguments: argparse.Namespace,
    header: dict[str, Any],
    result: dict[str, Any],
    state_counts: list[int],
    artifact_index: dict[str, Any],
) -> int:
    header["comparator_self_check"] = _comparator_self_check()
    print(f"COMPARATOR_SELF_CHECK=PASS {header['comparator_self_check']}", flush=True)
    return _persist(arguments, header, result, state_counts, artifact_index)


def _persist(
    arguments: argparse.Namespace,
    header: dict[str, Any],
    result: dict[str, Any],
    state_counts: list[int],
    artifact_index: dict[str, Any],
) -> int:
    """Write one receipt per frame, then the run's own record."""
    rows = result["members"]
    if len(rows) != len(state_counts):
        raise RuntimeError("the batched program returned a different member count")
    receipts = []
    refusals = []
    for index, row in enumerate(rows):
        identity = str(row["identity"])
        path = arguments.receipt_dir / f"{_slug(identity)}.json"
        try:
            receipt = _frame_receipt(
                header=header,
                row=row,
                index=index,
                member_state_count=state_counts[index],
                artifact_reference=artifact_index.get(identity),
            )
        except EmptyIdentitySetError as error:
            refusal = {
                "schema": "nova.diiid-gate-frame-identity/1",
                "header": header,
                "frame": {"identity": identity, "index": index},
                "identity_row_count": 0,
                "identity_refused": str(error),
            }
            _write_json(path, refusal)
            refusals.append(str(error))
            print(f"IDENTITY_REFUSED {identity} {error}", flush=True)
            continue
        _write_json(path, receipt)
        receipts.append((path, receipt))
        print(f"FRAME_RECEIPT={path} {_frame_narrative(receipt)}", flush=True)
    measured = [str(row["identity"]) for row in rows]
    run_receipt = {
        "schema": "nova.diiid-gate-frame-identity-run/1",
        "header": header,
        "frame_receipts": [_relative(path) for path, _ in receipts],
        "frames": {
            str(receipt["frame"]["identity"]): {
                "identity_row_count": receipt["identity_row_count"],
                "bit_identical_to_artifact": receipt["bit_identical_to_artifact"],
                "maximum_absolute_terminal_state_difference": receipt[
                    "maximum_absolute_terminal_state_difference"
                ],
            }
            for _, receipt in receipts
        },
        "artifact_frames_without_a_measured_member": sorted(
            set(artifact_index) - set(measured)
        ),
        "execution_contract": result["execution_contract"],
        "compile_seconds": result.get("compile_seconds"),
        "compile_cache": result.get("compile_cache"),
        "summary": result["summary"],
        "narrative": [_frame_narrative(receipt) for _, receipt in receipts],
        "identity_refusals": refusals,
    }
    _write_json(arguments.receipt_dir / "run.json", run_receipt)
    print(f"RUN_RECEIPT={arguments.receipt_dir / 'run.json'}", flush=True)
    if refusals:
        raise EmptyIdentitySetError("; ".join(refusals))
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
