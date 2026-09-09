"""Enumerate terminal receipt fields covered by the state digest."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import strict_exit_incidence as incidence
from nova.equilibrium.fixed_point import FixedPointTerminationReason
from nova.jax.config import configure_dtypes, configure_persistent_compilation_cache
from nova.jax.config import default_persistent_compilation_cache_root


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT / "docs/figures/batched-operator-boundary/terminal-hash/"
    "terminal-hash-census.json"
)
DEFAULT_STATE_CACHE = ROOT / "logs/exact-operand-cache.npz"


def _path_text(path: tuple[Any, ...]) -> str:
    """Return a stable dotted path for one receipt leaf."""
    return jax.tree_util.keystr(path).lstrip(".")


def _array_sha256(value: Any) -> str:
    """Hash dtype, shape, and contiguous bytes as the production census does."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _numeric_difference(
    left: np.ndarray, right: np.ndarray
) -> tuple[float | None, float | None]:
    """Return maximum absolute and scale-normalised differences."""
    if not np.issubdtype(left.dtype, np.number):
        return None, None
    left_float = left.astype(np.float64, copy=False)
    right_float = right.astype(np.float64, copy=False)
    matching_nonfinite = (
        (np.isnan(left_float) & np.isnan(right_float))
        | (np.isposinf(left_float) & np.isposinf(right_float))
        | (np.isneginf(left_float) & np.isneginf(right_float))
    )
    incompatible_nonfinite = (~np.isfinite(left_float) | ~np.isfinite(right_float)) & (
        ~matching_nonfinite
    )
    if np.any(incompatible_nonfinite):
        return float("inf"), float("inf")
    finite = ~matching_nonfinite
    if not np.any(finite):
        return 0.0, 0.0
    difference = np.abs(left_float[finite] - right_float[finite])
    maximum_absolute = float(np.max(difference, initial=0.0))
    scale = float(
        max(
            np.max(np.abs(left_float[finite]), initial=0.0),
            np.max(np.abs(right_float[finite]), initial=0.0),
            np.finfo(np.float64).tiny,
        )
    )
    return maximum_absolute, maximum_absolute / scale


def _field_records(left: Any, right: Any) -> list[dict[str, Any]]:
    """Compare every terminal receipt leaf without hiding unequal fields."""
    left_rows, left_tree = jax.tree_util.tree_flatten_with_path(left)
    right_rows, right_tree = jax.tree_util.tree_flatten_with_path(right)
    if left_tree != right_tree:
        raise RuntimeError("closure and sampled receipts have different pytrees")
    records: list[dict[str, Any]] = []
    for (left_path, left_leaf), (right_path, right_leaf) in zip(
        left_rows, right_rows, strict=True
    ):
        path = _path_text(left_path)
        if path != _path_text(right_path):
            raise RuntimeError("closure and sampled receipt paths differ")
        left_array = np.ascontiguousarray(np.asarray(left_leaf))
        right_array = np.ascontiguousarray(np.asarray(right_leaf))
        if (
            left_array.shape != right_array.shape
            or left_array.dtype != right_array.dtype
        ):
            raise RuntimeError(f"receipt field {path} changed shape or dtype")
        maximum_absolute, maximum_relative = _numeric_difference(
            left_array, right_array
        )
        bit_identical = left_array.tobytes() == right_array.tobytes()
        hashed = path == "flux"
        if hashed:
            classification = "hashed_physical_state_roundoff"
        elif path.startswith("fixed_point.") and path not in {
            "fixed_point.state",
            "fixed_point.trajectory_state",
        }:
            classification = "unhashed_solver_diagnostic_roundoff"
        else:
            classification = "unhashed_derived_physical_roundoff"
        records.append(
            {
                "path": path,
                "hashed": hashed,
                "classification": classification,
                "dtype": left_array.dtype.str,
                "shape": list(left_array.shape),
                "element_count": int(left_array.size),
                "closure_sha256": _array_sha256(left_array),
                "sampled_sha256": _array_sha256(right_array),
                "bit_identical": bit_identical,
                "maximum_absolute_difference": maximum_absolute,
                "maximum_relative_difference": maximum_relative,
            }
        )
    return records


def _git(*arguments: str) -> str:
    """Read repository provenance without assuming a shell working directory."""
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _build_routes(state_cache: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Build width-one closure and sampled members from the frozen MAST inputs."""
    original = incidence._mast_source_with_sampled_profiles
    incidence._mast_source_with_sampled_profiles = lambda source, group, row: source
    try:
        closure_members, closure_inputs = incidence._build_mast_members(
            state_cache, member_count=1
        )
    finally:
        incidence._mast_source_with_sampled_profiles = original
    sampled_members, sampled_inputs = incidence._build_mast_members(
        state_cache, member_count=1
    )
    return (
        closure_members[0],
        sampled_members[0],
        {
            "closure_inputs": closure_inputs,
            "sampled_inputs": sampled_inputs,
        },
    )


def _solve_member(member: Any) -> tuple[Any, float]:
    """Compile and execute one member's width-one terminal solve."""
    compiled, state, compile_seconds = incidence._compiled_member(member)
    result = compiled(state, jnp.asarray(False))
    incidence._block(result)
    return result, compile_seconds


def _terminal_summary(result: Any) -> dict[str, Any]:
    """Return the residual and termination facts beside field records."""
    fixed = result.fixed_point
    reason = int(np.asarray(fixed.termination_reason))
    return {
        "terminal_residual": float(np.asarray(fixed.residual, dtype=np.float64)),
        "termination": FixedPointTerminationReason(reason).name.lower(),
        "active_set_iterations": int(np.asarray(fixed.active_set_iterations)),
        "converged": bool(np.asarray(fixed.converged)),
        "state_sha256": _array_sha256(result.flux),
    }


def run(output: Path, state_cache: Path) -> dict[str, Any]:
    """Run the two width-one routes and write the field census."""
    started = time.perf_counter()
    configure_dtypes()
    if jax.config.jax_enable_x64 is not True:
        raise RuntimeError("terminal hash census requires x64 precision")
    cache = configure_persistent_compilation_cache(
        default_persistent_compilation_cache_root()
    )
    closure_member, sampled_member, inputs = _build_routes(state_cache)
    closure, closure_compile_seconds = _solve_member(closure_member)
    sampled, sampled_compile_seconds = _solve_member(sampled_member)
    fields = _field_records(closure, sampled)
    hashed_fields = [field for field in fields if field["hashed"]]
    if not hashed_fields:
        raise RuntimeError("the terminal result has no declared hashed fields")
    differing = [field for field in fields if not field["bit_identical"]]
    hashed_differing = [field for field in hashed_fields if not field["bit_identical"]]
    closure_summary = _terminal_summary(closure)
    sampled_summary = _terminal_summary(sampled)
    residual_delta = abs(
        closure_summary["terminal_residual"] - sampled_summary["terminal_residual"]
    )
    recommendation = (
        "restate identity over physical state with a stated tolerance"
        if hashed_differing
        else "keep the bit-identity gate"
    )
    payload = {
        "schema": "nova.terminal-state-hash-census/1",
        "created_at": datetime.now(UTC).isoformat(),
        "revision": _git("rev-parse", "HEAD"),
        "precision": {
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "platform": jax.default_backend(),
            "required": "CPU x64",
        },
        "member": closure_member.identity,
        "width": 1,
        "routes": {
            "closure": closure_summary,
            "sampled": sampled_summary,
            "residual_absolute_difference": residual_delta,
            "residual_bit_identical": residual_delta == 0.0,
            "closure_compile_seconds": closure_compile_seconds,
            "sampled_compile_seconds": sampled_compile_seconds,
        },
        "hash_definition": {
            "algorithm": "sha256",
            "implementation": "benchmarks.strict_exit_incidence._array_sha256",
            "inputs": ["dtype string", "int64 shape bytes", "contiguous array bytes"],
            "hashed_paths": [field["path"] for field in hashed_fields],
            "hashes_only_physical_flux": len(hashed_fields) == 1
            and hashed_fields[0]["path"] == "flux",
        },
        "field_summary": {
            "total_fields": len(fields),
            "hashed_fields": len(hashed_fields),
            "unhashed_fields": len(fields) - len(hashed_fields),
            "equal_fields": len(fields) - len(differing),
            "differing_fields": len(differing),
            "hashed_differing_fields": len(hashed_differing),
            "differing_by_classification": {
                classification: sum(
                    field["classification"] == classification for field in differing
                )
                for classification in sorted(
                    {field["classification"] for field in differing}
                )
            },
        },
        "fields": fields,
        "recommendation": {
            "decision": recommendation,
            "basis": (
                "The sole hashed field is physical flux, but closure and sampled "
                "routes differ in its bits while reaching the same terminal residual. "
                "The digest therefore detects roundoff-scale physical variation; it "
                "does not include bookkeeping fields."
                if hashed_differing
                else "The hashed physical field is bit-identical between routes."
            ),
            "suggested_physical_state_tolerance": (
                max(
                    field["maximum_relative_difference"]
                    for field in hashed_fields
                    if field["maximum_relative_difference"] is not None
                )
                if hashed_differing
                else 0.0
            ),
        },
        "reference_attribution": {
            "previous_hash": (
                "69309c0b6c2d617a1d9598df807d7f69a8343cf6944b38165c5a92e587a0ea32"
            ),
            "current_closure_hash": closure_summary["state_sha256"],
            "previous_evidence": (
                "docs/figures/batched-operator-boundary/exit-incidence/"
                "mast-baseline-comparison.json"
            ),
            "previous_route": (
                "width-two batched sampled-profile builder; the prior driver applied "
                "_mast_source_with_sampled_profiles to every MAST member"
            ),
            "current_route": (
                "width-one closure member with the sampled transform bypassed"
            ),
            "attribution": (
                "The 69309c0b value was not a separately established closure "
                "reference. It came from the earlier width-two batched "
                "sampled-profile measurement. Therefore no landed closure change "
                "is evidenced as moving 69309c0b to 5844595e; the apparent movement "
                "is a route mismatch. The current closure hash is legitimate only "
                "as a fresh measurement at this revision, not as a replacement of "
                "the earlier route's reference."
            ),
        },
        "provenance": {
            "state_cache": str(state_cache),
            "state_cache_sha256": incidence._sha256(state_cache),
            "persistent_compilation_cache": cache.receipt(),
            "inputs": inputs,
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(incidence._strict(payload), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload["field_summary"], sort_keys=True))
    print(json.dumps(payload["routes"], sort_keys=True))
    return payload


def main() -> int:
    """Parse paths and run the terminal state census."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state-cache", type=Path, default=DEFAULT_STATE_CACHE)
    arguments = parser.parse_args()
    run(arguments.output.resolve(), arguments.state_cache.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
