"""Paired old-tree/current-tree probe of the MAST corroboration operand solve.

The MAST corroboration bank was regenerated at the bank producer revision, and
re-running the same producer at a later revision moves the terminal residual on
all twelve arms.  This probe re-runs the declared solve in two revisions of the
tree and, per arm, digests each intermediate the solve produces, in order:

  profile_support      the geometry and source the member operator is built on
                       (wall coordinate, grid, sample, inside-material mask,
                       physical node number, connectivity grid axes, source)
  partition_structure  the operator's pytree structure -- whether it is a pytree
                       at all, its leaf count and its tree representation
  partition_values     the concatenated numeric dynamic leaves of that partition
  map_image            the first poloidal flux grid the terminal state yields
  terminal_residual    the fixed-point residual the solve reports

The first stage whose digest differs between the two trees names the read or
partition that moved.  The accepted operand cache is read-only here: this probe
never writes it.

The driver runs one tree per process so jax and nova module state stay isolated,
and it only compares the two emissions in a later invocation:

    <interpreter> bank_drift_paired_probe.py emit \\
        --tree-root <tree> --label <label> --out <label>.json \\
        --identity <shot>/<slice> --arm {pure,mixed}
    <interpreter> bank_drift_paired_probe.py compare \\
        --left <label>.json --right <label>.json --out receipt.json
    <interpreter> bank_drift_paired_probe.py merge \\
        --arms-dir <dir> --out receipt.json

One arm of one identity is the unit of work: it is the smallest slice that
carries a complete pair of emissions, so an allocation that dies loses one arm
rather than a whole campaign, and the arms that did land are already receipts.
An emission is therefore restricted to a single identity and arm, and `merge`
joins the per-arm receipts once they all exist.

The compilation cache is passed in rather than inferred, so every job of the
campaign writes the one directory the campaign names: the same traced programs
in the same tree then compile once for the whole set of jobs rather than once
per job.  The cache receipt the producer returns is stored in each emission, so
the directory a given row was compiled against is evidence rather than an
assumption.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRODUCER_RELATIVE = (
    "docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py"
)
GEOMETRY_STAGES = ("profile_support", "partition_structure", "partition_values")

_STAGE_DEFINITIONS = {
    "first_differing_stage": (
        "the earliest stage in stage_order whose digest differs between the two "
        "trees; a difference at an earlier stage propagates into every later "
        "one, so this names where the trees first diverge, not necessarily the "
        "stage that moved the residual"
    ),
    "residual_moved": (
        "the terminal residual differs between the trees in any bit; "
        "residual_ratio carries the magnitude, so a difference in the last bit "
        "and a difference of twelve decades both read as true here"
    ),
    "residual_ratio": "right-tree terminal residual divided by left-tree residual",
}


def _digest(array: Any) -> str:
    """Return a shape-and-value digest of an array-like operand."""

    values = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    hasher = hashlib.sha256()
    hasher.update(str(values.shape).encode())
    hasher.update(values.tobytes())
    return hasher.hexdigest()[:16]


def _flatten_operand(value: Any) -> list[np.ndarray]:
    """Return every numeric leaf of a nested operand in a deterministic order."""

    leaves: list[np.ndarray] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (bool, int, float)) or isinstance(item, np.generic):
            leaves.append(np.asarray([float(item)]))
            return
        if isinstance(item, (str, bytes)):
            return
        if isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key])
            return
        if isinstance(item, (tuple, list)):
            for element in item:
                visit(element)
            return
        try:
            array = np.asarray(item, dtype=np.float64)
        except (TypeError, ValueError):
            return
        leaves.append(array.reshape(-1) if array.ndim else array.reshape(1))

    visit(value)
    return leaves


def _profile_support_digest(profile: Any) -> dict[str, Any]:
    """Digest the geometry and source the member operator is built on."""

    operator = profile.operator
    parts: dict[str, str] = {}
    for name in ("wall", "grid", "sample", "inside_material"):
        leaves = _flatten_operand(getattr(operator, name, None))
        if leaves:
            parts[name] = _digest(np.concatenate(leaves))
    node_number = getattr(operator, "physical_node_number", None)
    if node_number is not None:
        parts["physical_node_number"] = _digest([float(node_number)])
    connectivity = getattr(operator, "connectivity_grid_axes", None)
    if callable(connectivity):
        try:
            axes, _shape = connectivity()
        except (TypeError, ValueError, RuntimeError):
            axes = None
        if axes is not None:
            leaves = _flatten_operand(axes)
            if leaves:
                parts["connectivity_grid_axes"] = _digest(np.concatenate(leaves))
    source = getattr(profile, "source", None)
    source_leaves = _flatten_operand(source) if source is not None else []
    if source_leaves:
        parts["source"] = _digest(np.concatenate(source_leaves))
    hasher = hashlib.sha256()
    for name in sorted(parts):
        hasher.update(name.encode())
        hasher.update(parts[name].encode())
    return {"digest": hasher.hexdigest()[:16], "parts": parts}


def _partition_report(operator: Any) -> dict[str, Any]:
    """Report the operator's pytree partition as structure and leaf values."""

    import jax

    leaves, treedef = jax.tree_util.tree_flatten(operator)
    numeric_leaves = [leaf for leaf in leaves if leaf is not operator]
    opaque = not numeric_leaves and len(leaves) == 1
    values = _flatten_operand(numeric_leaves)
    values_digest = _digest(np.concatenate(values)) if values else "empty"
    structure = {
        "is_pytree": not opaque,
        "leaf_count": len(leaves),
        "tree_node_count": int(getattr(treedef, "num_nodes", -1)),
        "leaf_dtypes": sorted(
            {str(np.asarray(leaf).dtype) for leaf in numeric_leaves}
        ),
        "pytree_repr": str(treedef)[:400],
        "operator_type": f"{type(operator).__module__}.{type(operator).__name__}",
    }
    return {
        "structure_digest": hashlib.sha256(
            json.dumps(structure, sort_keys=True).encode()
        ).hexdigest()[:16],
        "values_digest": values_digest,
        "summary": structure,
    }


def _identity_of(selected_row: Any) -> str:
    """Return the shot/slice identity a bank selection resolves to."""

    return f"{int(selected_row['shot'])}/{int(selected_row['slice_index'])}"


def _load_module(path: Path, name: str) -> Any:
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arm_stages(reachability: Any, profile: Any, state: Any) -> dict[str, Any]:
    """Digest the post-solve stage of one arm from its terminal state."""

    try:
        geometry = reachability._grid_geometry(profile, state)
    except Exception as error:  # noqa: BLE001 - the failure is the measurement
        return {
            "map_image": None,
            "stage_exception": f"{type(error).__name__}: {error}",
        }
    return {
        "map_image": {
            "flux": _digest(geometry["flux"]),
            "axis": _digest(geometry["axis"]),
            "class_margin": _digest([float(geometry["class_margin"])]),
        },
        "stage_exception": None,
    }


def _resume_rows(out_path: Path) -> list[dict[str, Any]]:
    """Return the rows already checkpointed for this tree, if any."""

    if not out_path.exists():
        return []
    try:
        payload = json.loads(out_path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    rows = payload.get("rows")
    return rows if isinstance(rows, (list,)) else []


def _emit(
    tree_label: str,
    tree_root: Path,
    out_path: Path,
    limit: int,
    identity: str | None,
    arm: str | None,
    compile_cache_root: Path | None,
) -> int:
    """Run the declared solve for one identity and arm and persist its digests."""

    import jax
    import jax.numpy as jnp

    producer = _load_module(tree_root / PRODUCER_RELATIVE, "probe_producer")
    reachability = producer._reachability_module()
    producer.configure_dtypes()
    cache_root = (
        compile_cache_root or producer.default_persistent_compilation_cache_root()
    )
    compile_cache = producer.configure_persistent_compilation_cache(cache_root)

    x64 = bool(jax.config.jax_enable_x64)
    response_cache, carrier_evidence = producer._persisted_response_cache(
        producer.response_carrier.DEFAULT_CARRIER,
        producer.response_carrier.DEFAULT_RECEIPT,
    )
    carrier_identity = producer._carrier_semantic_identity(carrier_evidence)
    selected = list(producer.select_slices_by_shot(producer.DECOMPOSITION_BANK))
    if identity is not None:
        selected = [
            (row, qualification)
            for row, qualification in selected
            if _identity_of(row) == identity
        ]
        if not selected:
            raise SystemExit(f"identity {identity} is not in the decomposition bank")
    elif limit > 0:
        selected = selected[:limit]
    rows = _resume_rows(out_path)
    done = {row["identity"] for row in rows}
    if done:
        print(f"resuming {out_path}: {sorted(done)} already emitted", flush=True)
    rows: list[dict[str, Any]] = []
    for selected_row, qualification in selected:
        shot = int(selected_row["shot"])
        slice_index = int(selected_row["slice_index"])
        arm_identity = _identity_of(selected_row)
        entry: dict[str, Any] = {
            "identity": arm_identity,
            "stages": {},
            "arms": {},
            "exception": None,
        }
        print(
            f"probe {tree_label} {arm_identity} arm={arm or 'both'} "
            f"cache={compile_cache.directory}",
            flush=True,
        )
        try:
            case, context = producer._mast_case_from_selection(
                producer.SHOT_STORE, selected_row, qualification
            )
            passive_case, profile, _policy = producer._passive_inclusive_case(
                case, context, response_cache
            )
            entry["stages"]["profile_support"] = _profile_support_digest(profile)
            partition = _partition_report(profile.operator)
            entry["stages"]["partition_structure"] = {
                "digest": partition["structure_digest"],
                "summary": partition["summary"],
            }
            entry["stages"]["partition_values"] = {
                "digest": partition["values_digest"]
            }
            observed = producer._ObservedProfile(profile)
            target_current = abs(
                float(passive_case["reference"]["plasma_current_a"])
            )
            states = reachability._mast_states(
                observed,
                jnp.asarray(passive_case["state"]),
                target_current,
                carrier_identity=f"mast:{shot}:{slice_index}:{carrier_identity}",
            )
            for arm_name, result in states.items():
                if arm is not None and str(arm_name) != arm:
                    continue
                entry["arms"][str(arm_name)] = {
                    "converged": bool(result.converged),
                    "terminal_residual": float(result.terminal_residual),
                    "termination_reason": str(result.termination_reason),
                    **_arm_stages(reachability, profile, result.state),
                }
        except Exception as error:  # noqa: BLE001 - the failure is the finding
            entry["exception"] = f"{type(error).__name__}: {error}"
        print(
            f"probe {tree_label} {arm_identity} done exception={entry['exception']} "
            f"residuals="
            + json.dumps(
                {
                    name: record.get("terminal_residual")
                    for name, record in entry["arms"].items()
                }
            ),
            flush=True,
        )
        rows.append(entry)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "tree_label": tree_label,
                    "tree_root": str(tree_root),
                    "emitted_at": datetime.now(UTC).isoformat(),
                    "jax_enable_x64": x64,
                    "carrier_identity": carrier_identity,
                    "compile_cache": compile_cache.receipt(),
                    "requested_identity": identity,
                    "requested_arm": arm,
                    "rows": rows,
                },
                indent=1,
                sort_keys=True,
            )
        )
    print(f"wrote {out_path}", flush=True)
    return 0




def _flat_stages(row: dict[str, Any] | None) -> dict[str, Any]:
    """Return the identity-level stage digests of one emission row.

    The map image and the terminal residual are properties of an arm, not of the
    identity, so they are read from the arm record being compared rather than
    from here.
    """

    if row is None:
        return {}
    stages = row.get("stages") or {}
    return {
        "profile_support": (stages.get("profile_support") or {}).get("digest"),
        "partition_structure": (stages.get("partition_structure") or {}).get("digest"),
        "partition_values": (stages.get("partition_values") or {}).get("digest"),
        "exception": row.get("exception"),
    }


def _compare(left_path: Path, right_path: Path, out_path: Path) -> int:
    """Join the two tree emissions into one twelve-row attribution receipt."""

    left = json.loads(left_path.read_text())
    right = json.loads(right_path.read_text())
    left_rows = {row["identity"]: row for row in left["rows"]}
    right_rows = {row["identity"]: row for row in right["rows"]}
    rows: list[dict[str, Any]] = []
    for identity in sorted(left_rows):
        left_row = left_rows[identity]
        right_row = right_rows.get(identity)
        left_stages = _flat_stages(left_row)
        right_stages = _flat_stages(right_row)
        first_stage = None
        for stage in GEOMETRY_STAGES:
            if left_stages.get(stage) != right_stages.get(stage):
                first_stage = stage
                break
            if left_stages.get(stage) is None:
                first_stage = "unavailable"
                break
        for arm in ("pure", "mixed"):
            left_arm = (left_row.get("arms") or {}).get(arm)
            right_arm = (right_row.get("arms") or {}).get(arm) if right_row else None
            residual_left = None if not left_arm else left_arm.get("terminal_residual")
            residual_right = (
                None if not right_arm else right_arm.get("terminal_residual")
            )
            residual_delta = (
                None
                if residual_left is None or residual_right is None
                else abs(residual_right - residual_left)
            )
            arm_first = first_stage
            if arm_first is None and left_arm and right_arm:
                left_map = (left_arm.get("map_image") or {}).get("flux")
                right_map = (right_arm.get("map_image") or {}).get("flux")
                if left_map != right_map:
                    arm_first = "map_image"
                elif residual_delta not in (None, 0.0):
                    arm_first = "terminal_residual"
            rows.append(
                {
                    "identity": identity,
                    "arm": arm,
                    "residual_left": residual_left,
                    "residual_right": residual_right,
                    "residual_delta": residual_delta,
                    "residual_moved": residual_delta not in (None, 0.0),
                    "residual_ratio": (
                        None
                        if residual_left in (None, 0.0) or residual_right is None
                        else residual_right / residual_left
                    ),
                    "converged_left": (
                        None if not left_arm else bool(left_arm.get("converged"))
                    ),
                    "converged_right": (
                        None if not right_arm else bool(right_arm.get("converged"))
                    ),
                    "stages_left": left_stages,
                    "stages_right": right_stages,
                    "first_differing_stage": arm_first,
                    "note": None if right_row else "arm absent in the right tree",
                }
            )
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["first_differing_stage"])
        counts[key] = counts.get(key, 0) + 1
    moved = [
        {
            "identity": row["identity"],
            "arm": row["arm"],
            "residual_left": row["residual_left"],
            "residual_moved": row["residual_moved"],
            "residual_ratio": row["residual_ratio"],
            "residual_right": row["residual_right"],
            "first_differing_stage": row["first_differing_stage"],
            "converged_left": row["converged_left"],
            "converged_right": row["converged_right"],
        }
        for row in rows
        if row["residual_moved"]
    ]
    focus = [
        row
        for row in rows
        if row["identity"] == "21983/35" and row["arm"] == "mixed"
    ]
    receipt = {
        "artifact": "paired old-tree against current-tree MAST operand-solve probe",
        "left": {
            "label": left["tree_label"],
            "root": left["tree_root"],
            "compile_cache": (left.get("compile_cache") or {}).get("directory"),
        },
        "right": {
            "label": right["tree_label"],
            "root": right["tree_root"],
            "compile_cache": (right.get("compile_cache") or {}).get("directory"),
        },
        "stage_order": list(GEOMETRY_STAGES) + ["map_image", "terminal_residual"],
        "definitions": dict(_STAGE_DEFINITIONS),
        "rows": rows,
        "summary": {
            "first_differing_stage_counts": counts,
            "residual_moved": moved,
            "focus_21983_35_mixed": focus,
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(f"wrote {out_path}", flush=True)
    print(json.dumps(receipt["summary"], indent=1, sort_keys=True), flush=True)
    return 0


def _merge(arms_dir: Path, out_path: Path, expected_rows: int) -> int:
    """Join the per-arm receipts into one campaign receipt.

    One arm per allocation means the arms land independently, so the campaign
    receipt is assembled from whatever arm receipts exist.  A missing arm is
    reported rather than silently absent: the campaign is only a twelve-arm
    attribution if all twelve are present.
    """

    receipts = sorted(arms_dir.glob("receipt-*.json"))
    if not receipts:
        raise SystemExit(f"no per-arm receipts under {arms_dir}")
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    trees: dict[str, dict[str, Any]] = {}
    for path in receipts:
        payload = json.loads(path.read_text())
        sources.append(
            {
                "path": str(path),
                "rows": len(payload.get("rows") or []),
                "generated_at": payload.get("generated_at"),
            }
        )
        for side in ("left", "right"):
            if payload.get(side):
                trees[side] = payload[side]
        rows.extend(payload.get("rows") or [])
    rows.sort(key=lambda row: (row["identity"], row["arm"]))
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row["first_differing_stage"])
        counts[key] = counts.get(key, 0) + 1
    focus = [
        row
        for row in rows
        if row["identity"] == "21983/35" and row["arm"] == "mixed"
    ]
    moved = [
        {
            "identity": row["identity"],
            "arm": row["arm"],
            "residual_left": row["residual_left"],
            "residual_moved": row["residual_moved"],
            "residual_ratio": row["residual_ratio"],
            "residual_right": row["residual_right"],
            "first_differing_stage": row["first_differing_stage"],
            "converged_left": row["converged_left"],
            "converged_right": row["converged_right"],
        }
        for row in rows
        if row["residual_moved"]
    ]
    missing = expected_rows - len(rows)
    receipt = {
        "artifact": "paired old-tree against current-tree MAST operand-solve probe",
        "left": trees.get("left"),
        "right": trees.get("right"),
        "stage_order": list(GEOMETRY_STAGES) + ["map_image", "terminal_residual"],
        "definitions": dict(_STAGE_DEFINITIONS),
        "rows": rows,
        "summary": {
            "rows_expected": expected_rows,
            "rows_present": len(rows),
            "arms_missing": missing,
            "first_differing_stage_counts": counts,
            "residual_moved": moved,
            "focus_21983_35_mixed": focus,
        },
        "source_receipts": sources,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(receipt, indent=1, sort_keys=True))
    print(f"wrote {out_path}", flush=True)
    print(json.dumps(receipt["summary"], indent=1, sort_keys=True), flush=True)
    return 0 if missing <= 0 else 1


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument("--tree-root", type=Path, required=True)
    emit.add_argument("--label", required=True)
    emit.add_argument("--out", type=Path, required=True)
    emit.add_argument("--limit", type=int, default=0)
    emit.add_argument(
        "--identity",
        default=None,
        help="restrict the emission to one shot/slice identity, e.g. 21978/35",
    )
    emit.add_argument(
        "--arm",
        default=None,
        choices=("pure", "mixed"),
        help="restrict the emission to one solve arm",
    )
    emit.add_argument(
        "--compile-cache-root",
        type=Path,
        default=None,
        help="root of the shared persistent compilation cache for this campaign",
    )
    compare = sub.add_parser("compare")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--schema", type=Path, required=False)
    compare.add_argument("--out", type=Path, required=True)
    merge = sub.add_parser("merge")
    merge.add_argument("--arms-dir", type=Path, required=True)
    merge.add_argument("--out", type=Path, required=True)
    merge.add_argument("--expected-rows", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    if args.command == "emit":
        return _emit(
            args.label,
            args.tree_root,
            args.out,
            args.limit,
            args.identity,
            args.arm,
            args.compile_cache_root,
        )
    if args.command == "merge":
        return _merge(args.arms_dir, args.out, args.expected_rows)
    return _compare(args.left, args.right, args.out)


if __name__ == "__main__":
    raise SystemExit(main())