"""Bisect the bank drift on the traced-anchor hook.

The forward operator carries three specialised member scalars -- the declared
axis flux, the declared boundary flux and the declared support mask.  A hook on
the operator decides whether they travel as traced pytree children, as member
data the compiler sees, or stay behind in the static specialisation aux as host
constants the compiler folds away.  When the hook names them the operator is
also built with them excluded from its geometry identity; when it does not, the
identity carries them.

This driver solves one banked identity under both treatments, once per solve
class, in a single allocation.  The traced treatment is the tree as it ships.
The constant treatment patches the hook empty from the driver for the duration
of one operand construction and its solve, then restores it, so nothing under
``nova/`` is edited.  The patch has to span the construction rather than only
the solve: the operator's geometry identity is computed in ``__post_init__``, so
a hook overridden afterwards would leave the identity naming the scalars as
member data while the flatten no longer carried them, and the arm would be a
mixture of the two treatments rather than either.

Each arm persists as it lands: terminal residual, termination reason, converged
flag, the active-set trip trace, the partition structure and value digests, the
hook's own answer to ``_dynamic_extra_names()``, an anchor snapshot, and the
terminal grid geometry the contour panel is drawn from.  An arm that raises is
recorded with its traceback rather than aborting the allocation, because the
exception is a measurement.

Subcommands:
  run       solve every (treatment, class) pair of one identity, persist each as
            it lands, then render the terminal-state contour pair
  receipt   join the engine emission and the committed bank reference into one
            side-by-side record
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRODUCER_RELATIVE = (
    "docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py"
)
DEFAULT_IDENTITY = "21978/35"
ANCHOR_NAMES = ("declared_axis_flux", "declared_boundary_flux", "declared_support")
TREATMENTS = ("traced", "constant")
SOLVE_CLASSES = ("pure", "mixed")
BANK_REFERENCE_RELATIVE = "docs/figures/null-identification-authority/bank-drift-probe"
TRACE_FIELDS = (
    "active_set_iterations",
    "active_set_residuals",
    "active_set_mask_differences",
    "active_set_cycle_damping_activations",
)
TRACE_LIMIT = 64
ROUND_OFF_FLOOR = float(np.finfo(np.float64).eps)
GEOMETRY_KEYS = (
    "radius",
    "height",
    "flux",
    "wall",
    "axis",
    "selected_x",
    "class_margin",
    "typed_saddle_coordinates_m",
    "typed_saddles_inside_wall",
)


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
        if isinstance(item, bool | int | float) or isinstance(item, np.generic):
            leaves.append(np.asarray([float(item)]))
            return
        if isinstance(item, str | bytes):
            return
        if isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key])
            return
        if isinstance(item, tuple | list):
            for element in item:
                visit(element)
            return
        try:
            array = np.asarray(item, dtype=np.float64)
        except TypeError, ValueError:
            return
        leaves.append(array.reshape(-1) if array.ndim else array.reshape(1))

    visit(value)
    return leaves


def _partition_report(operator: Any) -> dict[str, Any]:
    """Report the operator's pytree partition as structure and leaf values.

    The structure digest is the instrument this bisect rests on: it is computed
    from the treedef the flatten produces, so the constant treatment is proved
    to have removed the anchors from the traced partition only by its digest
    moving.  A treatment that silently changed nothing would leave this digest
    equal between arms.
    """

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
        "leaf_dtypes": sorted({str(np.asarray(leaf).dtype) for leaf in numeric_leaves}),
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


def _load_module(path: Path, name: str) -> Any:
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity_of(selected_row: Any) -> str:
    """Return the shot/slice identity a bank selection resolves to."""

    return f"{int(selected_row['shot'])}/{int(selected_row['slice_index'])}"


def _slug(identity: str) -> str:
    """Return the filename form of a shot/slice identity."""

    return identity.replace("/", "-")


@contextmanager
def _anchor_treatment(treatment: str):
    """Hold the declared anchors in the requested partition for one arm.

    ``traced`` yields the tree's own hook.  ``constant`` replaces it with an
    empty answer for the duration of the caller's block, which is what keeps
    the scalars in the static specialisation aux instead of the traced
    ``children``, and restores the original in a ``finally`` so a raise inside
    the block cannot leak the override into the next arm.
    """

    if treatment not in TREATMENTS:
        raise ValueError(f"unknown anchor treatment {treatment!r}")
    if treatment == "traced":
        yield
        return
    from nova.equilibrium.forward_operator import ForwardFluxOperator

    original = ForwardFluxOperator._dynamic_extra_names
    ForwardFluxOperator._dynamic_extra_names = lambda self: ()
    try:
        yield
    finally:
        ForwardFluxOperator._dynamic_extra_names = original


def _hook_names(operator: Any) -> list[str]:
    """Return the hook's own answer for this operator."""

    hook = getattr(operator, "_dynamic_extra_names", None)
    if not callable(hook):
        return []
    try:
        return [str(name) for name in hook()]
    except Exception:  # noqa: BLE001 - the failure is recorded as the answer
        return []


def _anchor_snapshot(operator: Any) -> dict[str, Any]:
    """Snapshot the three declared anchors as the operator carries them."""

    snapshot: dict[str, Any] = {}
    for name in ANCHOR_NAMES:
        value = getattr(operator, name, None)
        if value is None:
            snapshot[name] = {"present": False}
            continue
        array = np.asarray(value)
        record: dict[str, Any] = {
            "present": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "where": "traced" if name in _hook_names(operator) else "specialisation",
        }
        if array.dtype == bool:
            record["true_count"] = int(array.sum())
        elif array.size <= 8:
            record["values"] = [float(item) for item in array.reshape(-1)]
        else:
            record["digest"] = _digest(array)
        snapshot[name] = record
    return snapshot


def _trip_trace(fixed_point: Any) -> dict[str, Any]:
    """Read the active-set trip trace off a solved fixed-point result.

    The trace is what tells the two arms apart beyond the terminal residual: a
    residual that moves while the trip trace is identical means the hook changed
    the arithmetic, and a trip trace that moves means the hook changed an
    active-set decision, which is the question this bisect asks.
    """

    trace: dict[str, Any] = {}
    for name in TRACE_FIELDS:
        value = getattr(fixed_point, name, None)
        if value is None:
            trace[name] = {"present": False}
            continue
        try:
            array = np.asarray(value)
        except Exception as error:  # noqa: BLE001 - the failure is the reading
            trace[name] = {
                "present": True,
                "read_exception": f"{type(error).__name__}: {error}",
            }
            continue
        flat = array.reshape(-1)
        record: dict[str, Any] = {
            "present": True,
            "shape": list(array.shape),
            "size": int(flat.size),
        }
        if flat.size and np.issubdtype(flat.dtype, np.floating):
            finite = flat[np.isfinite(flat)]
            record["last_finite"] = float(finite[-1]) if finite.size else None
            record["values"] = [float(item) for item in flat[:TRACE_LIMIT]]
        elif flat.size:
            record["values"] = [int(item) for item in flat[:TRACE_LIMIT]]
        record["truncated"] = bool(flat.size > TRACE_LIMIT)
        trace[name] = record
    return trace


def _termination_reason(receipt: Any, reachability: Any) -> str:
    """Return the receipt's termination reason as its enum name."""

    try:
        reason_type = reachability.FixedPointTerminationReason
    except AttributeError:
        from nova.equilibrium.fixed_point import FixedPointTerminationReason

        reason_type = FixedPointTerminationReason
    value = int(np.asarray(receipt.termination_reason))
    try:
        return reason_type(value).name.lower()
    except ValueError:
        return f"unknown_{value}"


def _policy_overrides(reachability: Any, solve_class: str) -> dict[str, object] | None:
    """Return the policy overrides the requested solve class resolves to.

    The pure class pins the fixed-point policy to the values the bank reference
    was produced under; the mixed class leaves the request's own defaults in
    place.  Both are read off the producer's reachability module rather than
    restated here, so a change there reaches this driver.
    """

    if solve_class == "mixed":
        return None
    if solve_class != "pure":
        raise ValueError(f"unknown solve class {solve_class!r}")
    return {
        "newton_steps": reachability.NEWTON_STEPS,
        "gmres_iterations": reachability.GMRES_ITERATIONS,
        "warmup": reachability.WARMUP_SWEEPS,
        "relaxation": reachability.RELAXATION,
        "step_cap": reachability.STEP_CAP,
        "kernel_tolerance": reachability.FIXED_POINT_CRITERION,
        "qualification_tolerance": reachability.FIXED_POINT_CRITERION,
    }


def _geometry_arrays(reachability: Any, profile: Any, state: Any) -> dict[str, Any]:
    """Return the terminal grid geometry the contour panel is drawn from."""

    geometry = reachability._grid_geometry(profile, state)
    arrays: dict[str, Any] = {}
    for key in GEOMETRY_KEYS:
        value = geometry.get(key)
        if value is None:
            continue
        try:
            arrays[key] = np.asarray(value, dtype=float)
        except TypeError, ValueError:
            continue
    return arrays


def _arm_paths(
    out_dir: Path, identity: str, solve_class: str, treatment: str
) -> tuple[Path, Path]:
    """Return the JSON and array paths of one arm."""

    stem = f"arm-{_slug(identity)}-{solve_class}-{treatment}"
    return out_dir / f"{stem}.json", out_dir / f"{stem}.npz"


def _persist_arm(
    payload: dict[str, Any],
    reachability: Any,
    profile: Any,
    receipt: Any,
    json_path: Path,
    array_path: Path,
) -> None:
    """Write one arm's record and terminal geometry as soon as it lands."""

    try:
        arrays = _geometry_arrays(reachability, profile, receipt.equilibrium.flux)
    except Exception as error:  # noqa: BLE001 - the failure is the measurement
        payload["geometry_exception"] = f"{type(error).__name__}: {error}"
        arrays = {}
    if arrays:
        np.savez(array_path, **arrays)
        payload["geometry_path"] = array_path.name
        payload["geometry_digests"] = {
            key: _digest(value) for key, value in sorted(arrays.items())
        }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {json_path}", flush=True)


def _build_operands(
    producer: Any, selected_row: Any, qualification: Any, response_cache: Any
) -> tuple[Any, Any]:
    """Build the banked identity's passive-inclusive case and member operator."""

    case, context = producer._mast_case_from_selection(
        producer.SHOT_STORE, selected_row, qualification
    )
    passive_case, profile, _policy = producer._passive_inclusive_case(
        case, context, response_cache
    )
    return passive_case, profile


def _run(
    identity: str,
    treatments: tuple[str, ...],
    solve_classes: tuple[str, ...],
    out_dir: Path,
    compile_cache_root: Path | None,
) -> int:
    """Solve the identity under every requested treatment and solve class."""

    import jax
    import jax.numpy as jnp

    tree_root = ROOT
    producer = _load_module(tree_root / PRODUCER_RELATIVE, "anchor_bisect_producer")
    reachability = producer._reachability_module()
    producer.configure_dtypes()
    cache_root = (
        compile_cache_root or producer.default_persistent_compilation_cache_root()
    )
    compile_cache = producer.configure_persistent_compilation_cache(cache_root)
    x64 = bool(jax.config.jax_enable_x64)
    if not x64:
        print(
            "WARN x64 is not enabled; the solve would run in single precision",
            flush=True,
        )

    response_cache, carrier_evidence = producer._persisted_response_cache(
        producer.response_carrier.DEFAULT_CARRIER,
        producer.response_carrier.DEFAULT_RECEIPT,
    )
    carrier_identity = producer._carrier_semantic_identity(carrier_evidence)
    selected = [
        (row, qualification)
        for row, qualification in producer.select_slices_by_shot(
            producer.DECOMPOSITION_BANK
        )
        if _identity_of(row) == identity
    ]
    if not selected:
        raise SystemExit(f"identity {identity} is not in the decomposition bank")
    selected_row, qualification = selected[0]
    shot = int(selected_row["shot"])
    slice_index = int(selected_row["slice_index"])

    out_dir.mkdir(parents=True, exist_ok=True)
    arms: dict[str, Any] = {}
    for treatment in treatments:
        print(f"--- treatment {treatment} ---", flush=True)
        with _anchor_treatment(treatment):
            passive_case, profile = _build_operands(
                producer, selected_row, qualification, response_cache
            )
            operator = profile.operator
            hook_names = _hook_names(operator)
            anchors = _anchor_snapshot(operator)
            partition = _partition_report(operator)
            support = _profile_support_digest(profile)
            print(
                f"HOOK treatment={treatment} names={json.dumps(hook_names)} "
                f"leaves={partition['summary']['leaf_count']} "
                f"structure={partition['structure_digest']} "
                f"values={partition['values_digest']}",
                flush=True,
            )
            observed = producer._ObservedProfile(profile)
            seed = jnp.asarray(passive_case["state"])
            target_current = abs(float(passive_case["reference"]["plasma_current_a"]))
            for solve_class in solve_classes:
                key = f"{solve_class}-{treatment}"
                json_path, array_path = _arm_paths(
                    out_dir, identity, solve_class, treatment
                )
                arm_carrier = (
                    f"mast:{shot}:{slice_index}:{carrier_identity}"
                    if treatment == "traced"
                    else f"mast:{shot}:{slice_index}:{carrier_identity}:{treatment}"
                )
                payload: dict[str, Any] = {
                    "identity": identity,
                    "solve_class": solve_class,
                    "treatment": treatment,
                    "hook_names": hook_names,
                    "anchors": anchors,
                    "profile_support": support,
                    "partition": {
                        "structure_digest": partition["structure_digest"],
                        "values_digest": partition["values_digest"],
                        "summary": partition["summary"],
                    },
                    "carrier_identity": arm_carrier,
                    "target_current": target_current,
                    "exception": None,
                }
                try:
                    receipt = reachability._solve_with_defaults(
                        observed,
                        seed,
                        carrier_identity=arm_carrier,
                        target_current=target_current,
                        policy_overrides=_policy_overrides(reachability, solve_class),
                    )
                    equilibrium = receipt.equilibrium
                    fixed_point = equilibrium.fixed_point
                    payload.update(
                        {
                            "terminal_residual": float(
                                np.asarray(fixed_point.residual)
                            ),
                            "converged": bool(np.asarray(receipt.qualified)),
                            "termination_reason": _termination_reason(
                                receipt, reachability
                            ),
                            "trip_trace": _trip_trace(fixed_point),
                            "resolved_defaults": (
                                receipt.resolved_defaults.to_dict()
                                if hasattr(receipt, "resolved_defaults")
                                else None
                            ),
                            "flux_digest": _digest(equilibrium.flux),
                        }
                    )
                    print(
                        f"ARM identity={identity} class={solve_class} "
                        f"treatment={treatment} "
                        f"residual={payload['terminal_residual']:.12e} "
                        f"converged={payload['converged']} "
                        f"reason={payload['termination_reason']} "
                        f"iterations="
                        f"{np.asarray(fixed_point.active_set_iterations).reshape(-1).tolist()}",
                        flush=True,
                    )
                    _persist_arm(
                        payload, reachability, profile, receipt, json_path, array_path
                    )
                except Exception as error:  # noqa: BLE001 - the failure is the finding
                    payload["exception"] = (
                        f"{type(error).__name__}: {error}\n"
                        + "".join(traceback.format_exc(limit=12))
                    )
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    json_path.write_text(json.dumps(payload, indent=1, sort_keys=True))
                    print(
                        f"ARM identity={identity} class={solve_class} "
                        f"treatment={treatment} EXCEPTION "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                arms[key] = payload

    emission = {
        "identity": identity,
        "tree_root": str(tree_root),
        "emitted_at": datetime.now(UTC).isoformat(),
        "jax_enable_x64": x64,
        "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        "carrier_identity": carrier_identity,
        "compile_cache": compile_cache.receipt(),
        "treatments": list(treatments),
        "solve_classes": list(solve_classes),
        "arms": arms,
    }
    (out_dir / "emission.json").write_text(
        json.dumps(emission, indent=1, sort_keys=True)
    )
    print(f"wrote {out_dir / 'emission.json'}", flush=True)
    return 0


def _profile_support_digest(profile: Any) -> dict[str, Any]:
    """Digest the geometry and source the member operator is built on.

    The anchors are deliberately absent from this digest, so it is an
    invariance check rather than a treatment indicator: two treatments that
    disagree here are not measuring the same operands.

    The parts are the same set the bank probe digests, which is what makes the
    value comparable to the committed reference rather than only to this run:
    an equal digest is evidence that this driver built the same operands the
    banked run did, and the two solve classes agree here is evidence that a
    residual difference is not an operand difference.
    """

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
        except TypeError, ValueError, RuntimeError:
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


def _trace_digest(trace: dict[str, Any]) -> str:
    """Return one digest over the four trip-trace fields."""

    parts = {}
    for name in TRACE_FIELDS:
        record = trace.get(name)
        if not isinstance(record, dict) or record.get("present") is False:
            parts[name] = "absent"
            continue
        values = record.get("values")
        parts[name] = (
            _digest(values) if values is not None else str(record.get("shape"))
        )
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def _load_arm(
    out_dir: Path, identity: str, solve_class: str, treatment: str
) -> dict[str, Any]:
    json_path, _array = _arm_paths(out_dir, identity, solve_class, treatment)
    if not json_path.exists():
        return {
            "exception": f"missing arm file {json_path.name}",
            "treatment": treatment,
        }
    return json.loads(json_path.read_text())


def _close(left: float | None, right: float | None) -> bool | None:
    """Return whether two floats agree bit for bit, or None if either is absent."""

    if left is None or right is None:
        return None
    return float(left) == float(right)


def _arm_absence(record: dict[str, Any]) -> str | None:
    """Return why an arm carries no terminal state, or None when it does.

    An arm file that never landed is loaded as an exception stub and an arm
    whose solve raised is persisted with its traceback; neither carries a
    terminal residual, so neither supports a comparison of the treatments.
    """

    if record.get("terminal_residual") is not None:
        return None
    exception = record.get("exception")
    if exception:
        return str(exception).splitlines()[0]
    return "no terminal residual recorded"


def _residual_gap_below_round_off(
    left: float | None, right: float | None
) -> bool | None:
    """Whether two terminal residuals differ by less than the round-off floor.

    The terminal residual is the fixed point's normalised residual, so a solve
    that reaches the floor reports order one machine epsilon and no smaller.
    Two such values differ by the order their arithmetic ran in, not by a
    different terminal state, and reading that difference as a carrier would be
    reporting the summation order as physics.
    """

    if left is None or right is None:
        return None
    return abs(float(left) - float(right)) <= ROUND_OFF_FLOOR


def _receipt(out_dir: Path, identity: str, bank_dir: Path, out_path: Path) -> int:
    """Join this run's arms and the committed bank reference into one record."""

    slug = _slug(identity)
    reference: dict[str, Any] = {}
    for solve_class in SOLVE_CLASSES:
        reference_path = bank_dir / "arms" / f"receipt-{slug}-{solve_class}.json"
        if reference_path.exists():
            payload = json.loads(reference_path.read_text())
            reference[solve_class] = {
                "path": str(reference_path),
                "rows": payload.get("rows", []),
                "left_label": payload.get("left", {}).get("label"),
                "right_label": payload.get("right", {}).get("label"),
            }

    classes: dict[str, Any] = {}
    for solve_class in SOLVE_CLASSES:
        arms = {
            treatment: _load_arm(out_dir, identity, solve_class, treatment)
            for treatment in TREATMENTS
        }
        traced, constant = arms["traced"], arms["constant"]
        bank = None
        rows = (reference.get(solve_class) or {}).get("rows") or []
        if rows:
            bank = rows[0]
        verdict: dict[str, Any] = {}
        verdict["arms_absent"] = {
            treatment: why
            for treatment, arm in arms.items()
            if (why := _arm_absence(arm))
        }
        traced_trace = traced.get("trip_trace") or {}
        constant_trace = constant.get("trip_trace") or {}
        verdict["hook_names_differ"] = list(traced.get("hook_names") or []) != list(
            constant.get("hook_names") or []
        )
        verdict["partition_structure_differ"] = (traced.get("partition") or {}).get(
            "structure_digest"
        ) != (constant.get("partition") or {}).get("structure_digest")
        verdict["partition_leaf_delta"] = (
            ((traced.get("partition") or {}).get("summary") or {}).get("leaf_count")
            - ((constant.get("partition") or {}).get("summary") or {}).get("leaf_count")
            if traced.get("partition") and constant.get("partition")
            else None
        )
        verdict["profile_support_invariant"] = (
            traced.get("profile_support") or {}
        ).get("digest") == (constant.get("profile_support") or {}).get("digest")
        verdict["terminal_residual_differ"] = traced.get(
            "terminal_residual"
        ) != constant.get("terminal_residual")
        verdict["termination_reason_differ"] = traced.get(
            "termination_reason"
        ) != constant.get("termination_reason")
        verdict["trip_trace_differ"] = _trace_digest(traced_trace) != _trace_digest(
            constant_trace
        )
        verdict["trip_trace_digests"] = {
            "traced": _trace_digest(traced_trace),
            "constant": _trace_digest(constant_trace),
        }
        if bank is not None:
            verdict["traced_matches_bank_current_residual"] = _close(
                traced.get("terminal_residual"), bank.get("residual_right")
            )
            verdict["constant_matches_bank_producer_residual"] = _close(
                constant.get("terminal_residual"), bank.get("residual_left")
            )
            verdict["traced_matches_bank_current_stage"] = (
                traced.get("partition") or {}
            ).get("structure_digest") == (bank.get("stages_right") or {}).get(
                "partition_structure"
            )
            verdict["constant_matches_bank_producer_stage"] = (
                constant.get("partition") or {}
            ).get("structure_digest") == (bank.get("stages_left") or {}).get(
                "partition_structure"
            )
            verdict["bank_current_residual"] = bank.get("residual_right")
            verdict["bank_producer_residual"] = bank.get("residual_left")
        active_set_cycle = any(
            (arm.get("termination_reason") or "") == "active_set_cycle_detected"
            for arm in arms.values()
        )
        verdict["active_set_cycle_present"] = active_set_cycle
        traced_residual = traced.get("terminal_residual")
        constant_residual = constant.get("terminal_residual")
        verdict["round_off_floor"] = ROUND_OFF_FLOOR
        verdict["terminal_residual_gap"] = (
            abs(float(traced_residual) - float(constant_residual))
            if traced_residual is not None and constant_residual is not None
            else None
        )
        verdict["terminal_residual_gap_below_round_off"] = (
            _residual_gap_below_round_off(traced_residual, constant_residual)
        )
        if verdict["arms_absent"]:
            verdict["finding"] = (
                "absent arms: "
                + "; ".join(
                    f"{treatment} carries no terminal state ({why})"
                    for treatment, why in verdict["arms_absent"].items()
                )
                + ", so this record compares nothing about the hook"
            )
        elif (
            verdict["hook_names_differ"]
            and verdict["terminal_residual_differ"]
            and verdict["terminal_residual_gap_below_round_off"]
        ):
            verdict["finding"] = (
                "the anchor partition is not the carrier at this resolution: the "
                "two terminal residuals differ by "
                f"{verdict['terminal_residual_gap']:.3g} against a round-off "
                f"floor of {ROUND_OFF_FLOOR:.3g}, so the two arms reached the same "
                "state by a different arithmetic order"
            )
        elif verdict["hook_names_differ"] and verdict["terminal_residual_differ"]:
            verdict["finding"] = (
                "the anchor partition is the carrier: holding the three anchors "
                "constant changes the terminal state"
            )
        elif verdict["hook_names_differ"] and not verdict["terminal_residual_differ"]:
            verdict["finding"] = (
                "the anchor partition is not the carrier: the constant treatment "
                "changes the flatten but not the terminal state, so the drift lies "
                "downstream of the hook"
            )
        else:
            verdict["finding"] = (
                "the override did not take effect: both treatments report the same "
                "hook names, so this run measures nothing about the hook"
            )
        classes[solve_class] = {
            "traced": traced,
            "constant": constant,
            "bank_reference": bank,
            "verdict": verdict,
        }

    payload = {
        "artifact": (
            "anchor-treatment bisect on one banked identity: the traced-anchor arm "
            "as the tree ships it against a constant-anchor arm the driver holds by "
            "overriding the dynamic-extra hook for one construction and its solve"
        ),
        "identity": identity,
        "generated_at": datetime.now(UTC).isoformat(),
        "engine": "benchmarks/bank_drift_anchor_bisect.py",
        "classes": classes,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=True))
    for solve_class, record in classes.items():
        verdict = record["verdict"]
        print(
            f"VERDICT class={solve_class} "
            f"traced_residual={record['traced'].get('terminal_residual')} "
            f"traced_reason={record['traced'].get('termination_reason')} "
            f"constant_residual={record['constant'].get('terminal_residual')} "
            f"constant_reason={record['constant'].get('termination_reason')} "
            f"hook_moved={verdict['hook_names_differ']} "
            f"arms_absent={json.dumps(verdict['arms_absent'])} "
            f"residual_gap={verdict['terminal_residual_gap']} "
            f"finding={verdict['finding']}",
            flush=True,
        )
    print(f"wrote {out_path}", flush=True)
    return 0


def _hollow_after(axes: Any, count_before: int) -> None:
    """Turn the most recently added lines into hollow markers."""

    for line in axes.lines[count_before:]:
        line.set_markerfacecolor("none")


def _load_arrays(path: Path) -> dict[str, Any]:
    """Read one arm's terminal geometry archive."""

    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _panel(
    identity: str, solve_class: str, out_dir: Path, out_path: Path, levels: int
) -> int:
    """Draw both treatments' terminal states as one contour pair.

    One physical level array -- taken from the traced arm, the arm under test --
    is handed to both panels.  Two maps contoured on independently chosen levels
    can be made to look like anything, and the question here is exactly whether
    the two states differ.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nova.media.ink import DEFAULT_INK, poloidal_axes
    from nova.media.poloidal import (
        contour_levels,
        draw_flux_contours,
        draw_nulls,
        draw_wall,
    )

    panels = []
    for treatment in TREATMENTS:
        json_path, array_path = _arm_paths(out_dir, identity, solve_class, treatment)
        if not json_path.exists() or not array_path.exists():
            print(f"panel arm missing: {json_path}", flush=True)
            return 1
        payload = json.loads(json_path.read_text())
        panels.append(
            {
                "treatment": treatment,
                "record": payload,
                "arrays": _load_arrays(array_path),
            }
        )
    traced, constant = panels
    if "flux" not in traced["arrays"] or "flux" not in constant["arrays"]:
        print("panel skipped: an arm carries no terminal flux", flush=True)
        return 1

    wall = np.asarray(traced["arrays"]["wall"], dtype=float)
    radius = np.asarray(traced["arrays"]["radius"], dtype=float)
    height = np.asarray(traced["arrays"]["height"], dtype=float)
    shared_levels = contour_levels(
        np.asarray(traced["arrays"]["flux"], dtype=float), count=levels
    )

    own = DEFAULT_INK.variant(
        axis_marker="^", axis_markersize=9.0, xpoint_marker="x", xpoint_markersize=9.0
    )
    other = DEFAULT_INK.variant(
        axis_marker="o",
        axis_markersize=8.0,
        xpoint_marker="s",
        xpoint_markersize=8.0,
        axis_color="#666666",
        xpoint_color="#666666",
    )

    r_min, r_max = float(np.nanmin(wall[:, 0])), float(np.nanmax(wall[:, 0]))
    z_min, z_max = float(np.nanmin(wall[:, 1])), float(np.nanmax(wall[:, 1]))
    pad = 0.04 * max(r_max - r_min, z_max - z_min)
    extent = (r_min - pad, r_max + pad, z_min - pad, z_max + pad)

    figure, axes_row = plt.subplots(
        1, 2, figsize=(9.0, 5.0), dpi=DEFAULT_INK.figure_dpi
    )
    for axes, panel in zip(axes_row, panels, strict=True):
        record = panel["record"]
        arrays = panel["arrays"]
        poloidal_axes(axes)
        draw_flux_contours(
            axes, radius, height, np.asarray(arrays["flux"], dtype=float), shared_levels
        )
        draw_wall(axes, radius=wall[:, 0], height=wall[:, 1])
        for state, style, hollow in (
            (panel, own, False),
            (panels[1] if panel is panels[0] else panels[0], other, True),
        ):
            state_arrays = state["arrays"]
            before = len(axes.lines)
            draw_nulls(
                axes,
                magnetic_axis=np.asarray(state_arrays["axis"], dtype=float),
                x_points=np.asarray(state_arrays["selected_x"], dtype=float).reshape(
                    1, 2
                ),
                style=style,
                contain=wall,
            )
            if hollow:
                _hollow_after(axes, before)
        axes.set_xlim(extent[0], extent[1])
        axes.set_ylim(extent[2], extent[3])
        axes.set_autoscale_on(False)
        axes.set_title(
            f"{panel['treatment']}-anchor  converged={bool(record.get('converged'))}  "
            f"residual={float(record.get('terminal_residual', float('nan'))):.3e}  "
            f"{record.get('termination_reason')}",
            fontsize=7.0,
        )
    figure.suptitle(
        f"MAST {identity} {solve_class} terminal state, both anchor treatments on one "
        "shared level array.\nsolid: this panel's axis (triangle) and admitted saddle "
        "(cross); hollow grey: the other treatment's; wall drawn from the traced arm.",
        fontsize=7.5,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, facecolor=figure.get_facecolor())
    plt.close(figure)
    print(f"wrote {out_path}", flush=True)
    return 0


def _geometry_identity_report(operator: Any) -> dict[str, Any]:
    """Report the operator's cached geometry identity.

    The identity is computed once, in ``__post_init__``, from the host-owned
    geometry inputs plus every field that is not a declared dynamic extra.  It
    is therefore the instrument that shows whether a treatment reached the
    construction or was applied too late: the anchors join the specialisation
    when the hook does not name them, and the identity moves.
    """

    identity = getattr(operator, "_geometry_identity", None)
    if identity is None:
        return {"present": False, "digest": "absent"}
    if isinstance(identity, str):
        return {"present": True, "digest": identity}
    leaves = _flatten_operand(identity)
    return {
        "present": True,
        "digest": _digest(np.concatenate(leaves)) if leaves else "empty",
    }


def _control(
    identity: str, treatments: tuple[str, ...], compile_cache_root: Path | None
) -> int:
    """Report each treatment's partition without solving.

    This is the pre-flight positive control.  A treatment that silently changed
    nothing would leave the two arms' hook answers and partition digests equal,
    and the measurement would then be of two identical runs.  Building the
    operands costs seconds on a CPU host and no allocation, so the control is
    cheap next to the job it protects.
    """

    producer = _load_module(ROOT / PRODUCER_RELATIVE, "anchor_bisect_control_producer")
    producer.configure_dtypes()
    if compile_cache_root is not None:
        producer.configure_persistent_compilation_cache(compile_cache_root)
    response_cache, carrier_evidence = producer._persisted_response_cache(
        producer.response_carrier.DEFAULT_CARRIER,
        producer.response_carrier.DEFAULT_RECEIPT,
    )
    selected = [
        (row, qualification)
        for row, qualification in producer.select_slices_by_shot(
            producer.DECOMPOSITION_BANK
        )
        if _identity_of(row) == identity
    ]
    if not selected:
        raise SystemExit(f"identity {identity} is not in the decomposition bank")
    selected_row, qualification = selected[0]

    reports: dict[str, Any] = {}
    for treatment in treatments:
        with _anchor_treatment(treatment):
            _passive_case, profile = _build_operands(
                producer, selected_row, qualification, response_cache
            )
            operator = profile.operator
            partition = _partition_report(operator)
            report: dict[str, Any] = {
                "hook_names": _hook_names(operator),
                "partition": partition,
                "anchors": _anchor_snapshot(operator),
                "profile_support": _profile_support_digest(profile),
                "geometry_identity": _geometry_identity_report(operator),
            }
            reports[treatment] = report
            identity_report = report["geometry_identity"]
            print(
                f"CONTROL treatment={treatment} "
                f"hook_names={json.dumps(report['hook_names'])} "
                f"leaf_count={partition['summary']['leaf_count']} "
                f"structure={partition['structure_digest']} "
                f"values={partition['values_digest']} "
                f"geometry_identity={identity_report.get('digest')} "
                f"support={report['profile_support']['digest']}",
                flush=True,
            )

    names = {
        treatment: tuple(record["hook_names"]) for treatment, record in reports.items()
    }
    distinct_names = len({value for value in names.values()}) > 1
    leaves = {
        record["partition"]["summary"]["leaf_count"] for record in reports.values()
    }
    structures = {
        record["partition"]["structure_digest"] for record in reports.values()
    }
    supports = {record["profile_support"]["digest"] for record in reports.values()}
    identities = {
        record["geometry_identity"].get("digest") for record in reports.values()
    }
    print(
        "CONTROL_SUMMARY "
        + json.dumps(
            {
                "hook_names": {key: list(value) for key, value in names.items()},
                "hook_answers_differ": distinct_names,
                "leaf_counts": sorted(leaves),
                "leaf_count_delta": max(leaves) - min(leaves) if len(leaves) > 1 else 0,
                "structure_digests_differ": len(structures) > 1,
                "profile_support_invariant": len(supports) == 1,
                "geometry_identity_differs": len(identities) > 1,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="solve one identity under every treatment and solve class"
    )
    run_parser.add_argument("--identity", default=DEFAULT_IDENTITY)
    run_parser.add_argument(
        "--treatments", nargs="+", default=list(TREATMENTS), choices=list(TREATMENTS)
    )
    run_parser.add_argument(
        "--classes", nargs="+", default=list(SOLVE_CLASSES), choices=list(SOLVE_CLASSES)
    )
    run_parser.add_argument("--out-dir", required=True, type=Path)
    run_parser.add_argument("--compile-cache-root", type=Path, default=None)
    run_parser.add_argument("--panel-out", type=Path, default=None)
    run_parser.add_argument(
        "--panel-class", default="pure", choices=list(SOLVE_CLASSES)
    )
    run_parser.add_argument("--panel-levels", type=int, default=18)

    control_parser = subparsers.add_parser(
        "control", help="report each treatment's partition without solving"
    )
    control_parser.add_argument("--identity", default=DEFAULT_IDENTITY)
    control_parser.add_argument(
        "--treatments", nargs="+", default=list(TREATMENTS), choices=list(TREATMENTS)
    )
    control_parser.add_argument("--compile-cache-root", type=Path, default=None)

    receipt_parser = subparsers.add_parser(
        "receipt", help="join the arms against the committed bank reference"
    )
    receipt_parser.add_argument("--out-dir", required=True, type=Path)
    receipt_parser.add_argument("--identity", default=DEFAULT_IDENTITY)
    receipt_parser.add_argument(
        "--bank-dir", type=Path, default=ROOT / BANK_REFERENCE_RELATIVE
    )
    receipt_parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    if args.command == "receipt":
        return _receipt(args.out_dir, args.identity, args.bank_dir, args.out)
    if args.command == "control":
        return _control(args.identity, tuple(args.treatments), args.compile_cache_root)

    status = _run(
        args.identity,
        tuple(args.treatments),
        tuple(args.classes),
        args.out_dir,
        args.compile_cache_root,
    )
    if status == 0 and args.panel_out is not None:
        try:
            panel_status = _panel(
                args.identity,
                args.panel_class,
                args.out_dir,
                args.panel_out,
                args.panel_levels,
            )
        except Exception as error:  # noqa: BLE001 - the panel is not the measurement
            print(f"PANEL_FAILED {type(error).__name__}: {error}", flush=True)
            print(traceback.format_exc(limit=12), flush=True)
            panel_status = 0
        print(f"PANEL_STATUS {panel_status}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
