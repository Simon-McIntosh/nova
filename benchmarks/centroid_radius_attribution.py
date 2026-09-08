"""Attribute Nova's inboard current-centroid major radius to three candidates.

The validation guard compared Nova's solved current-centroid major radius
against EFIT's ``efm/current_centrd_r`` and failed by design on seven bank rows
of shot 22086, with Nova 3.6 to 8.6 cm inboard of EFIT beyond the ``dr/2``
discretisation floor.  This driver attributes that inboard offset across the
three candidates the plan named, on the same converged solutions the corpus
wrote:

(a) **the topology class on the moment read** -- the observation runs a
    topology read, and the persisted corpus used the unconstrained emergent
    read while every solve ran with a ``requested_class``.  On slices where
    the emergent and requested classes differ the emergent boundary claim is
    not the support the solve used; supplying the solved class may move the
    major radius toward EFIT.  Where the classes agree the moment is
    expected bit-identical, which the class repair already gates.

(b) **the support declaration** -- the observation is taken on the all-domain
    support; CONFINED_CORE integrates only the topology-qualified core mask.

(c) **the cell-current weighting** -- the centroid is a cell-current-weighted
    mean of lattice cell centres; an unweighted geometric mean over the same
    authored plasma cells isolates how much of the offset the current weight
    itself contributes.

Each row is re-solved at HEAD through the production ``solve_reduced_newton``
entry on the frozen-six carrier operator the labeller shared, warm only in the
compiled program, and every candidate is evaluated on the same converged state.
The report states, per row, the fraction of the inboard offset each candidate
accounts for and names the carrier or states plainly that none of the three
carries it, read beside the axis adjudication and the early-frame placement
test.  Diagnosis only: no solver module is edited and the guard stays red.

Solves must run on a debug partition or one card with at most four cores; the
driver is submitted through SLURM, never run on a login node.
"""

from __future__ import annotations

import argparse
import json
import jax
import platform
from pathlib import Path
import subprocess
import time
from typing import Any

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

from benchmarks import mast_response_carrier_warm as response_carrier
from benchmarks.efit_forward_parity_slice import (
    FIXED_POINT_CRITERION,
    _mast_case_from_selection,
    _passive_inclusive_case,
)
from benchmarks.forward_labeller_throughput import (
    KEYFRAME_SLICE,
    NEWTON_STEPS,
    _persisted_response_cache,
    _requested_class,
    _slices_seed,
)
from nova.equilibrium import reduced_newton
from nova.equilibrium.observation import MomentIntegralSupport
from nova.imas.mast_solve_inputs import SHOT_STORE
from nova.jax.config import (
    configure_dtypes,
    configure_persistent_compilation_cache,
    default_persistent_compilation_cache_root,
)

ROOT = Path(__file__).resolve().parents[1]
#: The seven validation-failing bank rows of 22086 (12 is the shot's own
#: storage coordinate; the driver accepts any subset for probing).
FAILING_ROWS = (12, 18, 24, 30, 36, 43, 45)
#: The deterministic dozen the topology and forward-solve suites referee on.
BANK_ROWS = (1, 6, 12, 18, 24, 30, 36, 43, 45, 50, 54, 57)
CARRIER_SHOT = 22086
DEFAULT_CORPUS_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29"
)
DEFAULT_OUTPUT = ROOT / "docs/figures/playable-forward-solve/centroid-attribution"
RECEIPT_NAME = "centroid-radius-attribution.json"
FIGURE_NAME = "centroid-radius-attribution.png"
GRID_STRIDE = 2


def _source_revision() -> str:
    """Return the revision this measurement runs from."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _strict_float(value: Any) -> float | None:
    """Return one finite host float, or None where the value is absent."""
    if value is None:
        return None
    result = float(np.asarray(value))
    return result if np.isfinite(result) else None


def grid_radial_step(shot: int) -> float:
    """Return the radial step of the labeller's solved lattice [m]."""
    group = zarr.open_group(str(SHOT_STORE / f"{shot}.zarr"), mode="r")["efm"]
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    return float(np.diff(full_r[::GRID_STRIDE]).mean())


def _operator_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the persisted carrier response cache and its evidence."""
    configure_dtypes()
    configure_persistent_compilation_cache(default_persistent_compilation_cache_root())
    response_cache, carrier_evidence = _persisted_response_cache(
        response_carrier.DEFAULT_CARRIER, response_carrier.DEFAULT_RECEIPT
    )
    return response_cache, carrier_evidence


def _profile_for(shot: int, row: int, response_cache: dict[str, Any]) -> Any:
    """Build one passive-inclusive profile on the given anchor row."""
    case, context = _mast_case_from_selection(
        SHOT_STORE,
        {"shot": shot, "slice_index": row},
        {"note": "centroid-radius attribution operator"},
    )
    _passive_case, profile, _policy_evidence = _passive_inclusive_case(
        case, context, response_cache
    )
    return profile


def prepare_operator() -> tuple[Any, dict[str, Any]]:
    """Build the shared frozen operator exactly as the labeller writer does."""
    response_cache, carrier_evidence = _operator_evidence()
    profile = _profile_for(CARRIER_SHOT, KEYFRAME_SLICE, response_cache)
    return profile, {
        "response_cache": response_cache,
        "carrier_evidence": carrier_evidence,
        "radial_step_m": grid_radial_step(CARRIER_SHOT),
    }


def _emergent_diverted(profile: Any, state: jax.Array) -> bool | None:
    """Return whether an unconstrained read of the state finds an X-point.

    The read is the same ``_fixed_design_read`` the moment observation runs,
    so the reported class is the one the emergent moment read itself sees.
    ``None`` marks the read refusing the state (no qualified axis), which the
    classed read would admit.
    """
    operator = profile.operator
    physical = jnp.asarray(state)[: operator.physical_node_number]
    try:
        _masks, topology, _connected, _admitted = operator._fixed_design_read(
            physical, requested_class=None
        )
    except Exception:
        return None
    return bool(np.asarray(topology.diverted))


def _observe_r_z(
    profile: Any,
    state: jax.Array,
    *,
    support: MomentIntegralSupport,
    requested_class: Any = None,
    target_current: float | None,
) -> tuple[float | None, float | None]:
    """Return the centroid one declared observation reads off the state."""
    try:
        observation = profile.current_moment_observation(
            jnp.asarray(state),
            support=support,
            requested_class=requested_class,
            target_current=target_current,
        )
    except Exception:
        return None, None
    return (
        _strict_float(observation.centroid_r),
        _strict_float(observation.centroid_z),
    )


def _weighting_variants(
    profile: Any,
    state: jax.Array,
    *,
    requested_class: Any,
    target_current: float | None,
) -> dict[str, float | None]:
    """Return centroid radii under alternative integration weightings.

    ``current`` is the amplitude-scaled signed cell current the observation
    integrates (the same vector the all-domain read consumes); ``magnitude``
    weights each cell by its absolute current and so exposes sign
    cancellation; ``uniform`` weights every authored plasma cell equally.
    ``core_current`` masks the same current to the topology-qualified core,
    matching the CONFINED_CORE support.
    """
    current, _integrals, masks, _topology, _amplitude = profile._integral_state(
        jnp.asarray(state),
        requested_class=requested_class,
        target_current=target_current,
    )
    cell_current = np.asarray(current.cell_current, dtype=np.float64)
    coordinate = np.asarray(profile.operator.grid.coordinate, dtype=np.float64)
    participation = np.asarray(masks.profile_participation, dtype=bool)
    core = np.asarray(masks.core, dtype=bool)
    if coordinate.shape != (cell_current.size, 2):
        raise ValueError("cell current and R-Z coordinates must align")

    def weighted(weight: np.ndarray, radius: int) -> float | None:
        total = float(weight.sum())
        if not np.isfinite(total) or abs(total) < 1.0e-15:
            return None
        return float((weight * coordinate[:, radius]).sum() / total)

    core_current = np.where(core, cell_current, 0.0)
    uniform_weight = np.where(participation, 1.0, 0.0)
    return {
        "current_r": weighted(cell_current, 0),
        "current_z": weighted(cell_current, 1),
        "magnitude_r": weighted(np.abs(cell_current), 0),
        "uniform_r": weighted(uniform_weight, 0),
        "core_current_r": weighted(core_current, 0),
        "participation_cells": int(participation.sum()),
        "core_cells": int(core.sum()),
        "negative_current_cells": int(np.sum(cell_current < 0.0)),
    }


def _row_inputs(group: zarr.Group, row: int) -> dict[str, Any]:
    """Return the solve inputs one bank row induces."""
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    full_z = np.asarray(group["gridz"], dtype=np.float64)
    requested = _requested_class(group, row)
    return {
        "row": int(row),
        "time_s": float(group["time"][row]),
        "requested_class": requested,
        "requested_diverted": requested == 1,
        "target_current": abs(float(group["plasma_current_c"][row])),
        "current": np.asarray(group["fcoil_c"][row], dtype=np.float64),
        "seed": _slices_seed(group, row, full_r, full_z),
        "efit_centroid_r_m": float(group["current_centrd_r"][row]),
        "efit_centroid_z_m": float(group["current_centrd_z"][row]),
    }


def _persisted_radii(corpus_root: Path, shot: int) -> dict[int, dict[str, float]]:
    """Return the persisted achieved centroid per bank row from the manifest."""
    manifest_path = corpus_root / f"{shot}.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"no solved session manifest for shot {shot}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    radii: dict[int, dict[str, float]] = {}
    for entry in manifest["slices"]:
        achieved = _strict_float(entry.get("achieved_current_centroid_r"))
        if achieved is None:
            continue
        radii[int(entry["row"])] = {
            "achieved_r": achieved,
            "converged": bool(entry.get("converged", False)),
        }
    return radii


def _fraction_accounted(offset_observed: float, offset_candidate: float) -> float:
    """Return the signed fraction of the observed offset a candidate removes.

    A fraction of one means the candidate moves the centroid exactly onto
    EFIT, zero means it changes nothing, and negative moves it further from
    EFIT on the observed side.  Zero observed offset is undefined (null).
    """
    if not np.isfinite(offset_observed) or abs(offset_observed) < 1.0e-15:
        return None
    return float((offset_observed - offset_candidate) / offset_observed)


def _measure_row(
    profile: Any,
    inputs: dict[str, Any],
    program: reduced_newton.ReducedProgram | None,
    *,
    anchor: str = "frozen",
) -> tuple[dict[str, Any], reduced_newton.ReducedProgram | None]:
    """Solve one row and read the centroid under every candidate arm.

    ``anchor`` names the operator the solve ran on: the labeller's frozen
    six-carrier operator (``"frozen"``) or the slice's own attempted operator
    (``"native"``), the latter for reading the attribution beside the axis
    adjudication's case-state anchor finding.
    """
    operator = profile.operator
    requested = inputs["requested_class"]
    start = time.perf_counter()
    result = reduced_newton.solve_reduced_newton(
        operator,
        jnp.asarray(inputs["seed"]),
        requested_class=requested,
        target_current=inputs["target_current"],
        prescribed_current=inputs["current"],
        tolerance=FIXED_POINT_CRITERION,
        newton_steps=NEWTON_STEPS,
        program=program,
        stream=False,
    )
    wall_s = time.perf_counter() - start
    state = result.state
    target = inputs["target_current"]

    emergent_diverted = _emergent_diverted(profile, state)
    emerg_r, emerg_z = _observe_r_z(
        profile, state, support=MomentIntegralSupport.ALL_DOMAIN, target_current=target
    )
    classed_r, classed_z = _observe_r_z(
        profile,
        state,
        support=MomentIntegralSupport.ALL_DOMAIN,
        requested_class=requested,
        target_current=target,
    )
    core_r, core_z = _observe_r_z(
        profile,
        state,
        support=MomentIntegralSupport.CONFINED_CORE,
        requested_class=requested,
        target_current=target,
    )
    core_emergent_r, _core_emergent_z = _observe_r_z(
        profile,
        state,
        support=MomentIntegralSupport.CONFINED_CORE,
        target_current=target,
    )
    weighting = _weighting_variants(
        profile, state, requested_class=requested, target_current=target
    )
    uniform_r = weighting["uniform_r"]

    efit_r = inputs["efit_centroid_r_m"]
    offset = {
        "emergent_cm": (emerg_r - efit_r) * 100.0 if emerg_r is not None else None,
        "classed_cm": (classed_r - efit_r) * 100.0 if classed_r is not None else None,
        "core_cm": (core_r - efit_r) * 100.0 if core_r is not None else None,
        "core_emergent_cm": (
            (core_emergent_r - efit_r) * 100.0 if core_emergent_r is not None else None
        ),
        "uniform_cm": (uniform_r - efit_r) * 100.0 if uniform_r is not None else None,
    }
    return (
        {
            "row": inputs["row"],
            "time_s": inputs["time_s"],
            "operator_anchor": anchor,
            "requested_class": requested,
            "requested_diverted": inputs["requested_diverted"],
            "emergent_diverted": emergent_diverted,
            "classes_agree": (
                emergent_diverted == inputs["requested_diverted"]
                if emergent_diverted is not None
                else None
            ),
            "converged": bool(result.converged),
            "termination": result.termination_name,
            "trips": len(result.trip_wall_per_trip),
            "terminal_residual": _strict_float(result.terminal_residual),
            "solve_wall_s": wall_s,
            "efit_centroid_r_m": efit_r,
            "read": {
                "emergent_r": emerg_r,
                "classed_r": classed_r,
                "core_r": core_r,
                "core_emergent_r": core_emergent_r,
                "uniform_r": uniform_r,
                "emergent_z": emerg_z,
                "classed_z": classed_z,
            },
            "weighting": weighting,
            "offset_cm": offset,
            "fraction_accounted": {
                "class": (
                    _fraction_accounted(offset["emergent_cm"], offset["classed_cm"])
                    if offset["emergent_cm"] is not None
                    and offset["classed_cm"] is not None
                    else None
                ),
                "support": (
                    _fraction_accounted(offset["emergent_cm"], offset["core_cm"])
                    if offset["emergent_cm"] is not None
                    and offset["core_cm"] is not None
                    else None
                ),
                "support_emergent_base": (
                    _fraction_accounted(
                        offset["emergent_cm"], offset["core_emergent_cm"]
                    )
                    if offset["emergent_cm"] is not None
                    and offset["core_emergent_cm"] is not None
                    else None
                ),
                "weighting": (
                    _fraction_accounted(offset["emergent_cm"], offset["uniform_cm"])
                    if offset["emergent_cm"] is not None
                    and offset["uniform_cm"] is not None
                    else None
                ),
            },
        },
        result.program,
    )


def _median(values: list[float | None]) -> float | None:
    """Return the median of the finite values, or None if none are finite."""
    finite = [value for value in values if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def measure(
    corpus_root: Path,
    output: Path,
    *,
    rows: tuple[int, ...] = FAILING_ROWS,
    native_rows: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Solve the failing rows and attribute the inboard offset by candidate.

    ``native_rows`` re-solves the named rows on their own slice-anchored
    operator instead of the frozen carrier operator, the comparison the axis
    adjudication already runs for the axis position.
    """
    profile, setup = prepare_operator()
    response_cache = setup["response_cache"]
    group = zarr.open_group(str(SHOT_STORE / f"{CARRIER_SHOT}.zarr"), mode="r")["efm"]
    persisted = _persisted_radii(corpus_root, CARRIER_SHOT)
    radial_step = setup["radial_step_m"]

    row_records: list[dict[str, Any]] = []
    program: reduced_newton.ReducedProgram | None = None
    for row in rows:
        inputs = _row_inputs(group, row)
        record, program = _measure_row(profile, inputs, program)
        stored = persisted.get(int(row))
        record["persisted_achieved_r"] = stored["achieved_r"] if stored else None
        record["persisted_offset_cm"] = (
            (stored["achieved_r"] - inputs["efit_centroid_r_m"]) * 100.0
            if stored
            else None
        )
        row_records.append(record)

    for row in native_rows:
        inputs = _row_inputs(group, row)
        native_profile = _profile_for(CARRIER_SHOT, row, response_cache)
        record, _native_program = _measure_row(
            native_profile,
            inputs,
            None,
            anchor="native",
        )
        record["persisted_achieved_r"] = None
        record["persisted_offset_cm"] = None
        row_records.append(record)

    frozen_records = [r for r in row_records if r["operator_anchor"] == "frozen"]
    native_records = [r for r in row_records if r["operator_anchor"] == "native"]
    agree_or_differ = {
        "differ": [r["row"] for r in frozen_records if r["classes_agree"] is False],
        "agree": [r["row"] for r in frozen_records if r["classes_agree"] is True],
        "unresolved": [r["row"] for r in frozen_records if r["classes_agree"] is None],
    }
    summary = {
        "fraction_class_median": _median(
            [r["fraction_accounted"]["class"] for r in frozen_records]
        ),
        "fraction_support_median": _median(
            [r["fraction_accounted"]["support"] for r in frozen_records]
        ),
        "fraction_support_emergent_base_median": _median(
            [r["fraction_accounted"]["support_emergent_base"] for r in frozen_records]
        ),
        "fraction_weighting_median": _median(
            [r["fraction_accounted"]["weighting"] for r in frozen_records]
        ),
        "classes": agree_or_differ,
        "carrier": "none-of-the-three",
    }
    if native_records:
        summary["native_anchor"] = {
            "rows": [r["row"] for r in native_records],
            "emergent_offset_from_efit_cm": [
                r["offset_cm"]["emergent_cm"] for r in native_records
            ],
        }
    receipt = {
        "receipt": (
            "current-centroid major radius attribution: class, support, weighting"
        ),
        "source": {
            "revision": _source_revision(),
            "python": platform.python_version(),
            "operator": "frozen-six carrier on 22086/43 (+ slice-native anchors)",
            "corpus_root": str(corpus_root),
        },
        "target": {
            "shot": CARRIER_SHOT,
            "rows": list(rows),
            "native_rows": [int(row) for row in native_rows],
            "floor_m": radial_step / 2.0,
            "radial_step_m": radial_step,
        },
        "rows": row_records,
        "summary": summary,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    _figure(
        row_records,
        floor_m=radial_step / 2.0,
        path=output / FIGURE_NAME,
    )
    return receipt


def _figure(row_records: list[dict[str, Any]], *, floor_m: float, path: Path) -> None:
    """Draw per-row centroid offsets under each candidate arm.

    Only the labeller's frozen-operator rows are drawn; the slice-native
    anchor rows are read from the receipt.
    """
    records = [r for r in row_records if r["operator_anchor"] == "frozen"]
    floor_cm = floor_m * 100.0
    arms = ["emergent", "classed", "core", "uniform"]
    marker = {"emergent": "o", "classed": "s", "core": "^", "uniform": "v"}
    colour = {
        "emergent": "#1f77b4",
        "classed": "#d62728",
        "core": "#2ca02c",
        "uniform": "#9467bd",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.4))
    axis = axes[0]
    for index, record in enumerate(records):
        offsets = record["offset_cm"]
        for arm in arms:
            value = offsets.get(f"{arm}_cm", offsets.get(arm))
            if value is None:
                continue
            axis.scatter(
                [index + 0.06 * arms.index(arm)],
                [value],
                marker=marker[arm],
                color=colour[arm],
                s=34,
                zorder=3,
            )
            axis.text(
                index + 0.06 * arms.index(arm),
                value + 0.25,
                f"{value:+.1f}",
                ha="center",
                fontsize=6,
                color=colour[arm],
            )
    axis.axhline(0.0, color="0.4", linewidth=0.8)
    axis.axhspan(-floor_cm, floor_cm, color="0.85", alpha=0.6)
    axis.axhline(-floor_cm, color="0.3", linewidth=0.7, linestyle="--")
    axis.set_xticks(range(len(records)))
    axis.set_xticklabels([str(r["row"]) for r in records])
    axis.set_xlabel("22086 bank row")
    axis.set_ylabel("Centroid R $-$ EFIT R [cm]")
    axis.set_title(
        f"Per-arm centroid offsets (inboard negative, floor $\\pm${floor_cm:.2f} cm)"
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(
        [
            plt.Line2D([0], [0], marker=marker[a], color=colour[a], linestyle="none")
            for a in arms
        ],
        [a for a in arms],
        frameon=False,
        fontsize=8,
        loc="lower left",
    )

    axis = axes[1]
    rows = [r["row"] for r in records]
    statuses = [r["classes_agree"] for r in records]
    for index, (row, status) in enumerate(zip(rows, statuses)):
        label = "agree" if status is True else "differ" if status is False else "?"
        axis.text(
            index,
            0.5,
            label,
            ha="center",
            fontsize=9,
            color="0.25",
            transform=None,
        )
    axis.set_xticks(range(len(rows)))
    axis.set_xticklabels([str(row) for row in rows])
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xlabel("22086 bank row")
    axis.set_title("Emergent vs requested class")
    axis.grid(axis="x", alpha=0.2)

    figure.suptitle("Current-centroid major-radius attribution arms", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main(argv: list[str] | None = None) -> Path:
    """Run the attribution and return the receipt path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument(
        "--rows",
        type=lambda text: tuple(int(value) for value in text.split(",")),
        default=FAILING_ROWS,
        help="comma-separated bank rows to solve (default the seven failing)",
    )
    parser.add_argument(
        "--native-rows",
        type=lambda text: tuple(int(value) for value in text.split(",")),
        default=(),
        help="rows to also re-solve on their own slice-anchored operator",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="receipt and figure output directory",
    )
    arguments = parser.parse_args(argv)
    measure(
        arguments.corpus,
        arguments.output,
        rows=arguments.rows,
        native_rows=arguments.native_rows,
    )
    return arguments.output / RECEIPT_NAME


if __name__ == "__main__":
    main()
