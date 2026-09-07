"""Census a contiguous zero-converged labeller block and its neighbours.

The census deliberately reads admission and reconstruction seeds through the
production shard helpers.  Its JSON-lines checkpoint is append-only per shot,
so a bounded compute allocation can resume without losing completed rows.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import zarr

from scripts.labeller_batch.shard import (
    SHOT_STORE,
    _slice_inputs,
    _slices_seed,
    prepare_labeller,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_ROOT = Path(
    "/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29"
)
DEFAULT_OUTPUT_DIRECTORY = (
    ROOT / "docs/figures/playable-forward-solve/zero-converged-block"
)
DEFAULT_REPORT = Path(
    "/home/ITER/mcintos/.config/reckon/crew/reports/nova/playable/"
    "zero-converged-block.md"
)
DEFAULT_BLOCK_START = 22_475
DEFAULT_BLOCK_END = 22_626
DEFAULT_NEIGHBOURS_PER_SIDE = 20


def _source_revision() -> str:
    """Return the revision supplying this census."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_write_text(path: Path, content: str) -> None:
    """Replace one text artifact only after its complete payload is durable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write one indented JSON artifact atomically."""
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject other top-level shapes."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return payload


def _manifest_summary(path: Path) -> dict[str, Any]:
    """Read the fields needed to select converging neighbours."""
    payload = _read_json(path)
    return {
        "shot": int(payload.get("shot", path.name.split(".", 1)[0])),
        "status": payload.get("status"),
        "converged": int(payload.get("converged_slice_count", 0)),
    }


def _select_shots(
    corpus_root: Path,
    block_start: int,
    block_end: int,
    neighbours_per_side: int,
) -> dict[str, list[int]]:
    """Select the full integer block and nearest converging neighbours."""
    converging: list[int] = []
    for path in sorted(corpus_root.glob("*.manifest.json")):
        try:
            summary = _manifest_summary(path)
        except OSError, ValueError, json.JSONDecodeError:
            continue
        shot = summary["shot"]
        if summary["status"] == "complete" and summary["converged"] > 0:
            converging.append(shot)
    left = sorted(shot for shot in converging if shot < block_start)[
        -neighbours_per_side:
    ]
    right = sorted(shot for shot in converging if shot > block_end)[
        :neighbours_per_side
    ]
    if len(left) != neighbours_per_side or len(right) != neighbours_per_side:
        raise RuntimeError(
            "not enough complete converging neighbours: "
            f"left={len(left)}, right={len(right)}, "
            f"required={neighbours_per_side}"
        )
    return {
        "left_neighbours": left,
        "block": list(range(block_start, block_end + 1)),
        "right_neighbours": right,
    }


def _selection(
    path: Path,
    corpus_root: Path,
    block_start: int,
    block_end: int,
    neighbours_per_side: int,
) -> dict[str, list[int]]:
    """Create or reuse the stable shot selection for resumed allocations."""
    if path.is_file():
        payload = _read_json(path)
        selected = payload.get("shots")
        expected = {
            "corpus_root": str(corpus_root.resolve()),
            "block_start": block_start,
            "block_end": block_end,
            "neighbours_per_side": neighbours_per_side,
        }
        observed = {key: payload.get(key) for key in expected}
        if observed != expected or not isinstance(selected, dict):
            raise ValueError(f"existing selection does not match this census: {path}")
        return {key: [int(shot) for shot in selected[key]] for key in selected}
    selected = _select_shots(corpus_root, block_start, block_end, neighbours_per_side)
    _atomic_write_json(
        path,
        {
            "corpus_root": str(corpus_root.resolve()),
            "block_start": block_start,
            "block_end": block_end,
            "neighbours_per_side": neighbours_per_side,
            "shots": selected,
        },
    )
    return selected


def _finite_range(values: Any) -> list[float] | None:
    """Return the inclusive range of finite values, or no range."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return None
    return [float(np.min(finite)), float(np.max(finite))]


def _contains(text: str, phrase: str) -> bool:
    """Return whether a case-insensitive diagnostic contains one phrase."""
    return phrase.casefold() in text.casefold()


def _slice_failure_stage(row: dict[str, Any]) -> str | None:
    """Return the earliest pipeline stage evidenced as failed by one row."""
    if bool(row.get("excluded")):
        return None
    free = str(row.get("free_solve_exception") or "")
    conditioned = str(row.get("conditioning_exception") or "")
    frame = str(row.get("frame_exception") or "")
    if _contains(free, "non-finite reconstruction flux seed"):
        return "seed-finiteness"
    if _contains(free, "NoQualifiedAxisError"):
        return "free-axis-admission"
    if free:
        return "free-solve"
    if _contains(conditioned, "non-finite reconstruction flux seed"):
        return "conditioning-seed"
    if _contains(conditioned, "NoQualifiedAxisError"):
        return "conditioning-axis-admission"
    if conditioned:
        return "conditioned-solve"
    if frame:
        return "frame-assembly"
    if not bool(row.get("converged")):
        if row.get("conditioned_converged") is False:
            return "conditioned-nonconvergence"
        if row.get("free_converged") is False:
            return "free-nonconvergence"
        return "terminal-nonconvergence"
    if not bool(row.get("qualified", True)):
        return "qualification"
    return None


def _first_failure(manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first admitted failed row and its earliest failed stage."""
    slices = manifest.get("slices", [])
    if not isinstance(slices, list):
        return {"row": None, "stage": "manifest-schema"}
    admitted = [row for row in slices if not bool(row.get("excluded"))]
    if not admitted:
        return {"row": None, "stage": "admission"}
    for row in sorted(admitted, key=lambda item: int(item.get("row", -1))):
        stage = _slice_failure_stage(row)
        if stage is not None:
            return {"row": int(row.get("row", -1)), "stage": stage}
    return None


def _efm_summary(shot: int, shot_store: Path) -> dict[str, Any]:
    """Return scalar ranges and fitted-coil finiteness for one EFM group."""
    path = shot_store / f"{shot}.zarr"
    root = zarr.open_group(str(path), mode="r")
    if "efm" not in root:
        raise KeyError(f"{path} has no efm group")
    group = root["efm"]
    fcoil = np.asarray(group["fcoil_c"], dtype=np.float64)
    finite = np.isfinite(fcoil)
    finite_rows = np.all(finite, axis=tuple(range(1, finite.ndim)))
    return {
        "path": str(path),
        "row_count": int(group["time"].shape[0]),
        "time_s": _finite_range(group["time"]),
        "plasma_current_a": _finite_range(group["plasma_current_c"]),
        "magnetic_axis_z_m": _finite_range(group["magnetic_axis_z"]),
        "current_centroid_z_m": _finite_range(group["current_centrd_z"]),
        "fcoil_finite": {
            "finite_values": int(np.count_nonzero(finite)),
            "total_values": int(finite.size),
            "finite_rows": int(np.count_nonzero(finite_rows)),
            "total_rows": int(finite_rows.size),
            "all_finite": bool(np.all(finite)),
        },
    }


def _shot_row(
    shot: int,
    cohort: str,
    corpus_root: Path,
    shot_store: Path,
) -> dict[str, Any]:
    """Build one durable census row without hiding either read failure."""
    reasons: list[str] = []
    manifest: dict[str, Any] | None = None
    manifest_path = corpus_root / f"{shot}.manifest.json"
    try:
        manifest = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        reasons.append(f"manifest unreadable: {type(error).__name__}: {error}")
    try:
        efm = _efm_summary(shot, shot_store)
    except (OSError, KeyError, ValueError) as error:
        efm = None
        reasons.append(f"efm unreadable: {type(error).__name__}: {error}")

    counts = None
    first_failure = None
    manifest_status = None
    if manifest is not None:
        manifest_status = manifest.get("status")
        counts = {
            key: int(manifest.get(key, 0))
            for key in (
                "slice_count",
                "admitted_slice_count",
                "written_slice_count",
                "converged_slice_count",
                "unconverged_slice_count",
                "excluded_slice_count",
            )
        }
        first_failure = _first_failure(manifest)
        if manifest_status != "complete":
            reasons.append(f"manifest status is {manifest_status!r}, not 'complete'")
    return {
        "cohort": cohort,
        "shot": shot,
        "readable": not reasons,
        "unreadable_reason": "; ".join(reasons) if reasons else None,
        "manifest_path": str(manifest_path),
        "manifest_status": manifest_status,
        "counts": counts,
        "first_failure": first_failure,
        "efm": efm,
    }


def _checkpoint_rows(path: Path) -> dict[int, dict[str, Any]]:
    """Read complete per-shot checkpoints and ignore a torn final line."""
    rows: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return rows
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        rows[int(row["shot"])] = row
    return rows


def _append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    """Append and fsync one shot before advancing to the next."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _ordered_selection(selected: dict[str, list[int]]) -> list[tuple[str, int]]:
    """Return the stable display and checkpoint order."""
    result: list[tuple[str, int]] = []
    for cohort in ("left_neighbours", "block", "right_neighbours"):
        result.extend((cohort, shot) for shot in selected[cohort])
    return result


def _sample_shots(
    rows: list[dict[str, Any]], selected: dict[str, list[int]]
) -> list[tuple[str, int]]:
    """Choose three distributed block shots and three nearby neighbours."""
    by_shot = {row["shot"]: row for row in rows}
    block_candidates = [
        shot
        for shot in selected["block"]
        if by_shot[shot]["manifest_status"] == "complete"
        and by_shot[shot]["counts"]
        and by_shot[shot]["counts"]["converged_slice_count"] == 0
        and by_shot[shot]["efm"] is not None
    ]
    if len(block_candidates) < 3:
        raise RuntimeError("fewer than three readable zero-converged block shots")
    targets = [
        selected["block"][0],
        sum(selected["block"]) / len(selected["block"]),
        selected["block"][-1],
    ]
    chosen_block: list[int] = []
    for target in targets:
        candidate = min(
            (shot for shot in block_candidates if shot not in chosen_block),
            key=lambda shot: (abs(shot - target), shot),
        )
        chosen_block.append(candidate)

    left = selected["left_neighbours"]
    right = selected["right_neighbours"]
    neighbour_candidates = [left[-1], right[0]]
    remaining = sorted(
        set(left + right) - set(neighbour_candidates),
        key=lambda shot: min(
            abs(shot - selected["block"][0]),
            abs(shot - selected["block"][-1]),
        ),
    )
    neighbour_candidates.append(remaining[0])
    return [("block", shot) for shot in chosen_block] + [
        ("neighbour", shot) for shot in neighbour_candidates
    ]


def _seed_check(
    shot: int,
    cohort: str,
    shot_store: Path,
    prepared: Any,
) -> dict[str, Any]:
    """Measure the production seed and its fixed-design topology inputs."""
    root = zarr.open_group(str(shot_store / f"{shot}.zarr"), mode="r")
    group = root["efm"]
    full_r = np.asarray(group["gridr"], dtype=np.float64)
    full_z = np.asarray(group["gridz"], dtype=np.float64)
    first_admitted = next(
        (
            row
            for row in range(int(group["time"].shape[0]))
            if _slice_inputs(group, row) is not None
        ),
        None,
    )
    if first_admitted is None:
        return {
            "cohort": cohort,
            "shot": shot,
            "first_admitted_row": None,
            "error": "no admitted reconstruction row",
        }
    seed = np.asarray(
        _slices_seed(group, first_admitted, full_r, full_z), dtype=np.float64
    )
    operator = prepared.profile.operator
    physical = jnp.asarray(seed[: operator.physical_node_number])
    topology = operator._fixed_design_topology
    grid_flux, wall_flux = topology.split_flux_map(physical)
    vmap_o, _vmap_x = topology.grid(grid_flux)
    wall_point = topology.wall(wall_flux, operator.polarity)
    grid_host = np.asarray(jax.device_get(grid_flux), dtype=np.float64)
    wall_boundary_flux = float(jax.device_get(wall_point[2]))
    finite_candidates = np.all(
        np.isfinite(np.asarray(jax.device_get(vmap_o))[:, :3]), axis=1
    )
    axis_r = float(group["magnetic_axis_r"][first_admitted])
    axis_z = float(group["magnetic_axis_z"][first_admitted])
    seed_finite = bool(np.all(np.isfinite(seed)))
    boundary_bounds = bool(
        np.isfinite(wall_boundary_flux)
        and np.nanmin(grid_host) <= wall_boundary_flux <= np.nanmax(grid_host)
    )
    axis_on_lattice = bool(
        full_r[0] <= axis_r <= full_r[-1] and full_z[0] <= axis_z <= full_z[-1]
    )
    return {
        "cohort": cohort,
        "shot": shot,
        "first_admitted_row": first_admitted,
        "writer_seed_finite": seed_finite,
        "seed_value_count": int(seed.size),
        "extrapolated_boundary_flux_wb": wall_boundary_flux,
        "grid_flux_range_wb": [
            float(np.nanmin(grid_host)),
            float(np.nanmax(grid_host)),
        ],
        "extrapolated_boundary_flux_bounds_map": boundary_bounds,
        "axis_candidate_count": int(np.count_nonzero(finite_candidates)),
        "stored_axis_rz_m": [axis_r, axis_z],
        "stored_axis_on_lattice": axis_on_lattice,
        "error": None,
    }


def _seed_checks(
    rows: list[dict[str, Any]],
    selected: dict[str, list[int]],
    shot_store: Path,
) -> list[dict[str, Any]]:
    """Run six seed checks through one shared writer operator."""
    prepared = prepare_labeller()
    checks: list[dict[str, Any]] = []
    for cohort, shot in _sample_shots(rows, selected):
        try:
            checks.append(_seed_check(shot, cohort, shot_store, prepared))
        except Exception as error:
            checks.append(
                {
                    "cohort": cohort,
                    "shot": shot,
                    "first_admitted_row": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return checks


def _degenerate(check: dict[str, Any]) -> bool:
    """Return whether any requested seed-degeneracy signal failed."""
    return bool(
        check.get("error")
        or not check.get("writer_seed_finite")
        or not check.get("extrapolated_boundary_flux_bounds_map")
        or int(check.get("axis_candidate_count", 0)) == 0
    )


def _range_across_rows(rows: list[dict[str, Any]], field: str) -> list[float] | None:
    """Return one scalar range across every readable EFM shot range."""
    endpoints = [
        value
        for row in rows
        if row["efm"] is not None and row["efm"].get(field) is not None
        for value in row["efm"][field]
    ]
    return _finite_range(endpoints)


def _recommendation(
    rows: list[dict[str, Any]], checks: list[dict[str, Any]]
) -> dict[str, Any]:
    """Choose one action from the measured block-specific seed contrast."""
    block_rows = [row for row in rows if row["cohort"] == "block"]
    affected = sum(
        row["manifest_status"] == "complete"
        and row["counts"] is not None
        and row["counts"]["converged_slice_count"] == 0
        for row in block_rows
    )
    block_checks = [check for check in checks if check["cohort"] == "block"]
    neighbour_checks = [check for check in checks if check["cohort"] == "neighbour"]
    block_degenerate = sum(_degenerate(check) for check in block_checks)
    neighbour_degenerate = sum(_degenerate(check) for check in neighbour_checks)
    block_current = _range_across_rows(block_rows, "plasma_current_a")
    neighbour_rows = [row for row in rows if row["cohort"] != "block"]
    neighbour_current = _range_across_rows(neighbour_rows, "plasma_current_a")
    polarity_contrast = bool(
        block_current is not None
        and neighbour_current is not None
        and block_current[1] < 0.0 < neighbour_current[0]
    )
    if block_degenerate >= 2 and neighbour_degenerate == 0:
        action = "repair-seed"
        text = (
            f"Repair the writer seed for the {affected} complete zero-converged "
            "shots in the block before admitting them to the demo cohort."
        )
        rationale = (
            f"{block_degenerate} of {len(block_checks)} block seeds are degenerate "
            f"against 0 of {len(neighbour_checks)} converging-neighbour seeds."
        )
    else:
        action = "exclude-block"
        text = (
            f"Exclude the {affected} complete zero-converged block shots from "
            "the demo cohort; the six-shot seed contrast does not isolate a "
            "block-specific repairable seed defect."
        )
        rationale = (
            f"Seed degeneracy appears in {block_degenerate} of {len(block_checks)} "
            f"block samples and {neighbour_degenerate} of "
            f"{len(neighbour_checks)} neighbour samples."
        )
        if polarity_contrast:
            rationale += (
                " The readable block EFM current is entirely negative "
                f"({_format_range(block_current, 1.0e-6)} MA), while the "
                "converging neighbours are entirely positive "
                f"({_format_range(neighbour_current, 1.0e-6)} MA)."
            )
    return {
        "action": action,
        "affected_shot_count": affected,
        "text": text,
        "rationale": rationale,
        "polarity_contrast": polarity_contrast,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate shot and first-stage counts without filtering failures."""
    result: dict[str, Any] = {}
    for cohort in ("left_neighbours", "block", "right_neighbours"):
        members = [row for row in rows if row["cohort"] == cohort]
        efm_members = [row for row in members if row["efm"] is not None]
        result[cohort] = {
            "shot_count": len(members),
            "readable_count": sum(row["readable"] for row in members),
            "efm_readable_count": len(efm_members),
            "complete_manifest_count": sum(
                row["manifest_status"] == "complete" for row in members
            ),
            "zero_converged_complete_count": sum(
                row["manifest_status"] == "complete"
                and row["counts"] is not None
                and row["counts"]["converged_slice_count"] == 0
                for row in members
            ),
            "converging_complete_count": sum(
                row["manifest_status"] == "complete"
                and row["counts"] is not None
                and row["counts"]["converged_slice_count"] > 0
                for row in members
            ),
            "plasma_current_a": _range_across_rows(members, "plasma_current_a"),
            "magnetic_axis_z_m": _range_across_rows(members, "magnetic_axis_z_m"),
            "current_centroid_z_m": _range_across_rows(members, "current_centroid_z_m"),
            "fcoil_all_finite_shot_count": sum(
                row["efm"]["fcoil_finite"]["all_finite"] for row in efm_members
            ),
            "first_failure_stages": dict(
                sorted(
                    Counter(
                        "unreadable"
                        if not row["readable"]
                        else (
                            row["first_failure"]["stage"]
                            if row["first_failure"] is not None
                            else "none"
                        )
                        for row in members
                    ).items()
                )
            ),
        }
    return result


def _format_range(values: list[float] | None, scale: float = 1.0) -> str:
    """Format one inclusive range compactly."""
    if values is None:
        return "—"
    return f"{values[0] * scale:.4g}–{values[1] * scale:.4g}"


def _markdown(receipt: dict[str, Any]) -> str:
    """Render the complete evidence table and the six-shot seed contrast."""
    summary = receipt["summary"]
    recommendation = receipt["recommendation"]
    lines = [
        "# Zero-converged labeller block census",
        "",
        (
            f"Generated at `{receipt['generated_at']}` from Nova "
            f"`{receipt['source_revision'][:12]}`. The table contains every "
            f"integer shot from {receipt['block']['start']} through "
            f"{receipt['block']['end']}, plus the nearest "
            f"{receipt['neighbours_per_side']} complete converging corpus shots "
            "on each side. Missing or partial inputs remain explicit rows."
        ),
        "",
        "## Headline",
        "",
        (
            f"The block has {summary['block']['zero_converged_complete_count']} "
            "complete zero-converged shots, "
            f"{summary['block']['converging_complete_count']} complete converging "
            "shots, and "
            f"{summary['block']['shot_count'] - summary['block']['readable_count']} "
            "rows with an unreadable or incomplete required input."
        ),
        "",
        (
            "The readable block current range is "
            f"{_format_range(summary['block']['plasma_current_a'], 1.0e-6)} MA; "
            "the selected converging neighbours span "
            f"{_format_range(summary['left_neighbours']['plasma_current_a'], 1.0e-6)} "
            "MA on the left and "
            f"{_format_range(summary['right_neighbours']['plasma_current_a'], 1.0e-6)} "
            "MA on the right."
        ),
        "",
        "## Seed-degeneracy contrast",
        "",
        (
            "The boundary value is the writer topology's wall extremum from the "
            "spline-extrapolated limiter tail of `_slices_seed`; it bounds the "
            "map when it lies inside the seed grid's finite flux range. The axis "
            "candidate count is the finite O-point census produced by the same "
            "fixed-design topology grid at the first admitted row."
        ),
        "",
        "|cohort|shot|row|seed finite|boundary bounds map|axis candidates|"
        "stored axis on lattice|boundary Wb|grid Wb range|error|",
        "|---|---:|---:|---|---|---:|---|---:|---|---|",
    ]
    for check in receipt["seed_degeneracy_checks"]:
        bounds = check.get("grid_flux_range_wb")
        lines.append(
            "|{cohort}|{shot}|{row}|{finite}|{bounds_map}|{candidates}|{on_lattice}|{boundary}|{grid}|{error}|".format(
                cohort=check["cohort"],
                shot=check["shot"],
                row=check.get("first_admitted_row", "—"),
                finite=check.get("writer_seed_finite", "—"),
                bounds_map=check.get("extrapolated_boundary_flux_bounds_map", "—"),
                candidates=check.get("axis_candidate_count", "—"),
                on_lattice=check.get("stored_axis_on_lattice", "—"),
                boundary=(
                    f"{check['extrapolated_boundary_flux_wb']:.6g}"
                    if "extrapolated_boundary_flux_wb" in check
                    else "—"
                ),
                grid=_format_range(bounds),
                error=str(check.get("error") or "—").replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "## Per-shot census",
            "",
            (
                "`first failure` is the earliest admitted manifest row that did "
                "not complete the writer pipeline, classified at the earliest "
                "stage explicitly evidenced by its exception and status fields. "
                "Excluded reconstruction rows are counted separately and do not "
                "mask the first admitted failure."
            ),
            "",
            "|cohort|shot|read|EFM rows|time s|Ip MA|axis z m|centroid z m|"
            "finite fcoil rows|admitted|excluded|converged|first failure|",
            "|---|---:|---|---:|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in receipt["shots"]:
        efm = row["efm"] or {}
        counts = row["counts"] or {}
        fcoil = efm.get("fcoil_finite") or {}
        first = row["first_failure"]
        first_text = "unreadable"
        if row["readable"]:
            first_text = (
                f"{first['row']}:{first['stage']}" if first is not None else "none"
            )
        read = "ok" if row["readable"] else str(row["unreadable_reason"])
        finite_rows = f"{fcoil['finite_rows']}/{fcoil['total_rows']}" if fcoil else "—"
        lines.append(
            "|{cohort}|{shot}|{read}|{efm_rows}|{time}|{current}|{axis}|{centroid}|{fcoil}|{admitted}|{excluded}|{converged}|{first}|".format(
                cohort=row["cohort"],
                shot=row["shot"],
                read=read.replace("|", "\\|"),
                efm_rows=efm.get("row_count", "—"),
                time=_format_range(efm.get("time_s")),
                current=_format_range(efm.get("plasma_current_a"), 1.0e-6),
                axis=_format_range(efm.get("magnetic_axis_z_m")),
                centroid=_format_range(efm.get("current_centroid_z_m")),
                fcoil=finite_rows,
                admitted=counts.get("admitted_slice_count", "—"),
                excluded=counts.get("excluded_slice_count", "—"),
                converged=counts.get("converged_slice_count", "—"),
                first=first_text,
            )
        )
    lines.extend(
        [
            "",
            "## First-failure census",
            "",
            "```json",
            json.dumps(
                {
                    cohort: values["first_failure_stages"]
                    for cohort, values in summary.items()
                },
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## Recommendation",
            "",
            f"**{recommendation['text']}** {recommendation['rationale']}",
            "",
        ]
    )
    return "\n".join(lines)


def measure(
    *,
    corpus_root: Path,
    shot_store: Path,
    output_directory: Path,
    report: Path,
    block_start: int,
    block_end: int,
    neighbours_per_side: int,
) -> dict[str, Any]:
    """Run or resume the census and publish its complete artifacts."""
    output_directory.mkdir(parents=True, exist_ok=True)
    selection_path = output_directory / "selection.json"
    checkpoint_path = output_directory / "census.checkpoint.jsonl"
    selected = _selection(
        selection_path,
        corpus_root,
        block_start,
        block_end,
        neighbours_per_side,
    )
    rows_by_shot = _checkpoint_rows(checkpoint_path)
    ordered = _ordered_selection(selected)
    for cohort, shot in ordered:
        if shot in rows_by_shot:
            continue
        row = _shot_row(shot, cohort, corpus_root, shot_store)
        _append_checkpoint(checkpoint_path, row)
        rows_by_shot[shot] = row
        print(
            "CENSUS_CHECKPOINT "
            + json.dumps(
                {
                    "shot": shot,
                    "cohort": cohort,
                    "readable": row["readable"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    rows = [rows_by_shot[shot] for _cohort, shot in ordered]
    if len(rows) != len(ordered):
        raise RuntimeError(f"checkpoint has {len(rows)} rows, expected {len(ordered)}")
    checks = _seed_checks(rows, selected, shot_store)
    receipt = {
        "schema": "nova-zero-converged-block-census",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_revision": _source_revision(),
        "corpus_root": str(corpus_root.resolve()),
        "shot_store": str(shot_store.resolve()),
        "block": {"start": block_start, "end": block_end},
        "neighbours_per_side": neighbours_per_side,
        "selection": selected,
        "checkpoint": str(checkpoint_path.resolve()),
        "summary": _summary(rows),
        "seed_degeneracy_checks": checks,
        "recommendation": _recommendation(rows, checks),
        "shots": rows,
    }
    markdown = _markdown(receipt)
    _atomic_write_json(output_directory / "census.json", receipt)
    _atomic_write_text(output_directory / "census.md", markdown)
    _atomic_write_text(report, markdown)
    return receipt


def _parser() -> argparse.ArgumentParser:
    """Return the command-line contract for the resumable census."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS_ROOT)
    parser.add_argument("--shot-store", type=Path, default=SHOT_STORE)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--block-start", type=int, default=DEFAULT_BLOCK_START)
    parser.add_argument("--block-end", type=int, default=DEFAULT_BLOCK_END)
    parser.add_argument(
        "--neighbours-per-side", type=int, default=DEFAULT_NEIGHBOURS_PER_SIDE
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the census and print its quantitative completion line."""
    arguments = _parser().parse_args(argv)
    if arguments.block_start > arguments.block_end:
        raise ValueError("block start must not exceed block end")
    if arguments.neighbours_per_side < 1:
        raise ValueError("neighbours per side must be positive")
    receipt = measure(
        corpus_root=arguments.corpus_root,
        shot_store=arguments.shot_store,
        output_directory=arguments.output_directory,
        report=arguments.report,
        block_start=arguments.block_start,
        block_end=arguments.block_end,
        neighbours_per_side=arguments.neighbours_per_side,
    )
    print(
        "ZERO_CONVERGED_BLOCK_CENSUS "
        + json.dumps(
            {
                "rows": len(receipt["shots"]),
                "block_zero_converged": receipt["summary"]["block"][
                    "zero_converged_complete_count"
                ],
                "recommendation": receipt["recommendation"]["action"],
                "affected_shots": receipt["recommendation"]["affected_shot_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
