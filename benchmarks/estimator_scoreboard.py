"""Publish one scoreboard for reconstructed equilibrium states.

Every column is a state supplied as an IMAS equilibrium, read at the data
dictionary version its own dataset was written at; every row is a case with a
reference state.  The comparator and the reference-free checks are the ones in
:mod:`nova.imas.mast_efit_referee`, and the bands are read from
:data:`nova.imas.parity_tolerances.STATE_SCOREBOARD_TOLERANCES`.

A new estimator joins by adding a column to the case manifest.  When that
estimator's states do not exist on disk yet, the manifest may declare a
perturbation of an existing state -- a deliberate defect the scoreboard must
rank worse -- and ``--emit-perturbed`` materialises it as its own IMAS
equilibrium before scoring, so the scoring path never sees a state that was not
supplied as one.

Run the table and its receipt::

    python benchmarks/estimator_scoreboard.py --manifest cases.json \
        --html <fragment>.html --receipt <receipt>.json

Materialise declared perturbed states first::

    python benchmarks/estimator_scoreboard.py --manifest cases.json \
        --state-dir <run directory>/states --emit-perturbed
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from nova.imas.mast_efit_referee import (
    EMPTY_SENTINEL,
    EquilibriumState,
    compare_states,
    read_equilibrium_state,
    reference_free_checks,
)
from nova.imas.parity_tolerances import (
    STATE_SCOREBOARD_FIELDS,
    STATE_SCOREBOARD_TOLERANCES,
)

DEFAULT_PERTURBATION = {
    "current_scale": 1.06,
    "radial_shift_m": 0.03,
    "flux_fraction": 0.02,
    "q_scale": 0.05,
}


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Load and shape-check the case manifest."""

    manifest = json.loads(Path(path).read_text())
    if "cases" not in manifest or not manifest["cases"]:
        raise ValueError(f"{path} carries no cases")
    for case in manifest["cases"]:
        if "id" not in case or "reference" not in case or "wall" not in case:
            raise ValueError(f"case {case!r} must name an id, a reference and a wall")
        names = [entry["name"] for entry in case.get("estimators", [])]
        if len(set(names)) != len(names):
            raise ValueError(f"case {case['id']} names an estimator twice")
    return manifest


def _state_source(entry: dict[str, Any], state_dir: Path | None) -> str | None:
    """Resolve the database a column reads, refusing an unresolvable path."""

    if "source" in entry:
        return str(entry["source"])
    if "out" in entry:
        if state_dir is None:
            raise ValueError(
                f"estimator {entry['name']!r} names a generated state {entry['out']!r} "
                "but no --state-dir was supplied"
            )
        return str(Path(state_dir) / entry["out"])
    return None


def perturb_state(
    source: Path | str,
    destination: Path | str,
    *,
    dd_version: str = "4.1.0",
    current_scale: float = 1.0,
    radial_shift_m: float = 0.0,
    flux_fraction: float = 0.0,
    q_scale: float = 1.0,
) -> Path:
    """Write a deliberately defective copy of one equilibrium as its own dataset.

    Every frame keeps its boundary shape and moves radially by ``radial_shift_m``
    about the machine axis, its plasma current and safety-factor profile are
    scaled, and its flux span is grown by ``flux_fraction`` at each end.  The
    copy carries no wall: the case manifest names the machine description the
    reference-free checks read.
    """

    import imas

    uri = str(source)
    if not uri.startswith("imas:"):
        uri = f"imas:hdf5?path={Path(source).expanduser().resolve()}"
    reader = imas.DBEntry(uri, "r")
    equilibrium = copy.deepcopy(reader.get("equilibrium"))
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    writer = imas.DBEntry(f"imas:hdf5?path={out.resolve()}", "w", dd_version=dd_version)
    for frame in equilibrium.time_slice:
        quantities = frame.global_quantities
        span = float(quantities.psi_axis) - float(quantities.psi_boundary)
        frame.boundary.outline.r = np.asarray(frame.boundary.outline.r, dtype=float) + (
            radial_shift_m
        )
        quantities.magnetic_axis.r = float(quantities.magnetic_axis.r) + radial_shift_m
        if np.isfinite(span) and abs(span) < EMPTY_SENTINEL:
            quantities.psi_axis = float(quantities.psi_axis) + flux_fraction * span
            quantities.psi_boundary = (
                float(quantities.psi_boundary) - flux_fraction * span
            )
        quantities.ip = float(quantities.ip) * current_scale
        profile = frame.profiles_1d
        profile.q = np.asarray(profile.q, dtype=float) * q_scale
    writer.put(equilibrium)
    writer.close()
    reader.close()
    return out


def emit_perturbed_states(
    manifest: dict[str, Any], state_dir: Path | str
) -> list[Path]:
    """Materialise every manifest column that declares a perturbed state."""

    written: list[Path] = []
    for case in manifest["cases"]:
        for entry in case.get("estimators", []):
            if "out" not in entry:
                continue
            perturbation = dict(DEFAULT_PERTURBATION)
            perturbation.update(entry.get("perturb", {}))
            origin = entry.get("perturbed_of", "reference")
            if origin == "reference":
                source = case["reference"]["source"]
            else:
                matches = [e for e in case["estimators"] if e["name"] == origin]
                if not matches or "source" not in matches[0]:
                    raise ValueError(f"case {case['id']} cannot perturb {origin!r}")
                source = matches[0]["source"]
            written.append(
                perturb_state(
                    source,
                    Path(state_dir) / entry["out"],
                    dd_version=str(case["reference"].get("dd_version", "4.1.0")),
                    **perturbation,
                )
            )
    return written


def score_case(case: dict[str, Any], state_dir: Path | None) -> dict[str, Any]:
    """Score every column of one case against its reference state."""

    reference = read_equilibrium_state(
        case["reference"]["source"],
        occurrence=int(case["reference"].get("occurrence", 0)),
        time_index=case["reference"].get("time_index"),
        time_s=case["reference"].get("time_s"),
        wall_source=case["wall"],
    )
    record: dict[str, Any] = {
        "case": case["id"],
        "machine": case.get("machine", ""),
        "reference": _state_record(reference),
        "estimators": [],
    }
    for entry in case.get("estimators", []):
        source = _state_source(entry, state_dir)
        if source is None or (
            not source.startswith("imas:") and not Path(source).exists()
        ):
            record["estimators"].append(
                {
                    "name": entry["name"],
                    "status": "absent",
                    "reason": "no state supplied for this case",
                }
            )
            continue
        state = read_equilibrium_state(
            source,
            occurrence=int(entry.get("occurrence", 0)),
            time_index=entry.get("time_index"),
            time_s=entry.get("time_s"),
            wall_source=case["wall"],
        )
        comparison = compare_states(reference, state)
        checks = reference_free_checks(state)
        record["estimators"].append(
            {
                "name": entry["name"],
                "status": "scored",
                "state": _state_record(state),
                "gauge_wb": comparison.gauge_wb,
                "metrics": comparison.metrics,
                "verdicts": dict(comparison.verdicts()),
                "reference_free": checks.values,
                "reference_free_verdicts": dict(checks.verdicts()),
                "lcfs_distance_m": comparison.lcfs_distance_m,
            }
        )
    return record


def _state_record(state: EquilibriumState) -> dict[str, Any]:
    """Return the receipt fields that name and identify one state."""

    return {
        "source": state.source,
        "digest": state.digest(),
        "dd_version": state.dd_version,
        "time_s": state.time_s,
        "flux_axis": state.flux_axis,
        "flux_boundary": state.flux_boundary,
        "flux_span": state.flux_span,
        "plasma_current_a": state.plasma_current_a,
        "q95": state.q95,
        "magnetic_axis_m": list(state.magnetic_axis_m),
        "boundary_points": int(state.boundary_m.shape[0]),
        "wall_rings": len(state.wall_loops),
    }


def _mark(verdict: bool | None) -> str:
    """Render one verdict as a marked cell."""

    if verdict is None:
        return "unavailable"
    return "pass" if verdict else "FAIL"


def _format(value: float, unit: str) -> str:
    """Render one metric value, naming an absent measure rather than zero."""

    if not np.isfinite(value):
        return "unavailable"
    return f"{value:.4g} {unit}".strip()


UNITS = {
    "magnetic_axis_distance_m": "m",
    "plasma_current_fraction": "of current",
    "axis_flux_fraction": "of span",
    "boundary_flux_fraction": "of span",
    "q95_difference": "q",
}


def render_table(records: list[dict[str, Any]]) -> str:
    """Render the scoreboard as an HTML table plus its pinned caption."""

    names = [entry["name"] for case in records for entry in case["estimators"]]
    columns: list[str] = []
    for name in names:
        if name not in columns:
            columns.append(name)
    width = len(STATE_SCOREBOARD_FIELDS) + 4
    lines = [
        '<section id="estimator-scoreboard-panel">',
        "<h2>Every reconstructed state, one comparator</h2>",
        "<p>Each row is a case read from a stored state; each column group is one "
        "estimator's state for that case, read through imas-python at the data "
        "dictionary version its own dataset was written at and identified by the "
        "digest printed in its last cell. Values are the state's distance from the "
        "case reference, marked pass or FAIL against the registered bands; the "
        "reference-free checks are reported beside them for every column, including "
        "where the reference comparison passes. A declared perturbed copy of the "
        "reference is scored as a third estimator from its own IMAS dataset, and a "
        "column whose states are not on disk reads <em>not supplied</em> rather than "
        "showing a zero.</p>",
        '<table id="estimator-scoreboard-table">',
        "<thead>",
        '<tr><th rowspan="2">case</th>'
        + "".join(f'<th colspan="{width}">{name}</th>' for name in columns)
        + "</tr>",
        "<tr>"
        + "".join(
            "<th>axis [m]</th><th>current</th><th>axis flux</th>"
            "<th>boundary flux</th><th>q95</th><th>axis in wall</th>"
            "<th>axis in boundary</th><th>boundary/bbox</th><th>digest</th>"
            for _ in columns
        )
        + "</tr>",
        "</thead>",
        "<tbody>",
    ]
    for case in records:
        by_name = {entry["name"]: entry for entry in case["estimators"]}
        cells = [f"<td><code>{case['case']}</code></td>"]
        for name in columns:
            entry = by_name.get(name)
            if entry is None or entry["status"] != "scored":
                cells.append(f'<td colspan="{width}">not supplied</td>')
                continue
            metrics = entry["metrics"]
            for field in STATE_SCOREBOARD_FIELDS:
                cells.append(
                    f"<td>{_format(metrics[field], UNITS[field])}"
                    f"<br/><small>{_mark(entry['verdicts'][field])}</small></td>"
                )
            checks = entry["reference_free"]
            cells.append(f"<td>{_mark(checks['axis_inside_wall'])}</td>")
            cells.append(f"<td>{_mark(checks['axis_inside_boundary'])}</td>")
            cells.append(
                "<td>"
                f"{_format(checks['boundary_area_fraction'], 'of bbox')}"
                f"<br/><small>{_mark(entry['reference_free_verdicts']['boundary_area_fraction'])}</small></td>"
            )
            cells.append(
                f"<td><code>{entry['state']['digest'][7:19]}</code><br/>"
                f"<small>dd {entry['state']['dd_version']}</small></td>"
            )
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines += ["</tbody>", "</table>", "</section>"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Score the manifest's cases and publish the table and its receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument(
        "--emit-perturbed",
        action="store_true",
        help="materialise declared perturbed states instead of scoring",
    )
    arguments = parser.parse_args(argv)
    manifest = load_manifest(arguments.manifest)
    if arguments.emit_perturbed:
        if arguments.state_dir is None:
            parser.error("--emit-perturbed requires --state-dir")
        written = emit_perturbed_states(manifest, arguments.state_dir)
        for path in written:
            print(path)
        return 0
    records = [score_case(case, arguments.state_dir) for case in manifest["cases"]]
    table = render_table(records)
    if arguments.html is not None:
        arguments.html.parent.mkdir(parents=True, exist_ok=True)
        arguments.html.write_text(table + "\n")
    if arguments.receipt is not None:
        arguments.receipt.parent.mkdir(parents=True, exist_ok=True)
        arguments.receipt.write_text(
            json.dumps(
                {
                    "cases": records,
                    "fields": list(STATE_SCOREBOARD_FIELDS),
                    "bands": {
                        field: {
                            "bound": tolerance.bound,
                            "direction": tolerance.direction.value,
                            "unit": tolerance.unit,
                            "evidence": tolerance.evidence,
                        }
                        for field, tolerance in STATE_SCOREBOARD_TOLERANCES.items()
                    },
                },
                indent=1,
                sort_keys=True,
            )
            + "\n"
        )
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
