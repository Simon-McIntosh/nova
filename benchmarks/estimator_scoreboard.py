"""Publish one scoreboard for reconstructed equilibrium states.

Every column is a state supplied as an IMAS equilibrium, read at the data
dictionary version its own dataset was written at; every row is a case with a
reference state.  The comparator and the reference-free checks are the ones in
:mod:`nova.imas.mast_efit_referee`, and the bands are read from
:data:`nova.imas.parity_tolerances.STATE_SCOREBOARD_TOLERANCES`.

A new estimator joins by adding a column to the case manifest.  When that
estimator's states do not exist on disk yet, the manifest may declare a
perturbation of an existing state -- a deliberate defect the scoreboard must
rank worse -- or a derivation from a stored topology record and flux map, and
``--emit-states`` materialises either as its own IMAS equilibrium before
scoring, so the scoring path never sees a state that was not supplied as one.

Run the table and its receipt::

    python benchmarks/estimator_scoreboard.py --manifest cases.json \
        --html <fragment>.html --receipt <receipt>.json

Materialise the declared states first::

    python benchmarks/estimator_scoreboard.py --manifest cases.json \
        --state-dir <run directory>/states --emit-states
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
    q95_from_flux_map,
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
    "flux_map_scale": 1.1,
}


def load_manifest(path: Path | str) -> dict[str, Any]:
    """Load and shape-check the case manifest."""

    manifest = json.loads(Path(path).read_text())
    if "cases" not in manifest or not manifest["cases"]:
        raise ValueError(f"{path} carries no cases")
    for case in manifest["cases"]:
        if "id" not in case or "reference" not in case or "wall" not in case:
            raise ValueError(f"case {case!r} must name an id, a reference and a wall")
        if "vacuum_b0" not in case:
            raise ValueError(f"case {case['id']} names no vacuum_b0")
        if "source" not in case["reference"] and "out" not in case["reference"]:
            raise ValueError(f"case {case['id']} names no reference state")
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
    flux_map_scale: float = 1.0,
) -> Path:
    """Write a deliberately defective copy of one equilibrium as its own dataset.

    Every frame keeps its boundary shape and moves radially by ``radial_shift_m``
    about the machine axis, its plasma current and safety-factor profile are
    scaled, and its flux span is grown by ``flux_fraction`` at each end.  When
    the state carries a stored flux map, the map's own minor radius about its
    magnetic axis is scaled by ``flux_map_scale``: the safety factor the one
    calculator reads must move with the map, so a defect that reaches only the
    scalar q profile cannot leave that column unchanged.  The copy carries no
    wall: the case manifest names the machine description the reference-free
    checks read.
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
        signed_span = float(quantities.psi_boundary) - float(quantities.psi_axis)
        frame.boundary.outline.r = np.asarray(frame.boundary.outline.r, dtype=float) + (
            radial_shift_m
        )
        quantities.magnetic_axis.r = float(quantities.magnetic_axis.r) + radial_shift_m
        if np.isfinite(span) and abs(span) < EMPTY_SENTINEL:
            axis_old = float(quantities.psi_axis)
            quantities.psi_axis = axis_old + flux_fraction * span
            quantities.psi_boundary = (
                float(quantities.psi_boundary) - flux_fraction * span
            )
            # the stored arrays are rescaled by the ratio of the boundary-minus-axis
            # spans, both signed: dividing by the absolute span would reflect a map
            # whose flux falls outward, which no perturbation declares
            stretch = (
                (float(quantities.psi_boundary) - float(quantities.psi_axis))
                / signed_span
                if signed_span != 0.0
                else 1.0
            )
            for stored in frame.profiles_2d:
                stored.psi = (
                    float(quantities.psi_axis)
                    + (np.asarray(stored.psi, dtype=float) - axis_old) * stretch
                )
            profile_psi = np.asarray(frame.profiles_1d.psi, dtype=float)
            if profile_psi.size:
                frame.profiles_1d.psi = (
                    float(quantities.psi_axis) + (profile_psi - axis_old) * stretch
                )
        quantities.ip = float(quantities.ip) * current_scale
        profile = frame.profiles_1d
        profile.q = np.asarray(profile.q, dtype=float) * q_scale
        if flux_map_scale != 1.0:
            for stored in frame.profiles_2d:
                axis_r = float(quantities.magnetic_axis.r)
                stored.grid.dim1 = (
                    axis_r
                    + (np.asarray(stored.grid.dim1, dtype=float) - axis_r)
                    * flux_map_scale
                )
    writer.put(equilibrium)
    writer.close()
    reader.close()
    return out


def _loop_area(points: np.ndarray) -> float:
    """Return the signed area enclosed by one closed polygon."""

    radius = points[:, 0]
    height = points[:, 1]
    return 0.5 * float(
        np.sum(radius * np.roll(height, -1) - np.roll(radius, -1) * height)
    )


def map_boundary(flux_map: Path | str, level: float) -> np.ndarray:
    """Return the closed contour of a stored flux map at one flux level.

    The map carries ``radius`` and ``height`` axes and a ``flux`` array indexed
    by them.  The contour at ``level`` is traced by marching squares, and the
    closed loop enclosing the largest area is returned, so a level crossed by
    the separatrix returns the bounding surface rather than a private lobe.
    """

    stored = np.load(Path(flux_map).expanduser(), allow_pickle=False)
    radius = np.asarray(stored["radius"], dtype=float)
    height = np.asarray(stored["height"], dtype=float)
    flux = np.asarray(stored["flux"], dtype=float)
    if flux.shape != (radius.size, height.size):
        raise ValueError(
            f"the flux map carries shape {flux.shape}; the axes index "
            f"{(radius.size, height.size)}"
        )
    import contourpy

    generator = contourpy.contour_generator(
        x=radius,
        y=height,
        z=flux.T,
        name="serial",
        line_type=contourpy.LineType.Separate,
    )
    lines = [np.asarray(line, dtype=float) for line in generator.lines(float(level))]
    closed = [
        line for line in lines if line.shape[0] > 3 and np.allclose(line[0], line[-1])
    ]
    candidates = closed or [line for line in lines if line.shape[0] > 3]
    if not candidates:
        raise ValueError(f"the flux map at {flux_map} crosses no level {level}")
    return max(candidates, key=lambda line: abs(_loop_area(line)))


def state_from_record(
    record: Path | str,
    flux_map: Path | str,
    destination: Path | str,
    *,
    section: str = "state",
    dd_version: str = "4.1.0",
    edge_psi_norm: float = 0.995,
    amplitude: float | None = None,
    b0: float | None = None,
    reference_radius: float | None = None,
) -> Path:
    """Write one state from a stored topology record and flux map.

    The record holds the state's magnetic axis, flux axis and boundary, plasma
    current and time; the flux map holds the poloidal flux on a uniform grid
    from which the bounding surface is traced.  The outline is taken at
    ``edge_psi_norm`` of the recorded span for the reason the flux-surface
    geometry record takes its outermost surface there: the contour exactly at
    the boundary flux of a diverted state joins the divertor legs, so the
    polygon spans the whole grid rather than the plasma.  Nothing is solved:
    the written state is the scalars and arrays the record carries and the
    outline the map crosses just inside its recorded boundary.

    The map the outline was traced on is written with the state, on the
    ``profiles_2d`` lattice the geometry read indexes, so the state carries the
    flux map its safety factor is scored from rather than only a boundary.  A
    declared ``amplitude`` writes that flux function's diamagnetic gradient
    ``F dF/dpsi = amplitude (1 - psi_N)`` on the record's own flux nodes, and
    ``b0`` writes the case's vacuum field on the axis.  A state written without
    ``amplitude`` carries no field data, so its safety factor reads as not
    supplied until the record declares one.
    """

    import imas
    from imas.ids_defs import IDS_TIME_MODE_HOMOGENEOUS

    stored_record = json.loads(Path(record).read_text())
    if section not in stored_record:
        raise ValueError(f"the record at {record} carries no {section!r} state")
    state = stored_record[section]
    span = float(state["flux_boundary"]) - float(state["flux_axis"])
    level = float(state["flux_axis"]) + float(edge_psi_norm) * span
    outline = map_boundary(flux_map, level)
    if not np.allclose(outline[0], outline[-1]):
        outline = np.vstack([outline, outline[0]])
    stored_map = np.load(Path(flux_map).expanduser(), allow_pickle=False)
    lattice_radius = np.asarray(stored_map["radius"], dtype=float)
    lattice_height = np.asarray(stored_map["height"], dtype=float)
    map_flux = np.asarray(stored_map["flux"], dtype=float)
    if map_flux.shape != (lattice_radius.size, lattice_height.size):
        raise ValueError(
            f"the flux map at {flux_map} carries shape {map_flux.shape} on axes "
            f"{(lattice_radius.size, lattice_height.size)}"
        )

    equilibrium = imas.IDSFactory().equilibrium()
    equilibrium.ids_properties.homogeneous_time = IDS_TIME_MODE_HOMOGENEOUS
    equilibrium.time = np.array([float(stored_record.get("time", 0.0))])
    equilibrium.time_slice.resize(1)
    frame = equilibrium.time_slice[0]
    frame.boundary.outline.r = outline[:, 0]
    frame.boundary.outline.z = outline[:, 1]
    frame.boundary.type = 1
    quantities = frame.global_quantities
    quantities.magnetic_axis.r = float(state["axis"][0])
    quantities.magnetic_axis.z = float(state["axis"][1])
    quantities.psi_axis = float(state["flux_axis"])
    quantities.psi_boundary = float(state["flux_boundary"])
    quantities.ip = float(state["plasma_current"])
    if amplitude is not None:
        norm = np.linspace(0.0, 1.0, lattice_radius.size)
        frame.profiles_1d.psi = float(state["flux_axis"]) + span * norm
        frame.profiles_1d.f_df_dpsi = float(amplitude) * (1.0 - norm)
    if reference_radius is None:
        reference_radius = state.get("reference_radius")
    if b0 is not None and reference_radius is not None:
        equilibrium.vacuum_toroidal_field.b0 = np.array([-abs(float(b0))])
        equilibrium.vacuum_toroidal_field.r0 = float(reference_radius)
    frame.profiles_2d.resize(1)
    stored_profile = frame.profiles_2d[0]
    stored_profile.grid.dim1 = lattice_radius
    stored_profile.grid.dim2 = lattice_height
    stored_profile.grid_type.index = 1
    stored_profile.psi = map_flux
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    writer = imas.DBEntry(f"imas:hdf5?path={out.resolve()}", "w", dd_version=dd_version)
    writer.put(equilibrium)
    writer.close()
    return out


def emit_perturbed_states(
    manifest: dict[str, Any], state_dir: Path | str
) -> list[Path]:
    """Materialise every manifest column that declares a perturbed state."""

    written: list[Path] = []
    for case in manifest["cases"]:
        for entry in case.get("estimators", []):
            if "out" not in entry or "from_record" in entry:
                continue
            if state_dir is None:
                raise ValueError(
                    f"case {case['id']} declares a generated state but no "
                    "--state-dir was supplied"
                )
            perturbation = dict(DEFAULT_PERTURBATION)
            perturbation.update(entry.get("perturb", {}))
            origin = entry.get("perturbed_of", "reference")
            if origin == "reference":
                source = _state_source(case["reference"], state_dir)
            else:
                matches = [e for e in case["estimators"] if e["name"] == origin]
                source = _state_source(matches[0], state_dir) if matches else None
            if source is None:
                raise ValueError(f"case {case['id']} cannot perturb {origin!r}")
            written.append(
                perturb_state(
                    source,
                    Path(state_dir) / entry["out"],
                    dd_version=str(case["reference"].get("dd_version", "4.1.0")),
                    **perturbation,
                )
            )
    return written


def emit_declared_states(manifest: dict[str, Any], state_dir: Path | str) -> list[Path]:
    """Materialise every manifest column that declares its state.

    A column is materialised either as a perturbation of an existing state or
    from a stored topology record and flux map; both are written as their own
    IMAS datasets so the scoring path reads only IMAS equilibria.
    """

    written: list[Path] = []
    for case in manifest["cases"]:
        derived_entries = [case["reference"]]
        derived_entries.extend(case.get("estimators", []))
        for entry in derived_entries:
            if "out" not in entry or "from_record" not in entry:
                continue
            derived = entry["from_record"]
            written.append(
                state_from_record(
                    derived["record"],
                    derived["map"],
                    Path(state_dir) / entry["out"],
                    section=derived.get("section", "state"),
                    dd_version=str(derived.get("dd_version", "4.1.0")),
                    amplitude=derived.get("amplitude"),
                    b0=case.get("vacuum_b0"),
                    reference_radius=derived.get("reference_radius"),
                )
            )
    written.extend(emit_perturbed_states(manifest, state_dir))
    return written


def score_case(case: dict[str, Any], state_dir: Path | None) -> dict[str, Any]:
    """Score every column of one case against its reference state."""

    reference_source = _state_source(case["reference"], state_dir)
    if reference_source is None:
        raise ValueError(f"case {case['id']} names no reference state")
    reference = read_equilibrium_state(
        reference_source,
        occurrence=int(case["reference"].get("occurrence", 0)),
        time_index=case["reference"].get("time_index"),
        time_s=case["reference"].get("time_s"),
        wall_source=case["wall"],
        q95=case["reference"].get("q95"),
    )
    vacuum_b0 = case.get("vacuum_b0")
    if vacuum_b0 is None:
        raise ValueError(f"case {case['id']} names no vacuum_b0 for the calculator")
    reference_q95 = q95_from_flux_map(reference, vacuum_b0=float(vacuum_b0))
    record: dict[str, Any] = {
        "case": case["id"],
        "machine": case.get("machine", ""),
        "reference": _state_record(reference),
        "reference_q95_flux_map": reference_q95,
        "estimators": [],
    }
    for entry in case.get("estimators", []):
        source = _state_source(entry, state_dir)
        if source is None or (
            not source.startswith("imas:") and not Path(source).exists()
        ):
            absent = {
                "name": entry["name"],
                "status": "absent",
                "reason": "no state supplied for this case",
            }
            if "note" in entry:
                absent["note"] = entry["note"]
            record["estimators"].append(absent)
            continue
        state = read_equilibrium_state(
            source,
            occurrence=int(entry.get("occurrence", 0)),
            time_index=entry.get("time_index"),
            time_s=entry.get("time_s"),
            wall_source=case["wall"],
            q95=entry.get("q95"),
        )
        state_q95 = q95_from_flux_map(state, vacuum_b0=float(vacuum_b0))
        comparison = compare_states(
            reference,
            state,
            q95_values=(reference_q95, state_q95),
        )
        checks = reference_free_checks(state)
        scored = {
            "name": entry["name"],
            "status": "scored",
            "state": _state_record(state),
            "gauge_wb": comparison.gauge_wb,
            "metrics": comparison.metrics,
            "verdicts": dict(comparison.verdicts()),
            "reference_free": checks.values,
            "q95_flux_map": state_q95,
            "q95_stored": float(state.q95),
            "reference_free_verdicts": dict(checks.verdicts()),
            "lcfs_distance_m": comparison.lcfs_distance_m,
        }
        if "note" in entry:
            scored["note"] = entry["note"]
        record["estimators"].append(scored)
    if "note" in case.get("reference", {}):
        record["reference"]["note"] = case["reference"]["note"]
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
        "flux_map_nodes": (
            None
            if state.flux_map is None
            else [
                int(state.flux_map.radius.size),
                int(state.flux_map.height.size),
            ]
        ),
        "reference_radius": (
            None if state.flux_map is None else float(state.flux_map.reference_radius)
        ),
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


def _q95_cell(entry: dict[str, Any]) -> str:
    """Render the calculator's q95 difference with the state's stored value beside.

    A state that carries no flux map reads *not supplied*: no scalar record
    stands in for the map the calculator reads.  Where the calculator returns a
    value, the state's own stored safety factor is printed beside it as a
    definition difference, because the two codes store q on different surfaces.
    """

    calculated = entry.get("q95_flux_map")
    stored = entry.get("q95_stored")
    stored_text = (
        "stored not supplied"
        if stored is None or not np.isfinite(stored)
        else f"stored {abs(float(stored)):.3f}"
    )
    if calculated is None or not np.isfinite(calculated):
        return f"<td>not supplied<br/><small>no flux map; {stored_text}</small></td>"
    return (
        f"<td>{_format(entry['metrics']['q95_difference'], 'q')}"
        f"<br/><small>{_mark(entry['verdicts']['q95_difference'])}; "
        f"this state {abs(float(calculated)):.3f}; {stored_text}</small></td>"
    )


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
        "where the reference comparison passes. The safety factor is read for "
        "every column by one calculator, "
        "<code>nova.imas.mast_efit_referee.q95_from_flux_map</code>, applied to "
        "that column's own stored flux map on one surface definition "
        "(<code>&#968;<sub>N</sub> = 0.95</code>), so the q95 column compares "
        "two reconstructions rather than two stored definitions; each state's "
        "own stored safety factor is printed beside it as that definition "
        "difference, and a state carrying no flux map reads <em>not supplied</em>. "
        "A declared perturbation is scored as "
        "its own column from its own IMAS dataset, and a column whose state is not on "
        "disk reads <em>not supplied</em> rather than showing a zero. The list below "
        "names, for each column, the state it reads and how that state was formed; the "
        "receipt beside this table carries each state's digest.</p>",
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
                if field == "q95_difference":
                    cells.append(_q95_cell(entry))
                    continue
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
    lines += ["</tbody>", "</table>"]
    notes = [("reference", records[0]["reference"].get("note"))]
    notes.extend(
        (entry["name"], entry.get("note")) for entry in records[0]["estimators"]
    )
    described = [(name, note) for name, note in notes if note]
    if described:
        lines.append('<dl class="state-sources">')
        for name, note in described:
            lines.append(f"<dt>{name}</dt><dd>{note}</dd>")
        lines.append("</dl>")
    lines.append("</section>")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Score the manifest's cases and publish the table and its receipt."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument(
        "--emit-states",
        "--emit-perturbed",
        dest="emit_states",
        action="store_true",
        help="materialise the declared states instead of scoring",
    )
    arguments = parser.parse_args(argv)
    manifest = load_manifest(arguments.manifest)
    if arguments.emit_states:
        if arguments.state_dir is None:
            parser.error("--emit-states requires --state-dir")
        written = emit_declared_states(manifest, arguments.state_dir)
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
