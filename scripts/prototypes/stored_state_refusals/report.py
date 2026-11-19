"""Assemble the standing-case evidence fragment from the arm records."""

from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np

RUN = Path(sys.argv[1]).resolve()
ROOT = Path(__file__).resolve().parents[3]
FRAGMENT = (
    ROOT
    / "docs/evidence/fragments/forward-solve-near-stored-states"
    / "stored-state-refusals.html"
)
FIGURES = ROOT / "docs/figures/forward-solve-near-stored-states/stored-state-refusals"
SRC = "/nova/figures/forward-solve-near-stored-states/stored-state-refusals"

LABELS = (
    "mastu-stored-default",
    "mastu-stored-continuation",
    "mastu-displaced-default",
    "mastu-displaced-continuation",
    "tcv-stored-default",
    "iter-stored-default",
)
CASES = {
    "mastu": "MAST-U 45272 at 0.2 s",
    "tcv": "TCV workshop at 0.4 s",
    "iter": "ITER 135013 at 75 s",
}


def records():
    merged = {}
    # Runs are read oldest first so the newest measurement of a key wins: an
    # arm re-measured by a repair job replaces the arm it supersedes, and every
    # case an older run measured still survives.
    paths = sorted(
        RUN.glob("*/measurement.json"),
        key=lambda path: (path.stat().st_mtime, str(path)),
    )
    for path in paths:
        for key, value in json.loads(path.read_text()).items():
            if isinstance(value, dict):
                merged.setdefault(key, {})
                merged[key].update(value)
            else:
                merged[key] = value
    for label in LABELS:
        arm = merged.get(label)
        if not arm:
            continue
        for path in sorted(RUN.glob(f"*/comparison/*{label}-generating-failure.json")):
            failure = json.loads(path.read_text())
            arm["exception"] = failure.get("exception")
            arm["axis_admitted"] = failure.get("axis_admitted")
            arm["failure_message"] = failure.get("message")
            arm["trips"] = failure.get("trips")
            arm["finite_residual_entries"] = failure.get("finite_residual_entries")
            arm["last_finite_residual"] = failure.get("last_finite_residual")
            for name, value in history_fields(failure.get("history_path")).items():
                if arm.get(name) is None:
                    arm[name] = value
    return merged


def number(value, digits=6):
    if value is None:
        return "unavailable"
    return f"{float(value):.{digits}g}"


def history_fields(path):
    """Return the residuals a refused solve kept when it wrote no receipt."""
    if not path:
        return {}
    history_path = Path(str(path))
    if not history_path.is_file():
        return {}
    import pickle

    import nova.equilibrium.fixed_point  # noqa: F401 - resolves the pickle type

    try:
        with history_path.open("rb") as stream:
            history = pickle.load(stream)
        values = np.asarray(history.active_set_residuals, dtype=float)
    except OSError, AttributeError, ValueError, pickle.UnpicklingError:
        return {}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {}
    return dict(
        last_finite_residual=float(finite[-1]),
        lowest_recorded_residual=float(finite.min()),
        finite_residual_entries=int(finite.size),
    )


def arm_row(label, record):
    if not record:
        return (
            "<tr><td><code>" + label + '</code></td><td colspan="12">not run</td></tr>'
        )
    cells = [
        "<code>" + label + "</code>",
        html.escape(str(record.get("case"))),
        html.escape(str(record.get("seed_policy") or "-")),
        number(record.get("residual")),
        html.escape(str(record.get("residual_state") or "-")),
        number(record.get("last_finite_residual")),
        "-"
        if record.get("finite_residual_entries") is None
        else str(record["finite_residual_entries"]),
        "-"
        if record.get("axis_admitted") is None
        else ("yes" if record["axis_admitted"] else "no"),
        "-"
        if record.get("termination_reason") is None
        else str(record["termination_reason"]),
        "yes" if record.get("qualified") else "no",
        "yes" if record.get("converged") else "no",
        number(record.get("wall_seconds"), 5),
        html.escape(str(record.get("exception") or "-")),
    ]
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def copy_figures(records_map):
    """Copy the panels the fragment cites; the harness previews are not cited.

    figure.savefig(preview, dpi=100) writes one byte-identical PNG for every
    MAST-U arm while the SVGs differ, so a preview adds no evidence and is
    not published.
    """
    FIGURES.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        record = records_map.get(label) or {}
        if not record.get("terminal_flux"):
            continue
        svg = Path(record.get("panel", ""))
        if svg.is_file():
            shutil.copy2(svg, FIGURES / svg.name)


def figure(label, record, caption):
    if not record or not record.get("terminal_flux"):
        return ""
    name = Path(str(record.get("panel", ""))).name
    return (
        '<figure id="stored-state-refusals-'
        + label
        + '"><img alt="'
        + html.escape(caption)
        + '" width="100%" src="'
        + SRC
        + "/"
        + name
        + '"><figcaption>'
        + html.escape(caption)
        + "</figcaption></figure>"
    )


def mesh_of(records_map, case):
    if case == "mastu":
        return records_map.get("mesh") or {}
    return (records_map.get("meshes") or {}).get(case) or {}


def coefficients_of(records_map, case):
    stored = records_map.get("stored_coefficients") or {}
    return stored.get(case) + [] if isinstance(stored.get(case), list) else []


def case_block(records_map, case):
    mesh = mesh_of(records_map, case)
    route = "-"
    for label in LABELS:
        record = records_map.get(label) or {}
        if record.get("case") == case:
            route = (record.get("policy") or {}).get("route", "-")
            break
    lines = [
        "case=" + CASES[case],
        "mesh="
        + json.dumps({key: mesh.get(key) for key in ("kind", "nodes", "stencil_rows")}),
        "route=" + str(route) + " with the from_defaults policy",
        "coefficients=" + json.dumps(coefficients_of(records_map, case)),
    ]
    return lines


def policy_table(records_map):
    default = {}
    for label in LABELS:
        record = records_map.get(label) or {}
        if record.get("policy"):
            default = record["policy"]
            break
    control = (records_map.get("mastu-stored-continuation") or {}).get("policy") or {}
    keys = (
        "route",
        "newton_steps",
        "gmres_iterations",
        "active_set_steps",
        "kernel_tolerance",
        "qualification_tolerance",
    )
    rows = "".join(
        "<tr><td><code>"
        + key
        + "</code></td><td>"
        + str(default.get(key, "-"))
        + "</td><td>"
        + str(control.get(key, "-"))
        + "</td></tr>"
        for key in keys
        if key in default or key in control
    )
    return (
        "<table><thead><tr><th>policy field</th><th>default</th>"
        "<th>MAST-U continuation</th></tr></thead><tbody>" + rows + "</tbody></table>"
    )


def assemble(records_map):
    parts = ['<section id="stored-state-refusals-record">']
    parts.append("<h2>Forward-solve refusals beside stored imas-efit states</h2>")
    parts.append(
        "<p>Each arm starts the public forward solve from the stored imas-efit flux "
        "map with imas-efit's stored conductor currents and profile coefficients. "
        "The table gives every arm's terminal state.</p>"
    )
    parts.append(
        "<table><thead><tr><th>arm</th><th>case</th><th>seed</th>"
        "<th>residual</th><th>residual state</th>"
        "<th>last finite residual</th><th>finite entries</th>"
        "<th>axis admitted</th><th>reason</th>"
        "<th>qualified</th><th>converged</th><th>wall s</th>"
        "<th>failure</th></tr></thead><tbody>"
        + "".join(arm_row(label, records_map.get(label)) for label in LABELS)
        + "</tbody></table>"
    )
    for label in LABELS:
        record = records_map.get(label) or {}
        if not record.get("exception"):
            continue
        parts.append(
            "<p><code>"
            + label
            + "</code> raises <code>"
            + html.escape(str(record["exception"]))
            + "</code>: "
            + html.escape(str(record.get("failure_message")))
            + ". "
            + "The solve admitted no stationary point, so no magnetic axis was "
            + "admitted; the terminal residual is recorded as "
            + html.escape(str(record.get("residual_state") or "unavailable"))
            + " with "
            + str(record.get("finite_residual_entries"))
            + " finite entries and a last finite residual of "
            + number(record.get("last_finite_residual"))
            + "; termination "
            + "reason "
            + str(record.get("termination_reason"))
            + " over "
            + str(record.get("trips"))
            + " trips.</p>"
        )
    return "\n".join(parts)


def main():
    records_map = records()
    copy_figures(records_map)
    parts = [assemble(records_map)]
    parts.append(
        '<section id="stored-state-refusals-inputs">'
        "<h3>Inputs, mesh and route per solve</h3>"
    )
    for case in ("mastu", "tcv", "iter"):
        lines = case_block(records_map, case)
        parts.append(
            '<pre id="stored-state-refusals-'
            + case
            + '">'
            + html.escape("\n".join(lines))
            + "</pre>"
        )
    parts.append("</section>")
    displaced = records_map.get("displacement") or {}
    if displaced:
        parts.append(
            '<section id="stored-state-refusals-displacement">'
            "<h3>The displaced coordinate</h3><pre>"
            + html.escape(json.dumps(displaced, indent=1))
            + "</pre></section>"
        )
    parts.append(
        '<section id="stored-state-refusals-panels">'
        "<h3>Terminal states on the stored state</h3>"
    )
    for label in LABELS:
        record = records_map.get(label) or {}
        if not record.get("terminal_flux"):
            continue
        caption = (
            label
            + ": terminal residual "
            + number(record.get("residual"))
            + ", reason "
            + str(record.get("termination_reason"))
            + ", qualified "
            + str(record.get("qualified"))
        )
        parts.append(figure(label, record, caption))
    parts.append("</section>")
    parts.append(
        '<section id="stored-state-refusals-policy">'
        "<h3>The default policy beside the continuation policy</h3>"
    )
    parts.append(policy_table(records_map))
    parts.append("</section>")
    parts.append("</section>")
    FRAGMENT.write_text("\n".join(parts) + "\n")
    print("FRAGMENT", FRAGMENT)


if __name__ == "__main__":
    main()
