"""Summarise durable refinement receipts without filling missing rows."""

# Markdown table rows are kept readable as single source lines.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MAJOR_RADIUS = {"limited": 6.2, "diverted": 1.7}
MEMBERSHIP_COEFFICIENT = {"limited": 0.274714, "diverted": 2.1805365680219544}
COUNTS = (550, 2000, 5000, 10000)


def _row(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _number(value, digits: int = 3) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}g}"


def _read_status(kind: str, row: dict) -> str:
    if not row["valid"]:
        return f"invalid ({row['reason']})"
    if not row["qualified"]:
        return "unqualified"
    if kind == "diverted" and row["x_error_pitches"] is None:
        return "no admitted X"
    return "admitted"


def _fit(rows: list[dict], key: str) -> str:
    pairs = [
        (row["pitch_m"], row[key])
        for row in rows
        if row.get(key) is not None and np.isfinite(row[key]) and row[key] > 0
    ]
    if len(pairs) < 2:
        return "—"
    pitch, error = map(np.asarray, zip(*pairs, strict=True))
    return f"{np.polyfit(np.log(pitch), np.log(error), 1)[0]:.3f} ({len(pairs)} rows)"


def _first(rows: list[dict], predicate) -> str:
    for row in sorted(
        rows, key=lambda item: item.get("cells", item.get("realised_cells", 0))
    ):
        if predicate(row):
            return str(row.get("cells", row.get("realised_cells")))
    return "none measured"


def build(rows_root: Path, map_root: Path, report: Path) -> None:
    quadratic_root = rows_root.parent / "quadratic-rows"
    rows = {
        (kind, cells, arm): _row(
            (rows_root / f"{kind}-{cells}-A.json")
            if arm == "A"
            else (quadratic_root / f"{kind}-{cells}.json")
        )
        for kind in MAJOR_RADIUS
        for cells in COUNTS
        for arm in "AB"
    }
    map_rows = {
        (kind, cells): _row(map_root / f"{kind}-{cells}.json")
        for kind in MAJOR_RADIUS
        for cells in COUNTS
    }
    baseline = {kind: _row(rows_root / f"{kind}-132-A.json") for kind in MAJOR_RADIUS}
    lines = [
        "# Forward map and topology resolution ladder",
        "",
        "## Method and controls",
        "",
        "The topology rows evaluate the closed-form analytic field through "
        "`nova.equilibrium.topology.read` on exact-count hex carriers. "
        "Arm A uses `TopologyPolicy` defaults with the calibrated physical "
        "normal-form radius. Arm B sets that radius and its pitch floor to zero, "
        "zeros the third and fourth curvature derivatives, and switches off "
        "saddle normal-form support through a source-checked prototype flag. "
        "The forward-map rows "
        "use the certificate machine, exact exterior and exact clip. The "
        "prototype changes only topology, so the map value is common to A and B. "
        "These are separate carriers, and any map row with a different realised "
        "count is identified below. No forward solve was run.",
        "",
        "Every measured row below names its SLURM job and its own log. Cold "
        "compiles start in fresh processes with the JAX persistent compilation "
        "cache disabled before import; warm execution is the shorter of two "
        "executions of that compiled program. Device memory is the executable's "
        "temporary allocation estimate, not a live device peak.",
        "",
        "| 132-cell control | Stored smooth | Reproduced smooth | Stored saddle | Reproduced saddle | Job / log |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for kind, row in baseline.items():
        if row is None:
            lines.append(f"| {kind} | — | — | — | — | missing |")
        else:
            log = rows_root.parent / "logs" / f"{kind}-132-A.log"
            lines.append(
                f"| {kind} | {_number(row['baseline_smooth_error'])} | "
                f"{_number(row['reproduced_smooth_error'])} | "
                f"{_number(row['baseline_saddle_error'])} | "
                f"{_number(row['reproduced_saddle_error'])} | "
                f"{row['job_id']} `{log}` |"
            )
    lines += [
        "",
        "## Measured rows",
        "",
        "Membership errors are absolute area-fraction differences in median-cell "
        "area units; the smooth and saddle sets are split at the larger of the "
        "physical normal-form radius and 1.5 pitches. `Nmap` is the certificate "
        "carrier's realised count. `map` is relative sup at the analytic state. "
        "`axis` and `X` show metres / pitches; a missing X on a diverted row "
        "means no X-point was admitted. `resid` and `solve` are absent "
        "because this run measures the map and read, not a solved state. "
        "Read reason 7 is `UNRESOLVED_COMPONENT`.",
        "",
        "| Case | N | Arm | Read | Nmap | map | smooth | saddle | axis m / h | X m / h | contact m | resid | solve | cold s | warm s | GPU temp GB | RSS GB | Job / log |",
        "|---|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for kind in MAJOR_RADIUS:
        for cells in COUNTS:
            mapped = map_rows[kind, cells]
            for arm in "AB":
                row = rows[kind, cells, arm]
                if row is None:
                    lines.append(
                        f"| {kind} | {cells} | {arm} | — | — | — | — | — | — | — | — | — | — | — | — | — | — | — | missing |"
                    )
                    continue
                top_log = (
                    rows_root.parent
                    / "logs"
                    / (
                        f"{kind}-{cells}-A.log"
                        if arm == "A"
                        else f"quadratic-{kind}-{cells}.log"
                    )
                )
                map_log = map_root.parent / "logs" / f"map-{kind}-{cells}.log"
                map_label = (
                    f"{mapped['job_id']} `{map_log}`" if mapped else "map pending"
                )
                lines.append(
                    f"| {kind} | {cells} | {arm} | {_read_status(kind, row)} | "
                    f"{mapped['realised_cells'] if mapped else '—'} | "
                    f"{_number(mapped['map_relative_sup']) if mapped else '—'} | "
                    f"{_number(row['smooth_membership_error'])} | "
                    f"{_number(row['saddle_membership_error'])} | "
                    f"{_number(row['axis_error_m'])} / {_number(row['axis_error_pitches'])} | "
                    f"{_number(row['x_error_m'])} / {_number(row['x_error_pitches'])} | "
                    f"{_number(row['limiter_contact_error_m'])} | — | — | "
                    f"{_number(row['cold_compile_seconds'])} | "
                    f"{_number(row['warm_execute_seconds'])} | "
                    f"{_number(row['device_temp_bytes'] / 1e9)} | "
                    f"{_number(row['host_peak_rss_kib'] / 1048576)} | "
                    f"{row['job_id']} `{top_log}`; {map_label} |"
                )
    lines += [
        "",
        "### Owner-cell straight-ray control",
        "",
        "This intermediate arm zeros the radius and curvature corrections "
        "through `TopologyPolicy` and a prototype flag, but retains straight-ray "
        "normal-form support in the saddle owner cell. It is not arm B.",
        "",
        "| Case | N | smooth | saddle | owner cells | Job / log |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for kind in MAJOR_RADIUS:
        for cells in COUNTS:
            control = _row(rows_root / f"{kind}-{cells}-B.json")
            if control is None:
                continue
            log = rows_root.parent / "logs" / f"{kind}-{cells}-B.log"
            lines.append(
                f"| {kind} | {cells} | "
                f"{_number(control['smooth_membership_error'])} | "
                f"{_number(control['saddle_membership_error'])} | "
                f"{control['normal_form_owner_cells']} | "
                f"{control['job_id']} `{log}` |"
            )
    lines += [
        "",
        "## Fitted orders and first measured acceptance",
        "",
        "Orders are least-squares slopes of log(error) against log(pitch). "
        "The membership bound is the section's measured coefficient "
        "`C (pitch / major radius)^2`, with C=0.274714 / 2.180536568 "
        "for limited / diverted. "
        "The map bound is relative sup ≤0.01; position bound is one pitch. "
        "Cold compile ≤60 s and host RSS ≤64 GB are production gates, so a "
        "pass in this closed-form prototype does not qualify the kernel-backed solve.",
        "",
        "| Case | Arm | map order | smooth order | saddle order | axis order | X order | first membership | first position | first map | first compile | first RSS |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kind in MAJOR_RADIUS:
        mapped = [map_rows[kind, cells] for cells in COUNTS if map_rows[kind, cells]]
        for arm in "AB":
            group = [
                rows[kind, cells, arm] for cells in COUNTS if rows[kind, cells, arm]
            ]

            def bound(row):
                return (
                    MEMBERSHIP_COEFFICIENT[kind]
                    * (row["pitch_m"] / MAJOR_RADIUS[kind]) ** 2
                )

            membership = _first(
                group,
                lambda row: (
                    row["valid"]
                    and row["qualified"]
                    and (kind == "limited" or row["x_error_pitches"] is not None)
                    and row["smooth_membership_error"] <= bound(row)
                    and (
                        row["saddle_membership_error"] is None
                        or row["saddle_membership_error"] <= bound(row)
                    )
                ),
            )
            position = _first(
                group,
                lambda row: (
                    row["valid"]
                    and row["qualified"]
                    and row["axis_error_pitches"] <= 1
                    and (
                        (row["x_error_pitches"] is None and kind == "limited")
                        or (
                            row["x_error_pitches"] is not None
                            and row["x_error_pitches"] <= 1
                        )
                    )
                ),
            )
            lines.append(
                f"| {kind} | {arm} | {_fit(mapped, 'map_relative_sup')} | "
                f"{_fit(group, 'smooth_membership_error')} | "
                f"{_fit(group, 'saddle_membership_error')} | "
                f"{_fit(group, 'axis_error_m')} | {_fit(group, 'x_error_m')} | "
                f"{membership} | {position} | "
                f"{_first(mapped, lambda row: row['map_relative_sup'] is not None and row['map_relative_sup'] <= 0.01)} | "
                f"{_first(group, lambda row: row['cold_compile_seconds'] <= 60)} | "
                f"{_first(group, lambda row: row['host_peak_rss_kib'] <= 64 * 1048576)} |"
            )
    lines += [
        "",
        "## Binding cost and verdict",
        "",
        "Pending manual interpretation of the complete row receipts and any "
        "timed-out or failed rows. The forward-solve residual, constraint rows, "
        "executable size and tracing gates were not measured by this prototype.",
        "",
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--map-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(args.rows, args.map_rows, args.out)


if __name__ == "__main__":
    main()
