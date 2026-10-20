"""Join completed row receipts into the compile ledger and landing record."""

# ruff: noqa: E501

import argparse
from html import escape
import json
from pathlib import Path

from scripts.prototypes.kernel_compile_census.run_job import KERNELS, PROGRAMS


ROOT = Path(__file__).resolve().parents[3]
START = "<!-- kernel census start -->"
END = "<!-- kernel census end -->"


def receipt(directories, program, size):
    for directory in directories:
        path = directory / f"{program}-{size}.json"
        if path.is_file():
            return json.loads(path.read_text()), path
    return {}, None


def value(row, key, digits=0):
    item = row.get(key)
    if item is None:
        return "—"
    return f"{item:,.{digits}f}" if isinstance(item, float) else f"{item:,}"


def peak(row):
    observed = max(
        row.get("sampled_peak_resident_kib") or 0,
        row.get("peak_host_rss_kib") or 0,
    )
    return f"{observed / 1048576:.1f}" if observed else "—"


def status(row):
    if not row:
        return "not yet measured"
    if row.get("phase") == "compiled" and row.get("process_exit", 0) == 0:
        misses = row.get("cache_guard_misses")
        hits = row.get("cache_guard_hits")
        return f"compiled; {misses} miss, {hits} hits"
    if row.get("error"):
        return escape(row["error"])
    if row.get("process_exit") == -9:
        return (
            f"killed during compile after {row.get('elapsed_seconds', 0):.0f} s "
            "at memory ceiling"
        )
    if row.get("process_exit") == -15:
        return f"terminated after {row.get('elapsed_seconds', 0):.0f} s"
    if row.get("process_exit") is None and row.get("phase") == "lowered":
        return "lowered; compile still in progress or interrupted"
    return escape(str(row.get("phase", "no receipt")))


def table_rows(cpu_dirs, gpu_dirs):
    rows = []
    for program in PROGRAMS:
        sizes = (4, 64) if program in KERNELS else (132, 550)
        sources = cpu_dirs[:1] if program == "packed_analytic_moments" else cpu_dirs
        pair = [receipt(sources, program, size)[0] for size in sizes]
        first, second = pair
        a, b = first.get("expanded_equations"), second.get("expanded_equations")
        hlo_a, hlo_b = first.get("stablehlo_bytes"), second.get("stablehlo_bytes")
        graph_class = (
            "constant"
            if a
            and b
            and hlo_a
            and hlo_b
            and abs(b - a) / a < 0.05
            and hlo_b / hlo_a < 1.2
            else "growing"
            if a and b and hlo_a and hlo_b
            else "unresolved"
        )
        for size, cpu in zip(sizes, pair, strict=True):
            gpu, _ = receipt(gpu_dirs, program, size)
            row = [
                escape(program),
                str(size),
                graph_class,
                value(cpu, "unique_equations"),
                value(cpu, "expanded_equations"),
                value(cpu, "stablehlo_bytes"),
                value(cpu, "executable_bytes"),
                value(cpu, "cold_compile_wall_seconds", 2),
                peak(cpu),
                status(cpu),
                escape(str(gpu.get("device", "—"))),
                value(gpu, "cold_compile_wall_seconds", 2),
                peak(gpu),
                value(gpu, "executable_bytes"),
                status(gpu),
            ]
            rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return "\n".join(rows)


def source_rows(cpu_dirs):
    point, point_path = receipt(cpu_dirs, "point_jet", 4)
    denominator = point.get("expanded_equations")
    if not denominator:
        return "<p>Point jet source attribution has not completed.</p>"
    rows = []
    for source in point["source_function_ranking"]:
        owner = f"{source['file']}:{source['line']}"
        count = source["expanded_equations"]
        rows.append(
            "<tr><td><code>"
            + escape(owner)
            + "</code></td><td><code>"
            + escape(source["function"])
            + "</code></td><td>"
            + f"{count:,}</td><td>{100 * count / denominator:.2f}%</td></tr>"
        )
    leaders = [
        item
        for item in point["source_function_ranking"]
        if item["file"] != "<unattributed>"
    ][:3]
    summary = ", ".join(
        f"<code>{escape(item['function'])}</code> "
        f"({100 * item['expanded_equations'] / denominator:.2f}%)"
        for item in leaders
    )
    return (
        f"<p>The point jet denominator is {denominator:,} expanded equations. "
        "Every equation is counted once; the unattributed bucket is retained where "
        "JAX supplied no project source frame. The full ranking below is derived "
        f"from <code>{escape(str(point_path))}</code>. "
        f"The leading attributed functions are {summary}.</p>\n"
        "<table><thead><tr><th>Owner file and line</th><th>Function</th>"
        "<th>Expanded equations</th><th>Share of point jet</th></tr></thead><tbody>\n"
        + "\n".join(rows)
        + "\n</tbody></table>"
    )


def other_sources(cpu_dirs):
    groups = []
    for program in PROGRAMS:
        if program == "point_jet":
            continue
        size = 4 if program in KERNELS else 132
        sources = cpu_dirs[:1] if program == "packed_analytic_moments" else cpu_dirs
        data, path = receipt(sources, program, size)
        ranking = data.get("source_function_ranking", [])
        if not ranking:
            size = 64 if program in KERNELS else 550
            data, path = receipt(sources, program, size)
            ranking = data.get("source_function_ranking", [])
        rows = []
        for source in ranking:
            rows.append(
                "<tr><td><code>"
                + escape(f"{source['file']}:{source['line']}")
                + "</code></td>"
                "<td><code>" + escape(source["function"]) + "</code></td>"
                f"<td>{source['expanded_equations']:,}</td></tr>"
            )
        groups.append(
            "<details><summary>"
            + escape(program)
            + f" ({len(rows)} owners)"
            + "</summary><p>Full receipt: <code>"
            + escape(str(path))
            + "</code>.</p><table><thead><tr><th>Owner file and line</th>"
            "<th>Function</th><th>Expanded equations</th></tr></thead><tbody>\n"
            + "\n".join(rows)
            + "\n</tbody></table></details>"
        )
    return "\n".join(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    cpu_dirs = [args.run_dir / "cpu-repair", args.run_dir / "cpu"]
    gpu_dirs = [
        args.run_dir / "gpu-retry" / "h200",
        args.run_dir / "gpu-retry" / "titan",
    ]
    fragment = (
        ROOT
        / "docs/evidence/fragments/kernel-compile-barrier/kernel-compile-census.html"
    )
    section = f"""
{START}
<h2 id="kernel-compile-census">Kernel-backed compile census</h2>
<p>Every row uses a fresh process with the persistent compilation cache disabled before JAX initialises. Because that setting bypasses JAX's persistent-cache hooks, the driver adds its observed backend compile to the cache ledger as one unpersisted miss; no row with a hit is called cold. Host memory is sampled every ten seconds and combined with the child process's peak RSS for short rows. A dash means no executable or measurement was produced. The CPU topology compile was killed at the 450 GB allocation limit; its lower and sampled peak remain evidence, and the row does not claim an executable. H200 was pending for resources after a real submission; the GPU retry therefore ran on titan, whose P100 identity stays explicit in its row receipts. A graph is classed constant when expanded equations differ by less than 5% and StableHLO bytes grow by less than 20% across the two sizes.</p>
<p>The standalone point and packed kernels map over 4 and 64 target points around one hexagonal source. Harmonic moments use 43 orders; pole moments use 10 requested orders with a 43-entry source family; harmonic multiplication uses two nine-term series. Production and prototype row programs batch three and four points respectively over 132- and 550-cell source couplings. The read uses the limited analytic field and realised hex carrier; the certificate map uses the limited certified carrier and booked moments.</p>
<p>The earlier 1.25-million-equation, 39 MB topology record measured a within-scope scalar-scan arm, while these rows trace the current <code>topology.read</code>; the program identities differ, so the smaller current count does not contradict that receipt. The earlier 7,044 s and 381 GB jets build compiled at 2,132 nodes; these rows batch three or four points with 132 or 550 source cells, so the wall and memory figures are different shapes. The earlier 83.6 s point-jet wall is an H200 measurement; titan's P100 wall is reported under its own device name.</p>
<p>The original packed-moment probe evaluated one point even when labelled 4 or 64, so those two receipts are excluded from the comparison pending the batched repair. The 132-cell certificate carrier was absent from the published fixture store; its row remains explicit until the in-memory carrier build completes.</p>
<table><thead><tr><th>Program</th><th>Scale</th><th>Graph</th><th>Unique eqns</th><th>Expanded eqns</th><th>StableHLO bytes</th><th>CPU executable bytes</th><th>CPU cold s</th><th>CPU peak GiB</th><th>CPU outcome</th><th>GPU device</th><th>GPU cold s</th><th>GPU peak GiB</th><th>GPU executable bytes</th><th>GPU outcome</th></tr></thead><tbody>
{table_rows(cpu_dirs, gpu_dirs)}
</tbody></table>
<h3>Point jet source ranking</h3>
{source_rows(cpu_dirs)}
<h3>Source rankings of the other programs</h3>
<p>Each program's complete ranking is ordered by expanded equations and can be opened below. Where the smaller size did not reach graph tracing, the larger size supplies its owner ranking.</p>
{other_sources(cpu_dirs)}
{END}
"""
    ledger = ROOT / "docs/research/compile-cost-evidence.html"
    content = ledger.read_text()
    if START in content:
        before, tail = content.split(START, 1)
        _, after = tail.split(END, 1)
        updated = before + section + after
    else:
        updated = content.replace("</main>", section + "\n</main>")
    ledger.write_text(updated)
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(
        '<section id="kernel-compile-census-landing">\n'
        "<h3>Kernel compile census</h3>\n"
        "<p>The compile ledger carries the two-size graph counts, cold compile receipts, "
        "host memory and traceback-derived source ranking. The point jet measured "
        "106,218 expanded equations on CPU; rangefunction._harmonic_pair_step "
        "owns 24,653 of them (23.21%). The read lowered at both sizes before "
        "the 450 GB CPU allocation killed compilation. GPU rows and any repaired "
        "certificate row are reported in the ledger with their measured outcome.</p>\n"
        "</section>\n"
    )


if __name__ == "__main__":
    main()
