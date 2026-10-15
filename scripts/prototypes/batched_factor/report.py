"""Compose the measured research record from complete H200 receipts."""

# ruff: noqa: E501

import argparse
import html
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def mb(value):
    return f"{value / 1e6:.3f}"


def number(value):
    return f"{value:.6g}"


def constant_source(shape):
    if len(shape) == 4 and shape[-2:] == [3, 6]:
        return "cached row coupling"
    if len(shape) == 3 and shape[-1] == 3 and shape[0] > 4000:
        return "quadrature flux matrix"
    if len(shape) == 5 and shape[-2:] == [3, 6]:
        return "linearized row-gather intermediate"
    if len(shape) == 2 and shape[0] == shape[1] and shape[0] > 800:
        return "predecessor LU factor"
    return "captured carrier or linearization intermediate"


def constant_rows(carrier, payload):
    rows = []
    for program, constants in payload["closure_constants"].items():
        for item in constants[:5]:
            shape = "×".join(map(str, item["shape"])) or "scalar"
            rows.append(
                f"<tr><td>{carrier}</td><td>{program}</td><td>"
                f"{shape}</td><td>{item['dtype']}</td><td>{mb(item['bytes'])}</td>"
                f"<td>{constant_source(item['shape'])}</td>"
                f"<td>{html.escape(item['operation'])}: HLO constant</td></tr>"
            )
    return "\n".join(rows)


def batch_rows(before, after):
    rows = []
    for width in (1, 16, 128):
        old, new = before[f"batch_{width}"], after[f"batch_{width}"]
        rows.append(
            f"<tr><th>{width}</th><td>{mb(old['executable_bytes'])}</td>"
            f"<td>{mb(new['executable_bytes'])}</td>"
            f"<td>{new['compile_seconds']:.2f}</td>"
            f"<td>{old['per_state_ms']:.3f}</td>"
            f"<td>{new['per_state_ms']:.3f}</td>"
            f"<td>{new['two_converged']}/{width}</td>"
            f"<td>{number(new['two_max_residual'])}</td>"
            f"<td>{number(new['alternate_state_max_residual'])}</td></tr>"
        )
    return "\n".join(rows)


def factor_rows(data, carrier):
    rows = []
    for width in (1, 16, 128):
        entry = data[f"batch_{width}"]
        rows.append(
            f"<tr><th>{carrier} {width}</th>"
            f"<td>{mb(entry['factor_executable_bytes'])}</td>"
            f"<td>{entry['factor_compile_seconds']:.2f}</td>"
            f"<td>{entry['factor_wall_ms']:.3f}</td>"
            f"<td>{entry['factor_per_state_ms']:.3f}</td></tr>"
        )
    return "\n".join(rows)


def diverted_rows(data):
    rows = []
    for width in (1, 16, 128):
        entry = data[f"batch_{width}"]
        rows.append(
            f"<tr><th>{width}</th><td>{mb(entry['executable_bytes'])}</td>"
            f"<td>{entry['compile_seconds']:.2f}</td>"
            f"<td>{entry['per_state_ms']:.3f}</td>"
            f"<td>{entry['two_converged']}/{width}</td>"
            f"<td>{number(entry['two_max_residual'])}</td></tr>"
        )
    return "\n".join(rows)


def write(args):
    before = load(args.before)
    limited = load(args.limited)
    diverted = load(args.diverted)
    limited_constants = load(args.constants_limited)
    diverted_constants = load(args.constants_diverted)
    for data in (limited, diverted):
        assert all(f"batch_{width}" in data for width in (1, 16, 128))
    assert all(str(width) in limited["panels"] for width in (1, 16, 128))
    root = "/nova/figures/converged-forward-solve/batched-factor-size"
    panels = "\n".join(
        f'<figure><img src="{root}/limited-{width}-terminal.svg" '
        f'alt="Limited terminal solved and analytic flux at width {width}, '
        f'both null sets and wall"><figcaption>Width {width}, first distinct terminal '
        f"state: scaled residual {number(limited['panels'][str(width)]['terminal_residual'])}; "
        f"converged {str(limited['panels'][str(width)]['converged']).lower()}. "
        "Solved and analytic line contours use shared levels; both null sets and "
        "the vessel wall are drawn.</figcaption></figure>"
        for width in (1, 16, 128)
    )
    max_size = max(
        limited[f"batch_{width}"]["executable_bytes"] for width in (1, 16, 128)
    )
    max_compile = max(
        limited[f"batch_{width}"]["compile_seconds"] for width in (1, 16, 128)
    )
    best_wall = limited["batch_128"]["per_state_ms"]
    research = f'''<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="docs-project" content="nova"><meta name="reckon-type" content="research">
<meta name="plan-slug" content="batched-factor-size"><meta name="plan-title" content="Batched factor executable size">
<meta name="plan-summary" content="H200 executable size, compile and wall for runtime-argument moment factors and chord batches">
<meta name="plan-status" content="reference"><meta name="plan-informs" content="converged-forward-solve">
<meta name="plan-evidence-for" content="converged-forward-solve"><meta name="plan-verifies" content="converged-forward-solve#s5">
<meta name="plan-source" content="Cache-disabled H200 carrier programs and lowered HLO constant operations">
<meta name="plan-source-quality" content="primary"><meta name="plan-tags" content="equilibrium,grad-shafranov,jax,convergence,topology">
<title>Batched factor executable size | nova</title>
<link rel="stylesheet" href="/_shared/foundation.css"><link rel="stylesheet" href="/_shared/dashboard.css">
</head><body><main class="plan-doc">
<header><p class="eyebrow">Research input</p><h1>Batched factor executable size</h1>
<p class="lead">On the 289-cell limited carrier, passing the state, carrier arrays and predecessor LU as traced operands reduces the largest two-correction batch executable to {mb(max_size)} MB. The width-128 synchronized wall is {best_wall:.3f} ms per distinct state, with {limited["batch_128"]["two_converged"]}/128 reaching a scaled residual at or below 10<sup>−12</sup>. The diverted carrier is measured for program size and compile; its separate convergence defect remains open.</p></header>
<section id="method"><h2>Method and provenance</h2>
<p>The prototype <code>scripts/prototypes/batched_factor/measure.py</code> imports the fixed-shape moment map without changing product code. It uses the cached 289-cell limited and 282-cell diverted certificate carriers, binary64 before array construction, and a cache-disabled H200 process for each carrier. The residual is evaluated from the scaled moment map. A vmapped JVP of identity directions forms every dense Jacobian column inside one jitted factor program; <code>jax.scipy.linalg.lu_factor</code> and <code>lu_solve</code> build and apply the device factor. The factor, state, scale and carrier arrays are runtime arguments to the two-correction chord program. A separate program builds fresh factors over state groups of four within one compiled scan to bound the temporary Jacobian memory.</p>
<p>The fresh closure-constant reproduction and the runtime and HLO jobs write complete logs and JSON receipts under <code>{html.escape(str(args.before.parent))}</code>. Each log begins with revision, tree, command, imported <code>nova</code> and measurement-module paths. All walls are synchronized medians of five calls, or three for the factor build. MB uses decimal bytes. The width-128 wall is a batch throughput measure, not a single-state latency.</p></section>
<section id="constants"><h2>Captured constants</h2>
<p>The fresh closure form reproduces {mb(before["batch_1"]["executable_bytes"])}, {mb(before["batch_16"]["executable_bytes"])} and {mb(before["batch_128"]["executable_bytes"])} MB at widths 1, 16 and 128. The table lists the five largest HLO <code>constant</code> operations in each oversized scalar-product, Jacobian-column and chord-batch lowering. The 50.65 MB row coupling and 36.08 MB quadrature matrix are carrier operands in the new program; the repeated 15.11 MB rows in the linearized products are captured primal intermediates of <code>jax.linearize</code>. A shape match identifies the source; the HLO operation name and byte count come directly from the lowered module with large literal values elided.</p>
<table><caption>Largest lowered HLO literals, by program and carrier. Shapes are element counts; size is decimal MB.</caption><thead><tr><th>Carrier</th><th>Program</th><th>Shape</th><th>Dtype</th><th>MB</th><th>Source</th><th>Producing operation</th></tr></thead><tbody>
{constant_rows("limited", limited_constants)}
{constant_rows("diverted", diverted_constants)}
</tbody></table></section>
<section id="limited"><h2>Limited two-correction batches</h2>
<table><caption>One predecessor factor is reused across distinct vertical-field offsets. Cold compile uses the disabled persistent cache; one compiled program also serves a second state with the same shape.</caption><thead><tr><th>Width</th><th>Closure MB</th><th>Runtime MB</th><th>Cold compile [s]</th><th>Closure [ms/state]</th><th>Runtime [ms/state]</th><th>Converged</th><th>Max residual</th><th>Second state residual</th></tr></thead><tbody>
{batch_rows(before, limited)}
</tbody></table>
<p>The maximum runtime batch cold compile is {max_compile:.2f} s, against 60 s. Width one is reported as a latency control and does not meet 0.90 ms per state; the lead's acceptance target applies to batched per-state wall. A 2% perturbation of the predecessor LU produces width-16 maximum residual {number(limited["batch_16"]["perturbed_factor_max_residual"])}, above 10<sup>−12</sup>; this rerun makes the factor-dependence check red. The closure control likewise reruns the oversized program rather than reading a frozen receipt.</p>
<figure><img src="{root}/program-costs.svg" alt="Before and after executable size and per-state wall against batch width"><figcaption>Closure capture (dashed grey) and runtime operands (blue) at widths 1, 16 and 128 on the same limited H200 carrier. Dotted limits mark 50 MB and 0.90 ms per state.</figcaption></figure>
</section>
<section id="factors"><h2>Fresh factor build</h2>
<p>The predecessor factor program itself is {mb(limited["runtime_factor"]["executable_bytes"])} MB and cold-compiles in {limited["runtime_factor"]["compile_seconds"]:.2f} s on the limited carrier; it is {mb(diverted["runtime_factor"]["executable_bytes"])} MB and cold-compiles in {diverted["runtime_factor"]["compile_seconds"]:.2f} s on the diverted carrier. The separate rows below include Jacobian columns and LU for fresh per-state factors. Their synchronized walls are not included in the reused-factor chord batch wall.</p>
<table><caption>Fresh factors compiled as one program per width, with groups of four states inside the program.</caption><thead><tr><th>Carrier / width</th><th>Executable [MB]</th><th>Cold compile [s]</th><th>Batch wall [ms]</th><th>Per state [ms]</th></tr></thead><tbody>
{factor_rows(limited, "limited")}
{factor_rows(diverted, "diverted")}
</tbody></table></section>
<section id="diverted"><h2>Diverted program costs</h2>
<table><caption>The 282-cell diverted carrier is included for executable size and compile; convergence belongs to the separate diverted-state investigation.</caption><thead><tr><th>Width</th><th>Runtime MB</th><th>Cold compile [s]</th><th>Wall [ms/state]</th><th>Converged</th><th>Max residual</th></tr></thead><tbody>
{diverted_rows(diverted)}
</tbody></table></section>
<section id="geometry"><h2>Limited terminal geometry</h2>
<p>Each panel shows the first terminal state from its batch, using shared levels for solved and analytic poloidal flux, both null sets, and the vessel wall. The field offset is 0.5 of the measured warm perturbation for the first state; the panels draw the base analytic exterior, whose omitted field change is order 10<sup>−8</sup> of the boundary span.</p>
{panels}
</section></main></body></html>
'''
    args.research.write_text(research)
    fragment = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="docs-project" content="nova"><meta name="reckon-type" content="evidence"><meta name="plan-slug" content="batched-factor-size"><meta name="plan-title" content="Batched factor size evidence"><meta name="plan-summary" content="H200 batch executable and wall measurements"><meta name="plan-evidence-for" content="converged-forward-solve"><meta name="plan-verifies" content="converged-forward-solve#s5"><meta name="plan-recorded-at" content="2026-10-15"><meta name="plan-verdict" content="partial"><title>Batched factor size evidence | nova</title><link rel="stylesheet" href="/_shared/foundation.css"><link rel="stylesheet" href="/_shared/dashboard.css"></head><body><main class="plan-doc"><h1 id="batched-factor-size-landing">Batched factor size evidence</h1><p id="batched-factor-size-record"><a href="/nova/research/batched-factor-size.html">Measured research record</a></p></main></body></html>\n"""
    args.fragment.write_text(fragment)


def main():
    parser = argparse.ArgumentParser()
    for name in (
        "before",
        "limited",
        "diverted",
        "constants-limited",
        "constants-diverted",
        "research",
        "fragment",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    write(parser.parse_args())


if __name__ == "__main__":
    main()
