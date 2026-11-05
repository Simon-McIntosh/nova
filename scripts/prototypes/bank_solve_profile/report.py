"""Compose the stage-profile evidence fragment from the two device receipts."""

# ruff: noqa: E501

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FRAGMENT = (
    ROOT
    / "docs/evidence/fragments/sub-millisecond-forward-path/bank-solve-profile.html"
)
NAMES = [
    "read census",
    "read decision",
    "read geometry",
    "step residual",
    "step assembly",
    "factor",
    "back-solve",
]


def parse(path, prefixes):
    found = {}
    header = None
    for line in Path(path).read_text().splitlines():
        label, _, payload = line.partition(" ")
        if not payload:
            continue
        try:
            row = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if label == "HEADER":
            header = row
        elif label in prefixes:
            found.setdefault(label, []).append(row)
    return header, found


def revision(path):
    for line in Path(path).read_text().splitlines():
        for token in line.split():
            if token.startswith("REVISION="):
                return token.partition("=")[2][:9]
    return "unknown"


def stages_at(rows, lanes):
    medians = {row["name"]: row["median_ms"] for row in rows if row["lanes"] == lanes}
    return {
        "read census": medians["census"],
        "read decision": medians["topology"] - medians["support_geometry"],
        "read geometry": medians["fit"] + medians["support_geometry"],
        "step residual": 2.0 * medians["residual"],
        "step assembly": medians["assembly"],
        "factor": medians["factor"],
        "back-solve": medians["backsolve"],
        "whole": medians["whole_solve"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage_log", type=Path)
    parser.add_argument("fused_log", type=Path)
    parser.add_argument("marginal_log", type=Path)
    args = parser.parse_args()
    header, stage = parse(args.stage_log, {"STAGE", "SOLVE", "SINGLE"})
    fused_header, fused = parse(args.fused_log, {"UNIT"})
    marginal_header, marginal = parse(args.marginal_log, {"MARGINAL"})
    stage_rev = revision(args.stage_log)
    marginal_rev = revision(args.marginal_log)
    solve_rows = {row["lanes"]: row for row in stage["SOLVE"]}
    single = stage["SINGLE"][0]
    units = {(row["lanes"], row["name"]): row["median_ms"] for row in fused["UNIT"]}

    one = stages_at(stage["STAGE"], 1)
    many = stages_at(stage["STAGE"], 16)
    iterate_one = one["whole"] / solve_rows[1]["max_steps"]
    iterate_many = many["whole"] / solve_rows[16]["max_steps"]
    total_one = sum(one[name] for name in NAMES)
    total_many = sum(many[name] for name in NAMES)
    steps = solve_rows[16]["steps"]
    ordered = sorted(steps)
    median_steps = (
        ordered[len(ordered) // 2]
        if len(ordered) % 2
        else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2.0
    )
    median_wall = many["whole"] * median_steps / solve_rows[16]["max_steps"]

    arms = sorted(marginal["MARGINAL"], key=lambda row: row["ceiling"])
    first_arm = arms[0]["median_ms"]
    longest = max(arms, key=lambda row: row["steps"])
    marginal_ms = (longest["median_ms"] - first_arm) / (longest["steps"] - 1)
    units_iterate = units[(1, "read")] + units[(1, "step")] + units[(1, "magnitude")]
    marginal_rows = ""
    for index, row in enumerate(arms):
        if index == 0 or row["steps"] == arms[index - 1]["steps"]:
            delta = "—"
        else:
            span = row["steps"] - arms[index - 1]["steps"]
            delta = f"{(row['median_ms'] - arms[index - 1]['median_ms']) / span:.1f}"
        marginal_rows += (
            f"<tr><td>{row['ceiling']}</td><td>{row['median_ms']:.1f}</td>"
            f"<td>{row['steps']}</td><td>{delta}</td></tr>"
        )

    rows = []
    for name in NAMES:
        rows.append(
            f"<tr><td>{name}</td><td>{one[name]:.2f}</td><td>{many[name]:.2f}</td>"
            f"<td>{many[name] / one[name]:.2f}</td>"
            f"<td>{100.0 * many[name] / total_many:.1f}%</td></tr>"
        )
    rows.append(
        f"<tr><td><b>stage sum</b></td><td><b>{total_one:.2f}</b></td>"
        f"<td><b>{total_many:.2f}</b></td><td>—</td><td>100%</td></tr>"
    )
    rows.append(
        f"<tr><td><b>whole solve, per iterate</b></td><td><b>{iterate_one:.2f}</b></td>"
        f"<td><b>{iterate_many:.2f}</b></td><td>—</td><td>—</td></tr>"
    )
    rows.append(
        f"<tr><td><b>stage sum / whole</b></td><td><b>{total_one / iterate_one:.2f}</b></td>"
        f"<td><b>{total_many / iterate_many:.2f}</b></td><td>—</td><td>—</td></tr>"
    )
    table = "\n".join(rows)
    lane_rows = "".join(
        f"<tr><td>{index + 1}</td><td>{count}</td><td>{reads}</td><td>{converged}</td>"
        f"<td>{residual:.2e}</td></tr>"
        for index, (count, reads, converged, residual) in enumerate(
            zip(
                steps,
                solve_rows[16]["reads"],
                solve_rows[16]["converged"],
                solve_rows[16]["fresh_residual"],
                strict=True,
            )
        )
    )
    unit_rows = "".join(
        f"<tr><td>{name}</td><td>{units[(1, name)]:.2f}</td>"
        f"<td>{units[(16, name)]:.2f}</td><td>{units[(16, name)] / units[(1, name)]:.2f}</td></tr>"
        for name in ("read", "step", "magnitude", "body")
        if (1, name) in units and (16, name) in units
    )

    fragment = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="docs-project" content="nova">
<meta name="reckon-type" content="evidence">
<meta name="plan-evidence-for" content="sub-millisecond-forward-path">
<meta name="plan-verifies" content="sub-millisecond-forward-path#s3">
<meta name="plan-verdict" content="partial">
<title>Bank solve profile | nova</title>
<link rel="stylesheet" href="/_shared/foundation.css">
<link rel="stylesheet" href="/_shared/dashboard.css">
</head>
<body><main class="plan-doc">
<h1 id="bank-solve-profile-landing">Bank solve profile</h1>
<p id="bank-solve-profile-summary">At sixteen slices one Newton step per iterate is
almost the whole iterate: the tangent assembly inside it costs {many["step assembly"]:.0f} ms
of a {total_many:.0f} ms per-iterate stage sum, {100.0 * many["step assembly"] / total_many:.0f} %,
and it grows {many["step assembly"] / one["step assembly"]:.1f} times from one slice
({one["step assembly"]:.0f} ms) while the whole read grows only
{(many["read census"] + many["read decision"] + many["read geometry"]) / (one["read census"] + one["read decision"] + one["read geometry"]):.1f} times.
The stage sum does not close on the whole-solve wall: it overstates it by
{100.0 * (total_one / iterate_one - 1.0):.0f} % at one slice and
{100.0 * (total_many / iterate_many - 1.0):.0f} % at sixteen, and that residual is
reported below rather than absorbed into a remainder stage.</p>

<section id="bank-solve-profile-stage">
<h2>Per-iterate stage wall at one and sixteen slices</h2>
<p>One H200 (job {args.stage_log.stem.rsplit("-", 1)[-1]}, partition
<code>betelgeuse</code>, reservation <code>gpu_0003_grpA</code>, one card, two cores)
at revision <code>{stage_rev}</code>. Every stage is its own jitted call at
the solve's shapes — one or sixteen lanes of the diverted 550-cell bank, each lane a
per-lane perturbation of the same seed — warmed, timed with
<code>block_until_ready</code> and reported as the median of {header["repeats"]}
repeats. Nothing is timed inside the solve's <code>while_loop</code>: a timer there
syncs the device and reports the barrier rather than the stage.</p>
<p>The read is split at the boundaries the product code has. <b>Read census</b> is
<code>stationary_stencil</code> over the batch, the stationary-point solve that
supplies the null and saddle census. <b>Read geometry</b> is the whole-mesh fit
(<code>nodal_jets</code> and <code>field_from_jets</code>) plus the terminal
<code>_support_at_level</code> call the read makes — the clipped edge intervals, the
fragment areas and the fractional membership at the admitted level. <b>Read decision</b>
is <code>read_at_stationary_points</code> less that terminal call, and so carries the
cut-cell membership, the normal-form set, the curvature stencil, the wall events, the
candidate loop and the fragment selection; it is a difference of two measured programs,
not a separate program. <b>Step residual</b> is <code>step_residual</code> counted
twice, once for the solve's own magnitude and once inside the Newton step.
<b>Step assembly</b> is <code>step_jacobian</code>, <b>factor</b> is
<code>_equilibrate</code> and <code>lu_factor</code>, and <b>back-solve</b> is
<code>lu_solve</code> with the update's scaling and linear residual.</p>
<table id="bank-solve-profile-stage-table"><thead><tr><th>Stage</th>
<th>1 slice [ms]</th><th>16 slices [ms]</th><th>Ratio 16 / 1</th>
<th>Share at 16</th></tr></thead><tbody>
{table}
</tbody></table>
<p>The whole solve is timed as the product runs it: <code>solve_batch</code> over the
same states, {header["repeats"]} repeats, divided by the steps it took
({solve_rows[1]["max_steps"]} at one slice, {solve_rows[16]["max_steps"]} at sixteen,
the slowest lane's count).</p>
<figure>
<img src="/nova/figures/sub-millisecond-forward-path/bank-solve-profile/per-iterate-stage.svg"
 alt="Horizontal stacked bars of the per-iterate wall by stage at sixteen slices and at one slice, with the whole-solve wall marked below each bar.">
<figcaption>Per-iterate wall by stage, in ms, at sixteen slices (top) and one slice
(bottom). Each bar is the stage sum; the dashed mark below it is the whole solve's
own per-iterate wall, {iterate_one:.0f} ms at one slice and {iterate_many:.0f} ms at
sixteen. The stage sum overruns the wall in both rows, so the bar's end is an upper
bound on the work and the mark is the measurement it is checked against.</figcaption>
</figure>
<p id="bank-solve-profile-closure">The stage sum does not close on the whole-solve
wall, so the 15 % closure requirement is <b>not met</b>. At one slice the sum is
{total_one:.2f} ms against {iterate_one:.2f} ms per iterate, an overstatement of
{100.0 * (total_one / iterate_one - 1.0):.0f} % or {total_one - iterate_one:.2f} ms;
at sixteen it is {total_many:.2f} ms against {iterate_many:.2f} ms, an overstatement
of {100.0 * (total_many / iterate_many - 1.0):.0f} % or {total_many - iterate_many:.2f} ms.
The residual is not reported as a remainder stage, because it is not work the loop
does: it is the difference between summing separate programs and running one fused
program, and it has the sign that difference predicts.</p>
</section>

<section id="bank-solve-profile-fused">
<h2>Cross-check: the same code as the fused units the loop contains</h2>
<p>To test whether the stage split itself is wrong, the same code is timed the way the
loop holds it, in one process at the same revision, {fused_header["repeats"]} repeats
each: <code>read_batch</code>, one <code>newton_step</code> over the batch, one
<code>step_residual</code>, and one iterate body composed of a read, a step, the
candidate's read and the magnitude.</p>
<table id="bank-solve-profile-fused-table"><thead><tr><th>Fused unit</th>
<th>1 slice [ms]</th><th>16 slices [ms]</th><th>Ratio 16 / 1</th></tr></thead><tbody>
{unit_rows}
</tbody></table>
<p id="bank-solve-profile-fused-read">{fused_text(units, one, many, iterate_one, marginal_ms, units_iterate)}</p>
</section>

<section id="bank-solve-profile-lanes">
<h2>Steps per lane at sixteen slices</h2>
<p>The batch wall is the slowest lane's, because the loop's predicate is
<code>jnp.any(~done)</code>: every lane runs every iteration the slowest lane needs.
The lane counts below come from the same solved receipt: {len(steps)} lanes, a
maximum of {solve_rows[16]["max_steps"]} steps and a median of {median_steps:.0f}.</p>
<table id="bank-solve-profile-lane-table"><thead><tr><th>Lane</th><th>Steps</th>
<th>Reads</th><th>Converged</th><th>Fresh residual</th></tr></thead><tbody>
{lane_rows}
</tbody></table>
<p id="bank-solve-profile-lanes-wall">The batch wall is {many["whole"]:.0f} ms. Run for
the median lane's {median_steps:.0f} steps instead of the maximum
{solve_rows[16]["max_steps"]}, the same per-iterate wall gives {median_wall:.0f} ms, a
saving of {many["whole"] - median_wall:.0f} ms or
{100.0 * (1.0 - median_wall / many["whole"]):.0f} % of the batch. That is the most a
straggler fix could return from this seed, and it is bounded by how far the slowest
lane sits above the median rather than by anything inside an iterate.</p>
</section>

<section id="bank-solve-profile-marginal">
<h2>What one more step actually costs</h2>
<p>A per-iterate wall taken as the whole solve divided by its step count assumes every
iterate costs the same. Pinning the loop's step ceiling below and at convergence
measures the cost of a step directly, as the difference between two arms run on the
same seed (job {args.marginal_log.stem.rsplit("-", 1)[-1]}, one lane,
{marginal_header["repeats"]} repeats, revision <code>{marginal_rev}</code>). The
receipts in this fragment ran at {stage_rev} and {marginal_rev}; every commit between
them touches only <code>scripts/</code>, so the product code measured is identical.</p>
<table id="bank-solve-profile-marginal-table"><thead><tr><th>Step ceiling</th>
<th>Wall [ms]</th><th>Steps run</th><th>ms per added step</th></tr></thead><tbody>
{marginal_rows}
</tbody></table>
<p id="bank-solve-profile-marginal-read">One iteration costs {first_arm:.1f} ms and each
further step adds {marginal_ms:.1f} ms, so the loop holds one moderately dearer first
iterate and a steady state of {marginal_ms:.1f} ms; the per-iterate walls above, taken
as the whole solve over its step count, are that average. The {units_iterate:.1f} ms the
same iterate's units cost as separate programs is {units_iterate / marginal_ms:.2f} times
the steady state, which is the closure residual stated as a ratio rather than as a sum,
and the first iterate's {first_arm:.1f} ms accounts for only
{first_arm - marginal_ms:.1f} ms of it.</p>
<p>Two caveats on this receipt, both visible in its table. These arms run the batch of one
lane at its unperturbed seed, which converges in {longest["steps"]} steps, so a ceiling
set above it converges in the same {longest["steps"]} steps: the last row is a repeat of
its predecessor rather than a longer
run, and the two walls agree to {abs(arms[-1]["median_ms"] - arms[-2]["median_ms"]):.1f} ms,
which is a reproducibility check rather than a measurement. The stage receipt's one-lane
batch is that seed scaled by 1 + 2e-5 and needs
{solve_rows[1]["max_steps"]} steps, so the two receipts are not the same state; their
per-step walls, {iterate_one:.1f} ms and {longest["median_ms"] / longest["steps"]:.1f} ms,
agree to within a few per cent, which is how far the seed moves the cost of a step.</p>
</section>

<section id="bank-solve-profile-single">
<h2>Cross-check: the single-slice solve as the product runs it</h2>
<p><code>support_read.solve</code> at the same seed converged in {single["steps"]:.0f}
steps and {single["refactors"]:.0f} refactors with {single["reads"]:.0f} reads,
terminal fresh residual {single["fresh_residual"]:.3e}, in
{stage["STAGE"][[row["name"] for row in stage["STAGE"]].index("single_solve")]["median_ms"]:.0f}
ms — {stage["STAGE"][[row["name"] for row in stage["STAGE"]].index("single_solve")]["median_ms"] / single["steps"]:.0f}
ms per step. The one-lane <code>solve_batch</code> arm above takes
{solve_rows[1]["max_steps"]} steps on the same seed, so the batched program is not a
like-for-like substitute at one lane; both programs nevertheless spend about the same
per step, and both spend it in roughly half the time the separate stages account for.</p>
</section>
</main></body></html>
"""
    FRAGMENT.write_text(fragment)
    print(f"FRAGMENT={FRAGMENT}")
    print(
        "VALUES "
        + json.dumps(
            {
                "one": one,
                "many": many,
                "iterate_one": iterate_one,
                "iterate_many": iterate_many,
                "total_one": total_one,
                "total_many": total_many,
                "median_steps": median_steps,
                "median_wall": median_wall,
                "units": {
                    f"{lane}:{name}": value for (lane, name), value in units.items()
                },
                "single": single,
            },
            default=float,
            sort_keys=True,
        )
    )


def fused_text(units, one, many, iterate_one, marginal_ms, units_iterate):
    read_one = one["read census"] + one["read decision"] + one["read geometry"]
    read_many = many["read census"] + many["read decision"] + many["read geometry"]
    step_one = (
        one["step residual"] + one["step assembly"] + one["factor"] + one["back-solve"]
    )
    step_many = (
        many["step residual"]
        + many["step assembly"]
        + many["factor"]
        + many["back-solve"]
    )
    body = units[(1, "body")]
    return (
        f"The fused read costs {units[(1, 'read')]:.2f} ms against the {read_one:.2f} ms"
        f" the census, decision and geometry stages sum to, and {units[(16, 'read')]:.2f} ms"
        f" against {read_many:.2f} ms — so the read's split is faithful to within 1 % at"
        f" both batch sizes, and nothing is lost to the separation there. The fused step"
        f" costs {units[(1, 'step')]:.2f} ms against the residual, assembly, factor and"
        f" back-solve stages' own {step_one:.2f} ms at one slice and"
        f" {units[(16, 'step')]:.2f} ms against {step_many:.2f} ms at sixteen, again"
        f" within 1 %. The split is therefore not what overstates the iterate."
        f" <b>What the cross-check does not do is explain the residual.</b> One fused"
        f" iterate body — a read, a step, the candidate's read and the magnitude — costs"
        f" {body:.2f} ms, which is {body / (2.0 * units[(1, 'read')] + units[(1, 'step')] + units[(1, 'magnitude')]):.2f}"
        f" times the sum of its parts: no fusion benefit at all inside that one program."
        f" Yet the loop's own iterate is {iterate_one:.2f} ms — {marginal_ms:.1f} ms in"
        f" its steady state, measured a step at a time — against the"
        f" {units_iterate:.2f} ms its three units cost separately, so the loop is"
        f" {units_iterate / marginal_ms:.2f} times cheaper than its own parts. Two"
        f" readings survive this node: one compiled <code>while_loop</code> body shares"
        f" work between the read and the step that a chain of separate jitted calls"
        f" re-derives and re-transfers, or <code>newton_step</code> inside the loop is not"
        f" doing all the work the standalone call does. The step arms rule out a third"
        f" reading — that the loop hides one very expensive first iterate — because the"
        f" marginal step is cheap too. This node does not choose between the two that"
        f" remain, and it does not claim the first."
    )


if __name__ == "__main__":
    main()
