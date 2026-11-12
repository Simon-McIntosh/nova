"""Compose the quantitative landing fragment from the persisted measurement."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


def table(headers, rows):
    head = "".join("<th>" + escape(str(value)) + "</th>" for value in headers)
    body = "".join(
        "<tr>"
        + "".join("<td>" + escape(str(value)) + "</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        "<table><thead><tr>" + head + "</tr></thead><tbody>" + body + "</tbody></table>"
    )


def write_fragment(content):
    output = Path(
        "docs/evidence/fragments/iron-core-magnetisation/icm-west-root-branch.html"
    )
    output.write_text(content)
    print("EVIDENCE=" + str(output))


def partial_record(run, record):
    last = record["steps"][-1]
    failed = record["attempts"][-1]
    nonconverged = sum(not row["converged"] for row in record["attempts"])
    rows = [
        [
            f"{row['drive']:.12g}",
            row["iterations"],
            f"{row['residual']:.12g}",
            f"{row['max_M_A_per_m'] / 1e6:.9g}",
            row["converged"],
            row["regularised_steps"],
        ]
        for row in record["attempts"]
    ]
    ramp_table = table(
        [
            "proposed drive",
            "corrections",
            "relative residual",
            "max |M| [MA/m]",
            "converged",
            "regularised corrections",
        ],
        rows,
    )
    weighted = []
    saved = run.parent / (
        "r-20261111T130417404611-icm-west-62751-solve/"
        "measurement-pair/0.0566/receipt.json"
    )
    origin_mu = json.loads(saved.read_text())["table_mu_relative"][0]
    fraction = 1 - 1 / origin_mu
    for name in ("area", "axisymmetric_volume"):
        item = record["G_" + name]
        low, high = item["symmetric_extremes_over_mu0"]
        weighted.append(
            [
                name,
                f"{item['asymmetry']:.12g}",
                f"{item['skew_fraction']:.12g}",
                f"{item['symmetric_fraction']:.12g}",
                f"{low:.12g}",
                f"{high:.12g}",
                f"{1 - fraction * high:.12g}",
                f"{1 - fraction * low:.12g}",
            ]
        )
    weighted_table = table(
        [
            "weight",
            "G asymmetry",
            "G skew fraction",
            "G symmetric fraction",
            "min sym G / mu0",
            "max sym G / mu0",
            "derived min sym J(0)",
            "derived max sym J(0)",
        ],
        weighted,
    )
    return f"""<section id="icm-west-root-branch-state">
<h3>WEST 62751: multiple roots reproduced; physical branch unresolved</h3>
<p><strong>Blocked.</strong> The bounded drive ramp stops before full drive.
No full-drive root C was produced, and no physical full-drive root is selected.
The strongest completed result is an energy defect in the saved collocation
operator, already present independently of the nonlinear roots.</p>
<p>Saved root A replays at relative residual
{record["replay"]["A"]:.15g}; root B at {record["replay"]["B"]:.15g}.
The maximum vector difference is
{record["root_pair_max_difference_A_per_m"] / 1e6:.12g} MA/m.
The imported material solver is byte-identical to the saved regularised-Newton
version; the smoothing branch is not imported or merged. These are two zeros
of the same discrete equation, not a comparison between two material laws.</p>
<h4>Continuation refusal</h4>
<p>On the saved {record["cells"]}-cell mesh, {len(record["steps"]) - 1}
nonzero drive steps were accepted in {len(record["attempts"])} attempts.
{nonconverged} attempts exhausted 400 corrections. Two additional converged
proposals were refined because they took more than 40 corrections.
The last accepted drive is {last["drive"]:.15g}, reached in
{last["iterations"]} corrections with residual {last["residual"]:.15g}
and maximum |M| {last["max_M_A_per_m"] / 1e6:.12g} MA/m.</p>
<p>The final proposal was drive {failed["drive"]:.15g}, an increment of
{failed["increment"]:.15g}. It reached residual {failed["residual"]:.15g}
after 400 corrections, including {failed["regularised_steps"]} regularised
corrections. Halving that increment would cross the declared 1e-5 floor;
the driver refused and the measurement log ends with <code>EXIT=1</code>.
This establishes a bounded continuation failure, not a proof that the
mathematical branch terminates. Neither a larger correction budget nor
pseudo-arclength continuation was tested.</p>
<figure><img
src="/nova/figures/iron-core-magnetisation/icm-west-root-branch/root-branch.svg"
alt="Incomplete drive ramp and strip profiles at their stated drives">
<figcaption>The left panels show accepted steps only, including the zero
state. The empty interval up to drive 1 is unreached. The 400-correction
refusals appear in the table below, not in these accepted-step curves.
The right panels show the saved 23-cell segment-3 strip column: full-drive
A (dashed grey), full-drive B (dotted dark grey), and the last accepted ramp
state at drive {last["drive"]:.9g} (purple). The purple profile is a partial
state and is not root C. It must not be interpreted as a same-drive
comparison or as a demonstrated full-drive physical root.
The 14 × 7.2 in SVG uses a 16 cm slide height. All 42 text artists passed the
layout check; its PNG was inspected without clipping or text collisions.
</figcaption></figure>
<h4>Every drive proposal</h4>
<p>The zero state has M = 0 and residual 0. Each proposal starts from the
last accepted root. The original residual is normalised by the greater of
||M|| and ||drive B_ext||/mu0. Acceptance tolerance is 1e-10.
The initial proposed increment is 0.05; difficult proposals are halved,
and easy accepted steps grow the increment by 1.5 up to 0.05.</p>
{ramp_table}
<h4>Weighted interaction and the zero-drive inference</h4>
<p>For diagonal W, the reported matrix is G_w = sqrt(W) G / sqrt(W).
Asymmetry is ||G_w − G_wᵀ||_F / ||G_w||_F. The skew fraction uses the
half-difference; the symmetric fraction uses the half-sum, each divided
by ||G_w||_F.
Area weights are the requested cell areas. Axisymmetric volume weights
include centroid R; the common 2π cancels.</p>
{weighted_table}
<p>The negative and above-mu0 eigenvalues violate the [0, mu0] bounds of an
energy-consistent B/M projection, also with axisymmetric volume weighting.
G is predominantly symmetric with a nonzero skew part; it is neither
exactly symmetric nor skew. Symmetrising it alone retains the out-of-bound
eigenvalues. This is a concrete collocation defect independent of any
assertion that a particular nonlinear root is physical.</p>
<p>The last two columns are an algebraic inference from measured G, not the
unreached raw-J diagnostic arm. At zero drive and M = 0,
J(0) = I − (1 − 1/mu_r(0)) G/mu0, with mu_r(0) = {origin_mu:.12g}.
The same weighted symmetric part therefore has the listed extrema.
Its negative direction disproves monotonicity in these inner products.
The unweighted Newton spectra, asymmetry, smallest singular values and cell
localisations at A, B and zero were not measured.</p>
<p>Deterministic-start Lanczos checks its explicit Ritz residuals.
The worst measured G/mu0 residual is
{max(record["G_area"]["eigen_residuals"]):.12g} for area weighting and
{max(record["G_axisymmetric_volume"]["eigen_residuals"]):.12g} for volume
weighting. Before WEST, the instrument detected the known negative
eigenvalue −2 and positive eigenvalue 4 of a control matrix, and reproduced
a nonsymmetric small matrix's minimum singular value against NumPy.</p>
<h4>Verdict, missing evidence and owner</h4>
<p>Multiple discrete roots and weighted collocation energy-bound violations
are demonstrated. A physical full-drive root has not been identified.
The candidate repair is an energy-consistent Galerkin interaction, with
reciprocity and spectral energy bounds checked before another shielding
claim. The iron formulation coordinator owns that repair and its assignment.
This node has not established the offending cells or ruled out an additional
constitutive-fit contribution; it does not prescribe agglomeration or merge
the smoothing branch.</p>
<p>The full-drive C magnetisation and its +1/+2/+4 m fields are absent.
The driver placed the four raw-J diagnostic arms after the full ramp, so none
ran when the ramp refused. The separate tiny-drive origin audit, invoked
after that driver, also did not run. These are unfulfilled acceptance
clauses, not passing checks. The next concrete action is a coordinator-owned
operator-only measurement at zero, A, B and this saved near-stall state,
followed by an explicit choice of larger-budget or arclength continuation.
An energy-consistent discretisation is required before physical validation.</p>
<h4>Durability and scope</h4>
<p>One Titan allocation, job 1311378, Tesla P100, x64 asserted before array
construction. The batch ended FAILED after 01:02:25 with status 1:0.
All dense solves, spectral iterations and rendering ran within that same
Titan allocation. A second figure attempt was
terminated when the failed batch ended, and its unrendered changes were
removed. The delivered renderer exactly matches the successful inspected
version. No product source, test, plan state, mesh or material law changed.
No G or probe-operator build was performed.</p>
<p>Measurement revision {escape(record["revision"])}; material source SHA-256
{escape(record["source_sha256"])}. The figure log binds the inspected render
to revision 39f76e65c226436d7dc56d90845dda53d8067392.</p>
<p>Durable run files: <code>{escape(str(run))}</code>:
<code>measurement.json</code>, <code>measurement.log</code>,
<code>ramp-latest.npy</code>, <code>figure-render.log</code>,
<code>figure-summary.json</code> and <code>root-branch.png</code>.
The 314,621-byte PNG stays in the run directory; the 34,154-byte SVG is
the sole published image. No root-C or negative-direction files are claimed.</p>
</section>
"""


def render(run):
    record = json.loads((run / "measurement.json").read_text())
    if not record.get("complete", False):
        write_fragment(partial_record(run, record))
        return
    origin = json.loads((run / "origin-audit.json").read_text())
    assert record["complete"] and len(record["spectra"]) == 4
    spectra = []
    locations = []
    for name, item in record["spectra"].items():
        singular = item["smallest_singular"]
        spectra.append(
            [
                name,
                *[f"{x:.9g}" for x in item["symmetric_extremes"]],
                f"{item['asymmetry']:.9g}",
                f"{singular['value']:.9g}",
                f"{singular['direct_norm']:.9g}",
                f"{singular['transpose_defect']:.3g}",
                ", ".join(f"{x:.9g}" for x in item["area_weighted_extremes"]),
            ]
        )
        direction = item["negative_direction"]
        cell = direction["cells"][0]
        fractions = ", ".join(
            f"{s}: {100 * f:.3f}%" for s, f in direction["segment_fraction"].items()
        )
        locations.append(
            [
                name,
                cell["cell"],
                cell["segment"],
                ", ".join(f"{x:.6g}" for x in cell["centroid_m"]),
                f"{cell['area_m2'] * 1e6:.6g}",
                f"{cell['fraction'] * 100:.3f}%",
                f"{direction['strip_fraction'] * 100:.3f}%",
                fractions,
            ]
        )
    fields = []
    for name, root in record["roots"].items():
        for offset, field, magnitude in zip(
            (1, 2, 4), root["field_T"], root["field_magnitude_T"], strict=True
        ):
            fields.append(
                [
                    name,
                    offset,
                    f"{field[0] * 1e6:.6f}",
                    f"{field[1] * 1e6:.6f}",
                    f"{magnitude * 1e6:.6f}",
                ]
            )
    comparisons = []
    for name, item in record["comparisons"].items():
        comparisons.append(
            [
                name,
                f"{item['max_difference_A_per_m'] / 1e6:.9g}",
                *[f"{100 * x:.6f}%" for x in item["relative_field_change"]],
            ]
        )
    ramp = [
        [
            f"{x['drive']:.10g}",
            x["iterations"],
            f"{x['residual']:.9g}",
            f"{x['max_M_A_per_m'] / 1e6:.9g}",
            x.get("regularised_steps", 0),
        ]
        for x in record["steps"]
    ]
    weighted = []
    for name in ("area", "axisymmetric_volume"):
        metric = record["G_" + name]
        weighted.append(
            [
                name,
                f"{metric['asymmetry']:.9g}",
                f"{metric['skew_fraction']:.9g}",
                f"{metric['symmetric_fraction']:.9g}",
                *[f"{x:.9g}" for x in metric["symmetric_extremes_over_mu0"]],
            ]
        )
    origin_status = (
        "Reached the first ramp drive."
        if origin["complete"]
        else "Stopped at the minimum drive increment; the local branch was "
        "not certified to the first ramp drive."
    )
    if origin["complete"]:
        origin_status += (
            " Refined/coarse endpoint difference: "
            f"{origin['endpoint_difference_A_per_m']:.9g} A/m "
            f"({origin['endpoint_relative_difference']:.9g} relative L2)."
        )
    ramp_table = table(
        [
            "drive",
            "corrections",
            "relative residual",
            "max |M| [MA/m]",
            "regularised corrections",
        ],
        ramp,
    )
    spectrum_table = table(
        [
            "state",
            "min eig sym J",
            "max eig sym J",
            "||J−Jᵀ||/||J||",
            "min singular",
            "direct ||Jv||",
            "||Jᵀu−σv||",
            "area-weighted eig extrema",
        ],
        spectra,
    )
    location_table = table(
        [
            "state",
            "leading cell",
            "segment",
            "(R,Z) [m]",
            "area [mm²]",
            "cell fraction",
            "study-strip fraction",
            "all segment fractions",
        ],
        locations,
    )
    weighted_table = table(
        [
            "weight",
            "asymmetry",
            "skew fraction",
            "symmetric fraction",
            "min eig sym G_w / mu0",
            "max eig sym G_w / mu0",
        ],
        weighted,
    )
    comparison_table = table(
        ["comparison", "max vector ΔM [MA/m]", "+1 m ΔB", "+2 m ΔB", "+4 m ΔB"],
        comparisons,
    )
    content = f"""<section id="icm-west-root-branch-state">
<h3>WEST 62751: continuation and discrete monotonicity</h3>
<p>The saved roots are two distinct zeros of one unchanged equation: A replays at
{record["replay"]["A"]:.12g}, B at {record["replay"]["B"]:.12g};
their maximum vector difference is
{record["root_pair_max_difference_A_per_m"] / 1e6:.9g} MA/m.
The ramp produces root C at full drive.
The collocation operator fails monotonicity. Root C is the prescribed ramp's endpoint;
this alone does not certify a physical continuum solution when the discretisation itself
has negative directions. A, B and C must not be used as validated
WEST shielding predictions.</p>
<p>The mechanism identified here is non-monotone centroid collocation on the saved mesh.
It violates weighted reciprocity and the positive contraction bound for a magnetic B/M
interaction, independently of a nonlinear root. A monotone H(B) does not
repair that defect. The repair belongs to the iron discretisation owner
(the coordinator's formulation/mesh work): replace centroid collocation with
an energy-consistent Galerkin interaction and validate the weighted energy bounds
and high-permeability shell convergence. Preserve strip resolution;
agglomeration or smoothing alone is not a certificate of that property.
This node changes no product source and does not merge the smoothing branch.</p>
<figure><img
src="/nova/figures/iron-core-magnetisation/icm-west-root-branch/root-branch.svg"
alt="Drive ramp and three root profiles in the outer-limb strip"><figcaption>
The left panels show accepted ramp steps only: maximum magnetisation and
Newton correction
count against drive. The right panels compare radial and vertical magnetisation at the
saved 23-cell segment-3 column; A is dashed grey, B dotted dark grey,
and C solid purple. Rejected adaptive attempts remain in the receipt.
This is a material solve, not a terminal Grad–Shafranov forward state.
</figcaption></figure>
<h4>Drive ramp</h4>
<p>Begin with M = 0 and drive 0; proposed increment 0.05,
halved after non-convergence or
more than 40 corrections (down to the declared refinement floor), with growth after easy
steps. Every accepted step meets relative residual 1e-10, normalised by the greater of
||M|| and ||B_ext||/mu0. There are {len(record["steps"]) - 1} accepted nonzero steps and
{len(record["attempts"])} attempts. All G and external/probe operands are reused.</p>
{ramp_table}
<h4>Origin check</h4>
<p>{origin_status} The first tiny-drive state's difference from the implicit-function
tangent is {origin.get("first_tangent_relative_error", float("nan")):.9g} relative L2.
The fitted polynomial H(B) has dH/dB extrema
{origin["law_min_dH_dB"]:.9g} and {origin["law_max_dH_dB"]:.9g} A/(m T)
(vacuum reluctivity {origin["vacuum_reluctivity"]:.9g}).
These derivative extrema include
all interval endpoints and interior quadratic extrema, rather than a sampled grid.</p>
<h4>Newton operator</h4>
<p>J = dF/dM is the exact unregularised Newton matrix, with
F = M − c(|B_ext+GM|)(B_ext+GM).
The zero row uses zero drive; A/B/C use full drive. Norms for asymmetry are Frobenius.
Negative eigenvalues at any measured state disprove monotonicity; positive values at a
finite set of states would not prove global monotonicity.
A nonlinear residual's J is not itself an energy Hessian,
so weighted G is examined separately below.</p>
{spectrum_table}
<p>Ritz values use deterministic-start Lanczos with explicit eigen residuals
in the receipt.
The smallest singular value comes from the largest eigenvalue of J⁻¹J⁻ᵀ using an LU
factor and transpose solves, checked against direct singular-vector equations.
A known indefinite diagonal matrix returns extrema −2 and 4; an independent nonsymmetric
small matrix reproduces NumPy's singular value before the WEST matrices
are measured.</p>
<h4>Negative-direction locations</h4>
<p>Fractions below are squared Euclidean eigenvector components summed per cell.
The strip fraction refers to the complete saved segment-3 column,
not every edge strip in the mesh.
The full twelve-cell witness lists and per-segment fractions are in the receipt.</p>
{location_table}
<h4>Weighted collocation G</h4>
<p>For diagonal weights W, report G_w = sqrt(W) G / sqrt(W), so the skew and symmetric
fractions refer to the requested inner product. Area weights are reported as requested;
axisymmetric volume weights include centroid R (the common 2π cancels).
An energy-consistent B/M projection has symmetric spectrum inside [0, mu0].
The measured negative and above-mu0
witnesses violate both ends of that interval, also with axisymmetric volume weighting.
G is neither exactly symmetric nor skew: most of its Frobenius norm is symmetric, with a
nonzero skew part. Merely symmetrising G would retain these
out-of-bound energy directions.</p>
{weighted_table}
<h4>Full-drive far field and root comparison</h4>
<p>Fields include the saved air-core contribution and the same saved probe response
for all roots, at R = 5.52, 6.52 and 8.52 m, Z = 0.
Relative differences are vector norms divided
by the corresponding A or B field norm, rather than a component-wise ratio.</p>
{table(["root", "outside coil edge [m]", "B_R [µT]", "B_Z [µT]", "|B| [µT]"], fields)}
{comparison_table}
<h4>Provenance and limits</h4>
<p>One Titan job, 1311378, on a Tesla P100; x64 asserted before array construction.
Measurement revision {escape(record["revision"])};
origin audit revision {escape(origin["revision"])}.
Material source SHA-256 {escape(record["source_sha256"])}; byte-identical to the saved
regularised-Newton solver. No G build, source mesh change, law repair,
production call-site change or suite claim. Data larger than the repository ceiling
stays in the run directory.</p>
<p>Durable receipts: <code>{escape(str(run / "measurement.json"))}</code>,
<code>{escape(str(run / "origin-audit.json"))}</code>,
<code>{escape(str(run / "measurement.log"))}</code>,
<code>{escape(str(run / "root-C.npy"))}</code>,
and per-state <code>negative-direction-STATE.npy</code>.
The SVG was mechanically checked for text clipping/overlap, rendered to a PNG
in the run directory, and inspected before delivery.</p>
</section>
"""
    write_fragment(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    render(parser.parse_args().run)
