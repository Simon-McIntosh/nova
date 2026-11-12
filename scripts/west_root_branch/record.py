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


def render(run):
    record = json.loads((run / "measurement.json").read_text())
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
    output = Path(
        "docs/evidence/fragments/iron-core-magnetisation/icm-west-root-branch.html"
    )
    output.write_text(content)
    print("EVIDENCE=" + str(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    render(parser.parse_args().run)
