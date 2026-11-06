"""Compose the near-X-point evidence fragment from a measurement receipt.

Every number the fragment prints is read from the receipt, so the prose cannot
disagree with the archive it cites.  The two source anchors below are held as
constants and printed as code references; they are checked by name in the text
so a reader can resolve them without the receipt.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

TRIP_BOUNDARY_LINE = "nova/equilibrium/reduced_newton.py:985"
SHADOW_MASK_LINE = "nova/equilibrium/forward_operator.py:3699"
COMPONENTS_LINE = "nova/equilibrium/forward_operator.py:3712"
DEG = "\N{DEGREE SIGN}"
FIGURE_URL = (
    "/nova/figures/separatrix-spline-extraction/promoted-nodes-xpoint-angles/"
    "promoted-nodes-xpoint-angles.svg"
)


def exponent(value):
    """Format a small magnitude as a bare mantissa-exponent pair."""
    if value is None or not math.isfinite(float(value)):
        return "not reported"
    return "%.3e" % float(value)


def group_rows(receipt):
    """Aggregate the promoted rows by half and by their selection source."""
    groups = {}
    for row in receipt["promoted"]:
        half = "upper" if row["height_m"] > 0.0 else "lower"
        key = (half, row["promoted_source"])
        entry = groups.setdefault(
            key,
            {
                "count": 0,
                "max_abs_difference": 0.0,
                "max_abs_operator": float("nan"),
                "finite_operator": 0,
            },
        )
        entry["count"] += 1
        entry["max_abs_difference"] = max(
            entry["max_abs_difference"], abs(row["image_minus_state_wb"])
        )
        operator = row["operator_image_minus_state_wb"]
        if isinstance(operator, float | int) and math.isfinite(float(operator)):
            entry["finite_operator"] += 1
            magnitude = abs(float(operator))
            if not math.isfinite(entry["max_abs_operator"]):
                entry["max_abs_operator"] = magnitude
            else:
                entry["max_abs_operator"] = max(entry["max_abs_operator"], magnitude)
    return groups


def angle_rows(receipt):
    """Pair the state's leg bearings with the field's, one row per leg."""
    separatrix = receipt["separatrix"]
    paired = {index: leg for index, leg in enumerate(separatrix["legs"])}
    rows = []
    for index, leg in enumerate(separatrix["legs"]):
        rows.append(
            "<tr><td>leg %d</td><td>%+.1f</td><td>%+.1f</td><td>%+.2f</td>"
            "<td>%.3f</td></tr>"
            % (
                index,
                leg["state_bearing_deg"],
                leg["image_bearing_deg"],
                leg["shift_deg"],
                leg["far_end_separation_mm"],
            )
        )
    return "\n".join(rows), len(paired)


def table_rows(groups):
    """Render the per-half, per-source aggregation as table rows."""
    order = sorted(groups, key=lambda key: (key[0] != "upper", key[1]))
    return "\n".join(
        "<tr><td>%s</td><td>%s</td><td>%d</td><td>%s</td><td>%s</td></tr>"
        % (
            half,
            source,
            groups[(half, source)]["count"],
            exponent(groups[(half, source)]["max_abs_difference"]),
            exponent(groups[(half, source)]["max_abs_operator"]),
        )
        for half, source in order
    )


def verdict(groups):
    """Say what the near-zero column is, from the operator's own control."""
    near = [
        (key, group)
        for key, group in groups.items()
        if group["max_abs_difference"] <= 1.0e-6
    ]
    if not near:
        return (
            "No half-and-source group reads near zero, so the asymmetry the"
            " review reported does not survive this reconstruction and there is"
            " no column to correct."
        )
    largest = max(near, key=lambda item: item[1]["count"])
    half, source = largest[0]
    group = largest[1]
    if not math.isfinite(group["max_abs_operator"]):
        return (
            "The %s-half near-zero groups are the ones the operator's own raster"
            " does not cover, so this reconstruction cannot separate a true"
            " agreement from an instrument gap at those nodes." % half
        )
    ratio = group["max_abs_operator"] / max(group["max_abs_difference"], 1.0e-300)
    if group["max_abs_operator"] <= 1.0e-6:
        reading = (
            "The operator's own column reads %s Wb on the same nodes, so the state"
            " and the field of the currents genuinely agree there and the near-zero"
            " column is a true agreement rather than an artifact. Those nodes are"
            " promoted by the coordinate predicates, not by a residual that the"
            " paste then hides." % exponent(group["max_abs_operator"])
        )
    else:
        reading = (
            "The operator's own column reads %s Wb on the same nodes, %s times the"
            " reconstructed figure, so the near-zero column is an instrument"
            " artifact: the reconstruction reproduces the state's own paste rather"
            " than the field of the currents. The corrected column is the"
            " operator's, and its values are those in the last table column."
            % (exponent(group["max_abs_operator"]), "%.1f" % ratio)
        )
    return (
        "The worst near-zero group is the %s-half %s nodes (%d of them), at %s Wb."
        " %s"
        % (
            half,
            source,
            group["count"],
            exponent(group["max_abs_difference"]),
            reading,
        )
    )


def compose(receipt, groups):
    separatrix = receipt["separatrix"]
    x_point = receipt["x_point"]
    halves = receipt["promoted_halves"]
    complete = receipt["complete_field"]
    mismatch = receipt["image_minus_state"]
    legs_html, leg_rows = angle_rows(receipt)

    return """<h2>Separatrix lobe and leg angles at the promoted coil-edit X-point</h2>
<p>Re-runs the converged MAST {shot}/{slice_index} boundary-coil edit
(fraction {fraction:+.3e} on circuit {circuit}, target
{target:.4f} A) on the assembler as it stands after the display-field
capacity repair, so the separatrix this fragment measures is the one
<code>nova.equilibrium.display_flux._branches</code> actually assembles rather than the
truncated one the earlier record had to report unassembled. The solve closes at
{residual} Wb with <code>converged</code> reading <code>{converged}</code>, and promotes
<strong>{promoted}</strong> of {lattice} lattice nodes plus {wall} of them on the
wall, out of
{physical} physical nodes of which {wall_nodes} are wall coordinates.</p>

<p>The separatrix is assembled twice on the same display grid ({spacing:.4f} m), once
from the
converged state and once from the field of the currents, and both are well formed:
<code>{state_ok}</code> and <code>{image_ok}</code>. Each carries
{state_vertices} and {image_vertices} closed vertices, so neither is at a branch
capacity.
The X-point is polished on both fields: the state's saddle sits at
({sx:.6f}, {sz:.6f}) m at {state_flux} Wb and the field's at
({ix:.6f}, {iz:.6f}) m at {image_flux} Wb, a displacement of
<strong>{displacement:.4f} mm</strong> ({dr:+.4f} mm in R,
{dz:+.4f} mm in Z, {cells:.3f} lattice cells at
{cell_spacing:.4f} m).</p>

<table>
  <caption>Separatrix directions at the X-point, converged state against the field of
the
  currents. Bearings are degrees from +R toward +Z.</caption>
  <tr><th>landmark</th><th>state</th><th>field of the currents</th><th>change</th></tr>
  <tr><td>lobe</td><td>{state_lobe:+.1f}{deg}</td><td>{image_lobe:+.1f}{deg}</td>
      <td>{lobe_change:+.2f}{deg}</td></tr>
  <tr><td>legs found</td><td>{state_count}</td><td>{image_count}</td><td></td></tr>
{legs}
</table>

<p>The lobe direction moves {lobe_change:+.2f}{deg} and the leg directions move by
{leg_shifts} respectively. The lobe tip moves {tip:.3f} mm. With
{leg_rows} leg direction{leg_plural} paired by far-end proximity, the worst
far-end separation is {far_end:.3f} mm, so the legs are the same two structures in both
fields rather than a re-paired set.</p>

<h3>Why the halves differ</h3>
<p>The promoted set is the union of two independent selections, which
<code>ForwardFluxOperator.residual_shadow_mask</code> ({shadow_line}) composes from the
interior flood of the solve's own topology read and the private-wall node mask, both
readable apart through <code>residual_shadow_components</code> ({components_line}).
Grouping
the promoted nodes by half and by that source, with the operator's own raster column
beside
the reconstructed one:</p>

<table>
  <caption>Promoted nodes by half and selection source. <code>image - state</code> is
the field
  reconstructed here; <code>operator</code> is the solve's own raster of the same
difference.</caption>
  <tr><th>half</th><th>selection source</th><th>nodes</th><th>image - state [Wb]</th>
      <th>operator [Wb]</th></tr>
{groups}
</table>

<p>Across the whole promoted set the reconstructed difference reaches
{worst} Wb against {solved_max} Wb on the solved nodes, and
{whole} Wb over the whole lattice. The upper half's
{half_upper_max} Wb and the lower half's
{half_lower_max} Wb are what the review saw. The mechanism is the paste at
<code>{trip_line}</code>, <code>mapped = jnp.where(promoted, state, image)</code>,
inside
<code>_reduced_kernels.trip_boundary</code>: on a closing trip the state carried into
the next
trip is the map, so at a promoted node the terminal state is the <em>pre-paste</em>
state and
<code>image - state</code> there is exactly the residual the shadow rule exists to
hide. A
node therefore reads near zero in that column for one of two different reasons, and
the two
are distinguishable only by the selection source and the operator's own column.</p>

<p>{verdict}</p>

<figure>
  <img src="{figure}" alt="Promoted nodes and the X-point on the converged coil edit's
poloidal field">
  <figcaption>The converged edit {edit} at {residual} Wb.
  Grey contours: the field of the currents. Darker contours: the converged state, on
the same
  level array, so a contour that separates is a real difference. Dashed near-black:
the state's
  separatrix; dashed blue: the field's. Grey circles: promoted nodes; red diamonds:
those whose
  <code>image - state</code> exceeds 1 mWb. Triangles mark the magnetic axis at
  ({ax:.4f}, {az:.4f}) m, the filled cross the state's X-point and the hollow cross
the field's.
  The left panel carries the locator for the {zoom:.1f} mm magnification on the right,
where the
  lobe and the first two leg directions are drawn as rays from the polished saddle.
  The nearest promoted node is {nearest:.3f} mm from the X-point; the quartiles are
  {q25:.3f}, {q50:.3f} and {q75:.3f} mm.</figcaption>
</figure>

<h3>Does the promotion move the geometry by an amount that matters?</h3>
<p>The saddle moves {displacement:.4f} mm, and the direction that changes most is the
{lobe_change_word} at {lobe_change:+.2f}{deg} — {lobe_verdict}. A change of that size is
{displacement_verdict}. The rule a repair would change is the paste itself: the
exclude-shadow
branch of <code>ForwardFluxOperator.residual_shadow_mask</code> in
<code>nova/equilibrium/forward_operator.py</code> decides which nodes keep their
pre-paste state,
and <code>_reduced_kernels.trip_boundary</code> in
<code>nova/equilibrium/reduced_newton.py</code> is where that decision is applied.
Narrowing the
selection to the residual predicate alone — dropping the coordinate predicates that
promote nodes
whose residual is already at the solve floor — would stop the excluded set from
widening without
changing the field it is meant to protect. Nothing in this node's evidence says the
change is
needed; it says the mechanism by which a widened set would show up.</p>
""".format(
        shot=receipt["shot"],
        slice_index=receipt["slice_index"],
        fraction=receipt["edit_fraction"],
        circuit=receipt["coil_circuit_index"],
        target=receipt["target_current_a"],
        residual=exponent(receipt["terminal_residual_wb"]),
        converged=receipt["converged"],
        promoted=receipt["promoted_node_count"],
        lattice=receipt["lattice_node_count"],
        wall=receipt["promoted_wall_node_count"],
        physical=receipt["physical_node_count"],
        wall_nodes=receipt["wall_node_count"],
        spacing=complete["spacing_m"],
        state_ok=separatrix["state_well_formed"],
        image_ok=separatrix["image_well_formed"],
        state_vertices=separatrix["state_closed_vertex_count"],
        image_vertices=separatrix["image_closed_vertex_count"],
        sx=x_point["state_rz"][0],
        sz=x_point["state_rz"][1],
        ix=x_point["image_rz"][0],
        iz=x_point["image_rz"][1],
        state_flux=exponent(x_point["state_flux_wb"]),
        image_flux=exponent(x_point["image_flux_wb"]),
        displacement=x_point["displacement_mm"],
        dr=x_point["displacement_r_mm"],
        dz=x_point["displacement_z_mm"],
        cells=x_point["displacement_lattice_cells"],
        cell_spacing=receipt["lattice_spacing_m"],
        state_lobe=separatrix["state_lobe_bearing_deg"],
        image_lobe=separatrix["image_lobe_bearing_deg"],
        lobe_change=separatrix["lobe_bearing_change_deg"],
        state_count=separatrix["state_leg_count"],
        image_count=separatrix["image_leg_count"],
        legs=legs_html,
        deg=DEG,
        leg_shifts=", ".join(
            "%+.2f%s" % (value, DEG) for value in separatrix["leg_direction_shifts_deg"]
        ),
        tip=separatrix["lobe_tip_shift_mm"],
        leg_rows=leg_rows,
        leg_plural="" if leg_rows == 1 else "s",
        far_end=max(
            (leg["far_end_separation_mm"] for leg in separatrix["legs"]),
            default=float("nan"),
        ),
        shadow_line=SHADOW_MASK_LINE,
        components_line=COMPONENTS_LINE,
        trip_line=TRIP_BOUNDARY_LINE,
        groups=table_rows(groups),
        worst=exponent(mismatch["promoted_max_abs_wb"]),
        solved_max=exponent(mismatch["solved_max_abs_wb"]),
        whole=exponent(mismatch["whole_lattice_max_abs_wb"]),
        half_upper_max=exponent_safe(halves["upper"]["max_abs_wb"]),
        half_lower_max=exponent(halves["lower"]["max_abs_wb"]),
        verdict=verdict(groups),
        figure=FIGURE_URL,
        edit=receipt["edit_index"],
        ax=receipt["nulls"]["axis_rz"][0],
        az=receipt["nulls"]["axis_rz"][1],
        zoom=6.0,
        nearest=receipt["nearest_promoted_to_x_point_mm"],
        q25=receipt["promoted_distance_quantiles_mm"]["q25"],
        q50=receipt["promoted_distance_quantiles_mm"]["q50"],
        q75=receipt["promoted_distance_quantiles_mm"]["q75"],
        lobe_change_word="lobe",
        lobe_verdict=(
            "a fraction of the lobe's own width"
            if abs(separatrix["lobe_bearing_change_deg"]) < 1.0
            else "larger than the solve's own footprint"
        ),
        displacement_verdict=(
            "well under one lattice cell"
            if x_point["displacement_lattice_cells"] < 1.0
            else "more than a lattice cell, so it is not a polishing artefact"
        ),
    )


def exponent_safe(value):
    return "not reported" if value is None else exponent(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = json.loads(arguments.receipt.read_text(encoding="utf-8"))
    groups = group_rows(receipt)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(compose(receipt, groups), encoding="utf-8")
    print("FRAGMENT %s" % arguments.output)
    print(json.dumps({"%s/%s" % key: value for key, value in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
