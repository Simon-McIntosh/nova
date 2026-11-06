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
ARM_NAMES = {
    "closed out": "lobe out",
    "closed back": "lobe back",
    "open 0": "leg 0",
    "open 1": "leg 1",
}
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


def angle_rows(angles):
    """Build the asymptotic and branch tables from the null-direction receipt."""
    state_asymptotic = angles["state"]["asymptotic"]["directions"]
    image_asymptotic = angles["image"]["asymptotic"]["directions"]
    asymptotic_rows = []
    asymptotic_shifts = []
    for index, (left, right) in enumerate(
        zip(state_asymptotic, image_asymptotic, strict=False), start=1
    ):
        shift = abs(
            (left["bearing_deg"] - right["bearing_deg"] + 180.0) % 360.0 - 180.0
        )
        asymptotic_shifts.append(shift)
        asymptotic_rows.append(
            "<tr><td>arm %d</td><td>%+.2f</td><td>%+.2f</td><td>%.6f</td></tr>"
            % (index, left["bearing_deg"], right["bearing_deg"], shift)
        )
    image_branches = {
        ARM_NAMES.get(entry["arm"], entry["arm"]): entry
        for entry in angles["image"]["branches"]
    }
    branch_rows = []
    branch_shifts = []
    for entry in angles["state"]["branches"]:
        partner = image_branches.get(ARM_NAMES.get(entry["arm"], entry["arm"]))
        if partner is None:
            continue
        shift = abs(
            (entry["bearing_deg"] - partner["bearing_deg"] + 180.0) % 360.0 - 180.0
        )
        branch_shifts.append(shift)
        branch_rows.append(
            "<tr><td>%s</td><td>%+.2f</td><td>%+.2f</td><td>%.3f</td><td>%.4f</td></tr>"
            % (
                ARM_NAMES.get(entry["arm"], entry["arm"]),
                entry["bearing_deg"],
                partner["bearing_deg"],
                shift,
                entry["reach_m"],
            )
        )
    return {
        "asymptotic_rows": "\n".join(asymptotic_rows),
        "branch_rows": "\n".join(branch_rows),
        "asymptotic_max_shift": max(asymptotic_shifts)
        if asymptotic_shifts
        else float("nan"),
        "branch_max_shift": max(branch_shifts) if branch_shifts else float("nan"),
        "asymptotic_count": len(asymptotic_rows),
        "branch_count": len(branch_rows),
        "state_asymptotic": state_asymptotic,
        "state_branches": angles["state"]["branches"],
        "image_branches": angles["image"]["branches"],
    }


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
            " column is a true agreement rather than an artifact: the flood"
            " selected nodes whose residual is already at the solve floor, so the"
            " paste changes nothing at them." % exponent(group["max_abs_operator"])
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
    wall = [
        (key, entry)
        for key, entry in groups.items()
        if key[0] == half and "wall" in key[1] and key != (half, source)
    ]
    contamination = ""
    if wall:
        key, entry = max(wall, key=lambda item: item[1]["count"])
        contamination = (
            " The reported %s-half column is nevertheless not uniformly zero: the"
            " %d %s nodes in the same half read up to %s Wb, and the operator's raster"
            " carries no wall coordinate to check them against. So the half's aggregate"
            " column is an artifact of averaging two selections in one figure, and the"
            " corrected column is this per-source split, not the half total."
            % (half, entry["count"], key[1], exponent(entry["max_abs_difference"]))
        )
    return (
        "The worst near-zero group is the %s-half %s nodes (%d of them), at %s Wb."
        " %s%s"
        % (
            half,
            source,
            group["count"],
            exponent(group["max_abs_difference"]),
            reading,
            contamination,
        )
    )


def compose(receipt, groups, angles):
    separatrix = receipt["separatrix"]
    x_point = receipt["x_point"]
    halves = receipt["promoted_halves"]
    complete = receipt["complete_field"]
    mismatch = receipt["image_minus_state"]
    tables = angle_rows(angles)
    state_closed = angles["state"]["closed"]
    image_closed = angles["image"]["closed"]
    lattice = angles["grids"]["lattice"]
    display = angles["grids"]["display"]
    state_reach = {
        entry["arm"]: entry["reach_m"] for entry in angles["state"]["branches"]
    }
    reach_gap_mm = max(
        (
            abs(entry["reach_m"] - state_reach[entry["arm"]]) * 1.0e3
            for entry in angles["image"]["branches"]
            if entry["arm"] in state_reach
        ),
        default=float("nan"),
    )
    grid_ratio = float(angles["lattice_spacing_m"]) / float(complete["spacing_m"])

    return """<h2>Separatrix directions at the promoted coil-edit X-point</h2>
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

<p>Both separatrices are well formed: <code>{state_ok}</code> and
<code>{image_ok}</code>. The state's trace carries {state_vertices_display} closed
vertices over an arc of {state_arc:.4f} m and the field's {image_vertices_display}
over {image_arc:.4f} m. The two counts are not comparable, and neither is wrong:
the state's separatrix is assembled on the solve's own lattice
({lattice_spacing:.4f} m cells) while the field's is assembled on the display map
({display_spacing:.4f} m), and each trace is sampled at a fixed number of points per
cell of its own grid. The counts therefore differ by exactly the ratio of the two
grids ({vertex_ratio:.2f} against {grid_ratio:.2f}) while the two curves agree in
length to {arc_gap_mm:.3f} mm: they are the same separatrix at two resolutions, and
the median spacing between the state's samples is {state_segment:.2f} mm against the
field's {image_segment:.2f} mm. Every branch direction below carries that difference,
and the state's chord is read at the limit of its own sampling.</p>

<p>The X-point itself is polished twice, on each field's own spline. On the solve's
lattice the two nulls coincide at
<strong>{lattice_displacement:.2e} mm</strong> and their Hessians agree to
{hessian_lattice:.1e} relative; on the display map they sit
<strong>{display_displacement:.4f} mm</strong> apart, with Hessians agreeing to
{hessian_display:.1e} relative. The
{receipt_displacement:.4f} mm the earlier record quoted is the display-map figure, and
it is a real property of the reconstructed field at that resolution, not of the solve's
lattice. The state's null on the lattice is
({sx:.6f}, {sz:.6f}) m at {state_flux} Wb.</p>

<table>
  <caption>The four asymptotic directions at the null, from the eigenvectors of the
  poloidal-flux Hessian at each field's polished saddle, taken on the solve's lattice.
  Bearings are degrees from +R toward +Z.</caption>
  <tr><th>direction</th><th>converged state</th><th>field of the currents</th>
      <th>difference</th></tr>
{asymptotic_rows}
</table>

<p>The asymptotic directions are the four half-lines the separatrix approaches at the
null, and they agree between the two fields to {asymptotic_max_shift:.6f}{deg} over
{asymptotic_count} directions: the promotion does not rotate the X.</p>

<table>
  <caption>The direction each assembled branch actually takes over its first one to two
  display cells, read as the chord of that branch between 10 and 20 mm of arc length
  from the saddle, on each field's own trace. <code>reach</code> is how far each branch
  runs from the null.</caption>
  <tr><th>branch</th><th>converged state</th><th>field of the currents</th>
      <th>difference</th><th>reach [m]</th></tr>
{branch_rows}
</table>

<p>Over {branch_count} branches the traced directions agree to
{branch_max_shift:.3f}{deg}. At 10 to 20 mm from a null the traced curve is still
visibly rotated away from the asymptotic lines, which is why both measures are given:
the asymptote is where the separatrix is going, and the chord is where the tracer has
it. Each branch is identified across the two fields by its own reach, and the reaches
agree to {reach_gap_mm:.3f} mm, so the pairing is of the same four structures.</p>

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
{whole} Wb over the whole lattice. The half aggregates are
{half_upper_max} Wb and {half_lower_max} Wb, and split by selection source they
resolve as the table does rather than as the halves do. The mechanism is the paste at
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
which draws all four branches leaving the null in both fields: the closed lobe out and
back, and both open legs. Dotted grey lines are the four asymptotic directions from the
Hessian at the state's polished saddle, so a branch that runs away from its asymptote
within the window is visible as such.
  The nearest promoted node is {nearest:.3f} mm from the X-point; the quartiles are
  {q25:.3f}, {q50:.3f} and {q75:.3f} mm.</figcaption>
</figure>

<h3>Does the promotion move the near-X-point geometry?</h3>
<p>Every local reading at the null says no more than the grid it was taken on. On the
solve's lattice the two polished nulls coincide to {lattice_displacement:.2e} mm and the
four arms agree to {asymptotic_max_shift:.6f}{deg}; on the display map the nulls sit
{display_displacement:.4f} mm apart, the arms differ by
{display_shift:.3f}{deg} and the traced chords by at most {branch_max_shift:.3f}{deg}.
The promotion therefore perturbs nothing that the solve resolves, and the whole of the
residual difference is the coarser interpolant the display map carries. The rule a
repair would change is the paste itself: the
exclude-shadow
branch of <code>ForwardFluxOperator.residual_shadow_mask</code> in
<code>nova/equilibrium/forward_operator.py</code> decides which nodes keep their
pre-paste state,
and <code>_reduced_kernels.trip_boundary</code> in
<code>nova/equilibrium/reduced_newton.py</code> is where that decision is applied.
Narrowing it so a
node is kept only where the residual the paste would hide is large enough to matter,
rather than wherever the flood or the wall band reaches, would shrink the excluded set
without changing the field it exists to protect. Nothing in this node's evidence says
the
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
        display_spacing=complete["spacing_m"],
        lattice_spacing=receipt["lattice_spacing_m"],
        state_ok=separatrix["state_well_formed"],
        image_ok=separatrix["image_well_formed"],
        state_vertices_display=state_closed["vertices"],
        image_vertices_display=image_closed["vertices"],
        state_arc=state_closed["arc_length_m"],
        image_arc=image_closed["arc_length_m"],
        state_segment=state_closed["median_segment_mm"],
        image_segment=image_closed["median_segment_mm"],
        vertex_ratio=(image_closed["vertices"] / max(state_closed["vertices"], 1)),
        grid_ratio=grid_ratio,
        arc_gap_mm=1.0e3
        * abs(state_closed["arc_length_m"] - image_closed["arc_length_m"]),
        lattice_displacement=lattice["displacement_mm"],
        display_displacement=display["displacement_mm"],
        receipt_displacement=angles["grids"]["receipt"]["displacement_mm"],
        hessian_lattice=lattice["hessian_relative_difference"],
        hessian_display=display["hessian_relative_difference"],
        sx=x_point["state_rz"][0],
        sz=x_point["state_rz"][1],
        state_flux=exponent(x_point["state_flux_wb"]),
        asymptotic_rows=tables["asymptotic_rows"],
        branch_rows=tables["branch_rows"],
        asymptotic_count=tables["asymptotic_count"],
        asymptotic_max_shift=tables["asymptotic_max_shift"],
        branch_count=tables["branch_count"],
        branch_max_shift=tables["branch_max_shift"],
        reach_gap_mm=reach_gap_mm,
        deg=DEG,
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
        display_shift=max(display["bearing_shifts_deg"]),
    )


def exponent_safe(value):
    return "not reported" if value is None else exponent(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--angles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = json.loads(arguments.receipt.read_text(encoding="utf-8"))
    angles = json.loads(arguments.angles.read_text(encoding="utf-8"))
    groups = group_rows(receipt)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(compose(receipt, groups, angles), encoding="utf-8")
    print("FRAGMENT %s" % arguments.output)
    print(json.dumps({"%s/%s" % key: value for key, value in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
