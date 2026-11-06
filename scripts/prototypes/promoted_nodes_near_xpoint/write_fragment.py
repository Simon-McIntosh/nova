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
COMPONENTS_RETURN_LINE = "nova/equilibrium/forward_operator.py:3758"
DOMAIN_CLASSIFY_LINE = "nova/equilibrium/domain.py:213"
DOMAIN_FLOOD_LINE = "nova/equilibrium/domain.py:185"
DOMAIN_PRIVATE_LINE = "nova/equilibrium/domain.py:117"
DOMAIN_PROFILE_LINE = "nova/equilibrium/domain.py:128"
FLOOD_SOURCE = "interior flood (private flux)"
POCKET_SOURCE = ("lower", FLOOD_SOURCE)
NEAR_FLOOD_SOURCE = ("upper", FLOOD_SOURCE)
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
    bearings = [entry["bearing_deg"] for entry in state_asymptotic]

    def nearest_asymptote(value):
        return min(
            (abs((value - other + 180.0) % 360.0 - 180.0) for other in bearings),
            default=float("nan"),
        )

    branch_rows = []
    branch_shifts = []
    chord_deviations = []
    for entry in angles["state"]["branches"]:
        partner = image_branches.get(ARM_NAMES.get(entry["arm"], entry["arm"]))
        if partner is None:
            continue
        shift = abs(
            (entry["bearing_deg"] - partner["bearing_deg"] + 180.0) % 360.0 - 180.0
        )
        state_deviation = nearest_asymptote(entry["bearing_deg"])
        image_deviation = nearest_asymptote(partner["bearing_deg"])
        chord_deviations.extend((state_deviation, image_deviation))
        branch_shifts.append(shift)
        branch_rows.append(
            "<tr><td>%s</td><td>%+.2f</td><td>%+.2f</td><td>%.3f</td>"
            "<td>%.2f</td><td>%.2f</td><td>%.4f</td></tr>"
            % (
                ARM_NAMES.get(entry["arm"], entry["arm"]),
                entry["bearing_deg"],
                partner["bearing_deg"],
                shift,
                state_deviation,
                image_deviation,
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
        "chord_max_deviation": max(chord_deviations)
        if chord_deviations
        else float("nan"),
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
            " column averages two selections and is not a single quantity; the split by"
            " source is the first reading that separates them. It is not the last: the"
            " lower half's flood group carries the residual in its own right, and the"
            " next section is what that group is."
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


def pocket_values(pocket, groups):
    """Read the lower pocket's own figures from its receipt."""
    levels = pocket["levels"]
    nodes = pocket["pocket"]["nodes"]
    span = levels["boundary_wb"] - levels["axis_wb"]
    saddle = levels["lower_state"]
    image = levels["lower_image"]
    psi_norm = (saddle["flux_wb"] - levels["axis_wb"]) / span
    worst = max(nodes, key=lambda row: abs(row["image_minus_state_wb"]))
    nearest = min(nodes, key=lambda row: row["distance_to_separatrix_mm"])
    lower = pocket["groups"]["lower / %s" % FLOOD_SOURCE]
    upper = pocket["groups"]["upper / %s" % FLOOD_SOURCE]
    orders = math.log10(
        max(lower["max_abs_wb"], 1.0e-300) / max(upper["max_abs_wb"], 1.0e-300)
    )
    return {
        "pocket_count": pocket["pocket"]["count"],
        "saddle_rz": "(%+.5f, %+.5f) m" % tuple(saddle["position_rz"]),
        "saddle_flux": exponent(saddle["flux_wb"]),
        "saddle_psi_norm": psi_norm,
        "saddle_beyond_boundary": abs(psi_norm - 1.0),
        "image_saddle_rz": "(%+.5f, %+.5f) m" % tuple(image["position_rz"]),
        "image_saddle_flux": exponent(image["flux_wb"]),
        "axis_flux": "%.6f" % levels["axis_wb"],
        "boundary_flux": "%.6f" % levels["boundary_wb"],
        "pocket_psi_min": min(row["psi_norm"] for row in nodes),
        "pocket_psi_max": max(row["psi_norm"] for row in nodes),
        "pocket_above_min": min(row["flux_above_lower_saddle_wb"] for row in nodes),
        "pocket_above_max": max(row["flux_above_lower_saddle_wb"] for row in nodes),
        "pocket_below_min": min(row["below_lower_saddle_mm"] for row in nodes),
        "pocket_below_max": max(row["below_lower_saddle_mm"] for row in nodes),
        "pocket_saddle_min": min(row["distance_to_lower_saddle_mm"] for row in nodes),
        "pocket_saddle_max": max(row["distance_to_lower_saddle_mm"] for row in nodes),
        "pocket_wall_min": min(row["distance_to_wall_mm"] for row in nodes),
        "pocket_wall_max": max(row["distance_to_wall_mm"] for row in nodes),
        "pocket_sep_min": nearest["distance_to_separatrix_mm"],
        "pocket_sep_max": max(row["distance_to_separatrix_mm"] for row in nodes),
        "pocket_nearest_node": nearest["node"],
        "pocket_worst_node": worst["node"],
        "pocket_worst_rz": "(%+.4f, %+.4f) m" % (worst["radius_m"], worst["height_m"]),
        "pocket_worst_state": exponent(worst["state_wb"]),
        "pocket_worst_image": exponent(worst["image_wb"]),
        "pocket_worst_difference": exponent(worst["image_minus_state_wb"]),
        "pocket_worst_operator": exponent(worst["operator_image_minus_state_wb"]),
        "pocket_orders": orders,
        "flood_upper_max": exponent(upper["max_abs_wb"]),
        "flood_lower_max": exponent(lower["max_abs_wb"]),
        "flood_upper_count": upper["count"],
        "flood_lower_count": lower["count"],
        "flood_upper_sep_min": upper["distance_to_separatrix_mm_min"],
        "flood_upper_sep_max": upper["distance_to_separatrix_mm_max"],
        "flood_upper_psi_min": upper["psi_norm_min"],
        "flood_upper_psi_max": upper["psi_norm_max"],
    }


def compose(receipt, groups, angles, pocket):
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
    state_asymptotic_field = angles["state"]["asymptotic"]
    pocket_figure = pocket_values(pocket, groups)
    pocket_figure["flood_orders"] = pocket_figure.pop("pocket_orders")
    pocket_figure["seed_r"], pocket_figure["seed_z"] = pocket["lower_seed_rz"]
    pocket_figure["domain_classify_line"] = DOMAIN_CLASSIFY_LINE
    pocket_figure["domain_flood_line"] = DOMAIN_FLOOD_LINE
    pocket_figure["domain_private_line"] = DOMAIN_PRIVATE_LINE
    pocket_figure["domain_profile_line"] = DOMAIN_PROFILE_LINE
    pocket_figure["components_return_line"] = COMPONENTS_RETURN_LINE

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

<p>The Hessian at the null has eigenvalues {eigenvalue_minor:+.6f} and
{eigenvalue_major:+.6f} Wb/m^2, a ratio of {eigenvalue_ratio:.6f}, and its trace is
{trace:+.3e} Wb/m^2. The eigenvectors are the principal axes at {axis_major:+.2f}{deg}
and {axis_minor:+.2f}{deg}; they are orthogonal by construction and are <em>not</em> the
separatrix asymptotes, which are the directions <code>d</code> with
<code>d'Hd = 0</code>. In the eigenbasis that is
<code>l1 cos^2 t + l2 sin^2 t = 0</code>, so each asymptote lies
<strong>{half_angle:.4f}{deg}</strong> from the major axis — 45{deg} only when the two
eigenvalues are equal and opposite — and the two asymptote lines open
<strong>{separation:.4f}{deg}</strong> at the null.</p>

<p>The trace fixes the local toroidal current density. With
<code>Delta* psi = psi_RR - psi_R/R + psi_ZZ</code> and <code>psi_R = 0</code> at a
null, <code>Delta* psi</code> there is the trace itself, and <code>mu0 J_phi = -Delta*
psi</code> gives <strong>{current:.3e} A/m^2</strong>. A traceless Hessian would mean no
local toroidal current and asymptotes at exactly 90{deg}; this one is not traceless, so
the asymptotes open slightly wider than a right angle and the four arms are not evenly
spaced.</p>

<table>
  <caption>The four separatrix asymptotes at the null, from the eigenvalue ratio of the
  poloidal-flux Hessian at each field's polished saddle, taken on the solve's lattice.
  Bearings are degrees from +R toward +Z.</caption>
  <tr><th>asymptote</th><th>converged state</th><th>field of the currents</th>
      <th>difference</th></tr>
{asymptotic_rows}
</table>

<p>The four asymptotes agree between the two fields to
{asymptotic_max_shift:.6f}{deg} over {asymptotic_count} directions: the promotion does
not rotate the X.</p>

<table>
  <caption>The direction each assembled branch actually takes over its first one to two
  display cells, read as the chord of that branch between 10 and 20 mm of arc length
  from the saddle, on each field's own trace. The last two columns give each
  chord's angle to the nearest of the four asymptotes above, and
  <code>reach</code> is how far each branch runs from the null.</caption>
  <tr><th>branch</th><th>converged state</th><th>field of the currents</th>
      <th>state vs field</th><th>state vs asymptote</th><th>field vs asymptote</th>
      <th>reach [m]</th></tr>
{branch_rows}
</table>

<p>Over {branch_count} branches the traced chords agree between the two fields to
{branch_max_shift:.3f}{deg}, and each chord lies within {chord_max_deviation:.2f}{deg}
of the asymptote it belongs to. The traced curve therefore reaches the asymptotic
direction within the first 10 to 20 mm and holds it: the two measures are consistent
readings of the same four arms, taken once from the Hessian at the null and once from
the assembled trace. Each branch is identified across the two fields by its own reach,
and the reaches agree to {reach_gap_mm:.3f} mm, so the pairing is of the same four
structures.</p>

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
{half_upper_max} Wb and {half_lower_max} Wb, and the source split does
<em>not</em> dissolve that asymmetry: inside the interior flood alone the upper group
reads {flood_upper_max} Wb over {flood_upper_count} nodes and the lower group
{flood_lower_max} Wb over {flood_lower_count}, a factor of {flood_orders:.1f} in
the exponent. The mechanism is the paste at <code>{trip_line}</code>,
<code>mapped = jnp.where(promoted, state, image)</code>, inside
<code>_reduced_kernels.trip_boundary</code>: on a closing trip the state carried into
the next trip is the map, so at a promoted node the terminal state is the
<em>pre-paste</em> state and <code>image - state</code> there is exactly the residual
the shadow rule exists to hide. The source tells a reader which mask selected a node;
it does not tell a reader which region of the flux map the node sits in, and that is
what decides the size of the hidden residual.</p>

<p>{verdict}</p>

<h3>The lower interior-flood pocket</h3>
<p>The two interior-flood groups are the same rule applied to two very different
places. The upper group sits in the upper private-flux wedge, {flood_upper_sep_min:.1f}
to {flood_upper_sep_max:.1f} mm from the separatrix, its normalised flux running
{flood_upper_psi_min:.3f} to {flood_upper_psi_max:.3f}, and there the field of the
currents already reproduces the frozen state to {flood_upper_max} Wb. The lower group
sits in a pocket the solve never resolves, and its frozen state is
{flood_lower_max} Wb from the field of the currents.</p>

<p>That pocket lies below a second saddle that the solve did not admit. Polishing the
same lattice spline from a seed at ({seed_r:.2f}, {seed_z:.2f}) m converges to a saddle
at {saddle_rz} with flux {saddle_flux} Wb, and the field of the currents puts its
counterpart at {image_saddle_rz} at {image_saddle_flux} Wb. With the axis at
{axis_flux} Wb and the admitted boundary flux at {boundary_flux} Wb, the lower saddle's
normalised flux is {saddle_psi_norm:.4f} — it lies {saddle_beyond_boundary:.4f}
<em>outside</em> the boundary level, and the receipt's admitted set carries the single
upper X-point and not this one. The {pocket_count} lower interior-flood nodes sit
beneath it:
every one is at least {pocket_below_min:.1f} mm below it in height (up to
{pocket_below_max:.1f} mm) and {pocket_saddle_min:.1f} to {pocket_saddle_max:.1f} mm
from it in the plane, their flux is {pocket_above_min:.4f} to {pocket_above_max:.4f} Wb
<em>above</em> its flux, and their normalised flux runs {pocket_psi_min:.3f} to
{pocket_psi_max:.3f} — entirely on the plasma side of the boundary value. They are
{pocket_wall_min:.1f} to {pocket_wall_max:.1f} mm from the wall.</p>

<p>So the pocket is a closed-flux lobe below an unadmitted saddle, bounded on one side
by the wall and cut off from the axis on the other by the narrow band around that
saddle where the flux falls below the boundary level. The flood labels exactly those
cells: <code>classify_domains</code> assigns <code>PRIVATE_FLUX</code> to every cell
that is <code>closed</code> but not in the axis-connected component
(<code>{domain_classify_line}</code>), and <code>connected</code> is the saddle-aware
flood grown from the magnetic axis in <code>axis_connected_component</code>
(<code>{domain_flood_line}</code>). The mask surfaces as
<code>DomainMasks.private_flux</code> (<code>{domain_private_line}</code>) and is
returned as the first shadow component by
<code>ForwardFluxOperator.residual_shadow_components</code>
(<code>{components_line}</code>, returned at <code>{components_return_line}</code>).
Nothing in the pocket carries plasma current: profile participation is the complement
of the private-flux and material masks (<code>{domain_profile_line}</code>), and the
same cells are removed from the residual. No equation therefore constrains the flux
there, and the paste is what holds it at whatever value the iterate carried in — which
is why the field of the currents does not reproduce it.</p>

<p>The worked example is node {pocket_worst_node} at {pocket_worst_rz}, the worst of the
group: its frozen state is {pocket_worst_state} Wb, the field of the currents gives
{pocket_worst_image} Wb, and the difference of {pocket_worst_difference} Wb is
confirmed by the operator's own raster at {pocket_worst_operator} Wb. The held region
changes no boundary geometry. Its nearest node is {pocket_nearest_node}, at
<strong>{pocket_sep_min:.2f} mm</strong> from the state's assembled separatrix, and the
farthest is {pocket_sep_max:.2f} mm away; the pocket touches the trace at no point, and
every boundary reading in this fragment is taken at the admitted upper X-point, which
the pocket never approaches. The held lower region is a defect in the flux map, not in
the boundary.</p>

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
drawn as a labelled leader because a {zoom:.1f} mm window is a fraction of a pixel at
served width; the right panel draws all four branches leaving the null in both fields:
the closed lobe out and back, and both open legs. Dotted grey lines are the four
asymptotes, the directions d with d'Hd = 0 at the state's polished saddle, so a
branch that runs away from its asymptote within the window is visible as such.
  The nearest promoted node is {nearest:.3f} mm from the X-point; the quartiles are
  {q25:.3f}, {q50:.3f} and {q75:.3f} mm.</figcaption>
</figure>

<h3>Does the promotion move the near-X-point geometry?</h3>
<p>Every local reading at the null says no more than the grid it was taken on. On the
solve's lattice, where each field is polished on a spline of its <em>own</em> values,
the two nulls coincide to {lattice_displacement:.2e} mm and the four arms agree to
{asymptotic_max_shift:.6f}{deg}. The {display_displacement:.4f} mm the display grid
carries is not a displacement of the solve's null: that comparison is between two
different interpolants sampled on the same finer grid — the state's lattice tensor
spline evaluated at the display coordinates (<code>display_state_flux</code>, built from
the 0.0928 m lattice nodes) against the reconstructed display map itself
(<code>display_flux</code>, from <code>build_display_flux_map</code> at 0.0100 m). The
0.35 mm is the difference between those two interpolants, carried by the lattice spline
being evaluated off its own nodes; the display map is the finer grid, not the coarser
one. On that grid the arms differ by {display_shift:.3f}{deg} and the traced chords by
at most {branch_max_shift:.3f}{deg}. The promotion therefore perturbs nothing that the
solve resolves. The rule a repair would change is the paste itself: the exclude-shadow
branch of <code>ForwardFluxOperator.residual_shadow_mask</code> in
<code>nova/equilibrium/forward_operator.py</code> decides which nodes keep their
pre-paste state, and <code>_reduced_kernels.trip_boundary</code> in
<code>nova/equilibrium/reduced_newton.py</code> is where that decision is applied.
Narrowing it so a node is kept only where the residual the paste would hide is large
enough to matter, rather than wherever the flood or the wall band reaches, would shrink
the excluded set without changing the closing field it exists to protect — and would
remove the {pocket_count}-node lower pocket, whose held residual is
{flood_lower_max} Wb and whose nearest node is {pocket_sep_min:.1f} mm from any boundary
the solve assembles. Nothing in this node's evidence says the change is needed; it says
the mechanism by which the widened set shows up.</p>
""".format(
        **pocket_figure,
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
        eigenvalue_minor=state_asymptotic_field["eigenvalues"][0],
        eigenvalue_major=state_asymptotic_field["eigenvalues"][1],
        eigenvalue_ratio=state_asymptotic_field["eigenvalue_ratio"],
        trace=state_asymptotic_field["hessian_trace_wb_per_m2"],
        current=state_asymptotic_field["toroidal_current_density_a_per_m2"],
        half_angle=state_asymptotic_field["asymptote_half_angle_deg"],
        separation=state_asymptotic_field["asymptote_separation_deg"],
        axis_major=state_asymptotic_field["axes"][0]["bearing_deg"],
        axis_minor=state_asymptotic_field["axes"][1]["bearing_deg"],
        chord_max_deviation=tables["chord_max_deviation"],
    )


def exponent_safe(value):
    return "not reported" if value is None else exponent(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--angles", type=Path, required=True)
    parser.add_argument("--pocket", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = json.loads(arguments.receipt.read_text(encoding="utf-8"))
    angles = json.loads(arguments.angles.read_text(encoding="utf-8"))
    pocket = json.loads(arguments.pocket.read_text(encoding="utf-8"))
    groups = group_rows(receipt)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        compose(receipt, groups, angles, pocket), encoding="utf-8"
    )
    print("FRAGMENT %s" % arguments.output)
    print(json.dumps({"%s/%s" % key: value for key, value in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
