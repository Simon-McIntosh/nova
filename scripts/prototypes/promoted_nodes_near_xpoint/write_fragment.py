"""Compose the evidence fragment from one measurement receipt.

Every number in the fragment is read from the receipt beside the log, so the
prose cannot drift from the measurement it reports.
"""

from __future__ import annotations

# ruff: noqa: E501  -- the composed fragment's prose lines are HTML, not code

import argparse
import json
from pathlib import Path


def _sci(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}e}"


def _table(rows: list[dict]) -> str:
    head = (
        "<table><thead><tr>"
        "<th>node</th><th><var>R</var> [m]</th><th><var>Z</var> [m]</th>"
        "<th>to X-point [mm]</th><th>to separatrix [mm]</th>"
        "<th>domain</th><th>image &minus; state [mWb]</th>"
        "</tr></thead><tbody>"
    )
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{row['node']}</td>"
            f"<td>{row['radius_m']:.3f}</td>"
            f"<td>{row['height_m']:.3f}</td>"
            f"<td>{row['distance_to_x_point_mm']:.1f}</td>"
            f"<td>{row['distance_to_separatrix_mm']:.1f}</td>"
            f"<td>{row['domain_label']}</td>"
            f"<td>{1.0e3 * row['image_minus_state_wb']:+.4f}</td>"
            "</tr>"
        )
    return head + "".join(body) + "</tbody></table>"


def compose(receipt: dict, figure_src: str) -> str:
    """Return the fragment body for one receipt."""
    promoted = receipt["promoted"]
    difference = receipt["image_minus_state"]
    x_point = receipt["x_point"]
    separatrix = receipt["separatrix"]
    counts = receipt["domain_counts"]
    quantiles = receipt["promoted_distance_quantiles_mm"]
    complete = receipt["complete_field"]
    compensation = receipt["constraint_compensation"]
    spacing_mm = 1.0e3 * receipt["lattice_spacing_m"]
    moves = x_point["displacement_mm"]
    cells = moves / spacing_mm if spacing_mm else float("nan")
    legs = ", ".join(f"{value:.2f}" for value in separatrix["leg_direction_shifts_deg"])
    inside = moves < spacing_mm
    verdict = (
        "The converged near-X-point geometry is therefore the complete field's own "
        f"to within {moves:.1f} mm"
        if inside
        else "The promotion moves the converged near-X-point geometry measurably: "
    )
    verdict += (
        f": re-polishing the saddle on the field of the complete currents moves it "
        f"{moves:.1f} mm ({cells:.2f} lattice cells of {spacing_mm:.1f} mm), turns the "
        f"lobe direction {separatrix['lobe_direction_shift_deg']:.2f}&deg; and the legs "
        f"{legs}&deg;, and leaves {_sci(difference['solved_max_abs_wb'])} Wb on the nodes "
        "the solve does claim, against "
        f"{_sci(difference['promoted_max_abs_wb'])} Wb on the promoted set"
        + (
            ", so the promotion, not the field, carries the gap."
            if inside
            else ", so the near-X-point reading is held by the promotion rather than solved."
        )
    )
    rule = (
        "A change to that reading belongs in the promotion rule: "
        "<code>ForwardFluxOperator.residual_shadow_mask</code> at "
        "<code>nova/equilibrium/forward_operator.py:3699</code>, which composes the "
        "mask, and the paste it feeds at "
        "<code>nova/equilibrium/reduced_newton.py:985</code> inside "
        "<code>trip_boundary</code> (<code>reduced_newton.py:935</code>)."
    )
    return f"""<h2>Promoted nodes near the X-point</h2>
<p>Re-solves the converged MAST {receipt["shot"]}/{receipt["slice_index"]} boundary-coil edit
(fraction {receipt["edit_fraction"]:+.2f} on circuit {receipt["coil_circuit_index"]}) through
<code>benchmarks/coil_edit_latency.py</code>'s own preparation, so the state measured here is the
sweep's edit {receipt["edit_index"]}, closing at {_sci(receipt["terminal_residual_wb"])} Wb. The
solve promotes <strong>{receipt["promoted_node_count"]}</strong> of
{receipt["lattice_node_count"]} lattice nodes on its closing trip, {counts.get("private flux", 0)}
of them in the private-flux region and {counts.get("wall shadow (outside material)", 0)} on the
private-wall shadow, out of {receipt["physical_node_count"]} physical nodes
({receipt["wall_node_count"]} of them wall nodes).</p>

<p>Every promoted node is selected by <code>ForwardFluxOperator.residual_shadow_mask</code>
(<code>nova/equilibrium/forward_operator.py:3699</code>), which composes the interior-flood shadow
of the solve's own topology read with the private-wall node mask. The active-set trip calls it on
its reconstructed state in <code>trip_boundary</code>
(<code>nova/equilibrium/reduced_newton.py:935</code>) and keeps the trial state on exactly those
nodes at <code>nova/equilibrium/reduced_newton.py:985</code>:
<code>mapped = jnp.where(promoted, state, image)</code>. Domain counts over the promoted set:
{", ".join(f"{name} {count}" for name, count in counts.items())}.</p>

<p>The field of the currents is evaluated on
<code>nova.equilibrium.display_flux.build_display_flux_map</code> itself, handed the solve's own
promoted mask and the constrained row's compensating conductor current
(<strong>{compensation["compensating_current_norm_a"]:.3f} A</strong>, carrying up to
{_sci(compensation["max_abs_flux_on_physical_nodes_wb"])} Wb on the physical nodes), so the
saddle, the separatrix and the node mismatch below are all read off one complete field. That
build reports {complete["promoted_node_count"]} promoted lattice nodes, a reproduction error of
{_sci(complete["reproduction_relative_error"])} against the operator's own raster, and a saddle
offset of {complete["saddle_offset_mm"]:.1f} mm between the map and the lattice spline.</p>

<figure>
  <img src="{figure_src}" alt="Promoted nodes on the converged coil edit's poloidal field">
  <figcaption>The converged edit {receipt["edit_index"]} at
  {_sci(receipt["terminal_residual_wb"])} Wb. Grey contours: the converged state's poloidal flux.
  Dashed near-black: the separatrix assembled on the converged state; solid blue: the separatrix
  assembled on the complete field of the currents. Orange diamonds: the
  {receipt["promoted_node_count"]} promoted nodes. The solve's magnetic axis is the solid
  triangle; the two polished saddles are both drawn as X-point glyphs, the converged state's in
  near-black and the field of the currents' in blue. The right-hand window magnifies the
  <strong>{moves:.1f} mm</strong> between those two saddles, which is under one pixel of the
  whole-machine panel; its scale bar is 10 mm. Lattice spacing {spacing_mm:.1f} mm.</figcaption>
</figure>

{_table(promoted)}

<p>Distances to the polished X-point over the {receipt["promoted_node_count"]} promoted nodes:
nearest <strong>{quantiles["q0"]:.1f} mm</strong>, lower quartile {quantiles["q25"]:.1f} mm,
median <strong>{quantiles["q50"]:.1f} mm</strong>, upper quartile {quantiles["q75"]:.1f} mm,
furthest <strong>{quantiles["q100"]:.1f} mm</strong>. The nearest sits
{quantiles["q0"] / spacing_mm:.2f} lattice cells from the saddle.</p>

<p>On the complete field, image &minus; state is
<strong>{_sci(difference["promoted_max_abs_wb"])} Wb</strong> at the worst promoted node, with an
rms of {_sci(difference["promoted_rms_wb"])} Wb over that set. Over the nodes the solve does
claim the same difference is {_sci(difference["solved_max_abs_wb"])} Wb, and over the whole
lattice {_sci(difference["whole_lattice_max_abs_wb"])} Wb. The solve's own residual is
{_sci(receipt["terminal_residual_wb"])} Wb, so the promoted set carries
{difference["promoted_max_abs_wb"] / max(receipt["terminal_residual_wb"], 1e-30):.0f} times the
residual the field reproduces everywhere else.</p>

<p>Re-polishing the X-point on that complete field moves the saddle
<strong>{moves:.1f} mm</strong> (<var>R</var> {x_point["displacement_r_mm"]:+.1f} mm,
<var>Z</var> {x_point["displacement_z_mm"]:+.1f} mm; <strong>{cells:.2f} lattice cells</strong>
of {spacing_mm:.1f} mm). The separatrix assembled on that field differs from the converged
state's by <strong>{separatrix["lobe_direction_shift_deg"]:.2f}&deg;</strong> in lobe direction and
{legs}&deg; in the leg directions, with the lobe tip
{separatrix["lobe_tip_shift_mm"]:.1f} mm away.</p>

<p>{verdict}</p>

<p>{rule}</p>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    parser.add_argument("--figure-src", required=True)
    arguments = parser.parse_args()
    receipt = json.loads(arguments.receipt.read_text(encoding="utf-8"))
    arguments.fragment.parent.mkdir(parents=True, exist_ok=True)
    arguments.fragment.write_text(
        compose(receipt, arguments.figure_src), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "fragment": str(arguments.fragment),
                "rows": len(receipt["promoted"]),
                "x_point_mm": receipt["x_point"]["displacement_mm"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
