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
    spacing_mm = 1.0e3 * receipt["lattice_spacing_m"]
    moves = x_point["displacement_mm"]
    cells = moves / spacing_mm if spacing_mm else float("nan")
    verdict = (
        "The converged near-X-point geometry is the field of the currents' own: "
        f"re-polishing the saddle on the image alone moves it {moves:.1f} mm, "
        f"{cells:.2f} lattice cells, and turns the lobe direction "
        f"{separatrix['lobe_direction_shift_deg']:.2f}&deg;."
        if moves < spacing_mm
        else "The promotion moves the near-X-point geometry: "
        f"re-polishing the saddle on the image alone moves it {moves:.1f} mm, "
        f"{cells:.2f} lattice cells."
    )
    return f"""<h2>Promoted nodes near the X-point</h2>
<p>Re-solves the converged MAST {receipt["shot"]}/{
        receipt["slice_index"]
    } boundary-coil edit
(fraction {receipt["edit_fraction"]:+.2f} on circuit {receipt["coil_circuit_index"]})
through <code>benchmarks/coil_edit_latency.py</code>'s own preparation, so the
state measured here is the sweep's edit {receipt["edit_index"]}, closing at
{_sci(receipt["terminal_residual_wb"])} Wb. The solve promotes
<strong>{receipt["promoted_node_count"]}</strong> of {receipt["lattice_node_count"]}
lattice nodes on its closing trip.</p>

<p>Every promoted node is selected by
<code>ForwardFluxOperator.residual_shadow_mask</code>
(<code>nova/equilibrium/forward_operator.py</code>, the method that composes the
interior-flood and private-wall shadows), which the active-set trip calls on its
own reconstructed state at <code>nova/equilibrium/reduced_newton.py</code> in
<code>trip_boundary</code>. The trip then keeps its trial state on exactly those
nodes with <code>mapped = jnp.where(promoted, state, image)</code>. The flood
component is the private-flux mask of the solve's own topology read; the wall
component is the private-wall node mask that read derives. Domain counts over
the promoted set: {", ".join(f"{name} {count}" for name, count in counts.items())}.</p>

<figure>
  <img src="{
        figure_src
    }" alt="Promoted nodes on the converged coil edit's poloidal field">
  <figcaption>The converged edit {receipt["edit_index"]} at
  {_sci(receipt["terminal_residual_wb"])} Wb. Grey contours: the converged
  state's poloidal flux. Dashed near-black: the separatrix the state assembles;
  blue: the separatrix the field of the currents assembles. Markers: the
  {receipt["promoted_node_count"]} promoted nodes. Both polished saddles are
  drawn. Lattice spacing {spacing_mm:.1f} mm.</figcaption>
</figure>

{_table(promoted)}

<p>The map of the currents misses the converged state by
<strong>{_sci(difference["whole_lattice_max_abs_wb"])} Wb</strong> over the whole
lattice, and by <strong>{_sci(difference["promoted_max_abs_wb"])} Wb</strong>
over the {receipt["promoted_node_count"]} promoted nodes alone. Over the nodes
the solve does claim the same difference is
{_sci(difference["solved_max_abs_wb"])} Wb, which is the solve's own residual
({_sci(receipt["terminal_residual_wb"])} Wb) rather than a geometry: the
promotion, not the field, carries the gap.</p>

<p>Re-polishing the X-point on the field of the currents alone moves it
<strong>{x_point["displacement_mm"]:.1f} mm</strong>
(<var>R</var> {x_point["displacement_r_mm"]:+.1f} mm,
<var>Z</var> {x_point["displacement_z_mm"]:+.1f} mm; {cells:.2f} lattice cells of
{spacing_mm:.1f} mm). The separatrix assembled on that field differs from the
converged state's by <strong>{separatrix["lobe_direction_shift_deg"]:.2f}&deg;</strong>
in lobe direction and {
        ", ".join(f"{value:.2f}" for value in separatrix["leg_direction_shifts_deg"])
        or "no"
    } in the leg directions, with the lobe tip
{separatrix["lobe_tip_shift_mm"]:.1f} mm away.</p>

<p>{verdict} A repair would change the promotion rule in
<code>nova/equilibrium/reduced_newton.py</code>: the
<code>jnp.where(promoted, state, image)</code> paste in <code>trip_boundary</code>
and the mask it is given, which is
<code>ForwardFluxOperator.residual_shadow_mask</code> in
<code>nova/equilibrium/forward_operator.py</code>.</p>
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
