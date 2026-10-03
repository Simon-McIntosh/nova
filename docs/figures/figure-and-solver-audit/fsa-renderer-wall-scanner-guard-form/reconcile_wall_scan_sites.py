"""Reconcile the wall-unmasked renderer inventory against the current tree.

The reviewed inventory
(``.config/reckon/crew/reports/nova/s22/fsa-renderer-call-sites-inside-wall.md``,
reviewed 92) named 78 call sites by base name and line. This report prints the
scanner's current list with repository-relative paths, then states which
inventory sites are no longer reported (fixed by a repair batch) and which
current sites the inventory does not carry. It always exits zero: it is an
instrument a reader runs to see drift, not a gate a repair can trip.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tests"))

import renderer_wall_scan as scan  # noqa: E402

ROOTS = ("benchmarks", "docs/figures")

# The reviewed inventory, keyed by base name and line as the reviewed report
# recorded it. Two files share the base name ``measure.py`` at line 70; they are
# distinguished below by repository-relative path.
INVENTORY = {
    "unit_amplitude_current_census.py": {681, 689},
    "sol_current_demonstration.py": {494, 527, 535},
    "sol_ledger_current_census.py": {415},
    "solovev_solver_discriminator.py": {542, 550},
    "solovev_off_axis_census.py": {427},
    "solovev_cut_cell_moments.py": {804, 812, 820},
    "efit_forward_parity_slice.py": {3338, 3437, 3684},
    "limited_shadow_before_after.py": {760, 425, 687, 696, 738, 748},
    "xpoint_cell_allocation_rca.py": {1493, 1531},
    "centroid_constrained_fixture_receipt.py": {1011, 1014, 1086, 1097},
    "centroid_field_response_probe.py": {259, 262},
    "diverted_chord_response_attribution.py": {200},
    "exact_clip_low_state_discriminator.py": {683, 686, 690},
    "exact_clip_seed_amplitude.py": {345, 349, 429, 432, 438},
    "exact_support_floor_attribution.py": {724, 755},
    "fixture_positional_stiffness.py": {580, 583},
    "limited_row_shadow_census.py": {739, 977, 1061, 1064},
    "null_census_assertion.py": {419},
    "oracle_start_newton_probe.py": {1819, 1822},
    "plasma_cell_first_step_directions.py": {203},
    "plasma_cell_fixed_point_attribution.py": {424, 819},
    "plasma_cell_map_fidelity.py": {526},
    "plasma_cell_production_ladder.py": {736},
    "plasma_cell_seed_policy.py": {489},
    "plasma_cell_terminal_state.py": {357},
    "plasma_cell_trip_panels.py": {517},
    "shafranov_combination_discriminator.py": {914},
    "shafranov_pair_receipt.py": {293, 309, 877},
    "solovev_certificate.py": {1967, 2045, 2048},
    "zero_residual_check.py": {593, 634},
    "solve_program_size_gate.py": {1541, 1544},
    "draw_trips.py": {72},
    "measure.py": {215, 299, 308, 599},
    "render_mechanism_evidence.py": {363, 366, 511, 516, 868},
}

# Sites the inventory predates: an unguarded painter in each of the two
# ``centroid-constrained-oracle-solve/repaired-remeasure*`` render scripts,
# whose raster comes from ``certificate._raster_field`` and which pass no wall.
# Both files were absent at the inventory revision and present at base.
POST_INVENTORY = {
    "docs/figures/centroid-constrained-oracle-solve/repaired-remeasure/measure.py": {
        70
    },
    "docs/figures/centroid-constrained-oracle-solve/repaired-remeasure-receipt"
    "/measure.py": {70},
}


def main() -> int:
    records = scan.scan_paths([str(REPO / root) for root in ROOTS])
    current = {(record.path, record.line) for record in records}

    print(f"current unguarded sites: {len(records)}")
    for record in records:
        print(f"  {record.path}:{record.line}: {record.kind}")

    inventory = {(name, line) for name, lines in INVENTORY.items() for line in lines}
    current_short = {(Path(path).name, line) for path, line in current}
    fixed = sorted(inventory - current_short)
    print(f"\ninventory sites no longer reported (repaired or moved): {len(fixed)}")
    for name, line in fixed:
        print(f"  {name}:{line}")

    extras = sorted(
        (path, line)
        for path, line in current
        if (Path(path).name, line) not in inventory
    )
    print(f"\ncurrent sites absent from the inventory: {len(extras)}")
    for path, line in extras:
        post = POST_INVENTORY.get(path) == {line}
        tag = " (disclosed post-inventory)" if post else ""
        print(f"  {path}:{line}{tag}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
