"""Tests for the scattered-field renderer scanner.

The scanner reports renderer calls that contour a field interpolated from
scattered nodes without masking it to the wall. The expectations below pin the
rule on named controls and on the base-pinned site count over the two renderer
trees.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import renderer_wall_scan as scan  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROOTS = ["benchmarks", "docs/figures"]

# The reviewed inventory of wall-unmasked renderer calls, keyed by base name.
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

# Sites the rule reports at this base which the inventory (taken before these
# files existed) does not carry: an unguarded painter in each of the two
# ``centroid-constrained-oracle-solve/repaired-remeasure*`` render scripts,
# whose raster comes from ``certificate._raster_field`` and which pass no
# ``wall``. The two records share one (base name, line) key.
INVENTORY_DRIFT_COUNT = 2


def _records():
    return scan.scan_paths([str(REPO / root) for root in ROOTS])


def _sites(records):
    return {(record.path, record.line) for record in records}


def _inventory_sites():
    return {
        (name, line)
        for name, lines in INVENTORY.items()
        for line in lines
    }


def test_scanner_is_importable_and_typed():
    record = scan.UnmaskedCall("a.py", 3, "tricontour")
    assert str(record) == "a.py:3: tricontour"


def test_flags_certificate_painter_without_wall():
    assert ("plasma_cell_fixed_point_attribution.py", 424) in _sites(_records())


def test_flags_direct_tricontour_on_scattered_nodes():
    assert ("sol_ledger_current_census.py", 415) in _sites(_records())


def test_does_not_flag_painter_that_passes_a_wall_argument():
    assert ("render.py", 52) not in _sites(_records())


def test_does_not_flag_painter_fed_by_a_pre_blanked_raster():
    assert ("contour_tree_explanatory_figure.py", 359) not in _sites(_records())


def test_does_not_flag_post_filtered_probe():
    assert ("plasma_cell_trip_panels.py", 406) not in _sites(_records())


def test_reproduces_every_inventory_site():
    missing = _inventory_sites() - _sites(_records())
    assert missing == set(), f"inventory sites not reproduced: {sorted(missing)}"


def test_base_count_is_inventory_plus_disclosed_drift():
    records = _records()
    sites = _sites(records)
    extras = sites - _inventory_sites()
    assert extras == {("measure.py", 70)}
    assert len(records) == len(_inventory_sites()) + INVENTORY_DRIFT_COUNT


def test_clean_tree_reports_nothing(tmp_path):
    (tmp_path / "render.py").write_text(
        "from nova.media.poloidal import draw_scattered_contours\n"
        "draw_scattered_contours(ax, r, z, f, levels, wall=wall)\n"
    )
    assert scan.scan_paths([str(tmp_path)]) == []


def test_cli_exits_nonzero_when_sites_exist():
    process = subprocess.run(
        [sys.executable, str(REPO / "tests" / "renderer_wall_scan.py"),
         str(REPO / "benchmarks")],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 1
    assert "plasma_cell_fixed_point_attribution.py:424" in process.stdout


def test_cli_exits_zero_on_a_clean_tree(tmp_path):
    (tmp_path / "clean.py").write_text("x = np.ones(3)\n")
    process = subprocess.run(
        [sys.executable, str(REPO / "tests" / "renderer_wall_scan.py"), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0
    assert process.stdout == ""