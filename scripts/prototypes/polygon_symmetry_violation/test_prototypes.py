"""Behavioural checks on the symmetry-diagnosis prototypes.

These run without JAX: every function under test is plain numpy over arrays the
caller supplies, so the measurements the evidence fragment quotes can be
re-derived from a known input instead of only from a job log.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.prototypes.polygon_symmetry_violation.compare import (
    central,
    central_step,
    psi_identity,
)
from scripts.prototypes.polygon_symmetry_violation import arms, mutation_control
from scripts.prototypes.polygon_symmetry_violation.split import (
    mirror_gap,
    paired_cells,
    print_report,
    print_rows,
    symmetric_cells,
)

STEPS = (1e-3, 1e-6)


def records_for(entries, steps=STEPS):
    """A plan of direct and psi ladder records, indexed as the runners expect:
    the direct records come first, exactly as ``plan`` emits them."""
    records = []
    for index in range(entries):
        records.append(dict(arm="direct", entry=index, index=len(records), step=0.0))
    for index in range(entries):
        for step in steps:
            for sign in (1.0, -1.0):
                records.append(
                    dict(
                        arm="psi",
                        entry=index,
                        index=len(records),
                        step=sign * step,
                    )
                )
    return records


def values_with(gap_entries, *, cells=1, amount=3e-6):
    """Current and base row archives that differ by ``amount`` at chosen entries."""
    records = records_for(gap_entries)
    shape = (len(records), cells, 3, 3)
    current = np.zeros(shape)
    base = np.zeros(shape)
    for record in records:
        if record["arm"] == "psi":
            current[record["index"], 0, 1, 0] = 1.0
            base[record["index"], 0, 1, 0] = 1.0 - amount
    return dict(current=current, base=base), records


def test_psi_identity_examines_every_ladder_record():
    values, records = values_with(2)
    entries = [dict(cell=0, moment=1), dict(cell=1, moment=1)]
    gap, examined = psi_identity(values, records, entries, {0: 0, 1: 0})
    assert examined == 2 * len(STEPS) * 2
    assert gap == pytest.approx(3e-6)


def test_psi_identity_does_not_stop_at_the_first_direct_record():
    """The failure this guards: a break on the arm test leaves the gap at zero."""
    values, records = values_with(1)
    assert records[0]["arm"] == "direct"
    entries = [dict(cell=0, moment=1)]
    gap, examined = psi_identity(values, records, entries, {0: 0})
    assert gap > 0.0, "the loop never reached a ladder record"
    assert examined == len(STEPS) * 2


def test_psi_identity_refuses_a_vacuous_measurement():
    values, records = values_with(0)
    entries = []
    with pytest.raises(SystemExit, match="examined 0"):
        psi_identity(values, records, entries, {})


def test_psi_identity_is_zero_without_a_base_tree():
    values, records = values_with(1)
    gap, examined = psi_identity(
        dict(current=values["current"]), records, [dict(cell=0)], {0: 0}
    )
    assert (gap, examined) == (0.0, 0.0)


def test_central_step_pairs_one_magnitude():
    ladder = [(1e-3, 2.0), (-1e-3, 0.0), (1e-6, 1.0), (-1e-6, 1.0)]
    assert central_step(ladder, 1e-3) == pytest.approx(1000.0)
    assert central_step(ladder, 1e-6) == pytest.approx(0.0)
    assert central_step(ladder, 1e-9) is None


def test_central_returns_one_row_per_paired_magnitude_coarsest_first():
    ladder = [(1e-3, 2.0), (-1e-3, 0.0), (1e-6, 1.0), (-1e-6, 1.0)]
    rows = central(ladder)
    assert [row["step"] for row in rows] == [1e-3, 1e-6]
    assert rows[0]["derivative"] == pytest.approx(1000.0)


def test_print_rows_emits_one_line_per_entry(capsys):
    entries = [
        dict(
            node=node,
            cell=0,
            moment=1,
            repaired_value=float(node),
            base_value=0.0,
            dz=0.0,
            mirror_gap=0.0,
            distance_to_cell_boundary=1.0,
        )
        for node in (11, 12, 13)
    ]
    print_rows("WORST_B", entries)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "WORST_B"
    assert len(lines) == 1 + len(entries)
    assert [json.loads(line)["node"] for line in lines[1:]] == [11, 12, 13]


def square(centre_z, radius=0.5):
    return np.array(
        [
            [-radius, centre_z - radius],
            [radius, centre_z - radius],
            [radius, centre_z + radius],
            [-radius, centre_z + radius],
        ]
    )


def test_mirror_gap_measures_the_departure_about_the_cell_axis():
    assert mirror_gap(square(0.25), 0.25) == pytest.approx(0.0)


def test_mirror_gap_reports_how_far_a_skewed_cell_departs():
    assert mirror_gap(skewed(0.0), 0.0) > 0.01


def skewed(centre_z, radius=0.5):
    cell = square(centre_z, radius)
    cell[0, 1] -= 0.05
    return cell


def test_symmetric_cells_is_one_gap_per_cell_and_reports_departures():
    geometry = dict(
        vertices=np.array([square(0.0), square(1.0), skewed(2.0)]),
        vertex_count=np.array([4, 4, 4]),
        centre=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]),
    )
    gaps = symmetric_cells(geometry)
    assert gaps.shape == (3,)
    assert gaps[0] == 0.0 and gaps[1] == 0.0
    assert gaps[2] > 0.01, "a skewed cell must report a departure, not a zero"


def test_paired_cells_asks_the_kernel_reflection():
    geometry = dict(
        vertices=np.array([square(0.0), skewed(2.0)]),
        vertex_count=np.array([4, 4]),
        centre=np.array([[0.0, 0.0], [0.0, 2.0]]),
    )
    assert list(paired_cells(geometry)) == [True, False]


def entry(node, value):
    return dict(
        node=node,
        cell=0,
        moment=1,
        repaired_value=value,
        base_value=0.0,
        dz=0.0,
        mirror_gap=0.0,
        distance_to_cell_boundary=1.0,
    )


def test_print_report_emits_rows_under_both_headers(capsys):
    """The call site, so a bare header print cannot come back unnoticed."""
    print_report([entry(11, 1e-6)], [entry(12, 2e-6), entry(13, 3e-6)])
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[0] == "VIOLATIONS_A"
    worst = lines.index("WORST_B")
    assert len(lines[1:worst]) == 1
    assert len(lines[worst + 1 :]) == 2
    assert [json.loads(line)["node"] for line in lines[worst + 1 :]] == [12, 13]


def test_red_arms_names_every_red_arm_but_an_expected_absence():
    table = dict(
        base=dict(exit_status=4, absent=True),
        head=dict(exit_status=0, absent=False),
    )
    assert arms.red_arms(table, {"base"}) == []
    table["head"]["exit_status"] = 1
    assert arms.red_arms(table, {"base"}) == ["head"]
    table["base"]["exit_status"] = 4
    assert arms.red_arms(table, set()) == ["base", "head"]


def test_control_exits_zero_only_when_the_guard_separates():
    assert mutation_control.run([]) == 0
    assert mutation_control.run(["--repaired"]) == 1
