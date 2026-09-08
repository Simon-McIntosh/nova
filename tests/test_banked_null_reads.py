"""Banked-read contract tests for the stationary-point receipt producers.

The three receipt producers (the MAST corroboration, the DIII-D forward match
gate, and the Solovev certificate) each bank, per row, Nova's axis and X-point
positions and flux beside the reference counterpart, the admitted O and X
candidate counts, the retained flux margin to the second-best candidate on
each type, and the read status with its exception text on failure.  The
shared helpers live in ``benchmarks/diiid_forward_gs_match.py``.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.diiid_forward_gs_match import (
    banked_failed_read_summary,
    banked_read_summary,
    candidate_flux_margins,
)


class _FakeOperator:
    """Fake operator carrying a census table over a 3x3 physical lattice."""

    def __init__(self, rows) -> None:
        self.topology = _FakeTopology()
        self._fixed_design_topology = _FakeTable(rows)


class _FakeTopology:
    """The split seam the census summary reads through."""

    def split_flux_map(self, physical):
        return jnp.asarray(physical), jnp.zeros(0)


class _FakeTable:
    """Census table whose status is the supplied fixed-shape census dict."""

    def __init__(self, rows) -> None:
        self.grid = _FakeGrid(rows)


class _FakeGrid:
    """Grid whose status reports the synthetic candidate census."""

    def __init__(self, rows) -> None:
        self._rows = rows

    def candidate_table_status(self, physical):
        valid = jnp.asarray(
            [
                [True, True, True, False, False, False, False, False, False, False],
                [True, True, False, False, False, False, False, False, False, False],
            ]
        )
        multiplicity = jnp.asarray(
            [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
        )
        origins = jnp.asarray(
            [[0, 1, 2, 0, 0, 0, 0, 0, 0, 0], [3, 4, 0, 0, 0, 0, 0, 0, 0, 0]]
        )
        return {
            "candidate_count": jnp.asarray((3, 2)),
            "capacity": jnp.asarray((30, 30)),
            "truncated": jnp.asarray((False, False)),
            "raw_ring_count": jnp.asarray((3, 2)),
            "polished_count": jnp.asarray((3, 2)),
            "typed_count": jnp.asarray((3, 2)),
            "same_root_count": jnp.asarray((3, 2)),
            "retained_count": jnp.asarray((3, 2)),
            "retained_multiplicity": multiplicity,
            "retained_representative_origin_index": origins,
            "retained_candidate": jnp.asarray(self._rows, dtype=jnp.float64),
            "retained_valid": valid,
            "ring_admitted_mask": jnp.asarray(
                [
                    [True, True, True, False, False, False],
                    [False, False, False, True, True, False],
                ]
            ),
            "census_slots_exhausted": jnp.asarray(False),
            "overflow": jnp.asarray((False, False)),
        }


def _synthetic_census() -> jax.Array:
    """Three O candidates and two X candidates on one shared flux scale.

    Rows are ``(R, Z, psi, kind)`` in the census ordering: extrema first with
    psi 1.0, 0.9, 0.8 (polarity +1 best is 1.0), saddles with psi 0.5, 0.4.
    """
    o_rows = np.asarray(
        ((0.3, 0.0, 0.8, 1.0), (0.5, 0.0, 1.0, 1.0), (0.7, 0.0, 0.9, 1.0))
    )
    x_rows = np.asarray(((0.5, 0.3, 0.5, 0.0), (0.5, -0.3, 0.4, 0.0)))
    padded = np.asarray(
        [
            np.pad(o_rows, ((0, 7), (0, 0)), constant_values=0.0),
            np.pad(x_rows, ((0, 8), (0, 0)), constant_values=0.0),
        ]
    )
    return jnp.asarray(padded)


def test_candidate_flux_margins_ranks_within_type_by_signed_flux():
    margins = candidate_flux_margins(
        _FakeOperator(_synthetic_census()), np.zeros(9), polarity=1.0
    )
    # best O psi = 1.0, second = 0.9 -> margin 0.1
    assert margins["o_candidate_count"] == 3
    assert margins["o_second_best_flux_margin_wb"] == pytest.approx(0.1)
    # best X psi = 0.5, second = 0.4 -> margin 0.1
    assert margins["x_candidate_count"] == 2
    assert margins["x_second_best_flux_margin_wb"] == pytest.approx(0.1)


def test_candidate_flux_margins_flips_ordering_under_negative_polarity():
    margins = candidate_flux_margins(
        _FakeOperator(_synthetic_census()), np.zeros(9), polarity=-1.0
    )
    # signed = -psi; best O psi 0.8, second 0.9 -> margin 0.1
    assert margins["o_second_best_flux_margin_wb"] == pytest.approx(0.1)


def test_banked_read_summary_emits_strict_json_block():
    operator = _FakeOperator(_synthetic_census())
    block = banked_read_summary(
        operator,
        np.zeros(9),
        axis_rz_m=np.asarray((0.5, 0.0)),
        x_point_rz_m=np.asarray((0.5, 0.3)),
        axis_flux_wb=1.0,
        boundary_flux_wb=-0.5,
        reference_axis_rz_m=np.asarray((0.51, 0.0)),
        reference_x_points_rz_m=np.asarray(((0.49, 0.3), (0.52, -0.3))),
        read_status="qualified_axis",
    )
    assert block["nova_axis_rz_m"] == [0.5, 0.0]
    assert block["nova_axis_flux_wb"] == pytest.approx(1.0)
    assert block["nova_x_point_rz_m"] == [0.5, 0.3]
    assert block["reference_x_points_rz_m"] == [
        [0.49, 0.3],
        [0.52, -0.3],
    ]
    assert block["o_candidate_count"] == 3
    assert block["x_candidate_count"] == 2
    assert block["read_status"] == "qualified_axis"
    assert block["read_exception_text"] is None
    json.dumps(block, allow_nan=False)


def test_banked_failed_read_summary_carries_exception_text():
    block = banked_failed_read_summary(
        axis_rz_m=None,
        x_point_rz_m=None,
        axis_flux_wb=float("nan"),
        reference_axis_rz_m=np.asarray((0.5, 0.0)),
        reference_x_points_rz_m=None,
        read_status="NoQualifiedAxisError",
        read_exception_text="synthetic axis disqualification",
    )
    assert block["nova_axis_rz_m"] is None
    assert block["o_candidate_count"] is None
    assert block["x_second_best_flux_margin_wb"] is None
    assert block["read_status"] == "NoQualifiedAxisError"
    assert block["read_exception_text"] == "synthetic axis disqualification"
    serialized = json.dumps(block, allow_nan=False)
    assert "synthetic axis disqualification" in serialized
