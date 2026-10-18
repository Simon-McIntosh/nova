"""The boundary-index certificate that lets a zero-saddle slice be trusted.

The census proposes stationary roots only from the cells it can condition, so
an admitted-saddle count of zero is a claim about the field that the census
cannot support on its own. These tests pin the certificate that decides whether
such a slice may be read as limited, and pin the refusal where it may not: the
degree of grad psi on the domain boundary against the index sum of the admitted
critical points inside it.
"""

from __future__ import annotations

import os
from pathlib import Path

import imas
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import fixed_points, topology
from nova.equilibrium.topology import TopologyPolicy
from nova.jax.config import configure_dtypes
from scripts.prototypes.efit_field_swap import crosscheck
from scripts.prototypes.efit_field_swap.spline_field import TraceSplineField
from scripts.prototypes.read_robustness.census import RootCensus
from scripts.prototypes.tcv_liuqe_known_answer import measure

configure_dtypes()

CONVENTION = topology.TopologyConvention.from_cocos(17, 1.0)
POLICY = TopologyPolicy()

LIUQE_STORE = Path(
    os.environ.get("TCV_LIUQE_STORE", "/home/ITER/mcintos/public/imasdb/tcv/3/77549/1")
)
WALL_STORE = Path(
    os.environ.get(
        "TCV_WALL_STORE", "/home/ITER/mcintos/Code/imas-efit/tests/data/imas/TCV/77549"
    )
)
LIMITED_SLICE = 34

GRID = np.linspace(-1.0, 1.0, 129)
RR, ZZ = np.meshgrid(GRID, GRID, indexing="ij")
RING = np.asarray(
    [
        (-0.5, -0.5),
        (0.5, -0.5),
        (0.5, 0.0),
        (0.5, 0.5),
        (-0.5, 0.5),
        (-0.5, -0.5),
    ],
    dtype=float,
)
OBSTACLE = np.asarray(
    [(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2)], dtype=float
)


def _field(values):
    state = measure.build_state(GRID, GRID, values, (RING,), 0.0, 0.0)
    return TraceSplineField.from_record(state)


def _well():
    """One maximum at the origin and nothing else: a genuinely limited field."""
    return _field(-(RR**2 + ZZ**2))


def _maxima_and_saddle():
    """Two maxima at R = plus/minus 0.35 and a saddle at the origin.

    A loop round all three winds once, and the index sum is one as well: two
    extrema at +1 and one saddle at -1. A census that reports the extrema and
    misses the saddle therefore claims an index sum of two against a winding of
    one, which is the coverage miss the certificate exists to catch.
    """
    return _field(-((RR**2 - 0.35**2) ** 2 + ZZ**2))


def test_field_with_a_missed_saddle_is_refused_not_read_limited(monkeypatch):
    field = _maxima_and_saddle()
    winding = crosscheck.coverage_certificate(field, RING, (), 1)
    assert winding.total == 1.0
    assert winding.certified, "the field's own index sum closes the certificate"
    missed = crosscheck.coverage_certificate(field, RING, (), 2)
    assert missed.total == 1.0
    assert not missed.certified

    def blind_census(f, geometry, convention, policy, previous):
        """A census that admits both maxima and reports no saddle.

        Two negative-definite roots carry index +1 each, so the census is
        claiming an index sum of two where the boundary winding is one: the
        saddle it did not admit is the difference.
        """
        hessian = jnp.asarray([[[-1.0, 0.0], [0.0, -1.0]]] * 3)
        jet = topology.FieldJet(
            value=jnp.zeros(3), gradient=jnp.zeros((3, 2)), hessian=hessian
        )
        roots = fixed_points.FixedPointResult(
            position=jnp.asarray([[-0.35, 0.0], [0.35, 0.0], [0.0, 0.0]]),
            jet=jet,
            valid=jnp.asarray([True, True, False]),
            status=jnp.zeros(3, jnp.int32),
            iterations=jnp.zeros(3, jnp.int32),
            retried=jnp.zeros(3, bool),
        )
        return RootCensus(roots, 3, 2, 0, 0, False, 0.0)

    monkeypatch.setattr(crosscheck, "census", blind_census)
    outcome = crosscheck.read_state(field, RING, (), CONVENTION, POLICY, (16,))
    assert outcome.label == "coverage uncertified"
    assert outcome.resolution is None
    assert outcome.coverage is not None and not outcome.coverage.certified
    assert outcome.coverage.total != outcome.coverage.index_sum


def test_cancelled_missed_pair_is_refused(monkeypatch):
    field = _maxima_and_saddle()
    # The field contains two extrema and one saddle. Reporting just one
    # extremum leaves the same index as the true field despite two missed roots.
    boundary = crosscheck.coverage_certificate(field, RING, (), 1)
    assert boundary.certified and boundary.total == boundary.index_sum == 1

    def blind_census(f, geometry, convention, policy, previous):
        hessian = jnp.asarray([[[-1.0, 0.0], [0.0, -1.0]]])
        jet = topology.FieldJet(
            value=jnp.zeros(1), gradient=jnp.zeros((1, 2)), hessian=hessian
        )
        roots = fixed_points.FixedPointResult(
            position=jnp.asarray([[-0.35, 0.0]]),
            jet=jet,
            valid=jnp.asarray([True]),
            status=jnp.zeros(1, jnp.int32),
            iterations=jnp.zeros(1, jnp.int32),
            retried=jnp.zeros(1, bool),
        )
        return RootCensus(roots, 1, 1, 0, 0, False, 0.0)

    monkeypatch.setattr(crosscheck, "census", blind_census)
    outcome = crosscheck.read_state(field, RING, (), CONVENTION, POLICY, (16,))
    assert outcome.label == "coverage uncertified", outcome.attempts
    assert outcome.resolution is None
    assert outcome.coverage.total == outcome.coverage.index_sum
    assert outcome.coverage.cells_uncertified > 0
    assert not outcome.coverage.certified


def test_constant_cell_reaches_depth_cap_and_is_uncertified():
    field = _field(np.zeros_like(RR))
    cell = np.asarray([RING[0], RING[1], RING[3], RING[4]])
    cleared, unresolved, occupied = crosscheck.certify_empty_cells(
        field, [cell], [], depth=3
    )
    assert cleared == 0
    assert occupied == 0
    assert unresolved == 1


def test_invalid_axis_slot_cannot_be_read_as_limited(monkeypatch):
    field = _well()

    def blind_census(f, geometry, convention, policy, previous):
        jet = topology.FieldJet(
            value=jnp.zeros(1),
            gradient=jnp.zeros((1, 2)),
            hessian=jnp.asarray([[[-2.0, 0.0], [0.0, -2.0]]]),
        )
        roots = fixed_points.FixedPointResult(
            position=jnp.asarray([[0.2, 0.2]]),
            jet=jet,
            valid=jnp.asarray([False]),
            status=jnp.ones(1, jnp.int32),
            iterations=jnp.zeros(1, jnp.int32),
            retried=jnp.zeros(1, bool),
        )
        return RootCensus(roots, 1, 0, 0, 0, False, 0.0)

    monkeypatch.setattr(crosscheck, "census", blind_census)
    outcome = crosscheck.read_state(field, RING, (), CONVENTION, POLICY, (16,))
    assert outcome.label == "coverage uncertified", outcome.attempts
    assert outcome.resolution is None
    assert outcome.coverage is not None and not outcome.coverage.certified
    assert "axis admitted False" in outcome.attempts


def test_certified_limited_field_reads_limited():
    outcome = crosscheck.read_state(_well(), RING, (), CONVENTION, POLICY, (16,))
    assert outcome.coverage is not None and outcome.coverage.certified
    assert outcome.coverage.index_sum == 1.0
    assert outcome.label == "limited"


def test_two_loop_domain_counts_outer_winding_less_obstacle():
    coverage = crosscheck.coverage_certificate(_well(), RING, (OBSTACLE,), 0)
    assert coverage.outer_winding != 0.0
    assert coverage.obstacle_winding == coverage.outer_winding
    assert coverage.total == pytest.approx(
        coverage.outer_winding - coverage.obstacle_winding
    )
    assert coverage.total == pytest.approx(0.0)
    assert coverage.certified, "a per-loop equality would have accepted this domain"


def test_critical_point_on_the_loop_leaves_coverage_uncertified():
    field = _field(-((RR - 0.5) ** 2 + ZZ**2))
    coverage = crosscheck.coverage_certificate(field, RING, (), 1)
    assert not coverage.reliable
    assert not coverage.certified


@pytest.mark.slow
@pytest.mark.skipif(
    not (LIUQE_STORE / "equilibrium.h5").exists(), reason="LIUQE store not present"
)
def test_limited_slice_is_read_only_when_coverage_is_certified():
    db = imas.DBEntry(f"imas:hdf5?path={LIUQE_STORE}", "r", dd_version="3.41.0")
    try:
        view = measure.slice_view(db.get("equilibrium"), LIMITED_SLICE)
    finally:
        db.close()
    assert not view["xpoints"], "the chosen slice must be a limited one"
    reading = measure.measure_slice(view, measure.read_wall(str(WALL_STORE)), (24,))
    assert reading["nova_label"] == "limited", reading["attempts"]
    assert reading["coverage_certified"] is True, reading["attempts"]
    assert reading["coverage_cells_uncertified"] == 0
