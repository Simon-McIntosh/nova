"""Driven-coil bank padding, persistence, and independent map qualification."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.biot import polygonanalytic as kernel
from nova.equilibrium import support_read, topology
from nova.equilibrium.fixed_points import _source_jets
from scripts.prototypes.iter_bank.measure import (
    PLASMA_JET_TOLERANCE,
    ZeroField,
    assert_accuracy,
    build_geometry,
    pack,
)
from scripts.prototypes.read_route.jets import build_jet_matrix


def analytic_jets(points, coupling, moments):
    """Evaluate the direct kernels without using the bank's response rows."""
    with jax.disable_jit():
        return pack(
            jax.vmap(coupling.evaluate, in_axes=(0, None))(points, tuple(moments.T))
        )


@pytest.fixture(scope="module")
def iter_bank():
    """Build one ITER plasma cell with two native driven-coil sections.

    Coordinates and signed turns come from DINA 105027/1, DD 3.38.1:
    pf_active PF1 and CS3U, and the wall-clipped carrier cell nearest
    (6.2, 0) m built by scripts/prototypes/iter_bank/measure.py. The cell
    boundary is the sub-mesh wall. Only input geometry is embedded; all
    operator rows and reference jets are evaluated during the test run.
    """
    started = perf_counter()
    assert jax.config.jax_enable_x64 is True
    cell = np.array(
        [
            [6.336119448749177, -0.11632048214432768],
            [6.261779593268597, 0.012439924575365918],
            [6.113099882307435, 0.012439924575365932],
            [6.038760026826854, -0.1163204821443277],
            [6.113099882307435, -0.24508088886402132],
            [6.261779593268597, -0.24508088886402132],
        ],
        dtype=np.float64,
    )
    coils = (
        np.array(
            [
                [4.4226, 7.08165],
                [4.4226, 8.06575],
                [3.4636, 8.06575],
                [3.4636, 7.08165],
            ],
            dtype=np.float64,
        ),
        np.array(
            [
                [2.057, 4.4175],
                [2.057, 6.5105],
                [1.3170000000000002, 6.5105],
                [1.3170000000000002, 4.4175],
            ],
            dtype=np.float64,
        ),
    )
    turns = np.array([248.6, 554.0], dtype=np.float64)
    geometry = topology.TopologyGeometry.from_cells((cell,), cell[None], (cell,))
    sources = topology.BiotMomentCoupling.from_polygons((cell, *coils))
    # Pack the plasma and conductors together so one kernel pass supplies all
    # response rows, at one target shape with no second padded point batch.
    with jax.disable_jit():
        rows, exterior, nodes, indices, _ = build_jet_matrix(
            sources,
            geometry,
            ZeroField(),
            (jnp.zeros(3),) * 3,
            batch=7,
            persist=False,
        )
    assert len(nodes) == 7
    operands = support_read.SupportOperands.from_rows(
        geometry,
        rows[:, :1],
        exterior,
        nodes,
        indices,
        bucket=64,
        coil_exterior=rows[:, 1:, 0, :] * turns[None, :, None],
    )
    currents = jnp.array(
        [
            [-16500.40721212742, 3328.9667992184472],
            [-16431.81795701071, 4715.148567535232],
            [-15841.299036467031, 5546.237210862071],
        ],
        dtype=jnp.float64,
    )
    moments = jnp.array(
        [[[12000.0, 40.0, -90.0]], [[25000.0, -60.0, 80.0]], [[-14000.0, 70.0, 50.0]]],
        dtype=jnp.float64,
    )
    plasma = jnp.pad(moments, ((0, 0), (0, 2), (0, 0)))
    coils = jnp.zeros_like(plasma).at[:, 1:, 0].set(turns * currents)
    reference_moments = jnp.stack((plasma, coils, plasma + coils), axis=1)
    oracle_started = perf_counter()
    # All three fields and current states share the direct kernel evaluation.
    # This oracle contracts physical source moments before differentiation;
    # it neither reads nor contracts the bank's prebuilt response rows.
    expected = jax.vmap(
        jax.vmap(analytic_jets, in_axes=(None, None, 0)), in_axes=(None, None, 0)
    )(nodes, sources, reference_moments)
    jax.block_until_ready(expected)
    print("DIRECT_ORACLE_SECONDS", perf_counter() - oracle_started)
    print("BANK_FIXTURE_SECONDS", perf_counter() - started)
    return operands, moments, currents, expected


def bank(monkeypatch, coil_exterior=None):
    monkeypatch.setattr(support_read.fixed_points, "build_mesh", lambda *args: None)
    return support_read.SupportOperands.from_rows(
        None,
        np.ones((3, 2, 3, 6)),
        np.ones((3, 6)),
        np.ones((3, 2)),
        np.array([[0, 1, 2]]),
        bucket=64,
        coil_exterior=np.arange(36.0).reshape(3, 2, 6)
        if coil_exterior is None
        else coil_exterior,
    )


def test_coil_responses_are_padded_and_count_is_traced(monkeypatch):
    operands = bank(monkeypatch)
    assert operands.coil_exterior.shape == (64, 2, 6)
    assert int(operands.coil_count) == 2
    assert np.count_nonzero(np.asarray(operands.coil_exterior)[3:]) == 0
    program = jax.jit(lambda b: b.coil_exterior.sum() + b.coil_count)
    first = program(operands)
    second = program(replace(operands, coil_count=jnp.asarray(1, jnp.int32)))
    assert float(first - second) == 1
    assert program._cache_size() == 1


def test_coil_responses_survive_content_store_roundtrip(monkeypatch, tmp_path):
    operands = bank(monkeypatch)
    digest = support_read.write_bank(operands, tmp_path)
    restored = support_read.read_bank(tmp_path, digest, None, None)
    np.testing.assert_array_equal(restored.coil_exterior, operands.coil_exterior)
    assert int(restored.coil_count) == 2
    changed = replace(operands, coil_exterior=2 * operands.coil_exterior)
    assert support_read.write_bank(changed, tmp_path) != digest


def test_malformed_coil_response_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="shape"):
        bank(monkeypatch, np.zeros((2, 2, 6)))


def test_absent_coil_response_roundtrip_keeps_none(monkeypatch, tmp_path):
    operands = replace(bank(monkeypatch), coil_exterior=None, coil_count=None)
    digest = support_read.write_bank(operands, tmp_path)
    restored = support_read.read_bank(tmp_path, digest, None, None)
    assert restored.coil_exterior is None
    assert int(restored.coil_count) == 0


def test_plasma_hessian_budget_preserves_strict_values_and_coils():
    expected = np.ones((10, 6))
    actual = expected.copy()
    actual[:, 4:] += 2e-9
    assert_accuracy(actual, expected, PLASMA_JET_TOLERANCE)
    with pytest.raises(AssertionError, match="relative jet errors"):
        assert_accuracy(actual, expected)
    actual[:, 5] += 2e-8
    with pytest.raises(AssertionError, match="relative jet errors"):
        assert_accuracy(actual, expected, PLASMA_JET_TOLERANCE)
    actual = expected.copy()
    actual[:, 0] += 2e-9
    with pytest.raises(AssertionError, match="relative jet errors"):
        assert_accuracy(actual, expected, PLASMA_JET_TOLERANCE)


def test_map_check_refuses_nonfinite_and_missing_coils():
    expected = np.ones((10, 6))
    assert_accuracy(expected.copy(), expected)
    with pytest.raises(AssertionError, match="relative jet errors"):
        assert_accuracy(expected * 0.99, expected)
    with pytest.raises(AssertionError, match="relative jet errors"):
        assert_accuracy(np.full_like(expected, np.nan), expected)
    with pytest.raises(AssertionError, match="measurable scale"):
        assert_accuracy(expected, np.zeros_like(expected))


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("sample", range(3))
def test_iter_nodal_jets_match_analytic_map(iter_bank, sample):
    """Check the live plasma contraction alone and with runtime coil currents."""
    operands, moments, currents, expected = iter_bank
    active = int(operands.active_nodes)
    for field, drive in ((0, jnp.zeros(2)), (2, currents[sample])):
        actual = support_read.nodal_jets(operands, tuple(moments[sample].T), drive)
        errors = assert_accuracy(
            actual[:active], expected[sample, field], PLASMA_JET_TOLERANCE
        )
        np.testing.assert_array_equal(actual[active:], 0.0)
        print("plasma" if not bool(jnp.any(drive)) else "total", errors)


def test_reentrant_wall_mesh_keeps_all_clipped_fragments():
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    wall = np.array(
        [
            [2.0, 0.0],
            [4.0, 0.0],
            [4.0, 2.0],
            [3.2, 2.0],
            [3.2, 0.3],
            [2.8, 0.3],
            [2.8, 2.0],
            [2.0, 2.0],
        ]
    )
    geometry = build_geometry(wall, target=40)
    cells = [
        Polygon(p[:n])
        for p, n in zip(
            np.asarray(geometry.vertices),
            np.asarray(geometry.vertex_count),
            strict=True,
        )
    ]
    vessel = Polygon(wall)
    assert abs(unary_union(cells).area - vessel.area) < 1e-10
    assert all(p.difference(vessel).area < 1e-12 for p in cells)


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize("sample", range(3))
def test_iter_coil_jets_match_analytic_map(iter_bank, sample):
    """Qualify every driven jet entry against the direct analytic kernels."""
    operands, moments, currents, expected = iter_bank
    moments = jnp.zeros_like(moments[sample])
    active = int(operands.active_nodes)
    actual = support_read.nodal_jets(operands, tuple(moments.T), currents[sample])
    errors = assert_accuracy(actual[:active], expected[sample, 1])
    np.testing.assert_array_equal(actual[active:], 0.0)
    print("coil", errors)


def extended_line_source():
    """A plasma cell and a target 1e-12 m from one edge's extended line."""
    angle = np.pi / 6
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    polygon = np.array([[-0.5, -0.6], [0.5, -0.6], [0.5, 0.6], [-0.5, 0.6]])
    polygon = polygon @ rotation.T + [6, 0]
    edge = polygon[2] - polygon[1]
    normal = np.array([edge[1], -edge[0]]) / np.linalg.norm(edge)
    point = polygon[1] + 2 * edge + 1e-12 * normal
    return polygon, point


def primal_gradient(point, coupling):
    """Evaluate the analytic gradient without invoking any tangent rule."""
    rows = kernel.packed_analytic_moments(
        np,
        np.asarray([point[0]]),
        np.asarray([point[1]]),
        np.asarray(coupling.edge),
        np.asarray(coupling.weight),
        np.asarray(coupling.norm),
        np.asarray(coupling.centre.T),
        np.asarray(coupling.centre.T),
        np.asarray(coupling.reflection_axis),
        np.asarray(coupling.reflection_partner),
    )
    stacked = np.stack((np.stack(rows[6:9]), -np.stack(rows[3:6])), axis=-1)
    return 2 * np.pi * point[0] * stacked[:, 0]


def test_plasma_row_hessian_matches_source_gradient_reference():
    """The bank's plasma row carries a Hessian this edge line does not damage.

    The reference differences the primal analytic gradient at three steps and
    Richardson-extrapolates it, so it never reads the polygon tangent the row
    is built from.
    """
    polygon, point = extended_line_source()
    coupling = topology.BiotMomentCoupling.from_polygons((polygon,))
    moments = np.array([[240000.0, 0.0, 0.0]])
    row = np.asarray(_source_jets(jnp.asarray(point), coupling))
    actual = np.einsum("cm,cmq->q", moments, row)
    estimates = []
    for step in (0.001, 0.0005, 0.00025):
        columns = []
        for direction in np.eye(2):
            offset = step * direction
            columns.append(
                (
                    primal_gradient(point + offset, coupling)
                    - primal_gradient(point - offset, coupling)
                )
                / (2 * step)
            )
        estimates.append(np.stack(columns, axis=-1))
    fourth = [
        (4 * estimates[1] - estimates[0]) / 3,
        (4 * estimates[2] - estimates[1]) / 3,
    ]
    sixth = (16 * fourth[1] - fourth[0]) / 15
    reference = np.einsum("c,cab->ab", moments[0], sixth)
    mixed, vertical = reference[0, 1], reference[1, 1]
    errors = np.array(
        [abs(actual[4] - mixed) / abs(mixed), abs(actual[5] - vertical) / abs(vertical)]
    )
    print("plasma_row_hessian", actual[[4, 5]].tolist(), errors.tolist())
    assert np.all(errors <= 1e-8), f"plasma-row Hessian errors {errors.tolist()}"


# The derive_rows instrument backs every first-derivative figure the evidence
# fragment quotes. These cases pin the three things that figure depends on: the
# derivative slice it reads, the magnitude window its filtered statistic uses,
# and the threshold at which an entry is listed; plus the opt-in status that
# replaced the withdrawn 1e-7 boundary, and the single ownership of the
# node-aggregated norm the certification script also prints.

from scripts.prototypes.iter_bank import derive_rows, recertify  # noqa: E402


def stored_rows(nodes=1, cells=1, moments=1, value=1.0):
    """A row archive shaped like the published bank's, with every jet at value."""
    rows = np.full((nodes, cells, moments, 6), value, dtype=np.float64)
    return rows


def test_derive_rows_reads_the_two_first_derivative_components():
    rows = stored_rows(cells=2)
    rows[..., 1] = 10.0
    rows[..., 2] = 20.0
    head = rows.copy()
    head[..., 0] = 1.5
    head[..., 3] = 5.0
    head[..., 4] = 5.0
    head[..., 5] = 5.0
    report = derive_rows.compare(head, rows)
    assert report["entries"] == 4
    assert report["components"] == ["dpsi/dR", "dpsi/dZ"]
    assert report["departing_count"] == 0
    assert report["unfiltered"]["max_relative"] == 0.0
    head[..., 2] = 30.0
    report = derive_rows.compare(head, rows)
    assert report["departing_count"] == 2
    assert {row["component"] for row in report["departures"]} == {"dpsi/dZ"}
    assert report["dpsi/dR"]["above_threshold"] == 0
    assert report["dpsi/dZ"]["above_threshold"] == 2


def test_derive_rows_window_keeps_near_zero_magnitudes_out_of_the_filtered_max():
    rows = stored_rows(cells=2)
    rows[0, 1, 0, 1] = 1e-12
    head = rows.copy()
    head[0, 0, 0, 1] = 1.0 + 1e-9
    head[0, 1, 0, 1] = 1e-12 + 1e-16
    report = derive_rows.compare(head, rows)
    assert report["unfiltered"]["max_relative"] == pytest.approx(1e-4, rel=1e-6)
    assert report["filtered"]["max_relative"] == pytest.approx(1e-9, rel=1e-6)
    assert report["filtered"]["count"] == 2


def test_derive_rows_lists_only_the_entries_above_the_relative_threshold():
    rows = stored_rows(cells=3)
    head = rows.copy()
    head[0, 0, 0, 1] += 5e-11
    head[0, 1, 0, 1] += 2e-10
    head[0, 2, 0, 1] += 9e-11
    report = derive_rows.compare(head, rows)
    assert report["departing_count"] == 1
    assert report["departures"][0]["cell"] == 1
    assert report["filtered"]["above_threshold"] == 1


def test_derive_rows_fails_only_on_a_named_absolute_boundary():
    rows = stored_rows()
    head = rows.copy()
    head[0, 0, 0, 1] += 1e-6
    quiet = derive_rows.compare(head, rows)
    assert quiet["material"]["boundary"] is None
    assert quiet["material"]["present"] is None
    assert derive_rows.exit_status(quiet) == 0
    loud = derive_rows.compare(head, rows, boundary=1e-7)
    assert loud["material"]["present"] is True
    assert loud["material"]["count"] == 1
    assert derive_rows.exit_status(loud) == 2


def test_derive_rows_shares_the_certification_relative_norm():
    assert derive_rows.relative is recertify.relative
