"""Focused contracts for exact-clip memory attribution."""

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import exact_clip_memory_scaling as memory_scaling
from nova.equilibrium.clip_quadrature import clipped_support_current_moments
from nova.equilibrium.forward_operator import _cell_banked_current_moments
from nova.equilibrium.separatrix_clip import TracedClippedSupports
from nova.equilibrium.stencil_mesh import FluxFieldPolynomial


def test_scaling_exponent_identifies_linear_and_pairwise_growth():
    """Power-law attribution distinguishes per-cell from all-cell pairs."""
    assert memory_scaling.scaling_exponent(100, 300, 10, 30) == pytest.approx(1.0)
    assert memory_scaling.scaling_exponent(100, 900, 10, 30) == pytest.approx(2.0)


@pytest.mark.parametrize("value", [0, -1])
def test_scaling_exponent_refuses_nonpositive_measurements(value):
    """An empty or signed measurement cannot support a memory-law claim."""
    with pytest.raises(ValueError, match="positive"):
        memory_scaling.scaling_exponent(value, 300, 10, 30)


def test_measure_uses_the_current_quadrature_node_owner(monkeypatch, tmp_path):
    """The memory probe reaches compilation after the quadrature module split."""
    monkeypatch.delattr(memory_scaling.certificate.observation, "_UNIT_NODE", False)
    monkeypatch.setattr(memory_scaling, "configure_dtypes", lambda: None)
    monkeypatch.setattr(memory_scaling, "support_clip_mode", lambda: "chord")
    monkeypatch.setattr(memory_scaling, "set_support_clip_mode", lambda _mode: None)
    monkeypatch.setattr(memory_scaling.certificate, "_source_revision", lambda: "abc")
    monkeypatch.setattr(
        memory_scaling.certificate, "_lane", lambda: {"platform": "cpu"}
    )

    def compiled(*_args, **_kwargs):
        assert (
            memory_scaling.certificate.observation._UNIT_NODE
            is memory_scaling.clip_quadrature._UNIT_NODE
        )
        return {
            "requested_cells": -110,
            "realised_cells": 132,
            "memory_analysis": {"temp_size_in_bytes": 1024},
            "qualifying_array_signatures": [],
        }

    monkeypatch.setattr(memory_scaling.certificate, "_compile_solve_memory", compiled)
    part_root = tmp_path / "parts"
    receipt = memory_scaling.measure(
        tmp_path / "receipt.json", tmp_path, [110], part_root=part_root
    )
    assert receipt["completed"] is True
    part = memory_scaling.json.loads(
        (part_root / "requested-110.json").read_text(encoding="utf-8")
    )
    assert part["row"] == receipt["rows"][0]


class _ConstantCurrentProfile:
    def current_density(self, radius, psi_norm):
        return jnp.ones_like(radius) * 2.0 + 0.0 * psi_norm


def _three_cell_support() -> TracedClippedSupports:
    vertices = np.asarray(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [2.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
            [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]],
        ]
    )
    zeros = np.zeros(3)
    return TracedClippedSupports(
        support_vertices=vertices,
        vertex_count=np.asarray([4, 3, 4]),
        centroids=np.asarray([[0.5, 0.5], [4.0 / 3.0, 1.0 / 3.0], [2.5, 0.5]]),
        included=np.asarray([True, False, False]),
        boundary=np.asarray([False, True, False]),
        area=np.asarray([1.0, 0.5, 0.0]),
        full_area=np.asarray([1.0, 1.0, 1.0]),
        first_area_moment=np.zeros((3, 2)),
        second_area_moment=np.zeros((3, 2, 2)),
        contour_area=zeros,
        patch_area_sum=np.asarray(1.5),
        branch_support_vertices=np.zeros((3, 2, 4, 2)),
        branch_vertex_count=np.zeros((3, 2), dtype=np.int32),
        branch_area=np.zeros((3, 2)),
        branch_first_area_moment=np.zeros((3, 2, 2)),
        branch_second_area_moment=np.zeros((3, 2, 2, 2)),
        saddle=np.zeros(3, dtype=bool),
        saddle_vertex=np.zeros((3, 2)),
    )


def test_cell_banked_current_moments_are_bit_identical():
    """Per-cell cut integration retains the existing current-moment result."""
    support = _three_cell_support()
    field = FluxFieldPolynomial(
        coefficient=jnp.zeros((3, 6)),
        centre=jnp.asarray(support.centroids),
        scale=jnp.ones((3, 2)),
        active=jnp.ones(3, dtype=bool),
    )
    selection = jnp.asarray([True, True, False])
    profile = _ConstantCurrentProfile()
    expected = clipped_support_current_moments(
        support,
        selection,
        field,
        profile,
        cut_cell_capacity=3,
    )
    actual = _cell_banked_current_moments(
        support,
        selection,
        field,
        profile,
        cut_cell_capacity=3,
    )
    for one, other in zip(expected, actual, strict=True):
        assert np.array_equal(np.asarray(one), np.asarray(other))
