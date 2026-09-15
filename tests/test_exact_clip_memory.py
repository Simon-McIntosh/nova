"""Focused contracts for exact-clip memory attribution."""

import pytest

from benchmarks import exact_clip_memory_scaling as memory_scaling


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
    receipt = memory_scaling.measure(tmp_path / "receipt.json", tmp_path, [110])
    assert receipt["completed"] is True
