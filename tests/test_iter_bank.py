"""Driven-coil bank padding, persistence, and independent map qualification."""

# ruff: noqa: E402
from nova.jax.config import configure_dtypes

configure_dtypes()

from dataclasses import replace
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nova.equilibrium import support_read
from scripts.prototypes.iter_bank.measure import assert_accuracy, build_geometry


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


def test_map_check_refuses_nonfinite_and_missing_coils():
    expected = np.ones((10, 6))
    assert_accuracy(expected.copy(), expected)
    with pytest.raises(AssertionError, match="relative jet errors"):
        assert_accuracy(expected * 0.99, expected)
    with pytest.raises(AssertionError, match="relative jet errors"):
        assert_accuracy(np.full_like(expected, np.nan), expected)
    with pytest.raises(AssertionError, match="measurable scale"):
        assert_accuracy(expected, np.zeros_like(expected))


@pytest.mark.parametrize("sample", range(3))
def test_dina_nodal_jets_match_analytic_map(sample):
    """Qualify recorded direct-kernel samples; the mutation removes PF1."""
    directory = os.environ.get("NOVA_ITER_BANK_RECEIPT")
    if directory is None:
        pytest.skip("set NOVA_ITER_BANK_RECEIPT to the qualified bank run directory")
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text())
    if sample >= len(receipt["accuracy"]):
        pytest.skip("the measurement did not produce this sampled state")
    state = receipt["accuracy"][sample]
    with np.load(
        directory / f"accuracy-{state['index']}.npz", allow_pickle=False
    ) as arrays:
        actual = arrays[
            "dropped" if os.environ.get("NOVA_ITER_BANK_DROP_COIL") else "actual"
        ]
        errors = assert_accuracy(actual, arrays["expected"])
    np.testing.assert_allclose(errors, state["relative_errors"], rtol=1e-12, atol=1e-16)


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


@pytest.mark.parametrize("sample", range(3))
def test_dina_coil_jets_match_analytic_map(sample):
    """Qualify the driven response independently of plasma derivative compilation."""
    directory = os.environ.get("NOVA_ITER_BANK_COIL_RECEIPT")
    if directory is None:
        pytest.skip("set NOVA_ITER_BANK_COIL_RECEIPT to the coil comparison directory")
    directory = Path(directory)
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["kind"] == "coil-only"
    state = receipt["accuracy"][sample]
    with np.load(
        directory / f"coil-accuracy-{state['index']}.npz", allow_pickle=False
    ) as arrays:
        actual = arrays[
            "dropped" if os.environ.get("NOVA_ITER_BANK_DROP_COIL") else "actual"
        ]
        assert_accuracy(actual, arrays["expected"])
