"""The persisted coil response: reuse, refusal and the slice wall.

Section 6 of the iron-core-magnetisation plan requires the external term to be
formed from a coil response persisted per mesh and per coil set, equal to the
direct coil evaluation to 1e-12 relative, and refused for any other mesh or coil
set. These tests pin the refusal paths on a temporary store, so a change that
lets a stored response serve a different problem fails here rather than in a
WEST solve.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _measure():
    spec = importlib.util.spec_from_file_location(
        "iron_field_measure", ROOT / "scripts/iron_field_source/measure.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store(tmp_path, monkeypatch):
    measure = _measure()

    def field_at(points, rows):
        points = np.asarray(points, dtype=np.float64)
        return np.column_stack((points[:, 0], -points[:, 1]))

    monkeypatch.setattr(measure, "COIL_RESPONSE", tmp_path)
    monkeypatch.setattr(measure, "field_at", field_at)
    return measure, tmp_path


def _points():
    return np.array([[1.0, 0.0], [1.1, 0.2], [1.2, -0.4]])


def test_a_matching_response_is_reused_bit_for_bit(store):
    measure, _ = store
    points = _points()
    first, reused = measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    assert reused is False
    second, reused = measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    assert reused is True
    np.testing.assert_array_equal(second, first)


def test_another_mesh_is_refused(store):
    """A response written for one mesh digest cannot serve another."""
    measure, directory = store
    points = _points()
    measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    (directory / f"{'a' * 16}.npz").rename(directory / f"{'c' * 16}.npz")
    with pytest.raises(ValueError, match="serves another mesh"):
        measure.persist_coil_field(points, None, "c" * 64, "b" * 64)


def test_another_coil_set_is_refused(store):
    """The same mesh with different coil currents must not reuse the response."""
    measure, directory = store
    points = _points()
    measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    with pytest.raises(ValueError, match="serves another coil set"):
        measure.persist_coil_field(points, None, "a" * 64, "d" * 64)


def test_another_quadrature_is_refused(store):
    """A response is refused when the quadrature points differ."""
    measure, directory = store
    points = _points()
    measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    with pytest.raises(ValueError, match="serves another quadrature"):
        measure.persist_coil_field(points[:2], None, "a" * 64, "b" * 64)


def test_the_stored_response_carries_its_identity(store):
    """The stored file records the digests, so a reader can audit the reuse."""
    measure, directory = store
    points = _points()
    measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    with np.load(directory / f"{'a' * 16}.npz") as stored:
        assert stored["mesh_digest"].item() == "a" * 64
        assert stored["coil_key"].item() == "b" * 64
        np.testing.assert_array_equal(stored["points"], points)


def test_the_direct_and_persisted_coil_fields_agree(store):
    """The reused values equal a fresh evaluation at the stored points."""
    measure, _ = store
    points = _points()
    persisted, _ = measure.persist_coil_field(points, None, "a" * 64, "b" * 64)
    direct = measure.field_at(points, None)
    np.testing.assert_allclose(persisted, direct, rtol=0, atol=0)


def test_the_receipt_records_the_response_written_this_run():
    """A receipt written by the state stage names the response it used."""
    receipt = (
        Path(
            "/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/"
            "field-source/west-62751-uniform-0.08-state-receipt.json"
        )
    )
    if not receipt.exists():
        pytest.skip("the state receipt is written by the measurement job")
    row = json.loads(receipt.read_text())
    if "slice_seconds_with_response" not in row:
        pytest.skip("the receipt predates the slice-wall fields; the state job rewrites it")
    assert row["completed"] is True
    # the persisted coils equal a direct coil evaluation at the same points
    assert row["coil_response_relative"] <= 1e-12
    # the prototype's single-quadrature external term differs only through the
    # summation order of the coils and plasma split
    assert row["external_relative_frobenius"] <= 1e-12
    assert "another mesh" in row["coil_response_refusals"]
    assert "another coil set" in row["coil_response_refusals"]
    assert row["slice_seconds_with_response"] > 0
    assert row["slice_seconds_without_response"] > 0