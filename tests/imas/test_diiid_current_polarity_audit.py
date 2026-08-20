from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as parquet


MODULE_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "diiid_current_polarity_audit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diiid_current_polarity_audit", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_orientation_is_gauge_invariant_and_preserves_sign():
    predicted = np.array([[0.0, 1.0], [2.0, 4.0]])
    np.testing.assert_allclose(
        audit.orientation_cosine(7.0 + 3.0 * predicted, predicted), 1.0
    )


def test_orientation_predicate_only_marks_negative_shapes():
    predicted = np.array([-2.0, -0.5, 1.0, 4.0])
    positive = audit.orientation_cosine(predicted + 11.0, predicted)
    negative = audit.orientation_cosine(-predicted + 11.0, predicted)
    assert not audit.affected_from_orientation(positive)
    assert audit.affected_from_orientation(negative)
    assert not audit.affected_from_orientation(0.0)


def test_raw_arrow_values_cross_reader_and_registry_without_sign_change(tmp_path):
    path = tmp_path / "shot.parquet"
    values = np.array([-3.5, -0.0, 0.0, 2.25, np.nan], dtype=np.float64)
    table = pa.table(
        {
            column: pa.array([values.tolist()], type=pa.list_(pa.float64()))
            for column in audit.CURRENT_COLUMNS
        }
    )
    parquet.write_table(table, path, compression="zstd")
    conductors = tuple(
        SimpleNamespace(
            input_column=column,
            turns=SimpleNamespace(applied_multiplier=1.0),
        )
        for column in audit.CURRENT_COLUMNS
    )
    receipt = audit.compare_read_boundary(path, SimpleNamespace(conductors=conductors))
    assert receipt["channel_count"] == 19
    assert receipt["all_channels_bitwise_equal"]
    assert receipt["total_raw_to_read_sign_mismatches"] == 0
    assert receipt["total_read_to_registry_ampere_sign_mismatches"] == 0
    for channel in receipt["channels"].values():
        assert channel["raw_page"]["compressed_size_bytes"] > 0
        assert len(channel["raw_page"]["sha256"]) == 64
        assert channel["raw_sign_counts"] == channel["nova_read_sign_counts"]


def test_gate_artifact_digests_are_fully_pinned():
    assert audit.EXPECTED_BANK_SHA256 == (
        "5b53ffc30bbe823fa50c43bb945c184c8033e81b4083aaa338661278d1f6adea"
    )
    assert audit.EXPECTED_PREREGISTRATION_SHA256 == (
        "7e60861de8c104a8d736bd5300993071da35fce93e206ad5bfb3010213f972fc"
    )


def test_exact_polygon_response_is_persisted_validated_and_reused(
    tmp_path, monkeypatch
):
    path = tmp_path / "response.npz"
    radius = np.array([1.0, 1.5, 2.0])
    height = np.array([-1.0, 0.0, 1.0])
    target_mask = audit._boundary_mask(radius, height)
    turns = SimpleNamespace(affects_axisymmetric_poloidal_flux=True)
    description = SimpleNamespace(
        physical_digest="geometry-digest",
        conductors=(
            SimpleNamespace(name="F1A", vertices=np.ones((4, 2)), turns=turns),
            SimpleNamespace(name="F2A", vertices=np.ones((4, 2)), turns=turns),
        ),
    )
    calls = []

    def exact_response(*_args):
        calls.append(True)
        return ("F1A", "F2A"), np.arange(16, dtype=float).reshape(2, 8)

    monkeypatch.setattr(audit, "_exact_polygon_response", exact_response)
    first, first_receipt = audit.persisted_exact_polygon_response(
        path, description, radius, height, target_mask
    )
    second, second_receipt = audit.persisted_exact_polygon_response(
        path, description, radius, height, target_mask
    )

    assert len(calls) == 1
    assert first_receipt["built_in_this_run"]
    assert not second_receipt["built_in_this_run"]
    assert first_receipt["kernel_route"] == "nova.biot.polygon.polygon_greens"
    assert first_receipt["filament_centre_proxy_used"] is False
    assert first[0] == second[0] == ("F1A", "F2A")
    np.testing.assert_array_equal(first[1], second[1])


def test_persisted_response_rejects_a_different_geometry(tmp_path, monkeypatch):
    path = tmp_path / "response.npz"
    radius = np.array([1.0, 1.5, 2.0])
    height = np.array([-1.0, 0.0, 1.0])
    target_mask = audit._boundary_mask(radius, height)
    turns = SimpleNamespace(affects_axisymmetric_poloidal_flux=True)
    conductor = SimpleNamespace(name="F1A", vertices=np.ones((4, 2)), turns=turns)
    description = SimpleNamespace(
        physical_digest="first-geometry", conductors=(conductor,)
    )
    monkeypatch.setattr(
        audit,
        "_exact_polygon_response",
        lambda *_args: (("F1A",), np.arange(8, dtype=float).reshape(1, 8)),
    )
    audit.persisted_exact_polygon_response(
        path, description, radius, height, target_mask
    )
    changed = SimpleNamespace(
        physical_digest="different-geometry", conductors=(conductor,)
    )

    with np.testing.assert_raises_regex(ValueError, "geometry digest"):
        audit.persisted_exact_polygon_response(
            path, changed, radius, height, target_mask
        )
