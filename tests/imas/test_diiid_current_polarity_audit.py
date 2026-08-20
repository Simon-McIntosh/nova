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
