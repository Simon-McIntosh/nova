"""Independent TF observables and agreement refusals."""

import numpy as np
import pytest

from benchmarks.mast_tf_average import (
    RELATIVE_TOLERANCE,
    aggregate_comparisons,
    array_metadata,
    candidate_reason,
    compare_shot,
    current_rbphi,
)


def synthetic_comparison(current):
    """Compare a declared winding's drive against an independently acquired field."""
    return compare_shot(
        current_rbphi(current, linked_turns=24),
        np.array([-0.24, -0.48, -0.36]),
        nominal_sources={"feed-current-digitiser"},
        measured_sources={"absolute-field-probe"},
    )


def test_independent_current_matches_field():
    current = np.array([-50_000.0, -100_000.0, -75_000.0])
    row = synthetic_comparison(current)
    assert row["within_tolerance"]
    assert row["relative_difference"] < 1e-14
    assert aggregate_comparisons([row], expected_shots=1)["verdict"] == "promoted"


def test_scaled_current_fails_tolerance():
    row = synthetic_comparison(1.1 * np.array([-50_000.0, -100_000.0, -75_000.0]))
    assert row["relative_difference"] == pytest.approx(0.1)
    assert not row["within_tolerance"]
    assert aggregate_comparisons([row], expected_shots=1)["verdict"] == "unresolved"


def test_upper_quantile_and_missing_shots_prevent_promotion():
    rows = [{"relative_difference": 0.0}] * 90 + [{"relative_difference": 0.1}] * 10
    assert aggregate_comparisons(rows, expected_shots=100)["verdict"] == "unresolved"
    assert (
        aggregate_comparisons(rows[:90], expected_shots=100)["verdict"] == "unresolved"
    )
    assert (
        aggregate_comparisons([], expected_shots=577)["median_relative_difference"]
        is None
    )
    assert RELATIVE_TOLERANCE == 0.02


@pytest.mark.parametrize("turns", [None, 0, -1, np.nan])
def test_unsourced_turns_refused(turns):
    with pytest.raises(ValueError, match="linked turn count"):
        current_rbphi(np.array([1.0]), linked_turns=turns)


@pytest.mark.parametrize("measured_sources", [set(), {"efm"}])
def test_reconstruction_cannot_validate_itself(measured_sources):
    with pytest.raises(ValueError, match="independent acquisition ancestry"):
        compare_shot(
            np.ones(3),
            np.ones(3),
            nominal_sources={"efm"},
            measured_sources=measured_sources,
        )


@pytest.mark.parametrize("measured", [[], [0.0], [np.nan], [np.inf], [[1.0]]])
def test_empty_zero_nonfinite_or_misaligned_refused(measured):
    with pytest.raises(ValueError):
        compare_shot(
            np.ones(1),
            measured,
            nominal_sources={"current"},
            measured_sources={"probe"},
        )


def test_orientation_and_ancestry_remain_unresolved():
    assert "orientation unresolved" in candidate_reason(
        "magnetics/b_field_tor_probe_cc_field"
    )
    assert "independent acquisition ancestry" in candidate_reason("efm/irod")
    assert "unsourced" in candidate_reason("amc/tf_current")


def test_inventory_reads_array_metadata(tmp_path):
    import json

    metadata = {
        "metadata": {
            "amc/tf_current/.zarray": {"shape": [3]},
            "amc/tf_current/.zattrs": {"units": "kA", "description": "TF feed current"},
        }
    }
    (tmp_path / ".zmetadata").write_text(json.dumps(metadata))
    arrays = array_metadata(tmp_path)
    assert arrays["amc/tf_current"]["units"] == "kA"
    assert arrays["amc/tf_current"]["stored_shape"] == [3]
    assert "amc/missing" not in arrays


def test_store_audit_refuses_derived_product_and_ignores_supply_monitor(
    tmp_path, monkeypatch
):
    import json
    import zarr

    from benchmarks.mast_tf_average import inspect_shot

    root = tmp_path / "level1"
    shot = root / "123.zarr"
    shot.mkdir(parents=True)
    arrays = {
        "amc/tf_current": np.array([-40.0, -50.0]),
        "amc/error_field_a": np.array([0.0, 0.1]),
        "amc/efps_current": np.array([20.0, 20.0]),
        "efm/bvac_r": np.ones(2),
        "efm/bvac_val": np.array([-0.2, -0.3]),
        "efm/irod": np.array([-1e6, -1.5e6]),
    }
    metadata = {}
    for path in arrays:
        metadata[path + "/.zarray"] = {"shape": [2]}
        metadata[path + "/.zattrs"] = {"description": path}
    (shot / ".zmetadata").write_text(json.dumps({"metadata": metadata}))
    monkeypatch.setattr(zarr, "open_group", lambda *args, **kwargs: arrays)
    row = inspect_shot(123, root, tmp_path / "level2")
    assert row["positive_control"]
    assert row["error_field_screen"] == "quiet"
    assert row["reconstruction_product"]["finite_count"] == 2
    assert row["reconstruction_product"]["irod_identity_max_relative_error"] < 1e-14
    assert row["verdict"] == "unresolved"
    assert row["measured_rbphi"] is None
    assert row["relative_difference"] is None
    assert all(not item["accepted"] for item in row["candidates"])
    assert len(row["paths_tried"]) == 2


def test_inventory_reads_inline_metadata_and_requires_consolidation(tmp_path):
    import json

    metadata = {
        "consolidated_metadata": {
            "metadata": {
                "magnetics": {"node_type": "group"},
                "magnetics/b_field_tor_probe_cc_field": {
                    "node_type": "array",
                    "shape": [3, 200],
                    "attributes": {"units": "T", "label": "Tesla/sec"},
                },
            }
        }
    }
    path = tmp_path / "zarr.json"
    path.write_text(json.dumps(metadata))
    arrays = array_metadata(tmp_path)
    assert len(arrays) == 1
    assert arrays["magnetics/b_field_tor_probe_cc_field"]["label"] == "Tesla/sec"
    assert arrays["magnetics/b_field_tor_probe_cc_field"]["stored_shape"] == [3, 200]
    path.write_text(json.dumps({"node_type": "group"}))
    with pytest.raises(KeyError):
        array_metadata(tmp_path)
