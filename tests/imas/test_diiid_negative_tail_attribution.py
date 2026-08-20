import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[2] / "benchmarks" / "diiid_negative_tail_attribution.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diiid_negative_tail_attribution", MODULE_PATH
)
attribution = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = attribution
SPEC.loader.exec_module(attribution)


def test_reproduce_bank_uses_pooled_error_energy_and_frame_median():
    bank = {
        "score": {
            "frame_records": [
                {
                    "population": "quiescent",
                    "shot": "a",
                    "r2": attribution.EXPECTED_MEDIAN_FRAME_R2,
                    "squared_error": 1.0 - attribution.EXPECTED_POOLED_R2,
                    "total_sum_squares": 1.0,
                },
                {
                    "population": "transient",
                    "shot": "a",
                    "r2": -20.0,
                    "squared_error": 21.0,
                    "total_sum_squares": 1.0,
                },
            ]
        }
    }
    result = attribution.reproduce_bank(bank)
    assert result["frames"] == 1
    np.testing.assert_allclose(result["pooled_delta_from_banked"], 0.0)
    np.testing.assert_allclose(result["median_delta_from_banked"], 0.0)


def test_centred_shape_slope_identifies_polarity_without_gauge():
    predicted = np.array([[1.0, 2.0], [4.0, 7.0]])
    actual = -1.25 * predicted + 18.0
    np.testing.assert_allclose(
        attribution.centred_shape_slope(actual, predicted), -1.25
    )


def test_deficit_shares_are_exclusive_and_sum_to_one():
    result = attribution.deficit_shares(
        baseline=np.array([-3.0, -2.8]),
        aligned=np.array([-2.9, -2.7]),
        polarity_corrected=np.array([0.99, 0.98]),
        control_reference=1.0,
    )
    shares = [
        result["geometry"],
        result["coil_current_time_alignment"],
        result["data_quality_current_polarity"],
        result["unexplained"],
    ]
    np.testing.assert_allclose(sum(shares), 1.0)
    assert result["data_quality_current_polarity"] > 0.9


def test_control_matching_never_uses_r2_or_selects_tail_shots():
    records = []
    for index, shot in enumerate((*attribution.TAIL_SHOTS, "c1", "c2", "c3")):
        records.append(
            {
                "shot": shot,
                "maximum_derivative": 20.0 + index,
                "represented_plasma_current_a": 1.0e6 + 1000.0 * index,
                "current_projection_fractional_rms": 0.02 + index * 1.0e-4,
                "r2": -1000.0 + index,
            }
        )
    matches = attribution.match_controls(records, controls_per_tail=2)
    for items in matches.values():
        assert len(items) == 2
        assert not ({str(item["shot"]) for item in items} & set(attribution.TAIL_SHOTS))
