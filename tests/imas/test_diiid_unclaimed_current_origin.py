"""Regression tests for exterior-current origin discrimination."""

from __future__ import annotations

import json

import numpy as np

from benchmarks import diiid_unclaimed_current_origin as origin


def test_preregistration_declares_all_discriminators() -> None:
    declaration = origin.preregistration()

    assert set(declaration) == {
        "cohort",
        "position_stability",
        "non_gs_accounting",
        "magnitude_plausibility",
        "resolution_dependence",
        "verdict",
    }
    assert declaration["cohort"]["frames"] == 60
    assert declaration["cohort"]["shots"] == 20
    assert declaration["cohort"]["detectable_patches"] == 391
    assert declaration["non_gs_accounting"]["coefficients_fitted"] == 0


def test_centroid_clustering_is_position_based() -> None:
    coordinates = np.asarray(
        [
            [1.00, -0.50],
            [1.01, -0.49],
            [0.99, -0.51],
            [2.00, 0.50],
            [2.01, 0.49],
            [1.99, 0.51],
            [1.50, 0.00],
        ]
    )

    labels = origin.cluster_centroids(coordinates, radius_m=0.03, minimum_patches=3)

    assert len(set(labels) - {-1}) == 2
    assert labels[-1] == -1
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:6])) == 1
    assert labels[0] != labels[3]


def test_position_summary_requires_cross_shot_cell_stability() -> None:
    records = [
        {
            "shot": f"shot-{index}",
            "frame": 1,
            "centroid_r_m": 1.50 + 0.0005 * index,
            "centroid_z_m": -0.20 + 0.0004 * index,
        }
        for index in range(origin.POSITION_MINIMUM_SHOTS)
    ]
    labels = np.zeros(len(records), dtype=int)

    summary = origin.position_summary(records, labels)

    assert summary["position_stable_cluster_count"] == 1
    assert summary["largest_clusters"][0]["shots"] == origin.POSITION_MINIMUM_SHOTS
    assert summary["largest_clusters"][0]["radial_peak_to_peak_m"] < 0.021


def test_origin_verdict_preserves_three_outcomes() -> None:
    artefact = origin.origin_verdict(0.9, 0, 0.05, 0.05)
    physical = origin.origin_verdict(0.1, 1, 0.05, -0.05)
    mixed = origin.origin_verdict(0.5, 0, 0.05, 0.05)

    assert artefact["verdict"] == "artefact"
    assert artefact["carrying_discriminator"] == "non-GS accounting"
    assert physical["verdict"] == "physical"
    assert mixed["verdict"] == "mixed"


def test_landed_receipt_retains_origin_population() -> None:
    source = json.loads(origin.SOURCE_RECEIPT.read_text())
    detectable, records = origin._source_patch_records(source)

    assert source["selection"]["frames"] == 60
    assert source["selection"]["shots"] == 20
    assert source["selection"]["all_selected_absent_from_polarity_population"]
    assert len(detectable) == 391
    assert len(records) == 60


def test_generated_receipt_carries_complete_quantitative_answer() -> None:
    path = origin.DEFAULT_OUTPUT / origin.RECEIPT_NAME
    receipt = json.loads(path.read_text())

    assert receipt["selection"] == {
        "all_selected_absent_from_polarity_population": True,
        "detectable_patches": 391,
        "frames": 60,
        "polarity_population_count": 603,
        "shots": 20,
    }
    assert (
        receipt["non_gs_accounting"]["landed_irreducible_strict_gs_residual_fraction"]
        == 0.9968
    )
    assert receipt["magnitude_plausibility"]["released_conductor_count"] >= 19
    assert receipt["resolution_dependence"]["native_detectable_patch_count"] == 391
    assert receipt["verdict"]["verdict"] in {"physical", "artefact", "mixed"}
    assert receipt["verdict"]["carrying_discriminator"]
    assert (origin.DEFAULT_OUTPUT / origin.FIGURE_NAME).is_file()
