from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmarks.mast_catalog_input_manifest import (
    ManifestError,
    _batch_digest,
    _digest,
    _state_digest,
    validate_manifest,
)


ARTIFACT = Path(
    "docs/figures/mast-catalog-gpu-solve/mast-catalog-throughput-inputs.json"
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(ARTIFACT.read_text())


def test_real_flat_top_population_is_large_unique_and_fully_provenanced(manifest):
    metrics = validate_manifest(manifest, minimum_slices=4096)

    assert metrics["eligible_real_flat_top_slices"] >= 4096
    assert metrics["unique_shot_index_input_coordinates"] >= 4096
    assert metrics["unique_state_digests"] >= 4096
    assert metrics["duplicate_state_digests"] == 0
    assert metrics["shot_index_time_campaign_configuration_provenance_complete"] is True
    assert metrics["machine_artifact_provenance_complete"] is True
    assert metrics["input_array_provenance_complete"] is True
    assert metrics["broadcast_state"] is False
    assert metrics["tiled_state"] is False
    assert metrics["benchmark_widths"] == [256, 512, 1024, 2048, 4096]


def test_every_state_and_batch_digest_is_reproducible_from_the_manifest(manifest):
    for row in manifest["slices"]:
        assert row["state_digest"] == _state_digest(row["arrays"])
    for batch in manifest["benchmark_batches"]:
        assert batch["population_digest"] == _batch_digest(
            manifest["slices"], batch["width"]
        )


def test_input_configuration_digest_covers_channels_units_shapes_and_dtypes(manifest):
    configuration = deepcopy(manifest["input_configuration"])
    recorded = configuration.pop("digest")

    assert recorded == _digest(configuration)
    assert (
        len(configuration["coil_channels"])
        == configuration["arrays"]["coil_currents_a"]["shape"][0]
    )
    assert (
        len(configuration["sensor_channels"])
        == configuration["arrays"]["sensor_signals"]["shape"][0]
    )
    assert len(configuration["sensor_channels"]) == len(configuration["sensor_units"])


def test_duplicate_numerical_state_is_refused_even_with_a_distinct_coordinate(manifest):
    altered = deepcopy(manifest)
    altered["slices"][1]["arrays"] = deepcopy(altered["slices"][0]["arrays"])
    for array in altered["slices"][1]["arrays"].values():
        array["row_index"] = altered["slices"][1]["input_index"]
    altered["slices"][1]["state_digest"] = altered["slices"][0]["state_digest"]

    with pytest.raises(ManifestError, match="duplicate states"):
        validate_manifest(altered, minimum_slices=4096)


@pytest.mark.parametrize("flag", ["synthetic", "broadcast", "tiled"])
def test_synthetic_broadcast_and_tiled_rows_are_refused(manifest, flag):
    altered = deepcopy(manifest)
    altered["slices"][0][flag] = True

    with pytest.raises(ManifestError, match="replicated or synthetic"):
        validate_manifest(altered, minimum_slices=4096)
