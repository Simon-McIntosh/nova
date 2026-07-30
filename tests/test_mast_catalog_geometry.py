from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nova.scripts.mast_catalog_geometry import (
    active_component_geometry,
    canonical_cycle,
    observed_ranges,
    passive_component_geometry,
    source_fingerprint,
)


def test_active_outline_ignores_cell_subdivision() -> None:
    whole = active_component_geometry(
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.4]),
        np.array([0.6]),
    )
    subdivided = active_component_geometry(
        np.array([0.9, 1.1, 0.9, 1.1]),
        np.array([-0.15, -0.15, 0.15, 0.15]),
        np.full(4, 0.2),
        np.full(4, 0.3),
    )

    assert subdivided == whole


def test_passive_union_ignores_element_subdivision() -> None:
    whole = passive_component_geometry(
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.4]),
        np.array([0.6]),
        np.array([0.0]),
        np.array([0.0]),
    )
    subdivided = passive_component_geometry(
        np.array([0.9, 1.1]),
        np.array([0.0, 0.0]),
        np.array([0.2, 0.2]),
        np.array([0.6, 0.6]),
        np.zeros(2),
        np.zeros(2),
    )

    assert subdivided == whole


def test_closed_path_ignores_start_and_direction() -> None:
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 90.0],
            [1.0, 0.0, 180.0],
            [1.0, -1.0, 270.0],
            [1.0, 0.0, 0.0],
        ]
    )

    assert canonical_cycle(points) == canonical_cycle(np.roll(points[:-1], 2, axis=0))
    assert canonical_cycle(points) == canonical_cycle(points[-2::-1])


def test_ranges_ignore_missing_shots_but_split_real_changes() -> None:
    ranges = observed_ranges(
        [
            (100, "a"),
            (101, None),
            (102, "a"),
            (103, "b"),
            (104, None),
            (105, "b"),
        ]
    )

    assert ranges == [
        {
            "first_observed_shot": 100,
            "last_observed_shot": 102,
            "physical_digest": "a",
            "observed_shots": 2,
        },
        {
            "first_observed_shot": 103,
            "last_observed_shot": 105,
            "physical_digest": "b",
            "observed_shots": 2,
        },
    ]


def test_real_catalog_setup_variants_share_physical_source() -> None:
    roots = (
        Path("/work/projects/imas_gpu/mast/level1/shots"),
        Path("/work/projects/imas_gpu/mast/level2/shots"),
    )
    if not all((root / "11766.zarr").exists() for root in roots) or not all(
        (root / "20400.zarr").exists() for root in roots
    ):
        pytest.skip("MAST L1/L2 catalogs are not mounted")

    early = source_fingerprint(11766, *roots)
    later = source_fingerprint(20400, *roots)

    assert early.representation_digest != later.representation_digest
    assert early.source_digest == later.source_digest


def test_partial_passive_store_is_missing_evidence() -> None:
    roots = (
        Path("/work/projects/imas_gpu/mast/level1/shots"),
        Path("/work/projects/imas_gpu/mast/level2/shots"),
    )
    if not all((root / "26963.zarr").exists() for root in roots):
        pytest.skip("MAST L1/L2 catalogs are not mounted")

    row = source_fingerprint(26963, *roots)

    assert not row.complete
    assert row.source_digest is None
    assert row.missing == ("level2-pf_passive-vertw",)
