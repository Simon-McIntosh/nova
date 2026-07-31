from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import nova.catalog.mast_geometry as mast_geometry
from nova.catalog.mast_geometry import (
    EvidenceState,
    MachineGeometryRegistry,
    SourceFingerprint,
    active_component_geometry,
    canonical_cycle,
    observed_ranges,
    passive_component_geometry,
    physical_digest,
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


def test_registry_resolves_aliases_to_one_physical_configuration() -> None:
    registry = MachineGeometryRegistry.default()

    configurations = {
        registry.resolve_representation(alias).physical_digest
        for alias in registry.representation_aliases
    }

    assert len(registry.representation_aliases) == 3
    assert len(configurations) == 1
    assert configurations == set(registry.configurations)
    assert registry.provenance["source_census_physical_digest"] == "67f789d3d8b40135"


def test_registry_shot_lookup_separates_evidence_from_identity() -> None:
    registry = MachineGeometryRegistry.default()
    early = registry.select(11695)
    observed = registry.select(11766)
    missing = registry.select(26963)
    late = registry.select(30473)

    assert early.evidence is EvidenceState.INHERITED
    assert observed.evidence is EvidenceState.OBSERVED
    assert missing.evidence is EvidenceState.MISSING
    assert missing.missing == ("level2-pf_passive-vertw",)
    assert late.evidence is EvidenceState.INHERITED
    assert {
        selection.configuration.physical_digest
        for selection in (early, observed, missing, late)
    } == set(registry.configurations)


def test_registry_contains_all_census_source_gaps() -> None:
    registry = MachineGeometryRegistry.default()

    missing_group_shots = {
        gap.shot
        for gap in registry.incomplete_evidence.values()
        if set(gap.missing) == {"level2-pf_passive", "level2-wall"}
    }

    assert len(registry.incomplete_evidence) == 17
    assert missing_group_shots == {
        26767,
        26829,
        26838,
        26846,
        26876,
        26884,
        26885,
        26941,
        26947,
        26950,
        26953,
        26957,
        26959,
        26972,
        26982,
        26990,
    }
    assert registry.incomplete_evidence[26963].missing == ("level2-pf_passive-vertw",)


def test_registry_angles_are_canonical_radians() -> None:
    geometry = next(
        iter(MachineGeometryRegistry.default().configurations.values())
    ).geometry
    magnetics = geometry["magnetics"]

    candidate_phi = [
        phi
        for probe in magnetics["poloidal_probes"]
        for phi in probe["position_phi_candidates"]
    ]
    saddle_phi = [
        point[2]
        for paths in magnetics["saddle_paths"].values()
        for path in paths
        for point in path
    ]
    additional_phi = [
        point[2]
        for points in magnetics["additional_points"].values()
        for point in points
    ]
    xray_phi = [
        chord[4]
        for chords in geometry["soft_x_ray_chords"].values()
        for chord in chords
    ]

    for values in (candidate_phi, saddle_phi, additional_phi, xray_phi):
        assert values
        assert np.all(np.asarray(values) >= 0.0)
        assert np.all(np.asarray(values) <= 2 * np.pi)


def test_registry_pins_catalog_degree_conversions() -> None:
    geometry = next(
        iter(MachineGeometryRegistry.default().configurations.values())
    ).geometry
    magnetics = geometry["magnetics"]

    first_probe = magnetics["poloidal_probes"][0]
    assert np.allclose(
        first_probe["position_phi_candidates"],
        np.deg2rad([150.0, 330.0]),
        atol=5e-6,
    )
    assert np.isclose(
        magnetics["saddle_paths"]["l"][0][0][2],
        np.deg2rad(4.96822),
        atol=5e-6,
    )
    assert np.isclose(
        magnetics["additional_points"]["poloidal_cc"][0][2],
        np.deg2rad(270.0),
        atol=5e-6,
    )
    assert np.isclose(
        magnetics["additional_points"]["toroidal_cc"][0][2],
        np.deg2rad(10.0),
        atol=5e-6,
    )
    assert np.isclose(
        geometry["soft_x_ray_chords"]["horizontal_cam_lower"][0][4],
        np.deg2rad(105.0),
        atol=5e-6,
    )


def test_physical_identity_changes_with_supported_geometry_and_pose() -> None:
    geometry = deepcopy(
        next(iter(MachineGeometryRegistry.default().configurations.values())).geometry
    )
    baseline = physical_digest(geometry)
    mutations = []

    active = deepcopy(geometry)
    active["active_components"]["sol"] += "00"
    mutations.append(active)

    passive = deepcopy(geometry)
    passive["passive_components"]["vertw"] += "00"
    mutations.append(passive)

    limiter = deepcopy(geometry)
    limiter["limiter"][0][0] += 1e-4
    mutations.append(limiter)

    probe = deepcopy(geometry)
    probe["magnetics"]["poloidal_probes"][0]["pose"][2] += 1e-3
    mutations.append(probe)

    probe_bank = deepcopy(geometry)
    probe_bank["magnetics"]["poloidal_probes"][0]["position_phi_candidates"][0] += 1e-3
    mutations.append(probe_bank)

    saddle = deepcopy(geometry)
    saddle["magnetics"]["saddle_paths"]["l"][0][0][2] += 1e-3
    mutations.append(saddle)

    for mutation in mutations:
        assert physical_digest(mutation) != baseline


def test_unsupported_semantics_are_explicit_authoring_gaps() -> None:
    configuration = next(
        iter(MachineGeometryRegistry.default().configurations.values())
    )
    gaps = " ".join(configuration.authoring_gaps)

    assert "turns" in gaps
    assert "polarity" in gaps
    assert "circuit topology" in gaps
    assert "material" in gaps
    assert "resistance" in gaps
    assert "independent toroidal probe orientation" in gaps
    assert "saddle traversal sign" in gaps


def test_catalog_scan_resumes_from_atomic_fingerprint_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    level1_root = tmp_path / "level1"
    level2_root = tmp_path / "level2"
    level1_root.mkdir()
    level2_root.mkdir()
    for shot in (10, 11):
        (level2_root / f"{shot}.zarr").mkdir()

    geometry = {
        "active_components": {},
        "passive_components": {},
        "limiter": [],
        "magnetics": {
            "poloidal_probes": [],
            "flux_loops": [],
            "saddle_paths": {},
        },
        "soft_x_ray_chords": {},
    }
    digest = physical_digest(geometry)
    calls: list[int] = []

    def fingerprint(
        shot: int,
        _level1_root: Path,
        _level2_root: Path,
    ) -> SourceFingerprint:
        calls.append(shot)
        return SourceFingerprint(shot, "source", f"representation-{shot}", True, ())

    monkeypatch.setattr(mast_geometry, "source_fingerprint", fingerprint)
    monkeypatch.setattr(
        mast_geometry,
        "physical_snapshot",
        lambda *_args: geometry,
    )
    checkpoint = tmp_path / "fingerprints.json"

    report = mast_geometry.scan_catalog(
        level1_root,
        level2_root,
        workers=1,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
    )

    assert calls == [10, 11]
    assert checkpoint.exists()
    assert report["physical_configuration_counts"] == {digest: 2}

    calls.clear()
    resumed = mast_geometry.scan_catalog(
        level1_root,
        level2_root,
        workers=1,
        checkpoint_path=checkpoint,
        checkpoint_every=1,
    )

    assert calls == []
    assert resumed == report
