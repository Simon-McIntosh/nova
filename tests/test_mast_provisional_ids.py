"""Cover the seeded DD 4.1.1 authoring and the published local artifact.

The load-bearing property is that adding electrical and material semantics is a
representation change: the physical configuration a shot resolves to must be the
same object before and after, so the physical digest is pinned here by value. The
rest of the file checks that a seed reaches the dictionary where a source
licenses one, and that an unresolved field is written nowhere at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nova.catalog.mast_geometry import MachineGeometryRegistry, physical_digest
from nova.imas.machine_evidence import FieldEvidence
from nova.imas.mast_artifact import (
    IncompleteMachineArtifactError,
    resolve_machine_artifact,
)
from nova.imas.mast_geometry import (
    DD_VERSION,
    REPRESENTATIVE_SHOT,
    artifact_shot_ranges,
    author_provisional_ids,
    publish_provisional_artifact,
    write_and_reopen,
)
from nova.imas.mast_seed_parameters import (
    CIRCUIT_RELATIONS,
    loop_sections,
    passive_material,
)
from nova.imas.test_utilities import mark

PHYSICAL_DIGEST = "ca06c8f64481114f"
REGISTRY_DIGEST = "7083e8029c879310d4b811ecc58f5eefdd40b2bfe01b4a1714b177b03a307366"
FIRST_SHOT = 11695
LAST_SHOT = 30473


@pytest.fixture(scope="module")
def registry():
    """Return the packaged physical registry."""

    return MachineGeometryRegistry.default()


@pytest.fixture(scope="module")
def bundle(registry):
    """Author the seeded set once for the representative shot."""

    return author_provisional_ids(
        registry.select(REPRESENTATIVE_SHOT),
        first_shot=FIRST_SHOT,
        last_shot=LAST_SHOT,
    )


@pytest.fixture(scope="module")
def reopened(bundle, tmp_path_factory):
    """Write and reopen the seeded set at the pinned dictionary version."""

    directory = tmp_path_factory.mktemp("machine_description") / "seeded"
    return write_and_reopen(bundle, directory)


@mark["imas"]
def test_seeding_leaves_the_physical_configuration_identical(registry, bundle) -> None:
    geometry = bundle.selection.configuration.geometry

    assert bundle.selection.configuration.physical_digest == PHYSICAL_DIGEST
    assert physical_digest(dict(geometry)) == PHYSICAL_DIGEST
    assert registry.registry_digest == REGISTRY_DIGEST
    assert set(geometry) == {
        "active_components",
        "passive_components",
        "limiter",
        "magnetics",
        "soft_x_ray_chords",
    }


@mark["imas"]
def test_seeded_set_validates_and_reopens_at_the_pinned_dictionary(reopened) -> None:
    assert set(reopened) == {"pf_active", "pf_passive", "wall", "magnetics", "tf"}
    for ids in reopened.values():
        ids.validate()
        assert str(ids.ids_properties.version_put.data_dictionary) == DD_VERSION


@mark["imas"]
def test_documented_circuit_grouping_reaches_the_dictionary(reopened) -> None:
    circuits = [
        (str(circuit.name), str(circuit.type))
        for circuit in reopened["pf_active"].circuit
    ]

    assert circuits == [
        (relation.name, relation.connection) for relation in CIRCUIT_RELATIONS
    ]
    assert ("P6", "anti-series") in circuits


@mark["imas"]
def test_unresolved_electrical_semantics_are_written_nowhere(reopened) -> None:
    pf_active = reopened["pf_active"]
    toroidal_field = reopened["tf"]

    assert not any(
        element.turns_with_sign.has_value
        for coil in pf_active.coil
        for element in coil.element
    )
    assert not pf_active.circuit[0].connections.has_value
    assert not toroidal_field.r0.has_value
    assert not toroidal_field.coils_n.has_value
    assert len(toroidal_field.coil) == 0
    assert len(pf_active.supply) == 0


@mark["imas"]
def test_unresolved_diagnostic_pose_is_written_nowhere(reopened) -> None:
    magnetics = reopened["magnetics"]
    primary = magnetics.b_field_pol_probe[0]

    assert primary.poloidal_angle.has_value
    assert not primary.position.phi.has_value
    assert not primary.toroidal_angle.has_value
    assert not primary.area.has_value
    assert len(magnetics.b_field_phi_probe) == 36
    for probe in magnetics.b_field_phi_probe:
        assert not probe.toroidal_angle.has_value
        assert not probe.poloidal_angle.has_value


@mark["imas"]
def test_material_seed_reaches_only_the_families_a_source_supports(
    bundle,
    reopened,
) -> None:
    sections = loop_sections(bundle.selection.configuration.geometry)
    loops = {str(loop.name): loop for loop in reopened["pf_passive"].loop}

    assert set(loops) == set(sections)
    for family, loop in loops.items():
        material = passive_material(family)
        if material is None:
            assert not loop.resistivity.has_value
            assert not loop.resistance.has_value
            continue
        assert float(loop.resistivity) == pytest.approx(material.resistivity)
        if sections[family].is_single_loop:
            assert float(loop.resistance) == pytest.approx(
                material.loop_resistance(
                    sections[family].area,
                    sections[family].major_radius,
                )
            )
        else:
            assert not loop.resistance.has_value


@mark["imas"]
def test_every_passive_section_carries_one_toroidal_turn(reopened) -> None:
    turns = {
        float(element.turns_with_sign)
        for loop in reopened["pf_passive"].loop
        for element in loop.element
    }

    assert turns == {1.0}


@mark["imas"]
def test_measured_geometry_survives_the_round_trip(bundle, reopened) -> None:
    geometry = bundle.selection.configuration.geometry

    assert len(reopened["pf_active"].coil) == len(geometry["active_components"]) == 13
    assert len(reopened["pf_passive"].loop) == len(geometry["passive_components"]) == 16
    assert len(reopened["magnetics"].flux_loop) == 44 + 36
    assert len(reopened["magnetics"].b_field_pol_probe) == 78 + 61


@mark["imas"]
def test_artifact_shot_ranges_carry_registry_evidence(registry) -> None:
    ranges = artifact_shot_ranges(registry)

    assert [(row.first_shot, row.last_shot, row.evidence) for row in ranges] == [
        (11695, 11765, "inherited"),
        (11766, 30471, "observed"),
        (30472, 30473, "inherited"),
    ]
    assert {row.physical_digest for row in ranges} == {PHYSICAL_DIGEST}


@mark["imas"]
def test_published_revision_verifies_but_is_not_operator_ready(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    artifact = publish_provisional_artifact(cache)
    manifest = artifact.manifest

    assert manifest.physical_digest == PHYSICAL_DIGEST
    assert manifest.registry_digest == REGISTRY_DIGEST
    assert manifest.dd_version == DD_VERSION
    assert manifest.oci.tag == f"dd-{DD_VERSION}-physical-{PHYSICAL_DIGEST}"
    assert not manifest.complete
    assert {artifact_file.name for artifact_file in manifest.files} >= {
        "master.h5",
        "pf_active.h5",
        "pf_passive.h5",
        "magnetics.h5",
        "wall.h5",
        "tf.h5",
    }
    assert manifest.forward_model_blockers() == (
        "pf_active/coil/element/turns_with_sign",
    )
    assert manifest.evidence.state_counts() == {
        "measured": 8,
        "published": 9,
        "generated": 29,
        "fitted": 0,
        "unresolved": 15,
    }
    resolved = resolve_machine_artifact(cache, artifact.digest, allow_incomplete=True)
    assert resolved.manifest == manifest
    with pytest.raises(IncompleteMachineArtifactError, match="operator-ready"):
        resolve_machine_artifact(cache, artifact.digest)


@mark["imas"]
def test_republishing_reproduces_the_semantics_not_the_container(
    tmp_path: Path,
) -> None:
    first = publish_provisional_artifact(tmp_path / "first")
    second = publish_provisional_artifact(tmp_path / "second")

    assert first.manifest.semantic_identity() == second.manifest.semantic_identity()
    assert first.manifest.oci.tag == second.manifest.oci.tag
    assert first.manifest.evidence == second.manifest.evidence
    assert first.manifest.physical_digest == second.manifest.physical_digest


@mark["imas"]
def test_representation_only_seeds_do_not_move_physical_identity(
    registry,
    tmp_path: Path,
) -> None:
    selection = registry.select(REPRESENTATIVE_SHOT)
    before = physical_digest(dict(selection.configuration.geometry))

    artifact = publish_provisional_artifact(tmp_path / "cache")

    after = physical_digest(dict(selection.configuration.geometry))
    assert before == after == PHYSICAL_DIGEST
    assert artifact.manifest.physical_digest == PHYSICAL_DIGEST
    assert artifact.manifest.evidence.paths_with_state(FieldEvidence.GENERATED)
    assert registry.registry_digest == REGISTRY_DIGEST
