from __future__ import annotations

import hashlib
from pathlib import Path

import imas
import pytest

import nova.imas.machine_artifact as machine_artifact_module
from nova.imas.machine_artifact import (
    ArtifactShotRange,
    MachineArtifactError,
    create_machine_artifact_manifest,
    latest_publication_dd_version,
    machine_name,
    manifest_schema,
    oci_artifact_reference,
    oci_artifact_tag,
    oci_artifact_type,
    oci_file_media_type,
    publication_dd_version,
)

DD_VERSION = latest_publication_dd_version()
REGISTRY_DIGEST = "a" * 64
PHYSICAL_DIGEST = "b" * 16


def _write_netcdf_wall(path: Path) -> None:
    wall = imas.IDSFactory(DD_VERSION).new("wall")
    wall.ids_properties.homogeneous_time = 0
    wall.description_2d.resize(1)
    wall.description_2d[0].limiter.unit.resize(1)
    outline = wall.description_2d[0].limiter.unit[0].outline
    outline.r = [1.0, 2.0, 1.0]
    outline.z = [0.0, 0.0, 1.0]
    with imas.DBEntry(path, "x", dd_version=DD_VERSION) as entry:
        entry.put(wall)


def _manifest(source: Path, machine: str):
    return create_machine_artifact_manifest(
        source,
        machine=machine,
        dd_version=DD_VERSION,
        registry_digest=REGISTRY_DIGEST,
        physical_digest=PHYSICAL_DIGEST,
        shot_ranges=(
            ArtifactShotRange(
                first_shot=0,
                last_shot=0,
                physical_digest=PHYSICAL_DIGEST,
                evidence="observed",
            ),
        ),
        complete=False,
        unresolved_gaps=("passive structure is absent",),
    )


def test_machine_name_parameterizes_manifest_and_oci_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "wall.nc").write_bytes(b"wall")

    mast = _manifest(source, "MAST")
    diiid = _manifest(source, "DIII-D")

    assert machine_name("DIII-D") == "diii-d"
    assert mast.schema == manifest_schema("mast") == "nova-mast-machine-artifact"
    assert diiid.schema == "nova-diii-d-machine-artifact"
    assert mast.machine == "mast"
    assert diiid.machine == "diii-d"
    assert mast.oci.artifact_type == oci_artifact_type("mast")
    assert diiid.oci.artifact_type == oci_artifact_type("DIII-D")
    assert mast.oci.file_media_type == oci_file_media_type("mast")
    assert diiid.oci.file_media_type == oci_file_media_type("DIII-D")

    mast_payload = mast.as_dict()
    diiid_payload = diiid.as_dict()
    differing_fields = {
        key for key in mast_payload if mast_payload[key] != diiid_payload[key]
    }
    assert differing_fields == {"oci", "schema"}


def test_diiid_artifact_reference_is_a_ghcr_reference(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "wall.nc").write_bytes(b"wall")
    manifest = _manifest(source, "DIII-D")

    reference = oci_artifact_reference(
        "ghcr.io/simon-mcintosh/diii-d-machine-description",
        manifest,
    )

    assert reference == (
        "ghcr.io/simon-mcintosh/diii-d-machine-description:"
        f"dd-{DD_VERSION}-physical-{PHYSICAL_DIGEST}"
    )


def test_publication_resolves_latest_dictionary_and_refuses_prior_major(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "wall.nc").write_bytes(b"wall")
    monkeypatch.setattr(
        machine_artifact_module,
        "dd_xml_versions",
        lambda: ["3.42.2", "4.0.0", "4.1.0", "4.1.2"],
    )
    shot_ranges = (
        ArtifactShotRange(
            first_shot=0,
            last_shot=0,
            physical_digest=PHYSICAL_DIGEST,
            evidence="observed",
        ),
    )

    assert latest_publication_dd_version() == "4.1.2"
    assert publication_dd_version() == "4.1.2"
    assert publication_dd_version("4.0.0") == "4.0.0"
    defaulted = create_machine_artifact_manifest(
        source,
        machine="DIII-D",
        registry_digest=REGISTRY_DIGEST,
        physical_digest=PHYSICAL_DIGEST,
        shot_ranges=shot_ranges,
        complete=False,
        unresolved_gaps=("passive structure is absent",),
    )
    assert defaulted.dd_version == "4.1.2"
    with pytest.raises(MachineArtifactError, match="requires data dictionary major 4"):
        create_machine_artifact_manifest(
            source,
            machine="DIII-D",
            dd_version="3.42.2",
            registry_digest=REGISTRY_DIGEST,
            physical_digest=PHYSICAL_DIGEST,
            shot_ranges=shot_ranges,
            complete=False,
            unresolved_gaps=("passive structure is absent",),
        )
    assert oci_artifact_tag("3.42.2", PHYSICAL_DIGEST) != oci_artifact_tag(
        "4.1.2",
        PHYSICAL_DIGEST,
    )


def test_imas_netcdf_inventory_has_repeatable_content_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_path = first / "wall.nc"
    second_path = second / "wall.nc"
    _write_netcdf_wall(first_path)
    _write_netcdf_wall(second_path)

    consistency_fields: list[tuple[int, int] | None] = []
    original = machine_artifact_module._hdf5_consistency_field

    def record_consistency_field(descriptor: int, size: int, path: Path):
        result = original(descriptor, size, path)
        consistency_fields.append(result)
        return result

    monkeypatch.setattr(
        machine_artifact_module,
        "_hdf5_consistency_field",
        record_consistency_field,
    )
    left = _manifest(first, "DIII-D")
    right = _manifest(second, "DIII-D")

    assert [artifact_file.name for artifact_file in left.files] == ["wall.nc"]
    assert left.files == right.files
    assert left.files[0].sha256 == hashlib.sha256(first_path.read_bytes()).hexdigest()
    assert first_path.read_bytes() == second_path.read_bytes()
    assert len(consistency_fields) == 2
    assert all(field is not None for field in consistency_fields)


@pytest.mark.parametrize("machine", ["", " DIII-D", "diii_d", "diii--d"])
def test_machine_name_rejects_noncanonical_registry_names(machine: str) -> None:
    with pytest.raises(MachineArtifactError, match="machine name"):
        machine_name(machine)
