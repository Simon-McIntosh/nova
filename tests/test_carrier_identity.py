"""A stored carrier changes identity when its construction sources change."""

from pathlib import Path

import pytest

from scripts.analytic_oracle_fixtures import coupling_build, measure


@pytest.mark.parametrize(
    "source",
    [
        "nova/biot/polygonanalytic.py",
        "nova/biot/polygon.py",
        "nova/biot/plasmagrid.py",
        "nova/biot/tiledassembly.py",
        "nova/equilibrium/stencil_mesh.py",
        "nova/frame/coilset.py",
        "nova/frame/frame.py",
        "nova/geometry/polygon.py",
        "coupling_assembly",
    ],
)
def test_carrier_revision_tracks_construction_sources(monkeypatch, source):
    root = Path(measure.__file__).resolve().parents[2]
    path = (
        Path(coupling_build.__file__).resolve()
        if source == "coupling_assembly"
        else root / source
    )
    assert path.is_file()
    before = measure.builder_revision()
    original = Path.read_bytes

    def changed_bytes(candidate):
        content = original(candidate)
        if candidate.resolve() == path:
            return content + b"\n# changed construction source\n"
        return content

    monkeypatch.setattr(Path, "read_bytes", changed_bytes)
    assert measure.builder_revision() != before


def test_fixture_measurement_edit_preserves_carrier_revision(monkeypatch):
    before = measure.builder_revision()
    original = Path.read_text
    fixture = Path(measure.__file__).resolve()

    def changed_text(candidate, *args, **kwargs):
        content = original(candidate, *args, **kwargs)
        if candidate.resolve() == fixture:
            return content + "\ndef measurement_only_marker():\n    pass\n"
        return content

    monkeypatch.setattr(Path, "read_text", changed_text)
    assert measure.builder_revision() == before


def test_carrier_revision_tracks_builder_function(monkeypatch):
    before = measure.builder_revision()
    original = Path.read_text
    fixture = Path(measure.__file__).resolve()

    def changed_text(candidate, *args, **kwargs):
        content = original(candidate, *args, **kwargs)
        if candidate.resolve() == fixture:
            return content.replace("area=area,", "area=np.asarray(area),", 1)
        return content

    monkeypatch.setattr(Path, "read_text", changed_text)
    assert measure.builder_revision() != before
