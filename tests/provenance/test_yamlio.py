"""Tests for deterministic, round-trippable YAML serialization."""

import math

import pytest
import yaml

from nova.assembly.provenance import yamlio

TORTURE_DOC = {
    "unicode": "Sécteur — Ω résumé ✓",
    "floats": [0.1, 1e-17, -3699.6178954, 2.5, 1.0, -0.0],
    "big_int": 123456789012345678901234567890,
    "nested": {
        "b": [1, 2, {"z": True, "a": None}],
        "a": {"deep": [False, "x"]},
    },
    "empty": {},
    "flags": [True, False, None],
}


def test_roundtrip_equals_source():
    """safe_load of the canonical bytes reproduces the original object."""
    data = yamlio.canonical_yaml_bytes(TORTURE_DOC)
    assert yaml.safe_load(data) == TORTURE_DOC


def test_redump_is_byte_identical():
    """Re-dumping a loaded document yields byte-identical output."""
    first = yamlio.canonical_yaml_bytes(TORTURE_DOC)
    reloaded = yaml.safe_load(first)
    second = yamlio.canonical_yaml_bytes(reloaded)
    assert first == second


@pytest.mark.parametrize(
    "value",
    [0.1, 1e-17, 1e17, -3699.6178954, 2.0, -0.0, 1234567890.123456, 3.141592653589793],
)
def test_float_roundtrip(value):
    """Individual float torture cases round-trip exactly."""
    data = yamlio.canonical_yaml_bytes({"v": value})
    loaded = yaml.safe_load(data)["v"]
    assert loaded == value
    assert isinstance(loaded, float)


def test_keys_sorted_deterministic():
    """Top-level and nested mapping keys are emitted in sorted order."""
    text = yamlio.canonical_yaml_bytes({"c": 1, "a": 2, "b": 3}).decode()
    lines = [line for line in text.splitlines() if line and not line[0].isspace()]
    keys = [line.split(":")[0] for line in lines]
    assert keys == sorted(keys)


def test_no_anchors_or_aliases():
    """Shared sub-objects are expanded rather than emitted as YAML anchors."""
    shared = {"x": 1}
    doc = {"first": shared, "second": shared}
    text = yamlio.canonical_yaml_bytes(doc).decode()
    assert "&" not in text
    assert "*" not in text


def test_rejects_non_serializable():
    """A non-YAML-safe object raises a clear error, not a silent pass."""
    with pytest.raises((TypeError, ValueError, yaml.YAMLError)):
        yamlio.canonical_yaml_bytes({"bad": object()})


def test_file_roundtrip(tmp_path):
    """dump_yaml then load_yaml reproduces the document and writes canonically."""
    path = tmp_path / "doc.yaml"
    yamlio.dump_yaml(TORTURE_DOC, path)
    assert path.read_bytes() == yamlio.canonical_yaml_bytes(TORTURE_DOC)
    assert yamlio.load_yaml(path) == TORTURE_DOC


def test_nan_and_inf_roundtrip():
    """Non-finite floats survive the round-trip (special-cased by YAML)."""
    data = yamlio.canonical_yaml_bytes({"inf": math.inf, "ninf": -math.inf})
    loaded = yaml.safe_load(data)
    assert loaded["inf"] == math.inf
    assert loaded["ninf"] == -math.inf
