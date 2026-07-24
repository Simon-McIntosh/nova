"""Tests for the schema-validated declarative fit configuration."""

import pytest

from nova.assembly.provenance import fitconfig


def _valid_payload():
    """Return a fully-specified valid configuration mapping."""
    return {
        "length_scales": [1.0, 2.0, 0.5],
        "nugget": 1e-6,
        "weights": [1.0, 1.0, 0.25],
        "constraints": {"clamp_ends": True, "periodic": False},
        "fiducial_index": [0, 4, 9],
        "radial_offset": -0.5,
        "augment_version": "8.1",
        "reference_version": "7.0",
        "random_state": 42,
        "extra": {"note": "forward-compat sink"},
    }


def test_valid_roundtrip_and_types():
    """A valid config parses with typed fields and round-trips through dict."""
    config = fitconfig.FitConfig.from_dict(_valid_payload())
    assert config.nugget == 1e-6
    assert config.weights == [1.0, 1.0, 0.25]
    assert config.random_state == 42
    assert config.fiducial_index == [0, 4, 9]
    assert fitconfig.FitConfig.from_dict(config.to_dict()) == config


def test_defaults_applied():
    """Omitted fields fall back to declared defaults."""
    config = fitconfig.FitConfig.from_dict({})
    assert config.nugget == 0.0
    assert config.extra == {}
    assert config.random_state is None


def test_unknown_top_level_key_rejected():
    """An unknown top-level key outside extra is rejected."""
    payload = _valid_payload()
    payload["mystery"] = 1
    with pytest.raises(ValueError):
        fitconfig.FitConfig.from_dict(payload)


def test_negative_nugget_rejected():
    """A negative nugget is rejected by validation."""
    payload = _valid_payload()
    payload["nugget"] = -0.1
    with pytest.raises(ValueError):
        fitconfig.FitConfig.from_dict(payload)


def test_wrong_type_rejected():
    """A wrongly-typed field is rejected."""
    payload = _valid_payload()
    payload["random_state"] = "not-an-int"
    with pytest.raises((TypeError, ValueError)):
        fitconfig.FitConfig.from_dict(payload)


def test_weights_must_be_numbers():
    """Weights must be a list of numbers."""
    payload = _valid_payload()
    payload["weights"] = [1.0, "x", 0.25]
    with pytest.raises((TypeError, ValueError)):
        fitconfig.FitConfig.from_dict(payload)


def test_sha256_stable_and_prefixed():
    """The content hash is stable across equal configs and sha256-tagged."""
    a = fitconfig.FitConfig.from_dict(_valid_payload())
    b = fitconfig.FitConfig.from_dict(_valid_payload())
    assert a.sha256 == b.sha256
    assert a.sha256.startswith("sha256:")


def test_sha256_changes_with_content():
    """A content change alters the hash."""
    a = fitconfig.FitConfig.from_dict(_valid_payload())
    payload = _valid_payload()
    payload["nugget"] = 2e-6
    b = fitconfig.FitConfig.from_dict(payload)
    assert a.sha256 != b.sha256


def test_file_roundtrip(tmp_path):
    """Writing then loading a config reproduces it."""
    config = fitconfig.FitConfig.from_dict(_valid_payload())
    path = tmp_path / "fit.yaml"
    config.write(path)
    assert fitconfig.FitConfig.load(path) == config
