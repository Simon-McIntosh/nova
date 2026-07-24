"""Tests for the immutable run-record model."""

import re

import pytest

from nova.assembly.provenance import runrecord

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _valid_payload():
    """Return a fully-specified valid run-record mapping."""
    return {
        "fit_id": "ccl-sector-1",
        "code_git_sha": "0" * 40,
        "code_dirty": False,
        "uv_lock_sha256": "sha256:" + "a" * 64,
        "input_digests": {
            "workbook": "sha256:" + "b" * 64,
            "reference": "sha256:" + "c" * 64,
        },
        "fit_config_sha256": "sha256:" + "d" * 64,
        "outputs": [
            {
                "name": "ccl.nc",
                "digest": "sha256:" + "e" * 64,
                "tolerance_class": "tight",
            },
        ],
        "env": {
            "packages": {"numpy": "1.26.4", "scipy": "1.13.0"},
            "blas_single_thread": True,
        },
        "operator": "ai-scd1@iter.org",
        "timestamp": "2026-07-24T09:00:00+00:00",
    }


def test_valid_roundtrip():
    """A valid record round-trips through dict and YAML."""
    record = runrecord.RunRecord.from_dict(_valid_payload())
    assert record.fit_id == "ccl-sector-1"
    assert record.code_dirty is False
    assert record.input_digests["workbook"].startswith("sha256:")
    assert runrecord.RunRecord.from_dict(record.to_dict()) == record


def test_record_digest_stable_and_prefixed():
    """The record digest is stable across equal records and sha256-tagged."""
    a = runrecord.RunRecord.from_dict(_valid_payload())
    b = runrecord.RunRecord.from_dict(_valid_payload())
    assert a.record_digest == b.record_digest
    assert a.record_digest.startswith("sha256:")


def test_record_digest_includes_timestamp():
    """The digest covers the whole document, timestamp included."""
    a = runrecord.RunRecord.from_dict(_valid_payload())
    payload = _valid_payload()
    payload["timestamp"] = "2026-07-24T10:00:00+00:00"
    b = runrecord.RunRecord.from_dict(payload)
    assert a.record_digest != b.record_digest


def test_bad_git_sha_rejected():
    """A git sha that is not 40 hex characters is rejected."""
    payload = _valid_payload()
    payload["code_git_sha"] = "deadbeef"
    with pytest.raises(ValueError):
        runrecord.RunRecord.from_dict(payload)


def test_bad_digest_format_rejected():
    """An input digest lacking the sha256 prefix is rejected."""
    payload = _valid_payload()
    payload["input_digests"]["workbook"] = "b" * 64
    with pytest.raises(ValueError):
        runrecord.RunRecord.from_dict(payload)


def test_missing_required_field_rejected():
    """Omitting a required field is rejected."""
    payload = _valid_payload()
    del payload["operator"]
    with pytest.raises((ValueError, TypeError, KeyError)):
        runrecord.RunRecord.from_dict(payload)


def test_file_roundtrip(tmp_path):
    """Writing then loading a record reproduces it."""
    record = runrecord.RunRecord.from_dict(_valid_payload())
    path = tmp_path / "run.yaml"
    record.write(path)
    assert runrecord.RunRecord.load(path) == record


def test_capture_environment_real_repo():
    """capture_environment reports the repo's real 40-hex sha and dirty flag."""
    env = runrecord.capture_environment()
    assert _HEX40.match(env["code_git_sha"])
    assert isinstance(env["code_dirty"], bool)
    assert env["uv_lock_sha256"].startswith("sha256:")
