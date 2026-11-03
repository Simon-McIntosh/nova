"""Contracts shared by coordinate readers and receipt writers."""

from __future__ import annotations

import json
import hashlib

import numpy as np
import pytest

from nova.utilities.axes import uniform_axis


def test_float32_uniform_axis_reconstructs_from_endpoints() -> None:
    stored = np.linspace(1.0, 2.0, 64, dtype=np.float32)
    axis, step = uniform_axis(stored, "radius", minimum_size=3)
    np.testing.assert_array_equal(axis, np.linspace(1.0, 2.0, 64))
    assert step == pytest.approx(1.0 / 63.0)


def test_uniform_axis_rejects_a_ragged_or_short_axis() -> None:
    with pytest.raises(ValueError, match="uniformly spaced"):
        uniform_axis([0.0, 1.0, 2.1], "height")
    with pytest.raises(ValueError, match="at least 3 nodes"):
        uniform_axis([0.0, 1.0], "height", minimum_size=3)


def test_atomic_writer_preserves_requested_json_bytes(tmp_path) -> None:
    from nova.io.json import write_atomic

    destination = tmp_path / "receipt.json"
    write_atomic(destination, {"b": 2, "a": 1}, indent=2, newline=True)
    assert destination.read_bytes() == b'{\n  "a": 1,\n  "b": 2\n}\n'
    write_atomic(destination, {"z": 3}, separators=(",", ":"))
    assert destination.read_bytes() == b'{"z":3}'
    assert list(tmp_path.iterdir()) == [destination]
    with pytest.raises(TypeError):
        write_atomic(destination, {"bad": object()})
    assert destination.read_bytes() == b'{"z":3}'


def test_canonical_json_and_trimmed_strings_share_machine_evidence_owner() -> None:
    from nova.imas.machine_evidence import EvidenceError, canonical_json, require_string

    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'
    assert canonical_json({"b": 2, "a": 1}, newline=False) == b'{"a":1,"b":2}'
    assert require_string(" value ", "field", EvidenceError) == " value "
    with pytest.raises(EvidenceError, match="trimmed"):
        require_string(" value ", "field", EvidenceError, trimmed=True)
    assert json.loads(canonical_json({"a": 1})) == {"a": 1}


def test_catalog_digest_retains_its_exact_compact_bytes() -> None:
    from nova.catalog.mast_geometry import stable_digest

    payload = {"b": 2, "a": 1}
    expected = hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
    assert stable_digest(payload) == expected
