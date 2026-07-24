"""Tests for streaming content digests."""

import hashlib
import os

import pytest

from nova.assembly.provenance import digest


def test_digest_bytes_empty():
    """Empty byte string matches the canonical sha256 of no bytes."""
    expected = hashlib.sha256(b"").hexdigest()
    assert digest.digest_bytes(b"") == f"sha256:{expected}"


def test_digest_bytes_known_vector():
    """A short byte string hashes to the known sha256 vector."""
    data = b"abc"
    expected = hashlib.sha256(data).hexdigest()
    assert digest.digest_bytes(data) == f"sha256:{expected}"


def test_digest_file_empty(tmp_path):
    """An empty file digests to the empty-input sha256."""
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert digest.digest_file(path) == f"sha256:{hashlib.sha256(b'').hexdigest()}"


def test_digest_file_small(tmp_path):
    """A small file digests identically to hashing its bytes in one shot."""
    path = tmp_path / "small.txt"
    payload = b"hello provenance\n"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert digest.digest_file(path) == f"sha256:{expected}"


def test_digest_file_chunking_matches_oneshot(tmp_path):
    """Chunked streaming of a multi-MB file matches a single-shot hash."""
    path = tmp_path / "big.bin"
    payload = os.urandom(5 * 1024 * 1024 + 12345)
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    # A tiny chunk size forces many read iterations.
    assert digest.digest_file(path, chunk_size=4096) == f"sha256:{expected}"


def test_digest_tree_deterministic_ordering(tmp_path):
    """Tree digest ordering is sorted and independent of creation order."""
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "two.txt").write_bytes(b"two")
    (tmp_path / "a" / "one.txt").write_bytes(b"one")
    (tmp_path / "root.txt").write_bytes(b"root")

    entries = digest.digest_tree(tmp_path)
    keys = list(entries)
    assert keys == sorted(keys)
    assert set(keys) == {"a/one.txt", "b/two.txt", "root.txt"}


def test_digest_tree_entry_fields(tmp_path):
    """Each tree entry carries digest, size, and a UTC ISO mtime."""
    payload = b"payload"
    (tmp_path / "f.txt").write_bytes(payload)
    entries = digest.digest_tree(tmp_path)
    entry = entries["f.txt"]
    assert entry["digest"] == f"sha256:{hashlib.sha256(payload).hexdigest()}"
    assert entry["size"] == len(payload)
    assert entry["mtime"].endswith("+00:00")


def test_digest_file_missing(tmp_path):
    """A missing path raises rather than returning a bogus digest."""
    with pytest.raises(FileNotFoundError):
        digest.digest_file(tmp_path / "nope.bin")
