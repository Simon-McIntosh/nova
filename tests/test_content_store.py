"""Publication, verified read, and tamper refusal for the content store."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from nova.database import content_store
from nova.database.content_store import (
    MANIFEST_FILENAME,
    ContentStoreError,
    digest_hex,
    publish_files,
    read_regular_bytes,
    sha256_bytes,
    verified_destination,
    verified_object_root,
    verify_directory_files,
)


class _Entry:
    """Minimal declaration that verify_directory_files checks a file against."""

    def __init__(self, name: str, sha256: str, size: int) -> None:
        self.name = name
        self.sha256 = sha256
        self.size = size


def _descriptor(files: dict[str, bytes]) -> bytes:
    payload = {
        name: {"sha256": sha256_bytes(data), "size": len(data)}
        for name, data in files.items()
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _entries(files: dict[str, bytes]) -> list[_Entry]:
    return [_Entry(name, sha256_bytes(data), len(data)) for name, data in files.items()]


def _publish(cache: Path, files: dict[str, bytes]) -> tuple[str, list[_Entry]]:
    descriptor = _descriptor(files)
    digest = "sha256:" + sha256_bytes(descriptor)
    publish_files(cache, digest, _entries(files), files, descriptor)
    return digest, _entries(files)


def _verified_read(cache: Path, digest: str, entries: list[_Entry]) -> Path:
    hex_digest = digest_hex(digest)
    object_root = verified_object_root(cache, create=False)
    directory = verified_destination(object_root, hex_digest)
    assert directory is not None
    descriptor = read_regular_bytes(directory / MANIFEST_FILENAME)
    assert sha256_bytes(descriptor) == hex_digest
    verify_directory_files(directory, entries, allow_manifest=True)
    return directory


def test_publish_then_verified_read_returns_the_files(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    digest, entries = _publish(cache, {"wall.nc": b"payload", "meta.json": b"{}"})

    directory = _verified_read(cache, digest, entries)

    assert (directory / "wall.nc").read_bytes() == b"payload"
    assert (directory / "meta.json").read_bytes() == b"{}"


def test_flipping_one_byte_makes_the_verified_read_refuse(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    digest, entries = _publish(cache, {"wall.nc": b"payload-bytes"})
    directory = _verified_read(cache, digest, entries)

    payload = bytearray((directory / "wall.nc").read_bytes())
    payload[0] ^= 0x01
    (directory / "wall.nc").write_bytes(bytes(payload))

    with pytest.raises(ContentStoreError, match="checksum mismatch"):
        _verified_read(cache, digest, entries)


def test_repeated_publication_keeps_the_first_object(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    digest, entries = _publish(cache, {"wall.nc": b"first"})
    directory = _verified_read(cache, digest, entries)
    assert (
        publish_files(
            cache,
            digest,
            entries,
            {"wall.nc": b"first"},
            _descriptor({"wall.nc": b"first"}),
        )
        == directory
    )
    assert (directory / "wall.nc").read_bytes() == b"first"


def test_publish_files_rejects_changed_source_before_visibility(tmp_path: Path) -> None:
    expected = b"expected"
    actual = tmp_path / "wall.nc"
    actual.write_bytes(b"altered!")
    descriptor = _descriptor({"wall.nc": expected})
    digest = "sha256:" + sha256_bytes(descriptor)
    cache = tmp_path / "cache"

    with pytest.raises(ContentStoreError, match="checksum mismatch"):
        publish_files(
            cache,
            digest,
            _entries({"wall.nc": expected}),
            {"wall.nc": actual},
            descriptor,
        )

    assert (
        verified_destination(
            verified_object_root(cache, create=False), digest_hex(digest)
        )
        is None
    )


def test_publish_files_concurrent_writers_leave_one_verified_object(
    tmp_path: Path,
) -> None:
    files = {"wall.nc": b"same-payload"}
    descriptor = _descriptor(files)
    digest = "sha256:" + sha256_bytes(descriptor)
    cache = tmp_path / "cache"
    rendezvous = Barrier(2)

    def publish() -> Path:
        def create_together(root: int, name: str) -> tuple[str, int]:
            private = content_store._create_private_directory(root, name)
            rendezvous.wait(timeout=10)
            return private

        return publish_files(
            cache,
            digest,
            _entries(files),
            files,
            descriptor,
            private_directory_factory=create_together,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = tuple(pool.map(lambda _: publish(), range(2)))

    assert first == second
    assert _verified_read(cache, digest, _entries(files)) == first
    assert (first / "wall.nc").read_bytes() == files["wall.nc"]
    assert [path.name for path in first.parent.iterdir()] == [digest_hex(digest)]
