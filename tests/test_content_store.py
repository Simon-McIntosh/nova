"""Publication, verified read, and tamper refusal for the content store."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from nova.database.content_store import (
    MANIFEST_FILENAME,
    ContentStoreError,
    create_private_directory,
    digest_hex,
    linux_rename_no_replace,
    open_pinned_object_root,
    publish_directory_no_replace,
    read_regular_bytes,
    sha256_bytes,
    verified_destination,
    verified_object_root,
    verify_directory_files,
    write_bytes_at,
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
    return [
        _Entry(name, sha256_bytes(data), len(data)) for name, data in files.items()
    ]


def _publish(cache: Path, files: dict[str, bytes]) -> tuple[str, list[_Entry]]:
    descriptor = _descriptor(files)
    digest = "sha256:" + sha256_bytes(descriptor)
    hex_digest = digest_hex(digest)
    object_root = verified_object_root(cache, create=True)
    descriptor_fd = open_pinned_object_root(object_root)
    try:
        name, temporary_fd = create_private_directory(descriptor_fd, hex_digest)
        try:
            for relative, data in files.items():
                write_bytes_at(temporary_fd, relative, data)
            write_bytes_at(temporary_fd, MANIFEST_FILENAME, descriptor)
        finally:
            os.close(temporary_fd)
        published = publish_directory_no_replace(
            linux_rename_no_replace(), descriptor_fd, name, hex_digest
        )
    finally:
        os.close(descriptor_fd)
    assert published is True
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
    hex_digest = digest_hex(digest)

    object_root = verified_object_root(cache, create=False)
    descriptor_fd = open_pinned_object_root(object_root)
    try:
        name, temporary_fd = create_private_directory(descriptor_fd, hex_digest)
        os.close(temporary_fd)
        published = publish_directory_no_replace(
            linux_rename_no_replace(), descriptor_fd, name, hex_digest
        )
    finally:
        os.close(descriptor_fd)

    assert published is False
    assert (directory / "wall.nc").read_bytes() == b"first"