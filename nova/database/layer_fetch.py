"""Fetch a pinned OCI dataset layer by digest into the content store."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Sequence

from nova.database import content_store
from nova.database.content_store import (
    MANIFEST_FILENAME,
    ContentStoreError,
    digest_hex,
    publish_files,
    read_regular_bytes,
    verified_destination,
    verified_object_root,
    verify_directory_files,
)

REGISTRY = "ghcr.io/iterorganization/efitpp-test-data"
ORAS_EXECUTABLE = "oras"

# Transport failures leave the layer's state unknown, so callers may skip; a
# decisive registry answer (an unknown digest) is a fetch error instead.
_UNREACHABLE_MARKERS = (
    "connection refused",
    "connection reset",
    "connection timed out",
    "no such host",
    "dial tcp",
    "i/o timeout",
    "network is unreachable",
    "tls handshake timeout",
    "unauthorized",
    "authentication required",
)

Runner = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


class LayerFetchError(Exception):
    """A layer could not be retrieved or admitted into the content store."""


class LayerRegistryUnreachable(LayerFetchError):
    """The registry could not be reached, so the layer's state is unknown."""


class LayerDigestMismatch(LayerFetchError):
    """The fetched bytes do not hash to the digest the record pins."""


class _Declaration:
    """Per-file identity the content store verifies a directory against."""

    __slots__ = ("name", "sha256", "size")

    def __init__(self, name: str, sha256: str, size: bool | int) -> None:
        self.name = name
        self.sha256 = sha256
        self.size = int(size)


def verify_layer_digest(blob: Path | str, layer_digest: str) -> int:
    """Return the blob size after confirming its sha256 equals the pinned digest."""

    expected = _validated_digest(layer_digest)
    path = Path(blob)
    actual = _sha256_file(path)
    if actual != expected:
        raise LayerDigestMismatch(
            f"layer digest mismatch: record pins sha256:{expected}, "
            f"fetched bytes hash to sha256:{actual}"
        )
    return path.stat().st_size


def fetch_layer(
    layer_digest: str,
    *,
    cache_directory: Path | str,
    name: str | None = None,
    registry: str = REGISTRY,
    oras: str = ORAS_EXECUTABLE,
    runner: "Runner | None" = None,
) -> Path:
    """Fetch, verify and publish one OCI layer, returning its verified store path.

    The layer is addressed by its digest, never by a mutable tag: the reference
    handed to the registry is ``<registry>@<layer_digest>``.  A layer already
    published under that digest is returned from the store without contacting
    the registry.  The fetched bytes are refused unless their sha256 equals the
    pinned digest, and the published object is re-verified through the content
    store's read path before it is returned.
    """

    hex_digest = _validated_digest(layer_digest)
    object_root = verified_object_root(cache_directory, create=True)
    published = verified_destination(object_root, hex_digest)
    if published is not None:
        return _verified_layer(published)

    runner = _default_runner if runner is None else runner
    with tempfile.TemporaryDirectory(prefix="layer-fetch-") as scratch:
        blob = Path(scratch) / "blob"
        reference = f"{registry}@{layer_digest}"
        completed = runner((oras, "blob", "fetch", reference, "--output", str(blob)))
        if completed.returncode != 0:
            raise _classify_failure(reference, completed)
        if not blob.is_file():
            raise LayerFetchError(
                f"oras reported success for {reference} but wrote no blob"
            )
        verify_layer_digest(blob, layer_digest)
        return _publish_layer(cache_directory, hex_digest, blob, name)


def _publish_layer(
    cache_directory: Path | str,
    hex_digest: str,
    blob: Path,
    name: str | None,
) -> Path:
    stored_name = _stored_name(name, hex_digest)
    identity_digest, size = content_store.file_content_identity(blob)
    manifest = _manifest_bytes(stored_name, identity_digest, size)

    directory = publish_files(
        cache_directory,
        f"sha256:{hex_digest}",
        [_Declaration(stored_name, identity_digest, size)],
        {stored_name: blob},
        manifest,
    )
    return _verified_layer(directory)


def _verified_layer(directory: Path | None) -> Path:
    if directory is None:
        raise LayerFetchError("layer is absent from the content store")
    path = Path(directory)
    manifest = read_regular_bytes(path / MANIFEST_FILENAME)
    declarations = _declarations_from_manifest(manifest)
    if not declarations:
        raise LayerFetchError(f"stored layer declares no files: {path}")
    verify_directory_files(path, declarations, allow_manifest=True)
    return path


def _declarations_from_manifest(manifest: bytes) -> list[_Declaration]:
    try:
        payload = json.loads(manifest)
    except json.JSONDecodeError as error:
        raise LayerFetchError("stored layer manifest is not valid JSON") from error
    if not isinstance(payload, dict):
        raise LayerFetchError("stored layer manifest must be an object")
    return [
        _Declaration(name, str(record["sha256"]), int(record["size"]))
        for name, record in payload.items()
    ]


def _manifest_bytes(name: str, sha256: str, size: int) -> bytes:
    payload = {name: {"sha256": sha256, "size": size}}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _stored_name(name: str | None, hex_digest: str) -> str:
    return name if name is not None else f"{hex_digest}.blob"


def _default_runner(argv: Sequence[str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(list(argv), capture_output=True, text=True)


def _classify_failure(
    reference: str,
    completed: "subprocess.CompletedProcess[str]",
) -> LayerFetchError:
    output = (completed.stderr or completed.stdout or "").strip()
    text = f"{completed.stdout or ''}\n{completed.stderr or ''}".lower()
    summary = output[:200]
    if any(marker in text for marker in _UNREACHABLE_MARKERS):
        return LayerRegistryUnreachable(
            f"registry unreachable for {reference}: {summary}"
        )
    return LayerFetchError(
        f"oras could not fetch {reference} (exit {completed.returncode}): {summary}"
    )


def _validated_digest(layer_digest: str) -> str:
    try:
        return digest_hex(layer_digest)
    except ContentStoreError as error:
        raise LayerFetchError(str(error)) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise LayerFetchError(f"cannot read fetched layer {path}") from error
    with os.fdopen(descriptor, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
