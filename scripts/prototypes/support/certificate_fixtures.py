"""Shared content-store backend for the analytic-certificate booking fixtures.

The certificate booking gate composes a carrier machine and a clipped exterior
for every analytic case and resolution. Both are expensive to build, identical
across runs, and addressed by the semantic identity the fixture sources define,
so they are published whole through the shared content store and a later run in
a fresh process loads them instead of rebuilding them.

This module is only the store. Which fixture exists, what its identity is, how
its dataset is encoded and decoded, and whether a read is a hit or a build are
all owned by the fixture owners in
:mod:`scripts.analytic_oracle_fixtures.measure` - ``cached_machine`` and
``cached_fixture_exterior`` - which take a store for the one thing they cannot
express themselves: where the object lives and how it is made visible to a
concurrent reader. :class:`ContentStoreFixtureStore` supplies that, and the two
helpers below name it as the store the booking gate uses.

The contract this store keeps:

* **The key is the defining identity.** The fixture owners include computed
  wall and state digests in their local identities. The shared store omits
  those device-rounded fields and retains the machine, discretisation, source
  parameters, support arm and kernel revision. Stored metadata records this
  canonical identity while each owner sees its local metadata on load.
* **Publication is the store's rename-onto-digest.** :func:`publish_fixture`
  hands the payload to ``nova.database.content_store.publish_files``, which
  lands it in a private directory and renames it onto its digest, so a reader
  sees the whole object or none of it and two concurrent publishers of one key
  cannot interleave. That primitive is the reason this store publishes through
  the content store rather than writing a group in place: it is safe on the GPFS
  operand root, where an in-place group write is not.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
import json
import os
from pathlib import Path
import tempfile

import xarray

from nova.database.content_store import (
    MANIFEST_FILENAME,
    ContentStoreError,
    entry_metadata,
    file_content_identity,
    publish_files,
    read_regular_bytes,
    sha256_bytes,
    verified_destination,
    verified_object_root,
)
from nova.database.filepath import operand_cache_root

CARRIER_KIND = "carrier"
EXTERIOR_KIND = "exterior"
FIXTURE_KIND_ROOT = "certificate-fixtures"
PAYLOAD_NAME = "fixture.nc"
NETCDF_ENGINE = "h5netcdf"


def fixture_root() -> Path:
    """Return the shared root the certificate fixtures are published under.

    The default is a subdirectory of the shared operand root, on the GPFS
    evidence mount, so a fixture built by one run is visible to all of them. A
    caller that needs a private root - a unit test, a scratch probe - sets
    ``NOVA_BOOK_FIXTURE_ROOT``.
    """
    override = os.environ.get("NOVA_BOOK_FIXTURE_ROOT")
    if override is not None:
        return Path(override)
    return operand_cache_root() / FIXTURE_KIND_ROOT


def kind_root(kind: str) -> Path:
    """Return the cache directory one fixture kind is published under."""
    return fixture_root() / kind


def fixture_digest(identity: dict[str, object]) -> str:
    """Return the digest of defining inputs, independent of device rounding."""
    encoded = _encoded_identity(_defining_identity(identity)).encode()
    return "sha256:" + sha256_bytes(encoded)


def _encoded_identity(identity: dict[str, object]) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def _defining_identity(identity: dict[str, object]) -> dict[str, object]:
    """Keep source inputs and revisions while omitting computed array digests."""
    canonical = json.loads(_encoded_identity(identity))
    canonical["certificate_store_identity"] = "defining-inputs"
    if "discretisation" in canonical:
        canonical["discretisation"].pop("wall_content", None)
    if "machine_semantic_key" in canonical:
        canonical.pop("analytic_state_sha256_binary64", None)
        canonical.pop("analytic_boundary_sha256_binary64", None)
    return canonical


def fixture_entry(kind: str, digest: str) -> Path | None:
    """Return the verified object directory for one fixture key, else ``None``.

    A cold cache is a miss, not an error: the object root does not exist until
    the first fixture of that kind is published. Every other fault the store
    reports about the root or the destination - a root that is a symlink, a
    destination that escaped it - is raised rather than read.
    """
    try:
        object_root = verified_object_root(kind_root(kind), create=False)
    except ContentStoreError:
        return None
    return verified_destination(object_root, digest.removeprefix("sha256:"))


def read_fixture(kind: str, digest: str) -> xarray.Dataset | None:
    """Return the published fixture dataset for one key, else ``None``.

    The store verifies every declared file's content digest before the object
    becomes visible, so a returned dataset is the one that was published under
    this key. The manifest is re-read for the payload name so a directory left
    without a manifest is a miss rather than a partial read.
    """
    directory = fixture_entry(kind, digest)
    if directory is None:
        return None
    manifest_path = directory / MANIFEST_FILENAME
    if entry_metadata(manifest_path) is None:
        return None
    payload = json.loads(read_regular_bytes(manifest_path))
    declared = {item["name"] for item in payload.get("files", [])}
    if PAYLOAD_NAME not in declared:
        return None
    payload_path = directory / PAYLOAD_NAME
    if entry_metadata(payload_path) is None:
        return None
    with xarray.open_dataset(payload_path, engine=NETCDF_ENGINE) as data:
        return data.load()


def publish_fixture(kind: str, digest: str, data: xarray.Dataset) -> Path:
    """Publish one fixture dataset through the content store.

    The dataset is serialised to a private staging file, then handed to
    :func:`content_store.publish_files`, which renames the object onto its
    digest without replacement where the filesystem admits it and with a plain
    rename where it does not. Two concurrent publishers of one key therefore
    resolve to one visible object rather than interleaving writes.
    """
    root = kind_root(kind)
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=".pending-") as staging:
        staging_path = Path(staging) / PAYLOAD_NAME
        data.to_netcdf(staging_path, engine=NETCDF_ENGINE)
        content, size = file_content_identity(staging_path)
        manifest = json.dumps(
            {
                "kind": kind,
                "files": [{"name": PAYLOAD_NAME, "sha256": content, "size": size}],
            },
            sort_keys=True,
        ).encode()
        destination = publish_files(
            root,
            digest,
            [_DeclaredFile(PAYLOAD_NAME, content, size)],
            {PAYLOAD_NAME: staging_path},
            manifest,
        )
    return destination


class _DeclaredFile:
    """One declared payload file, in the form the content store verifies."""

    def __init__(self, name: str, sha256: str, size: int):
        self.name = name
        self.sha256 = sha256
        self.size = size


class _ContentSession:
    """One fixture kind's session, moving a dataset through the content store.

    The session is opened on a digest and reads or publishes under it. Its
    location is the verified object directory on a hit, and the directory the
    store renamed the object onto on a miss; both lie beneath the kind root, so
    a caller can assert that a fixture landed where it meant to publish.
    """

    def __init__(
        self, kind: str, digest: str, location: str, identity: dict[str, object]
    ):
        self.kind = kind
        self.digest = digest
        self.location = location
        self.identity = identity
        self.lock_wait = 0.0

    @property
    def key(self) -> str:
        return self.digest

    def load(self) -> xarray.Dataset | None:
        data = read_fixture(self.kind, self.digest)
        if data is None:
            return None
        if data.attrs.get("semantic_identity") != _encoded_identity(
            _defining_identity(self.identity)
        ):
            raise ValueError("certificate store identity mismatch")
        if data.attrs.get("cache_key") != self.digest:
            raise ValueError("certificate store key mismatch")
        data.attrs["semantic_identity"] = _encoded_identity(self.identity)
        return data

    def publish(self, data: xarray.Dataset) -> None:
        canonical = data.copy(deep=False)
        canonical.attrs = dict(data.attrs)
        canonical.attrs["semantic_identity"] = _encoded_identity(
            _defining_identity(self.identity)
        )
        canonical.attrs["cache_key"] = self.digest
        self.location = str(publish_fixture(self.kind, self.digest, canonical))


class ContentStoreFixtureStore:
    """Publish one fixture kind through the shared content store.

    No lock is taken: publication is the store's rename-onto-digest, so two
    processes racing one identity resolve to one visible object without either
    waiting on the other, which is what lets this backend sit on the shared
    GPFS operand root.
    """

    def __init__(self, kind: str):
        self.kind = kind

    @contextmanager
    def session(
        self, identity: dict[str, object]
    ) -> AbstractContextManager[_ContentSession]:
        digest = fixture_digest(identity)
        entry = fixture_entry(self.kind, digest)
        yield _ContentSession(
            self.kind,
            digest,
            str(entry) if entry is not None else "",
            identity,
        )


def carrier_store() -> ContentStoreFixtureStore:
    """Return the shared root's store for the carrier fixture kind."""
    return ContentStoreFixtureStore(CARRIER_KIND)


def exterior_store() -> ContentStoreFixtureStore:
    """Return the shared root's store for the clipped exterior fixture kind."""
    return ContentStoreFixtureStore(EXTERIOR_KIND)


def cached_carrier(case, requested_cells: int, wall):
    """Return the carrier machine for one identity, loading or publishing it.

    The build, the identity and the receipt belong to
    :func:`measure.cached_machine`; this names the shared content store as the
    backend, so the booking gate builds a carrier once and a later run loads it.
    """
    from scripts.analytic_oracle_fixtures import measure as fixture

    return fixture.cached_machine(
        case,
        requested_cells,
        wall_nodes=fixture.WALL_POINT_COUNT,
        wall=wall,
        store=carrier_store(),
    )


def cached_exterior(case, analytic, machine, operator, state):
    """Return the clipped exterior for one identity, loading or publishing it.

    The build, the identity and the receipt belong to
    :func:`measure.cached_fixture_exterior`; this names the shared content store
    as the backend.
    """
    from scripts.analytic_oracle_fixtures import measure as fixture

    return fixture.cached_fixture_exterior(
        case, analytic, machine, operator, state, store=exterior_store()
    )
