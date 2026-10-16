"""Identity-keyed shared cache for the analytic-certificate booking fixtures.

The certificate booking gate composes a carrier machine and a clipped exterior
for every analytic case and resolution. Both are expensive to build, identical
across runs, and addressed by the semantic identity the fixture sources define,
so they are published whole through the shared content store and a later run in
a fresh process loads them instead of rebuilding them.

The contract this module keeps:

* **The key is the identity.** ``fixture.cache_identity`` (carrier) and
  ``fixture._exterior_cache_identity`` (exterior) name every input a fixture
  depends on - the closed-form constants, the discretisation, the builder and
  its revision, the analytic state digest. The store digest is the SHA-256 of
  that identity's canonical JSON, so a source change that moves any one input
  addresses a different object rather than reloading a stale fixture.
* **One owner per fixture.** :func:`cached_carrier` owns the carrier and
  :func:`cached_exterior` owns the exterior; each is the only writer of its
  object. Publication goes through ``nova.database.content_store``, which lands
  the payload in a private directory and renames it onto its digest, so a
  reader sees the whole object or none of it and two concurrent publishers of
  one key cannot interleave. That primitive is the reason this module publishes
  through the content store rather than writing a group in place: it is safe on
  the GPFS operand root, where an in-place group write is not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from time import perf_counter
from typing import Callable

import numpy as np
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
IDENTITY_NAME = "identity.json"
NETCDF_ENGINE = "h5netcdf"


def fixture_root() -> Path:
    """Return the shared root the certificate fixtures are published under.

    The default is a subdirectory of the shared operand root, on the GPFS
    evidence mount, so a fixture built by one run is visible to the next. A
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
    """Return the content-store digest naming one fixture from its identity."""
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + sha256_bytes(encoded)


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


def _load_or_build(
    kind: str,
    identity: dict[str, object],
    build: Callable[[], xarray.Dataset],
) -> tuple[xarray.Dataset, dict[str, object]]:
    """Return the fixture dataset for one identity, loading or publishing it.

    On a hit the load wall is what this process paid; on a miss the build wall
    is the assembly and the payload is published under the identity's digest.
    The receipt records which of the two happened, so a run can be read for
    whether it built or loaded without consulting the store.
    """
    digest = fixture_digest(identity)
    started = perf_counter()
    data = read_fixture(kind, digest)
    if data is not None:
        receipt = {
            "kind": kind,
            "digest": digest,
            "store": str(fixture_entry(kind, digest)),
            "hit": True,
            "load_seconds": perf_counter() - started,
            "build_seconds": 0.0,
        }
        return data, receipt
    build_started = perf_counter()
    data = build()
    build_seconds = perf_counter() - build_started
    destination = publish_fixture(kind, digest, data)
    receipt = {
        "kind": kind,
        "digest": digest,
        "store": str(destination),
        "hit": False,
        "load_seconds": 0.0,
        "build_seconds": build_seconds,
    }
    return data, receipt


def _encoded(identity: dict[str, object]) -> str:
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def cached_carrier(case, requested_cells: int, wall: np.ndarray | None):
    """Return the carrier machine for one identity, loading or publishing it.

    The carrier build is the host assembly the fixture sources define; its
    identity and its dataset codec are taken from the fixture module, so the
    object stored here is bit-identical to one the in-place builder would have
    produced and carries the same schema and identity attributes.
    """
    from scripts.analytic_oracle_fixtures import measure as fixture

    identity = fixture.cache_identity(
        case,
        requested_cells=requested_cells,
        wall_nodes=fixture.WALL_POINT_COUNT,
        wall=wall,
    )
    key = fixture_digest(identity)

    def build() -> xarray.Dataset:
        machine = fixture.build_machine(
            case,
            requested_cells,
            wall_nodes=fixture.WALL_POINT_COUNT,
            wall=wall,
        )
        return fixture._dataset(machine, identity, key)

    data, receipt = _load_or_build(CARRIER_KIND, identity, build)
    machine = fixture._from_dataset(data, identity, key)
    machine.cache.update(
        {
            "store": receipt["store"],
            "semantic_key": key,
            "hit": receipt["hit"],
            "load_seconds": receipt["load_seconds"],
            "build_seconds": receipt["build_seconds"],
        }
    )
    return machine, receipt


def cached_exterior(case, analytic, machine, operator, state):
    """Return the clipped exterior fixture for one identity, loading or publishing.

    The exterior build and the analytic state it is posed against are taken from
    the fixture module, so the published object is the same moments and target
    vector the in-place builder produced. Its identity keys on the carrier's own
    semantic key, the analytic state digest and the support arm. The arm is
    named explicitly on both the identity and the build: the two arms drive
    different clips and different quadratures, so an object built on one arm
    must never be addressed by the other.
    """
    from scripts.analytic_oracle_fixtures import measure as fixture

    support_arm = fixture.EXTERIOR_SUPPORT_ARM
    identity = fixture._exterior_cache_identity(
        case, analytic, machine, state, support_arm
    )
    key = fixture_digest(identity)

    def build() -> xarray.Dataset:
        moments = fixture.exact_current_moments(
            case, operator, state, analytic=analytic, support_arm=support_arm
        )
        coefficients = operator.coupling_current_moments(moments)
        exterior = np.asarray(state) - fixture._internal_flux_image(
            operator, coefficients
        )
        return xarray.Dataset(
            {
                "cell_current": (("cell",), np.asarray(moments.cell_current)),
                "radial_moment": (("cell",), np.asarray(moments.radial_moment)),
                "vertical_moment": (("cell",), np.asarray(moments.vertical_moment)),
                "exterior": (("target",), exterior),
            },
            attrs={
                "cache_schema": fixture.EXTERIOR_CACHE_SCHEMA,
                "cache_key": key,
                "semantic_identity": _encoded(identity),
            },
        )

    data, receipt = _load_or_build(EXTERIOR_KIND, identity, build)
    if data.attrs.get("cache_schema") != fixture.EXTERIOR_CACHE_SCHEMA:
        raise ValueError("fixture exterior cache schema mismatch")
    if data.attrs.get("semantic_identity") != _encoded(identity):
        raise ValueError("fixture exterior cache identity mismatch")
    from nova.equilibrium.stencil_mesh import CellCurrentMoments

    moments = CellCurrentMoments(
        *(
            np.asarray(data[name].values, dtype=np.float64)
            for name in ("cell_current", "radial_moment", "vertical_moment")
        )
    )
    exterior = np.asarray(data["exterior"].values, dtype=np.float64)
    return moments, exterior, receipt
