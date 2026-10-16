"""Publication, verified read, and tamper refusal for the content store."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
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


# Publication on a parallel filesystem is exercised in fresh processes, because
# the property under test is what two independent publishers observe rather than
# what one interpreter's threads observe.  The cache lives on a shared mount
# whose rename semantics (POSIX rename over a non-empty directory) the strategy
# relies on, so the test root defaults to the GPFS home and is overridable.

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_GPFS_ROOT = Path(
    os.environ.get(
        "NOVA_CONTENT_STORE_GPFS_ROOT",
        str(Path.home() / ".cache" / "nova" / "content-store-gpfs-tests"),
    )
)

_PUBLISH_DRIVER = '''\
"""Publish one content-store entry, or crash before it becomes visible."""

import json
import os
import sys
import time
from pathlib import Path

from nova.database import content_store


class _Entry:
    def __init__(self, name, data):
        self.name = name
        self.sha256 = content_store.sha256_bytes(data)
        self.size = len(data)


def main():
    cache = Path(sys.argv[1])
    payload = bytes.fromhex(sys.argv[2])
    rendezvous = Path(sys.argv[3]) if sys.argv[3] else None
    crash = "--crash" in sys.argv
    files = {"matrix.npy": payload}
    entries = [_Entry(name, data) for name, data in files.items()]
    descriptor = json.dumps(
        {e.name: {"sha256": e.sha256, "size": e.size} for e in entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = "sha256:" + content_store.sha256_bytes(descriptor)
    if rendezvous is not None:
        (rendezvous / ("ready-%d" % os.getpid())).write_text("")
        deadline = time.monotonic() + 30.0
        while not (rendezvous / "go").exists():
            if time.monotonic() > deadline:
                return 2
            time.sleep(0.002)
    rename_factory = (lambda: os._exit(1)) if crash else None
    destination = content_store.publish_files(
        cache, digest, entries, files, descriptor, rename_factory=rename_factory
    )
    print(str(destination))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


@pytest.fixture()
def gpfs_cache() -> Iterator[Path]:
    root = _GPFS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    unique = root / f"{os.getpid()}-{secrets.token_hex(6)}"
    unique.mkdir()
    try:
        yield unique / "store"
    finally:
        shutil.rmtree(unique, ignore_errors=True)


def _driver(tmp_path: Path) -> Path:
    script = tmp_path / "publish_driver.py"
    script.write_text(_PUBLISH_DRIVER)
    return script


def _driver_env() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_REPOSITORY_ROOT), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment["JAX_PLATFORMS"] = "cpu"
    return environment


def _payload_digest(payload: bytes) -> str:
    descriptor = _descriptor({"matrix.npy": payload})
    return "sha256:" + sha256_bytes(descriptor)


def test_gpfs_publish_then_verified_read(tmp_path: Path, gpfs_cache: Path) -> None:
    payload = b"gpfs-value-matrix" * 512
    entries = _entries({"matrix.npy": payload})
    completed = subprocess.run(
        [sys.executable, str(_driver(tmp_path)), str(gpfs_cache), payload.hex(), ""],
        env=_driver_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr

    directory = _verified_read(gpfs_cache, _payload_digest(payload), entries)
    assert (directory / "matrix.npy").read_bytes() == payload
    assert directory.is_relative_to(gpfs_cache.resolve())


def test_gpfs_concurrent_same_key_publishers_leave_one_entry(
    tmp_path: Path, gpfs_cache: Path
) -> None:
    payload = b"gpfs-shared-key" * 512
    rendezvous = tmp_path / "rendezvous"
    rendezvous.mkdir()
    driver = _driver(tmp_path)
    command = [
        sys.executable,
        str(driver),
        str(gpfs_cache),
        payload.hex(),
        str(rendezvous),
    ]
    processes = [
        subprocess.Popen(
            command, env=_driver_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        for _ in range(2)
    ]
    deadline = time.monotonic() + 30.0
    while len(list(rendezvous.glob("ready-*"))) < 2:
        assert time.monotonic() < deadline, "publishers did not reach the rendezvous"
        time.sleep(0.005)
    (rendezvous / "go").write_text("")
    results = [process.communicate(timeout=120) for process in processes]
    for process, (_, stderr) in zip(processes, results, strict=True):
        assert process.returncode == 0, stderr.decode()

    destinations = {stdout.decode().strip() for stdout, _ in results}
    assert len(destinations) == 1
    directory = _verified_read(
        gpfs_cache, _payload_digest(payload), _entries({"matrix.npy": payload})
    )
    assert str(directory) in destinations
    store_root = verified_object_root(gpfs_cache, create=False)
    assert [entry.name for entry in store_root.iterdir()] == [
        digest_hex(_payload_digest(payload))
    ]


def test_gpfs_crash_mid_publish_leaves_no_visible_entry(
    tmp_path: Path, gpfs_cache: Path
) -> None:
    payload = b"gpfs-crashed-publish" * 512
    completed = subprocess.run(
        [
            sys.executable,
            str(_driver(tmp_path)),
            str(gpfs_cache),
            payload.hex(),
            "",
            "--crash",
        ],
        env=_driver_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 1, completed.stderr

    object_root = verified_object_root(gpfs_cache, create=False)
    assert (
        verified_destination(object_root, digest_hex(_payload_digest(payload))) is None
    )
    visible = [
        entry.name for entry in object_root.iterdir() if not entry.name.startswith(".")
    ]
    assert visible == []


def test_gpfs_corrupted_file_is_refused_on_read(
    tmp_path: Path, gpfs_cache: Path
) -> None:
    payload = b"gpfs-tamper-target" * 512
    entries = _entries({"matrix.npy": payload})
    subprocess.run(
        [sys.executable, str(_driver(tmp_path)), str(gpfs_cache), payload.hex(), ""],
        env=_driver_env(),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    directory = _verified_read(gpfs_cache, _payload_digest(payload), entries)

    stored = bytearray((directory / "matrix.npy").read_bytes())
    stored[0] ^= 0x01
    (directory / "matrix.npy").write_bytes(bytes(stored))

    with pytest.raises(ContentStoreError, match="checksum mismatch"):
        _verified_read(gpfs_cache, _payload_digest(payload), entries)


def test_entry_without_completion_marker_is_not_served(gpfs_cache: Path) -> None:
    payload = b"gpfs-marker-less" * 512
    digest, entries = _publish(gpfs_cache, {"matrix.npy": payload})
    directory = _verified_read(gpfs_cache, digest, entries)

    (directory / MANIFEST_FILENAME).unlink()

    object_root = verified_object_root(gpfs_cache, create=False)
    assert verified_destination(object_root, digest_hex(digest)) is None
