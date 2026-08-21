"""Concurrent verification contracts for content-addressed machine artifacts."""

from __future__ import annotations

import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
from queue import Empty
from typing import Any

import pytest

from nova.imas.machine_artifact import (
    ArtifactShotRange,
    MachineArtifactError,
    create_machine_artifact_manifest,
    materialize_machine_artifact,
    resolve_machine_artifact,
)


SHARED_CACHE = Path("/run/user/39486/imas-ambix-machine-artifact")
SHARED_DIGEST = (
    "sha256:44f71af85061ff8463d673601cc9db8d4a9b7c605430e73486db1a475b4e2d26"
)
SHOT_STORE = Path("/work/projects/imas_gpu/mast/level1/shots")
SHOT = 21978
_HDF5_USER_BLOCK_SIZE = 1024
_HDF5_CONSISTENCY_OFFSET = _HDF5_USER_BLOCK_SIZE + 20
_HDF5_CONSISTENCY_WIDTH = 4


def _hdf5_image() -> bytes:
    """Return a small image with a version-zero superblock after a user block."""

    image = bytearray((index * 17 + 3) % 256 for index in range(2048))
    image[_HDF5_USER_BLOCK_SIZE : _HDF5_USER_BLOCK_SIZE + 8] = b"\x89HDF\r\n\x1a\n"
    image[_HDF5_USER_BLOCK_SIZE + 8] = 0
    image[
        _HDF5_CONSISTENCY_OFFSET : (_HDF5_CONSISTENCY_OFFSET + _HDF5_CONSISTENCY_WIDTH)
    ] = b"\x00" * _HDF5_CONSISTENCY_WIDTH
    return bytes(image)


def _materialized_image(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "master.h5"
    payload.write_bytes(_hdf5_image())
    manifest = create_machine_artifact_manifest(
        source,
        machine="mast",
        dd_version="4.1.1",
        registry_digest="a" * 64,
        physical_digest="b" * 16,
        shot_ranges=(
            ArtifactShotRange(
                first_shot=11766,
                last_shot=30471,
                physical_digest="b" * 16,
                evidence="observed",
            ),
        ),
        complete=False,
        unresolved_gaps=("test fixture is not operator-ready",),
    )
    stored = materialize_machine_artifact(source, tmp_path / "cache", manifest)
    return tmp_path / "cache", stored, payload


def test_manifest_identity_canonicalizes_only_hdf5_consistency_flags(
    tmp_path: Path,
) -> None:
    cache, stored, source = _materialized_image(tmp_path)
    recorded = stored.manifest.files[0]

    assert recorded.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    opened = bytearray((stored.directory / recorded.name).read_bytes())
    opened[
        _HDF5_CONSISTENCY_OFFSET : (_HDF5_CONSISTENCY_OFFSET + _HDF5_CONSISTENCY_WIDTH)
    ] = b"\x05\x00\x00\x00"
    (stored.directory / recorded.name).write_bytes(opened)

    verified = resolve_machine_artifact(
        cache,
        stored.digest,
        allow_incomplete=True,
    )

    assert verified.manifest == stored.manifest
    assert verified.digest == stored.digest


def test_every_other_hdf5_byte_remains_verified(tmp_path: Path) -> None:
    cache, stored, _source = _materialized_image(tmp_path)
    payload = stored.directory / stored.manifest.files[0].name
    original = payload.read_bytes()
    consistency_offsets = range(
        _HDF5_CONSISTENCY_OFFSET,
        _HDF5_CONSISTENCY_OFFSET + _HDF5_CONSISTENCY_WIDTH,
    )
    for offset in range(len(original)):
        if offset in consistency_offsets:
            continue
        altered = bytearray(original)
        altered[offset] ^= 1
        payload.write_bytes(altered)

        with pytest.raises(MachineArtifactError, match="checksum mismatch"):
            resolve_machine_artifact(cache, stored.digest, allow_incomplete=True)
    payload.write_bytes(original)


def _open_artifact_worker(
    worker_index: int,
    command_queue: Any,
    ready_queue: Any,
) -> None:
    import imas

    digest_hex = SHARED_DIGEST.removeprefix("sha256:")
    directory = SHARED_CACHE / "sha256" / digest_hex
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    uri = f"imas:hdf5?path={directory}"
    while True:
        command, trial = command_queue.get()
        if command == "stop":
            return
        entry = imas.DBEntry(uri, "r", dd_version=str(manifest["dd_version"]))
        ready_queue.put((worker_index, trial))
        close_command, close_trial = command_queue.get()
        if (close_command, close_trial) != ("close", trial):
            raise RuntimeError("artifact worker received an invalid close command")
        entry.close()


def _require_shared_artifact() -> None:
    directory = SHARED_CACHE / "sha256" / SHARED_DIGEST.removeprefix("sha256:")
    if not (directory / "manifest.json").is_file():
        pytest.skip("shared content-addressed MAST artifact is not mounted")


def test_verified_artifact_survives_repeated_concurrent_opens() -> None:
    _require_shared_artifact()
    directory = SHARED_CACHE / "sha256" / SHARED_DIGEST.removeprefix("sha256:")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    master_identity = next(
        row["sha256"] for row in manifest["files"] if row["name"] == "master.h5"
    )
    context = get_context("spawn")
    ready_queue = context.Queue()
    command_queues = [context.Queue() for _ in range(4)]
    workers = [
        context.Process(
            target=_open_artifact_worker,
            args=(index, command_queues[index], ready_queue),
        )
        for index in range(4)
    ]
    for worker in workers:
        worker.start()
    raw_mismatches = 0
    try:
        for trial in range(20):
            for command_queue in command_queues:
                command_queue.put(("open", trial))
            opened = {ready_queue.get(timeout=30) for _ in workers}
            assert opened == {(index, trial) for index in range(4)}
            raw_digest = hashlib.sha256(
                (directory / "master.h5").read_bytes()
            ).hexdigest()
            raw_mismatches += raw_digest != master_identity
            resolved = resolve_machine_artifact(
                SHARED_CACHE,
                SHARED_DIGEST,
                allow_incomplete=True,
            )
            assert resolved.digest == SHARED_DIGEST
            for command_queue in command_queues:
                command_queue.put(("close", trial))
    finally:
        for command_queue in command_queues:
            command_queue.put(("stop", -1))
        for worker in workers:
            worker.join(30)
            if worker.is_alive():
                worker.kill()
                worker.join()
    assert [worker.exitcode for worker in workers] == [0, 0, 0, 0]
    assert raw_mismatches == 20


def _score_partition_worker(
    slice_start: int,
    artifact_path: Path,
    start_event: Any,
    result_queue: Any,
) -> None:
    from nova.imas.mast_parity_gate import bank_production_partition

    try:
        start_event.wait(30)
        report = bank_production_partition(
            SHOT,
            slice_start=slice_start,
            slice_stop=slice_start + 1,
            artifact_path=artifact_path,
            artifact_cache=SHARED_CACHE,
            artifact_digest=SHARED_DIGEST,
            store=SHOT_STORE,
        )
    except Exception as error:
        result_queue.put((slice_start, type(error).__name__, str(error)))
    else:
        result_queue.put(
            (
                slice_start,
                "complete",
                len(report.scored_slices),
                len(report.skipped_slices),
            )
        )


def test_four_scoring_partitions_share_one_verified_artifact(tmp_path: Path) -> None:
    _require_shared_artifact()
    if not (SHOT_STORE / f"{SHOT}.zarr").is_dir():
        pytest.skip("corrected MAST shot store is not mounted")
    context = get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    starts = (0, 1, 2, 3)
    workers = [
        context.Process(
            target=_score_partition_worker,
            args=(
                start,
                tmp_path / f"partition-{start}.json",
                start_event,
                result_queue,
            ),
        )
        for start in starts
    ]
    for worker in workers:
        worker.start()
    start_event.set()
    results = []
    try:
        for _ in workers:
            results.append(result_queue.get(timeout=300))
    except Empty as error:
        raise AssertionError("a scoring partition did not complete") from error
    finally:
        for worker in workers:
            worker.join(30)
            if worker.is_alive():
                worker.kill()
                worker.join()

    by_start = {result[0]: result[1:] for result in results}
    assert set(by_start) == set(starts)
    assert all(result[0] == "complete" for result in by_start.values())
    assert all(result[1] + result[2] == 1 for result in by_start.values())
    assert [worker.exitcode for worker in workers] == [0, 0, 0, 0]
    assert all((tmp_path / f"partition-{start}.json").is_file() for start in starts)
