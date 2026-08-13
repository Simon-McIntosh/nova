from pathlib import Path
import subprocess

import pytest

from benchmarks.measurement_provenance import (
    MeasurementProvenanceError,
    measurement_stamp,
    resolve_checkout,
)


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repository(path: Path) -> Path:
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "Nova test")
    _git(path, "config", "user.email", "nova-test@example.invalid")
    (path / "tracked.txt").write_text("measured\n")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "--quiet", "-m", "test: create measured tree")
    return path


def test_clean_checkout_emits_its_commit_stamp(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    expected_sha = _git(repository, "rev-parse", "HEAD")

    state = resolve_checkout(repository)

    assert state.commit_sha == expected_sha
    assert state.has_porcelain_output is False
    assert measurement_stamp(repository) == expected_sha


def test_modified_tracked_file_prevents_a_stamp(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "tracked.txt").write_text("changed during measurement\n")

    state = resolve_checkout(repository)

    assert state.has_porcelain_output is True
    with pytest.raises(MeasurementProvenanceError, match="status --porcelain"):
        measurement_stamp(repository)


def test_non_repository_prevents_a_stamp(tmp_path: Path) -> None:
    with pytest.raises(MeasurementProvenanceError, match="Git worktree"):
        measurement_stamp(tmp_path)
