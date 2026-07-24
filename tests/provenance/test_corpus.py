"""Tests for the corpus resolver and registry sync.

The resolver-location tests and the sync/extract/verify tests use fabricated
trees and a stub puller so no network or real corpus is required. One opt-in
test exercises the real GHCR pull; it skips unless ``NOVA_CORPUS_INTEGRATION``
is set so the default suite stays offline.
"""

from __future__ import annotations

import os
from pathlib import Path
import tarfile

import pytest

from nova.assembly.provenance import corpus, digest, yamlio


def _fake_kind(archive: str = "fake.tar.gz", verify: bool = False) -> corpus.CorpusKind:
    return corpus.CorpusKind(
        name="appdata",  # 'appdata' enables the legacy-mount fallback branch
        tag="fake-tag",
        digest="sha256:" + "0" * 64,
        archive=archive,
        verify=verify,
    )


def _stage(base: Path, tag: str, nest: str = "nova") -> Path:
    """Create ``base/tag/<nest>/sector_modules`` and return the sector dir."""
    sector = base / tag / nest / "sector_modules"
    sector.mkdir(parents=True)
    (sector / "Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.xlsx").write_bytes(b"wb")
    return sector


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Point the resolver at empty scratch homes and drop the env override."""
    monkeypatch.setattr(corpus, "NORMA_DATA_HOME", tmp_path / "home", raising=True)
    monkeypatch.setattr(
        corpus, "LEGACY_APPDATA", tmp_path / "nonexistent-legacy", raising=True
    )
    monkeypatch.delenv("NOVA_CORPUS_ROOT", raising=False)


def test_resolve_none_when_absent():
    assert corpus.resolve_corpus(_fake_kind()) is None


def test_resolve_from_norma_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    sector = _stage(home, "fake-tag", nest="nova")
    monkeypatch.setattr(corpus, "NORMA_DATA_HOME", home)
    assert corpus.resolve_corpus(_fake_kind()) == sector


def test_resolve_finds_directly_nested_tree(monkeypatch, tmp_path):
    """A tarball that unpacks straight to ``<tag>/sector_modules`` resolves."""
    home = tmp_path / "home"
    sector = home / "fake-tag" / "sector_modules"
    sector.mkdir(parents=True)
    monkeypatch.setattr(corpus, "NORMA_DATA_HOME", home)
    assert corpus.resolve_corpus(_fake_kind()) == sector


def test_resolve_prefers_env_root(monkeypatch, tmp_path):
    home = tmp_path / "home"
    _stage(home, "fake-tag")
    env_root = tmp_path / "elsewhere"
    env_sector = _stage(env_root, "fake-tag")
    monkeypatch.setattr(corpus, "NORMA_DATA_HOME", home)
    monkeypatch.setenv("NOVA_CORPUS_ROOT", str(env_root))
    assert corpus.resolve_corpus(_fake_kind()) == env_sector


def test_resolve_legacy_appdata_fallback(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy" / "sector_modules"
    legacy.mkdir(parents=True)
    monkeypatch.setattr(corpus, "LEGACY_APPDATA", legacy)
    assert corpus.resolve_corpus(_fake_kind()) == legacy


def test_preserved_pickle_returns_path_when_present(monkeypatch, tmp_path):
    home = tmp_path / "home"
    sector = _stage(home, "fake-tag")
    (sector / "ILIS_nominal.pickle").write_bytes(b"\x80\x04.")
    monkeypatch.setattr(corpus, "NORMA_DATA_HOME", home)
    got = corpus.preserved_pickle("ILIS_nominal.pickle", _fake_kind())
    assert got == sector / "ILIS_nominal.pickle"


def test_preserved_pickle_none_when_absent():
    assert corpus.preserved_pickle("ILIS_nominal.pickle", _fake_kind()) is None


def test_sync_idempotent_when_present(monkeypatch, tmp_path):
    home = tmp_path / "home"
    sector = _stage(home, "fake-tag")
    monkeypatch.setattr(corpus, "NORMA_DATA_HOME", home)

    def _boom(*_args, **_kwargs):
        raise AssertionError("puller must not run when a copy already resolves")

    assert corpus.sync(_fake_kind(), puller=_boom) == sector


def _archive_puller(payload_root: Path, archive_name: str):
    """Return a puller stub that drops a tarball of ``payload_root`` into dest."""

    def puller(_repo: str, _digest: str, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest / archive_name, "w:gz") as tar:
            tar.add(payload_root, arcname="nova")

    return puller


def test_sync_pulls_and_extracts(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setattr(corpus, "NORMA_DATA_HOME", home)

    payload = tmp_path / "payload"
    (payload / "sector_modules").mkdir(parents=True)
    (payload / "sector_modules" / "unit.xlsx").write_bytes(b"content")

    kind = _fake_kind(archive="fake.tar.gz", verify=False)
    resolved = corpus.sync(kind, puller=_archive_puller(payload, kind.archive))
    assert resolved == home / kind.tag / "nova" / "sector_modules"
    assert (resolved / "unit.xlsx").read_bytes() == b"content"


def test_sync_raises_when_archive_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(corpus, "NORMA_DATA_HOME", tmp_path / "home")

    def _noop_puller(_repo, _digest, dest):
        dest.mkdir(parents=True, exist_ok=True)  # pulls nothing

    with pytest.raises(corpus.CorpusError, match="expected archive"):
        corpus.sync(_fake_kind(), puller=_noop_puller)


def _write_manifest(path: Path, sector_rel: str, content: bytes) -> None:
    doc = {
        "schema": "nova.assembly.provenance/manifest/1",
        "corpus": "test",
        "files": [{"path": sector_rel, "digest": digest.digest_bytes(content)}],
    }
    yamlio.dump_yaml(doc, path)


def test_sync_verifies_against_manifest(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setattr(corpus, "NORMA_DATA_HOME", home)

    payload = tmp_path / "payload"
    (payload / "sector_modules").mkdir(parents=True)
    (payload / "sector_modules" / "unit.xlsx").write_bytes(b"content")

    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, "sector_modules/unit.xlsx", b"content")

    kind = _fake_kind(archive="fake.tar.gz", verify=True)
    resolved = corpus.sync(
        kind,
        puller=_archive_puller(payload, kind.archive),
        manifest_path=manifest_path,
    )
    assert (resolved / "unit.xlsx").exists()


def test_sync_detects_corruption(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setattr(corpus, "NORMA_DATA_HOME", home)

    payload = tmp_path / "payload"
    (payload / "sector_modules").mkdir(parents=True)
    (payload / "sector_modules" / "unit.xlsx").write_bytes(b"content")

    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, "sector_modules/unit.xlsx", b"different-bytes")

    kind = _fake_kind(archive="fake.tar.gz", verify=True)
    with pytest.raises(corpus.CorpusError, match="verification failed"):
        corpus.sync(
            kind,
            puller=_archive_puller(payload, kind.archive),
            manifest_path=manifest_path,
        )


@pytest.mark.skipif(
    not os.environ.get("NOVA_CORPUS_INTEGRATION"),
    reason="opt-in GHCR pull; set NOVA_CORPUS_INTEGRATION=1 with oras logged in",
)
def test_real_ghcr_pull(monkeypatch, tmp_path):
    """Pull the real appdata corpus from GHCR into a scratch home and verify."""
    monkeypatch.setattr(corpus, "NORMA_DATA_HOME", tmp_path / "home")
    resolved = corpus.sync("appdata")
    assert resolved.is_dir()
    assert (resolved / "ILIS_nominal.pickle").exists()
