"""Resolve and materialize the coil-metrology corpus on any machine.

The measured sector-module corpus is too large and too access-restricted to
commit to git, so it lives as content-addressed artifacts in a container
registry (GHCR, ``ghcr.io/simon-mcintosh/norma-corpus``). This module lets any
machine find a local copy of the corpus -- and, when none is present, pull one
from the registry *by digest*, verify it against the committed preservation
manifest, and extract it into the per-user data directory.

Two corpus kinds are published:

``appdata``
    The raw AppData snapshot: the sector-module ``*.xlsx`` workbooks with their
    pickle caches and the preserved ``ILIS_nominal.pickle`` reference cloud.
``canonical``
    The transcoded canonical units (long-format ``*.csv`` + provenance
    sidecars) rebuilt from the workbooks.

``resolve_corpus(kind)`` returns the local ``sector_modules`` directory for a
kind, or ``None`` when no copy is staged. ``sync(kind)`` guarantees a verified
local copy, pulling from the registry only when one is absent -- it is
idempotent and never re-downloads a tree that already resolves.

Pulling requires the ``oras`` CLI on ``PATH`` (or at ``~/bin/oras``) already
authenticated to the registry (``oras login ghcr.io``). No Python dependency on
a registry client is introduced; the pull is a subprocess call.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from nova.assembly.provenance import manifest as manifest_module

# Per-user home for staged corpus trees; each kind lives under its tag there.
NORMA_DATA_HOME = Path.home() / ".local" / "share" / "norma"

# The historical WSL AppData location the facility laptop wrote to; kept as a
# last-resort resolver fallback so a machine with the real mount still works.
LEGACY_APPDATA = Path("/mnt/c/Users/mcintos/AppData/Local/nova/sector_modules")

# Registry the corpus artifacts are published to.
CORPUS_REPO = "ghcr.io/simon-mcintosh/norma-corpus"

# The committed preservation manifest content-addresses the appdata tree.
CORPUS_MANIFEST_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "Assembly" / "corpus_manifest.yaml"
)


@dataclass(frozen=True)
class CorpusKind:
    """A published corpus artifact and the archive it unpacks to.

    Parameters
    ----------
    name : str
        Short kind key (``"appdata"`` / ``"canonical"``).
    tag : str
        Registry tag; also the directory name the tree is staged under.
    digest : str
        Manifest digest of the artifact, ``"sha256:<hex>"``. The pull pins this
        rather than the mutable tag so the bytes are reproducible.
    archive : str
        Name of the ``*.tar.gz`` blob inside the artifact to extract.
    verify : bool
        Whether the extracted tree is checked against the corpus manifest. Only
        the appdata tree is described by that manifest.
    """

    name: str
    tag: str
    digest: str
    archive: str
    verify: bool


KINDS: dict[str, CorpusKind] = {
    "appdata": CorpusKind(
        name="appdata",
        tag="appdata-snapshot-2026-07-23",
        digest="sha256:6507c3768ea34e22971cc5cab667ecbdfc16ed67deffbc3cd8ce9d20256ec8cd",
        archive="corpus-appdata-2026-07-23.tar.gz",
        verify=True,
    ),
    "canonical": CorpusKind(
        name="canonical",
        tag="canonical-2026-07-24",
        digest="sha256:9870e2616e9457eae7eb847d80ef06d9c555efbaeefb7905ee2234097ac9df60",
        archive="canonical-2026-07-24.tar.gz",
        verify=False,
    ),
}


class CorpusError(RuntimeError):
    """Raised when a corpus cannot be pulled, extracted, or verified."""


def _kind(kind: str | CorpusKind) -> CorpusKind:
    if isinstance(kind, CorpusKind):
        return kind
    try:
        return KINDS[kind]
    except KeyError as error:
        raise KeyError(
            f"unknown corpus kind {kind!r}; known kinds: {sorted(KINDS)}"
        ) from error


def _find_sector_modules(base: Path) -> Path | None:
    """Return the ``sector_modules`` directory at or just below ``base``.

    The two artifacts nest their payload differently (the appdata tarball
    unpacks to ``nova/sector_modules``, the canonical tarball to
    ``<tag>/sector_modules``), so a shallow search rather than a fixed depth
    keeps the resolver robust to either layout.
    """
    if not base.is_dir():
        return None
    direct = base / "sector_modules"
    if direct.is_dir():
        return direct
    for child in sorted(base.iterdir()):
        if child.is_dir():
            candidate = child / "sector_modules"
            if candidate.is_dir():
                return candidate
    return None


def _search_bases() -> list[Path]:
    """Return the corpus-home roots to search, honouring ``$NOVA_CORPUS_ROOT``."""
    bases: list[Path] = []
    env = os.environ.get("NOVA_CORPUS_ROOT")
    if env:
        bases.append(Path(env))
    bases.append(NORMA_DATA_HOME)
    return bases


def resolve_corpus(kind: str | CorpusKind = "appdata") -> Path | None:
    """Return the local ``sector_modules`` directory for a corpus kind.

    Parameters
    ----------
    kind : str or CorpusKind, optional
        Which corpus to locate. The default is the raw ``appdata`` snapshot.

    Returns
    -------
    pathlib.Path or None
        The directory holding the sector-module artifacts, or ``None`` when no
        copy is staged on this machine. The search order is
        ``$NOVA_CORPUS_ROOT``, then :data:`NORMA_DATA_HOME`, then (for the
        appdata kind only) the legacy WSL AppData mount.
    """
    corpus = _kind(kind)
    for base in _search_bases():
        for candidate_root in (base / corpus.tag, base):
            found = _find_sector_modules(candidate_root)
            if found is not None:
                return found
    if corpus.name == "appdata" and LEGACY_APPDATA.is_dir():
        return LEGACY_APPDATA
    return None


def _oras_binary() -> str:
    """Return the path to the ``oras`` CLI, or raise if it is not installed."""
    found = shutil.which("oras")
    if found:
        return found
    home_oras = Path.home() / "bin" / "oras"
    if home_oras.exists():
        return str(home_oras)
    raise CorpusError(
        "oras CLI not found; install it and authenticate with "
        "`oras login ghcr.io` before syncing the corpus"
    )


def _oras_pull(repo: str, digest: str, dest: Path) -> None:
    """Pull an artifact by digest into ``dest`` with the ``oras`` CLI."""
    dest.mkdir(parents=True, exist_ok=True)
    reference = f"{repo}@{digest}"
    try:
        subprocess.run(
            [_oras_binary(), "pull", reference, "-o", str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise CorpusError(
            f"oras pull of {reference} failed (rc={error.returncode}): "
            f"{error.stderr.strip() or error.stdout.strip()}"
        ) from error


def _extract(archive: Path, dest: Path) -> None:
    """Extract a ``*.tar.gz`` into ``dest`` with the safe data filter."""
    with tarfile.open(archive, "r:gz") as tar:
        # ``filter="data"`` (Python 3.12+) blocks path traversal / device
        # entries; the corpus archives contain only regular files.
        try:
            tar.extractall(dest, filter="data")
        except TypeError:  # pragma: no cover - older Python without the filter
            tar.extractall(dest)


def _verify_extracted(sector_modules: Path, manifest_path: Path) -> None:
    """Verify the extracted tree against the corpus manifest.

    The manifest paths are relative to the tree root that *contains*
    ``sector_modules``; a digest mismatch on any file present under that root
    is a corruption and raises. Files listed in the manifest but absent from a
    partial tree are not an error -- only files that are present and differ.
    """
    tree_root = sector_modules.parent
    document = manifest_module.load_manifest(manifest_path)
    report = manifest_module.verify(tree_root, document)
    if report["changed"]:
        raise CorpusError(
            "corpus verification failed: digest mismatch for "
            + ", ".join(report["changed"][:5])
            + (" ..." if len(report["changed"]) > 5 else "")
        )


def sync(
    kind: str | CorpusKind = "appdata",
    *,
    puller=_oras_pull,
    manifest_path: Path | None = None,
) -> Path:
    """Guarantee a verified local copy of a corpus and return its path.

    When a copy already resolves, it is returned untouched -- no network
    access. Otherwise the artifact is pulled from the registry by pinned
    digest, its archive extracted into :data:`NORMA_DATA_HOME`, verified
    against the corpus manifest where entries apply, and the resulting
    ``sector_modules`` directory returned.

    Parameters
    ----------
    kind : str or CorpusKind, optional
        Which corpus to materialize.
    puller : callable, optional
        ``puller(repo, digest, dest)`` performing the fetch. Defaults to the
        ``oras`` subprocess; overridable so unit tests need no network.
    manifest_path : pathlib.Path, optional
        Corpus manifest to verify against. Defaults to the committed manifest.

    Returns
    -------
    pathlib.Path
        The verified local ``sector_modules`` directory.
    """
    corpus = _kind(kind)
    existing = resolve_corpus(corpus)
    if existing is not None:
        return existing

    dest = NORMA_DATA_HOME / corpus.tag
    puller(CORPUS_REPO, corpus.digest, dest)

    archive = dest / corpus.archive
    if not archive.exists():
        raise CorpusError(
            f"expected archive {corpus.archive} not found in {dest} after pull"
        )
    _extract(archive, dest)

    resolved = resolve_corpus(corpus)
    if resolved is None:
        raise CorpusError(
            f"corpus {corpus.name} extracted to {dest} but no sector_modules "
            "directory was found"
        )
    if corpus.verify:
        _verify_extracted(resolved, manifest_path or CORPUS_MANIFEST_PATH)
    return resolved


def preserved_pickle(name: str, kind: str | CorpusKind = "appdata") -> Path | None:
    """Return the path to a preserved sector-module artifact, or ``None``.

    Parameters
    ----------
    name : str
        Artifact file name, for example ``"ILIS_nominal.pickle"``.
    kind : str or CorpusKind, optional
        Which corpus to look in.

    Returns
    -------
    pathlib.Path or None
        The artifact path if the corpus resolves and the file exists.
    """
    sector_modules = resolve_corpus(kind)
    if sector_modules is None:
        return None
    candidate = sector_modules / name
    return candidate if candidate.exists() else None
