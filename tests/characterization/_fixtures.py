"""Materialize sector-module workbooks from the in-repo canonical corpus.

The sector/pit fitting code reads Excel workbooks from a facility data
directory (``SectorFile.datadir``) that is absent here. This module rebuilds
those workbooks on demand from the git-safe canonical units committed under
``data/Assembly/sector_modules`` (long-format CSV + provenance sidecar), writes
them into a private cache, and redirects the reader at that cache -- without
editing the assembly source, which the harness only observes.

The redirection patches the *default* ``datadir`` (and the pickle-cache
``dirname``) baked into the generated ``__init__`` of the two reader dataclasses
so that the many internal ``SectorData(...)`` / ``SectorFile(...)`` calls, which
pass neither, pick up the cache. The pickle cache is keyed by a fingerprint of
the seeded units so it self-invalidates when the corpus changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ._environment import repo_root

# Where the committed canonical units live, and where rebuilt workbooks are cached.
CANONICAL_DIR = repo_root() / "data" / "Assembly" / "sector_modules"
CACHE_ROOT = Path.home() / ".cache" / "nova-characterization"
WORKBOOK_CACHE = CACHE_ROOT / "sector_modules"

# Index into the defaults tuple of the reader dataclasses' generated __init__.
# Both SectorFile and SectorData carry ``datadir`` at the same position; only
# SectorData carries the pickle-cache ``dirname``.
_DATADIR_INDEX = 3
_DIRNAME_INDEX = 5

_seeded = False


def canonical_units() -> dict[str, str]:
    """Return ``{stem: "sha256:<hex>"}`` for every committed canonical unit.

    The digest is the content address of the canonical CSV (the unit's own
    :attr:`CanonicalUnit.digest`), the value recorded as a provenance input.
    """
    from nova.assembly.canonical import read_unit

    units: dict[str, str] = {}
    for csv_path in sorted(CANONICAL_DIR.glob("*.csv")):
        units[csv_path.stem] = read_unit(csv_path).digest
    return units


def _corpus_fingerprint(units: dict[str, str]) -> str:
    """Return a short stable hash over the unit digests for cache keying."""
    joined = "\n".join(f"{stem}\t{digest}" for stem, digest in sorted(units.items()))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


def sector_modules_available() -> bool:
    """Return whether the sector-module fixtures can be materialized here.

    True when the in-repo canonical units are present and the workbook
    transcoder imports -- i.e. workbooks can be rebuilt without the facility
    share. Rebuilt/cached ``*.xlsx`` also count.
    """
    if WORKBOOK_CACHE.is_dir() and any(WORKBOOK_CACHE.glob("*.xlsx")):
        return True
    if not (CANONICAL_DIR.is_dir() and any(CANONICAL_DIR.glob("*.csv"))):
        return False
    try:
        import nova.assembly.canonical  # noqa: F401
    except Exception:  # noqa: BLE001 - any import failure means no transcoder
        return False
    return True


def seed_workbooks(cache_dir: Path = WORKBOOK_CACHE) -> dict[str, str]:
    """Rebuild every canonical unit into an ``.xlsx`` under ``cache_dir``.

    Returns the ``{stem: digest}`` map of the units that were materialized.
    A workbook is (re)written only when absent or stale relative to its unit
    digest, recorded in a sidecar stamp file.
    """
    from nova.assembly.canonical import egress_workbook, read_unit

    cache_dir.mkdir(parents=True, exist_ok=True)
    units: dict[str, str] = {}
    for csv_path in sorted(CANONICAL_DIR.glob("*.csv")):
        unit = read_unit(csv_path)
        units[unit.stem] = unit.digest
        xlsx_path = cache_dir / f"{unit.stem}.xlsx"
        stamp_path = cache_dir / f"{unit.stem}.digest"
        current = stamp_path.read_text().strip() if stamp_path.exists() else ""
        if xlsx_path.exists() and current == unit.digest:
            continue
        egress_workbook(unit).save(xlsx_path)
        stamp_path.write_text(unit.digest + "\n")
    return units


def _patch_reader_defaults(datadir: str, pickle_dir: Path) -> None:
    """Redirect the reader dataclasses' default datadir and pickle cache."""
    from nova.assembly.sectordata import SectorData
    from nova.assembly.sectorfile import SectorFile

    for cls in (SectorData, SectorFile):
        defaults = list(cls.__init__.__defaults__)
        defaults[_DATADIR_INDEX] = datadir
        cls.__init__.__defaults__ = tuple(defaults)

    defaults = list(SectorData.__init__.__defaults__)
    defaults[_DIRNAME_INDEX] = pickle_dir
    SectorData.__init__.__defaults__ = tuple(defaults)


def ensure_sector_cache() -> dict[str, str]:
    """Idempotently seed the workbook cache and redirect the readers at it.

    Safe to call from any entry-point callable or generator: the first call
    materializes the workbooks and patches the reader defaults; later calls
    return the cached unit map. The pickle cache is namespaced by a corpus
    fingerprint so a corpus change forces a clean rebuild.
    """
    global _seeded
    units = seed_workbooks()
    if not _seeded:
        pickle_dir = CACHE_ROOT / "pickles" / _corpus_fingerprint(units)
        pickle_dir.mkdir(parents=True, exist_ok=True)
        _patch_reader_defaults(str(WORKBOOK_CACHE), pickle_dir)
        _seeded = True
    return units
