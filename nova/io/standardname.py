"""Resolve store keys against the managed standard-name vocabulary.

The zarr signal store keys every signal by an IMAS standard name. Those names
are owned by the ``imas-standard-names`` machinery (grammar + name API, pint
units) and the ``imas-standard-names-catalog`` data package (pre-built SQLite
catalog); nova owns no standard-name schema. This module is the consumption
seam: it wraps those two packages behind a small resolver so the transcoders
never touch a nova-side vocabulary file.

Resolution proceeds from the most authoritative source to the least:

#. the packaged catalog (``catalog.db`` when the data package ships it, else
   the machinery's bundled reference catalog) supplies the unit, kind, and
   governance status for a known name;
#. failing a catalog hit, the standard-name grammar is asked to parse the
   token -- a well-formed name that is simply absent from the installed
   catalog resolves as *provisional* with no authoritative unit;
#. a token the grammar rejects does not resolve.

A provisional resolution is the signal that a name is a candidate contribution
to the catalog fork, not that nova should mint schema locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property
import os

from nova.utilities.importmanager import check_import

try:  # the standard-name machinery is an optional (io extra) dependency
    from imas_standard_names import parse as _parse_standard_name
    from imas_standard_names.grammar.parser import ParseError
    from imas_standard_names.repository import StandardNameCatalog

    HAS_STANDARD_NAMES = True
except ImportError:  # pragma: no cover - exercised only without the io extra
    HAS_STANDARD_NAMES = False


class NameSource(str, Enum):
    """Provenance of a resolved standard name, most authoritative first."""

    CATALOG = "catalog"
    GRAMMAR = "grammar"


@dataclass(frozen=True)
class Resolution:
    """A resolved standard name and the metadata the store carries for it.

    Attributes
    ----------
    name:
        The standard name, unchanged from the query.
    unit:
        UDUNITS unit string from the catalog entry, or ``None`` when the name
        resolved only through the grammar (no authoritative unit).
    kind:
        ``"scalar"`` or ``"vector"`` from the catalog, else ``None``.
    status:
        Catalog governance status (``draft``/``active``/...), or
        ``"provisional"`` for a grammar-only resolution.
    source:
        Which layer resolved the name.
    """

    name: str
    unit: str | None
    kind: str | None
    status: str
    source: NameSource

    @property
    def provisional(self) -> bool:
        """Return True when the name is not backed by a catalog entry."""
        return self.source is not NameSource.CATALOG


class UnknownStandardName(KeyError):
    """Raised when a token resolves through neither catalog nor grammar."""


@dataclass
class StandardNameResolver:
    """Resolve signal keys against ISN/ISNC without a nova-side vocabulary.

    Parameters
    ----------
    catalog_db:
        Explicit path to a ``catalog.db`` SQLite catalog. When ``None`` the
        resolver uses the data package's bundled database if it ships one and
        otherwise falls back to the machinery's packaged reference catalog.
    """

    catalog_db: str | os.PathLike | None = None

    def __post_init__(self):
        """Fail fast with the install hint when the io extra is absent."""
        with check_import("imas-standard-names", "imas-standard-names-catalog"):
            if not HAS_STANDARD_NAMES:
                raise ImportError("imas-standard-names is not installed")

    @staticmethod
    def _packaged_catalog_db() -> str | None:
        """Return the data package's catalog.db path when it exists on disk.

        A git-tag install of the catalog data package ships no built
        ``catalog.db`` (it is a CI/wheel artifact); in that case the resolver
        falls back to the machinery's bundled reference catalog.
        """
        try:
            from imas_standard_names_catalog import get_catalog_db
        except ImportError:  # pragma: no cover - only without the data package
            return None
        path = str(get_catalog_db())
        return path if os.path.exists(path) else None

    @cached_property
    def catalog(self) -> StandardNameCatalog:
        """Return the standard-name catalog reader."""
        database = self.catalog_db or self._packaged_catalog_db()
        if database is not None:
            return StandardNameCatalog(str(database))
        return StandardNameCatalog()

    def resolve(self, name: str) -> Resolution:
        """Resolve a standard name, raising :class:`UnknownStandardName`.

        Notes
        -----
        Catalog hits carry the authoritative unit and kind. A grammar-valid
        name absent from the catalog resolves as provisional -- record such
        names as candidate catalog-fork contributions rather than inventing
        schema for them.
        """
        entry = self.catalog.get(name)
        if entry is not None:
            # Metadata-kind entries (loci, groupings) carry no unit; only
            # quantity entries do. Read defensively so both resolve.
            unit = getattr(entry, "unit", None)
            kind = getattr(entry, "kind", None)
            return Resolution(
                name=name,
                unit=None if unit is None else str(unit),
                kind=None if kind is None else str(kind),
                status=str(getattr(entry, "status", "active")),
                source=NameSource.CATALOG,
            )
        try:
            result = _parse_standard_name(name)
        except ParseError as error:
            raise UnknownStandardName(
                f"{name!r} is neither a catalog entry nor a grammar-valid "
                f"standard name: {error}"
            ) from error
        if getattr(result, "diagnostics", None):
            raise UnknownStandardName(
                f"{name!r} is not a valid standard name: {result.diagnostics}"
            )
        return Resolution(
            name=name,
            unit=None,
            kind=None,
            status="provisional",
            source=NameSource.GRAMMAR,
        )

    def known(self, name: str) -> bool:
        """Return True when ``name`` is a catalog entry."""
        return self.catalog.get(name) is not None
