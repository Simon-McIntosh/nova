"""Structured metadata parsed from corpus filenames.

The measured coil-alignment corpus names workbooks with a convention that
encodes the sector, a human-readable description, the controlling IDM document
number, and a two-part version, for example::

    Sector_Module_#1_CCL_as-built_data_8LMK6A_v8_1.xlsx

A leading underscore marks a private in-work revision
(``_Sector_Module_#6_..._v9_0.xlsx``). Files that do not follow the convention
(nominal pickles, ad-hoc spreadsheets, NetCDF exports, text notes, archives)
are not errors: they are classified ``"opaque"`` and carry only the fields that
can be read from the name itself.
"""

from dataclasses import dataclass
from pathlib import Path
import re

from packaging.version import Version

KIND_SECTOR_MODULE = "sector_module"
KIND_OPAQUE = "opaque"

# Sector-module stem: the sector number follows "#", a non-greedy description
# absorbs the middle tokens, and the trailing "<idm>_v<major>_<minor>" anchors
# the IDM document and version. The IDM token is upper-case alphanumeric and
# must contain at least one letter so it is never confused with the version.
_SECTOR_MODULE = re.compile(
    r"^Sector_Module_#(?P<sector>\d+)_"
    r"(?P<description>.+)_"
    r"(?P<idm>(?=[0-9A-Z]*[A-Z])[0-9A-Z]+)_"
    r"v(?P<major>\d+)_(?P<minor>\d+)$"
)


@dataclass(frozen=True)
class SourceMeta:
    """Metadata read from a single corpus filename.

    Parameters
    ----------
    stem : str
        Filename without its suffix and without any private-revision prefix.
    suffix : str
        File extension including the leading dot, for example ``".xlsx"``.
    kind : str
        Either ``"sector_module"`` for conforming workbooks or ``"opaque"``.
    private : bool
        ``True`` when the original name began with an underscore, marking a
        private in-work revision.
    sector : int or None
        Sector number for conforming names, otherwise ``None``.
    idm_doc : str or None
        Controlling IDM document number for conforming names.
    version : str or None
        Dotted ``"major.minor"`` version string for conforming names.
    description : str or None
        Human-readable description tokens for conforming names.
    """

    stem: str
    suffix: str
    kind: str
    private: bool
    sector: int | None = None
    idm_doc: str | None = None
    version: str | None = None
    description: str | None = None

    @property
    def parsed_version(self) -> Version | None:
        """Return the version as a comparable :class:`packaging.version.Version`.

        Returns
        -------
        packaging.version.Version or None
            Parsed version, or ``None`` for opaque names.
        """
        return Version(self.version) if self.version is not None else None

    def to_dict(self) -> dict:
        """Return a YAML-serializable mapping of the metadata.

        Returns
        -------
        dict
            Mapping with only scalar values, suitable for canonical YAML.
        """
        return {
            "stem": self.stem,
            "suffix": self.suffix,
            "kind": self.kind,
            "private": self.private,
            "sector": self.sector,
            "idm_doc": self.idm_doc,
            "version": self.version,
            "description": self.description,
        }


def parse_filename(name) -> SourceMeta:
    """Parse a corpus filename into structured metadata.

    Parameters
    ----------
    name : str or os.PathLike
        Filename or path. Only the basename is inspected.

    Returns
    -------
    SourceMeta
        Structured metadata. Non-conforming names return a ``kind="opaque"``
        record rather than raising.
    """
    basename = Path(name).name
    private = basename.startswith("_")
    core = basename[1:] if private else basename

    suffix = Path(core).suffix
    stem = Path(core).stem

    match = _SECTOR_MODULE.match(stem)
    if match is None:
        return SourceMeta(
            stem=stem,
            suffix=suffix,
            kind=KIND_OPAQUE,
            private=private,
        )

    return SourceMeta(
        stem=stem,
        suffix=suffix,
        kind=KIND_SECTOR_MODULE,
        private=private,
        sector=int(match.group("sector")),
        idm_doc=match.group("idm"),
        version=f"{int(match.group('major'))}.{int(match.group('minor'))}",
        description=match.group("description"),
    )
