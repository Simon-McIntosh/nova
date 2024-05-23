"""Manage access to IMAS data entry."""

from dataclasses import dataclass
import logging
import os
from packaging.version import Version
from typing import ClassVar

from nova.utilities.importmanager import check_import

with check_import("imaspy"):
    import imaspy as imas

logging.basicConfig(level=logging.WARNING)


@dataclass
class Dataset:
    """Locate an IDS dataset on a local or remote machine using a pulse-run layout.

    Parameters
    ----------
    pulse: int, optional (required when ids not set)
        Pulse number. The default is 0.
    run: int, optional (required when ids not set)
        Run number. The default is 0.
    machine: str, optional (required when ids not set)
        Machine name. The default is iter.
    user: str, optional (required when ids not set)
        User name. The default is public.
    backend: str, optional (required when ids not set)
        Access layer backend. The default is hdf5.

    Attributes
    ----------
    uri : str
        IDS unified resorce identifier.

    home : os.Path, read-only
        Path to IMAS database home.

    ids_path : os.path, read-only
        Path to IDS database entry.

    dd_version : Version
        Data Dictionary version.
    """

    pulse: int = 0
    run: int = 0
    machine: str = "iter"
    user: str = "public"
    backend: str = "hdf5"
    dd_version: Version | str | None = None

    dir_attrs: ClassVar[list[str]] = ["pulse", "run", "machine", "user", "backend"]

    def __post_init__(self):
        """Set dd_version."""
        self.version = self.dd_version

    @property
    def version(self):
        """Return a Version instance of dd_version."""
        return self.dd_version

    @version.setter
    def version(self, dd_version: Version | str | None):
        if isinstance(dd_version, Version):
            dd_version = str(dd_version)
        self.dd_version = Version(imas.IDSFactory(dd_version).version)

    @property
    def attrs(self) -> list[str]:
        """Return data dir attributes. Subclass to append."""
        return self.dir_attrs

    @classmethod
    def default_ids_attrs(cls) -> dict:
        """Return dict of default ids attributes."""
        return {attr: getattr(cls, attr) for attr in cls.dir_attrs}

    @property
    def ids_attrs(self):
        """Manage dict of ids attributes."""
        return {attr: getattr(self, attr) for attr in self.attrs}

    @ids_attrs.setter
    def ids_attrs(self, attrs: dict):
        for attr, value in attrs.items():
            if attr not in self.attrs:
                raise AttributeError(f"attr {attr} not in self.attrs {self.attrs}")
            setattr(self, attr, value)

    @property
    def uri(self):
        """Return IDS URI."""
        return (
            f"imas:{self.backend}?user={self.user};"
            f"pulse={self.pulse};run={self.run};"
            f"database={self.machine};version={self.dd_version.major};"
        )

    @property
    def home(self):
        """Return database root."""
        if self.user == "public":
            return os.path.join(os.environ["IMAS_HOME"], "shared")
        return os.path.join(os.path.expanduser(f"~{self.user}"), "public")

    @property
    def database_path(self):
        """Return top level of database path."""
        return os.path.join(
            self.home, "imasdb", self.machine, str(self.dd_version.major)
        )

    @property
    def ids_path(self) -> str:
        """Return path to database entry."""
        match self.backend:
            case str(backend) if backend == "hdf5":
                return os.path.join(self.database_path, str(self.pulse), str(self.run))
            case _:
                raise NotImplementedError(
                    f"not implemented for {self.backend}" " backend"
                )
