"""Manage access to IMAS database."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from nova.database.filepath import FilePath
from imas import imasdef, DBEntry

# pylint: disable=too-many-ancestors


@dataclass
class IDS:
    """Structure IDS input as leading arguments."""

    shot: int
    run: int
    ids_name: str = None


@dataclass
class IMASdb:
    """Methods to access IMAS database."""

    user: str = 'public'
    tokamak: str = 'iter_md'
    backend: int = imasdef.MDSPLUS_BACKEND

    @contextmanager
    def _database(self, shot: int, run: int, ids_name: str):
        """Database context manager."""
        database = DBEntry(self.backend, self.tokamak, shot, run,
                           user_name=self.user)
        database.open()
        yield database.get(ids_name)
        database.close()

    def ids(self, shot: int, run: int, ids_name: str):
        """Return filled ids from dataabase."""
        with self._database(shot, run, ids_name) as ids_data:
            return ids_data


@dataclass
class Database(FilePath, IMASdb, IDS):
    """Provide access to imasdb data."""

    datapath: str = 'data/Imas'
    ids_data: Any = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Set filepath."""
        self.set_path(self.datapath)

    @property
    def filename(self):
        """Return filename."""
        return f'{self.tokamak}_{self.shot}{self.run:04d}'

    def load_ids_data(self):
        """Return ids_data."""
        self.ids_data = self.ids(self.shot, self.run, self.ids_name)
        return self.ids_data

    @property
    def ids_attrs(self):
        """Return dict of ids attributes."""
        return dict(tokamak=self.tokamak, shot=self.shot, run=self.run,
                    ids=self.ids_name)
