"""Manage access to IMAS database."""
from contextlib import contextmanager
from dataclasses import dataclass

from imas import imasdef, DBEntry

# pylint: disable=too-many-ancestors


@dataclass
class IDS:
    """Structure IDS input as leading arguments."""

    shot: int
    run: int
    ids_name: str


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
class Database(IMASdb, IDS):
    """Provide access to imasdb data."""

    def load_ids_data(self):
        """Return ids_data."""
        return self.ids(self.shot, self.run, self.ids_name)
