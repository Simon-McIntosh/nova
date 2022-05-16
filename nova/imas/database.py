"""Manage access to IMAS database."""
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from nova.database.filepath import FilePath

# pylint: disable=too-many-ancestors


@dataclass
class IDS:
    """Structure IDS input as leading arguments."""

    pulse: int
    run: int
    ids_name: str = ''


@dataclass
class IMASdb:
    """Methods to access IMAS database."""

    user: str = 'public'
    machine: str = 'iter_md'
    backend: int = 13

    @contextmanager
    def _database(self, pulse: int, run: int, ids_name: str):
        """Database context manager."""
        from imas import DBEntry
        database = DBEntry(self.backend, self.machine, pulse, run,
                           user_name=self.user)
        database.open()
        yield database.get(ids_name)
        database.close()

    def ids(self, pulse: int, run: int, ids_name: str):
        """Return filled ids from dataabase."""
        with self._database(pulse, run, ids_name) as ids_data:
            return ids_data


@dataclass
class Database(FilePath, IMASdb, IDS):
    """Provide access to imasdb data."""

    datapath: str = 'data/Imas'
    ids_data: Any = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Set filepath."""
        self.filename = f'{self.machine}_{self.pulse}{self.run:04d}'
        self.group = self.ids_name
        self.set_path(self.datapath)

    def load_ids_data(self):
        """Return ids_data."""
        self.ids_data = self.ids(self.pulse, self.run, self.ids_name)
        return self.ids_data

    @property
    def ids_attrs(self):
        """Return dict of ids attributes."""
        return dict(machine=self.machine, pulse=self.pulse, run=self.run,
                    ids_name=self.ids_name)
