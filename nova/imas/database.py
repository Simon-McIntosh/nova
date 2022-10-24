"""Manage access to IMAS database."""
from contextlib import contextmanager
from dataclasses import dataclass, field

try:
    from imas import DBEntry
    import_imas = True
except ImportError:
    import_imas = False

from nova.database.filepath import FilePath

# pylint: disable=too-many-ancestors


@dataclass
class IDS:
    """Structure IDS input as leading arguments."""

    pulse: int | None = None
    run: int | None = None
    ids_name: str | None = None
    ids: object | None = field(repr=False, default=None)


@dataclass
class IMASal:
    """Define IMAS access layer attributes."""

    user: str = 'public'
    machine: str = 'iter_md'
    backend: int = 13


@dataclass
class IMASdb(IMASal, IDS):
    """Methods to access IMAS database."""

    @contextmanager
    def get_ids(self):
        """Yield database with context manager."""
        if not import_imas:
            raise ImportError('imas module not found')
        db_entry = DBEntry(self.backend, self.machine,
                           self.pulse, self.run, user_name=self.user)
        try:
            db_entry.open()
        except TypeError as error:
            raise TypeError(f'malformed input to DBEntry\n{error}\n'
                            f'backend: {self.backend}\n'
                            f'machine {self.machine}\n'
                            f'pulse {self.pulse}\n'
                            f'run {self.run}\n'
                            f'user {self.user}\n')
        try:
            ids_name, ids_path = self.ids_name.split('.', 1)
            yield db_entry.partial_get(ids_name, ids_path)
        except ValueError:
            yield db_entry.get(self.ids_name)
        db_entry.close()

    def set_ids(self):
        """Set ids from pulse/run."""
        if self.ids is None:
            with self.get_ids() as ids:
                self.ids = ids
        self.ids_name = self.ids.__name__


@dataclass
class Database(FilePath, IMASdb):
    """Provide access to imasdb data."""

    directory: str = 'user_data'
    datapath: str = 'imas'

    def __post_init__(self):
        """Set ids and filepath."""
        self.set_ids()
        try:
            self.filename = f'{self.machine}_{self.pulse}{self.run:04d}'
        except TypeError:
            self.filename = None
        self.group = self.ids_name
        self.set_path(self.datapath)

    @property
    def ids_attrs(self):
        """Return dict of ids attributes."""
        return dict(machine=self.machine, pulse=self.pulse, run=self.run,
                    ids_name=self.ids_name)


if __name__ == '__main__':

    database = Database(130506, 403, 'equilibrium', machine='iter')

    db2 = Database(ids=database.ids)
