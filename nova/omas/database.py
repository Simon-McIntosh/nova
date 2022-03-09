"""Manage access to IMAS database."""
from dataclasses import dataclass, field

import omas

# pylint: disable=too-many-ancestors


@dataclass
class ODS:
    """Structure ODS input as leading arguments."""

    pulse: int
    run: int
    name: str = None


@dataclass
class OMASdb:
    """Methods to access IMAS database."""

    user: str = 'public'
    machine: str = 'iter_md'

    def ods(self, pulse: int, run: int, paths: list[str], verbose=True):
        """Return filled ids from dataabase."""
        return omas.load_omas_imas(user=self.user, machine=self.machine,
                                   pulse=pulse, run=self.run, paths=paths,
                                   verbose=verbose)


@dataclass
class Database(OMASdb, ODS):
    """Provide access to imasdb data."""

    datapath: str = 'data/imasdb'
    ods_data: omas.omas_core.ODS = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Set filepath."""
        super().__post_init__()
        self.set_path(self.datapath)

    @property
    def filename(self):
        """Return filename."""
        return f'{self.machine}_{self.pulse}{self.run:04d}'

    def load_ods_data(self):
        """Return ids_data."""
        self.ods_data = self.ods(self.pulse, self.run, [self.name])[self.name]
        return self.ods_data

    @property
    def ods_attrs(self):
        """Return dict of ids attributes."""
        return dict(machine=self.machine, shot=self.shot, run=self.run,
                    paths=self.paths)
