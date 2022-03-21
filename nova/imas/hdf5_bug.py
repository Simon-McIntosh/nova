"""Demonstrate potential bug associated with idscp / HDF5 database."""
from contextlib import contextmanager
from dataclasses import dataclass

from imas import DBEntry, imasdef


@dataclass
class DataBase:
    """Manage acess to imas database."""

    shot: int
    run: int
    ids_name: str = 'equilibrium'
    user: str = 'public'
    tokamak: str = 'iter'
    backend: int = imasdef.MDSPLUS_BACKEND

    def __post_init__(self):
        """Load equilibrium ids and print index zero length arrays."""
        ids_data = self.ids(self.shot, self.run, self.ids_name)
        self.ids_data = ids_data
        # collect itime indexes for all zero-length profile_2d arrays.
        profiles_2d_array_empty = [
            itime for itime in range(len(ids_data.time))
            if len(ids_data.time_slice[itime].profiles_2d.array) == 0]
        print(f'backend: {self.backend}',
              f'profiles_2d empty: {profiles_2d_array_empty}')

    def ids(self, shot: int, run: int, ids_name: str):
        """Return filled ids from dataabase."""
        with self._database(shot, run, ids_name) as ids_data:
            return ids_data

    @contextmanager
    def _database(self, shot: int, run: int, ids_name: str):
        database = DBEntry(self.backend, self.tokamak, shot, run,
                           user_name=self.user)
        database.open()
        yield database.get(ids_name)
        database.close()


if __name__ == '__main__':

    db = DataBase(135011, 7, backend=12)
    db = DataBase(135011, 7, backend=13)
