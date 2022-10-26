"""Manage access to IMAS database."""
from contextlib import contextmanager
from dataclasses import dataclass, field

try:
    from imas import DBEntry
    IMPORT_IMAS = True
except ImportError:
    IMPORT_IMAS = False
import xxhash

from nova.database.filepath import FilePath

# pylint: disable=too-many-ancestors

ImasIds = object


@dataclass
class Database:
    """
    Methods to access IMAS database.

    Parameters
    ----------
    pulse: int, optional (required when ids not set)
        Pulse number. The default is 0.
    run: int, optional (required when ids not set)
        Run number. The default is 0.
    name: str, optional (required when ids not set)
        Ids name. The default is ''.
    user: str, optional (required when ids not set)
        User name. The default is public.
    machine: str, optional (required when ids not set)
        Machine name. The default is iter.
    backend: int, optional (required when ids not set)
        Access layer backend. The default is 13 (HDF5).
    ids: ImasIds, optional
        When set the ids parameter takes prefrence. The default is None.

    Attributes
    ----------
    ids: ImasIds
        IMAS ids.
    ids_attrs: dict
        Ids attributes as dict with keys [pulse, run, name, user, machine]

    Notes
    -----
    The Database class regulates access to IMAS ids data. Requests may be made
    via pulse, run, name identifiers or as direct referances to
    open ids handels.

    See Also
    --------
    nova.imas.Datafile: Cached access to ids data.

    Raises
    ------
    ImportError
        Imas module not found. IMAS access layer not loaded or installed.
    TypeError
        Malformed imput passed to database instance.
    ValueError
        Insufficient parameters passed to define ids.
        self.ids is None and pulse, run, and name set to defaults.

    Examples
    --------
    Load an equilibrium ids from file with defaults for user, machine and
    backend:

    >>> equilibrium = Database(130506, 403, 'equilibrium')
    >>> equilibrium.pulse, equilibrium.run, equilibrium.name
    (130506, 403, 'equilibrium')
    >>> equilibrium.user, equilibrium.machine, equilibrium.backend
    ('public', 'iter', 13)

    Minimum input requred for Database is 'ids' or 'pulse', 'run' and 'name':

    >>> Database()
    Traceback (most recent call last):
        ...
    ValueError: When self.ids is None require:
    pulse 0 > 0, run 0 > 0 and name "" != ""

    Malformed inputs are thrown as TypeErrors:

    >>> Database(None, 403, 'equilibrium')
    Traceback (most recent call last):
        ...
    TypeError: malformed input to DBEntry
    an integer is required
    pulse None, run 403, user public
    machine iter, backend: 13

    The database class may also be initiated with an ids from which the
    name attribute may be recovered:

    >>> database = Database(ids=equilibrium.ids)
    >>> database.name
    'equilibrium'

    Other database attributes such as pulse and run, are
    not avalable when an ids is passed. These values are set to the hash of
    the ids. This enables automatic caching of ids derived data by downstream
    actors:

    >>> database.pulse, database.run
    (3600040824, 3600040824)

    The equilibrium and database instances may be shown to share the same ids
    by comparing their respective hashes:

    >>> hash(equilibrium) == hash(database)
    True

    However, due to differences if the pulse and run numbers of the database
    instance, which was iniciated directly from an ids, these instances are not
    considered to be equal to one another

    >>> equilibrium != database
    True

    The ids_attrs property returns a dict of key instance attributes which may
    be used to identify the instance

    >>> equilibrium.ids_attrs == dict(pulse=130506, run=403, \
                                      name='equilibrium', user='public', \
                                      machine='iter')
    True

    """

    pulse: int = field(default=0)
    run: int = field(default=0)
    name: str = field(default='')
    user: str = field(default='public')
    machine: str = field(default='iter')
    backend: int = field(default=13)
    ids: ImasIds | None = field(repr=False, default=None)

    def __post_init__(self):
        """Load parameters and set ids."""
        if self.ids is None:
            return self.get_ids()
        return self.load_ids()

    @classmethod
    def default_ids_attrs(cls):
        """Return dict of ids attributes."""
        return dict(pulse=cls.pulse, run=cls.run, name=cls.name,
                    user=cls.user, machine=cls.machine)

    @property
    def ids_attrs(self):
        """Return dict of ids attributes."""
        return dict(pulse=self.pulse, run=self.run, name=self.name,
                    user=self.user, machine=self.machine)

    def get_ids(self):
        """Set ids from pulse/run."""
        if self.pulse == 0 or self.run == 0 or self.name == '':
            raise ValueError(
                f'When self.ids is None require:\n'
                f'pulse {self.pulse} > 0, run {self.run} > 0 and '
                f'name "{self.name}" != ""')
        with self._get_ids() as ids:
            self.ids = ids

    @contextmanager
    def _get_ids(self):
        """Yield database with context manager."""
        if not IMPORT_IMAS:
            raise ImportError('imas module not found'
                              'try module load IMAS')
        db_entry = DBEntry(self.backend, self.machine,
                           self.pulse, self.run, user_name=self.user)
        try:
            db_entry.open()
        except TypeError as error:
            raise TypeError(f'malformed input to DBEntry\n{error}\n'
                            f'pulse {self.pulse}, '
                            f'run {self.run}, '
                            f'user {self.user}\n'
                            f'machine {self.machine}, '
                            f'backend: {self.backend}') from error
        try:
            name, ids_path = self.name.split('.', 1)
            yield db_entry.partial_get(name, ids_path)
        except ValueError:
            yield db_entry.get(self.name)
        db_entry.close()

    def __hash__(self) -> int:
        """
        Return ids hash.

        This method is a placeholder based the has of ids.__str__() which
        represents only a partial representation of the ids object. Work is
        underway to provide ids hashes via the IMAS access layer.
        """
        xxh32 = xxhash.xxh32()
        xxh32.update(self.ids.__str__())
        return xxh32.intdigest()

    def load_ids(self):
        """
        Initialize database class directly from an ids.

        Set unknown pulse and run numbers to the ids hash
        Update name to match ids.__name__
        """
        self.pulse = self.run = hash(self)
        self.name = self.ids.__name__


@dataclass
class Datafile(FilePath, Database):
    """
    Provide cached acces to imas ids data.

    Extends Database class via the provision of load and store methods.

    See Also
    --------
    nova.imas.Database

    """

    directory: str = 'user_data'
    datapath: str = 'imas'

    def __post_init__(self):
        """Set ids and filepath."""
        super().__post_init__()
        try:
            self.filename = f'{self.machine}_{self.pulse}{self.run:04d}'
        except TypeError:
            self.filename = None
        self.group = self.name
        self.set_path(self.datapath)


if __name__ == '__main__':

    import doctest
    doctest.testmod()
