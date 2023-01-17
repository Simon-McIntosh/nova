"""Manage access to IMAS database."""
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, InitVar
from importlib import import_module
from typing import Any, ClassVar, Optional, Type

import xxhash

from nova.database.netcdf import netCDF


# _pylint: disable=too-many-ancestors

ImasIds = Any
Ids = (ImasIds | dict[str, int | str] | tuple[int | str])


@dataclass
class IDS:
    """High level IDS attributes."""

    pulse: int = 0
    run: int = 0
    machine: str = 'iter'
    occurrence: int = 0
    user: str = 'public'
    name: str | None = None
    backend: int = 13

    attrs: ClassVar[list[str]] = ['pulse', 'run', 'machine', 'occurrence',
                                  'user', 'name', 'backend']


@dataclass
class Database(IDS):
    """
    Methods to access IMAS database.

    Parameters
    ----------
    pulse: int, optional (required when ids not set)
        Pulse number. The default is 0.
    run: int, optional (required when ids not set)
        Run number. The default is 0.
    occurrence: int, optional (required when ids not set)
        Occurrence number. The default is 0.
    machine: str, optional (required when ids not set)
        Machine name. The default is iter.
    user: str, optional (required when ids not set)
        User name. The default is public.
    name: str, optional (required when ids not set)
        Ids name. The default is ''.
    backend: int, optional (required when ids not set)
        Access layer backend. The default is 13 (HDF5).
    ids: ImasIds, optional
        When set the ids parameter takes prefrence. The default is None.

    Attributes
    ----------
    ids_data: ImasIds
        IMAS ids.
    ids_attrs: dict
        Ids attributes as dict with keys [pulse, run, machine, user, name]

    Notes
    -----
    The Database class regulates access to IMAS ids data. Requests may be made
    via pulse, run, name identifiers or as direct referances to
    open ids handles.

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

    >>> equilibrium = Database(130506, 403, name='equilibrium')
    >>> equilibrium.pulse, equilibrium.run, equilibrium.name
    (130506, 403, 'equilibrium')
    >>> equilibrium.user, equilibrium.machine, equilibrium.backend
    ('public', 'iter', 13)

    Minimum input requred for Database is 'ids' or 'pulse', 'run' and 'name':

    >>> Database()
    Traceback (most recent call last):
        ...
    ValueError: When self.ids is None require:
    pulse (0 > 0) & run (0 > 0) & name (None != None)

    Malformed inputs are thrown as TypeErrors:

    >>> malformed_database = Database(None, 403, name='equilibrium')
    >>> malformed_database.ids_data
    Traceback (most recent call last):
        ...
    TypeError: malformed input to imas.DBEntry
    an integer is required
    pulse None, run 403, user public
    machine iter, backend: 13

    The database class may also be initiated with an ids from which the
    name attribute may be recovered:

    >>> database = Database(ids=equilibrium.ids_data)
    >>> database.name
    'equilibrium'

    Other database attributes such as pulse and run, are
    not avalable when an ids is passed. These values are set to the hash of
    the ids. This enables automatic caching of ids derived data by downstream
    actors:

    >>> database.pulse != 130506
    True
    >>> database.run != 403
    True

    The equilibrium and database instances may be shown to share the same ids
    by comparing their respective hashes:

    >>> equilibrium.ids_hash == database.ids_hash
    True

    However, due to differences if the pulse and run numbers of the database
    instance, which was iniciated directly from an ids, these instances are not
    considered to be equal to one another

    >>> equilibrium != database
    True

    The ids_attrs property returns a dict of key instance attributes which may
    be used to identify the instance

    >>> equilibrium.ids_attrs == dict(pulse=130506, run=403, occurrence=0,\
                                      name='equilibrium', user='public', \
                                      machine='iter', backend=13)
    True

    Database instances may be created via the from_ids_attrs class method:

    >>> database = Database.from_ids_attrs(equilibrium.ids)
    >>> database.pulse != 130506
    True
    >>> database.run != 403
    True
    >>> database.name
    'equilibrium'

    """

    ids: ImasIds | None = field(repr=False, default=None)

    def __post_init__(self):
        """Load parameters and set ids."""
        self.rename()
        self.load_database()
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

    def rename(self):
        """Reset name to default if default is not None."""
        if (name := next(field for field in fields(self)
                         if field.name == 'name').default) is not None:
            self.name = name

    @property
    def ids_data(self):
        """Return ids data, lazy load."""
        if self.ids is None:
            self.ids = self.get_ids()
        return self.ids

    def load_database(self):
        """Load instance database attributes."""
        if self.ids is not None:
            return self._load_from_ids()
        return self._load_from_attrs()

    def _load_from_ids(self):
        """
        Initialize database class directly from an ids.

        Set unknown pulse and run numbers to the ids hash
        Update name to match ids.__name__
        """
        self.pulse = self.run = self.ids_hash
        if self.name is not None and self.name != self.ids_data.__name__:
            raise NameError(f'missmatch between instance name {self.name} '
                            f'and ids_data {self.ids_data.__name__}')
        self.name = self.ids_data.__name__

    def _load_from_attrs(self):
        """Confirm minimum working set of input attributes."""
        if self.pulse == 0 or self.run == 0 or self.name == '':
            raise ValueError(
                f'When self.ids is None require:\n'
                f'pulse ({self.pulse} > 0) & run ({self.run} > 0) & '
                f'name ({self.name} != None)')

    @classmethod
    def update_ids_attrs(cls, ids_attrs: bool | Ids):
        """Return class attributes."""
        return DataAttrs(ids_attrs, cls).attrs

    @classmethod
    def merge_ids_attrs(cls, ids_attrs: bool | Ids, base_attrs: dict):
        """Return merged class attributes."""
        return DataAttrs(ids_attrs, cls).merge_ids_attrs(base_attrs)

    @classmethod
    def from_ids_attrs(cls, ids_attrs: bool | Ids):
        """Initialize database instance from ids attributes."""
        if isinstance(attrs := cls.update_ids_attrs(ids_attrs), dict):
            return cls(**attrs)
        return False

    @classmethod
    def default_ids_attrs(cls) -> dict:
        """Return dict of ids attributes."""
        return {attr: getattr(cls, attr) for attr in cls.attrs}

    @property
    def ids_attrs(self):
        """Return dict of ids attributes."""
        return {attr: getattr(self, attr) for attr in self.attrs}

    @property
    def group_attrs(self):
        """
        Return database attributes.

        Group attrs used by :func:`~nova.database.filepath.FilePath.hash_attrs`
        to generate a unique hex hash to label data within a netCDF file.
        """
        return self.ids_attrs

    def get_ids(self, ids_path: Optional[str] = None):
        """Return ids. Extend name with ids_path if not None."""
        ids_name = '/'.join((item for item in [self.name, ids_path]
                             if item is not None)).split('/', 1)
        with self._get_ids() as db_entry:
            if len(ids_name) == 2:
                return db_entry.partial_get(*ids_name,
                                            occurrence=self.occurrence)
            return db_entry.get(*ids_name, occurrence=self.occurrence)

    @contextmanager
    def _get_ids(self):
        """Yield database with context manager."""
        try:
            imas = import_module('imas')
        except ImportError as error:
            raise ImportError('imas module not found'
                              'try module load IMAS') from error
        db_entry = imas.DBEntry(self.backend, self.machine,
                                self.pulse, self.run, user_name=self.user)
        try:
            db_entry.open()
        except TypeError as error:
            raise TypeError(f'malformed input to imas.DBEntry\n{error}\n'
                            f'pulse {self.pulse}, '
                            f'run {self.run}, '
                            f'user {self.user}\n'
                            f'machine {self.machine}, '
                            f'backend: {self.backend}') from error
        yield db_entry
        db_entry.close()

    @property
    def ids_hash(self) -> int:
        """
        Return ids hash.

        This method is a placeholder based the has of ids.__str__() which
        represents only a partial representation of the ids object. Work is
        underway to provide ids hashes via the IMAS access layer.
        """
        xxh32 = xxhash.xxh32()
        xxh32.update(str(self.ids_data))
        return xxh32.intdigest()


@dataclass
class DataAttrs:
    """
    Methods to handle the formating of database attributes.

    Parameters
    ----------
    attrs: bool | Database | Ids
        Input attributes.
    subclass: Type[Database], optional
        Subclass instance or class. The default is Database.

    Attributes
    ----------
    attrs: dict
        Resolved attributes.

    Raises
    ------
    TypeError
        Malformed attrs input passed to class.

    Examples
    --------
    The get_ids_attrs method is used to resolve a full ids_attrs dict from
    a partial input. If the input to get_ids_attrs is boolean then True
    returns the instance's default':

    >>> DataAttrs(True).attrs == Database.default_ids_attrs()
    True

    whilst False returns an empty dict:

    >>> DataAttrs(False).attrs
    False

    Database attributes may be extracted from any class derived from Database:

    >>> database = Database(130506, 403, 'iter', 0, name='equilibrium')
    >>> DataAttrs(database).attrs == database.ids_attrs
    True

    Passing a fully defined attribute dict returns this input:

    >>> attrs = dict(pulse=130506, run=403, occurrence=0, \
                     name='pf_active', user='other', \
                     machine='iter', backend=13)
    >>> DataAttrs(attrs).attrs == attrs
    True

    DataAttrs updates attrs with defaults for all missing values:

    >>> _ = attrs.pop('user')
    >>> DataAttrs(attrs).attrs == attrs | dict(user='public')
    True

    An additional example with attrs as a partial dict:

    >>> attrs = DataAttrs(dict(pulse=3, run=4)).attrs
    >>> attrs['pulse'], attrs['run'], attrs['machine']
    (3, 4, 'iter')

    Attrs may be input as an ids. In this case attrs is returned with
    hashed pulse and run numbers in additional to the original ids attribute:

    >>> attrs = DataAttrs(database.ids_data).attrs
    >>> attrs['pulse'] != 130506
    True
    >>> attrs['run'] != 403
    True
    >>> attrs['ids'].__name__
    'equilibrium'

    Attrs may be input as a list or tuple of args. This input is
    expanded by the passed subclass and must resolve to a valid ids. Partial
    input is acepted as long as the defaults enable a correct resolution.

    >>> DataAttrs(dict(pulse=130506, run=403,\
                       name='equilibrium')).attrs == database.ids_attrs
    True

    Raises TypeError when input attrs are malformed:

    >>> DataAttrs('equilibrium').attrs
    Traceback (most recent call last):
        ...
    TypeError: malformed attrs: <class 'str'>

    """

    ids_attrs: bool | Database | Ids
    subclass: InitVar[Type[Database]] = Database
    default_attrs: dict = field(init=False, default_factory=dict)

    def __post_init__(self, subclass):
        """Update database attributes."""
        self.default_attrs = subclass.default_ids_attrs()

    @property
    def attrs(self) -> dict | bool:
        """Return output from update_attrs."""
        return self.update_ids_attrs()

    def merge_ids_attrs(self, base_attrs: dict):
        """Merge database attributes."""
        attrs = self.update_ids_attrs(dict(name=self.default_attrs['name']))
        if isinstance(attrs, bool):
            return attrs
        return base_attrs | attrs

    def update_ids_attrs(self, default_attrs=None) -> dict | bool:
        """Return formated database attributes."""
        if default_attrs is None:
            default_attrs = self.default_attrs
        if self.ids_attrs is False:
            return False
        if self.ids_attrs is True:
            return default_attrs
        if isinstance(self.ids_attrs, Database):
            return self.ids_attrs.ids_attrs
        if isinstance(self.ids_attrs, dict):
            return default_attrs | self.ids_attrs
        if hasattr(self.ids_attrs, 'ids_properties'):  # IMAS ids
            database = Database(**default_attrs, ids=self.ids_attrs)
            return database.ids_attrs | dict(ids=self.ids_attrs)
        if isinstance(self.ids_attrs, list | tuple):
            return default_attrs | dict(zip(Database.attrs, self.ids_attrs))
        raise TypeError(f'malformed attrs: {type(self.ids_attrs)}')


@dataclass
class Datafile(netCDF):
    """
    Provide cached acces to imas ids data.

    Extends Database class via the provision of load and store methods.

    .. _RST Overview:

    See Also
    --------
    nova.imas.Database

    """

    def __post_init__(self):
        """Set ids and filepath."""
        super().__post_init__()
        self.load_build()

    def load_build(self):
        """
        Load netCDF data.

        Raises
        ------
        FileNotFoundError
            File not present: self.filepath
        OSError
            Group not present in netCDF file: self.group
        """
        try:
            self.load()
        except (FileNotFoundError, OSError):
            self.build()

    def build(self):
        """Build ids dataset."""
        raise NotImplementedError()


@dataclass
class IdsData(Datafile, Database):
    """Provide cached acces to imas ids data."""

    dirname: str = '.nova.imas'

    def __post_init__(self):
        """Update filename and group."""
        self.rename()
        if self.filename is None:
            self.filename = f'{self.machine}_{self.pulse}_{self.run}'
            if self.occurrence > 0:
                self.filename += f'_{self.occurrence}'
            self.group = self.name
        super().__post_init__()

    def merge_data(self, data):
        """Merge external data, interpolating to existing dataset timebase."""
        self.data = self.data.merge(data.interp(time=self.data.time),
                                    combine_attrs='drop_conflicts')

    def load_data(self, ids_class):
        """Load data from IdsClass and merge."""
        try:
            data = ids_class(**self.ids_attrs, ids=self.ids).data
        except NameError:  # load from single ids_data instance
            return
        self.data = self.data.merge(data, combine_attrs='drop_conflicts')

    def build(self):
        """Build ids dataset."""
        raise NotImplementedError()


@dataclass
class CoilData(Datafile):
    """
    Provide cached access to coilset data.

    Extends: :class:`~nova.imas.database.Datafile`

    See Also
    --------
    :class:`~nova.imas.database.Datafile`
    """

    dirname: str = '.nova'

    def __post_init__(self):
        """Update filename and group."""
        self.group = self.hash_attrs(self.group_attrs)
        super().__post_init__()

    @property
    def group_attrs(self):
        """
        Return group attributes.

        Group attrs used by :func:`~nova.database.filepath.FilePath.hash_attrs`
        to generate a unique hex hash to label data within a netCDF file.
        """
        if hasattr(super(), 'group_attrs'):
            return super().group_attrs
        return {}

    def build(self):
        """Build ids dataset."""
        raise NotImplementedError()


if __name__ == '__main__':

    import doctest
    doctest.testmod()
