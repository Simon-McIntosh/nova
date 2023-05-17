"""Manage access to IMAS database."""
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, InitVar
from operator import attrgetter
import os
from typing import Any, ClassVar, Optional, Type

try:
    import imas
    from imas.hli_exception import ALException
    from imas.hli_utils import imasdef
    IMAS_MODULE_NOT_FOUND = False
    EMPTY_INT = imasdef.EMPTY_INT
    EMPTY_FLOAT = imasdef.EMPTY_FLOAT
except (ModuleNotFoundError, SystemExit):
    IMAS_MODULE_NOT_FOUND = True
    EMPTY_INT = -999999999
    EMPTY_FLOAT = -9e40
import numpy as np
import xxhash

from nova.database.datafile import Datafile

# _pylint: disable=too-many-ancestors

ImasIds = Any
Ids = (ImasIds | dict[str, int | str] | tuple[int | str])


@dataclass
class IDS:
    """High level IDS attributes.

    Parameters
    ----------
    pulse: int, optional (required when ids not set)
        Pulse number. The default is 0.
    run: int, optional (required when ids not set)
        Run number. The default is 0.
    machine: str, optional (required when ids not set)
        Machine name. The default is iter.
    occurrence: int, optional (required when ids not set)
        Occurrence number. The default is 0.
    user: str, optional (required when ids not set)
        User name. The default is public.
    name: str, optional (required when ids not set)
        Ids name. The default is ''.
    backend: str, optional (required when ids not set)
        Access layer backend. The default is hdf5.
    ids: ImasIds, optional
        When set the ids parameter takes prefrence. The default is None.

    Attributes
    ----------
    ids_attrs : dict
        Ids attributes as dict with keys [pulse, run, machine, occurence,
                                          user, name, backend]
    uri : str, read-only
        IDS unified resorce identifier.

    home : os.Path, read-only
        Path to IMAS database home.

    path : os.path, read-only
        Path to IMAS database entry.

    Methods
    -------
    get_ids()
        Return bare ids.



    """

    pulse: int = 0
    run: int = 0
    machine: str = 'iter'
    occurrence: int = 0
    user: str = 'public'
    name: str | None = None
    backend: str = 'hdf5'

    dd_version: ClassVar[int] = 3
    attrs: ClassVar[list[str]] = ['pulse', 'run', 'machine', 'occurrence',
                                  'user', 'name', 'backend']

    @property
    def uri(self):
        """Return IDS URI."""
        return f"imas:{self.backend}?user={self.user};name={self.name};"\
               f"shot={self.pulse};run={self.run};"\
               f"occurrence={self.occurrence};"\
               f"database={self.machine};version={self.dd_version};"

    @property
    def home(self):
        """Return database root."""
        if self.user == 'public':
            return os.path.join(os.environ['IMAS_HOME'], 'shared')
        return os.path.join(os.path.expanduser(f'~{self.user}'), 'public')

    @property
    def _path(self):
        """Return top level of database path."""
        return os.path.join(self.home, 'imasdb', self.machine,
                            str(self.dd_version))

    @property
    def path(self):
        """Return path to database entry."""
        match self.backend:
            case str(backend) if backend == 'hdf5':
                return os.path.join(self._path, str(self.pulse), str(self.run))
            case _:
                raise NotImplementedError(f'not implemented for {self.backend}'
                                          ' backend')

    def get_ids(self):
        """Return empty ids."""
        return getattr(imas, self.name)()

    @classmethod
    def default_ids_attrs(cls) -> dict:
        """Return dict of default ids attributes."""
        return {attr: getattr(cls, attr) for attr in cls.attrs}

    @property
    def ids_attrs(self):
        """Return dict of ids attributes."""
        return {attr: getattr(self, attr) for attr in self.attrs}

    @classmethod
    def update_ids_attrs(cls, ids_attrs: bool | Ids):
        """Return class attributes."""
        return DataAttrs(ids_attrs, cls).attrs

    @classmethod
    def merge_ids_attrs(cls, ids_attrs: bool | Ids, base_attrs: dict):
        """Return merged class attributes."""
        return DataAttrs(ids_attrs, cls).merge_ids_attrs(base_attrs)


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
    Skip doctest if IMAS instalation or requisite IDS(s) not found.

    >>> import pytest
    >>> try:
    ...     _ = Database(130506, 403).get_ids('equilibrium')
    ...     _ = Database(130506, 403).get_ids('pf_active')
    ... except:
    ...     pytest.skip('IMAS not found or 130506/403 unavailable')

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
                     machine='iter', backend='hdf5')
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

    ids_attrs: Ids | bool | str
    subclass: InitVar[Type[IDS]] = IDS
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
        attrs = self.update_ids_attrs({'name': self.default_attrs['name']})
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
            return database.ids_attrs | {'ids': self.ids_attrs}
        if isinstance(self.ids_attrs, list | tuple):
            return default_attrs | dict(zip(Database.attrs, self.ids_attrs))
        raise TypeError(f'malformed attrs: {type(self.ids_attrs)}')


@dataclass
class Database(IDS):
    """
    Methods to access IMAS database.

    Attributes
    ----------
    ids_data: ImasIds
        IMAS ids.
    ids_attrs: dict
        Ids attributes as dict with keys [pulse, run, machine, occurence,
                                          user, name, backend]

    Notes
    -----
    The Database class regulates access to IMAS ids data. Requests may be made
    via pulse, run, name identifiers or as direct referances to
    open ids handles.

    Raises
    ------
    ImportError
        Imas module not found. IMAS access layer not loaded or installed.
    TypeError
        Malformed imput passed to database instance.
    imas.hli_exceptionALException
        Insufficient parameters passed to define ids.
        self.ids is None and pulse, run, and name set to defaults or None.

    Examples
    --------
    Skip doctest if IMAS instalation or requisite IDS(s) not found.

    >>> import pytest
    >>> try:
    ...     _ = Database(130506, 403).get_ids('equilibrium')
    ... except:
    ...     pytest.skip('IMAS not installed or 130506/403 unavailable')

    Load an equilibrium ids from file with defaults for user, machine and
    backend:

    >>> equilibrium = Database(130506, 403, name='equilibrium')
    >>> equilibrium.pulse, equilibrium.run, equilibrium.name
    (130506, 403, 'equilibrium')
    >>> equilibrium.user, equilibrium.machine, equilibrium.backend
    ('public', 'iter', 'hdf5')

    Minimum input requred for Database is 'ids' or 'pulse', 'run' and 'name':

    >>> Database().ids_data
    Traceback (most recent call last):
        ...
    imas.hli_exception.ALException: When self.ids is None require:
    pulse (0 > 0) & run (0 > 0) & name (None != None)

    Malformed inputs are thrown as TypeErrors:

    >>> malformed_database = Database(None, 403, name='equilibrium')
    >>> malformed_database.ids_data
    Traceback (most recent call last):
        ...
    imas.hli_exception.ALException: When self.ids is None require:
    pulse (None > 0) & run (403 > 0) & name (equilibrium != None)

    The database class may also be initiated with an ids from which the
    name attribute may be recovered:

    >>> database = Database(ids=equilibrium.ids_data)
    >>> database.name
    'equilibrium'

    Other database attributes such as pulse and run, are
    not avalable when an ids is passed. These values are set to the hash of
    the ids. This enables automatic caching of ids derived data by downstream
    actors. The ids hash is always negative:

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
                                      machine='iter', backend='hdf5')
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

    filename: str = field(default='', repr=False)
    group: str | None = field(default=None, repr=False)
    ids: ImasIds | None = field(repr=False, default=None)

    def __post_init__(self):
        """Load parameters and set ids."""
        self.rename()
        self.load_database()
        self.update_filename()

    def rename(self):
        """Reset name to default if default is not None."""
        if (name := next(field for field in fields(self)
                         if field.name == 'name').default) is not None:
            self.name = name

    @property
    def ids_data(self):
        """Return ids data, lazy load."""
        if self.ids is None:
            self._check_ids_attrs()
            self.ids = self.get_ids()
        return self.ids

    def load_database(self):
        """Load instance database attributes."""
        if self.ids is not None:
            return self._load_attrs_from_ids()
        return None

    @property
    def classname(self):
        """Return base filename."""
        classname = f'{self.__class__.__name__.lower()}'.replace('data', '')
        if classname == self.name:
            return self.machine
        return f'{classname}_{self.machine}'

    def update_filename(self):
        """Update filename."""
        if self.filename == '':
            self.filename = self.classname
            if self.pulse is not None and self.pulse > 0 and \
                    self.run is not None and self.run > 0:
                self.filename += f'_{self.pulse}_{self.run}'
            if self.occurrence > 0:
                self.filename += f'_{self.occurrence}'
        if self.filename == 'machine_description':
            self.filename = self.classname
        if self.group is None and self.name is not None:
            self.group = self.name

    @property
    def _unset_attrs(self) -> bool:
        """Return True if any required input attributes are unset."""
        return self.pulse == 0 or self.pulse is None or \
            self.run == 0 or self.run is None or self.name is None

    def _load_attrs_from_ids(self):
        """
        Initialize database class directly from an ids.

        Set unknown pulse and run numbers to the ids hash
        Update name to match ids.__name__
        """
        if self._unset_attrs:
            self.pulse = 0
            self.run = 0
        if self.name is not None and self.name != self.ids_data.__name__:
            raise NameError(f'missmatch between instance name {self.name} '
                            f'and ids_data {self.ids_data.__name__}')
        self.name = self.ids_data.__name__

    def _check_ids_attrs(self):
        """Confirm minimum working set of input attributes."""
        if self._unset_attrs:
            raise ALException(
                f'When self.ids is None require:\n'
                f'pulse ({self.pulse} > 0) & run ({self.run} > 0) & '
                f'name ({self.name} != None)')

    @classmethod
    def from_ids_attrs(cls, ids_attrs: bool | Ids):
        """Initialize database instance from ids attributes."""
        if isinstance(attrs := cls.update_ids_attrs(ids_attrs), dict):
            return cls(**attrs)
        return False

    @property
    def group_attrs(self):
        """
        Return database attributes.

        Group attrs used by :func:`~nova.database.filepath.FilePath.hash_attrs`
        to generate a unique hex hash to label data within a netCDF file.
        """
        return self.ids_attrs

    def get_ids(self, ids_path: Optional[str] = None, occurrence=None):
        """Return ids. Override IDS.get_ids. Extend name with ids_path."""
        ids_name = '/'.join((item for item in [self.name, ids_path]
                             if item is not None)).split('/', 1)
        if occurrence is None:
            occurrence = self.occurrence
        with self.db_open() as db_entry:
            if len(ids_name) == 2:
                return db_entry.partial_get(*ids_name, occurrence=occurrence)
            return db_entry.get(*ids_name, occurrence=occurrence)

    def next_occurrence(self, limit=10000) -> int:
        """Return index of next available occurrence."""
        ids_path = 'ids_properties/homogeneous_time'
        for i in range(limit):
            try:
                if self.get_ids(ids_path, i) == imas.imasdef.EMPTY_INT:
                    return i
            except ALException:
                return i
        raise IndexError(f'no empty occurrences found for i < {limit}')

    @property
    def backend_id(self):  # TODO remove once uri interface is released
        """Return backend id from backend."""
        return getattr(imas.hli_utils.imasdef,
                       f'{self.backend.upper()}_BACKEND')

    @contextmanager
    def _db_entry(self):
        """Yield database with context manager."""
        if IMAS_MODULE_NOT_FOUND:
            raise ImportError('imas module not found, try `ml load IMAS`')
        db_entry = imas.DBEntry(self.backend_id, self.machine, self.pulse,
                                self.run, self.user)
        # db_entry = imas.DBEntry()  # TODO uri update
        yield db_entry
        db_entry.close()

    @contextmanager
    def db_open(self):
        """Yield open database entry."""
        with self._db_entry() as db_entry:
            try:
                db_entry.open()  # (uri=self.uri)  # TODO uri update
            except ALException as error:
                raise ALException(
                    f'malformed input to imas.DBEntry\n{error}\n'
                    f'pulse {self.pulse}, '
                    f'run {self.run}, '
                    f'user {self.user}\n'
                    f'machine {self.machine}, '
                    f'backend: {self.backend}') from error
            yield db_entry

    @property
    def db_empty(self):
        """Return true if database entry does not exist."""
        if os.path.isdir(self.path) and os.listdir(self.path):
            return False
        return True

    @property
    def db_mode(self):
        """Return db_entry mode."""
        if self.db_empty:
            return 'create'
        return 'open'

    @contextmanager
    def db_write(self):
        """Yeild bare database entry."""
        with self._db_entry() as db_entry:
            getattr(db_entry, self.db_mode)()  # (uri=self.uri)  # TODO uri
            yield db_entry

    def put_ids(self, ids, occurrence=None):
        """Write ids data to database entry."""
        if occurrence is None:
            occurrence = self.occurrence
        with self.db_write() as db_entry:
            db_entry.put(ids, occurrence=occurrence)

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
class IdsIndex:
    """
    Methods for indexing data as arrays from an ids.

    Parameters
    ----------
    ids_data : ImasIds
        IMAS IDS (in-memory).
    ids_node : str
        Array extraction node.

    Examples
    --------
    Check access to required IDS(s).

    >>> import pytest
    >>> try:
    ...     _ = Database(105028, 1).get_ids('pf_active')
    ...     _ = Database(105028, 1).get_ids('equilibrium')
    ...     _ = Database(135007, 4).get_ids('pf_active')
    ... except:
    ...     pytest.skip('IMAS not installed or 105028/1, 135007/4 unavailable')

    First load an ids, accomplished here using the Database class from
    nova.imas.database.

    >>> from nova.imas.database import Database, IdsIndex
    >>> pulse, run = 105028, 1  # DINA scenario data
    >>> pf_active = Database(pulse, run, name='pf_active')

    Initiate an instance of IdsIndex using ids_data from pf_active and
    specifying 'coil' as the array extraction node.

    >>> ids_index = IdsIndex(pf_active.ids_data, 'coil')

    Get first 5 coil names.

    >>> ids_index.array('name')[:5]
    array(['CS3U', 'CS2U', 'CS1', 'CS2L', 'CS3L'], dtype=object)

    Get full array of current data (551 time slices for all 12 coils).

    >>> current = ids_index.array('current.data')
    >>> current.shape
    (551, 12)

    Get vector of coil currents at single time slice (itime=320)

    >>> current = ids_index.vector(320, 'current.data')
    >>> current.shape
    (12,)

    Load equilibrium ids and initiate new instance of ids_index.
    >>> equilibrium = Database(pulse, run, name='equilibrium')
    >>> ids_index = IdsIndex(equilibrium.ids_data, 'time_slice')

    Get psi at itime=30 from profiles_1d and profiles_2d.

    >>> ids_index.vector(30, 'profiles_1d.psi').shape
    (50,)
    >>> ids_index.vector(30, 'profiles_2d.psi').shape
    (65, 129)

    Load pf_active ids containing force data.

    >>> pulse, run = 135007, 4  # DINA scenario including force data
    >>> pf_active = Database(pulse, run, name='pf_active')
    >>> ids_index = IdsIndex(pf_active.ids_data, 'coil')

    Use context manager to temporarily switch the ids_node to radial_force
    and vertical_force and extract force data at itime=100 from each node.

    >>> with ids_index.node('radial_force'):
    ...     ids_index.vector(100, 'force.data').shape
    (12,)

    >>> with ids_index.node('vertical_force'):
    ...     ids_index.vector(100, 'force.data').shape
    (17,)

    """

    ids_data: ImasIds
    ids_node: str = 'time_slice'
    transpose: bool = field(init=False, default=False)
    length: int = field(init=False, default=0)
    shapes: dict[str, tuple[int, ...] | tuple[()]] = \
        field(init=False, default_factory=dict)

    def __post_init__(self):
        """Initialize ids node."""
        self.ids = self.ids_node

    @contextmanager
    def node(self, ids_node: str):
        """
        Permit tempary change to an instance's ids_node.

        Example
        -------
        Check access to required IDS(s).

        >>> import pytest
        >>> try:
        ...     _ = Database(135007, 4).get_ids('pf_active')
        ... except:
        ...     pytest.skip('IMAS not installed or 135007/4 unavailable')

        Demonstrate use of context manager for switching active ids_node.

        >>> from nova.imas.database import IdsIndex
        >>> ids_data = Database(135007, 4, name='pf_active').ids_data
        >>> ids_index = IdsIndex(ids_data, 'coil')
        >>> with ids_index.node('vertical_force'):
        ...     ids_index.array('force.data').shape
        (2338, 17)
        """
        _ids_node = self.ids_node
        self.ids = ids_node
        yield
        self.ids = _ids_node

    @property
    def ids(self):
        """Return ids_data node."""
        if self.ids_node:
            return attrgetter(self.ids_node)(self.ids_data)
        return self.ids_data

    @ids.setter
    def ids(self, ids_node: str | None):
        """Update ids node."""
        if ids_node is not None:
            self.transpose = ids_node != 'time_slice'
            self.ids_node = ids_node
        try:
            self.length = len(self.ids)
        except (AttributeError, TypeError):
            self.length = 0

    def __getitem__(self, path: str) -> tuple[int, ...] | tuple[()]:
        """Return cached dimension length."""
        _path = self.ids_path(path)
        try:
            return self.shapes[_path]
        except KeyError:
            self.shapes[_path] = self._path_shape(path)
            return self[path]

    def ids_path(self, path: str) -> str:
        """Return full ids path."""
        if self.ids_node is None:
            return path
        return f'{self.ids_node}.{path}'

    def shape(self, path) -> tuple[int, ...]:
        """Return attribute array shape."""
        if self.length == 0:
            return self[path]
        return (self.length,) + self[path]

    def _path_shape(self, path: str) -> tuple[int, ...] | tuple[()]:
        """Return data shape at itime=0 on path."""
        match data := self.get_slice(0, path):
            case np.ndarray():
                return data.shape
            case float() | int() | str():
                return ()
            case _:
                raise ValueError(f'unable to determine data length {path}')

    def get(self, path: str):
        """Return attribute from ids path."""
        return attrgetter(path)(self.ids)

    def resize(self, path: str, number: int):
        """Resize structured array."""
        attrgetter(path)(self.ids_data).resize(number)

    def __setitem__(self, key, value):
        """Set attribute on ids path."""
        match key:
            case str(attr):
                index, subindex = 0, 0
            case (str(attr), index):
                subindex = 0
            case (str(attr), index, int(subindex)):
                pass
            case _:
                raise KeyError(f'invalid key {key}')

        if isinstance(index, slice):
            # recursive update for all indicies specified in slice.
            for _index, _value in zip(range(len(value))[index], value[index]):
                self.__setitem__((attr, _index, subindex), _value)
            return

        path = self.get_path(self.ids_node, attr)
        split_path = path.split('.')
        node = '.'.join(split_path[:-1])
        leaf = split_path[-1]
        match node.split(':'):
            case (str(node),):
                branch = attrgetter(node)(self.ids_data)
            case (str(array), str(node)):
                trunk = attrgetter(array)(self.ids_data)[index]
                branch = attrgetter(node)(trunk)
            case _:
                raise IndexError(f'invalid node {node}')
        match leaf.split(':'):
            case (str(leaf),):
                setattr(branch, leaf, value)
            case str(stem), str(leaf):
                try:
                    shoot = attrgetter(stem)(branch)[subindex]
                except IndexError:
                    attrgetter(stem)(branch).resize(subindex+1)
                    shoot = attrgetter(stem)(branch)[subindex]
                setattr(shoot, leaf, value)
            case _:
                raise NotImplementedError(f'invalid leaf {leaf}')

    def get_slice(self, index: int, path: str):
        """Return attribute slice at node index."""
        try:
            return attrgetter(path)(self.ids[index])
        except AttributeError:  # __structArray__
            node, path = path.split('.', 1)
            return attrgetter(path)(
                attrgetter(node)(self.ids[index])[0])
        except TypeError:  # object is not subscriptable
            return self.get(path)

    def vector(self, itime: int, path: str):
        """Return attribute data vector at itime."""
        if len(self[path]) == 0:
            raise IndexError(f'attribute {path} is 0-dimensional '
                             'access with self.array(path)')
        if self.transpose:
            data = np.zeros(self.shape(path)[:-1], dtype=self.dtype(path))
            for index in range(self.length):
                try:
                    data[index] = self.get_slice(index, path)[itime]
                except (ValueError, IndexError):  # empty slice
                    pass
            return data
        return self.get_slice(itime, path)

    def array(self, path: str):
        """Return attribute data array."""
        if self.length == 0:
            return self.get(path)
        data = np.zeros(self.shape(path), dtype=self.dtype(path))
        for index in range(self.length):
            try:
                data[index] = self.get_slice(index, path)
            except ValueError:  # empty slice
                pass
        if self.transpose:
            return data.T
        return data

    def valid(self, path: str):
        """Return validity flag for ids path."""
        try:
            self.empty(path)
            return True
        except TypeError:
            return False

    def empty(self, path: str):
        """Return status based on first data point extracted from ids_data."""
        try:
            data = self.get_slice(0, path)
        except IndexError:
            return True
        if hasattr(data, 'flat'):
            try:
                data = data.flat[0]
            except IndexError:
                return True
        try:  # string
            return len(data) == 0
        except TypeError:
            return data is None or np.isclose(data, -9e40) \
                or np.isclose(data, -999999999)

    def dtype(self, path: str):
        """Return data point type."""
        if self.empty(path):
            raise ValueError(f'data entry at {path} is empty')
        data = self.get_slice(0, path)
        if isinstance(data, str):
            return object
        if hasattr(data, 'flat'):
            return type(data.flat[0])
        return type(data)

    @staticmethod
    def get_path(branch: str, attr: str) -> str:
        """Return ids attribute path."""
        if not branch:
            return attr
        if '*' in branch:
            return branch.replace('*', attr)
        return '.'.join((branch, attr))


@dataclass
class IdsEntry(IdsIndex, IDS):
    """Methods to facilitate sane ids entry."""

    ids_data: ImasIds = None
    ids_node: str = ''
    database: Database | None = field(init=False, default=None)

    def __post_init__(self):
        """Initialize ids_data and create database instance."""
        if self.ids_data is None:
            self.ids_data = self.get_ids()
        self.database = Database(**self.ids_attrs, ids=self.ids_data)
        super().__post_init__()

    def put_ids(self, occurrence=None):
        """Expose Database.put_ids."""
        self.database.put_ids(self.ids_data, occurrence)


@dataclass
class IdsData(Datafile, Database):
    """Provide cached acces to imas ids data."""

    dirname: str = '.nova.imas'

    #def assert_final(self, classname: str):
    #    """

    def merge_data(self, data):
        """Merge external data, interpolating to existing dataset timebase."""
        self.data = self.data.merge(data.interp(time=self.data.time),
                                    combine_attrs='drop_conflicts')

    def load_data(self, ids_class):
        """Load data from IdsClass and merge."""
        if self.pulse == 0 and self.run == 0 and self.ids is None:
            return
        try:
            data = ids_class(**self.ids_attrs, ids=self.ids).data
        except NameError:  # name missmatch when loading from ids node
            return
        if hasattr(self.data, 'time') and hasattr(data, 'time'):
            data = data.interp({'time': self.data.time})
        self.data = self.data.merge(data, compat='override',
                                    combine_attrs='drop_conflicts')

    def build(self):
        """Build netCDF dataset."""
        super().build()


@dataclass
class CoilData(IdsData):
    """
    Provide cached access to coilset data.

    Extends: :class:`~nova.imas.database.IdsData`

    See Also
    --------
    :class:`~nova.imas.database.IdsData`
    """

    dirname: str = field(default='.nova', repr=False)

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
        """Build netCDF dataset."""
        super().build()
