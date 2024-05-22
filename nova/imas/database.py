"""Manage access to IMAS database."""

from contextlib import contextmanager
from dataclasses import dataclass, field, fields, InitVar
from functools import cached_property
import os
import packaging
from typing import Any, ClassVar, Optional, Type
import xxhash

from nova.database.datafile import Datafile
from nova.imas.db_entry import DBEntry
from nova.imas.datadir import DataDir
from nova.utilities.importmanager import check_import

with check_import("imaspy"):
    import imaspy as imas

EMPTY_INT = imas.ids_defs.EMPTY_INT
EMPTY_FLOAT = imas.ids_defs.EMPTY_FLOAT


# _pylint: disable=too-many-ancestors

ImasIds = Any
Ids = ImasIds | dict[str, int | str] | tuple[int | str]


@dataclass
class IDS(DataDir):
    """High level IDS attributes.

    name: str, optional (required when ids not set)
        Ids name. The default is ''.
    occurrence: int, optional (required when ids not set)
        Occurrence number. The default is 0.

    ids: ImasIds, optional
        When set the ids parameter takes prefrence. The default is None.

    Attributes
    ----------
    ids_attrs : dict
        Ids attributes as dict with keys [pulse, run, machine, occurence,
                                          user, name, backend]
    uri : str, read-only
        IDS unified resorce identifier.


    Methods
    -------
    get_ids()
        Return bare ids.

    dd_version()
        Return DD version.

    """

    name: str | None = None
    occurrence: int = 0

    attrs: ClassVar[list[str]] = [
        "occurrence",
        "name",
    ]

    @property
    def uri(self):
        """Return IDS URI, Extend DataEntry.uri with name:occurrence fragment."""
        return f"{super().uri}#idsname={self.name}:occurrence={self.occurrence}"

    def get_ids(self):
        """Return empty ids."""
        return getattr(imas.IDSFactory(), self.name)()

    @classmethod
    def update_ids_attrs(cls, ids_attrs: bool | Ids):
        """Return class attributes."""
        return DataAttrs(ids_attrs, cls).attrs

    @classmethod
    def merge_ids_attrs(cls, ids_attrs: bool | Ids, base_attrs: dict):
        """Return merged class attributes."""
        return DataAttrs(ids_attrs, cls).merge_ids_attrs(base_attrs)


@dataclass
class Database(IDS):
    """
    Methods to access IMAS database.

    Attributes
    ----------
    ids: ImasIds
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
    imas.exception.ALException
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

    >>> Database().ids_data  # doctest: +IGNORE_EXCEPTION_DETAIL
    >>> Database().ids
    Traceback (most recent call last):
        ...
    imas.exception.ALException: When self.ids is None require:
    pulse (0 > 0) & run (0 > 0) & name (None != None)

    Malformed inputs are thrown as TypeErrors:

    >>> malformed_database = Database(None, 403, name='equilibrium')
    >>> malformed_database.ids_data  # doctest: +IGNORE_EXCEPTION_DETAIL
    >>> malformed_database.ids
    Traceback (most recent call last):
        ...
    imas.exception.ALException: When self.ids is None require:
    pulse (None > 0) & run (403 > 0) & name (equilibrium != None)

    The database class may also be initiated with an ids from which the
    name attribute may be recovered:

    >>> database = Database(ids=equilibrium.ids)
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

    filename: str = field(default="", repr=False)
    group: str | None = field(default=None, repr=False)
    ids: ImasIds | None = field(repr=False, default=None)

    '''
    def __post_init__(self):
        """Load parameters and set ids."""
        self.rename()
        self.load_database()
        self.update_filename()
    '''

    @cached_property
    def db_entry(self):
        """Return nova.DBEntry instance."""
        return DBEntry(uri=self.uri, mode="a", database=self)

    def __enter__(self):
        """Access imas DBEntry.enter."""
        return self.db_entry.__enter__()

    def __exit__(self, exc_type, exc_val, traceback):
        """Patch."""
        self.db_entry.__exit__(exc_type, exc_val, traceback)

    def rename(self):
        """Reset name to default if default is not None."""
        if (
            name := next(
                field for field in fields(self) if field.name == "name"
            ).default
        ) is not None:
            self.name = name

    @property
    def _unset_attrs(self) -> bool:
        """Return True if any required input attributes are unset."""
        return (
            (self.pulse == 0 or self.pulse is None)
            or (self.run == 0 or self.run is None)
            or self.name is None
        )

    def _load_attrs_from_ids(self):
        """
        Initialize database class directly from an ids.

        Set unknown pulse and run numbers to zero if unset
        Update name to match ids.metadata.name
        """
        if self._unset_attrs:
            self.pulse = 0
            self.run = 0
        if self.name is not None and self.name != self.ids.metadata.name:
            raise NameError(
                f"missmatch between instance name {self.name} "
                f"and ids.metadata.name {self.ids.metadata.name}"
            )
        self.name = self.ids.metadata.name

    @cached_property
    def ids_dd_version(self) -> packaging.version.Version:
        """Return DD version used to write the IDS."""
        version_put = self.get_ids("ids_properties/version_put/data_dictionary")
        return packaging.version.parse(version_put.split("-")[0])

    def load_database(self):
        """Load instance database attributes."""
        if self.ids is not None:
            return self._load_attrs_from_ids()
        return None

    @property
    def classname(self):
        """Return base filename."""
        classname = f"{self.__class__.__name__.lower()}".replace("data", "")
        if classname == self.name:
            return self.machine
        return f"{classname}_{self.machine}"

    def update_filename(self):
        """Update filename."""
        if self.filename == "":
            self.filename = self.classname
            if (
                self.pulse is not None
                and self.pulse > 0
                and self.run is not None
                and self.run > 0
            ):
                self.filename += f"_{self.pulse}_{self.run}"
            if self.occurrence > 0:
                self.filename += f"_{self.occurrence}"
        if self.filename == "machine_description":
            self.filename = self.classname
        if self.group is None and self.name is not None:
            self.group = self.name

    def _check_ids_attrs(self):
        """Confirm minimum working set of input attributes."""
        if self._unset_attrs:
            raise imas.exception.ALException(
                f"When self.ids is None require:\n"
                f"pulse ({self.pulse} > 0) & run ({self.run} > 0) & "
                f"name ({self.name} != None)"
            )

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
        ids_name = "/".join(
            (item for item in [self.name, ids_path] if item is not None)
        ).split("/", 1)
        if occurrence is None:
            occurrence = self.occurrence
        with self.db_open() as db_entry:
            if len(ids_name) == 2:
                return getattr(
                    db_entry.get(ids_name[0], occurrence=occurrence, lazy=True),
                    ids_name[1],
                )
            return db_entry.get(*ids_name, occurrence=occurrence, lazy=True)

    def next_occurrence(self, limit=10000) -> int:
        """Return index of next available occurrence."""
        ids_path = "ids_properties/homogeneous_time"
        for i in range(limit):
            try:
                if self.get_ids(ids_path, i) == imas.ids_defs.EMPTY_INT:
                    return i
            except imas.exception.ALException:
                return i
        raise IndexError(f"no empty occurrences found for i < {limit}")

    @property
    def backend_id(self):  # TODO remove once uri interface is released
        """Return backend id from backend."""
        return getattr(imas.hli_utils.imasdef, f"{self.backend.upper()}_BACKEND")

    @property
    def dd_version(self) -> packaging.version.Version:
        """Return imas DD version."""
        try:
            return packaging.version.parse(imas.al_dd_version)
        except ValueError:
            return packaging.version.parse("1")
        return getattr(imas.ids_defs, f"{self.backend.upper()}_BACKEND")

    @contextmanager
    def _db_entry(self):
        """Yield database with context manager."""
        db_entry = imas.DBEntry(
            self.backend_id, self.machine, self.pulse, self.run, self.user
        )
        # db_entry = imas.DBEntry()  # TODO uri update
        yield db_entry
        db_entry.close()

    @contextmanager
    def db_open(self):
        """Yield open database entry."""
        with self._db_entry() as db_entry:
            try:
                db_entry.open()  # (uri=self.uri)  # TODO uri update
            except imas.exception.ALException as error:
                raise imas.exception.ALException(
                    f"malformed input to imas.DBEntry\n{error}\n"
                    f"pulse {self.pulse}, "
                    f"run {self.run}, "
                    f"user {self.user}\n"
                    f"machine {self.machine}, "
                    f"backend: {self.backend}"
                ) from error
            yield db_entry

    @property
    def db_empty(self):
        """Return true if database entry does not exist."""
        if os.path.isdir(self.ids_path) and os.listdir(self.ids_path):
            return False
        return True

    @property
    def db_mode(self):
        """Return db_entry mode."""
        if self.db_empty:
            return "create"
        return "open"

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

        This method is a placeholder.
        This method creates a hash of ids.__str__() which
        is only a partial representation of the ids object. Work is
        underway to provide ids hashes via the IMAS access layer.
        """
        xxh32 = xxhash.xxh32()
        xxh32.update(str(self.ids))
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

    >>> attrs = DataAttrs(database.ids).attrs
    >>> attrs['pulse'] != 130506
    True
    >>> attrs['run'] != 403
    True
    >>> attrs['ids'].metadata.name
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
        attrs = self.update_ids_attrs({"name": self.default_attrs["name"]})
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
        if hasattr(self.ids_attrs, "ids_properties"):  # IMAS ids
            database = Database(**default_attrs, ids=self.ids_attrs)
            return database.ids_attrs | {"ids": self.ids_attrs}
        if isinstance(self.ids_attrs, list | tuple):
            return default_attrs | dict(zip(Database.attrs, self.ids_attrs))
        raise TypeError(f"malformed attrs: {type(self.ids_attrs)}")


@dataclass
class IdsData(Datafile, Database):
    """Provide cached acces to imas ids data."""

    dirname: str = ".nova.imas"

    def merge_data(self, data):
        """Merge external data, interpolating to existing dataset timebase."""
        self.data = self.data.merge(
            data.interp(time=self.data.time), combine_attrs="drop_conflicts"
        )

    def load_data(self, ids_class):
        """Load data from IdsClass and merge."""
        if self.pulse == 0 and self.run == 0 and self.ids is None:
            return
        if self.ids is not None:
            ids_attrs = {"ids": self.ids}
        else:
            ids_attrs = self.ids_attrs
        try:
            data = ids_class(**ids_attrs).data
        except NameError:  # name missmatch when loading from ids node
            return
        if self.ids is not None:  # override when using ids input
            self.data = data
            return
        if hasattr(self.data, "time") and hasattr(data, "time"):
            data = data.interp({"time": self.data.time})
        self.data = data.merge(
            self.data, compat="override", combine_attrs="drop_conflicts"
        )

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

    dirname: str = field(default=".nova", repr=False)

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
        if hasattr(super(), "group_attrs"):
            return super().group_attrs
        return {}

    def build(self):
        """Build netCDF dataset."""
        super().build()


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
