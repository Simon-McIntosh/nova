"""Manage access to IMAS data entry."""

from dataclasses import dataclass, field, InitVar, KW_ONLY
from functools import cached_property
import logging
import os
from packaging.version import Version
from typing import Any, ClassVar, Optional, Type

from nova.utilities.importmanager import check_import

with check_import("imaspy"):
    import imaspy as imas
    from imaspy.ids_toplevel import IDSToplevel

logging.basicConfig(level=logging.WARNING)

ImasIds = Any
Ids = ImasIds | dict[str, int | str] | tuple[int | str]


@dataclass(kw_only=True)
class DDVersion:
    """Manage Data Dictionary version.

    Parameters
    ----------
    dd_version: Version | str | None. optional
        Data Dictionary version. The default is None

    Examples
    --------
    The data dictionary version may be set with a string, Version, or an envvar.

    >>> version = Version('3.22.0')
    >>> ddversion = DDVersion(dd_version=str(version))
    >>> ddversion.dd_version == version
    True
    >>> ddversion = DDVersion(dd_version=version)
    >>> ddversion.dd_version == version
    True
    >>> IMAS_VERSION = os.environ.get('IMAS_VERSION', '')
    >>> try:
    ...     os.environ['IMAS_VERSION'] = '3.30.0'
    ...     ddversion = DDVersion()
    ... finally:
    ...     if IMAS_VERSION:
    ...         os.environ['IMAS_VERSION'] = IMAS_VERSION
    ...     else:
    ...         del os.environ['IMAS_VERSION']
    >>> ddversion.dd_version == Version('3.30.0')
    True

    The dd_version must be a know data dictionary version.

    >>> DDVersion(dd_version='3.4') # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    UnknownDDVersion
    """

    dd_version: Version | str | None = None
    _dd_version: Version | str | None = field(init=False, repr=False, default=None)

    @property  # type: ignore[no-redef]
    def dd_version(self):  # noqa
        """Return a Version instance of dd_version."""
        return self._dd_version

    @dd_version.setter
    def dd_version(self, dd_version: Version | str | None):
        if type(dd_version) is property:
            dd_version = self._dd_version
        if isinstance(dd_version, Version):
            dd_version = str(dd_version)
        self._dd_version = Version(imas.IDSFactory(dd_version).version)


@dataclass
class IDSBase(DDVersion):  # noqa: D207
    """Manage IDS filesystem index attributes.

    Parameters
    ----------
    pulse: int, optional (required when ids not set)
        Pulse number. The default is 0.
    run: int, optional (required when ids not set)
        Run number. The default is 0.
    name: str, optional (required when ids not set)
        Ids name. The default is None.
    occurrence: int, optional (required when ids not set)
        Occurrence number. The default is 0.

    _: KW_ONLY
        The following attributes are keyword only.

    machine: str, optional (required when ids not set)
        Machine name. The default is iter.
    user: str, optional (required when ids not set)
        User name. The default is public.
    backend: str, optional (required when ids not set)
        Access layer backend. The default is hdf5.


    Raises
    ------
    TypeError
        Malformed imput passed to database instance.
    imas.exception.ALException
        Insufficient parameters passed to define ids if uri is empty.
        self.ids is None and pulse, run, and name set to defaults or None.


    Examples
    --------
    The IDSBase class manages the following filesystem index IDS attributes.

    >>> IDSBase().attrs
    ['pulse', 'run', 'name', 'occurrence', 'machine', 'user', 'backend']
    >>> list(Dataset.default_ids_attrs().keys()) == IDSBase().attrs
    True

    Attributes may be set positionaly or as keyword arguments.

    >>> ids = IDSBase(100, run=12)
    >>> ids.pulse, ids.run
    (100, 12)

    IDSBase attributes may be accessed from instance.

    >>> equilibrium = IDSBase(130506, 403, 'equilibrium')
    >>> equilibrium.pulse, equilibrium.run, equilibrium.name
    (130506, 403, 'equilibrium')
    >>> equilibrium.user, equilibrium.machine, equilibrium.backend
    ('public', 'iter', 'hdf5')

    IDS attributes can be accessed en-block via the ids_attrs property.

    >>> ids.ids_attrs = {"pulse": 7, "run": 8}
    >>> ids.ids_attrs["pulse"], ids.ids_attrs["run"]
    (7, 8)

    Default values may be accessed on the class.

    >>> ids.default_ids_attrs()["pulse"]
    0

    Updating undefined attributes raises an AttributeError.

    >>> ids.ids_attrs = {"shot": 6}
    Traceback (most recent call last):
        ...
    AttributeError: attr shot not in self.attrs ['pulse', 'run', 'name', 'occurrence', \
'machine', 'user', 'backend']

    The has_default_attrs property may be used to check if any IDSattrs have been set.

    >>> IDSBase().has_default_attrs
    True
    >>> IDSBase(pulse=7).has_default_attrs
    False

    The home, database_path, and ids_path properties are set based on the user value.

    >>> import os
    >>> imashome = os.environ['IMAS_HOME']
    >>> IDSBase(user='public').home == os.path.join(imashome, "shared")
    True
    >>> userhome = os.path.expanduser("~username")
    >>> IDSBase(user='username').home == os.path.join(userhome, "public")
    True
    >>> path = os.path.join(userhome, "public", "imasdb", "iter", "3")
    >>> IDSBase(user='username', machine='iter', dd_version='3.22.0').database_path\
== path
    True
    >>> idsattrs = IDSBase(1, 2, user='username', machine='iter', dd_version='3.22.0')
    >>> idsattrs.ids_path == \
os.path.join(userhome, 'public', 'imasdb', 'iter', '3', '1', '2')
    True

    Check for the existance of an IDS and set mode accordingly.

    >>> IDSBase(1, 3, 'pf_active').is_empty
    True

    """

    pulse: int = field(repr=False, default=0)
    run: int = field(repr=False, default=0)
    name: str | None = field(repr=False, default=None)
    occurrence: int = field(repr=False, default=0)
    _: KW_ONLY
    machine: str = field(repr=False, default="iter")
    user: str = field(repr=False, default="public")
    backend: str = field(repr=False, default="hdf5")

    index_attrs: ClassVar[list[str]] = [
        "pulse",
        "run",
        "name",
        "occurrence",
        "machine",
        "user",
        "backend",
    ]

    @property
    def attrs(self) -> list[str]:
        """Return IDS filesystem index attributes. Subclass to append."""
        return self.index_attrs

    @classmethod
    def default_ids_attrs(cls) -> dict:
        """Return dict of default ids attributes."""
        return {attr: getattr(cls, attr) for attr in cls.index_attrs}

    @property
    def ids_attrs(self):
        """Manage dict of ids attributes."""
        return {attr: getattr(self, attr) for attr in self.attrs}

    @ids_attrs.setter
    def ids_attrs(self, attrs: dict):
        for attr, value in attrs.items():
            if attr not in self.attrs:
                raise AttributeError(f"attr {attr} not in self.attrs {self.attrs}")
            setattr(self, attr, value)

    @property
    def has_default_attrs(self):
        """Return True if IDSattrs are default, else False."""
        return self.ids_attrs == self.default_ids_attrs()

    @property
    def uri(self):
        """Return URI build from IDSBase parameters."""
        if self.has_default_attrs:
            return ""
        self.check_ids_attrs()
        return (
            f"imas:{self.backend}?user={self.user};"
            f"pulse={self.pulse};run={self.run};"
            f"database={self.machine};version={self.dd_version.major};"
            f"#idsname={self.name}:occurrence={self.occurrence}"
        )

    @property
    def home(self):
        """Return database root."""
        if self.user == "public":
            return os.path.join(os.environ["IMAS_HOME"], "shared")
        return os.path.join(os.path.expanduser(f"~{self.user}"), "public")

    @property
    def database_path(self):
        """Return top level of database path."""
        return os.path.join(
            self.home, "imasdb", self.machine, str(self.dd_version.major)
        )

    @property
    def ids_path(self) -> str:
        """Return path to database entry."""
        match self.backend:
            case str(backend) if backend == "hdf5":
                return os.path.join(self.database_path, str(self.pulse), str(self.run))
            case _:
                raise NotImplementedError(
                    f"not implemented for {self.backend}" " backend"
                )

    @property
    def is_empty(self):
        """Return True if ids_path is empty, else False."""
        if os.path.isdir(self.ids_path) and os.listdir(self.ids_path):
            return False
        return True

    @property
    def _unset_attrs(self) -> bool:
        """Return True if any required input attributes are unset."""
        return (
            (self.pulse == 0 or self.pulse is None)
            or (self.run == 0 or self.run is None)
            or self.name is None
        )

    def check_ids_attrs(self):
        """Confirm minimum working set of input attributes."""
        if self._unset_attrs:
            raise imas.exception.ALException(
                f"When self.ids is None require:\n"
                f"pulse ({self.pulse} > 0) & run ({self.run} > 0) & "
                f"name ({self.name} != None)"
            )

    def _load_attrs_from_ids(self):
        """
        Initialize database class directly from an IDS.

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


@dataclass(kw_only=True)
class Datastore(IDSBase):  # noqa: D207
    """Locate IDS datastore. Manage URI and dd_version attributes.

    Parameters
    ----------
    uri : str. optional
        IDS unified resorce identifier. The default is constructed from IDSBase.

    Examples
    --------
    The uri property returns a URI built from the instance's attributes.

    >>> Datastore(101, 202, 'pf_active', 0, machine='iter', dd_version='3.22.0').uri
    'imas:hdf5?user=public;pulse=101;run=202;database=iter;version=3;\
#idsname=pf_active:occurrence=0'

    >>> datastore = Datastore(uri='imas:hdf5?path=/tmp/datastore')
    >>> datastore.uri = 'imas:hdf5?path=/tmp/datastore'

    The datastore may be identified via ether IDS attributes.

    >>> Datastore(101, 1, 'pf_active').uri != ''
    True

    or an explicit uri>

    >>> Datastore(uri='imas:hdf5?path=/tmp/datastore').uri != ''
    True

    But not both.

    >>> Datastore(101, uri='imas:hdf5?') # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    AttributeError

    A Datastore URI mush have a string type.

    >>> Datastore(uri=None)
    Traceback (most recent call last):
        ...
    TypeError: type(uri) <class 'NoneType'> is not str
    """

    uri: str = ""
    _uri: str = field(init=False, repr=False, default="")

    def __post_init__(self):
        """Update Datastore URI."""
        self._update_uri()

    def _update_uri(self):
        """Validate URI."""
        match self.uri:
            case "":
                self.uri = super().uri
            case str():
                if not self.has_default_attrs:
                    raise AttributeError(
                        f"Set ether IDSBase {self.ids_attrs} or " f"uri {self.uri}"
                    )
            case _:
                raise TypeError(f"type(uri) {type(self.uri)} is not str")

    @property  # type: ignore[no-redef]
    def uri(self) -> str:  # noqa
        """Overide IDSBase uri. Return Datastore URI."""
        return self._uri

    @uri.setter
    def uri(self, uri: str) -> None:
        if type(uri) is property:
            uri = self._uri
        self._uri = uri

    @property
    def _unset_attrs(self) -> bool:
        """Return True if any required input attributes are unset."""
        return (self.uri == "" or self.uri is None) and super()._unset_attrs


@dataclass(kw_only=True)
class Dataset(Datastore):  # noqa: D207
    """Locate an IDS dataset on a local or remote machine using a pulse-run layout.

    Parameters
    ----------
    mode: str | None. optional
        DBEntry file access mode. The default is ``"r"``.
              - ``"r"``: Open an existing data entry. Raises an error when the data
                entry does not exist.

                .. note:: The opened data entry is not read-only, it can be written to.
              - ``"a"``: Open an existing data entry, create the data entry if it does
                not exist.
              - ``"w"``: Create a data entry, overwriting any existing.

                .. caution:: This will irreversibly delete any existing data.
              - ``"x"``: Create a data entry. Raises an error when a data entry already
                exists.

    Attributes
    ----------
    home : os.Path, read-only
        Path to IMAS database home.

    ids_path : os.path, read-only
        Path to IDS database entry.


    Examples
    --------
    The Dataset class extends IDSBase adding attributes and methods for locating an IDS
    dataset.

    Arguments in positions > 4 are keyword only.

    >>> Dataset(0, 0, 'pf_active', 0, 'iter')
    Traceback (most recent call last):
        ...
    TypeError: Dataset.__init__() takes from 1 to 5 positional arguments \
but 6 were given

    The minimum input requred for Database is 'ids' or 'uri',
    'pulse', 'run', and 'name':

    >>> Dataset(uri='imas:hdf5?path=/tmp', mode=None, dd_version='3.22.0')
    Dataset(dd_version=<Version('3.22.0')>, uri='imas:hdf5?path=/tmp', \
mode=None, ids=None)

    >>> Dataset(pulse=101)
    Traceback (most recent call last):
        ...
    imas_core.exception.ALException: When self.ids is None require:
    pulse (101 > 0) & run (0 > 0) & name (None != None)

    >>> Dataset()
    Traceback (most recent call last):
        ...
    ValueError: No URI provided.

    Dataset instances can produce empty IDSs.

    >>> dataset = Dataset(1, 1, name='equilibrium', dd_version='3.22.0', mode=None)
    >>> equilibrium = dataset.new_ids()
    >>> equilibrium.has_value
    False
    >>> equilibrium.metadata.name == 'equilibrium'
    True
    >>> equilibrium._version == '3.22.0'
    True
    >>> equilibrium._lazy == False
    True

    Accessing an data entry that can not be found will raise a ImasCoreBackendException.

    >>> Dataset(1, 1, 'pf_active') # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    ImasCoreBackendException

    Skip remaining doctest if IMAS instalation or requisite Database(s) not found.

    >>> import pytest
    >>> try:
    ...     _ = Dataset(130506, 403, 'equilibrium')
    ... except:
    ...     pytest.skip('IMAS not installed or 130506/403 unavailable')

    Access to a lazy-loaded IDS is avaiable on instances created in a read mode='r'.

    >>> dataset = Dataset(130506, 403, 'equilibrium')
    >>> dataset.ids.code.name == 'CORSICA'
    True

    Subsiquent context access will close the file.

    >>> with dataset as ids:
    ...     dataset.ids == ids
    True

    >>> dataset.ids.code.name  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    RuntimeError:

    The link to the ids object may be patched with a subsiquent call to get().

    >>> dataset.get(lazy=False)  # load all data for hash example below
    <IDSToplevel (IDS:equilibrium)>
    >>> dataset.ids.code.name == 'CORSICA'
    True

    The database class may be initiated with an ids from which the
    name attribute may be recovered:

    >>> ids_dataset = Dataset(ids=dataset.ids)
    >>> ids_dataset.name
    'equilibrium'
    >>> ids_dataset.uri == ""
    True
    >>> ids_dataset = Dataset(ids=dataset.ids, uri='imas:hdf5?')
    >>> ids_dataset.name, ids_dataset.uri, ids_dataset.has_default_attrs
    ('equilibrium', 'imas:hdf5?', False)

    Other database attributes such as pulse and run, are
    not avalable when an ids is passed. These values are set to their default values.

    >>> ids_dataset.pulse == 0
    True
    >>> ids_dataset.run == 0
    True

    The equilibrium and database instances may be shown to share the same ids
    by comparing their respective hashes:

    >>> ids_dataset.ids_hash == dataset.ids_hash
    True

    However, due to differences if the pulse and run numbers of the database
    instance, which was iniciated directly from an ids, these instances are not
    considered to be equal to one another

    >>> ids_dataset != dataset
    True


    The ids_attrs property returns a dict of key instance attributes which may
    be used to identify the instance

    >>> dataset.ids_attrs == {"pulse": 130506, "run": 403, "name": "equilibrium",
    ...                       "occurrence": 0, "user": 'public', "machine": "iter",
    ...                       "backend": "hdf5"}
    True

    Dataset instances may be created via the from_ids_attrs class method:

    >>> ids_dataset = Dataset.from_ids_attrs(dataset.ids)
    >>> ids_dataset.pulse != 130506
    True
    >>> ids_dataset.run != 403
    True
    >>> ids_dataset.name
    'equilibrium'


    """

    mode: str | None = "r"  # Open an existing data entry. Raise error if not present.
    ids: IDSToplevel | None
    _ids: IDSToplevel | None = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Load ids."""
        super().__post_init__()
        self._open()

    def check_ids_attrs(self):
        """Extend to skip IDS attrs check when ids is not None."""
        if self.ids is not None:
            return
        super().check_ids_attrs()

    def get(
        self,
        name: Optional[str] = None,
        occurrence: Optional[int] = None,
        *,
        lazy=True,
    ) -> IDSToplevel:
        """Return IDS from Dataset."""
        if name is not None:
            self.name = name
        if occurrence is not None:
            self.occurrence = occurrence
        self.ids = self.db_entry.get(self.name, self.occurrence, lazy=lazy)
        return self.ids

    def put(self, ids: IDSToplevel, occurrence: int = 0):
        """Write ids to database entry."""
        self.db_entry.put(ids, occurrence)

    def _open(self):
        """Return."""
        if self.ids is not None:
            self.name = self.ids.metadata.name
            return
        match self.mode:
            case "r":
                self.get()

    @cached_property
    def db_entry(self):
        """Return imas.DBEntry instance."""
        return imas.DBEntry(uri=self.uri, mode=self.mode)

    @property
    def is_valid(self):
        """Return database entry validity flag."""
        # try:
        imas.DBEntry(uri=self.uri, mode="r").get(self.name, self.occurrence)
        # _mode = self.mode

    def __enter__(self):
        """Return DBEntry IDS with context."""
        self.db_entry.__enter__()
        return self.get()

    def __exit__(self, exc_type, exc_val, traceback):
        """Delete db_entry cached property on exit."""
        self.db_entry.__exit__(exc_type, exc_val, traceback)
        del self.db_entry

    @property  # type: ignore[no-redef]
    def ids(self) -> IDSToplevel:  # noqa
        """Manage ids attribute."""
        return self._ids

    @ids.setter
    def ids(self, ids: IDSToplevel):
        if type(ids) is property:
            ids = self._ids
        self._ids = ids

    @cached_property
    def ids_hash(self):
        """Return xx_hash of ids data."""
        if self.ids._lazy:
            self.get(lazy=False)
        return self.ids._xxhash()

    def new_ids(self, name: Optional[str] = None) -> None:
        """Return a new ids from IDSFactory and update name, and ids attributes."""
        if name is not None:
            self.name = name
        self.ids = imas.IDSFactory(version=str(self.dd_version)).new(self.name)
        return self.ids

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


@dataclass
class DataAttrs:  # noqa: D207
    """
    Methods to handle the formating of dataset attributes.

    Parameters
    ----------
    attrs: bool | Dataset | Ids
        Input attributes.
    subclass: Type[Dataset], optional
        Subclass instance or class. The default is Dataset.

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
    ...     _ = Dataset(130506, 403, 'equilibrium')
    ...     _ = Dataset(130506, 403, 'pf_active')
    ... except:
    ...     pytest.skip('IMAS not found or 130506/403 unavailable')

    The get_ids_attrs method is used to resolve a full ids_attrs dict from
    a partial input. If the input to get_ids_attrs is boolean then True
    returns the instance's default':

    >>> DataAttrs(True).attrs == Dataset.default_ids_attrs()
    True

    whilst False returns an empty dict:

    >>> DataAttrs(False).attrs
    False

    Dataset attributes may be extracted from any class derived from Dataset:

    >>> dataset = Dataset(130506, 403, 'equilibrium', machine='iter')
    >>> DataAttrs(dataset).attrs == dataset.ids_attrs
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

    >>> attrs = DataAttrs(dataset.ids).attrs
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
                       name='equilibrium')).attrs == dataset.ids_attrs
    True

    Raises TypeError when input attrs are malformed:

    >>> DataAttrs('equilibrium').attrs
    Traceback (most recent call last):
        ...
    TypeError: malformed attrs: <class 'str'>

    """

    ids_attrs: Ids | bool | str
    subclass: InitVar[Type[IDSBase]] = IDSBase
    default_attrs: dict = field(init=False, default_factory=dict)

    def __post_init__(self, subclass):
        """Update dataset attributes."""
        self.default_attrs = subclass.default_ids_attrs()

    @property
    def attrs(self) -> dict | bool:
        """Return output from update_attrs."""
        return self.update_ids_attrs()

    def merge_ids_attrs(self, base_attrs: dict):
        """Merge dataset attributes."""
        attrs = self.update_ids_attrs({"name": self.default_attrs["name"]})
        if isinstance(attrs, bool):
            return attrs
        return base_attrs | attrs

    def update_ids_attrs(self, default_attrs=None) -> dict | bool:
        """Return formated dataset attributes."""
        if default_attrs is None:
            default_attrs = self.default_attrs
        if self.ids_attrs is False:
            return False
        if self.ids_attrs is True:
            return default_attrs
        if isinstance(self.ids_attrs, IDSBase):
            return self.ids_attrs.ids_attrs
        if isinstance(self.ids_attrs, dict):
            return default_attrs | self.ids_attrs
        if hasattr(self.ids_attrs, "ids_properties"):  # IMAS ids
            dataset = Dataset(**default_attrs, ids=self.ids_attrs)
            return dataset.ids_attrs | {"ids": self.ids_attrs}
        if isinstance(self.ids_attrs, list | tuple):
            return default_attrs | dict(zip(Dataset.attrs, self.ids_attrs))
        raise TypeError(f"malformed attrs: {type(self.ids_attrs)}")


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
