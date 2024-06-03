"""Manage access to IMAS database."""

from dataclasses import dataclass, field, fields
from functools import cached_property
import packaging
from typing import Optional

from nova.database.datafile import Datafile
from nova.imas.dataset import Dataset, ImasIds
from nova.utilities.importmanager import check_import

with check_import("imaspy"):
    import imaspy as imas

EMPTY_INT = imas.ids_defs.EMPTY_INT
EMPTY_FLOAT = imas.ids_defs.EMPTY_FLOAT


# _pylint: disable=too-many-ancestors


@dataclass
class _IDS(Dataset):
    """High level access to single IDS.

    Parameters
    ----------
    ids: ImasIds, optional
        When set the ids parameter takes prefrence. The default is None.


    Examples
    --------

    """

    ids: ImasIds | None = field(repr=False, default=None)

    def get_ids(self, ids_path: Optional[str] = None, occurrence=None):
        """Return ids. Extend name with ids_path."""
        ids_name = "/".join(
            (item for item in [self.name, ids_path] if item is not None)
        ).split("/", 1)
        if occurrence is None:
            occurrence = self.occurrence
        self.db_entry.callstate["kwargs"] = {
            "ids_name": ids_name[0],
            "occurrence": occurrence,
        }
        with self.db_entry as ids:
            if len(ids_name) == 2:
                return getattr(ids, ids_name[1])
            return ids

    @property
    def ids_data(self):
        """Return ids data, lazy load."""
        if self.ids is None:
            self._check_ids_attrs()
            self.ids = self.get_ids()
        return self.ids


@dataclass
class Database(Dataset):
    r"""Methods to access an IMAS Database entry.

    Parameters
    ----------
    filename: str, optional
        Database filename. The default is "".
    group: str | None, optional
        netCDF group. The default is None.

    Notes
    -----
    The Database class regulates access to IMAS IDS data. Requests may be made
    via pulse, run, name identifiers or as direct referances to
    open ids handles.

    Examples
    --------

    """

    filename: str = field(default="", repr=False)
    group: str | None = field(default=None, repr=False)

    '''
    def __post_init__(self):
        """Load parameters and set ids."""
        self.rename()
        self.load_database()
        self.update_filename()
    '''

    def rename(self):
        """Reset name to default if default is not None."""
        if (
            name := next(
                field for field in fields(self) if field.name == "name"
            ).default
        ) is not None:
            self.name = name

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

    @property
    def group_attrs(self):
        """
        Return database attributes.

        Group attrs used by :func:`~nova.database.filepath.FilePath.hash_attrs`
        to generate a unique hex hash to label data within a netCDF file.
        """
        return self.ids_attrs

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

    '''
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
    '''


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
