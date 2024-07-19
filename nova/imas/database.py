"""Manage access to IMAS database."""

from dataclasses import dataclass, field, fields
from functools import cached_property
import packaging

from nova.database.datafile import Datafile
from nova.imas.dataset import Dataset


# _pylint: disable=too-many-ancestors


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

    """

    filename: str = field(default="", repr=False)
    group: str | None = field(default=None, repr=False)

    def __post_init__(self):
        """Load parameters and set ids."""
        self.rename()
        self.load_database()
        self.update_filename()
        super().__post_init__()

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
        version_put = self.ids.ids_properties.version_put.data_dictionary.value
        return packaging.version.parse(version_put.split("-")[0])

    def load_database(self):
        """Load instance database attributes."""
        if self._ids is not None:
            return self._load_attrs_from_ids()
        return None

    @property
    def classname(self):
        """Return base filename."""
        classname = f"{self.__class__.__name__.lower()}".replace("data", "")
        classname = classname.replace("poloidalfield", "pf_")
        # if classname == self.name:
        #    return self.machine
        return f"{classname}_{self.machine}"

    def update_filename(self):
        """Update filename."""
        if self.filename == "":
            self.filename = self.classname
            if self.pulse is not None and self.pulse > 0 and self.run is not None:
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


@dataclass
class IdsData(Datafile, Database):
    """Provide cached acces to imas ids data."""

    dirname: str = ".nova.imas"

    def merge_data(self, data):
        """Merge external data, interpolating to existing dataset timebase."""
        self.data = self.data.merge(
            data.interp(time=self.data.time), combine_attrs="drop_conflicts"
        )

    def load_data(self, ids_class, **ids_attrs):
        """Load data from IdsClass and merge."""
        if self.pulse == 0 and self.run == 0 and self.ids is None:
            return
        if self.ids is not None:
            ids_attrs = {"ids": self.ids}
        else:
            ids_attrs = self.ids_attrs | ids_attrs
        try:
            data = ids_class(**ids_attrs).data
        except NameError:  # name missmatch when loading from ids node
            return
        if self.ids is not None:  # override when using ids input
            self.data = data
            return

        if hasattr(self.data, "time") and hasattr(data, "time"):
            data = data.interp({"time": self.data.time}, assume_sorted=True)

        print("pre merge")
        self.data = data.merge(
            self.data, compat="override", combine_attrs="drop_conflicts"
        )
        print("post merge")


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
        if self.group is None:
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


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=False)
