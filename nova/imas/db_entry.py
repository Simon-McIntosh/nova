"""Manage access to IMAS database via DBEntry class."""
from dataclasses import dataclass, field
from functools import cached_property

from nova.imas.callstate import Callstate
from nova.utilities.importmanager import check_import

with check_import("imaspy"):
    import imaspy


@dataclass
class DBEntry(imaspy.DBEntry):
    """Extend imaspy.DBEntry to provide context aware access to ids data."""

    uri: str
    mode: str
    database: None = None
    dd_version: str | None = None
    xml_path: str | None = None
    callstate: Callstate = field(init=False, repr=False, default_factory=Callstate)

    def __post_init__(self):
        """Extend imaspy and restrict to AL5."""
        super().__init__(
            self.uri, self.mode, dd_version=self.dd_version, xml_path=self.dd_version
        )

    @cached_property
    def idsname(self):
        """Return idsname."""
        return self.uri.split("#")[1].split(":")[0].split("=")[1]

    def __call__(self, *args, lazy=False, **kwargs):
        """
        Implement interface to db_entry get.

        Parameters
        ----------
        lazy: {True, False}, optional
            Lazy read flag.

            - True: Return self. Use within 'with'. Update callstate and return self.
            - False: Return full ids and store on self.ids. Close db_entry.
        """
        if lazy is False:
            return self.get_data(*args, lazy=False, **kwargs)
        self.callstate(args, kwargs)
        return self

    def get(self, *args, **kwargs):
        """Return data from db_entry."""
        if "time_requested" in kwargs:
            return super().get_slice(*args, **kwargs)
        return super().get(*args, **kwargs)

    @cached_property
    def is_valid(self):
        """Retrun True if ids_properties.homogeneous_time is set else False."""
        try:
            self.get_data(lazy=True)
            return True
        except RuntimeError:
            return False

    def __enter__(self):
        """Extend imaspy DBEntry.enter."""
        super().__enter__()
        return self.get(
            self.idsname,
            *self.callstate.args,
            lazy=True,
            **self.callstate.kwargs,
        )

    def close(self):
        """Extend imaspy.db_entry.close. Clear callstate before exit."""
        self.callstate.clear()
        super().close()

    def get_data(self, *args, lazy=False, **kwargs):
        """Return ids data from db_entry.get."""
        self.database.ids = self.get(self.idsname, *args, lazy=lazy, **kwargs)
        return self.database.ids
