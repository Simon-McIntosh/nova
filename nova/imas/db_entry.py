"""Manage access to IMAS database via DBEntry class."""

from dataclasses import dataclass, field
from functools import cached_property

from nova.imas.callstate import Callstate
from nova.imas.dataset import Ids
from nova.utilities.importmanager import check_import

with check_import("imaspy"):
    import imaspy as imas


@dataclass
class DBEntry(imas.DBEntry):
    """Extend imas.DBEntry to provide context aware access to ids data."""

    uri: str
    mode: str
    dd_version: str | None = None
    xml_path: str | None = None
    ids: Ids = field(repr=False, default=None)
    callstate: Callstate = field(init=False, repr=False, default_factory=Callstate)

    def __post_init__(self):
        """Extend imaspy and restrict operation to AL5."""
        super().__init__(
            self.uri, self.mode, dd_version=self.dd_version, xml_path=self.xml_path
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
            return self.get(*args, lazy=False, **kwargs)
        self.callstate(args, kwargs)
        return self

    @property
    def is_valid(self):
        """Retrun True if ids_properties.homogeneous_time is set else False."""
        try:
            self.get(self.name, self.occurrence, lazy=True)
            return True
        except RuntimeError:
            return False

    def __enter__(self):
        """Return open ids. Extend imaspy DBEntry.enter."""
        super().__enter__()
        return self.get(
            *self.callstate.args,
            lazy=True,
            **self.callstate.kwargs,
        )

    def close(self):
        """Extend imaspy.db_entry.close. Clear callstate before exit."""
        self.callstate.clear()
        super().close()

    def _get_data(self, *args, lazy=False, **kwargs):
        """Return ids data from db_entry.get."""
        self.ids = self.get(self.name, self.occurrence, *args, lazy=lazy, **kwargs)
        return self.ids
