"""Manage cached access to the identity-cache data store."""

from abc import abstractmethod
from dataclasses import dataclass

from nova.database.zarrstore import ZarrStore


@dataclass
class Datafile(ZarrStore):
    """
    Provide cached access to the identity-cache data store.

    The identity cache -- compiled frames and solved operators keyed by a
    content hash of the source identity -- persists through zarr, whose native
    group eviction avoids the whole-file rewrite the netCDF backend requires.
    netCDF remains available through :class:`~nova.database.netcdf.netCDF` for
    interchange and export. Extends the store with load and build methods.

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
        except FileNotFoundError, OSError:
            # A single-IDS source populates self.ids via get() before build;
            # a composite (a machine description assembled from several sources)
            # carries no name of its own and builds directly from its sources.
            if self.ids is None and self.name is not None:
                self.get()
            self.build()
            self.store()

    @abstractmethod
    def build(self):
        """Build netCDF dataset."""
