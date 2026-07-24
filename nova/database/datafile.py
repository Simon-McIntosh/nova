"""Manage access to netCDF data."""

from abc import abstractmethod
from dataclasses import dataclass

from nova.database.netcdf import netCDF


@dataclass
class Datafile(netCDF):
    """
    Provide cached acces to netCDF data.

    Extends netCDF class via the provision of load and store methods.

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
