"""Methods to manage sane ids entry."""

from dataclasses import dataclass, field

from nova.imas.database import Database
from nova.imas.dataset import IdsBase
from nova.imas.ids_index import IdsIndex


@dataclass
class IdsEntry(IdsIndex, IdsBase):
    """Methods to facilitate sane ids entry."""

    mode: str | None = None
    database: Database | None = field(init=False, default=None)
    # lazy: bool = False

    def __post_init__(self):
        """Initialize ids and create database instance."""
        if self._ids is None:
            self.ids = self.new_ids()
        self.database = Database(**self.ids_attrs, ids=self._ids, mode=self.mode)

    def put_ids(self, occurrence=None):
        """Expose Database.put_ids."""
        self.database.put_ids(self.ids, occurrence)
