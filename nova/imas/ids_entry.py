"""Methods to manage sane ids entry."""

from dataclasses import dataclass, field

from nova.imas.database import Database
from nova.imas.dataset import IdsBase, ImasIds
from nova.imas.ids_index import IdsIndex


@dataclass
class IdsEntry(IdsIndex, IdsBase):
    """Methods to facilitate sane ids entry."""

    ids: ImasIds = None
    ids_node: str = ""
    mode: str | None = None
    database: Database | None = field(init=False, default=None)

    def __post_init__(self):
        """Initialize ids and create database instance."""

        if self.ids is None:
            self.ids = self.new_ids()
        self.database = Database(**self.ids_attrs, ids=self.ids, mode=self.mode)
        super().__post_init__()

    def put_ids(self, occurrence=None):
        """Expose Database.put_ids."""
        self.database.put_ids(self.ids, occurrence)
