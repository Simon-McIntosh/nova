"""Update ids properties."""
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from nova.imas.attrs import Attrs


@dataclass
class Properties(Attrs):
    """Manage imas ids_property attributes."""

    comment: str = None
    homogeneous_time: int = 2
    provider: str = None
    provenance_ids: object = None

    attributes: ClassVar[list[str]] = \
        ['comment', 'homogeneous_time', 'provider', 'creation_date',
         'provenance']

    def __call__(self, ids_data: object):
        """Extend Attrs call to include provenance ids."""
        super().__call__(ids_data)
        if self.provenance_ids is not None:
            ids_data.provenance.node.resize(1)
            ids_data.provenance.node[0].sources = self.provenance_ids.code

    @property
    def creation_date(self):
        """Return creation date."""
        return datetime.today().strftime('%d-%m-%Y')
