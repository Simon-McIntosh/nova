"""Update ids metadata."""
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import git
import numpy as np
import pandas

import nova
from nova.imas.database import Ids


@dataclass
class Attrs(ABC):
    """Provide IdsData baseclass."""

    attributes: ClassVar[list[str]] = []

    def update(self, ids: object):
        """Update code metadata."""
        for attr in self.attributes:
            try:
                attribute = getattr(self, attr)
            except AttributeError:
                continue
            if attribute is None:
                continue
            setattr(ids, attr, attribute)


@dataclass
class Properties(Attrs):
    """Manage imas ids_property attributes."""

    comment: str | None = None
    source: str | None = None
    homogeneous_time: int = 1
    provider: str | None = 'Simon McIntosh, simon.mcintosh@iter.org'
    provenance: list[str] | None = None

    attributes: ClassVar[list[str]] = \
        ['comment', 'homogeneous_time', 'source', 'provider', 'creation_date']

    def update(self, ids):
        """Extend Attrs update to include provenance ids."""
        super().update(ids)
        if self.provenance is not None:
            ids.provenance.node.resize(1)
            ids.provenance.node[0].sources = self.provenance

    @property
    def creation_date(self):
        """Return creation date."""
        return datetime.today().strftime('%d-%m-%Y')


@dataclass
class Code(Attrs):
    """Methods for retriving and formating code metadata."""

    parameter_dict: dict | None = None
    output_flag: list[int] | np.ndarray | None = None

    name: ClassVar[str] = 'Nova'
    attributes: ClassVar[list[str]] = \
        ['name', 'commit', 'version', 'repository',
         'parameters', 'output_flag']

    def __post_init__(self):
        """Load git repository."""
        self.repo = git.Repo(search_parent_directories=True)
        if self.output_flag is not None:
            self.output_flag = np.array(self.output_flag, int)

    @property
    def parameters(self):
        """Return code parameters as HTML table."""
        if self.parameter_dict is None:
            return None
        return pandas.Series(self.parameter_dict).to_frame().to_html()

    @parameters.setter
    def parameters(self, parameters: dict):
        """Update code parameters."""
        self.parameter_dict = parameters

    @property
    def commit(self):
        """Return git commit hash."""
        return self.repo.head.object.hexsha

    @property
    def version(self):
        """Return code version."""
        return nova.__version__

    @property
    def repository(self):
        """Return repository url."""
        return self.repo.remotes.origin.url


@dataclass
class Metadata:
    """Write ids metadata."""

    ids: Ids

    def put_properties(self, comment, source=None, *args, **kwargs):
        """Update ids_properties."""
        properties = Properties(comment, source, *args, **kwargs)
        properties.update(self.ids.ids_properties)

    def put_code(self, parameter_dict=None, output_flag=None):
        """Update referances to Nova code."""
        code = Code(parameter_dict, output_flag)
        code.update(self.ids.code)


if __name__ == '__main__':

    import imas
    ids = imas.equilibrium()

    metadata = Metadata(ids)
    metadata.put_properties('Equilibrium extrapolation', 'Yuri\'s database',
                            provider='Simon McIntosh')
    metadata.put_code('test code attribute update')
