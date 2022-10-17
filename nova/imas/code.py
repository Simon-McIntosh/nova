"""Manage nova code metadata."""
from dataclasses import dataclass, field
from typing import ClassVar
import git
import pandas

import nova
from nova.imas.attrs import Attrs


@dataclass
class Code(Attrs):
    """Methods for retriving and formating code metadata."""

    parameter_dict: dict | None = field(default=None, repr=False)
    output_flag: int = 1

    name: ClassVar[str] = 'Nova'
    attributes: ClassVar[list[str]] = \
        ['name', 'commit', 'version', 'repository', 'parameters',
         'output_flag']

    def __post_init__(self):
        """Load git repository."""
        self.repo = git.Repo(search_parent_directories=True)

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


if __name__ == '__main__':

    code = Code()
    code.parameters = dict(dcoil=30, plasma=-50)

    print(code.parameters)
