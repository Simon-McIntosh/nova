"""Manage local data directories."""
from dataclasses import dataclass

from nova.utilities.localdata import LocalData


@dataclass
class LocalData(LocalData):
    """
    Manage local data directories.

    Methods implemented for group creation and deletion of directory structure.
    """

    experiment: str
    parent_dir: str = 'Naka'
    source_dir: str = 'source'
    binary_dir: str = 'local'
    metadata_dir: str = 'metadata'

    def __post_init__(self):
        """Extend utilities.localdata.LocalData."""
        super().__init__()
        self._directories.append(self.metadata_directory)

    @property
    def metadata_directory(self):
        """Return full path to metadata directory."""
        return self.getdir(self.experiment_directory, self.metadata_dir)


if __name__ == '__main__':
    local = LocalData('MRun028_SRun001')
