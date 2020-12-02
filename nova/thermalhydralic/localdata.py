"""Manage local data directories."""
import os
from dataclasses import dataclass

from nova.definitions import root_dir


@dataclass
class LocalData:
    """
    Manage local data directories.

    Methods implemented for group creation and deletion of directory structure.
    """

    experiment: str
    parent: str = ''
    data: str = 'data'
    post: str = 'post'

    def __post_init__(self):
        """Create hierarchical list used by makedir and removedir methods."""
        self._directories = [self.experiment_directory,
                             self.data_directory,
                             self.post_directory]

    @property
    def parent_directory(self):
        """Return full path to parent directory."""
        return self.getdir(os.path.join(root_dir, 'data'), self.parent)

    @property
    def experiment_directory(self):
        """Return full path to experiment directory."""
        return self.getdir(self.parent_directory, self.experiment)

    @property
    def data_directory(self):
        """Return full path to data directory."""
        return self.getdir(self.experiment_directory, self.data)

    @property
    def post_directory(self):
        """Return full path to data directory."""
        return self.getdir(self.experiment_directory, self.post)

    @staticmethod
    def getdir(directory, subfolder=''):
        """Return directory, append subfolder if passed."""
        if subfolder:
            directory = os.path.join(directory, subfolder)
        return directory

    @staticmethod
    def _mkdir(directory):
        if not os.path.isdir(directory):
            os.mkdir(directory)

    def makedir(self):
        """Create physical directories."""
        for directory in self._directories:
            self._mkdir(directory)

    def removedir(self):
        """Remove experiment and sub directories."""
        for directory in self._directories[::-1]:
            os.rmdir(directory)


if __name__ == '__main__':

    local = LocalData('CS1', 'Sultan')
    local.makedir()
    local.removedir()
    print(local.experiment)
