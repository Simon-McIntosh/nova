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
    parent_dir: str = 'Sultan'
    source_dir: str = 'ftp'
    binary_dir: str = 'local'

    def __post_init__(self):
        """Extend utilities.localdata.LocalData."""
        super().__init__()

if __name__ == '__main__':
    local = LocalData('CSJA12', 'Sultan')
    #print(local.locate('*.xls'))
