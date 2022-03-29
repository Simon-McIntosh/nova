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
    parent: str = 'Sultan'
    source: str = 'ftp'
    binary: str = 'local'

    def __post_init__(self):
        """Extend utilities.localdata.LocalData."""
        super().__post_init__()


if __name__ == '__main__':
    local = LocalData('CSJA12', '')
    # print(local.locate('*.xls'))
