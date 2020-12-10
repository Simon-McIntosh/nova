"""Manage local data directories."""
import os
import glob
from dataclasses import dataclass

import numpy as np

from nova.definitions import root_dir


@dataclass
class LocalData:
    """
    Manage local data directories.

    Methods implemented for group creation and deletion of directory structure.
    """

    experiment: str
    binary: bool = True
    parent_dir: str = 'Sultan'
    source_dir: str = 'ftp'
    binary_dir: str = 'local'

    def __post_init__(self):
        """Create hierarchical list used by makedir and removedir methods."""
        self._directories = [self.experiment_directory,
                             self.source_directory,
                             self.binary_directory]

    @property
    def parent_directory(self):
        """Return full path to parent directory."""
        return self.getdir(os.path.join(root_dir, 'data'), self.parent_dir)

    @property
    def experiment_directory(self):
        """Return full path to experiment directory."""
        return self.getdir(self.parent_directory, self.experiment)

    @property
    def source_directory(self):
        """Return full path to raw data directory."""
        return self.getdir(self.experiment_directory, self.source_dir)

    @property
    def binary_directory(self):
        """Return full path to binary data directory."""
        return self.getdir(self.experiment_directory, self.binary_dir)

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

    def checkdir(self):
        """Return booliean status of all tracked local directories."""
        isdir = np.full(len(self._directories), False)
        for i, directory in enumerate(self._directories):
            isdir[i] = os.path.isdir(directory)
        return isdir.all()

    def makedir(self):
        """Create physical directories."""
        for directory in self._directories:
            self._mkdir(directory)

    def removedir(self):
        """Remove experiment and sub directories."""
        for directory in self._directories[::-1]:
            os.rmdir(directory)

    def locate(self, file, directory_prefix='source'):
        """
        Locate file on local host.

        Parameters
        ----------
        file : str
            Filename.
        directory_prefix : str
            Directory label, evaluated as f'{directory_prefix}_directory'.

        Raises
        ------
        IndexError
            Evaluation of filename wild card returns multiple files.
        AttributeError
            f'{directory_prefix}_directory' undefined.

        Returns
        -------
        localfile : str
            Full path of local file.

        """
        try:
            directory = getattr(self, f'{directory_prefix}_directory')
        except AttributeError as error:
            raise AttributeError('directory prefix undefined '
                                 f'{directory_prefix}') from error
        filepath = os.path.join(directory, file)
        localfile = ''
        if '*' in file:
            localfile = glob.glob(filepath)
            if len(localfile) == 0:
                raise FileNotFoundError(f'No files found matching {file}')
            if len(localfile) > 1:
                raise IndexError(f'multiple files found {file} > {localfile}')
            localfile = os.path.split(localfile[0])[1]
        else:
            if os.path.isfile(filepath):
                localfile = file
            else:
                raise FileNotFoundError(f'File {file} not found')
        return localfile


if __name__ == '__main__':

    local = LocalData('CSJA_3', 'Sultan')
    print(local.locate('*.xls'))
