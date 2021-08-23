"""Manage Ansys simulation data directories."""
from dataclasses import dataclass
import os

from nova.definitions import root_dir


@dataclass
class DataDir:
    """Manage file paths."""

    folder: str
    file: str = None
    subset: str = 'all'
    data_dir: str = 'data/Ansys'
    rst_dir: str = '//io-ws-ccstore1/ANSYS_Data/mcintos'

    def __post_init__(self):
        """Set data directory."""
        self.directory = os.path.join(root_dir, self.data_dir, self.folder)
        self.mkdir(self.directory)

    def mkdir(self, path):
        """Create dir if not present."""
        if not os.path.isdir(path):
            os.mkdir(path)

    @property
    def rst_folder(self):
        """Return rst folder."""
        return os.path.join(root_dir, self.rst_dir, self.folder)

    @property
    def rst_file(self):
        """Return rst file path."""
        return os.path.join(self.rst_folder, f'{self.file}.rst')

    @property
    def ansys_folder(self):
        """Return path of Ansys part vtk folder."""
        ansys_folder = os.path.join(self.directory, 'ansys')
        self.mkdir(ansys_folder)
        return ansys_folder

    @property
    def ansys_file(self):
        """Return vtk file path."""
        if self.subset == 'all':
            return os.path.join(self.ansys_folder, f'{self.file}.vtk')
        return os.path.join(self.ansys_folder,
                            f'{self.file}_{self.subset.lower()}.vtk')

    @property
    def ccl_folder(self):
        """Return path of conductor centerline vtk folder."""
        ccl_folder = os.path.join(self.directory, 'ccl')
        self.mkdir(ccl_folder)
        return ccl_folder

    @property
    def ccl_file(self):
        """Return ccl file path."""
        postfix = 'ccl' if self.cluster is None else 'uccl'
        return os.path.join(self.ccl_folder, f'{self.file}_{postfix}.vtk')

    @property
    def args(self):
        """Return data dir args."""
        return self.folder, self.file, self.subset, self.data_dir
