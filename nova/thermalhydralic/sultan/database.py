"""Manage Sultan database. Unify local and remote data access."""
import os.path
from dataclasses import dataclass, field

from nova.thermalhydralic.sultan.localdata import LocalData
from nova.thermalhydralic.sultan.remotedata import FTPData


@dataclass
class DataBase:
    """
    Manage local and remote data soruces.

    Parameters
    ----------
    experiment : str
        Experiment label
    local_args : array-like
        Argument list passed.
    ftp : FTPData, optional
        Remote data instance. The default is None.

    """

    _experiment: str
    _local_args: list[str] = field(default_factory=list, repr=False)
    _ftp_args: list[str] = field(default_factory=list, repr=False)
    local: LocalData = field(init=False, repr=False)
    ftp: FTPData = field(init=False, repr=False)
    datapath: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self):
        """Initialize local and ftp data instances."""
        if not self.datapath:
            self.datapath = ['ac/dat', 'AC/dat', 'TEST/AC/ACdat']
        self.experiment = self._experiment

    @property
    def experiment(self):
        """Manage sultan experiment name."""
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        self._experiment = experiment
        self.ftp = FTPData(self.experiment, *self.ftp_args)
        self.local = LocalData(self.experiment, *self.local_args)

    def datafile(self, filename):
        """Return full local path of datafile."""
        try:  # local search
            datafile = self.local.locate(f'{filename}.dat')
            return self.source_filepath(f'{filename}.dat')
        except FileNotFoundError:  # ftp search
            for relative_path in self.datapath:
                try:
                    datafile = self.locate(filename, relative_path)
                    break
                except FileNotFoundError as file_not_found:
                    file_not_found_error = file_not_found
        try:
            return self.source_filepath(datafile)
        except AttributeError:
            err_txt = f'datafile {filename} '
            err_txt += f'not found on datapath {self.datapath}'
            raise FileNotFoundError(err_txt) from file_not_found_error

    @property
    def local_args(self):
        """Return local args, read-only."""
        return self._local_args

    @property
    def ftp_args(self):
        """Return ftp args, read-only."""
        return self._ftp_args

    def locate(self, file, *relative_path):
        """
        Return full filename of file mached on ftp server.

        Parameters
        ----------
        file : str
            Filename, names of type '*.ext' permited.

        Returns
        -------
        file : str
            Full filename.

        """
        filename = self.ftp.locate(file, *relative_path)
        makedir = ~self.local.checkdir()  # generate structure if requred
        if makedir:
            self.local.makedir()
        try:
            self.ftp.download(filename, self.local.source_directory,
                              *relative_path)
        except FileNotFoundError as file_not_found:
            if makedir:
                self.local.removedir()  # remove if generated bare
            raise FileNotFoundError(f'File {filename} not found on '
                                    'ftp server') from file_not_found
        return self.source_filepath(filename)

    def binary_filepath(self, filename):
        """Return binary filepath."""
        return os.path.join(self.local.data_directory, filename)

    def source_filepath(self, filename):
        """Return source filepath."""
        return os.path.join(self.local.source_directory, filename)
