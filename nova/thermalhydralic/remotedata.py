"""Manage access to remote data."""
import os
from dataclasses import dataclass
from warnings import warn
import pandas.api.types

import ftputil

from nova.thermalhydralic.localdata import LocalData


@dataclass
class FTPData:
    """Manage access to FTP database."""

    local: LocalData
    parent: str = 'Daten'
    server: str = 'ftp.psi.ch'
    username: str = 'sultan'
    password: str = '3g8S4Nbq'

    def __post_init__(self):
        """Initialize localdata and assemble ftp arguments."""
        if not isinstance(self.local, LocalData):
            if not pandas.api.types.is_list_like(self.local):
                self.local = [self.local]
            self.local = LocalData(*self.local)
        self.experiment = self.local.experiment
        self.ftp_args = (self.server, self.username, self.password)

    def locate(self, file, *relative_path):
        """
        Locate file no ftp server.

        Parameters
        ----------
        file : str
            Filename, names of type '*.ext' permited.
        *relative_path : str, optional
            relative path below parent/experiment/.

        Raises
        ------
        FileNotFoundError
            File not found on ftp server.

        Returns
        -------
        remotefile : str
            Valid full remote filename (no wildcards).

        """
        with ftputil.FTPHost(*self.ftp_args) as host:
            self.changedir(host, self.parent, self.experiment, *relative_path)
            files = host.listdir('./')
        file_not_found_error = f'file {file} not found in {files}'
        if '*' in file:
            ext = file.split('*')[-1]
            ftpfile = [f for f in files if ext in f]
            nmatch = len(ftpfile)
            if nmatch == 0:
                raise FileNotFoundError(file_not_found_error)
            if nmatch > 1:
                warn_txt = f'multiple files found {file} > {ftpfile}'
                warn_txt += f'\nusing {ftpfile[0]}'
                warn(warn_txt)
            remotefile = ftpfile[0]
        elif file not in files:
            raise FileNotFoundError(file_not_found_error)
        else:
            remotefile = file
        return remotefile

    def download(self, file, *relative_path):
        """
        Download file from ftp server.

        Parameters
        ----------
        file : str
            Full filename.
        *relative_path : str, optional
            relative path below parent/experiment/.

        Raises
        ------
        ftputil
            File not found.
        IndexError
            Evaluation of filename wild card returns multiple files.

        Returns
        -------
        None.

        """
        remotefile = self.locate(file, *relative_path)
        isdir = self.local.checkdir()
        if not isdir:
            self.local.makedir()  # generate local file strucutre if required
        localfile = os.path.join(self.local.source_directory, remotefile)
        with ftputil.FTPHost(*self.ftp_args) as host:
            self.changedir(host, self.parent, self.experiment, *relative_path)
            try:
                host.download(remotefile, localfile)
            except ftputil.error.PermanentError as file_not_found:
                if not isdir:
                    self.local.removedir()  # remove if generated bare
                raise FileNotFoundError(
                    f'file {file} not found in {host.listdir("./")}') \
                    from file_not_found

    @staticmethod
    def changedir(host, *relative_path):
        """
        Change directory on host directed by relative_path.

        Parameters
        ----------
        host : stputil.FTPHost
            Host instance.
        *relative_path : str or array-like
            Relative path.

        Raises
        ------
        ftputil
            Folder not found on ftp server.

        Returns
        -------
        None.

        """
        if not pandas.api.types.is_list_like(relative_path):
            relative_path = [relative_path]
        folders = [folder for folder in relative_path if folder]
        for folder in folders:
            try:
                host.chdir(f'./{folder}')
            except ftputil.error.PermanentError as file_not_found:
                pwd = host.listdir('./')
                raise FileNotFoundError(
                    f'folder {folder} not found in {pwd}') \
                    from file_not_found

    def listdir(self, *relative_path, select=''):
        """
        Return file/directory list.

        Parameters
        ----------
        *relative_path : str or array-like
            Relative path below parent directory.
        select : str, optional
            File select sub string. The default is '' (match all).

        Returns
        -------
        names : array-like
            List of file/directory names.

        """
        relative_path = [self.parent, *list(relative_path)]
        with ftputil.FTPHost(*self.ftp_args) as host:
            self.changedir(host, *relative_path)
            names = host.listdir('./')
        if select:
            names = [file for file in names if select in file]
        return names


if __name__ == '__main__':

    ftp = FTPData('CSJA_3')
    print(ftp.listdir(ftp.experiment, select='.OPJ'))
