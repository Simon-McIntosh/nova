"""Manage access to remote data."""
import os
from dataclasses import dataclass, field
from warnings import warn

import pandas.api.types
import ftputil


@dataclass
class FTPData:
    """Manage access to FTP database."""

    _experiment: str
    parent: str = 'Daten'
    server: str = 'ftp.psi.ch'
    username: str = 'sultan'
    password: str = '3g8S4Nbq'
    ftp_args: tuple[str] = field(init=False)

    def __post_init__(self):
        """Assemble ftp arguments."""
        self.ftp_args = (self.server, self.username, self.password)
        self.experiment = self._experiment

    @property
    def experiment(self):
        """Manage experiment attribute."""
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        self._experiment = experiment
        '''
        try:
            experiment = self.locate(experiment, '../')
            self._experiment = experiment
        except FileNotFoundError as file_not_found:
            raise FileNotFoundError() from file_not_found
        '''

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
            try:
                remotefile = next(filename for filename in files
                                  if file in filename)
            except StopIteration as stop_error:
                raise FileNotFoundError(file_not_found_error) from stop_error
        else:
            remotefile = file
        return remotefile

    def download(self, file, directory, *relative_path):
        """
        Download file from ftp server.

        Parameters
        ----------
        file : str
            Full filename.
        directory : str
            Local directory path.
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
        if not os.path.isdir(directory):
            raise FileNotFoundError(f'lLocal directory {directory} not found.')
        remotefile = self.locate(file, *relative_path)
        localfile = os.path.join(directory, remotefile)
        with ftputil.FTPHost(*self.ftp_args) as host:
            self.changedir(host, self.parent, self.experiment, *relative_path)
            try:
                host.download(remotefile, localfile)
            except ftputil.error.PermanentError as file_not_found:
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
                raise FileNotFoundError(
                    f'folder {folder} not found on {host.getcwd()} '
                    f'in {host.listdir("./")}') \
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

    ftp = FTPData('CSJA13')
    #print(ftp.listdir(ftp.experiment, select='.OPJ'))
