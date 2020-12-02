"""Manage access to remote data."""
import os
from dataclasses import dataclass
import glob
from warnings import warn

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

    def locate(self, file, subdir=None):
        """
        Locate file on local host. If not found, download from ftp server.

        Parameters
        ----------
        file : str
            filename.
        subdir : str, optional
            subdir on ftp server, see _download. The default is None.

        Raises
        ------
        IndexError
            Evaluation of filename wild card returns multiple files.

        Returns
        -------
        localfile : str
            Full path of local file.

        """
        file = os.path.join(self.local.data_directory, file)
        localfile = []
        if '*' in file:
            localfile = glob.glob(file)
            if len(localfile) > 1:
                raise IndexError(f'multiple files found {file} > {localfile}')
        else:
            if os.path.isfile(file):
                localfile = [file]
        if localfile:
            localfile = os.path.split(localfile[0])[1]
        else:
            localfile = self.download(file, subdir=subdir)
        return localfile

    def download(self, file, subdir=None):
        """
        Download file from ftp server.

        Parameters
        ----------
        file : str
            Filename, names of type '*.ext' permited.
        subdir : str, optional
            subdirectory, evaluated as parentdir/experiment/subdir.
            The default is None.

        Raises
        ------
        ftputil
            File not found.
        IndexError
            Evaluation of filename wild card returns multiple files.

        Returns
        -------
        file : str
            Full filename.

        """
        with ftputil.FTPHost(self.server, self.username, self.password) as host:
            chdir = [self.parent, self.local.experiment, subdir]
            for cd in chdir:
                if cd:
                    try:
                        host.chdir(f'./{cd}')
                    except ftputil.error.PermanentError as file_not_found:
                        pwd = host.listdir('./')
                        raise ftputil.error.PermanentError(
                            f'folder {cd} not found in {pwd}') \
                            from file_not_found
            if '*' in file:
                ext = os.path.split(file)[1].split('*')[-1]
                ftpfile = [f for f in host.listdir('./') if ext in f]
                if len(ftpfile) > 1:
                    warn_txt = f'multiple files found {file} > {ftpfile}'
                    warn_txt += f'\nusing {ftpfile[0]}'
                    warn(warn_txt)
                remotefile = ftpfile[0]
            else:
                remotefile = os.path.split(file)[1]
            try:
                file = os.path.join(self.local.data_directory, remotefile)
                host.download(remotefile, file)
            except ftputil.error.FTPError:
                err_txt = f'file {remotefile} '
                err_txt += f'not found in {host.listdir("./")}'
                raise ftputil.error.FTPError(err_txt)
        return file

    def listdir(self, substr='', experiment='', subdir=''):
        """
        Return file/directory list.

        Parameters
        ----------
        substr : str, optional
            File select substr. The default is ''.
        experiment : str, optional
            First sub-directory. The default is None.
        subdir : str, optional
            Second sub-directory. The default is None.

        Returns
        -------
        ls : array-like
            List of file/directory names.

        """
        chdir = [self.parent, experiment, subdir]
        with ftputil.FTPHost('ftp.psi.ch', 'sultan', '3g8S4Nbq') as host:
            for cd in chdir:
                if cd:
                    host.chdir(f'./{cd}')
            ls = host.listdir('./')
        if substr:
            ls = [file for file in ls if substr in file]
        return ls

if __name__ == '__main__':

    local = LocalData('CS3U')