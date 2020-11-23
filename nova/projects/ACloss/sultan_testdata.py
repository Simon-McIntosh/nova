
import os
import glob
import re

import ftputil
import pandas
import numpy as np

from nova.definitions import root_dir

class FTPSultan:

    def __init__(self, testID):
        self.testID = testID
        self.data_dir = os.path.join(root_dir, f'data/Sultan/{self.testID}')
        if not os.path.isdir(self.data_dir):
            os.mkdir(self.data_dir)

    def locate(self, file, parent_dir='Daten'):
        file = os.path.join(self.data_dir, file)
        local_file = []
        if '*' in file:
            local_file = glob.glob(file)
            if len(local_file) > 1:
                raise IndexError(f'multiple files found {file} > {local_file}')
        else:
            if os.path.isfile(file):
                local_file = [file]
        if local_file:
            localfile = os.path.split(local_file[0])[1]
        else:
            localfile = self._download(file, parent_dir=parent_dir)
        return localfile

    def _download(self, file, parentdir='Daten', subdir=None):
        with ftputil.FTPHost('ftp.psi.ch', 'sultan', '3g8S4Nbq') as host:
            try:
                host.chdir(f'./{parentdir}/{self.testID}')
            except ftputil.error.PermanentError:
                ls = host.listdir(f'./{parentdir}')
                raise ftputil.error.PermanentError(
                    f'testID folder {self.testID} not found in {ls}')
            if '*' in file:
                ext = os.path.split(file)[1].split('*')[-1]
                ftpfile = [f for f in host.listdir('./') if ext in f]
                if len(ftpfile) > 1:
                    err_txt = f'multiple files found {file} > {ftpfile}'
                    raise IndexError(err_txt)
                remotefile = ftpfile[0]
            else:
                remotefile = os.path.split(file)[1]
            try:
                host.download(remotefile, file)
            except ftputil.error.FTPError:
                err_txt = f'file {remotefile} '
                err_txt += f'not found in {host.listdir("./")}'
                raise ftputil.error.FTPError(err_txt)
        return file

    def load_testplan(self):
        testplan = os.path.join(self.data_dir, self.locate('*.xls'))
        with pandas.ExcelFile(testplan) as xls:
            index = pandas.read_excel(xls, usecols=[0], header=None)
        self.name = index[0][0]
        self.index = {}
        previouslabel = None
        for i, label in enumerate(index.values):
            if isinstance(label[0], str):
                if label[0][:2] in ['AC', 'DC']:
                    self.index[label[0]] = [i+1]
                    if previouslabel is not None:
                        self.index[previouslabel].append(i-1)
                    previouslabel = label[0]
        if len(self.index) > 0:
            self.index[previouslabel].append(i)
        self.data = {}
        self.note = {}
        previouscolumns = None
        with pandas.ExcelFile(testplan) as xls:
            for label in self.index:
                index = self.index[label]
                df = pandas.read_excel(
                    xls, skiprows=index[0],
                    nrows=np.diff(index)[0], header=None)
                if df.iloc[0, 0] == 'File':
                    df.columns = pandas.MultiIndex.from_arrays(
                        df.iloc[:2].values)
                    df = df.iloc[2:]
                    previouscolumns = df.columns
                elif previouscolumns is not None:
                    df.columns = previouscolumns
                # extract note
                note = [c for c in df.columns.get_level_values(0) if
                        'note' in c.lower()]
                if note:
                    self.note[label] = df[note[0]]
                    df.drop(columns=note, inplace=True, level=0)
                df.fillna(method='pad', inplace=True)
                self.data[label] = df.reset_index()

    def load_file(self, label, index):
        file = self.data[label].loc[index, 'File'][0]
        self.locate(file)
        #print(file)



if __name__ == '__main__':

    ftps = FTPSultan('CSJA_7')
    ftps.load_testplan()

    ftps.load_file('AC Loss Initial', 1)