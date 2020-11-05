import os.path
from subprocess import Popen, PIPE
from shlex import split
import pickle
import datetime
import hashlib

import numpy as np
from pandas import DataFrame, MultiIndex
import xarray as xr
import netCDF4 as nc4


def human_format(number, precision=2):
    # adapté d'https://stackoverflow.com/questions/45310254/
    # fixed-digits-after-decimal-with-f-strings
    magnitude = 0
    txt_type = 'f'
    while abs(number) >= 1000 or round(abs(number), precision) >= 1000:
        magnitude += 1
        number = round(number / 1000, precision)
    magnitude = ['', 'K', 'M', 'G', 'T', 'P'][magnitude]
    number = round(number, precision)
    value = '.{}{}'.format(precision, txt_type)
    return '{:{value}}{}'.format(number, magnitude, value=value)


class pythonIO:

    _illegal = [('/', '_DIV_'), ('<','_LR_'), ('>','_RR_'),
                ('(','_OB_'), (')','_CB_')]  # illegal netCDF symbols

    def mkdir(self, filepath):
        filedir = os.path.dirname(filepath)
        if not os.path.isdir(filedir):  # make dir
            os.mkdir(filedir)

    def save_pickle(self, filepath, attributes):
        self.mkdir(filepath)
        protocol = pickle.HIGHEST_PROTOCOL
        with open(filepath + '.pk', mode='wb') as f:
            pickle.dump(attributes, f, protocol=protocol)
            for attribute in attributes:
                data = getattr(self, attribute)
                pickle.dump(data, f, protocol=protocol)

    def load_pickle(self, filepath):
        with open(filepath + '.pk', mode='rb') as f:
            try:
                attributes = pickle.load(f)
                for attribute in attributes:
                    setattr(self, attribute, pickle.load(f))
            except ValueError:
                return True

    def netCDF_columns(self, columns, decode=False):
        for illegal in self._illegal:
            if decode:
                illegal = illegal[::-1]
            columns = [c.replace(*illegal) for c in columns]
        return columns

    def save_netCDF(self, filepath, attributes, **metadata):
        'convert list of dataframes (attributes) to xarrays and save as NetCDF'
        self.mkdir(filepath)
        with nc4.Dataset(filepath+'.nc', 'w', format='NETCDF4') as dataset:
            for md in metadata:
                print(md)
                setattr(dataset, md, metadata[md])
            for i, attribute in enumerate(attributes):
                dataframe = getattr(self, attribute).copy()
                if not isinstance(dataframe, DataFrame):
                    raise TypeError(f'attribute {type(attribute)} not DataFrame')
                # compress multi-index
                if isinstance(dataframe.columns, MultiIndex):
                    dataframe.columns = \
                        ['|'.join(c) for c in dataframe.columns.values]
                    attribute += '.mi'
                # encode columns
                dataframe.columns = self.netCDF_columns(dataframe.columns)
                dataframe.to_xarray().to_netcdf(
                    filepath+'.nc', mode='a', format='NETCDF4',
                    group=attribute)

    def load_netCDF(self, filepath):
        with nc4.Dataset(filepath+'.nc', 'r') as dataset:
            print(dataset)
            attributes = dataset.groups.keys()  # read dataset groups
        for attribute in attributes:
            with xr.open_dataset(filepath+'.nc', group=attribute) as xarray:
                dataframe = xarray.to_dataframe()
            # decode columns
            dataframe.columns = self.netCDF_columns(dataframe.columns,
                                                    decode=True)
            if attribute[-3:] == '.mi':  # re-create multi-index
                dataframe.columns = MultiIndex.from_tuples(
                    [(c.split('|')) for c in dataframe.columns],
                    names=('name', 'unit'))
                attribute = attribute[:-3]
            setattr(self, attribute, dataframe)

    @staticmethod
    def hash_file(file, algorithm='sha256'):
        secure_hash = getattr(hashlib, algorithm)()
        with open(file, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                secure_hash.update(chunk)
        return secure_hash.hexdigest()


def read_block(txt, string=False):
    # read block of text without labels
    txt = txt.split('\n')
    nC = len(txt[-2].split())
    nR = len(txt)-2
    if string:
        data = [[[] for _ in range(nC)] for _ in range(nR)]
    else:
        data = np.zeros((nR, nC))
    for i, line in enumerate(txt[1:-1]):
        ls = line.split()
        if string:
            for j, k in enumerate(ls):
                try:
                    k = float(k)
                except ValueError:
                    pass
                data[i][j] = k
        else:
            data[i, :] = np.array([float(k.replace('-', 'nan')) for k in ls])
    return data


def read_table(txt, plot=False, labels=None):
    # read 2d block of text
    labels = [r'\Delta \sigma', 'K'] if labels is None else labels
    if len(labels) != 2:
        raise ValueError('2D list of labels required')
    txt = txt.split('\n')
    nS = len(txt[-2].split())-1
    secondary = np.array([float(s) for s in txt[1].split()[-nS:]])
    nP = len(txt)-3
    primary = np.zeros(nP)
    data = np.zeros((nP, nS))
    for i, line in enumerate(txt[2:-1]):
        ls = line.split()
        if nP > 1:
            primary[i] = float(ls[0])
        data[i, :] = np.array([float(k.replace('-', 'nan')) for k in ls[-nS:]])
    if nP == 1:
        return data.flatten(), secondary
    else:
        return data, primary, secondary


class readtxt:

    def __init__(self, filename):
        self.nline = 0
        self.f = open(filename, 'r')

    def __enter__(self):
        return self

    def trim(self, label, index=-1, delimiter=' ', rewind=False):
        # search for label at index
        itrim = 0
        while True:
            sol = self.f.tell()  # start of line
            line = self.readline()
            try:
                if line.split(delimiter)[index] == label:
                    break
                else:
                    itrim += 1
            except IndexError:
                continue
        if rewind:
            self.rewind(False, sol)
        return itrim

    def skiplines(self, nskip, verbose=False):
        for nline in range(nskip):
            ln = self.readline()
            if verbose:
                print('skip:', ln)

    def skipnumber(self, nskip, verbose=False):
        n = 0
        while True:
            if verbose:
                ln = self.readline()
                no = len(ln.split())
                n += no
                print('skipnumber: {}'.format(no), ln)
            else:
                n += len(self.readline().split())
            if n == nskip:
                break
            elif n > nskip:
                raise ValueError('n > nskip')

    def readarray(self, nread):
        n = 0
        number_list = []
        while True:
            line = self.readline(True)
            number_list.extend(line)
            n += len(line)
            if n == nread:
                break
            elif n > nread:
                raise ValueError('readarray splits across line')
        return np.array(number_list)

    def readline(self, split=False, string=False, sep=None):
        while True:
            ln = self.f.readline()
            if len(ln) == 0:  # end of file
                self.f.close()  # close file
            ln = ln.strip()  # strip whitespace
            if ln:  # break when full line found
                break
        self.nline += 1  # increment line counter
        if split:
            ln = ln.split(sep=sep)
            if not string:
                try:
                    ln = [int(n) for n in ln]
                except ValueError:
                    try:
                        ln = [float(n) for n in ln]
                    except ValueError:
                        txt = '\nline not a sequence of numbers\n'
                        txt += 'call readline with str=True for list of str'
                        raise ValueError(txt)
        return ln

    def skipblock(self, ncol=6):
        n = 0
        while True:
            no = len(self.readline().split())
            n += no
            if no != ncol:
                break
        return n

    def readblock(self):  # skip blanklines
        dblock = []
        eof = False
        while True:
            sol = self.f.tell()  # start of line
            try:
                ln = self.readline()
            except ValueError:  # end of file
                eof = True
                break
            try:  # string of floats
                line = [float(d.replace('D', 'E')) for d in ln.split()]
                dblock.extend(line)
            except ValueError:  # string of characters
                break
        self.rewind(eof, sol)
        return dblock

    def rewind(self, eof, sol):
        if not eof:
            self.f.seek(sol)  # rewind to start of line
            self.nline -= 1  # rewind

    def readfblock(self, ncol_skip=0, ncol_read=None, string=False):
        fr = []
        eof = False
        while True:
            sol = self.f.tell()  # start of line
            try:
                line = self.readline(split=True, string=True)
                print(line)
            except ValueError:  # end of file
                eof = True
                break
            if ncol_read is None:
                ncol_read = len(line)
            if len(line) < ncol_skip+ncol_read:  # end of block
                    break
            try:
                if string:
                    fr.append([line[ncol_skip+i] for i in range(ncol_read)])
                else:
                    fr.append([float(line[ncol_skip+i].replace('D', 'E'))
                              for i in range(ncol_read)])
            except ValueError:  # end of block
                break
        self.rewind(eof, sol)
        return np.array(fr).T

    def readnumber(self):
        ln = self.readline()
        try:
            number = int(ln)
        except ValueError:
            try:
                number = float(ln)
            except ValueError:
                raise ValueError('{} is not a number'.format(ln))
        return number

    def checkline(self, startswith, rewind=False):
        sol = self.f.tell()  # start of line
        ln = self.readline()
        if not ln.startswith(startswith):
            errtxt = 'unexpected line\n'
            errtxt += 'expected to startswith: {}\n'.format(startswith)
            errtxt += 'found: {}'.format(ln)
            raise IOError(errtxt)
        if rewind:
            self.f.seek(sol)  # rewind to start of line

    def __exit__(self, type, value, traceback):
        self.f.close()

'''
def class_dir(name):
    root = list(name.__path__)[0]
    return root
'''

def trim_dir(check_dir):
    nlevel, dir_found = 5, False
    for i in range(nlevel):
        if os.path.isdir(check_dir):
            dir_found = True
            break
        else:
            if '../' in check_dir:
                check_dir = check_dir.replace('../', '', 1)
    if not dir_found:
        errtxt = check_dir + ' not found\n'
        raise ValueError(errtxt)
    return check_dir


def get_filepath(path, subfolder='', filename='', label='', **kwargs):
    date = kwargs.get('date', datetime.date.today().strftime('%d_%m_%Y'))
    filepath = path
    if subfolder:
        filepath += '/' + subfolder + '/'
    if filename:
        filepath += filename
    if date is not None:
        filepath += f'_{date}'
    if label:
        filepath += '_' + label
    return filepath, date


def qsub(script, jobname='analysis', t=1, freiahost=1,
         verbose=True, Code='Nova'):
    p = Popen(split('ssh -T freia{:03.0f}.hpc.l'.format(freiahost)),
              stdin=PIPE, stdout=PIPE, stderr=PIPE)
    p.stdin.write('cd ~/Code/Nova/nova\n'.encode())

    if '.py' not in script:
        script += '.py'
    #wd = ''
    py3 = '~/Code/anaconda3/bin/python3'
    flags = '-V -N {} -j y -wd -S {}'.format(jobname, py3)
    if t > 1:
        flags += ' -t 1:{:1.0f}'.format(t)
    flags += ' '
    qsub = 'qsub ' + flags + script + '\n'

    #
    p.stdin.write(qsub.encode())  # submit job
    p.stdin.flush()
    stdout, stderr = p.communicate()
    if verbose:
        print(stdout.decode(), stderr.decode())


class PATH(object):  # file paths

    def __init__(self, jobname, overwrite=True):
        import datetime
        import os

        self.path = {}
        self.jobname = jobname

        self.date_str = datetime.date.today().strftime('%Y_%m_%d')  # today

        # get root dir
        wd = os.getcwd()
        os.chdir('../')
        root = os.getcwd()
        os.chdir(wd)

        self.data = self.make_dir(root + '/Data')  # data dir
        if overwrite:
            self.folder = self.rep_dir(
                self.data + '/' + self.date_str + '.' + self.jobname)
        else:
            self.folder = self.make_dir(
                self.data + '/' + self.date_str + '.' + self.jobname)
        self.logfile = self.folder + '/run_log.txt'

    def new(self, task):
        self.task = task
        if task == 0:
            self.job = self.folder
        else:
            self.job = self.make_dir(self.data + '/' + self.date_str +
                                     '.' + self.jobname)
            self.job = self.rep_dir(self.data + '/' + self.date_str +
                                    '.' + self.jobname + '/task.' + str(task))
        self.screenID = self.jobname + '_task-' + str(task)

        # copy config files

    def make_dir(self, d):  # check / make
        import os
        if not os.path.exists(d):
            os.makedirs(d)
        return d

    def rep_dir(self, d):  # check / replace
        import os
        import shutil as sh
        if os.path.exists(d):
            sh.rmtree(d)
        os.makedirs(d)
        return d

    def go(self):
        import os
        self.home = os.getcwd()
        os.chdir(self.job)

    def back(self):
        import os
        os.chdir(self.home)


class SET_PATH(object):  # file paths

    def __init__(self, jobname, **kw):
        import os
        import datetime
        self.os = os
        self.file = {}
        self.jobname = jobname

        if 'date' in kw.keys():
            self.date_str = kw['date']
        else:
            self.date_str = datetime.date.today().strftime('%Y_%m_%d')  # today

        # get root dir
        self.home = self.os.getcwd()
        os.chdir('../')
        self.root = os.getcwd()
        os.chdir(self.home)

        self.data = self.check_dir(self.root + '/Data')  # data
        self.folder = self.check_dir(
            self.data + '/' + self.date_str + '.' + jobname)  # job

    def check_dir(self, d):  # check / replace
        if not self.os.path.exists(d):
            print(d)
            print('dir not found')
        return d

    def go(self, task):
        self.job = self.check_dir(self.data + '/' + self.date_str + '.' + self.jobname +
                                  '/task.' + str(task))  # job
        if 'Rfolder' in self.__dict__.keys():
            self.Rjob = self.make_dir(self.Rfolder + '/task.' + str(task))
        self.os.chdir(self.job)

    def back(self):
        self.os.chdir(self.home)

    def goto(self, there):
        self.home = self.os.getcwd()
        self.os.chdir(there)

    def make_dir(self, d):  # check / make
        import os
        if not os.path.exists(d):
            os.makedirs(d)
        return d

    def result(self, task=0, postfix=''):
        import shutil as sh
        self.Rdata = self.make_dir(self.root + '/Results')  # data
        self.Rfolder = self.make_dir(self.Rdata + '/' +
                                     self.date_str + '.' + self.jobname + postfix)
        # initalise folder + copy inputs
