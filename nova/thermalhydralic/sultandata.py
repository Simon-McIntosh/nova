"""Postprocess Sultan AC loss test data."""
import os.path
from dataclasses import dataclass, field
from typing import List, Any
import itertools

import pandas
import numpy as np

from nova.thermalhydralic.localdata import LocalData
from nova.thermalhydralic.remotedata import FTPData


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
    _local_args: List[str] = field(default_factory=list)
    _ftp_args: List[str] = field(default_factory=list)
    local: LocalData = field(init=False, repr=False)
    ftp: FTPData = field(init=False, repr=False)
    datapath: str = 'ac/dat AC/ACdat TEST/AC/ACdat'

    def __post_init__(self):
        """Initialize local and ftp data instances."""
        self.experiment = self._experiment

    def datafile(self, filename):
        """Return full local path of datafile."""
        for relative_path in self.datapath.split():
            try:
                datafile = self.locate(filename, relative_path)
                break
            except FileNotFoundError as file_not_found:
                file_not_found_error = file_not_found
                pass
        try:
            return self.source_filepath(datafile)
        except AttributeError:
            err_txt = f'datafile not found on datapath {self.datapath}'
            raise FileNotFoundError(err_txt) from file_not_found_error

    @property
    def experiment(self):
        """Manage sultan experiment."""
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        self.ftp = FTPData(experiment, *self.ftp_args)
        self.local = LocalData(experiment, *self.local_args)
        self._experiment = experiment

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
        Return full filename. Search localy first, download if not found.

        Parameters
        ----------
        file : str
            Filename, names of type '*.ext' permited.

        Returns
        -------
        file : str
            Full filename.

        """
        try:
            filename = self.local.locate(file)
        except FileNotFoundError:
            filename = self.ftp.locate(file, *relative_path)
            makedir = ~self.local.checkdir()  # generate structure if requred
            if makedir:
                self.local.makedir()
            try:
                self.ftp.download(filename, self.local.source_directory)
            except FileNotFoundError as file_not_found:
                if makedir:
                    self.local.removedir()  # remove if generated bare
                raise FileNotFoundError(f'File {filename} not found on '
                                        'ftp server') from file_not_found
        return self.source_filepath(filename)

    def binary_filepath(self, filename):
        """Return binary filepath."""
        return os.path.join(self.local.binary_directory, filename)

    def source_filepath(self, filename):
        """Return source filepath."""
        return os.path.join(self.local.source_directory, filename)


@dataclass
class TestData:
    """
    Load Sultan experiment testplan and testdata.

    Parameters
    ----------
    experiment : str
        Experiment label.
    binary : bool
        Load data from binary file.
    """

    _experiment: str
    binary: bool = True
    database: DataBase = field(init=False, repr=False, default=None)
    testplan: pandas.DataFrame = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Initialize properties."""
        self.experiment = self._experiment  # initialize experiment

    def __repr__(self):
        """Return string representation of dataclass."""
        _vars = vars(self)
        attributes = ", ".join(f"{name.replace('_', '')}={_vars[name]!r}"
                               for name in _vars)
        return f"{self.__class__.__name__}({attributes})"

    @property
    def experiment(self):
        """Manage experiment name."""
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        self._experiment = experiment
        self.database = DataBase(self.experiment)
        self.load_testplan()

    @property
    def testfile(self):
        """Return testplan filename."""
        return self.database.binary_filepath('testplan.h5')

    def load_testplan(self):
        """Load testplan."""
        if os.path.isfile(self.testfile) and self.binary:
            self._load_testplan()
        else:
            self._read_testplan()

    def _read_testplan(self):
        """Extract data from *.xls testplan."""
        testplan_xls = self.database.locate('*.xls')
        with pandas.ExcelFile(testplan_xls) as xls:
            testplan_index = self._read_testplan_index(xls)
            testplan = self._read_testplan_metadata(xls, testplan_index)
            self._write_testplan(testplan)
        self.testplan = testplan

    @staticmethod
    def _read_strand(_xls_index):
        """
        Extract strand name from testplan _xls_index.

        Parameters
        ----------
        _xls_index : pandas.Series
            First columns of testplan.

        Raises
        ------
        StopIteration
            Strand label not found.

        Returns
        -------
        None.

        """
        mode = 'c'
        labels = [label[0] for label in _xls_index.values
                  if isinstance(label[0], str)]
        try:
            strand = next(label for label in labels
                          if f'{mode}XXYYZZ' in label
                          or f'{mode.upper()}XXYYZZ' in label)
        except StopIteration as stop:
            raise StopIteration(f'{mode}XXYYZZ not found in {labels}') \
                from stop
        return strand.split('XXYYZZ')[0][:-1].replace('-', '')

    @staticmethod
    def _isshot(label):
        """Return True if label contains test, AC, or DC."""
        isshot = 'test' in label[0].lower()
        isshot |= label[0][:2] == 'AC'
        isshot |= label[0][:2] == 'DC'
        return isshot

    @staticmethod
    def _nextlabel(index, i):
        """
        Return next label in index.

        Parameters
        ----------
        index : array-like
            list of strings.
        i : int
            Current location in list.

        Returns
        -------
        nextlabel : str
            Next label, '' if at end of list.

        """
        try:
            nextlabel = index.values[i+1][0]
        except IndexError:
            nextlabel = ''
        return nextlabel

    @staticmethod
    def _isfile(label):
        """
        Return True if label == File.

        Parameters
        ----------
        label : str
            Index label.

        Returns
        -------
        isfile : bool
            True if label == File.

        """
        try:
            isfile = label.strip().capitalize() == 'File'
        except AttributeError:
            isfile = False
        return isfile

    @staticmethod
    def _isnext_strand(nextlabel, strand):
        """
        Return True if nextlabel contains strand.

        Parameters
        ----------
        nextlabel : str
            Next label.
        strand : str
            Strand ID.

        Returns
        -------
        isnext_strand : bool
            True if nextlabel contains strand.

        """
        try:
            isnext_strand = strand in nextlabel
        except TypeError:
            isnext_strand = False
        return isnext_strand

    @staticmethod
    def _format_testname(index, j, strand, _testplan_index):
        """
        Return formated testname.

        Ensure test name is unique. Set name to f'noname {j}' if strand
        not found.

        Parameters
        ----------
        index : array-like
            First column of testplan.
        j : int
            Shot start index.
        strand : str
            Strand ID.
        _testplan_index : dict
            Testplan index.

        Returns
        -------
        testname : str
            Formated testname.

        """
        testname = index.values[j-1][0]
        try:
            if strand in testname:
                testname = f'noname {j}'
        except TypeError:
            testname = f'noname {j}'
        testname = testname.split(':')[0].split('(')[0]
        if testname in _testplan_index:
            testname += f' {j}'
        return testname

    @staticmethod
    def _format_testplan_index(_testplan_index):
        """Remove open indeces [j, 0] from testplan index."""
        # remove open indices
        for testname in list(_testplan_index.keys()):
            if _testplan_index[testname][1] == 0:
                _testplan_index.pop(testname)
        # convert to DataFrame
        _testplan_index = pandas.DataFrame(_testplan_index,
                                           index=['start', 'stop', 'mode']).T
        _testplan_index.reset_index(inplace=True)
        _testplan_index['name'] = _testplan_index['index']
        ac_index, dc_index = itertools.count(0), itertools.count(0)
        null_index = itertools.count(0)
        for i, name in enumerate(_testplan_index['index']):
            if name.capitalize() == 'Instrumentation':
                name = 'cal'
            elif name[:2].upper() == 'AC':
                name = f'ac{next(ac_index)}'
            elif name[:2].upper() == 'DC':
                name = f'dc{next(dc_index)}'
            else:
                name = f'ex{next(null_index)}'
            _testplan_index.loc[i, 'index'] = name
        _testplan_index = _testplan_index.astype(
            {'start': int, 'stop': int, 'mode': str, 'name': str})
        _testplan_index.set_index('index', inplace=True)
        return _testplan_index

    def _istest(self, strand, label, _xls_index, i):
        """Return True if current label is identified as a test."""
        isshot = self._isshot(label)
        nextlabel = self._nextlabel(_xls_index, i)
        isnext_file = self._isfile(nextlabel)
        isnext_strand = self._isnext_strand(nextlabel, strand)
        return isshot and (isnext_file or isnext_strand)

    def _read_testplan_index(self, xls):
        """Extract testplan indices."""
        _xls_index = pandas.read_excel(xls, usecols=[0], header=None)
        _testplan_index = {}
        testname = None
        skip_file = False
        strand = self._read_strand(_xls_index)
        for i, label in enumerate(_xls_index.values):
            if isinstance(label[0], str):
                if label[0][:len(strand)] == strand \
                        and testname is not None:
                    _testplan_index[testname][1] = i  # advance stop index
                else:
                    if self._istest(strand, label, _xls_index, i):
                        j = i+1
                        skip_file = True
                    elif self._isfile(label[0]):
                        if skip_file:
                            skip_file = False
                            continue
                        j = i
                    else:
                        testname = None
                        continue
                    testname = self._format_testname(
                        _xls_index, j, strand, _testplan_index)
                    _testplan_index[testname] = [j, 0, '']  # test start
        mode_index = len(strand)
        for testname in _testplan_index:
            shotlabel = _xls_index.iloc[_testplan_index[testname][1], 0]
            shotmode = shotlabel[mode_index]
            _testplan_index[testname][2] = shotmode.lower()
        _testplan_index = self._format_testplan_index(_testplan_index)
        return _testplan_index

    @staticmethod
    def _append_note(note, _testplan):
        columns = [col for col in _testplan.columns.get_level_values(0) if
                   isinstance(col, str)]
        note_column = [col for col in columns if 'note' in col.lower()]
        if note_column:
            _note = pandas.Series(_testplan.loc[:, note_column[0]].values,
                                  index=_testplan.iloc[:, 0])
            note = pandas.concat([note, _note], axis=0)
            _testplan.drop(columns=note_column, inplace=True, level=0)
        return note

    @staticmethod
    def _format_columns(_testplan):
        # remove column nans
        columns = _testplan.columns.get_level_values(0)
        drop_columns = columns[columns.isna()]
        if len(drop_columns) > 0:
            _testplan.drop(columns=drop_columns, inplace=True, level=0)
        _testplan.fillna(method='pad', inplace=True)
        # rename columns
        columns = {'I pulse': 'Ipulse', 'I Pulse': 'Ipulse',
                   'B Sultan': 'Be', 'B SULTAN':  'Be',
                   'B sultan': 'Be', 'frequency': 'frequency',
                   'Frequency': 'frequency',
                   'P. Freq': 'frequency',
                   'I Sample': 'Isample'}
        _testplan.rename(columns=columns, inplace=True)
        _testplan.rename(columns={np.nan: ''}, inplace=True, level=1)

    @staticmethod
    def _format_frequency_label(frequency_label):
        """
        Return frecuency extracted from frequency/duration string.

        Parameters
        ----------
        x : str or float
            frequency (float) or frequency/duration (str).

        Returns
        -------
        x : float
            frequency.

        """
        try:
            frequency_label = float(frequency_label.split('/')[0])
        except ValueError:  # split string is string (BPTrapez)
            frequency_label = -1
        except AttributeError:  # x already float
            pass
        return frequency_label

    @staticmethod
    def _format_frequency(_testplan):
        frequency_duration = ('frequency', 'Hz/duration')
        frequency_hz = ('frequency', 'Hz')
        if frequency_duration in _testplan:
            _testplan[('frequency', 'Hz')] = \
                _testplan[frequency_duration].apply(
                    TestData._format_frequency_label)
        elif frequency_hz in _testplan:
            if _testplan[frequency_hz].dtype == object:
                _testplan[frequency_hz] = _testplan[frequency_hz].apply(
                    TestData._format_frequency_label)

    def _read_testplan_metadata(self, xls, testplan_index):
        """Return testplan metadata."""
        testplan = {'index': testplan_index}
        note = pandas.Series(name='note', dtype=str)
        previouscolumns = None
        for testname in testplan['index'].index:
            testindex = testplan['index'].loc[testname, :]
            start, stop = testindex.loc[['start', 'stop']]
            _header = pandas.read_excel(
                xls, skiprows=start, nrows=stop-start+1, header=None)
            if _header.iloc[0, 0] == 'File':
                start += 2
                columns = pandas.MultiIndex.from_arrays(
                    _header.iloc[:2].values)
                previouscolumns = columns
            elif previouscolumns is not None:
                columns = previouscolumns
            _testplan = pandas.read_excel(xls, skiprows=start,
                                          nrows=stop-start+1, header=None)
            _testplan.columns = columns
            self._format_columns(_testplan)
            note = self._append_note(note, _testplan)
            self._format_frequency(_testplan)
            _testplan.sort_values([('Be', 'T'), ('Isample', 'kA')],
                                  inplace=True)
            try:  # AC data
                _testplan.sort_values([('Ipulse', 'A'), ('frequency', 'Hz')],
                                      inplace=True)
            except KeyError:
                pass
            _testplan.reset_index(inplace=True)
            _testplan.drop(columns=['index'], level=0, inplace=True)
            _testplan.dropna(axis=1, inplace=True)
            # convert object dtypes to str
            dtypes = _testplan.dtypes
            astype = {c: str for c in dtypes.index if dtypes[c] == object}
            _testplan = _testplan.astype(astype)
            # save to dict
            testplan[testname] = _testplan
        testplan['note'] = pandas.DataFrame(note, columns=['note'])
        return testplan

    def _write_testplan(self, testplan):
        """Save testplan to json file."""
        with pandas.HDFStore(self.testfile, mode='w') as store:
            for key in testplan:
                store.put(key, testplan[key], format='table', append=True)

    def _load_testplan(self):
        testplan = {}
        with pandas.HDFStore(self.testfile, mode='r') as store:
            for key in store.keys():
                testplan[key[1:]] = store[key]
        self.testplan = testplan

    @property
    def testindex(self):
        """Return testplan index, read-only."""
        return self.testplan['index']

    def datafile(self, filename):
        """Return full local filepath of datafile."""
        return self.database.datafile(filename)


@dataclass
class SultanTest:

    _experiment: str
    _mode: str = 'ac'
    _testname: Any = 0
    _shot: int = 0
    testdata: TestData = field(init=False, repr=False)

    def __post_init__(self):
        self.testdata = TestData(self.experiment)
        self.testname = self._testname

    @property
    def experiment(self):
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        self.testdata.experiment = experiment
        self._experiment = experiment

    @property
    def mode(self):
        """
        Manage sultan test mode.

        Parameters
        ----------
        mode : str
            Sultan test mode.

        Raises
        ------
        IndexError
            Mode not in [ac, dc, full].

        Returns
        -------
        mode : str

        """
        return self._mode

    @mode.setter
    def mode(self, mode):
        mode = mode.lower()
        if mode not in ['cal', 'ac', 'dc', 'full']:
            raise IndexError('mode not in [cal, ac, dc, full]')
        self._mode = mode

    @property
    def testindex(self):
        """Return testplan index, read-only."""
        index = self.testdata.testindex
        if self.mode == 'full':
            names = index.loc[:, 'name']
        else:
            names = index.loc[index['mode'] == self.mode[0], 'name']
        return names

    @property
    def testname(self):
        """
        Manage testname.

        Parameters
        ----------
        testname : str or int
            Test identifier.

        Raises
        ------
        IndexError
            testname out of range.

        Returns
        -------
        testname : str

        """
        return self._testname

    @testname.setter
    def testname(self, testname):
        if isinstance(testname, int):
            testindex = testname
            try:
                testname = self.testindex.index[testindex]
            except IndexError:
                raise IndexError(f'testname index {testindex} out of range\n\n'
                                 f'{self.testindex}')
        elif isinstance(testname, str):
            if testname not in self.testindex.index:
                raise IndexError(f'testname {testname} not found in '
                                 f'\n{self.testindex}')
        self._testname = testname
        if testname != self._testname:
            self._testname = testname
            self.reload = True

    @property
    def testplan(self):
        """Return testplan, read-only."""
        return self.testdata.testplan[self.testname]

    @property
    def shot(self):
        """
        Shot identifier.

        Parameters
        ----------
        shot : int
            Shot identifier.

        Raises
        ------
        IndexError
            Shot not set (is None) or is set out of range.

        Returns
        -------
        shot : pandas.Series
            Shot identifier.

        """
        if self._shot is None:
            raise IndexError('shot index not set, '
                             'valid range '
                             f'{self.shot_range[0]}-{self.shot_range[1]-1}')
        return self.testplan.iloc[self._shot, :]

    @shot.setter
    def shot(self, shot):
        try:
            _shot = self._shot  # store previous
            self._shot = shot
            self.shot
        except IndexError:
            self._shot = _shot  # rewind
            raise IndexError(f'shot index {shot} out of bounds \n'
                             f'{self.testdata}')
        if shot != _shot:
            self.reload = True

    @property
    def shot_range(self):
        """Return valid shot range, (int, int)."""
        index = self.testdata.index
        return index[0], index[-1]+1

    @property
    def file(self):
        """Return shot filename."""
        return self.shot.loc['File'][0]

    @property
    def filename(self):
        """Return datafile filename."""
        return f'{self.file}.dat'

    @property
    def note(self):
        """Return shot note."""
        return self.testdata.testplan['note'].loc[self.file][0]

    @property
    def testnotes(self):
        """Return testplan notes."""
        note = self.testdata.testplan['note'].loc[self.testplan.File]
        return note.reset_index()

    @property
    def datafile(self):
        """Return full local filepath of datafile."""
        return self.testdata.datafile(self.filename)


@dataclass
class SultanData:
    """Access Sultan timeseries data."""

    test: SultanTest = field(repr=False)
    _raw: pandas.DataFrame = field(init=False, repr=False, default=None)
    _lowpass: pandas.DataFrame = field(init=False, repr=False, default=None)

    def _read_datafile(self):
        """
        Return sultan dataframe.

        Returns
        -------
        sultandata : pandas.DataFrame
            Shot data.

        """
        sultandata = pandas.read_csv(self.test.datafile, encoding='ISO-8859-1')
        columns = {}
        for c in sultandata.columns:
            if 'left' in c or 'right' in c:
                columns[c] = c.replace('left', 'Left')
                columns[c] = columns[c].replace('right', 'Right')
                columns[c] = columns[c].replace('  ', ' ')
            if c[-7:] == ' (320K)':
                columns[c] = c[:-7]
        sultandata.rename(columns=columns, inplace=True)
        if 'T in' in sultandata.columns:
            sultandata['T in Left'] = sultandata['T in']
            sultandata['T in Right'] = sultandata['T in']
        if 'P in' in sultandata.columns:
            sultandata['P in Left'] = sultandata['P in']
            sultandata['P in Right'] = sultandata['P in']
        return sultandata


if __name__ == '__main__':

    test = SultanTest('CSJA_3', 'ac')
    sd = SultanData(test)
    print(sd._read_datafile())
