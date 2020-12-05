"""Postprocess Sultan AC loss test data."""
import os.path
from dataclasses import dataclass, field
from typing import List

import pandas

from nova.utilities.IO import pythonIO
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

    def __post_init__(self):
        """Initialize local and ftp data instances."""
        self.experiment = self._experiment

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

    def locate(self, file):
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
            filename = self.ftp.locate(file)
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
class TestPlan:
    """
    Load Sultan experiment testplan.

    Parameters
    ----------
    database : str or DataBase
        Experiment label or DataBase instance
    strand : str, read-only
        Strand name.

    """

    _experiment: str
    _mode: str = 'ac'
    binary: bool = True
    database: DataBase = field(init=False, repr=False, default=None)
    _index: pandas.DataFrame = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Initialize properties."""
        self.experiment = self._experiment  # initialize experiment
        self.mode = self._mode  # initialize mode

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
        self.database = DataBase(experiment)
        self._experiment = experiment
        self._index = None

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
        if mode not in ['ac', 'dc', 'full']:
            raise IndexError('mode not in [ac, dc, full]')
        self._mode = mode
        self._index = None

    @property
    def index(self):
        """Return testplan index."""
        if self._index is None:
            self.load_testplan()
        return self._index

    def load_testplan(self):
        """Load testplan index."""
        testplan = self.database.binary_filepath(f'{self.mode}_testplan.pq')
        if os.path.isfile(testplan) and self.binary:
            self._index = pandas.read_parquet(testplan)
        else:
            self._index = self.read_testplan()
            self._index.to_parquet(testplan)

    def read_testplan(self):
        """Extract data from *.xls testplan."""
        testplan = self.database.locate('*.xls')
        with pandas.ExcelFile(testplan) as xls:
            index = pandas.read_excel(xls, usecols=[0], header=None)
        _testplan_index = self._read_testplan_index(index)
        return pandas.DataFrame(_testplan_index, index=['start', 'stop']).T

    def _read_strand(self, index):
        """
        Extract strand name from testplan index.

        Parameters
        ----------
        index : pandas.Series
            First columns of testplan.

        Raises
        ------
        StopIteration
            Strand label not found.

        Returns
        -------
        None.

        """
        if self.mode == 'full':
            mode = 'a'
        else:
            mode = self.mode[0]
        labels = [label[0] for label in index.values
                  if isinstance(label[0], str)]
        try:
            strand = next(label for label in labels
                          if f'{mode}XXYYZZ' in label
                          or f'{mode.upper()}XXYYZZ' in label)
        except StopIteration as stop:
            raise StopIteration(f'{mode}XXYYZZ not found in {labels}') \
                from stop
        return strand.split('XXYYZZ')[0][:-1]

    def _shot_prefix(self, index):
        """Return shot prefix."""
        shot_prefix = self._read_strand(index)
        if self.mode != 'full':
            if f'{self.mode[0]}XXYYZZ' in shot_prefix:  # lower case
                shot_prefix += self.mode[0]
            else:  # upper case
                shot_prefix += self.mode[0].upper()
        return shot_prefix.replace('-', '')

    @staticmethod
    def _islabel(label):
        """Return True if index contains test, AC, or DC."""
        islabel = 'test' in label[0].lower()
        islabel |= label[0][:2] == 'AC'
        islabel |= label[0][:2] == 'DC'
        return islabel

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
    def _is_file(label):
        """
        Return True if label == File.

        Parameters
        ----------
        label : str
            Next label.

        Returns
        -------
        is_file : bool
            True if label == File.

        """
        try:
            is_file = label.strip().capitalize() == 'File'
        except AttributeError:
            is_file = False
        return is_file

    @staticmethod
    def _isnext_strand(nextlabel, shot_prefix):
        """
        Return True if nextlabel contains shot_prefix.

        Parameters
        ----------
        nextlabel : str
            Next label.
        shot_prefix : str
            Shot prefix.

        Returns
        -------
        isnext_strand : bool
            True if nextlabel contains shot_prefix.

        """
        try:
            isnext_strand = shot_prefix in nextlabel
        except TypeError:
            isnext_strand = False
        return isnext_strand

    @staticmethod
    def _format_testname(index, j, shot_prefix, _testplan_index):
        """
        Return formated testname.

        Ensure test name is unique. Set name to f'noname {j}' if shot_prefix
        not found.

        Parameters
        ----------
        index : array-like
            First column of testplan.
        j : int
            Shot start index.
        shot_prefix : str
            Shot ID prefix.
        _testplan_index : dict
            Testplan index.

        Returns
        -------
        testname : str
            Formated testname.

        """
        testname = index.values[j-1][0]
        try:
            if shot_prefix in testname:
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
        for testname in list(_testplan_index.keys()):  # remove open indices
            if _testplan_index[testname][1] == 0:
                _testplan_index.pop(testname)
        return _testplan_index

    def _read_testplan_index(self, index):
        """Extract testplan indices."""
        _testplan_index = {}
        testname = None
        skip_file = False
        shot_prefix = self._shot_prefix(index)
        for i, label in enumerate(index.values):
            if isinstance(label[0], str):
                if label[0][:len(shot_prefix)] == shot_prefix \
                        and testname is not None:
                    _testplan_index[testname][1] = i  # advance stop index
                else:
                    islabel = self._islabel(label)
                    nextlabel = self._nextlabel(index, i)
                    isnext_file = self._is_file(nextlabel)
                    isnext_strand = self._isnext_strand(nextlabel, shot_prefix)
                    if islabel and (isnext_file or isnext_strand):
                        j = i+1
                        skip_file = True
                    elif self._is_file(label[0]):
                        if skip_file:
                            skip_file = False
                            continue
                        j = i
                    else:
                        testname = None
                        continue
                    testname = self._format_testname(
                        index, j, shot_prefix, _testplan_index)
                    _testplan_index[testname] = [j, 0]  # create new test start
        return self._format_testplan_index(_testplan_index)


@dataclass
class TestMatrix(pythonIO):
    """
    Provide access to sultan data.

    Datafiles stored localy. Files downloaded from ftp server if required.

    """

    experiment: str
    mode: str = 'ac'
    testname: int = 0  # test identifier
    shot: int = 0  # shot identifier
    binary: bool = True

    def __post_init__(self):
        self.database = DataBase(self.experiment)
        self.testplan = TestPlan(self.experiment, self.mode, self.binary)

    '''
    def read_testmatrix(self):
        # store test matrix data
        previouscolumns = None
        with pandas.ExcelFile(testplan) as xls:
            for label in _testplan_index:
                index = _testplan_index[label]
                df = pandas.read_excel(
                    xls, skiprows=index[0],
                    nrows=np.diff(index)[0]+1, header=None)
                if df.iloc[0, 0] == 'File':
                    df.columns = pandas.MultiIndex.from_arrays(
                        df.iloc[:2].values)
                    df = df.iloc[2:]
                    previouscolumns = df.columns
                elif previouscolumns is not None:
                    df.columns = previouscolumns
                # remove column nans
                columns = df.columns.get_level_values(0)
                df = df.loc[:, columns.notna()]
                # extract note
                columns = [col for col in df.columns.get_level_values(0) if
                           isinstance(col, str)]
                note = [col for col in columns if 'note' in col.lower()]
                if note:
                    self._note[label] = df[note[0]]
                    self._note[label].columns = ['Note']
                    df.drop(columns=note, inplace=True, level=0)
                df.fillna(method='pad', inplace=True)
                # rename columns
                columns = {'I pulse': 'Ipulse', 'I Pulse': 'Ipulse',
                           'B Sultan': 'Be', 'B SULTAN':  'Be',
                           'B sultan': 'Be', 'frequency': 'frequency',
                           'Frequency': 'frequency',
                           'P. Freq': 'frequency',
                           'I Sample': 'Isample'}
                df.rename(columns=columns, inplace=True)
                df.rename(columns={np.nan: ''}, inplace=True, level=1)
                # format frequency
                frequency_duration = ('frequency', 'Hz/duration')
                frequency_Hz = ('frequency', 'Hz')
                if frequency_duration in df:
                    df[('frequency', 'Hz')] = df[frequency_duration].apply(
                            self._format_frequency)
                elif frequency_Hz in df:
                    if df[frequency_Hz].dtype == object:
                        df[frequency_Hz] = df[frequency_Hz].apply(
                            self._format_frequency)
                df.sort_values([('Be', 'T'), ('Isample', 'kA')], inplace=True)
                try:  # AC data
                    df.sort_values([('Ipulse', 'A'), ('frequency', 'Hz')],
                                   inplace=True)
                except KeyError:
                    pass
                df.reset_index(inplace=True)
                df.drop(columns=['index'], level=0, inplace=True)
                self._testmatrix[label] = df
    '''


'''

@dataclass
class SultanData(pythonIO):

    _attributes = ['experiment', 'testname', 'shot', 'mode']
    _default_attributes = {'mode': 'ac', 'read_txt': False}
    _input_attributes = ['testname', 'shot', 'mode']

    def __init__(self, *args, **kwargs):
        self._experiment = None
        self._testname = None  # test identifier
        self._shot = None  # shot identifier
        self._mode = None
        self._reload = True  # reload data from file
        self._testmatrix = None
        self._note = {}
        self._sultandata = None
        self._set_data_attributes(*args, **kwargs)

    @property
    def reload(self):
        """
        Manage data pipeline.

        Parameters
        ----------
        reload : bool
            Reload status - reinitialize instance when set to True.

        Returns
        -------
        reload : bool

        """
        return self._reload

    @reload.setter
    def reload(self, reload):
        if reload:
            self._sultandata = None
            if hasattr(self, 'postprocess'):  # update postprocess chain
                self.postprocess = True
        self._reload = reload

    @property
    def testkeys(self):
        """Return testmatrix keys as pandas.Series, read-only."""
        return pandas.Series(list(self.testmatrix.keys()), dtype=object)

    @property
    def testmatrix(self):
        """Return testmatrix, read-only."""
        if self._testmatrix is None:
            self.load_testmatrix()
        return self._testmatrix

    @property
    def testname(self):
        """
        Manage testname, testmatrix key.

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
        if self._testname is None:
            raise IndexError('testname not set, '
                             'valid range (str or int): '
                             f'\n\n{self.testkeys}')
        return self._testname

    @testname.setter
    def testname(self, testname):
        if isinstance(testname, int):
            try:
                testname = self.testkeys.iloc[testname]
            except IndexError:
                raise IndexError(f'testname index {testname} out of range\n\n'
                                 f'{self.testkeys}')
        elif isinstance(testname, str):
            if testname not in self.testmatrix:
                raise IndexError(f'testname {testname} not found in '
                                 f'\n{self.testkeys}')
        if testname != self._testname:
            self._testname = testname
            self.reload = True

    @property
    def testindex(self):
        """Return testname index."""
        return next((i for i, name in enumerate(self.testmatrix)
                     if name == self.testname))

    @property
    def testplan(self):
        """Return testplan, read-only."""
        return self.testmatrix[self.testname]

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
            Shot not set (is None) or set out of range.

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
                             f'{self.testplan}')
        if shot != _shot:
            self.reload = True

    @property
    def shot_range(self):
        """Return valid shot range, (int, int)."""
        index = self.testplan.index
        return index[0], index[-1]+1

    @property
    def note(self):
        """Return shot note."""
        return self._note[self.testname].iloc[self._shot]

    @staticmethod
    def _check_stopindex(_testplan_index, testname):
        """Pop testname if stopindex == 0."""
        if testname is not None:
            if _testplan_index[testname][1] == 0:
                _testplan_index.pop(testname)
        return _testplan_index

    def load_testmatrix(self, **kwargs):
        """
        Load testmatrix from file.

        Parameters
        ----------
        **kwargs : dict
            force read_txt with read_txt=True keyword argument.

        Returns
        -------
        None.

        """
        read_txt = kwargs.get('read_txt', self.read_txt)
        if self._experiment is None:
            raise IndexError('experiment not set')
        filepath = os.path.join(self.localdir, f'testmatrix_{self.mode}')
        if not os.path.isfile(filepath + '.pk') or read_txt:
            self.read_testmatrix()
            self.save_pickle(filepath, ['strand', '_testmatrix'])
        else:
            self.load_pickle(filepath)


    @staticmethod
    def _format_frequency(x):
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
            x = float(x.split('/')[0])
        except ValueError:  # split string is string (BPTrapez)
            x = -1
        except AttributeError:  # x already float
            pass
        return x

    def _get_filename(self, testname, shot):
        """
        Return shot filename.

        Parameters
        ----------
        testname : str or float
            Test identifier.
        shot : int
            Shot identifier.

        Returns
        -------
        filename : str
            filename.

        """
        if testname is not None:
            self.testname = testname
        if shot is not None:
            self.shot = shot
        return f'{self.shot[("File", "")]}.dat'

    def _load_datafile(self, testname=None, shot=None,
                       subdir=['ac/dat', 'AC/ACdat', 'TEST/AC/ACdat']):
        """
        Return sultan dataframe and associated shot metadata.

        Parameters
        ----------
        testname : str or int, optional
            Test identifier. The default is None.
        shot : int, optional
            Shot identifier. The default is None.
        subdir : array-like, optional
            List of trial data subdirectories.
            The default is ['ac/dat', 'AC/ACdat', 'TEST/AC/ACdat'].

        Returns
        -------
        sultandata : pandas.DataFrame
            Shot data.
        sultanmetadata : pandas.Series
            Shot metadata.

        """
        filename = self._get_filename(testname, shot)
        for sdir in subdir:
            try:
                datafile = self.locate(filename, subdir=sdir)
                break
            except ftputil.error.PermanentError as error:
                ftp_err = error
                pass
        try:
            datafile = os.path.join(self.datadir, datafile)
        except UnboundLocalError:
            raise ftputil.error.PermanentError(f'{ftp_err}')
        sultandata = pandas.read_csv(datafile, encoding='ISO-8859-1')
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

    @property
    def metadata(self):
        """Return shot metadata."""
        return self.shot

    @property
    def sultandata(self):
        """Return sultandata, reload if necessary."""
        if self.reload or self._sultandata is None:
            self._sultandata = self._load_datafile()
            self.reload = False
        return self._sultandata
'''

if __name__ == '__main__':


    tp = TestPlan('CSJA_3')
