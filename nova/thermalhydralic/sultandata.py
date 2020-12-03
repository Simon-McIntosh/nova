"""Postprocess Sultan AC loss test data."""
import os.path
from dataclasses import dataclass

from nova.utilities.IO import pythonIO
from nova.thermalhydralic.localdata import LocalData
from nova.thermalhydralic.remotedata import FTPData


@dataclass
class TestPlan:


    @stataticmethod
    def _read_strand(index):
        """Return strand label extracted from testplan index."""
        labels = [label[0] for label in index.values
                  if isinstance(label[0], str)]
        try:
            strand = next(label for label in labels
                          if f'{mode}XXYYZZ' in label
                          or f'{mode.upper()}XXYYZZ' in label)
        except StopIteration:
            raise StopIteration(f'{mode}XXYYZZ not found in {labels}')
        self.strand = strand.split('XXYYZZ')[0][:-1]

    def _read_testplan_index(self, index):
        """Extract testplan indices."""
        _testplan_index = {}
        testname = None
        skip_file = False
        strandID = self.strand
        if self.mode == 'full':
            mode = 'a'
        else:
            mode = self.mode[0]
            if f'{mode}XXYYZZ' in strand:  # lower case
                strandID += self.mode[0]
            else:  # upper case
                strandID += self.mode[0].upper()
        strandID = strandID.replace('-', '')
        for i, label in enumerate(index.values):
            if isinstance(label[0], str):
                if label[0][:len(strandID)] == strandID \
                        and testname is not None:
                    _testplan_index[testname][1] = i  # advance stop index
                else:
                    islabel = 'test' in label[0].lower() \
                        or label[0][:2] == 'AC' \
                        or label[0][:2] == 'DC'
                    try:
                        nextlabel = index.values[i+1][0]
                    except IndexError:
                        nextlabel = ''
                    try:
                        isnext_file = nextlabel.strip().capitalize() == 'File'
                    except AttributeError:
                        isnext_file = False
                    try:
                        isnext_strand = strandID in nextlabel
                    except TypeError:
                        isnext_strand = False
                    if islabel and (isnext_file or isnext_strand):
                        j = i+1
                        skip_file = True
                    elif label[0].strip().capitalize() == 'File':
                        if skip_file:
                            skip_file = False
                            continue
                        j = i
                    else:
                        testname = None
                        continue
                    testname = index.values[j-1][0]
                    try:
                        if strandID in testname:
                            testname = f'noname {j}'
                    except TypeError:
                        testname = f'noname {j}'
                    testname = testname.split(':')[0].split('(')[0]
                    if testname in _testplan_index:
                        testname += f' {j}'
                    _testplan_index[testname] = [j, 0]
        for testname in list(_testplan_index.keys()):  # remove open indices
            if _testplan_index[testname][1] == 0:
                _testplan_index.pop(testname)
        return _testplan_index

    def read_testplan(self):
        """Read *.xls testplan."""
        testplan = self.locate('*.xls')
        testplan = os.path.join(self.local.source_directory, testplan)
        with pandas.ExcelFile(testplan) as xls:
            index = pandas.read_excel(xls, usecols=[0], header=None)
        self.strand = self._read_strand(index)


@dataclass
class SultanData(pythonIO):
    """
    Provide access to sultan data.

    Datafiles stored localy. Files downloaded from ftp server if required.

    """

    _experiment: str
    _testname: int = 0  # test identifier
    _shot: int = 0  # shot identifier
    _mode: str = 'ac'

    def __post_init__(self):
        self.local = LocalData(self.experiment, 'Sultan', 'ftp', 'local')
        self.ftp = FTPData(self.local)

    @property
    def experiment(self):
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        self.local.experiment = experiment
        self._experiment = experiment

    #def optional(self, *args):
    #    read_text = args + (True,)
    #    print(read_text[0])

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
            localfile = self.local.locate(file)
            return localfile
        except FileNotFoundError:
            remotefile = self.ftp.locate(file)
            self.ftp.download(remotefile)
            return remotefile




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


if __name__ == '__main__':

    data = SultanData('CSJA_3')
    data.read_testplan()

    '''
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
    '''

    '''
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
        if self._mode is None:
            raise IndexError('mode not set, valid entries [ac, dc, full]')
        return self._mode

    @mode.setter
    def mode(self, mode):
        _mode = self._mode  # store previous
        mode = mode.lower()
        if mode not in ['ac', 'dc', 'full']:
            raise IndexError('mode not in [ac, dc, full]')
        if _mode != mode:
            self._mode = mode
            self._testmatrix = None

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
