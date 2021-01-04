"""Manage access to Sultan campaign metadata."""
import os.path
from dataclasses import dataclass, field
import itertools

import pandas
import numpy as np

from nova.thermalhydralic.sultan.database import DataBase


@dataclass
class Campaign:
    """
    Load Sultan experiment campaign metadata.

    Parameters
    ----------
    experiment : str
        Experiment label.
    binary : bool
        Load data from binary file.

    """

    _experiment: str
    database: DataBase = field(init=False, repr=False, default=None)
    metadata: pandas.DataFrame = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Initialize properties."""
        self.experiment = self._experiment

    @property
    def experiment(self):
        """Manage experiment name."""
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        self._experiment = experiment
        self.database = DataBase(self.experiment)
        self.load_metadata()

    @property
    def index(self):
        """Return metadata index, read-only."""
        return self.metadata['index']

    @property
    def note(self):
        """Return metadata notes."""
        note = self.metadata['note']
        return note.reset_index()

    @property
    def metadatafile(self):
        """Return full local filepath of the binary metadata file."""
        return self.database.binary_filepath('metadata.h5')

    @property
    def binaryfile(self):
        """Return full local filepath of bindary data file."""
        return self.database.binary_filepath('testdata.h5')

    def load_metadata(self):
        """Load campaign metadata."""
        if os.path.isfile(self.metadatafile):
            self._load_metadata()
        else:
            self._read_metadata()

    def _read_metadata(self):
        """Extract data from *.xls campaign metadata."""
        metadata_xls = self.database.locate('*.xls')
        with pandas.ExcelFile(metadata_xls) as xls:
            metadata_index = self._read_metadata_index(xls)
            metadata = self._read_metadata_testplan(xls, metadata_index)
            self._save_metadata(metadata)
        self.metadata = metadata

    def _save_metadata(self, metadata):
        """Save metadata to hdf file."""
        with pandas.HDFStore(self.metadatafile, mode='w') as store:
            for key in metadata:
                store.put(key, metadata[key], format='table', append=True)

    def _load_metadata(self):
        metadata = {}
        with pandas.HDFStore(self.metadatafile, mode='r') as store:
            for key in store.keys():
                metadata[key[1:]] = store[key]
        self.metadata = metadata

    @staticmethod
    def _read_strand(_xls_index):
        """
        Extract strand name from metadata _xls_index.

        Parameters
        ----------
        _xls_index : pandas.Series
            First columns of metadata.

        Raises
        ------
        StopIteration
            Strand label not found.

        Returns
        -------
        None.

        """
        testmode = 'c'
        labels = [label[0] for label in _xls_index.values
                  if isinstance(label[0], str)]
        try:
            strand = next(label for label in labels
                          if f'{testmode}XXYYZZ' in label
                          or f'{testmode.upper()}XXYYZZ' in label)
        except StopIteration as stop:
            raise StopIteration(f'{testmode}XXYYZZ not found in {labels}') \
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
    def _format_testname(index, j, strand, metadata_index):
        """
        Return formated testname.

        Ensure test name is unique. Set name to f'noname {j}' if strand
        not found.

        Parameters
        ----------
        index : array-like
            First column of metadata.
        j : int
            Shot start index.
        strand : str
            Strand ID.
        metadata_index : dict
            metadata index.

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
        if testname in metadata_index:
            testname += f' {j}'
        return testname

    @staticmethod
    def _format_metadata_index(metadata_index):
        """Remove open indeces [j, 0] from metadata index."""
        # remove open indices
        for testname in list(metadata_index.keys()):
            if metadata_index[testname][1] == 0:
                metadata_index.pop(testname)
        # convert to DataFrame
        metadata_index = pandas.DataFrame(metadata_index,
                                          index=['start', 'stop',
                                                 'testmode']).T
        metadata_index.reset_index(inplace=True)
        metadata_index['name'] = metadata_index['index']
        ac_index, dc_index = itertools.count(0), itertools.count(0)
        null_index = itertools.count(0)
        for i, name in enumerate(metadata_index['index']):
            if name.capitalize() == 'Instrumentation':
                name = 'cal'
            elif name[:2].upper() == 'AC':
                name = f'ac{next(ac_index)}'
            elif name[:2].upper() == 'DC':
                name = f'dc{next(dc_index)}'
            else:
                name = f'ex{next(null_index)}'
            metadata_index.loc[i, 'index'] = name
        metadata_index = metadata_index.astype(
            {'start': int, 'stop': int, 'testmode': str, 'name': str})
        metadata_index.set_index('index', inplace=True)
        return metadata_index

    def _istest(self, strand, label, _xls_index, i):
        """Return True if current label is identified as a test."""
        isshot = self._isshot(label)
        nextlabel = self._nextlabel(_xls_index, i)
        isnext_file = self._isfile(nextlabel)
        isnext_strand = self._isnext_strand(nextlabel, strand)
        return isshot and (isnext_file or isnext_strand)

    def _read_metadata_index(self, xls):
        """Extract metadata indices."""
        _xls_index = pandas.read_excel(xls, usecols=[0], header=None)
        metadata_index = {}
        testname = None
        skip_file = False
        strand = self._read_strand(_xls_index)
        for i, label in enumerate(_xls_index.values):
            if isinstance(label[0], str):
                if label[0][:len(strand)] == strand \
                        and testname is not None:
                    metadata_index[testname][1] = i  # advance stop index
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
                        _xls_index, j, strand, metadata_index)
                    metadata_index[testname] = [j, 0, '']  # test start
        testmode_index = len(strand)
        for testname in metadata_index:
            shotlabel = _xls_index.iloc[metadata_index[testname][1], 0]
            testmode = shotlabel[testmode_index]
            metadata_index[testname][2] = testmode.lower()
        metadata_index = self._format_metadata_index(metadata_index)
        return metadata_index

    @staticmethod
    def _append_note(note, testplan):
        columns = [col for col in testplan.columns.get_level_values(0) if
                   isinstance(col, str)]
        note_column = [col for col in columns if 'note' in col.lower()]
        if note_column:
            _note = pandas.Series(testplan.loc[:, note_column[0]].values,
                                  index=testplan.iloc[:, 0])
            note = pandas.concat([note, _note], axis=0)
            testplan.drop(columns=note_column, inplace=True, level=0)
        return note

    @staticmethod
    def _format_columns(testplan):
        # remove column nans
        columns = testplan.columns.get_level_values(0)
        drop_columns = columns[columns.isna()]
        if len(drop_columns) > 0:
            testplan.drop(columns=drop_columns, inplace=True, level=0)
        testplan.fillna(method='pad', inplace=True)
        # rename columns
        columns = {'I pulse': 'Ipulse', 'I Pulse': 'Ipulse',
                   'B Sultan': 'Be', 'B SULTAN':  'Be',
                   'B sultan': 'Be', 'frequency': 'frequency',
                   'Frequency': 'frequency',
                   'P. Freq': 'frequency',
                   'I Sample': 'Isample'}
        testplan.rename(columns=columns, inplace=True)
        testplan.rename(columns={np.nan: ''}, inplace=True, level=1)

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
    def _format_frequency(testplan):
        frequency_duration = ('frequency', 'Hz/duration')
        frequency_hz = ('frequency', 'Hz')
        if frequency_duration in testplan:
            testplan[('frequency', 'Hz')] = \
                testplan[frequency_duration].apply(
                    Campaign._format_frequency_label)
        elif frequency_hz in testplan:
            if testplan[frequency_hz].dtype == object:
                testplan[frequency_hz] = testplan[frequency_hz].apply(
                    Campaign._format_frequency_label)

    def _read_metadata_testplan(self, xls, metadata_index):
        """Return metadata."""
        metadata = {'index': metadata_index}
        note = pandas.Series(name='note', dtype=str)
        previouscolumns = None
        for testname in metadata['index'].index:
            testindex = metadata['index'].loc[testname, :]
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
            testplan = pandas.read_excel(xls, skiprows=start,
                                         nrows=stop-start+1, header=None)
            testplan.columns = columns
            self._format_columns(testplan)
            note = self._append_note(note, testplan)
            self._format_frequency(testplan)
            testplan.sort_values([('Be', 'T'), ('Isample', 'kA')],
                                 inplace=True)
            try:  # AC data
                testplan.sort_values([('Ipulse', 'A'), ('frequency', 'Hz')],
                                     inplace=True)
            except KeyError:
                pass
            testplan.reset_index(inplace=True)
            testplan.drop(columns=['index'], level=0, inplace=True)
            testplan.dropna(axis=1, inplace=True)
            # convert object dtypes to str
            dtypes = testplan.dtypes
            astype = {c: str for c in dtypes.index if dtypes[c] == object}
            testplan = testplan.astype(astype)
            # save to dict
            metadata[testname] = testplan
        metadata['note'] = pandas.DataFrame(note, columns=['note'])
        return metadata


if __name__ == '__main__':

    campaign = Campaign('CSJA_3')
