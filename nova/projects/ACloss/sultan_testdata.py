
import os
import glob
import re

import ftputil
import pandas
import numpy as np
import scipy.signal
import CoolProp.CoolProp as CoolProp

from nova.definitions import root_dir
from nova.utilities.pyplot import plt


class FTPSultan:
    """
    Provide access to sultan data.

    Datafiles stored localy. Files downloaded from ftp server if not found.

    """

    def __init__(self, experiment):
        self._experiment = experiment
        self.datadir = self._set_datadir()
        self._testname = None  # test identifier
        self._shot = None  # shot identifier
        self._reload = True  # reload data from file
        self._testmatrix = {}
        self._note = {}
        self.load_testmatrix()

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

    def _set_datadir(self):
        datadir = os.path.join(root_dir, f'data/Sultan/{self.experiment}')
        if not os.path.isdir(datadir):
            os.mkdir(datadir)
        return datadir

    @property
    def experiment(self):
        """
        Manage experiment identifier.

        Reinitialize if changed.

        Parameters
        ----------
        experiment : str
            Test directory name, evaluated as ftp/parentdir/experiment.

        Returns
        -------
        experiment : str

        """
        return self._experiment

    @experiment.setter
    def experiment(self, experiment):
        if self._experiment != experiment:  # reinitialize
            self.__init__(experiment)

    @property
    def testmatrix(self):
        """Return testmatrix keys as pandas.Series, read-only."""
        return pandas.Series(self._testmatrix.keys())

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
                             f'\n\n{self.testmatrix}')
        return self._testname

    @testname.setter
    def testname(self, testname):
        if isinstance(testname, int):
            try:
                testname = self.testmatrix.iloc[testname]
            except IndexError:
                raise IndexError(f'testname index {testname} out of range\n\n'
                                 f'{self.testmatrix}')
        elif isinstance(testname, str):
            if testname not in self._testmatrix:
                raise IndexError(f'testname {testname} not found in '
                                 f'\n{self.testmatrix}')
        if testname != self._testname:
            self._testname = testname
            self.reload = True

    @property
    def testplan(self):
        """Return testplan, read-only."""
        return self._testmatrix[self.testname]

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
                             f'valid range {self.shot_range}')
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
        return index[0], index[-1]

    @property
    def note(self):
        """Return shot note."""
        return self._note[self.testname].iloc[self._shot]

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
        file = os.path.join(self.datadir, file)
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
            localfile = self._download(file, subdir=subdir)
        return localfile

    def _download(self, file, parentdir='Daten', subdir=None):
        """
        Download file from ftp server.

        Parameters
        ----------
        file : str
            Filename, names of type '*.ext' permited.
        parentdir : str, optional
            Root dir on ftp server. The default is 'Daten'.
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
        with ftputil.FTPHost('ftp.psi.ch', 'sultan', '3g8S4Nbq') as host:
            chdir = [parentdir, self.experiment, subdir]
            for cd in chdir:
                if cd is not None:
                    try:
                        host.chdir(f'./{cd}')
                    except ftputil.error.PermanentError:
                        pwd = host.listdir('./')
                        raise ftputil.error.PermanentError(
                            f'folder {cd} not found in {pwd}')
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
                file = os.path.join(self.datadir, remotefile)
                host.download(remotefile, file)
            except ftputil.error.FTPError:
                err_txt = f'file {remotefile} '
                err_txt += f'not found in {host.listdir("./")}'
                raise ftputil.error.FTPError(err_txt)
        return file

    def load_testmatrix(self):
        """Load testmatrix."""
        testplan = os.path.join(self.datadir, self.locate('*.xls'))
        with pandas.ExcelFile(testplan) as xls:
            index = pandas.read_excel(xls, usecols=[0], header=None)
        self.testlabel = index[0][0]  # testplan name
        # extract testplan indices
        _testplan_index = {}
        previouslabel = None
        for i, label in enumerate(index.values):
            if isinstance(label[0], str):
                if label[0][:2] in ['AC', 'DC']:
                    _testplan_index[label[0]] = [i+1]
                    if previouslabel is not None:
                        _testplan_index[previouslabel].append(i-1)
                    previouslabel = label[0]
        if len(_testplan_index) > 0:
            _testplan_index[previouslabel].append(i)
        # store test matrix data
        previouscolumns = None
        with pandas.ExcelFile(testplan) as xls:
            for label in _testplan_index:
                index = _testplan_index[label]
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
                columns = [col for col in df.columns.get_level_values(0) if
                           isinstance(col, str)]
                note = [col for col in columns if 'note' in col.lower()]
                if note:
                    self._note[label] = df[note[0]]
                    self._note[label].columns = ['Note']
                    df.drop(columns=note, inplace=True, level=0)
                df.fillna(method='pad', inplace=True)
                df.reset_index(inplace=True)
                df.iloc[:, 0] = df.index
                self._testmatrix[label] = df

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
        return f'{self.shot[("File", None)]}.dat'

    def _load_datafile(self, testname=None, shot=None, subdir='ac/dat'):
        """
        Return sultan dataframe and associated shot metadata.

        Parameters
        ----------
        testname : str or int, optional
            Test identifier. The default is None.
        shot : int, optional
            Shot identifier. The default is None.
        subdir : str, optional
            Data subdirectory. The default is 'ac/dat'.

        Returns
        -------
        sultandata : pandas.DataFrame
            Shot data.
        sultanmetadata : pandas.Series
            Shot metadata.

        """
        filename = self._get_filename(testname, shot)
        datafile = self.locate(filename, subdir=subdir)
        datafile = os.path.join(self.datadir, datafile)
        sultandata = pandas.read_csv(datafile, encoding='ISO-8859-1')
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


class SultanPostProcess(FTPSultan):
    """Post processing methods for single leg sultan coupling loss data."""

    def __init__(self, experement):
        """
        Import data and initialize data structure.

        Parameters
        ----------
        sultandata : pandas.DataFrame
            Sultan test data.
        side : str
            Side of Sultan experement ['Left', 'Right'].

        Returns
        -------
        None.

        """
        FTPSultan.__init__(self, experement)  # link to sultan data
        self._side = None
        self._rawdata = None
        self._lowpassdata = None

    @staticmethod
    def _initialize_dataframe():
        """
        Return calclation dataframe.

        Returns
        -------
        dataframe : pandas.DataFrame
            Empty dataframe with time index and default columns names.

        """
        variables = [('t', 's'), ('mdot', 'kg/s'),
                     ('Tin', 'K'), ('Tout', 'K'),
                     ('Pin', 'Pa'), ('Pout', 'Pa'),
                     ('hin', 'J/Kg'), ('hout', 'J/Kg'),
                     ('Q', 'W')]
        columns = pandas.MultiIndex.from_tuples(variables)
        return pandas.DataFrame(columns=columns)

    @property
    def postprocess(self):
        """
        Manage postproces flags.

        Parameters
        ----------
        postprocess : bool
            Clear raw and lowpass data if True.

        Returns
        -------
        postprocess : pandas.Series
            Postproces flags.

        """
        return pandas.Series({'raw': self._rawdata is None,
                              'lowpass': self._lowpassdata is None})

    @postprocess.setter
    def postprocess(self, postprocess):
        if postprocess:
            self._rawdata = None
            self._lowpassdata = None

    @property
    def rawdata(self):
        """Return rawdata, read-only."""
        if self._rawdata is None:
            self._rawdata = self._extract_data(lowpass=False)
        return self._rawdata

    @property
    def lowpassdata(self):
        """Return filtered data, read-only."""
        if self._lowpassdata is None:
            self._lowpassdata = self._extract_data(lowpass=True)
        return self._lowpassdata

    @property
    def side(self):
        """
        Manage side property. (Re)initialize data if changed.

        Parameters
        ----------
        side : str
            Side of Sultan experement ['Left', 'Right'].

        Returns
        -------
        side : str

        """
        if self._side is None:
            raise IndexError('side of Sultan experement not set [Left, Right]')
        return self._side

    @side.setter
    def side(self, side):
        side = side.capitalize()
        if side not in ['Left', 'Right']:
            raise IndexError(f'side {side} not in [Left, Right]')
        if not hasattr(self, side):
            self._side = None
        update = side != self._side
        self._side = side
        if update:
            self.postprocess = True

    def _extract_data(self, lowpass=False):
        """
        Extract relivant data variables and calculate Q.

        Parameters
        ----------
        lowpass : bool, optional
            Apply lowpass filter.
            Window length set equal to 2.5*period of driving waveform.
            The default is False.

        Returns
        -------
        data : pandas.DataFrame
            ACloss dataframe.

        """
        data = self._initialize_dataframe()
        data['t'] = self.sultandata['Time']
        data['mdot'] = self.sultandata[f'dm/dt {self.side}'] * 1e-3
        for end in ['in', 'out']:
            data[f'T{end}'] = self.sultandata[f'T {end} {self.side}']
            data[f'P{end}'] = self.sultandata[f'P {end} {self.side}'] * 1e5
        if lowpass:
            dt = np.diff(data['t'], axis=0).mean()
            freq = self.shot[('P. Freq', 'Hz')]
            windowlength = int(2.5 / (dt*freq))
            if windowlength % 2 == 0:
                windowlength += 1
            for attribute in ['mdot', 'Tin', 'Tout', 'Pin', 'Pout']:
                data[attribute] = scipy.signal.savgol_filter(
                    np.squeeze(data[attribute]), windowlength, polyorder=3)
        for end in ['in', 'out']:  # Calculate enthapy
            T, P = data[f'T{end}'].values, data[f'P{end}'].values
            data[f'h{end}'] = CoolProp.PropsSI('H', 'T', T, 'P', P, 'Helium')
        # net heating
        data['Q'] = data[('mdot', 'kg/s')] * \
            (data[('hout', 'J/Kg')] - data[('hin', 'J/Kg')])
        # normalize
        Ipulse = float(re.findall(r'\d+', self.shot[('Ipulse', 'A')])[0])
        print(Ipulse)
        Bpulse = Ipulse * 0.2/230
        freq = self.shot[('P. Freq', 'Hz')]
        omega = 2*np.pi*freq
        Bdot = omega*Bpulse
        data['Qnorm'] = data['Q'] / Bdot**2
        return data

    def _threshold(self, data, threshold=0.9):
        """
        Return slice of first and last indices meeting condition.

        Condition evaluated as data > threshold * data.abs().max()

        Parameters
        ----------
        data : array-like
            Data vector.
        threshold : float, optional
            Threshold factor applied to data.abs().max(). The default is 0.9.

        Returns
        -------
        index : slice
            Threshold index.

        """
        Imax = data.abs().max()
        threshold_index = np.where(data.abs() > threshold*Imax)[0]
        index = slice(threshold_index[0], threshold_index[-1])
        return index

    def plot_single(self, variable, ax=None, lowpass=False):
        if lowpass:
            data = self.lowpassdata
        else:
            data = self.rawdata
        if variable not in data:
            raise IndexError(f'variable {variable} not in {data.columns}')
        if ax is None:
            ax = plt.gca()
        ax.plot(data.t, data[variable])


if __name__ == '__main__':

    spp = SultanPostProcess('CSJA_7')

    spp.testname = 0
    spp.shot = 13
    spp.side = 'left'
    spp.lowpassdata
    #sultan = spp.load_datafile('AC Loss before DC', 12)
    #sultanmetadata

    #SultanPostProcess

    spp.plot_single('Qnorm')
    spp.plot_single('Qnorm', lowpass=True)
