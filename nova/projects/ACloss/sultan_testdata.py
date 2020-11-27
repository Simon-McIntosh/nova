
import os
import glob
import re
from warnings import warn

import ftputil
import pandas
import numpy as np
import scipy.signal
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import CoolProp.CoolProp as CoolProp

from nova.definitions import root_dir
from nova.utilities.pyplot import plt
from nova.utilities.time import clock
from nova.utilities.IO import pythonIO


class FTPSultan(pythonIO):
    """
    Provide access to sultan data.

    Datafiles stored localy. Files downloaded from ftp server if not found.

    """

    def __init__(self, experiment, read_txt=False):
        self._experiment = experiment
        self.datadir = self._set_datadir()
        self._testname = None  # test identifier
        self._shot = None  # shot identifier
        self._reload = True  # reload data from file
        self._testmatrix = {}
        self._note = {}
        self._sultandata = None
        self.read_txt = read_txt
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
        return pandas.Series(list(self._testmatrix.keys()))

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
    def testindex(self):
        """Return testname index."""
        return next((i for i, name in enumerate(self._testmatrix)
                     if name == self.testname))

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
                    warn_txt = f'multiple files found {file} > {ftpfile}'
                    warn_txt += f'\nusing {ftpfile[0]}'
                    warn(warn_txt)
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
        filepath = os.path.join(self.datadir, 'testmatrix')
        if not os.path.isfile(filepath + '.pk') or read_txt:
            self.read_testmatrix()
            self.save_pickle(filepath,
                             ['strand', '_testmatrix'])
        else:
            self.load_pickle(filepath)

    def read_testmatrix(self, mode='AC'):
        """Load testmatrix."""
        try:
            testplan = os.path.join(self.datadir, self.locate('*.xls'))
        except ftputil.error.PermanentError as error:  # folder not found
            os.rmdir(self.datadir)
            raise ftputil.error.PermanentError(f'{error}')
        with pandas.ExcelFile(testplan) as xls:
            index = pandas.read_excel(xls, usecols=[0], header=None)
        strand = next(label[0] for label in index.values
                      if 'XXYYZZ' in label[0])
        self.strand = strand.split('XXYYZZ')[0][:-1] + mode[0]
        self.strand = self.strand.replace('-', '')
        # extract testplan indices
        _testplan_index = {}
        testname = None
        skip_file = False
        for i, label in enumerate(index.values):
            if isinstance(label[0], str):
                if label[0][:len(self.strand)] == self.strand \
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
                        isnext_strand = self.strand in nextlabel
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
                        if self.strand in testname:
                            testname = f'noname {j}'
                    except TypeError:
                        testname = f'noname {j}'
                    testname = testname.split(':')[0].split('(')[0]
                    if testname in _testplan_index:
                        testname += f' {j}'
                    _testplan_index[testname] = [j, 0]
        # remove open indices
        for testname in list(_testplan_index.keys()):
            if _testplan_index[testname][1] == 0:
                _testplan_index.pop(testname)
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
                # reset index
                df.reset_index(inplace=True)
                df.iloc[:, 0] = df.index
                # format frequency
                frequency_duration = ('P. Freq', 'Hz/duration')
                frequency_Hz = ('P. Freq', 'Hz')
                if frequency_duration in df:
                    df[('P. Freq', 'Hz')] = df[frequency_duration].apply(
                            self._format_frequency)
                elif frequency_Hz in df:
                    if df[frequency_Hz].dtype == object:
                        df[frequency_Hz] = df[frequency_Hz].apply(
                            self._format_frequency)
                self._testmatrix[label] = df

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
        try:
            datafile = self.locate(filename, subdir=subdir)
        except ftputil.error.PermanentError:
            datafile = self.locate(filename, subdir='TEST/AC/ACdat')
        datafile = os.path.join(self.datadir, datafile)
        sultandata = pandas.read_csv(datafile, encoding='ISO-8859-1')
        columns = {}
        for c in sultandata.columns:
            if 'left' in c or 'right' in c:
                columns[c] = c.replace('left', 'Left')
                columns[c] = columns[c].replace('right', 'Right')
        sultandata.rename(columns=columns, inplace=True)
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

    def __init__(self, experement, read_txt=False):
        """
        Import data and initialize data structure.

        Parameters
        ----------
        experement : str
            Experement label.

        Returns
        -------
        None.

        """
        FTPSultan.__init__(self, experement, read_txt)  # link to sultan data
        self._side = None
        self._rawdata = None
        self._lowpassdata = None
        self._Qdot_threshold = 0.95
        self._iQdot = None
        self._Bdot = None

    @staticmethod
    def _initialize_dataframe():
        """
        Return calclation dataframe.

        Returns
        -------
        dataframe : pandas.DataFrame
            Empty dataframe with time index and default columns names.

        """
        variables = [('t', 's'), ('mdot', 'kg/s'), ('Ips', 'A'),
                     ('Tin', 'K'), ('Tout', 'K'),
                     ('Pin', 'Pa'), ('Pout', 'Pa'),
                     ('hin', 'J/Kg'), ('hout', 'J/Kg'),
                     ('Qdot', 'W'), ('Qdot_norm', 'W')]
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
                              'lowpass': self._lowpassdata is None,
                              'iQdot': self._iQdot is None,
                              'Bdot': self._Bdot is None})

    @postprocess.setter
    def postprocess(self, postprocess):
        if postprocess:
            self._rawdata = None
            self._lowpassdata = None
            self._iQdot = None
            self._Bdot = None

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
        Extract relivant data variables and calculate Qdot.

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
        data['Ips'] = self.sultandata['PS EEI (I)']
        for end in ['in', 'out']:
            data[f'T{end}'] = self.sultandata[f'T {end} {self.side}']
            data[f'P{end}'] = self.sultandata[f'P {end} {self.side}'] * 1e5
        if lowpass:
            dt = np.diff(data['t'], axis=0).mean()
            freq = self.shot[('P. Freq', 'Hz')]
            windowlength = int(2.5 / (dt*freq))
            if windowlength % 2 == 0:
                windowlength += 1
            if windowlength < 5:
                windowlength = 5
            for attribute in ['mdot', 'Ips', 'Tin', 'Tout', 'Pin', 'Pout']:
                data[attribute] = scipy.signal.savgol_filter(
                    np.squeeze(data[attribute]), windowlength, polyorder=3)
        for end in ['in', 'out']:  # Calculate enthapy
            T, P = data[f'T{end}'].values, data[f'P{end}'].values
            data[f'h{end}'] = CoolProp.PropsSI('H', 'T', T, 'P', P, 'Helium')
        # net heating
        data['Qdot'] = data[('mdot', 'kg/s')] * \
            (data[('hout', 'J/Kg')] - data[('hin', 'J/Kg')])
        # normalize Qdot heating by |Bdot|**2
        data['Qdot_norm'] = data['Qdot'] / self.Bdot**2
        return data

    @property
    def Bdot(self):
        """Return field rate amplitude."""
        if self._Bdot is None:
            self._evaluate_Bdot()
        return self._Bdot

    @property
    def Be(self):
        """Return amplitude of excitation field, T."""
        return self._transform_Ipulse(self.shot[('Ipulse', 'A')])

    def _transform_Ipulse(self, Ipulse):
        """
        Return excitation field.

        Parameters
        ----------
        Ipulse : str
            Sultan Ipulse field.

        Returns
        -------
        Be : float
            Excitation field.

        """
        try:
            Ips = float(re.findall(r'\d+', Ipulse)[0])
        except TypeError:
            Ips = 230
        Be = Ips * 0.2/230  # excitation field amplitude
        return Be

    def _evaluate_Bdot(self):
        freq = self.shot[('P. Freq', 'Hz')]
        omega = 2*np.pi*freq
        self._Bdot = omega*self.Be  # pulse field rate amplitude

    def _zero_offset(self):
        """Correct t=0 offset in Qdot_norm heating."""
        zero_offset = self.lowpassdata.loc[0, ('Qdot_norm', 'W')]
        if not np.isclose(zero_offset, 0):
            for attribute in ['rawdata', 'lowpassdata']:
                data = getattr(self, attribute)
                data['Qdot_norm'] -= zero_offset

    @property
    def Qdot_threshold(self):
        """
        Manage heat threshold parameter.

        Parameters
        ----------
        Qdot_threshold : float
            Heating idexed as Ips.abs > Qdot_threshold * Ips.abs.max.

        Raises
        ------
        ValueError
            Qdot_threshold must lie between 0 and 1.

        Returns
        -------
        Qdot_threshold : float

        """
        return self._Qdot_threshold

    @Qdot_threshold.setter
    def Qdot_threshold(self, Qdot_threshold):
        if Qdot_threshold != self._Qdot_threshold:
            self._iQdot = None
        if Qdot_threshold < 0 or Qdot_threshold > 1:
            raise ValueError(f'heat threshold {Qdot_threshold} '
                             'must lie between 0 and 1')
        self._Qdot_threshold = Qdot_threshold

    @property
    def iQdot(self):
        """Return heat index, slice."""
        if self._iQdot is None:
            self._evaluate_iQdot()
        return self._iQdot

    def _evaluate_iQdot(self):
        """
        Return slice of first and last indices meeting threshold condition.

        Condition evaluated as Ips.abs() > Qdot_threshold * Ips.abs().max()

        Parameters
        ----------
        data : array-like
            Data vector.
        Qdot_threshold : float, optional property
            Threshold factor applied to data.abs().max(). The default is 0.95.

        Returns
        -------
        index : slice
            Threshold index.

        """
        Ips = self.sultandata['PS EEI (I)']
        Imax = Ips.abs().max()
        threshold_index = np.where(Ips.abs() >= self.Qdot_threshold*Imax)[0]
        self._iQdot = slice(threshold_index[0], threshold_index[-1]+1)

    def extract_response(self, transient_factor=1.05, plot=False, ax=None):
        """
        Extract heating response at end of heat and max heat.

        Flag transient when max heat >> end of heat.

        Parameters
        ----------
        transient_factor : float, optional
            Limit factor applied to ratio of eoh and max heat.
            Heating is considered transient of ratio exceeds factor.
            The default is 1.05.
        plot : bool, optional
            plotting flag. The default is False
        ax : axis, optional
            plot axis. The default is None (plt.gca())

        Returns
        -------
        t_eoh : float
            end of heating time.
        Qdot_eoh : float
            end of heating value (Qdot_norm).
        t_max : float
            max heating time.
        Qdot_max : float
            max heating value (Qdot_norm).
        steady : bool
            transient flag, False if Qdot_max/Qdot_eoh > transient_factor.

        """
        # extract lowpass data
        self._zero_offset()
        t = self.lowpassdata[('t', 's')]
        Qdot_norm = self.lowpassdata[('Qdot_norm', 'W')]
        # end of heating
        t_eoh = t[self.iQdot.stop-1]
        Qdot_eoh = Qdot_norm[self.iQdot.stop-1]
        argmax = Qdot_norm.argmax()
        t_max = t[argmax]
        Qdot_max = Qdot_norm[argmax]
        steady = True
        if Qdot_max/Qdot_eoh > transient_factor:
            steady = False
        elif t[self.iQdot][Qdot_norm[self.iQdot].argmax()] - t_eoh > 1:
            steady = False
        if plot:
            if ax is None:
                ax = plt.gca()
            ax.plot(t_eoh, Qdot_eoh, **self._get_marker(steady, 'eoh'))
            ax.plot(t_max, Qdot_max, **self._get_marker(steady, 'max'))
        return t_eoh, Qdot_eoh, t_max, Qdot_max, steady

    def _get_marker(self, steady, location):
        marker = {'ls': 'none', 'alpha': 1}
        if location == 'eoh':
            marker.update({'color': 'C6', 'label': 'eoh', 'marker': 'o'})
        elif location == 'max':
            marker.update({'color': 'C4', 'label': 'max', 'marker': 'd'})
        else:
            raise IndexError(f'location {location} not in [eof, max]')
        if steady:
            marker.update({'ms': 6, 'mew': 1.5})
        else:
            marker.update({'ms': 6, 'mew': 1.5, 'mfc': 'w'})
        return marker

    def plot_single(self, variable, ax=None, lowpass=False):
        self._zero_offset()
        if lowpass:
            data = self.lowpassdata
        else:
            data = self.rawdata
        if variable not in data:
            raise IndexError(f'variable {variable} not in {data.columns}')
        if ax is None:
            ax = plt.gca()
        bg_color = 0.4 * np.ones(3) if lowpass else 'lightgray'
        color = 'C3' if lowpass else 'C0'
        label = 'lowpass' if lowpass else 'raw'
        ax.plot(data.t, data[variable], color=bg_color)
        ax.plot(data.t[self.iQdot], data[variable][self.iQdot],
                color=color, label=label)
        ax.legend()
        ax.set_xlabel('$t$ s')
        ax.set_ylabel(r'$\hat{\dot{Q}}$ W')
        plt.despine()

    def title(self, ax=None):
        if ax is None:
            ax = plt.gca()
        I = self.shot[('Ipulse', 'A')][1:]
        f = self.shot[('P. Freq', 'Hz')]
        ax.set_title(rf'$I_{{ps}}$ = {I}(2$\pi$ {f} $t$)')

    def plot_Qdot_norm(self):
        self._zero_offset()
        self.plot_single('Qdot_norm')
        self.plot_single('Qdot_norm', lowpass=True)
        self.title()
        self.extract_response(plot=True)
        plt.legend(loc='upper right')


class SultanEnsemble(SultanPostProcess):

    def __init__(self, experiment, testname, side, read_txt=False):
        SultanPostProcess.__init__(self, experiment, read_txt)
        self.testname = testname
        self.side = side
        self.read_txt = read_txt
        self.load_testdata()

    def load_testdata(self, **kwargs):
        """Load testdata from file."""
        read_txt = kwargs.get('read_txt', self.read_txt)
        if read_txt or not os.path.isfile(self.ensemble_filename):
            self._extract_testdata()
            self._save_testdata()
        else:
            self.testdata = pandas.read_parquet(self.ensemble_filename)

    def _extract_testdata(self):
        self._initialize_testdata()
        self._extract_response()

    def _initialize_testdata(self):
        try:
            testdata = self.testplan.loc[:, ['B Sultan', 'P. Freq']]
        except KeyError:
            testdata = self.testplan.loc[:, ['B SULTAN', 'P. Freq']]
        if ('P. Freq', 'Hz/duration') in testdata:
            testdata.drop(columns=[('P. Freq', 'Hz/duration')],
                          inplace=True)
        testdata = testdata.droplevel(1, axis=1)
        testdata.rename(columns={'B Sultan': 'Be', 'P. Freq': 'f'},
                        inplace=True)
        testdata['B'] = [self._transform_Ipulse(Ips)
                         for Ips in self.testplan.loc[:, ('Ipulse', 'A')]]
        testdata['Bdot'] = 2*np.pi*testdata['f'] * testdata['B']
        self.testdata = testdata

    def _extract_response(self):
        tick = clock(self.shot_range[1],
                     header='extracting frequency response')
        for shot in range(*self.shot_range):
            self.shot = shot
            response = self.extract_response()
            self.testdata.loc[shot, ['Qdot_eof', 'Qdot_max', 'steady']] = \
                response[1], response[3], response[-1]
            tick.tock()
        self.testdata.sort_values(['Be', 'f', 'B'], inplace=True)

    @property
    def ensemble_filename(self):
        """Return ensemble filename."""
        file = f'{self.experiment}_test{self.testindex}_{self.side}.parquet'
        return os.path.join(self.datadir, file)

    def _save_testdata(self):
        self.testdata.to_parquet(self.ensemble_filename)

    def plot_response(self, ax=None):
        """Plot ensemble response."""
        if ax is None:
            ax = plt.gca()
        for Be in self.testdata.Be.unique():
            index = self.testdata.Be == Be
            f = self.testdata.f[index]
            Qdot = self.testdata.Qdot_max[index]
            steady = self.testdata.steady[index].astype(bool)
            unsteady_marker = self._get_marker(False, 'max')
            steady_marker = self._get_marker(True, 'max')
            steady_marker.update({'ls': '-'})
            ax.plot(f[~steady], Qdot[~steady], **unsteady_marker)
            ax.plot(f[steady], Qdot[steady], **steady_marker)
        ax.set_yscale('log')
        ax.set_xscale('log')


if __name__ == '__main__':

    #spp = SultanPostProcess('MIT_Alpha', read_txt=True)
    #spp.testname = 11
    #spp.shot = 1
    #spp.side = 'Left'
    #spp.plot_Qdot_norm()

    se = SultanEnsemble('CSJA_3', -1, 'left', read_txt=True)
    se.plot_response()

    se.shot = 26
    se.plot_Qdot_norm()
    #spp = SultanPostProcess('CSJA_7')
    #spp.testname = 0
    #spp.shot = 12
    #spp.side = 'left'


    #spp.plot_Qdot_norm()

    #plt.figure()
    #spp.plot_single('Ips')
    #spp.plot_single('Ips', lowpass=True)

