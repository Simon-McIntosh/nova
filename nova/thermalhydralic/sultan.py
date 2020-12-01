
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
from matplotlib.lines import Line2D

from nova.definitions import root_dir
from nova.utilities.pyplot import plt
from nova.utilities.time import clock
from nova.utilities.IO import pythonIO


class SultanPostProcess(FTPSultan):
    """Post processing methods for single leg sultan coupling loss data."""

    _attributes = FTPSultan._attributes + ['side']
    _default_attributes = {**FTPSultan._default_attributes,
                           **{'side': 'Left'}}
    _input_attributes = FTPSultan._input_attributes + ['side']

    def __init__(self, *args, **kwargs):
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
        self._side = None
        FTPSultan.__init__(self, *args, **kwargs)  # link to sultan
        self._rawdata = None
        self._lowpassdata = None
        self._Qdot_threshold = 0.75
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
        variables = [('t', 's'), ('mdot', 'kg/s'), ('Ipulse', 'A'),
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
        data['Ipulse'] = self.sultandata['PS EEI (I)']
        for end in ['in', 'out']:
            data[f'T{end}'] = self.sultandata[f'T {end} {self.side}']
            data[f'P{end}'] = self.sultandata[f'P {end} {self.side}'] * 1e5
        if lowpass:
            dt = np.diff(data['t'], axis=0).mean()
            freq = self.shot[('frequency', 'Hz')]
            windowlength = int(2.5 / (dt*freq))
            if windowlength % 2 == 0:
                windowlength += 1
            if windowlength < 5:
                windowlength = 5
            for attribute in ['mdot', 'Ipulse', 'Tin', 'Tout', 'Pin', 'Pout']:
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
            Ipulse = float(re.findall(r'\d+', Ipulse)[0])
        except TypeError:
            Ipulse = 230
        Be = Ipulse * 0.2/230  # excitation field amplitude
        return Be

    def _evaluate_Bdot(self):
        freq = self.shot[('frequency', 'Hz')]
        omega = 2*np.pi*freq
        self._Bdot = omega*self.Be  # pulse field rate amplitude

    def _zero_offset(self):
        """Correct start of heating offset in Qdot_norm."""
        zero_offset = self.lowpassdata.loc[self.iQdot.start,
                                           ('Qdot_norm', 'W')]
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
            Heating idexed as Ipulse.abs > Qdot_threshold * Ipulse.abs.max.

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

        Evaluated as Ipulse.abs() > Qdot_threshold * Ipulse.abs().max()

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
        Ipulse = self.sultandata['PS EEI (I)']
        Imax = Ipulse.abs().max()
        threshold_index = np.where(Ipulse.abs() >= self.Qdot_threshold*Imax)[0]
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
        dQdot_heat = Qdot_norm[self.iQdot].max() - Qdot_norm[self.iQdot].min()
        argmax = Qdot_norm.argmax()
        t_max = t[argmax]
        Qdot_max = Qdot_norm[argmax]

        steady = True
        if Qdot_max/Qdot_eoh > transient_factor:
            steady = False
        elif Qdot_norm[self.iQdot.start] > Qdot_norm[self.iQdot.stop-1]:
            steady = False
        elif Qdot_norm[self.iQdot.stop-1] - Qdot_norm[self.iQdot].min() < \
                0.95 * dQdot_heat:
            steady = False
        if plot:
            if ax is None:
                ax = plt.gca()
            ax.plot(t_eoh, Qdot_eoh, **self._get_marker(steady, 'eoh'))
            ax.plot(t_max, Qdot_max, **self._get_marker(steady, 'max'))
        return t_eoh, Qdot_eoh, t_max, Qdot_max, steady

    def _get_marker(self, steady=True, location='max'):
        marker = {'ls': 'none', 'alpha': 1, 'ms': 4, 'mew': 1}
        if location == 'eoh':
            marker.update({'color': 'C6', 'label': 'eoh', 'marker': 'o'})
        elif location == 'max':
            marker.update({'color': 'C4', 'label': 'max', 'marker': 'd'})
        else:
            raise IndexError(f'location {location} not in [eof, max]')
        if not steady:
            marker.update({'mfc': 'w'})
        return marker

    def _initialize_testdata(self):
        testdata = self.testplan.loc[:, ['Be', 'Isample', 'frequency']]
        if ('frequency', 'Hz/duration') in testdata:
            testdata.drop(columns=[('frequency', 'Hz/duration')],
                          inplace=True)
        testdata = testdata.droplevel(1, axis=1)
        testdata['B'] = [self._transform_Ipulse(Ipulse)
                         for Ipulse in self.testplan.loc[:, ('Ipulse', 'A')]]
        testdata['Bdot'] = 2*np.pi*testdata['frequency'] * testdata['B']
        self.testdata = testdata

    def _extract_testdata(self):
        self._initialize_testdata()
        self._extract_response()

    def load_testdata(self, **kwargs):
        """Load testdata from file."""
        read_txt = kwargs.pop('read_txt', self.read_txt)
        if read_txt or not os.path.isfile(self.testdata_filename):
            self._extract_testdata()
            self._save_testdata()
        else:
            self.testdata = pandas.read_parquet(self.testdata_filename)

    def _extract_response(self):
        header = 'Extracting frequency response: '
        header += f'{os.path.split(self.testdata_filename)[1].split(".")[0]}'
        tick = clock(self.shot_range[1], header=header)
        for shot in range(*self.shot_range):
            self.shot = shot
            response = self.extract_response()
            self.testdata.loc[shot, ['Qdot_eof', 'Qdot_max', 'steady']] = \
                response[1], response[3], response[-1]
            tick.tock()
        self.testdata.sort_values(['Be', 'Isample', 'B',
                                   'frequency'], inplace=True)

    @property
    def testdata_filename(self):
        """Return testdata filename."""
        file = f'{self.experiment}_{self.mode.upper()}'
        file += f'{self.testindex}{self.side[0]}.pq'
        return os.path.join(self.localdir, file)

    def _save_testdata(self):
        self.testdata.to_parquet(self.testdata_filename)

    def plot_response(self, unsteady=False, ax=None, color='C0',
                      Bexternal=None, Isample=0):
        """Plot ensemble response."""
        if ax is None:
            ax = plt.gca()
        field_marker = {1: 'D', 2: 11, 9: 10}
        plot_kwargs = self._get_marker()
        plot_kwargs.update({'color': color})
        if Bexternal is None:
            Bexternal = self.testdata.Be.unique()
        else:
            if not pandas.api.types.is_list_like(Bexternal):
                Bexternal = [Bexternal]
        if Isample is None:
            Isample = self.testdata.Isample.unique()
        else:
            if not pandas.api.types.is_list_like(Isample):
                Isample = [Isample]
        for Be in Bexternal:
            marker = field_marker.get(Be, '*')
            for Is in Isample:
                # index data
                index = self.testdata.Be == Be
                index &= self.testdata.Isample == Is
                testdata = self.testdata.loc[index]
                # sort
                testdata = testdata.sort_values(['Bdot'])
                frequency = testdata.frequency
                Qdot = testdata.Qdot_max
                steady = testdata.steady.astype(bool)
                plot_kwargs.update({'ls': '-', 'marker': marker,
                                    'color': color, 'mfc': color,
                                    'ms': 6})

                ax.plot(frequency[steady], Qdot[steady],
                        **plot_kwargs)
                if unsteady:
                    plot_kwargs.update({'mfc': 'none', 'ls': 'none'})
                    ax.plot(frequency[~steady], Qdot[~steady],
                            **plot_kwargs)
        ax.set_yscale('log')
        ax.set_xscale('log')
        return sum(index) > 0

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
        Ipulse = self.shot[('Ipulse', 'A')][1:]
        f = self.shot[('frequency', 'Hz')]
        ax.set_title(rf'$I_{{pulse}}$ = {Ipulse }(2$\pi$ {f} $t$)')

    def plot_Qdot_norm(self):
        self._zero_offset()
        self.plot_single('Qdot_norm')
        self.plot_single('Qdot_norm', lowpass=True)
        self.title()
        self.extract_response(plot=True)
        plt.legend(loc='upper right')


class SultanEnsemble(SultanPostProcess):

    def __init__(self, *args, **kwargs):
        SultanPostProcess.__init__(self, *args, **kwargs)
        if self.isvalid:
            self.load_testdata()

    @property
    def substr(self):
        """
        Manage experiment substr.

        Parameters
        ----------
        substr : str or None
            Selection substr searched for in experiment name.

        Returns
        -------
        substr : str
            Returns self.experiment if null.

        """
        if self._substr:
            return self._substr
        else:
            return self.experiment

    @substr.setter
    def substr(self, substr):
        self._substr = substr

    def extract(self, substr=None):
        self.substr = substr
        for experiment in self.listdir(self.substr):
            self.experiment = experiment
            for testname in self.testkeys:
                self.testname = testname
                for side in ['left', 'right']:
                    self.side = side
                    self.load_testdata()

    def plot_ensemble(self, substr=None, testname=0, ax=None):
        self.substr = substr
        if ax is None:
            ax = plt.gca()
        legend = [[], []]
        for i, experiment in enumerate(self.listdir(self.substr)):
            self.experiment = experiment
            self.side = 'left'
            self.testname = testname
            self.load_testdata()
            color = f'C{i % 10}'
            if self.plot_response(unsteady=True, color=color):
                legend[0].append(Line2D([0], [0], color=color, ls='-'))
                legend[1].append(self.strand)
        # set marker legend
        marker_legend = plt.legend(
            [Line2D([0], [0], marker=marker, color='C7', mew=1,
                    ms=5, ls='none', mfc=mfc)
             for mfc, marker in zip(['C7', 'none', 'C7', 'C7', 'C7'],
                                    ['o', 'o', 'D', 11, 10])],
            ['steady', 'unsteady (under-estimate)',
             r'$B_e$ 1T', r'$B_e$ 2T', r'$B_e$ 9T'],
            loc='lower left')
        # label axes and add primary legend
        ax.set_xlabel('$f$ Hz')
        ax.set_ylabel(r'$\dot{Q}$ W')
        ax.legend(*legend, loc='upper right', ncol=1,
                  bbox_to_anchor=(1.1, 1))
        ax.add_artist(marker_legend)
        plt.despine()
        test_str = 'virgin' if testname == 0 else 'cycled'
        ax.set_title('Sultan AC loss CS frequency response '
                     f'({test_str})')


if __name__ == '__main__':

    #ftp = FTPSultan('CSJA_3', 0, 0)

    #spp = SultanPostProcess('CSJA_3', read_txt=True)
    #spp.testname = 11
    #spp.shot = 1
    #spp.side = 'Left'
    #spp.plot_Qdot_norm()

    #se = SultanEnsemble('CSJA_10', 0, 0, side='right')
    #se = SultanEnsemble()
    #se.plot_Qdot_norm()

    #se.extract('KOCS')
    se = SultanEnsemble()
    se.plot_ensemble('CSJA', testname=-1)
    #se.plot_response(unsteady=True)

    #plt.figure()
    se = SultanEnsemble('JACS_1', -1, 0, side='left', read_txt=False)
    #se.plot_response(unsteady=True, color='k')

    '''
    V = A Bdot
    V = IR
    P = IV = IA Bdot = 1/R (A Bdot)**2 = A**2/R |Bdot|**2
    '''

    '''
    import scipy.signal

    system = scipy.signal.lti([], [3, 50], 3000)
    f = np.logspace(-1, 1.6, 50)
    mag = scipy.signal.bode(system, 2*np.pi*f)[1]
    Qbode = 10**(mag/20)
    plt.plot(f, Qbode)
    '''



