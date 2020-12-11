
from dataclasses import dataclass, field
import os


import pandas
import numpy as np
import scipy.signal
import scipy.integrate
import scipy.interpolate
import scipy.optimize
import CoolProp.CoolProp as CoolProp
from matplotlib.lines import Line2D

from nova.utilities.pyplot import plt
from nova.utilities.time import clock


@dataclass
class SultanPostProcess:
    """
    Post processing methods for single leg sultan coupling loss data.

    Parameters
    ----------
    experement : str
        Experement label.

    """

    _side: str = 'left'
    _Qdot_threshold = 0.75
    _iQdot = None
    _Bdot = None


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



