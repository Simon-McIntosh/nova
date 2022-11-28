
from dataclasses import dataclass, field, InitVar
from typing import Union
import os


import pandas
import numpy as np
import scipy.signal
import scipy.integrate
import scipy.interpolate
from matplotlib.lines import Line2D

from nova.thermalhydralic.sultan.testplan import TestPlan
from nova.thermalhydralic.sultan.shotinstance import ShotInstance
from nova.thermalhydralic.sultan.shotprofile import ShotProfile
from nova.thermalhydralic.sultan.sultanio import SultanIO

import matplotlib.pyplot as plt
from nova.utilities.time import clock

'''
    def plot(self, ax=None):
        """

        Parameters
        ----------
        ax : axis, optional
            plot axis. The default is None (plt.gca())

        Returns
        -------
        None

        """
        if ax is None:
            ax = plt.gca()
        ax.plot(t_eoh, Qdot_eoh, **self._get_marker(steady, 'eoh'))
        ax.plot(t_max, Qdot_max, **self._get_marker(steady, 'max'))
'''


@dataclass
class FrequencyResponse(SultanIO):
    """Post processing methods for single leg sultan coupling loss data."""

    shotprofile: Union[ShotInstance, TestPlan, str]
    shotinstance: ShotInstance = field(init=False, repr=False, default=None)
    testplan: TestPlan = field(init=False, repr=False, default=None)
    _data: pandas.DataFrame = field(init=False, repr=False, default=None)

    def __post_init__(self):
        """Link shot profile and shot instance methods."""
        if not isinstance(self.shotprofile, ShotInstance):
            self.shotprofile = ShotProfile(self.shotprofile)
        self.shotinstance = self.shotprofile.shotinstance
        self.testplan = self.shotinstance.testplan
        self._data = self.load_data()

    @property
    def data(self):
        """Return frequency response data."""
        if self.testplan.reload.response or self.shotprofile.reload.response:
            self._data = self.load_data()
        return self._data

    def load_data(self):
        """Extend SultanIO load_data."""
        data = SultanIO.load_data(self)
        self.testplan.reload.response = False
        self.shotprofile.reload.response = False
        return data

    @property
    def binaryfilepath(self):
        """Return full path of binary datafile."""
        return self.testplan.database.binary_filepath('response.h5')

    @property
    def filename(self):
        """Return frequency response filename."""
        experiment = self.testplan.experiment
        testname = self.testplan.testname
        side = self.shotprofile.side
        return f'{experiment}_{testname}_{side}'

    def _read_data(self):
        data = pandas.DataFrame(
            index=range(self.testplan.shotnumber),
            columns=['file', 'external', 'excitation', 'current', 'frequency',
                     'stop', 'maximum', 'impulse', 'steady'])
        header = f'Extracting frequency response: {self.filename}'
        tick = clock(self.testplan.shotnumber, header=header)
        for index, shot in enumerate(self.shotinstance.sequence()):
            profile = self.shotprofile
            metadata = self.shotinstance.metadata.droplevel(1)
            data.loc[
                index, ['file', 'external', 'current', 'frequency']] = \
                metadata.loc[['File', 'Be', 'Isample', 'frequency']].values
            data.loc[index, 'excitation'] = self.shotprofile.excitation_field
            shotresponse = ShotResponse(profile.lowpassdata, profile.heatindex)
            ####### >>>> ##### test shotresponse

            data.loc[
                index, ['stop', 'maximum', 'step', 'steady']] = \
                self.shotprofile.shotresponse.dataseries
            tick.tock()
        data.sort_values(['external', 'current', 'excitation', 'frequency'],
                         inplace=True)
        data.fillna(-1, inplace=True)
        return data


if __name__ == '__main__':

    testplan = TestPlan('CSJA_3', 0)
    response = FrequencyResponse(testplan)

    response.shotinstance.index = 20
    response.shotprofile.plot()


    '''
    #response.experiment = 'CSJA_5'
    response.testname = -1
    #print(response.testplan)
    print(response.shotprofile)
    response.side = 'Right'
    response.testname = -1
    response.experiment = 'CSJA_5'

    print(response.testname)
    response.experiment = 'CSJA_3'
    print(response.testname)
    '''


    '''
        @property
    def experiment(self):
        """Manage experiment name."""
        return self.testplan.experiment

    @experiment.setter
    def experiment(self, experiment):
        self.testplan.experiment = experiment

    @property
    def testname(self):
        """Manage test name."""
        return self.testplan.testname

    @testname.setter
    def testname(self, testname):
        self.testplan.testname = testname

    @property
    def testmode(self):
        """Manage test mode, ac or dc."""
        return self.testplan.testmode

    @testmode.setter
    def testmode(self, testmode):
        self.testplan.mode = testmode

    @property
    def side(self):
        """Manage leg side, Left or Right."""
        return self.shotprofile.side

    @side.setter
    def side(self, side):
        self.shotprofile.side = side
    '''


    '''
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
