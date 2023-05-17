import nep_data
from amigo.IO import class_dir
from amigo.pyplot import plt
import pandas as pd
from os.path import join
from nep.DINA.read_plasma import read_plasma
from nep.DINA.read_dina import timeconstant
import numpy as np
from scipy.interpolate import interp1d


class elm_data:

    def __init__(self):
        self.ddir = join(class_dir(nep_data), 'scenario_database', 'elm')
        self.readENP()
        self.readLTC_lower()
        self.pl = read_plasma('disruptions')

    def readLTC_lower(self):
        xlsfile = 'ELM_LOW_MD_DW_II_16ms_ECQ_AllCurrents.xlsx'
        filename = join(self.ddir, xlsfile)
        self.LTC = {'lower': {}}
        with pd.ExcelFile(filename) as xlsx:
            self.LTC['lower']['coils'] = pd.read_excel(
                    xlsx, sheet_name='data', usecols=range(3))
            self.LTC['lower']['coils'].columns = ['t', 'Iind', 'Id']
            self.LTC['lower']['coils'].loc[:, 't'] *= 1e-3  # ms to s
            self.LTC['lower']['coils'].loc[:, 'Iind':'Id'] *= 1e3  # kA to A
            if self.LTC['lower']['coils'].loc[0, 'Id'] < 0:
                # discharge from positive
                self.LTC['lower']['coils'].loc[:, 'Id'] *= -1
        self.get_discharge('LTC')

    def readENP(self):
        xlsfile = 'Currents_induced_in_ELM_and_VS_coils_at__Y9FVQJ_v1_0.xlsx'
        filename = join(self.ddir, xlsfile)
        self.ENP = {}
        with pd.ExcelFile(filename) as xlsx:
            for sheet in xlsx.sheet_names:
                name = sheet.split()[0].lower()
                self.ENP[name] = {}
                name = sheet.split()[0].lower()
                self.ENP[name]['coils'] = pd.read_excel(
                        xlsx, sheet_name=sheet, skiprows=7)
                self.ENP[name]['coils'].columns = ['t', 'Iind', 'Id']
                self.ENP[name]['coils'].loc[:, 'Iind':'Id'] *= 1e3  # -kA to A
                if self.ENP[name]['coils'].loc[0, 'Id'] < 0:
                    # discharge from positive
                    self.ENP[name]['coils'].loc[:, 'Id'] *= -1
        self.get_discharge('ENP')

    def get_data(self, code):
        return getattr(self, code)

    def get_discharge(self, code, eps=1e-5):
        data = self.get_data(code)
        for name in data:
            Io = data[name]['coils'].loc[0, 'Id']
            trim_index = next(i for i, Id in
                              enumerate(data[name]['coils'].loc[:, 'Id'])
                              if Id < (1-eps)*Io)-1
            data[name]['discharge'] = pd.DataFrame()
            data[name]['discharge']['t'] = \
                data[name]['coils'].loc[trim_index:, 't']
            data[name]['discharge']['t'] -= \
                data[name]['coils'].loc[trim_index, 't']
            data[name]['discharge']['I'] = \
                data[name]['coils'].loc[trim_index:, 'Id']
            data[name]['Id'] = \
                interp1d(data[name]['discharge']['t'],
                         data[name]['discharge']['I'],
                         bounds_error=False,
                         fill_value=(data[name]['discharge'].iloc[0]['I'],
                                     data[name]['discharge'].iloc[-1]['I']))
            tc = timeconstant(data[name]['discharge']['t'],
                              data[name]['discharge']['I'],
                              trim_fraction=0.3)
            data[name]['Io'], data[name]['tau'] = \
                tc.nfit(2, plot=False, Io=Io, )[:2]
            data[name]['Ifit'] = \
                np.zeros(len(data[name]['discharge']['t']))
            for Io, tau in zip(data[name]['Io'], data[name]['tau']):
                data[name]['Ifit'] += \
                    Io*np.exp(-data[name]['discharge']['t']/tau)

    def plot_discharge(self, code, t_trip=0, coils=None, ax=None, ic=0):
        data = self.get_data(code)
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        if coils is None:
            coils = data.keys()
        for i, name in enumerate(coils):
            label = rf'{code} $\tau$ ('
            for tau_ in data[name]['tau']:
                label += f'{1e3*tau_:1.1f}, '
            label = label[:-2] + ')ms'
            ax.plot(t_trip+data[name]['discharge']['t'],
                    1e-3*data[name]['discharge']['I'], f'C{i+ic}',
                    label=label)
            ax.plot(t_trip+data[name]['discharge']['t'],
                    1e-3*data[name]['discharge']['I'], '-.',
                    color='gray', label='', alpha=0.75)
        ax.legend()
        plt.despine()
        ax.set_ylabel('discharge $I$ kA')

    def plot_induced(self, code, coils=None, ax=None, ic=0):
        data = self.get_data(code)
        if coils is None:
            coils = data.keys()
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        for i, name in enumerate(coils):
            label = f'{code} {name}'
            ax.plot(data[name]['coils']['t'],
                    1e-3*data[name]['coils']['Iind'], f'C{i+ic}', label=label)
        ax.set_ylabel('induced $I$ kA')
        plt.despine()

    def get_t_trip(self, trip='t_cq', scenario=0):
        self.pl.load_file(scenario)
        trip = self.pl.get_vs3_trip()
        return trip['t_cq']

    def plot_combined(self, code, t_trip=0, coils=None, ax=None, ic=0):
        data = self.get_data(code)
        if coils is None:
            coils = data.keys()
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        for i, name in enumerate(coils):
            t, Iind = data[name]['coils']['t'], data[name]['coils']['Iind']
            Icombined = Iind + data[name]['Id'](t-t_trip)
            label = f'{code} $I_{{max}}$ {1e-3*np.max(Icombined):1.1f}kA'
            ax.plot(t, 1e-3*Icombined, f'C{i+ic}', label=label)
        ax.legend()
        ax.set_xlim([0.26, 0.38])
        ax.set_ylim([15, 26])
        ax.set_ylabel('combined $I$ kA')
        plt.despine()

    def plot_plasma_current(self, scenario=0, ax=None):
        self.pl.load_file(scenario)
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        ax.plot(self.pl.t, 1e-6*self.pl.Ipl, 'C3', label=self.pl.name)
        ax.legend()
        ax.set_xlabel('$t$ s')
        ax.set_ylabel('$I_{pl}$ MA')

    def compare_discharge(self, t_trip=0, coils=['lower'], ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        elm.plot_discharge('LTC', t_trip=t_trip, coils=coils, ax=ax, ic=0)
        elm.plot_discharge('ENP', t_trip=t_trip, coils=coils, ax=ax, ic=1)

    def compare_induced(self, coils=['lower'], ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        elm.plot_induced('LTC', coils=coils, ax=ax, ic=0)
        elm.plot_induced('ENP', coils=coils, ax=ax, ic=1)

    def compare_combined(self, t_trip=0, coils=['lower'], ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        elm.plot_combined('LTC', t_trip=t_trip, coils=coils, ax=ax, ic=0)
        elm.plot_combined('ENP', t_trip=t_trip, coils=coils, ax=ax, ic=1)

    def compare(self, coil='lower'):
        ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True)[1]
        t_trip = elm.get_t_trip()
        elm.compare_discharge(coils=[coil], t_trip=t_trip, ax=ax[0])
        elm.compare_induced(coils=[coil], ax=ax[1])
        elm.compare_combined(coils=[coil], t_trip=t_trip, ax=ax[2])
        ax[-1].set_xlabel('$t$ s')
        plt.detick(ax)
        ax[0].set_title(coil+' elm')


if __name__ is '__main__':

    elm = elm_data()
    elm.compare()

