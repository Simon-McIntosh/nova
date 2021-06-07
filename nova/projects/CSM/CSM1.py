import datetime
import os
from glob import glob
import string

# import matplotlib.dates as mdates
import matplotlib.gridspec
import numpy as np
import pandas
from scipy.interpolate import interp1d
import scipy.signal

from nova.definitions import root_dir
from nova.utilities.IO import pythonIO
from nova.utilities.pyplot import plt
from nova.utilities.time import clock
from nova.utilities.addtext import linelabel

'''
import pandas as pd
from nep.IO.project_path import project_path
from amigo.IO import pythonIO
from os import path
from amigo.pyplot import plt
import numpy as np
from glob import glob
from scipy.interpolate import interp1d
from scipy.signal import welch
from amigo.addtext import linelabel
from amigo.time import clock
# import matplotlib.dates as mdates
import datetime
from matplotlib.gridspec import GridSpec
from amigo.addtext import linelabel
from pandas.api.types import is_list_like
'''


class cold_test(pythonIO):

    _ID = {'TCS': 'TCS', 'DS': 'displace', 'SG': 'strain', 'TC': 'TCS',
           'EX': 'extend'}
    _units = {'%Open': 'valve',
              'kA': 'current',
              'K': 'temperature',
              'mOhm': 'resistance',
              'V': 'voltage',
              'mV': 'mvoltage',
              'W': 'power',
              'bara': 'apressure',
              'mbar': 'pressure',
              'g/s': 'flow',
              'ppm': 'strain',
              'mm': 'extend'}
    _labels = {'current': '$I$ kA',
               'temperature': '$T$ K',
               'strain': '$\epsilon$ ppm',
               'extend': '$\Delta W$ mm',
               'displace': '$\Delta W$ mm'}

    def __init__(self, project_dir='CSM1', read_txt=False):
        super().__init__()  # python read/write
        self.directory = os.path.join(root_dir, 'data/CSM', project_dir)
        self.groups, self.channels = [], {}
        self._units_r = {self._units[u]: u for u in self._units}
        self.to = np.datetime64('2020-01-07T08:33:00.000000000')
        self.read_txt = read_txt

    def split_channels(self, data):
        index = {**{self._ID[l]: [] for l in self._ID},
                 **{self._units[u]: [] for u in self._units},
                 **{'other': []}}
        for channel in data.columns:
            if np.array([l in channel[0] for l in self._ID]).any():
                for label in self._ID:
                    if label in channel[0]:
                        index[self._ID[label]].append(channel)
                        break
            elif channel[1] in self._units:
                index[self._units[channel[1]]].append(channel)
            else:
                index['other'].append(channel)
        for channel in list(index.keys()):  # remove empty channels
            if len(index[channel]) == 0:
                index.pop(channel)
        return index

    def load_coldtest(self, group, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        filepath = os.path.join(self.directory, group)
        rawfilepath = os.path.join(self.directory, 'rawdata')
        if read_txt or not os.path.isfile(filepath + '.pk'):
            if group in ['strain', 'extend']:
                data = self.read_coldtest(group, 'strain')
            else:
                if group != 'rawdata' and os.path.isfile(rawfilepath + '.pk'):
                    if not hasattr(self, 'rawdata'):
                        self.load_pickle(rawfilepath)  # load rawdata
                    data = self.read_rawdata(group)
                else:
                    #cooldown = self.read_coldtest(group, 'cooldown')
                    data = self.read_coldtest(group, 'test')
                    #data = pandas.concat((cooldown, test), sort=True)
                    data.sort_index(inplace=True)
            if group != 'rawdata':
                self.condition_signal(data)
            setattr(self, group, data)
            self.save_pickle(filepath, [group])
        else:
            self.load_pickle(filepath)
        self.groups.append(group)
        self.channels.update(**{channel: group for channel in
                                getattr(self, group).columns.droplevel(1)})

    def read_folder(self, *args):
        files = glob(os.path.join(self.directory, *args))
        files = np.sort([f.split('\\')[-1] for f in files])
        return files

    def read_rawdata(self, group):
        channel_index = self.split_channels(self.rawdata)
        if group == 'resistance':
            self.rawdata[('RCoil', 'mOhm')].fillna(
                    1e-3*self.rawdata[('RCoil', 'ohm')], inplace=True)
        data = self.rawdata.loc[:, channel_index[group]]
        if group == 'resistance':
            data.loc[:] *= 1e3  # all to mOhm
        return data

    def read_coldtest(self, group, folder, to=None):
        if to is None:
            to = self.to
        files = self.read_folder(folder, '*.csv')
        nfiles = len(files)
        data = [[] for __ in range(nfiles)]
        tick = clock(nITER=nfiles, header=f'reading {nfiles} {folder} files')
        if group in ['strain', 'extend']:
            folder = 'strain'
        for i, file in enumerate(files):
            if folder == 'strain':
                file_reader = self.read_strain_file
            else:
                file_reader = self.read_coldtest_file
            data[i] = file_reader(folder, file)
            channel_index = self.split_channels(data[i])
            if group != 'rawdata':
                data[i] = data[i].loc[:, channel_index[group]]
            tick.tock()
        data = pandas.concat(data, sort=True)
        data.sort_index(inplace=True)
        return data

    def read_coldtest_file(self, folder, filename):
        data = pandas.read_csv(os.path.join(self.directory, folder, filename),
                               header=7)
        data.loc[:, 'timestamp'] = pandas.to_datetime(data.loc[:, 'timestamp'])
        self.format_columns(data)
        if '20200107' in filename:  # trim start of file
            data = data.iloc[28:, :]
        return data

    def read_strain_file(self, folder, filename):
        filepath = os.path.join(self.directory, folder, filename)
        data = pandas.read_csv(filepath, skiprows=[0, 1, 2, 4, 5, 6])
        data = data.iloc[:, 3:]  # drop first three rows
        elapsed = pandas.read_csv(filepath,
                skiprows=6, usecols=[2])
        columns = {}
        for c in data.columns:
            if c[:2] == 'EX':
                columns[c] = f'{c.split(" on ")[-1]}_(mm)'
            elif c[:2] == 'SG':
                columns[c] = f'{c}_(ppm)'
            else:
                columns[c] = c
        data.rename(columns=columns, inplace=True)
        data['timestamp'] = pandas.to_datetime(elapsed.iloc[:, 0])
        if '021320 - 021420' in filename:  # correct time shift
            data.loc[:, 'timestamp'] -= datetime.timedelta(seconds=21*60+21)
        self.format_columns(data)
        return data

    def format_columns(self, data):
        data.rename(columns={'timestamp': 'timestamp_(time)'},
                    inplace=True)
        columns = data.columns.values
        _columns = [c.replace('mean', '').replace('_', '') for c in columns]
        _columns = [c.replace('value', '') for c in _columns]
        labels = [c.split('(')[0] for c in _columns]
        units = [c.split('(')[1][:-1] for c in _columns]
        data.rename(columns={c: l for c, l in zip(columns, labels)},
                             inplace=True)
        data.columns = pandas.MultiIndex.from_arrays(
                [data.columns, units],
                names=('ID', 'unit'))
        data.set_index(('timestamp', 'time'), inplace=True)
        data.index.name = ('timestamp')

    def t(self, index, to=None):
        if to is None:
            to = self.to
        elif isinstance(to, int):
            to = index[to]
        return (index - to).astype('timedelta64[ms]')*1e-3

    def condition_signal(self, data):
        t = self.t(data.index)
        Tcol = [c for c in data.columns if c[0][:2] == 'TT']
        DScol = [c for c in data.columns if c[0][:2] in ['DS', 'EX']]
        strain_col = [c for c in data.columns if c[0][:2] == 'SG']
        current_col = [c for c in data.columns
                       if c[1] == 'A' or c[1] == 'kA']
        for col in data.columns:
            if col in Tcol:
                signal_minimum = data.loc[:, col] < 1e-3
                signal_maximum = data.loc[:, col] > 300
                gradient_max = 0.1
            elif col in DScol:
                signal_minimum = data.loc[:, col] < -30
                signal_maximum = data.loc[:, col] > -1e-4
                gradient_max = 0.1
            elif col in current_col:
                if col[0] == 'IBus':
                    data.loc[data.loc[:, col] > 45, col] *= 1e-3
                if col[1] == 'kA':
                    signal_minimum = data.loc[:, col] < -45
                    signal_maximum = data.loc[:, col] > 45
                    gradient_max = 0.1
                else:
                    signal_minimum = False
                    signal_maximum = False
                    gradient_max = False#0.01
            elif col in strain_col:
                data.loc[:, col] *= 2
                signal_minimum = data.loc[:, col] < -5000
                signal_maximum = data.loc[:, col] > 5000
                gradient_max = False
            elif col[0] == 'RCoil':
                signal_minimum = data.loc[:, col] < 0
                signal_maximum = False
                gradient_max = 0.1
            else:
                signal_minimum = False
                signal_maximum = False
                gradient_max = False
            if gradient_max:
                value = data.loc[:, col].copy()
                isna = value.isna()
                value.loc[isna] = 0
                _t, _i = np.unique(t, return_index=True)
                _v = value[_i]
                gradient = interp1d(_t, np.gradient(_v, _t))(t)
                gradient_maximum = (abs(gradient) > gradient_max) | isna
            else:
                gradient_maximum = np.zeros(data.shape[0], dtype=bool)
            index = signal_minimum | signal_maximum | gradient_maximum
            # interpolate
            if sum(~index) > 2:  # surficent data for interpolation
                interp = interp1d(t[~index], data.loc[~index, col],
                                   bounds_error=False,
                                   fill_value=None)
                data.loc[index, col] = interp(t[index])
                # calculate high frequency content
                dt = np.median(np.diff(t))
                __, psd = scipy.signal.welch(data.loc[:, col], 1/dt)
                if psd[-1] > 1 and col in Tcol:
                    data.drop(columns=col, inplace=True)
            elif col in Tcol:
                data.drop(columns=col, inplace=True)
        dt = np.append(t[1:] - t[:-1], 0)
        gap = np.median(dt) + 1.5 * np.std(dt)
        data.loc[dt > gap, :] = None

    def plot_temperature(self):
        ax = plt.subplots(1, 1)[1]
        text = linelabel(Ndiv=50, value='')
        for col in self.temperature:
            T = self.temperature.loc[:, col]
            if T[-1] > 250:
                color = 'C0'
            elif T[-1] > 50:
                color = 'C1'
            elif T[-1] > 10:
                color = 'C2'
            elif T[-1] > 6:
                color = 'C4'
            else:
                color = 'C3'
            ax.plot(self.temperature.loc[:, col], T, color=color,
                    label=col[0][2:])
            text.add(col[0])
        plt.despine()
        ax.set_ylim([0, 300])
        ax.set_xlabel('$t$ hr')
        ax.set_ylabel('$T$ K')
        plt.legend(ncol=7, loc='upper center', bbox_to_anchor=(0.5, 1.4))
        plt.despine()
        # text.plot(fs=6)

    def offset(self, dataframe, offset_dt):
        index = [[], []]
        index[0] = dataframe.index[0]
        index[1] = index[0] + datetime.timedelta(seconds=offset_dt)
        offset = dataframe.loc[index[0]:index[1]].mean()
        return offset

    def get_dataframe(self, label, index):
        index = self.get_index(index)
        if isinstance(label, pandas.DataFrame):
            dataframe = label
            group = self.channels[dataframe.columns.droplevel(1)[0]]
        else:
            if pandas.api.types.is_list_like(label):
                group = self._ID[label[0][:2]]
                self.load_coldtest(group)
                groups = [self.channels[l] for l in label]
                if not np.array([g == groups[0] for g in groups]).all():
                    raise IndexError('list labels must belong to same group')
                group = groups[0]
                channels = [(l, self._units_r.get(group, 'mm')) for l in label]
            elif label in self.groups:
                group = label
                channels = slice(None)
            elif label in self.channels:
                group = self.channels[label]
                channels = [(label, self._units_r[group])]
            else:
                group = label
                channels = slice(None)
            if not hasattr(self, group):
                self.load_coldtest(group)  # load dataframe
            dataframe = getattr(self, group).copy() # get dataframe
            if group in ['current']:
                dataframe.drop(columns=['Ibusfast', 'IBus'], inplace=True)
                channels = dataframe.columns[::-1]
            dataframe = dataframe.loc[:, channels]
        dataframe = dataframe[index]
        return dataframe, group

    def plot(self, label, index=None, ax=None,
             offset=None, offset_dt=5*60, legend=True, labels=True,
             xlabel=True, ylabel=True, ncol=2, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = plt.gcf()
        dataframe, group = self.get_dataframe(label, index)
        if group in ['current', 'temperature']:
            offset_dt = 0
        #self.condition_signal(dataframe)
        if offset is not None:
            dataframe -= offset
        elif offset_dt > 0:  # calculate offset
            offset = self.offset(dataframe, offset_dt)
            dataframe -= offset
        else:
            offset = 0
        ax.plot(dataframe, **kwargs)
        if legend and labels:
            shift = np.floor(dataframe.shape[1] / ncol) * 0.2
            ax.legend([c for c in
                        dataframe.columns.droplevel(1)],
                        ncol=ncol, loc='upper center',
                        bbox_to_anchor=(0.5, 1+shift))
        plt.despine()
        fig.autofmt_xdate()
        if labels:
            if xlabel:
                ax.set_xlabel('timestamp')
            if group in self._labels and ylabel:
                ax.set_ylabel(self._labels[group])
        return offset

    def plot_col(self, label, index=['cooldown', 'test'], offset_dt=5*60):
        ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'wspace': 0.1})[1]
        offset = self.plot(label, index=index[0], ax=ax[0],
                           offset_dt=offset_dt)
        self.plot(label, index=index[1], ax=ax[1], labels=False,
                  offset=offset)

    def plot_row(self, label, index='test', ncol=2, color=None,
                 offset_dt=5*60):
        plt.set_aspect(0.7)
        fig = plt.figure()
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[6, 1])
        ax = []
        ax.append(fig.add_subplot(gs[0]))
        ax.append(fig.add_subplot(gs[1], sharex=ax[0]))
        plt.setp(ax[0].get_xticklabels(), visible=False)
        legend = ncol > 0
        self.plot(label, index=index, ax=ax[0], xlabel=False, ncol=ncol,
                  color=color, legend=legend, offset_dt=offset_dt)
        self.plot('current', index=index, ax=ax[1], legend=False,
                  color='gray')
        Imax_iloc = np.nanargmax(self.current['PSIOut'])
        Imax = self.current.iloc[Imax_iloc]['PSIOut']
        #ax[1].text(, Imax, f'{Imax}')

    def extract_shrinkage(self, sensors=['displace', 'extend'], plot=True,
                          Imax=48.5):
        for group in sensors + ['current']:
            if group not in self.groups:
                self.load_coldtest(group)
        shrink = pandas.DataFrame(index=self.current.index)
        if 'extend' in sensors:
            self.condition_signal(self.extend)
            self.extend.loc[slice('2020-02-13 12:00',
                                  '2020-02-14'), 'EX-170-ID'] = None
            t = self.t(shrink.index)  # timebase
            t_ex = self.t(self.extend.index)  # strain timebase
            for col in self.extend:  # load extension results
                shrink[col] = interp1d(
                        t_ex, self.extend[col], bounds_error=False,
                        fill_value=None)(t)
        if 'displace' in sensors:
            for col in self.displace:  # load displacment results
                shrink[col] = self.displace[col]
        # clip
        for col in shrink:
            index = shrink[col] < -9
            shrink.loc[index, col] = None
        shrink.columns = pandas.MultiIndex.from_tuples(shrink.columns)
        setattr(self, 'shrink', shrink)
        self.groups.append('shrink')
        self.fit(shrink, index='test', plot=plot, Imax=Imax)

    def fit(self, label, index='test', Imin=7.5, Imax=40, plot=True,
            ncol=2, color=None):
        index = self.get_index(index)
        # extract current
        if not hasattr(self, 'current'):
            self.load_coldtest('current')
        I = self.current.loc[index, ('PSIOut', 'kA')]
        # extract dataframe
        dataframe, group = self.get_dataframe(label, index)
        # interpolate
        if I.shape[0] == dataframe.shape[0]:
            interpolate = not np.equal(I.index, dataframe.index).all()
        if I.shape[0] != dataframe.shape[0] or interpolate:
            t = self.t(I.index)  # current time index
            t_data = self.t(dataframe.index)  # data time index
            _dataframe = pandas.DataFrame(index=I.index)
            for col in dataframe:  # load extension results
                _dataframe[col] = interp1d(
                    t_data, dataframe.loc[:, col], bounds_error=False,
                    fill_value=None)(t)
            dataframe = _dataframe
            dataframe.columns = pandas.MultiIndex.from_tuples(dataframe.columns)
        # zero offset
        dataframe -= self.offset(dataframe, 5)
        # trim low current
        current_index = I > Imin
        I = I[current_index]
        dataframe = dataframe[current_index]
        coef = np.zeros(dataframe.shape[1])
        for i, col in enumerate(dataframe):
            if not np.isnan(dataframe.loc[:, col]).all():
                index = dataframe.loc[:, col].notna()
                coef[i] = np.linalg.lstsq(
                        I[index].to_numpy().reshape(-1, 1)**2,
                        dataframe.loc[index, col], rcond=None)[0][0]
        if plot:
            ax = plt.subplots(1, 1)[1]
            if Imax is None:
                Imax = I.max()
                xtick = None
            else:
                xtick = Imax
            _I = np.linspace(0, Imax, 100)  # plot fit
            group = self.channels[dataframe.columns.droplevel(1)[0]]
            if group == 'strain':
                value = '1.0f'
            else:
                value = '1.2f'
            text = linelabel(value=value,
                             postfix=self._units_r.get(group, 'mm'),
                             Ndiv=20)
            for i, col in enumerate(dataframe):
                if color is None:
                    c = f'C{i}'
                else:
                    try:
                        c = color[i]
                    except IndexError:
                        c = color
                if not np.isnan(dataframe.loc[:, col]).all():
                    ax.plot(I, dataframe[col], '.',
                            color=c, alpha=0.75, ms=4, zorder=-20)
                    ax.plot(_I, coef[i]*_I**2, '--',
                            color=c, alpha=1, label=col[0])

                    #plt.plot(I, dataframe[col], '.', label=col[0],
                    #         color=c, alpha=1)
                    #plt.plot(_I, coef[i]*_I**2, color=c, alpha=1)
                    text.add('')
            if xtick is not None:
                xticks = ax.get_xticks()
                dx = xticks[-1] - xticks[0]
                xticks = [x for x in xticks if abs(x - xtick) / dx > 0.1]
                xticks = np.sort(np.append(xticks, xtick))
                ax.set_xticks(xticks)
            plt.despine()
            plt.xlabel('$I$ kA')
            plt.ylabel(self._labels[group])
            shift = np.floor(dataframe.shape[1] / ncol) * 0.12
            plt.legend(ncol=ncol, loc='upper center',
                       bbox_to_anchor=(0.5, 1+shift))
            text.plot()

    def get_index(self, index):
        if index is None:
            index = slice(None)
        elif index == 'cooldown':
            index = slice('2020-01-07', '2020-01-27')  # cooldown
        elif index == 'test':
            index = slice('2020-02-11 9:00', '2020-02-14 18:00')
        elif index == 'strain':
            index = slice('2020-02-11 9:00', '2020-02-14 12:00')
        elif index == 'low_strain':
            index = slice('2020-02-11 9:00', '2020-02-11 18:00')
        elif index == 'low_strain_trim':
            index = slice('2020-02-11 9:00', '2020-02-11 12:40')
        elif index == 'medium_strain':
            index = slice('2020-02-12 9:00', '2020-02-12 18:00')
        elif index == 'medium_strain_trim':
            index = slice('2020-02-12 11', '2020-02-12 13:50')
        elif index == 'high_strain_trim':
            index = slice('2020-02-12', '2020-02-15')
        elif index == 'fit':
            index = slice('2020-02-11 9:00', '2020-02-12 18:00')
        elif index == 'drop':
            index = slice('2020-02-12 16:00', '2020-02-12 17:00')
        elif index == 'drop_trim':
            index = slice('2020-02-12 16:10', '2020-02-12 16:27:30')
        elif index == 'comp':
            index = slice('2020-02-12 11', '2020-02-12 13:50')
        return index

if __name__ == '__main__':
    plt.set_context('talk')


    ct = cold_test(read_txt=False)
    #ct.load_coldtest('strain', read_txt=True)

    #ct.plot('temperature')

    #ct.plot('current')

    #ct.plot('strain', index='cooldown', ncol=3)
    #ct.plot_col('strain')

    #ct.plot_row('voltage', index='high_strain_trim')
    #ct.plot_row('strain', index='strain', ncol=3)
    #ct.fit('strain', index='high_strain_trim')

    # ct.plot_row('temperature', index='high_strain_trim', ncol=0)

    plt.set_aspect(0.5)
    #ct.plot_row('strain', index='low_strain')
    #ct.plot_row(['SG-340-ID', 'SG-220-OD'], index='low_strain')
    #ct.plot_row(['SG-340-ID', 'SG-220-OD'], index='low_strain_trim')
    #ct.fit(['SG-340-ID', 'SG-220-OD'], index='low_strain_trim')

    #ct.plot_row('strain', index='medium_strain')
    #ct.plot_row(['SG-340-ID', 'SG-220-OD'], index='medium_strain')

    ct.plot_row(['SG-340-ID', 'SG-220-OD'], 'comp')
    ct.fit(['SG-340-ID', 'SG-220-OD'], 'comp')

    #ct.plot_row(['SG-340-ID', 'SG-220-OD'], index=None)
    #ct.fit(['SG-340-ID', 'SG-220-OD'], index=None)

    #ct.plot('extend', offset_dt=0)
    #ct.plot_col('extend', offset_dt=0)
    #ct.plot_col('displace', offset_dt=0)

    ct.plot_row('extend', 'comp', ncol=2)
    ct.plot_row('displace', 'comp', ncol=4)


    ct.fit(['EX-170-ID', 'EX-270-OD', 'EX-4310-OD'], 'comp', ncol=3,
           color=['C0', 'C1', 'C3'])
    ct.fit(['DS001', 'DS002', 'DS004'], 'comp', ncol=3,
           color=['C0', 'C1', 'C3'])



    #ct.plot_row(['EX-270-OD'], index='drop', color='C1')
    #ct.plot_row(['DS002'], index='drop', color='C1')

    #ct.plot_row(['EX-270-OD'], index='drop_trim', color='C1')
    #ct.plot_row(['DS002'], index='drop_trim', color='C1')

    #plt.set_aspect(0.9)
    #ct.fit(['EX-270-OD'], index='drop_trim', Imin=0, color='C1')
    #ct.fit(['DS002'], index='drop_trim', Imin=0, color='C1')

    #plt.set_aspect(0.85)
    #ct.extract_shrinkage(sensors=['displace'])
    #ct.extract_shrinkage(sensors=['extend'])

    #ct.plot_row('displace', index='test')
    #ct.plot_row('extend', index='strain')

    #ct.plot('displace', offset_dt=0)
    #ct.plot_col('displace', offset_dt=0)
    #ct.plot_row('displace', index='test')


    #ct.extract_shrinkage(sensors=['displace', 'extend'])
    #ct.plot('shrink', offset_dt=0)
    #ct.plot_col('shrink', offset_dt=0)
    #ct.plot_row('shrink', index='test', ncol=4)

    #ct.plot_row('voltage')


    #myFmt = mdates.DateFormatter('%d %B')
    #ax.xaxis.set_major_formatter(myFmt)
