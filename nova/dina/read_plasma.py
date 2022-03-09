
import os

import pandas
import numpy as np
from scipy.interpolate import interp1d

from nova.electromagnetic.IO.read_waveform import read_dina
from nova.utilities.pyplot import plt


class read_plasma(read_dina):

    def __init__(self, database_folder='disruptions', read_txt=False):
        super().__init__(database_folder, read_txt)  # read utilities

    def load_file(self, folder, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        filepath = self.locate_file('plasma', folder=folder)
        filepath = '.'.join(filepath.split('.')[:-1])
        self.name = filepath.split(os.path.sep)[-2]
        if read_txt or not os.path.isfile(filepath + '.pk'):
            self.read_file(filepath)  # read txt file
            self.save_pickle(filepath,
                             ['data', 'columns', 't', 'Ip',
                              'x', 'z', 'nt', 'dt', 'zdir'])
        else:
            self.load_pickle(filepath)

    def read_file(self, filepath, dropnan=True):
        filename = filepath + '.dat'
        self.data = pandas.read_csv(filename, delim_whitespace=True,
                                    skiprows=40, na_values='NAN')
        if dropnan:
            self.data = self.data.dropna()  # remove NaN values
        self.columns = {}
        for c in list(self.data):
            self.columns[c] = c.split('[')[0]
        self.data = self.data.rename(index=str, columns=self.columns)
        self.load_data()

    def load_data(self):
        data = {}  # convert units
        data['t'] = 1e-3*self.data['t']  # ms - s
        data['Ip'] = -1e3*self.data['Ip']  # -kA to A
        data['x'] = 1e-2*self.data['Rcur']  # cm - m
        data['z'] = 1e-2*self.data['Zcur']  # cm - m
        dt_min = np.nanmin(np.diff(data['t']))
        t_max = np.nanmax(np.array(data['t']))
        self.nt = int(t_max / dt_min)
        self.t = np.linspace(0, t_max, self.nt)  # equal spaced time stencil
        self.dt = t_max/(self.nt-1)
        for var in data:
            if var != 't':  # interpolate
                setattr(self, var, interp1d(data['t'], data[var])(self.t))
        self.zdir = np.sign(self.z[np.argmax(abs(self.z))])


if __name__ == '__main__':

    plasma = read_plasma('disruptions', read_txt=True)
    plasma.load_file(-1)

    plt.plot(plasma.t, plasma.Ip)
