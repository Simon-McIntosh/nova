import nep_data.scenario_database
from amigo.IO import class_dir
from os.path import join, isfile, isdir
from os import listdir
import numpy as np
from scipy.optimize import minimize
from amigo.pyplot import plt
import pandas as pd


class timeconstant:

    def __init__(self, td, Id, trim_fraction=0.2):
        self.load(td, Id)
        self.trim_fraction = trim_fraction

    def load(self, td, Id):  # load profile
        self.td = np.copy(td)
        self.Id = np.copy(Id)

    def trim(self, **kwargs):
        self.trim_fraction = kwargs.get('trim_fraction', self.trim_fraction)
        if self.trim_fraction > 0 and self.trim_fraction < 1\
                and self.Id[0] != 0:
            dI = self.Id[-1] - self.Id[0]
            i1 = next((i for i, Id in enumerate(self.Id)
                       if abs(Id) < abs(self.Id[0] +
                                        (1-self.trim_fraction)*dI)))
            td, Id = self.td[:i1], self.Id[:i1]
        else:
            td, Id = self.td, self.Id
        return td, Id

    def fit_tau(self, x, *args):
        to, Io, tau = x
        t_exp, I_exp = args
        I_fit = Io*np.exp(-(t_exp-to)/tau)
        err = np.sum((I_exp-I_fit)**2)  # sum of squares
        return err

    def get_tau(self, plot=False, **kwargs):
        td, Id = self.trim(**kwargs)
        to = kwargs.get('to', 10e-3)
        Io = kwargs.get('Io', -60e3)
        tau = kwargs.get('tau', 30e-3)
        x = minimize(self.fit_tau, [to, Io, tau], args=(td, Id)).x
        err = self.fit_tau(x, td, Id)
        to, Io, tau = x
        Iexp = Io*np.exp(-(td-to)/tau)
        if plot:
            ax = kwargs.get('ax', plt.gca())
            ax.plot(1e3*td, 1e-3*Iexp, '-', label='exp')
        return to, Io, tau, err, Iexp

    def get_td(self, plot=False, **kwargs):  # linear discharge time
        td, Id = self.trim(**kwargs)
        A = np.ones((len(td), 2))
        A[:, 1] = td
        a, b = np.linalg.lstsq(A, Id, rcond=None)[0]
        tlin = abs(self.Id[0]/b)  # discharge time
        Ilin = a + b*td  # linear fit
        err = np.sum((Id-Ilin)**2)
        if plot:
            plt.plot(1e3*td, 1e-3*Ilin, '-', label='lin')
        return a, b, tlin, err, Ilin

    def fit_ntau(self, x, *args):
        texp, Iexp = args
        Ifit = self.I_nfit(texp, x)
        err = np.sum((Iexp-Ifit)**2)  # sum of squares
        return err

    def I_nfit(self, t, x):
        n = int(len(x)/2)
        Io, tau = x[:n], x[n:]
        Ifit = np.zeros(len(t))
        for Io, tau in zip(x[:n], x[n:]):
            Ifit += Io*np.exp(-t/tau)
        return Ifit

    def nfit(self, n, plot=False, **kwargs):
        tau_o = kwargs.get('tau_o', 50e-3)  # inital timeconstant
        td, Id = self.trim(**kwargs)
        to = td[0]
        td -= to  # time shift
        xo = np.append(Id[0]/n*np.ones(n), tau_o*np.ones(n))
        # xo *= np.random.random(2*n)
        bounds = [(None, None) for __ in range(n)]
        bounds.extend([(1e-6, None) for __ in range(n)])
        x = minimize(self.fit_ntau, xo, args=(td, Id),
                     method='L-BFGS-B', bounds=bounds).x
        Ifit = self.I_nfit(td, x)
        Io, tau = x[:n], x[n:]
        if plot:
            if 'ax' in kwargs:
                ax = kwargs['ax']
            else:
                ax = plt.subplots(1, 1)[1]
            ax.plot(1e3*td+to, 1e-3*Id, '-', label='data')
            ax.plot(1e3*td+to, 1e-3*Ifit, '--', label='fit')
            ax.set_xlabel('$t$ ms')
            ax.set_ylabel('$I$ kA')
            plt.despine()
            plt.legend()
        return Io, tau, td+to, Ifit

    def ntxt(If, tau):
        txt = r'$\alpha$=['
        for i, I in enumerate(If):
            if i > 0:
                txt += ','
            txt += '{:1.2f}'.format(I)
        txt += ']'
        txt += r' $\tau$=['
        for i, t in enumerate(tau):
            if i > 0:
                txt += ','
            txt += '{:1.1f}'.format(1e3*t)
        txt += ']ms'
        return txt

    def fit(self, plot=False, **kwargs):
        tfit, Idata = self.trim(**kwargs)
        tau, tau_err, Iexp = self.get_tau(plot=False, **kwargs)[-3:]
        tlin, tlin_err, Ilin = self.get_td(plot=False, **kwargs)[-3:]
        if tau_err < tlin_err:
            self.discharge_type = 'exponential'
            self.discharge_time = tau
            Ifit = Iexp
        else:
            self.discharge_type = 'linear'
            self.discharge_time = tlin
            Ifit = Ilin
        if plot:
            if 'ax' in kwargs:
                ax = kwargs['ax']
            else:
                ax = plt.subplots(1, 1)[1]

            ax.plot(1e3*tfit, 1e-3*Idata, '-', label='data')
            ax.plot(1e3*tfit, 1e-3*Ifit, '--',
                    label='{} fit'.format(self.discharge_type))
            ax.set_xlabel('$t$ ms')
            ax.set_ylabel('$I$ kA')
            plt.despine()
            # plt.legend()
            txt = '{} discharge'.format(self.discharge_type)
            txt += ', t={:1.1f}ms'.format(self.discharge_time)
            # plt.title(txt)
        return self.discharge_time, self.discharge_type, tfit, Ifit


class dina:

    def __init__(self, database_folder):
        self.set_directory(database_folder)

    def set_directory(self, database_folder):
        self.database_folder = database_folder
        self.get_directory()
        self.get_folders()

    def get_directory(self):
        self.directory = class_dir(nep_data.scenario_database)
        if self.database_folder is not None:
            self.directory = join(self.directory, self.database_folder)

    def get_folders(self):
        folders = [f for f in listdir(self.directory)]
        self.folders = sorted(folders)
        self.nfolder = len(self.folders)
        files = [f for f in listdir(self.directory) if isfile(f)]
        self.files = sorted(files)
        self.nfile = len(self.files)

    def select_folder(self, folder):  # folder entered as string, index or None
        if isinstance(folder, int):  # index (int)
            if folder > self.nfolder-1:
                txt = '\nfolder index {:d} greater than '.format(folder)
                txt += 'folder number {:d}'.format(self.nfolder)
                raise IndexError(txt)
            folder = self.folders[folder]
        elif isinstance(folder, str):
            if folder not in self.folders:
                txt = '\nfolder {} '.format(folder)
                txt += 'not found in {}'.format(self.directroy)
                raise IndexError(txt)
        elif folder is None:
            folder = self.directory
        else:
            raise ValueError('folder required as int, str or None')
        return join(self.directory, folder)

    def locate_file(self, file_type, folder=None):
        if self.nfolder == 0:
            folder = None
        folder = self.select_folder(folder)
        ext = file_type.split('.')[-1].lower()
        if ext in ['xls', 'qda', 'txt']:  # *.*
            file_type = file_type.split('.')[0].lower()
            subfolders = listdir(folder)
            for subfolder in subfolders:
                subfolder = join(folder, subfolder)
                if isdir(subfolder):
                    files = [f for f in listdir(subfolder) if
                             isfile(join(subfolder, f))]
                    folder_ext = [file.split('.')[-1].lower()
                                  for file in files]
                    if ext in folder_ext:
                        folder = subfolder
                        break
                    else:
                        files = []
            if not files:
                raise IndexError('file {}.{} not found'.format(file_type,
                                                               ext))
        else:
            files = [f for f in listdir(folder) if isfile(join(folder, f))]
        files = [f for f in files if file_type.lower() in f.lower()]
        if len(files) == 0:
            txt = '\nfile key {} not found '.format(file_type)
            txt += 'in: \n{}'.format(files)
            raise IndexError(txt)
        try:
            file = [f for f in files if ext in f.lower()][0]
        except IndexError:
            raise IndexError('ext {} not found in {}'.format(ext, files))
        return join(folder, file)

    def read_csv(self, filename, dropnan=True, split='', dataframe=False):
        data = pd.read_csv(filename, delimiter='\t', na_values='NAN')

        columns = {}
        for c in list(data):
            if split:
                co = c.split(split)[0]
            if co in [key.split(split)[0] for key in columns]:
                co += '_extra'  # seperate duplicates
            columns[c] = co
        data = data.rename(index=str, columns=columns)
        data_keys = list(data.keys())
        for var in data_keys:
            if len(data[var]) == 0 or np.isnan(data[var]).all():
                data.pop(var)
        if dropnan:
            data = data.dropna(axis=0)  # remove NaN values
        if not dataframe:
            data = data.to_dict(orient='list')
        return data


if __name__ == '__main__':

    dina = dina('operations')
    filename = dina.locate_file('data2.txt', folder=0)
