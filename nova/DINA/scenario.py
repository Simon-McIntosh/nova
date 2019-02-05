from nep.DINA.read_dina import dina
from amigo.qdafile import QDAfile
from amigo.pyplot import plt
import numpy as np
from nep.coil_geom import PFgeom
from scipy.interpolate import interp1d
from amigo.stat import histopeaks
import matplotlib.lines as mlines
from collections import OrderedDict
from nep.DINA.read_eqdsk import read_eqdsk
import nova.cross_coil as cc
from os import sep
from amigo.IO import pythonIO
from os.path import isfile
from datetime import datetime
from amigo.geom import turning_points, lowpass
from sklearn.cluster import KMeans, DBSCAN
from rdp import rdp
from nova.force import force_field
from itertools import count
from nova.cross_coil import get_coil_psi
from amigo import geom
from nova.inverse import INV
from nova.coils import PF
from nova.streamfunction import SF
from amigo.time import clock
import copy
import pandas as pd
from amigo.addtext import linelabel


class scenario(pythonIO):

    def __init__(self, database_folder='operations', folder=None,
                 read_txt=False, VS=False, file_type='txt', dCoil=0.25,
                 setname='split_f'):
        self.date_switch = datetime.strptime('2016-02', '%Y-%m')
        self.read_txt = read_txt
        self.setname = setname
        self.dCoil = dCoil
        self.dina = dina(database_folder)
        if folder is not None:
            self.load_file(folder, file_type=file_type)
        pythonIO.__init__(self)  # python read/write

    def get_folders(self, exclude=[]):
        folders = np.ones(self.dina.nfolder,
                          dtype=[('name', 'U25'),
                                 ('year', int), ('mode', 'U25'),
                                 ('month', int), ('version', 'U25')])
        for i in range(self.dina.nfolder):
            folders[i]['name'] = self.dina.folders[i]
            folders[i]['mode'] = self.dina.folders[i].split('DINA')[0][:-1]
            timestamp = self.dina.folders[i].split('DINA')[-1]
            folders[i]['year'] = int(timestamp.split('-')[0])
            timestamp = ''.join(timestamp.split('-')[1:])
            folders[i]['month'] = int(timestamp[:2])
            folders[i]['version'] = timestamp[2:].replace('_', '')
        folders.sort(order=['year', 'month', 'version'])
        if exclude:
            index = [name not in exclude for name in folders['name']]
            folders = folders[index]
        return folders

    def locate_file_type(self, file_type, folder):
        file_types = ['txt', 'qda']
        file_types.remove(file_type)
        file_types = [file_type, *file_types]
        filepath = None
        for file_type in file_types:
            try:
                filepath = self.dina.locate_file(
                        'data2.{}'.format(file_type), folder=folder)
                break
            except IndexError:
                pass
        if filepath is None:
            raise FileNotFoundError()
        return filepath, file_type

    def load_file(self, folder, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        setname = kwargs.get('setname', self.setname)
        dCoil = kwargs.get('dCoil', self.dCoil)
        file_type = kwargs.get('file_type', 'txt')
        self.load_coilset(dCoil=dCoil, read_txt=False)
        self.load_functions(setname=setname)
        filepath, file_type = self.locate_file_type(file_type, folder)
        filepath = '.'.join(filepath.split('.')[:-1])
        self.folder = folder
        self.filepath = '/'.join(filepath.split('\\')[:-1]) + '/'
        attributes = ['name', 'date', 'data', 'post', 'tie_plate',
                      't', 'dt', 'fun', 'Icoil', 'Ipl']
        if read_txt or not isfile(filepath + '.pk'):
            self.read_file(folder, file_type=file_type, dCoil=dCoil)
            self.save_pickle(filepath, attributes)
        else:
            self.load_pickle(filepath)
        self.set_limits()
        self.loadCScurrents()

    def load_coilset(self, read_txt=False, **kwargs):
        dCoil = kwargs.get('dCoil', self.dCoil)
        filepath = '/'.join(self.dina.directory.split('\\')[:-1]) + '/'
        filepath = filepath + f'coilset_{dCoil}'
        attributes = ['coilset', 'dCoil', 'boundary']
        if read_txt or not isfile(filepath + '.pk'):
            self.compute_coilset(dCoil=dCoil, VS=False)  # coil geometory
            self.load_boundary()  # load flux map limits and fw profile
            self.save_pickle(filepath, attributes)
        else:
            self.load_pickle(filepath)

    def load_functions(self, setname='link'):
        self.setname = setname
        self.pf = PF()  # create bare pf coil instance
        self.pf(self.coilset[self.setname])
        if setname != self.setname:
            self.update_coilset()
        self.sf = SF()  # create bare sf instance
        self.ff = force_field(self.coilset[self.setname])
        if self.coilset[self.setname]['force']['update_passive']:
            self.ff.set_force_field(state='passive')
        self.ff.get_force()
        self.inv = INV(self.coilset[self.setname], boundary='sf')
        self.inv.sf = self.sf  # link sf instance
        self.inv.ff = self.ff  # link ff instance

    def read_file(self, folder, file_type='txt', dCoil=0.5, verbose=False):
        if verbose:
            print('reading {}'.format(folder))
        if file_type == 'txt':
            self.read_txt_file(folder)
        elif file_type == 'qda':
            self.read_qda_file(folder)
        self.date = datetime.strptime(
                self.name.split('DINA')[-1].split('_')[0][:7], '%Y-%m')
        if 'time' in self.data2:  # rename time field
            self.data2['t'] = self.data2['time']
            self.data2.pop('time')
        self.load_scenario_data()
        self.load_force_data()
        self.postprocess()  # calculate Faxial
        # self.operate()  # identify operating modes

    def get_current(self, t, names):
        Ic = pd.Series(index=names)
        for i, name in enumerate(names):
            key = 'I'+name.lower()
            if 'cs1' in key:
                key = key[:-1]
            if key == 'Iplasma':
                key = 'Ip'
                factor = 1e3  # MA to kA
            else:
                factor = 1
            Ic.iloc[i] = factor*self.fun[key](t)
        return Ic

    def get_time(self, dt=None):
        if dt is None:
            dt = self.dt  # data-set resolution
        n = int((self.t[-1] - self.t[0]) / dt)
        time = np.linspace(self.t[0], self.t[-1], n)
        return time, n

    def stored_energy(self, M, names, dt=20, plot=False):
        time, n = self.get_time(dt=dt)
        E = np.zeros(n)
        for i, t in enumerate(time):
            Ic = self.get_current(t, names)  # kA
            E[i] = 0.5*M.dot(Ic).dot(Ic)
            E[i] *= 1e-3  # stored energy GJ
        energy = pd.Series()
        index = np.argmax(E)
        energy['t [s]'] = time[index]
        energy['E [GJ]'] = E[index]
        Ic = self.get_current(time[index], names)
        for Io, name in zip(Ic, names):
            energy[f'Io [kA] {name}'] = Io
        if plot:
            ax = plt.subplots(1, 1)[1]
            ax.plot(time, E)
            ax.set_xlabel('$t$ s')
            ax.set_ylabel('$E$ GJ')
            index = np.argmax(E)
            ax.text(time[index], E[index],
                    f'{E[index]:1.1f}GJ, {time[index]:1.1f}s',
                    ha='left', va='bottom')
            plt.despine()
        return energy

    def scenario_fault(self, M, names, dt=None, plot=False):
        time, n = self.get_time(dt=dt)
        Ic = np.zeros((n, len(M)))
        dI = np.zeros((n, len(M)))
        Eo = np.zeros((n, len(M)))
        Efault = np.zeros((n, len(M)))
        for i, t in enumerate(time):
            Ic[i, :], dI[i, :], Eo[i, :], Efault[i, :] = \
                self.fault_current(t, M, names)
        Ic = pd.DataFrame(Ic, columns=names)
        dI = pd.DataFrame(dI, columns=names)
        Eo = pd.DataFrame(Eo, columns=names)
        Efault = pd.DataFrame(Efault, columns=names)
        fault = pd.DataFrame(columns=names)
        for name in names:
            i = Efault.loc[:, name].idxmax()
            fault.loc['t', name] = time[i]
            fault.loc['Io', name] = Ic.loc[i, name]
            fault.loc['dI', name] = dI.loc[i, name]
            fault.loc['Ifault', name] = Ic.loc[i, name] + dI.loc[i, name]
            fault.loc['Eo', name] = Eo.loc[i, name]
            fault.loc['Efault', name] = Efault.loc[i, name]
        if plot:
            ax = plt.subplots(2, 1)[1]
            for i, name in enumerate(names):
                ls = f'C{i}-' if 'PF' in name else f'C{i-6}--'
                ax[0].plot(time, Ic.loc[:, name], ls, label=name)
                ax[1].plot(time, Efault.loc[:, name], ls, label=name)
            ax[0].legend(bbox_to_anchor=(1.1, 1.4), ncol=6)
            plt.despine()
            plt.detick(ax)
            ax[1].set_xlabel('$t$ s')
            ax[0].set_ylabel('$I$ kA')
            ax[1].set_ylabel('$E_{fault}$ GJ')
        return fault

    def fault_current(self, t, Mt, names):
        Ic = self.get_current(t, names)  # kA
        L = np.diag(Mt)
        M = Mt.copy()
        np.fill_diagonal(M, 0)
        Eo = 1e-3 * 0.5 * L * Ic**2  # GJ
        dI = pd.Series(1/L * np.dot(M, Ic), index=Ic.index)
        Efault = 1e-3 * 0.5 * L * (Ic+dI)**2  # GJ
        return Ic, dI, Eo, Efault

    def fix_shape(self, plot=False, **kwargs):  # set colocation points
        t = kwargs.get('t', self.flattop['t'][-1])  # eof
        n = kwargs.get('n', self.boundary['n'])
        limit = kwargs.get('limit', self.boundary['limit'])
        self.update_scenario(t)
        eqdsk = self.update_psi(n=n, limit=limit, plot=plot)
        self.inv.colocate(eqdsk)

    def set_limits(self):  # default limits for ITER coil-set
        self.inv.initalise_limits()  # reset
        self.inv.set_limit(ICS=45)  # kA current limits
        self.inv.set_limit(IPF1=48, IPF2=55, IPF3=55, IPF4=55, IPF5=52,
                           IPF6=52)
        self.inv.set_limit(FCSsep=240, side='upper')  # force limits
        self.inv.set_limit(FCSsum=60, side='both')

        self.inv.set_limit(FPF1=-150, FPF2=-75, FPF3=-90, FPF4=-40,
                           FPF5=-10, FPF6=-190, side='lower')
        self.inv.set_limit(FPF1=110, FPF2=15, FPF3=40, FPF4=90,
                           FPF5=160, FPF6=170, side='upper')

    def solve(self, **kwargs):
        flux_factor = kwargs.get('flux_factor', 1)
        flux = (flux_factor - 1) * self.flattop['dpsi'] / (2*np.pi)
        kwargs.pop('flux_factor', None)
        plasma_factor = kwargs.get('plasma_factor', 1)
        kwargs.pop('plasma_factor', None)
        for key in kwargs:
            self.inv.set_limit(**dict([[key, kwargs[key]]]), side='equal')
        rms = self.inv.solve_slsqp(flux)
        if plasma_factor != 1:
            self.scale_plasma_current(plasma_factor)
        Ipl = self.coilset[self.setname]['plasma_parameters']['Ipl']
        self.ff.get_force()
        return rms, Ipl

    def scale_plasma_current(self, factor):
        Ipl = self.checkpoint['coilset']['plasma_parameters']['Ipl']
        self.coilset[self.setname]['plasma_parameters']['Ipl'] = Ipl*factor
        for filament in self.coilset[self.setname]['plasma']:
            self.coilset[self.setname]['plasma'][filament]['If'] = Ipl*factor
        self.inv.update_plasma()
        self.ff.set_force_field(state='passive')

    def set_tie_plate(self):
        self.tie_plate = {}
        self.tie_plate['preload'] = 201.33
        self.tie_plate['limit_load'] = -26  # minimum axial load
        self.tie_plate['alpha'] = 6.85e-3  # alpha Fx (Poisson)
        self.tie_plate['beta'] = np.array([-0.0165, -0.0489, -0.0812,
                                           -0.113, -0.145, -0.178])  # beta Fz
        self.tie_plate['gamma'] = -2.95e-2  # gamma Fc (crush)
        self.tie_plate['mg'] = 1.18  # coil weight MN

    def read_txt_file(self, folder, dropnan=True, force=False):
        filename = self.dina.locate_file('data2.txt', folder=folder)
        self.name = filename.split(sep)[-3]
        self.data2 = self.dina.read_csv(filename, dropnan=True, split=',',
                                        dataframe=True)
        filename = self.dina.locate_file('data3.txt', folder=folder)
        self.data3 = self.dina.read_csv(filename, dropnan=True, split=',',
                                        dataframe=True)

    def read_qda_file(self, folder):
        filename = self.dina.locate_file('data2.qda', folder=folder)
        self.name = filename.split(sep)[-3]
        self.qdafile = QDAfile(filename)
        self.data2 = {}
        columns = {}
        for i, (var, nrow) in enumerate(zip(self.qdafile.headers,
                                            self.qdafile.rows)):
            var = var.decode()
            if nrow > 0:
                columns[var] = var.split(',')[0]
                self.data2[columns[var]] = np.array(self.qdafile.data[i, :])
        filename = self.dina.locate_file('data3.qda', folder=folder)
        self.name = filename.split(sep)[-3]
        self.qdafile = QDAfile(filename)
        self.data3 = {}
        columns = {}
        for i, (var, nrow) in enumerate(zip(self.qdafile.headers,
                                            self.qdafile.rows)):
            var = var.decode()
            if nrow > 0:
                columns[var] = var.split(',')[0]
                self.data3[columns[var]] = np.array(self.qdafile.data[i, :])

    def compute_coilset(self, dCoil=0.25, VS=False, nlevel=5):
        self.dCoil = dCoil
        dCoil = [self.dCoil/np.sqrt(2)**i for i in range(nlevel)]
        self.coilset = {}
        for i, dC in enumerate(dCoil):
            f = '' if i == 0 else '_' + i*'f'
            self.coilset['split'+f] = self.load_coil(dC, VS, joinCS=False)
            self.coilset['link'+f] = self.load_coil(dC, VS, joinCS=True)

    def load_coil(self, dCoil, VS, joinCS=True):
        pf = PFgeom(VS=VS, dCoil=dCoil).pf
        if joinCS:
            pf.join_coils('CS1', ['CS1L', 'CS1U'])
        coilset = pf.coilset
        coilset['mesh'] = {'dCoil': dCoil}
        coilset['force'] = {'update_passive': True,
                            'update_active': False}  # defaults
        self.load_force(coilset)  # append interaction matrices
        return coilset

    def load_boundary(self):
        eqdsk = read_eqdsk(file='burn').eqdsk  # load referance sf instance
        self.update_boundary(eqdsk)

    def update_boundary(self, eqdsk):
        limit = [[eqdsk['x'][0], eqdsk['x'][-1]],
                 [eqdsk['z'][0], eqdsk['z'][-1]]]
        self.boundary = {'n': eqdsk['nx'] * eqdsk['nz'], 'limit': limit,
                         'xlim': eqdsk['xlim'], 'zlim': eqdsk['zlim']}

    def load_force(self, coilset):
        ff = force_field(coilset, multi_filament=True)
        coilset['force']['Fa'] = ff.Fa  # interaction matrices
        coilset['force']['Fa_filament'] = ff.Fa_filament
        coilset['force']['Fp'] = ff.Fp
        coilset['force']['Fp_filament'] = ff.Fp_filament

    def space_data(self):  # generate interpolators and space timeseries
        if self.date > self.date_switch:
            coordinate_switch = 1
        else:  # old file - correct coordinates
            coordinate_switch = -1
        to, unique_index = np.unique(self.data2['t'], return_index=True)
        dt = np.mean(np.diff(to))
        tmax = np.nanmax(to)
        tmin = np.nanmin(to)
        nt = int(tmax/dt)
        dt = tmax/(nt-1)
        self.t = np.linspace(tmin, tmax, nt)
        self.fun, self.data = {}, {}
        extract = [var for var in self.data2 if ('I' in var and len(var) <= 5)]
        extract += ['Rcur', 'Zcur', 'ap', 'kp', 'Rp', 'Zp', 'a',
                    'Ksep', 'BETAp', 'li(3)', 't', 'PSI(axis)']
        extract = [var for var in extract if var in self.data2]
        for var in extract:  # interpolate
            if ('I' in var and len(var) <= 5) or ('V' in var):
                self.data2[var] *= coordinate_switch
            self.fun[var] = interp1d(to, self.data2[var][unique_index],
                                     fill_value='extrapolate')
            self.data[var] = self.fun[var](self.t)

    def load_scenario_data(self):
        self.space_data()
        self.Icoil = {}
        for var in self.data:
            if var == 'Ip':  # plasma
                self.Ipl = 1e6*self.data[var]  # MA to A
            elif var[0] == 'I' and len(var) <= 5:
                # kAturn to Aturn
                self.Icoil[var[1:].upper()] = 1e3*self.data[var]
        self.t = self.data['t']
        self.dt = np.mean(np.diff(self.t))

    def load_force_data(self, ticktock=False):
        coils = list(self.pf.coilset['coil'].keys())
        CSname = self.pf.coilset['index']['CS']['name']
        self.post = {'DINA': {}, 'Nova': {}}
        # DINA
        self.post['DINA']['t'] = pd.Series(self.data3['time'])
        nC, nt = self.pf.coilset['nC'], len(self.post['DINA']['t'])
        Fx, Fz, B = np.zeros((nt, nC)), np.zeros((nt, nC)), np.zeros((nt, nC))
        for i, name in enumerate(self.pf.coilset['coil']):
            B[:, i] = self.data3[f'B_{name.lower()}']
            Fx[:, i] = self.data3[f'Fr_{name.lower()}']
            Fz[:, i] = self.data3[f'Fz_{name.lower()}']
        self.post['DINA']['B'] = pd.DataFrame(B, columns=coils)
        self.post['DINA']['Fx'] = pd.DataFrame(Fx, columns=coils)
        self.post['DINA']['Fz'] = pd.DataFrame(Fz, columns=coils)
        self.post['DINA']['Fsep'] = self.calculate_Fsep(
                self.post['DINA']['Fz'].loc[:, CSname])
        # Nova
        Fx, Fz, Fc = np.zeros((nt, nC)), np.zeros((nt, nC)), np.zeros((nt, nC))
        tick = clock(nt)
        for i, t in enumerate(self.post['DINA']['t']):
            self.update_scenario(t)  # recalculate coil forces (+ crush)
            Fx[i, :] = self.ff.Fcoil['F'][:, 0]
            Fz[i, :] = self.ff.Fcoil['F'][:, 1]
            Fc[i, :] = self.ff.Fcoil['F'][:, 3]
            if ticktock:
                tick.tock()
        self.post['Nova']['t'] = self.post['DINA']['t']
        self.post['Nova']['Fx'] = pd.DataFrame(Fx, columns=coils)
        self.post['Nova']['Fz'] = pd.DataFrame(Fz, columns=coils)
        self.post['Nova']['Fc'] = pd.DataFrame(Fc, columns=coils)
        self.post['Nova']['Fsep'] = self.calculate_Fsep(
                self.post['Nova']['Fz'].loc[:, CSname])
        # patch DINA Faxial with Nova Fc calculation
        self.post['DINA']['Fc'] = self.post['Nova']['Fc']

    def loadCScurrents(self):
        CSname = self.pf.coilset['index']['CS']['name']
        self.post['DINA']['Ic'] = pd.DataFrame(columns=CSname)
        for name in CSname:
            data_name = 'CS1' if 'CS1' in name else name
            Ic_interp = interp1d(self.t, self.data[f'I{data_name.lower()}'],
                                 fill_value='extrapolate')
            self.post['DINA']['Ic'][name] = Ic_interp(self.post['DINA']['t'])

    def postprocess(self):
        # postprocess Faxial calculation
        self.set_tie_plate()  # set tip-plate constants
        CSname = self.pf.coilset['index']['CS']['name']
        self.post['Nova']['Faxial'] = self.calculate_Faxial(
            self.post['Nova']['Fx'].loc[:, CSname],
            self.post['Nova']['Fz'].loc[:, CSname],
            self.post['Nova']['Fc'].loc[:, CSname])
        self.post['Nova']['Fsum'] = np.sum(
                self.post['Nova']['Fz'].loc[:, CSname], axis=1)
        self.post['DINA']['Faxial'] = self.calculate_Faxial(
            self.post['DINA']['Fx'].loc[:, CSname],
            self.post['DINA']['Fz'].loc[:, CSname],
            self.post['DINA']['Fc'].loc[:, CSname])
        self.post['DINA']['Fsum'] = np.sum(
                self.post['DINA']['Fz'].loc[:, CSname], axis=1)

    def filter_vector(self, x, dt, dt_window=1.0, polyorder=2):
        x_filter = {}
        for name in x:
            x_filter[name] = lowpass(x[name], dt, dt_window=dt_window,
                                     polyorder=polyorder)
        return x_filter

    def plot_plasma_current(self, flattop_mask=True, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        alpha = 0.5 if flattop_mask else 1
        t_ft = self.flattop['t']
        dpsi = self.flattop['dpsi']
        ft_slice = slice(*self.flattop['index'])
        ax.plot(self.t, 1e-6*self.Ipl, 'C0', alpha=alpha)
        ax.plot(self.t[ft_slice], 1e-6*self.Ipl[ft_slice], 'C0', alpha=alpha)
        txt = 'flattop:\n'
        txt += '$t$      {:.1f}-{:1.1f}s\n'.format(*t_ft)
        txt += '$\Delta  t$    {:.1f}s\n'.format(np.diff(t_ft)[0])
        txt += '$\Delta \Psi$  {:.1f}Wb'.format(dpsi)
        ax.text(np.mean(t_ft), 1e-6*self.Ipl[self.flattop['index'][0]]/2,
                txt, ma='left', ha='center', va='center',
                bbox=dict(facecolor='w', ec='gray', lw=1,
                          boxstyle='round', pad=0.5))
        ax.set_ylabel('$I_{pl}$ MA')

    def plot_turning_points(self, n=500, read_txt=False):
        self.load_force_history(n=n, read_txt=read_txt)
        ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))[1]
        self.plot_plasma_current(ax=ax[0])
        self.get_turning_points('Icoil', coils=['CS'],
                                dt_window=2.0, n=500, epsilon=50,
                                plot=True, ax=ax[1])
        self.get_turning_points('Fsep', t=self.Fcoil['t'],
                                dt_window=0.0, n=500, epsilon=10,
                                plot=True, ax=ax[2])
        ax[0].set_title(self.name)
        plt.detick(ax)

    def get_turning_points(self, x, dt_window=1.0, n=500, epsilon=50,
                           plot=False, ax=None, **kwargs):
        if isinstance(x, str):
            variable_name = x
            x = getattr(self, x)  # load vector
        else:
            variable_name = ''
        names = list(x.keys())
        if 'coils' in kwargs:
            coils = kwargs['coils']
            for coil in x:
                label = self.get_coil_label(coil)
                if label not in coils:
                    names.remove(coil)
        if 't' in kwargs:
            t = kwargs['t']
            dt = np.mean(np.diff(t))
        else:
            t, dt = self.t, self.dt
        if dt_window > 0:
            x = self.filter_vector(x, dt, dt_window=dt_window)
        turn_index = {}
        to = np.linspace(t[0], t[-1], n)  # down-sample
        for name in names:
            xo = interp1d(t, x[name])(to)
            M = np.append(to.reshape(-1, 1), xo.reshape(-1, 1), axis=1)
            Mrdp = rdp(M, epsilon=epsilon)
            to_index = np.hstack(turning_points(Mrdp[:, 1])).astype(int)
            turn_index[name] = np.zeros(len(to_index), dtype=int)
            for i, t_ in enumerate(Mrdp[to_index, 0]):
                turn_index[name][i] = np.argmin(abs(t - t_))
        clusters = self.cluster(t, x, turn_index)
        if plot:
            self.plot_clusters(clusters, variable_name=variable_name, ax=ax)

    def cluster(self, t, x, turn_index, plot=False, ax=None):
        clusters = {}
        clusters['index'] = np.array([], dtype=int)
        clusters['time'] = np.array([], dtype=int)
        clusters['name'] = np.array([])
        clusters['t'] = t
        clusters['vector'] = {}
        for name in turn_index:
            clusters['vector'][name] = x[name]
            clusters['index'] = np.append(clusters['index'], turn_index[name])
            clusters['time'] = \
                np.append(clusters['time'], clusters['t'][turn_index[name]])
            clusters['name'] = np.append(
                    clusters['name'],
                    [name for __ in range(len(turn_index[name]))])
        db = DBSCAN(eps=5, min_samples=1).fit(clusters['time'].reshape(-1, 1))
        ncl = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        km = KMeans(n_clusters=ncl)
        clusters['fit'] = km.fit(clusters['time'].reshape(-1, 1))
        return clusters

    def plot_clusters(self, clusters, variable_name='',
                      flattop_mask=True, ax=None):
        scale, ylabel = self.get_variable(variable_name)
        ft_t = self.t[self.flattop['index']]
        ft_slice = slice(np.argmin(abs(ft_t[0]-clusters['t'])),
                         np.argmin(abs(ft_t[1]-clusters['t'])))
        dtype = [('t', float), ('i', int), ('value', float)]
        max_vector = np.ones(len(clusters['vector']), dtype=dtype)
        for i, name in enumerate(clusters['vector']):
            imax = np.argmax(clusters['vector'][name])
            max_vector['i'][i] = i
            max_vector['t'][i] = clusters['t'][imax]
            max_vector['value'][i] = clusters['vector'][name][imax]
        max_vector = np.sort(max_vector, order='value')[::-1]
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        for i, name in enumerate(clusters['vector']):
            color = 'C{}'.format(i)
            if flattop_mask:
                ax.plot(clusters['t'], scale*clusters['vector'][name],
                        alpha=0.5, color=color)
                ax.plot(clusters['t'][ft_slice],
                        scale*clusters['vector'][name][ft_slice],
                        label=name, color=color)
            else:
                ax.plot(clusters['t'], scale*clusters['vector'][name],
                        label=name, color=color)
        for i in range(len(clusters['index'])):
            index = clusters['index'][i]
            name = clusters['name'][i]
            ax.plot(clusters['t'][index],
                    scale*clusters['vector'][name][index],
                    'X', color='gray', zorder=30)
        for t_center in clusters['fit'].cluster_centers_:
            index = np.argmin(abs(clusters['t']-t_center))
            instance_vector = [clusters['vector'][name][index]
                               for name in clusters['vector']]
            ylim = np.array([np.min(instance_vector),
                             np.max(instance_vector)])
            ax.plot(t_center*np.ones(2), scale*ylim, '--', color='gray')
        if variable_name == 'Fsep':
            for i in range(2):
                color = 'C{}'.format(max_vector['i'][i])
                plt.plot(max_vector['t'][i], max_vector['value'][i],
                         '.', color=color, ms=10, zorder=50)
                ha = 'left' if max_vector['t'][i] < np.mean(ft_t) else 'right'
                plt.text(max_vector['t'][i], max_vector['value'][0],
                         '{:1.1f}MN'.format(max_vector['value'][i]),
                         va='bottom', ha=ha, color=color)

        plt.despine()
        ax.set_xlabel('$t$ s')
        ax.set_ylabel(ylabel)
        ax.legend(loc=4, framealpha=1)

    def get_variable(self, variable_name):
        scale, variable, unit = 1, '', ''
        if variable_name == 'Fsep':
            scale = 1
            variable = '$F_{sep}$'
            unit = 'MN'
        elif variable_name == 'Icoil':
            scale = 1e-3
            variable = '$I_{coil}$'
            unit = 'kA'
        ylabel = '{} {}'.format(variable, unit)
        return scale, ylabel

    def operate(self, plot=False, dt_window=10, nstd=3):
        # identify operating modes
        trim = np.argmax(self.Ipl[::-1] < -1e-3)
        ind = len(self.Ipl)-trim
        # low-pass filter current
        Ipl_lp = lowpass(self.Ipl, self.dt, dt_window=dt_window)
        dIpldt_lp = np.gradient(Ipl_lp, self.t)  # slope
        self.hip = histopeaks(self.t[:ind], dIpldt_lp[:ind], nstd=nstd, nlim=9,
                              nbins=300)  # modes
        opp_index = self.hip.timeseries(Ip=self.Ipl[:ind], plot=plot)
        self.flattop = {}  # plasma current flattop
        self.flattop['index'] = opp_index[0]
        self.flattop['t'] = self.t[opp_index[0]]
        self.flattop['dt'] = np.diff(self.flattop['t'])[0]
        self.flattop['dpsi'] = \
            np.diff(self.data['PSI(axis)'][self.flattop['index']])[0]

    def get_coil_current(self, index, VS3=True):
        It = {}
        for coil in self.Icoil:
            if VS3 or 'VS' not in coil:
                if index <= 0 and isinstance(index, int):  # frame_index
                    It[coil] = self.Icoil[coil][-index]
                else:  # time value
                    It[coil] = 1e3*self.fun['I'+coil.lower()](index)
        if 'CS1' in It and 'CS1' not in self.pf.coilset['coil']:
            # split central pair
            for coil in ['CS1L', 'CS1U']:
                It[coil] = It['CS1']  # prior to turn multiplication
            It.pop('CS1')  # remove CS1 value
        elif 'CS1' in self.pf.coilset['coil'] and 'CS1' not in It:
            # combine central pair
            It['CS1'] = np.mean([It['CS1L'], It['CS1U']])
            It.pop('CS1L')  # remove CS1L value
            It.pop('CS1U')  # remove CS1U value
        for name in It:  # A to A.turn
            if name in self.pf.coilset['coil']:
                It[name] *= self.pf.coilset['coil'][name]['Nt']
        return It  # return dict of coil currents

    def update_scenario(self, t=None):
        # update from DINA interpolators
        if t is None:
            t = self.flattop['t'][-1]  # eof
        self.to = t  # referance time
        self.set_coil_current(t)
        self.set_plasma(self.extract_plasma(t))
        self.update_coilset()
        self.ff.get_force()

    def set_checkpoint(self):
        self.checkpoint = {
                'name': self.name, 'setname': self.setname,
                'to': self.to, 'Xpsi': self.inv.sf.Xpsi,
                'coilset': copy.deepcopy(self.coilset[self.setname]),
                'fix': copy.deepcopy(self.inv.fix),
                'limit': copy.deepcopy(self.inv.limit),
                'Fcoil': copy.deepcopy(self.ff.Fcoil)}

    def update_coilset(self):  # synchronize coilset currents
        plasma = self.coilset[self.setname]['plasma']
        It = self.pf.get_coil_current()
        for setname in self.coilset:
            if setname != self.setname:
                self.coilset[setname]['plasma'] = plasma
                self.update_coil_current(self.coilset[setname], It)

    @staticmethod
    def update_coil_current(coilset, It):
        for name in It:
            if name in coilset['coil']:
                setname_array = [name]
                current_array = [It[name]]
            elif name == 'CS1':  # single to pair
                setname_array = ['CS1L', 'CS1U']
                current_array = 0.5 * It[name] * np.ones(2)
            else:  # pair to single
                setname_array = ['CS1']
                current_array = [2 * It[name]]
            for setname, current in zip(setname_array, current_array):
                coilset['coil'][setname]['It'] = current
                Nf = coilset['coil'][setname]['Nf']
                for i in range(Nf):
                    subname = setname+'_{}'.format(i)
                    coilset['subcoil'][subname]['If'] = current / Nf

    def update_psi(self, n=None, limit=None, plot=False, ax=None,
                   current='A', plot_nulls=False):
        if n is not None:
            self.boundary['n'] = n
        if limit is not None:
            self.boundary['limit'] = limit
        x2d, z2d, x, z = geom.grid(self.boundary['n'],
                                   self.boundary['limit'])[:4]
        psi = get_coil_psi(x2d, z2d, self.pf.coilset['subcoil'],
                           self.pf.coilset['plasma'])
        eqdsk = {'x': x, 'z': z, 'psi': psi,
                 'nx': len(x), 'nz': len(z)}
        eqdsk['xlim'] = self.boundary['xlim']
        eqdsk['zlim'] = self.boundary['zlim']
        if 'plasma_parameters' in self.coilset[self.setname]:
            plasma_parameters = self.coilset[self.setname]['plasma_parameters']
            for variable in ['beta', 'li', 'Ipl']:
                eqdsk[variable] = plasma_parameters[variable]
        self.sf.update_eqdsk(eqdsk)
        if plot:
            self.plot_plasma(ax=ax, current=current, plot_nulls=plot_nulls)
        return eqdsk

    def plot_plasma(self, ax=None, plot_nulls=False, current='A'):
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(8, 10))[1]
        self.sf.contour(Xnorm=True, boundary=True,
                        separatrix='', ax=ax)
        try:
            self.sf.sol(update=True, plot=True)
            self.sf.plot_sol(core=True, ax=ax)
        except ValueError:  # legs not found
            pass
        except AttributeError:
            pass
        if plot_nulls:
            self.sf.plot_nulls(labels=['X', 'M'])
        self.sf.plot_firstwall(ax=ax)
        self.inv.plot_fix()
        self.pf.plot(label=True, current=current, patch=False, ax=ax)
        self.pf.plot(subcoil=True, plasma=True, ax=ax)
        self.ff.get_force()
        self.ff.plot(label=True)
        self.ff.plotCS()

    def extract_plasma(self, t):
        if t <= 0 and isinstance(t, int):  # frame_index
            x = self.plasma['x'][-t]
            z = self.plasma['z'][-t]
            # dx, dz = self.plasma['dx'], self.plasma['dz']
            Ipl = self.plasma['I'][-t]
            apl = np.nan
            kpl = np.nan
            beta = np.nan
            li = np.nan
        else:  # time interpolant
            try:
                x = self.fun['Rcur'](t)
                z = self.fun['Zcur'](t)
                apl = self.fun['ap'](t)
                kpl = self.fun['kp'](t)
            except KeyError:
                x = 1e-2*self.fun['Rp'](t)
                z = 1e-2*self.fun['Zp'](t)
                apl = 1e-2*self.fun['a'](t)
                kpl = self.fun['Ksep'](t)
            # dx = 2 * apl * 0.025
            # dz = kpl * dx
            Ipl = 1e6*self.fun['Ip'](t)
            beta = self.fun['BETAp'](t)
            li = self.fun['li(3)'](t)
        plasma_parameters = {'Ipl': Ipl, 'xcur': x, 'zcur': z, 'a': apl,
                             'kappa': kpl, 'beta': beta, 'li': li}
        return plasma_parameters

    def set_plasma(self, plasma_parameters):
        for name in self.coilset:
            self.coilset[name]['plasma_parameters'] = plasma_parameters
        ###
        # eq update here
        ###
        dx, dz = 0.15, 0.25
        self.coilset[self.setname]['plasma'].clear()
        if plasma_parameters['xcur'] > 0:
            plasma = {'x': plasma_parameters['xcur'],
                      'z': plasma_parameters['zcur'],
                      'It': plasma_parameters['Ipl'],
                      'dx': dx, 'dz': dz}
            plasma_subcoil = PF.mesh_coil(plasma, dCoil=0.25)
            for i, filament in enumerate(plasma_subcoil):
                subname = 'Plasma_{}'.format(i)
                self.coilset[self.setname]['plasma'][subname] = filament
        self.inv.update_plasma()
        self.ff.set_force_field(state='passive')
        for setname in self.coilset:  # set passive_field flag
            update = False if setname == self.setname else True
            self.coilset[setname]['force']['update_passive'] = update

    def set_coil_current(self, t):
        It = self.get_coil_current(t, VS3=False)  # get coil currents
        self.pf.update_current(It)

    def load_force_history(self, n=500, plot=False, ax=None, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        filepath = self.filepath + 'force'
        if read_txt or not isfile(filepath + '.pk'):
            self.get_force_history(filepath, n)  # calculate force profiles
        else:
            self.load_pickle(filepath)  # load force profiles
            if len(self.Fcoil) != n:  # nstep mismatch, re-read
                self.get_force_history(filepath, n)
        self.Fsep = {csgap: self.Fcoil[csgap] for csgap in self.CSgap}
        if plot:
            self.plot_force_history(ax=ax)

    def get_force_history(self, filepath, n):
        attributes = ['CSgap', 'Fcoil']
        dtype = [('t', float), ('sep', float), ('zsum', float)]
        CSname = self.pf.coilset['index']['CS']['name']
        self.CSgap = [[] for __ in
                      range(self.pf.coilset['index']['CS']['n'] - 1)]
        PFcoil = [[] for __ in range(self.pf.coilset['index']['PF']['n'])]
        for i in range(self.pf.coilset['index']['CS']['n'] - 1):
            self.CSgap[i] = '{}-{}'.format(CSname[i], CSname[i+1])
            dtype.append((self.CSgap[i], float))
        for i in range(self.pf.coilset['index']['PF']['n']):
            PFcoil[i] = self.pf.coilset['index']['PF']['name'][i]
            dtype.append((PFcoil[i], float))
        self.Fcoil = np.zeros(n, dtype=dtype)
        self.Fcoil['t'] = np.linspace(self.t[1], self.t[-2], n)
        tick = clock(n, header='calculating force history')
        for i, t in enumerate(self.Fcoil['t']):
            self.update_scenario(t)
            Fcoil = self.ff.Fcoil
            self.Fcoil['sep'][i] = Fcoil['CS']['sep']
            self.Fcoil['zsum'][i] = Fcoil['CS']['zsum']
            for j, csgap in enumerate(self.CSgap):
                self.Fcoil[csgap][i] = Fcoil['CS']['sep_array'][j]
            for j, pfcoil in enumerate(PFcoil):
                self.Fcoil[pfcoil][i] = Fcoil['PF']['z_array'][j]
            tick.tock()
        self.save_pickle(filepath, attributes)

    def plot_force_history(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        ax.plot(self.Fcoil['t'], self.Fcoil['sep'], 'gray')
        for csgap in self.CSgap:
            ax.plot(self.Fcoil['t'], self.Fcoil[csgap], label=csgap)
        ax.legend()
        ylim = ax.get_ylim()
        ax.set_ylim([0, ylim[1]])
        ax.set_xlabel('$t$ s')
        ax.set_ylabel('$F_{sep}$ MN')
        imax = np.argmax(self.Fcoil['sep'])
        ax.text(self.Fcoil['t'][imax], self.Fcoil['sep'][imax],
                '$F_{{max}}$ {:1.1f}MN'.format(self.Fcoil['sep'][imax]),
                ha='left', va='bottom')
        plt.despine()

    def load_VS3(self, n=51, plot=False):
        point = OrderedDict()  # VS coil locations
        dtype = [('Bx', float), ('Bz', float), ('Bmag', float),
                 ('t', float), ('index', int)]
        self.VS3 = {}
        for VScoil in ['upper', 'lower']:
            point[VScoil] = np.array(
                    [self.pf.coilset['coil'][VScoil+'VS']['x']+1e-3,
                     self.pf.coilset['coil'][VScoil+'VS']['z']])
            self.VS3[VScoil] = np.zeros(n, dtype=dtype)
        index = self.flattop['index']  # flattop index
        index_array = np.linspace(index[0], index[1], n, dtype=int)
        t = self.t[index_array]
        for VScoil in point:  # set time and index
            self.VS3[VScoil]['t'] = t
            self.VS3[VScoil]['index'] = index_array
        for i, ind in enumerate(index_array):
            self.set_current(ind)
            for VScoil in point:
                B = cc.Bpoint(point[VScoil], self.pf)
                self.VS3[VScoil]['Bx'][i] = B[0]
                self.VS3[VScoil]['Bz'][i] = B[1]
                self.VS3[VScoil]['Bmag'][i] = np.linalg.norm(B)

        if plot:
            ax = plt.subplots(2, 1, sharex=True)[1]
            for i, VScoil in enumerate(self.VS3):
                ax[0].plot(t, self.VS3[VScoil]['Bmag'], '-C{}'.format(i),
                           label=VScoil)
                ax[1].plot(t, self.VS3[VScoil]['Bx'], '--C{}'.format(i))
                ax[1].plot(t, self.VS3[VScoil]['Bz'], '-.C{}'.format(i))
            ax[0].legend()
            h_x = mlines.Line2D([], [], color='gray',
                                linestyle='--', label='$B_x$')
            h_z = mlines.Line2D([], [], color='gray',
                                linestyle='-.', label='$B_z$')
            ax[1].legend(handles=[h_x, h_z])

            plt.xlabel('$t$ s')
            ax[0].set_ylabel('$|B|$ T')
            ax[1].set_ylabel('$B_*$ T')
            plt.despine()

    def get_index(self, index):
        if index <= 0:  # interger frame index
            frame_index = -index
        else:  # time
            if index < np.min(self.t) or index > np.max(self.t):
                errtxt = '\ntime index out of bounds'
                errtxt += '\nindex: {:1.1f}'.format(index)
                errtxt += '\ntmin: {:1.1f}s, tmax: {:1.1f}s'.format(
                        np.min(self.t), np.max(self.t))
                raise IndexError(errtxt)
            frame_index = np.argmin(abs(self.t - index))
        return frame_index

    def get_coil_label(self, coil):
        label = coil[:3] if coil[:2] == 'VS' else coil[:2]
        return label

    def plot_flux(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        ax.plot(self.t, self.data['PSI(axis)'])
        ax.set_xlabel('$t$ s')
        ax.set_ylabel('$\Psi$ Wb')
        plt.despine()

    def plot_current(self, mask=False, apply_filter=False,
                     coils=['CS', 'PF'], line_color=None, ax=None,
                     plot_turn=False):
        if ax is None:
            ax = plt.subplots(1, 1)[1]
        if not hasattr(self, 'opp_index'):
            mask=False
        else:
            index = self.opp_index[1]  # mode index
        if mask:
            alpha = 0.15
        else:
            alpha = 1
        if apply_filter:
            Icoil = self.Icoil_filter
        else:
            Icoil = self.Icoil
        plot_parameters = {}
        plot_parameters['CS'] = {'color': 'C3', 'zorder': 10}
        plot_parameters['PF'] = {'color': 'C0', 'zorder': 5}
        plot_parameters['VS3'] = {'color': 'C9', 'zorder': 2}
        plot_parameters['TF'] = {'color': 'C6', 'zorder': -1}
        plot_parameters['default'] = {'color': 'gray', 'zorder': 0}
        self.inflection_points = {}
        cindex = count(0)
        for coil in self.Icoil:
            label = self.get_coil_label(coil)
            if label in coils:
                plabel = label if label in plot_parameters else 'default'
                if len(coils) == 1:
                    color = 'C{}'.format(next(cindex))
                elif line_color is None:
                    color = plot_parameters[plabel]['color']
                else:
                    color = line_color
                zorder = plot_parameters[plabel]['zorder']
                ax.plot(self.t, 1e-3*Icoil[coil],
                        color=color, zorder=zorder, alpha=alpha, label=coil)
                if mask:
                    ax.plot(self.t[index[0]:index[1]],
                            1e-3*Icoil[coil][index[0]:index[1]],
                            color=color, zorder=zorder)
                turn_index = turning_points(Icoil[coil])
                self.inflection_points[coil] = np.hstack(turn_index)
                if plot_turn:
                    ax.plot(self.t[turn_index],
                            1e-3*Icoil[coil][turn_index],
                            'X', color='C0', zorder=20)
        plt.despine()
        ax.set_xlabel('$t$ s')
        ax.set_ylabel('$I$ kA')
        ax.legend(loc=8, ncol=3)

    def get_VS3_rms(self):
        index = self.opp_index[0]  # flattop
        Ivs3 = self.Icoil['VS3'][index[0]:index[1]]  # line current
        VS3_rms = np.std(Ivs3)
        t = self.t[index[0]:index[1]]
        VS3_rms = np.sqrt(np.trapz(Ivs3**2, t)/(t[-1]-t[0]))
        return VS3_rms

    def get_gap_name(self):
        nCS = self.pf.coilset['index']['CS']['n']
        CSname = self.pf.coilset['index']['CS']['name']
        gap_name = [f'{CSname[i]}_{CSname[i+1]}' for i in range(nCS - 1)]
        gap_name = [f'LDP_{CSname[0]}'] + gap_name + [f'{CSname[-1]}_LDP']
        return gap_name

    def calculate_Fsep(self, FzCS):
        Fsum = np.sum(FzCS, axis=1)
        Fsep = pd.DataFrame(columns=self.get_gap_name())
        Fsep.iloc[:, 0] = Fsum
        Fsep.iloc[:, 1:-1] = np.array(
                [np.sum(FzCS.iloc[:, i+1:], axis=1) -
                 np.sum(FzCS.iloc[:, :i+1], axis=1)
                 for i in range(np.shape(FzCS)[1]-1)]).T
        Fsep.iloc[:, -1] = -Fsum
        return Fsep

    def calculate_Faxial(self, FxCS, FzCS, FcCS):
        gap_name = self.get_gap_name()
        Faxial = pd.DataFrame(columns=gap_name)
        # adjustment to tie-plate tension
        Faxial.iloc[:, -1] = self.tie_plate['alpha'] * FxCS.sum(axis=1)
        Faxial.iloc[:, -1] += (self.tie_plate['beta'] * FzCS).sum(axis=1)
        Faxial.iloc[:, -1] += self.tie_plate['gamma'] * FcCS.sum(axis=1)
        uppergap = gap_name[-1]  # top to bottom
        for i, gap in enumerate(gap_name[::-1][1:]):
            Faxial[gap] = Faxial[uppergap] + FzCS.iloc[:, -(i+1)]
            Faxial[gap] -= self.tie_plate['mg']  # module weight
            uppergap = gap
        # caculate Fbase
        Flimit = 109.04 - 0.526 * Faxial[gap_name[0]]
        Faxial['base'] = FzCS.sum(axis=1) / Flimit *\
            (self.tie_plate['preload'] + self.tie_plate['limit_load'])
        return Faxial

    def get_max_value(self, t, df, ax=None, plot=False, unit='MN', sign=1):
        if hasattr(df, 'stack'):
            index = df.stack().index[np.argmax(sign*df.values)]
            ic = list(df.columns).index(index[1])
            time = t[int(index[0])]
            value = df.loc[int(index[0]), index[1]]
        else:
            index = np.argmax(sign*df.values)
            ic = 0
            time = t[index]
            value = df.loc[index]
        if plot:
            if ax is None:
                ax = plt.gca()
            color = f'C{ic%10}'
            va = 'bottom' if sign == 1 else 'top'
            va = 'center'
            ax.plot(time, value, 'o', color=color)
            ax.text(time, value, f' {value:1.1f}{unit}',
                    va=va, ha='left', color=color)
        return time, value

    def compare_force(self, Nova=False, title=False):
        ax = plt.subplots(2, 1)[1]
        text = [linelabel(postfix='MN', ax=ax[i], value='') for i in range(2)]
        for i, gap in enumerate(self.post['DINA']['Fsep']):
            ax[0].plot(self.post['DINA']['t'], self.post['DINA']['Fsep'][gap],
                       '-', color=f'C{i%10}', label=gap)
            if Nova:
                ax[0].plot(self.post['Nova']['t'],
                           self.post['Nova']['Fsep'][gap].values,
                           '-.', color=f'C{i%10}')
        self.get_max_value(self.post['DINA']['t'], self.post['DINA']['Fsep'],
                           ax=ax[0], plot=True)
        for i, gap in enumerate(self.post['DINA']['Faxial']):
            label = gap.replace('_', '-')
            if label != 'base':
                label = f'gap {6-i}'
            ax[1].plot(self.post['DINA']['t'],
                       self.post['DINA']['Faxial'][gap],
                       '-', color=f'C{i%10}', label=label)
            if Nova:
                ax[1].plot(self.post['Nova']['t'],
                           self.post['Nova']['Faxial'][gap].values,
                           '-.', color=f'C{i%10}')
        self.get_max_value(self.post['DINA']['t'], self.post['DINA']['Faxial'],
                           ax=ax[1], plot=True)
        ax[0].plot(self.post['DINA']['t'],
                   240*np.ones(len(self.post['DINA']['t'])),
                   '--', color='gray', label='limit')
        text[0].add('', value='1.1f')
        axial_limit = self.tie_plate['preload'] + self.tie_plate['limit_load']
        ax[1].plot(self.post['DINA']['t'],
                   axial_limit * np.ones(len(self.post['DINA']['t'])),
                   '--', color='gray')
        text[1].add('', value='1.1f')
        plt.despine()
        plt.detick(ax)
        ax[0].set_ylabel('$F_{sep}$ MN')
        ax[1].set_ylabel('$F_{axial}^*$ MN')
        ax[-1].set_xlabel('$t$ s')
        ax[0].set_ylim([0, 280])
        ax[1].set_ylim([0, 200])
        for i in range(2):
            text[i].plot()
        ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, 2.7),
                     ncol=4)
        if title:
            ax[0].set_title(self.name, y=1.5)

    def compare(self):
        ax = plt.subplots(5, 1, figsize=(9, 9), sharex=True)[1]
        text = [linelabel(ax=ax_, Ndiv=7, value='') for ax_ in ax]

        for name in self.post['DINA']['Ic'].columns[::-1]:
            ax[0].plot(self.post['DINA']['t'], self.post['DINA']['Ic'][name])
            text[0].add(name)
        # text[0].plot()
        ax[0].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1,
                     fontsize='xx-small')
        self.get_max_value(self.post['DINA']['t'],
                           self.post['DINA']['Ic'].loc[:, ::-1],
                           ax=ax[0], plot=True, unit='kA')
        self.get_max_value(self.post['DINA']['t'],
                           self.post['DINA']['Ic'].loc[:, ::-1],
                           ax=ax[0], plot=True, unit='kA', sign=-1)
        ax[0].plot(self.post['DINA']['t'],
                   -45*np.ones(len(self.post['DINA']['t'])),
                   '--', color='gray', zorder=-10, alpha=0.6)
        ax[0].plot(self.post['DINA']['t'],
                   45*np.ones(len(self.post['DINA']['t'])),
                   '--', color='gray', zorder=-10, alpha=0.6)
        ax[0].set_ylim([-60, 55])

        Blimit = interp1d([40, 45], [13, 12.6], fill_value='extrapolate')
        CSname = self.pf.coilset['index']['CS']['name']
        B = self.post['DINA']['B'].loc[:, CSname]
        Bstar = B / Blimit(abs(self.post['DINA']['Ic'].values)) * 12.6
        for name in Bstar.columns[::-1]:
            ax[1].plot(self.post['DINA']['t'], Bstar[name])
            text[1].add(name)
        ax[1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1,
                     fontsize='xx-small')
        # text[1].plot()
        self.get_max_value(self.post['DINA']['t'], Bstar.loc[::-1],
                           ax=ax[1], plot=True, unit='T')
        ax[1].plot(self.post['DINA']['t'],
                   12.6*np.ones(len(self.post['DINA']['t'])), '--',
                   color='gray', zorder=-10, alpha=0.6)

        ax[2].plot(self.post['DINA']['t'], self.post['DINA']['Fsum'])
        ax[2].plot(self.post['DINA']['t'],
                   -60*np.ones(len(self.post['DINA']['t'])), '--',
                   color='gray', zorder=-10, alpha=0.6)
        ax[2].plot(self.post['DINA']['t'],
                   60*np.ones(len(self.post['DINA']['t'])), '--',
                   color='gray', zorder=-10, alpha=0.6)
        self.get_max_value(self.post['DINA']['t'], self.post['DINA']['Fsum'],
                           ax=ax[2], plot=True, unit='MN')
        self.get_max_value(self.post['DINA']['t'], self.post['DINA']['Fsum'],
                           ax=ax[2], plot=True, unit='MN', sign=-1)

        for name in self.post['DINA']['Fsep'].columns[::-1]:
            ax[3].plot(self.post['DINA']['t'],
                       self.post['DINA']['Fsep'][name],
                       label=name.replace('_', '-'))
            text[3].add(name.replace('_', '-'))
        self.get_max_value(self.post['DINA']['t'],
                           self.post['DINA']['Fsep'].iloc[:, ::-1],
                           ax=ax[3], plot=True, unit='MN')
        ax[3].plot(self.post['DINA']['t'],
                   240*np.ones(len(self.post['DINA']['t'])), '--',
                   color='gray', zorder=-10, alpha=0.6)
        ax[3].set_ylim([0, 250])
        # text[3].plot()
        ax[3].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1,
                     fontsize='xx-small')

        index = np.argmin(abs(self.data['t']-self.post['DINA']['t'][-1]))
        ax[4].plot(self.data['t'][:index], self.data['Ip'][:index])

        ax[0].set_ylabel('$I_c$ kA')
        ax[1].set_ylabel('$B^*$ T')
        ax[2].set_ylabel('$F_{sum}$ MN')
        ax[3].set_ylabel('$F_{sep}$ MN')
        ax[4].set_ylabel('$I_p$ MA')
        ax[-1].set_xlabel('$t$ s')
        plt.despine()
        plt.detick(ax)


if __name__ is '__main__':

    scn = scenario(read_txt=False)

    # scn.load_file(folder='15MA DT-DINA2017-04_v1.2')
    # scn.load_file(folder='15MA H-DINA2017-05')
    #exclude = ['15MA DT-DINA2008-01', '15MA DT-DINA2012-01',

    #'15MA DT-DINA2012-03',
    #'15MA DT-DINA2012-05', '15MA DT-DINA2018-05_v1.1']


    scn.load_file(folder='15MA DT-DINA2010-07b', read_txt=False)
    scn.update_scenario(t=100)
    scn.update_psi(plot=True, current='AT', n=5e3)


    # scn.compare_force(title=False)
    # scn.compare()
    #scn.load_file(folder='15MA DT-DINA2014-01', read_txt=False)


    #scn.read_data3(plot=True)


    #scn.update_scenario()
    #scn.update_psi(plot=True, current='AT')
    #scn.operate(plot=True)

    #scn.hip.get_peaks(plot=True)
    '''
    scn.update_scenario()
    scn.update_psi(plot=True, current='AT')
    '''

    # scn.update_scenario(150)
    # scn.update_psi(n=5e3, plot=True, current='AT')

    '''

    for name in scn.pf.coilset['coil']:
        print(name, scn.pf.coilset['coil'][name]['It'])


    scn.load_functions('split')
    # print(scn.pf.coilset['subcoil'])

    for name in scn.pf.coilset['coil']:
        print(name, scn.pf.coilset['coil'][name]['It'])

    scn.update_psi(n=5e3, plot=True)
    '''



    # scn.load_plasma(frame_index=100, plot=True)
    # scn.plot_currents()
    #scn.hip.plot_timeseries()
    #scn.hip.plot_peaks()
    # scn.load_VS3(n=100, plot=True)

    '''
    # It = scn.get_current(ind, VS3=False)  # get coil currents (no VS3)
    #scn.plot_currents()

    scn.load_plasma()
    scn.pf.coil['upperVS']['It'] = -60e3
    scn.pf.coil['lowerVS']['It'] = 60e3
    scn.plot_plasma(scn.flattop_index[-1])
    # scn.load_coils(plot=True)
    '''
