from nep.DINA.read_dina import dina
from amigo.qdafile import QDAfile
from amigo.pyplot import plt
import numpy as np
from nep.coil_geom import PFgeom
from scipy.interpolate import interp1d
from amigo.stat import histopeaks
import matplotlib.lines as mlines
from scipy.signal import periodogram
from nova.elliptic import EQ
from collections import OrderedDict
from nep.DINA.read_eqdsk import read_eqdsk
import nova.cross_coil as cc
from os import path, mkdir, sep
import pickle
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


class read_scenario(pythonIO):

    def __init__(self, database_folder='operations', folder=None,
                 read_txt=False, VS=False, file_type='txt', dCoil=0.25):
        self.date_switch = datetime.strptime('2016-02', '%Y-%m')
        self.read_txt = read_txt
        self.dCoil = dCoil
        self.dina = dina(database_folder)
        if folder is not None:
            self.load_file(folder, file_type=file_type)
        pythonIO.__init__(self)  # python read/write

    def load_file(self, folder, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        dCoil = kwargs.get('dCoil', self.dCoil)
        file_type = kwargs.get('file_type', 'txt')
        filepath = self.dina.locate_file('data2.{}'.format(file_type),
                                         folder=folder)
        filepath = '.'.join(filepath.split('.')[:-1])
        self.filepath = '/'.join(filepath.split('\\')[:-1]) + '/'
        attributes = ['data', 'columns', 'tmax', 'tmin', 'nt', 'dt',
                      't', 'fun', 'Icoil',
                      'Ipl', 'Ipl_lp', 'dIpldt', 'dIpldt_lp', 'noise',
                      'opp_index', 'flattop_index', 'vs3', 'name',
                      'hip', 'date', 'coordinate_switch', 'dCoil',
                      'pf_index', 'pf_coil', 'pf_subcoil',
                      'pf_plasma_coil', 'x', 'z', 'psi', 'xlim',
                      'zlim', 'Fa', 'Fp']  #
        if read_txt or not isfile(filepath + '.pk'):
            self.read_file(folder, file_type=file_type)  # read txt file
            self.save_pickle(filepath, attributes)
        else:
            self.load_pickle(filepath)
            attribute_error = self.load_functions()
            if self.dCoil != dCoil or attribute_error:  # re-read txt file
                self.read_file(folder, file_type=file_type, dCoil=dCoil)
                self.save_pickle(filepath, attributes)

    def load_functions(self):
        attribute_error = False
        try:
            self.pf = PF()  # create bare pf coil instance
            self.pf(self.pf_index, self.pf_coil, self.pf_subcoil,
                    self.pf_plasma_coil)  # __call__
            self.sf = SF(eqdsk={'x': self.x, 'z': self.z, 'psi': self.psi,
                                'Ipl': self.Ipl,
                                'xlim': self.xlim, 'zlim': self.zlim})
            self.ff = force_field(self.pf_index, self.pf_coil, self.pf_subcoil,
                                  self.pf_plasma_coil, multi_filament=True,
                                  Fa=self.Fa, Fp=self.Fp)
            self.inv = INV(self.pf_index, self.pf_coil, self.pf_subcoil,
                           self.pf_plasma_coil, boundary='sf')
            self.inv.update_coils(regrid=False)
            self.inv.set_ITER_limits()
        except AttributeError:
            attribute_error = True
        return attribute_error

    def read_file(self, folder, file_type='txt', dCoil=0.5):
        print('reading {}'.format(folder))
        if file_type == 'txt':
            filename = self.read_txt_file(folder)
        elif file_type == 'qda':
            filename = self.read_qda_file(folder)
        self.name = filename.split(sep)[-3]
        self.date = datetime.strptime(
                self.name.split('DINA')[-1].split('_')[0], '%Y-%m')
        if 'time' in self.data:  # rename time field
            self.data['t'] = self.data['time']
            self.data.pop('time')
        self.trim_nan()
        self.space_data()
        self.load_data()
        self.set_noise()  # level of dz/dt diagnostic noise
        self.opperate()  # identify operating modes
        self.load_coils(dCoil=dCoil, VS=False)  # coil geometory
        self.load_psi()  # load flux map
        self.load_force()  # load force field matrices

    def fix_shape(self):  # set colocation points
        self.inv.colocate(self.sf, ff=self.ff)
        self.inv.set_foreground()
        self.inv.set_background()
        self.inv.get_weight()

    def set_noise(self):
        noise_dict = {'2014-01': 3, '2015-02': 0.6,
                      '2016-01': 0.6, '2017-05': 0.6,
                      '2015-05': 0.6, '2011-06': 0.6,
                      '2012-02': 0.6, '2018-04': 0.6,
                      '2016-01_v1.1': 0.6,
                      '2017-04_v1.2': 0.2,
                      '2017-01': 0.6,
                      '2016-02': 0.0,
                      '2008-01': 0.0}
        key = self.name.split('DINA')[-1]
        try:
            noise = noise_dict[key]
        except KeyError:
            errtxt = 'simulation {}'.format(self.name)
            errtxt += ' not present in noise_dict'
            raise KeyError(errtxt)
        self.noise = noise

    def read_txt_file(self, folder, dropnan=True, force=False):
        filename = self.dina.locate_file('data2.txt', folder=folder)
        binary_data = path.join(self.dina.directory, folder, 'pickle')
        pk_filename = path.join(binary_data, 'data2.pk')
        if path.isfile(pk_filename) and not force:
            with open(pk_filename, 'rb') as f:
                self.data = pickle.load(f)
                self.columns = pickle.load(f)
        else:  # read text data
            self.data, self.columns = \
                self.dina.read_csv(filename, dropnan=False, split=',')
            if not path.isdir(binary_data):
                mkdir(binary_data)
            with open(pk_filename, 'wb') as f:
                pickle.dump(self.data, f)
                pickle.dump(self.columns, f)
        return filename

    def read_qda_file(self, folder):
        filename = self.dina.locate_file('data2.qda', folder=folder)
        self.qdafile = QDAfile(filename)
        self.data = {}
        self.columns = {}
        for i, (var, nrow) in enumerate(zip(self.qdafile.headers,
                                            self.qdafile.rows)):
            var = var.decode()
            if nrow > 0:
                self.columns[var] = var.split(',')[0]
                self.data[self.columns[var]] = \
                    np.array(self.qdafile.data[i, :])
        return filename

    def trim_nan(self):
        nan_index = np.zeros(len(self.data['t']), dtype=int)
        for var in self.data:
            nan_index += np.isnan(self.data[var])
        nan_index = nan_index > 0
        for var in self.data:
            self.data[var] = np.array(self.data[var])[~nan_index]

    def load_coils(self, dCoil=0.15, VS=False, joinCS=True):
        pf_geom = PFgeom(VS=VS, dCoil=dCoil)
        pf = pf_geom.pf
        if joinCS:
            pf.join_coils('CS1', ['CS1L', 'CS1U'])
        self.pf_index = pf.index
        self.pf_coil = pf.coil
        self.pf_subcoil = pf.subcoil
        self.pf_plasma_coil = pf.plasma_coil
        self.dCoil = dCoil

    def load_psi(self):
        sf = read_eqdsk(file='burn').sf  # load referance sf instance
        self.x, self.z, self.psi = sf.x, sf.z, sf.psi
        self.xlim, self.zlim = sf.xlim, sf.zlim

    def load_force(self):
        ff = force_field(self.pf_index, self.pf_coil, self.pf_subcoil,
                         self.pf_plasma_coil, multi_filament=True)
        self.Fa, self.Fp = ff.Fa, ff.Fp  # interaction matrices

    def plot_coils(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(8, 10))[1]
        self.pf.plot(label=True, current=True, patch=False, ax=ax)
        self.pf.plot(subcoil=True, plasma=True, ax=ax)

    def space_data(self):  # generate interpolators and space timeseries
        if self.date > self.date_switch:
            self.coordinate_switch = 1
        else:  # old file - correct coordinates
            self.coordinate_switch = -1
        to = np.copy(self.data['t'])
        dt = np.mean(np.diff(to))
        self.tmax = np.nanmax(to)
        self.tmin = np.nanmin(to)
        self.nt = int(self.tmax/dt)
        self.dt = self.tmax/(self.nt-1)
        self.t = np.linspace(self.tmin, self.tmax, self.nt)
        self.fun = {}
        for var in self.data:  # interpolate
            if ('I' in var and len(var) <= 5) or ('V' in var):
                self.data[var] *= self.coordinate_switch
            self.fun[var] = interp1d(to, self.data[var])
            self.data[var] = self.fun[var](self.t)

    def load_data(self):
        self.Icoil = {}
        for var in self.data:
            if var == 'Ip':  # plasma
                self.Ipl = 1e6*self.data[var]  # MA to A
            elif var[0] == 'I' and len(var) <= 5:
                # kAturn to Aturn
                self.Icoil[var[1:].upper()] = 1e3*self.data[var]
        self.t = self.data['t']
        self.dt = np.mean(np.diff(self.t))

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
        t_ft = self.t[self.flattop_index]
        dpsi = np.diff(self.data['PSI(axis)'][self.flattop_index])[0]
        ft_slice = slice(*self.flattop_index)
        ax.plot(self.t, 1e-6*self.Ipl, 'C0', alpha=alpha)
        ax.plot(self.t[ft_slice], 1e-6*self.Ipl[ft_slice], 'C0', alpha=alpha)
        txt = 'flattop:\n'
        txt += '$t$      {:.1f}-{:1.1f}s\n'.format(*t_ft)
        txt += '$\Delta  t$    {:.1f}s\n'.format(np.diff(t_ft)[0])
        txt += '$\Delta \Psi$  {:.1f}Wb'.format(dpsi)
        ax.text(np.mean(t_ft), 1e-6*self.Ipl[self.flattop_index[0]]/2,
                txt, ma='left', ha='center', va='top',
                bbox=dict(facecolor='w', ec='gray', lw=1,
                          boxstyle='round', pad=0.5))
        ax.set_ylabel('$I_{pl}$ MA')

    def plot_turning_points(self):
        self.load_force_history(n=500)
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
        ft_t = self.t[self.flattop_index]
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
    def opperate(self, plot=False):  # identify operating modes
        trim = np.argmax(self.Ipl[::-1] < 0)
        ind = len(self.Ipl)-trim
        self.dIpldt = np.gradient(self.Ipl[:ind], self.t[:ind])  # gradient
        self.Ipl_lp = lowpass(self.Ipl[:ind], self.dt, dt_window=1)  # current
        self.dIpldt_lp = np.gradient(self.Ipl_lp[:ind], self.t[:ind])  # slope
        self.hip = histopeaks(self.t[:ind], self.dIpldt_lp, nstd=3, nlim=10,
                              nbins=75)  # modes
        self.opp_index = self.hip.timeseries(Ip=self.Ipl[:ind], plot=plot)
        self.flattop_index = self.opp_index[0]  # flattop index
        if 'VS3' in self.Icoil:
            self.vs3 = {
                    't': self.t[self.flattop_index[0]:self.flattop_index[1]],
                    'I': self.Icoil['VS3'][self.flattop_index[0]:
                                           self.flattop_index[1]]}
        else:
            self.vs3 = None

    def get_coil_current(self, index, VS3=True):
        Ic = {}
        for coil in self.Icoil:
            if VS3 or 'VS' not in coil:
                if index < 0:  # frame_index
                    Ic[coil] = self.Icoil[coil][-index]
                else:  # time value
                    Ic[coil] = 1e3*self.fun['I'+coil.lower()](index)
        if 'CS1' not in self.pf.coil:  # split central pair
            for coil in ['CS1L', 'CS1U']:
                Ic[coil] = Ic['CS1']
            Ic.pop('CS1')  # remove CS1 value
        for coil in Ic:  # A to A.turn
            if coil in self.pf.coil:
                Ic[coil] *= self.pf.coil[coil]['N']
        return Ic  # return dict of coil currents

    def update(self, t):  # update from DINA interpolators
        self.set_plasma(t)
        self.set_coil_current(t)
        self.inv.update_coils(regrid=False)
        self.ff.set_force_field(state='passive')
        self.ff.set_current()
        Fcoil = self.ff.get_force()
        return Fcoil

    def update_psi(self, grid=None, plot=False):
        if grid:  # grid={'n': 1e3, 'limit':[3.5, 9, -5.5, 5.5]}
            x2d, z2d, x, z = geom.grid(grid['n'], grid['limit'])[:4]
            psi = get_coil_psi(x2d, z2d, self.pf_subcoil, self.pf_plasma_coil)
            eqdsk = {'x': x, 'z': z, 'psi': psi}
        else:
            psi = get_coil_psi(self.sf.x2d, self.sf.z2d,
                               self.pf.subcoil, self.pf.plasma_coil)
            eqdsk = {'x': self.sf.x, 'z': self.sf.z, 'psi': psi}

        self.sf.update_psi(eqdsk)
        try:
            self.sf.sol()
        except ValueError:
            pass
        if plot:
            levels = self.plot_plasma()
            return levels

    def plot_plasma(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(8, 10))[1]
        levels = self.sf.contour(Xnorm=True, boundary=True,
                                 separatrix='both', ax=ax)
        self.sf.plot_firstwall(ax=ax)
        self.plot_coils(ax=ax)
        self.sf.plot_sol(ax=ax)
        self.ff.plot()
        return levels

    def set_plasma(self, t):
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
        dx = 2 * apl * 0.4
        dz = kpl * dx
        Ipl = 1e6*self.fun['Ip'](t)
        self.pf_plasma_coil.clear()
        if x > 0.0:
            plasma_coil = {'x': x, 'z': z, 'dx': dx, 'dz': dz, 'Ic': Ipl}
            plasma_subcoil = PF.mesh_coil(plasma_coil, 0.25)
            for i, filament in enumerate(plasma_subcoil):
                subname = 'Plasma_{}'.format(i)
                self.pf_plasma_coil[subname] = filament

    def set_coil_current(self, t):
        Ic = self.get_coil_current(t, VS3=False)  # get coil currents
        self.pf.update_current(Ic)

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
        CSname = self.pf.index['CS']['name']
        self.CSgap = [[] for __ in range(self.pf.index['CS']['n'] - 1)]
        PFcoil = [[] for __ in range(self.pf.index['PF']['n'])]
        for i in range(self.pf.index['CS']['n'] - 1):
            self.CSgap[i] = '{}-{}'.format(CSname[i], CSname[i+1])
            dtype.append((self.CSgap[i], float))
        for i in range(self.pf.index['PF']['n']):
            PFcoil[i] = self.pf.index['PF']['name'][i]
            dtype.append((PFcoil[i], float))
        self.Fcoil = np.zeros(n, dtype=dtype)
        self.Fcoil['t'] = np.linspace(self.t[1], self.t[-2], n)
        tick = clock(n, header='calculating force history')
        for i, t in enumerate(self.Fcoil['t']):
            Fcoil = self.update(t)
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
        '''
        for i, center_index in\
                enumerate(self.clusters['fit'].cluster_centers_):
            t_center = self.t[int(center_index.round())]
            Fs = interp1d(self.Fcoil['t'], self.Fcoil['sep'])(t_center)
            ax.plot(t_center, Fs, 'C{}X'.format(i))
        '''
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
            point[VScoil] = np.array([self.pf.coil[VScoil+'VS']['x']+1e-3,
                                      self.pf.coil[VScoil+'VS']['z']])
            self.VS3[VScoil] = np.zeros(n, dtype=dtype)
        index = self.flattop_index  # flattop index
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
        if index < 0:  # interger frame index
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

    def get_noise(self):
        fs = 1/np.mean(np.diff(self.t))  # requires equal spaced data
        f, Pxx = periodogram(self.data['dZ/dt'], fs,
                             'flattop', scaling='spectrum')
        plt.figure()
        plt.semilogy(f, np.sqrt(Pxx))


if __name__ is '__main__':

    scn = read_scenario(read_txt=False)
    scn.load_file(folder='15MA H-DINA2018-04')

    scn.load_plasma(frame_index=100, plot=True)

    # scn.plot_currents()

    #scn.hip.plot_timeseries()
    #scn.hip.plot_peaks()

    # scn.load_VS3(n=100, plot=True)

    '''
    # Ic = scn.get_current(ind, VS3=False)  # get coil currents (no VS3)
    #scn.plot_currents()

    scn.load_plasma()
    scn.pf.coil['upperVS']['Ic'] = -60e3
    scn.pf.coil['lowerVS']['Ic'] = 60e3
    scn.plot_plasma(scn.flattop_index[-1])

    # scn.get_noise()
    # scn.load_coils(plot=True)
    '''
