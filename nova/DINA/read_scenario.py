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
                 read_txt=False, VS=False, file_type='txt', dCoil=0.25,
                 setname='link'):
        self.date_switch = datetime.strptime('2016-02', '%Y-%m')
        self.read_txt = read_txt
        self.setname = setname
        self.dCoil = dCoil
        self.dina = dina(database_folder)
        if folder is not None:
            self.load_file(folder, file_type=file_type)
        pythonIO.__init__(self)  # python read/write

    def load_file(self, folder, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        setname = kwargs.get('setname', self.setname)
        dCoil = kwargs.get('dCoil', self.dCoil)
        file_type = kwargs.get('file_type', 'txt')
        filepath = self.dina.locate_file('data2.{}'.format(file_type),
                                         folder=folder)
        filepath = '.'.join(filepath.split('.')[:-1])
        self.filepath = '/'.join(filepath.split('\\')[:-1]) + '/'
        attributes = ['name', 'date', 'data', 'columns',
                      't', 'dt', 'fun', 'Icoil', 'Ipl', 'flattop', 'hip',
                      'boundary', 'coilset', 'dCoil']
        if read_txt or not isfile(filepath + '.pk'):
            self.read_file(folder, file_type=file_type, dCoil=dCoil)
            self.save_pickle(filepath, attributes)
        else:
            try:
                self.load_pickle(filepath)
            except pickle.UnpicklingError:
                print('pickle error - lfs pointer found')
                self.read_file(folder, file_type=file_type, dCoil=dCoil)
                self.save_pickle(filepath, attributes)
            if dCoil != self.dCoil:  # reload coilset
                print('reloading - dCoil {}'.format(dCoil))
                self.load_coilset(dCoil=dCoil, VS=False)  # coil geometory
                self.save_pickle(filepath, attributes)
        self.load_functions(setname=setname)
        self.set_limits()

    def load_functions(self, setname='link'):
        if setname != self.setname:
            self.update_coilset()
        self.setname = setname
        self.pf = PF()  # create bare pf coil instance
        self.pf(self.coilset[self.setname])
        self.sf = SF()  # creat bare sf instance
        self.ff = force_field(self.coilset[self.setname])
        if self.coilset[self.setname]['update_passive_field']:
            self.ff.set_force_field(state='passive')
        self.ff.get_force()
        self.inv = INV(self.coilset[self.setname], boundary='sf')
        self.inv.sf = self.sf  # link sf instance
        self.inv.ff = self.ff  # link ff instance

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
        self.opperate()  # identify operating modes
        self.load_boundary()  # load flux map limits and fw profile
        self.load_coilset(dCoil=dCoil, VS=False)  # coil geometory

    def fix_shape(self, plot=False, **kwargs):  # set colocation points
        t = kwargs.get('t', self.flattop['t'][-1])  # eof
        n = kwargs.get('n', self.boundary['n'])
        limit = kwargs.get('limit', self.boundary['limit'])
        self.update_DINA(t)
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

    #def solve(self, fff):  # flattop flux factor
    #    flux =

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

    def load_coilset(self, dCoil=0.25, VS=False):
        self.dCoil = dCoil
        dCoil_f = self.dCoil / np.sqrt(2)  # fine mesh
        self.coilset = {}
        self.coilset['split'] = self.load_coil(self.dCoil, VS, joinCS=False)
        self.coilset['split_f'] = self.load_coil(dCoil_f, VS, joinCS=False)
        self.coilset['link'] = self.load_coil(self.dCoil, VS, joinCS=True)
        self.coilset['link_f'] = self.load_coil(dCoil_f, VS, joinCS=True)

    def load_coil(self, dCoil, VS, joinCS=True):
        pf = PFgeom(VS=VS, dCoil=dCoil).pf
        if joinCS:
            pf.join_coils('CS1', ['CS1L', 'CS1U'])
        coilset = {'index': pf.index, 'coil': pf.coil, 'subcoil': pf.subcoil,
                   'plasma_coil': pf.plasma_coil, 'dCoil': dCoil,
                   'update_passive_field': True}
        self.load_force(coilset)  # append interaction matrices
        return coilset

    def load_boundary(self):
        sf = read_eqdsk(file='burn').sf  # load referance sf instance
        self.boundary = {'n': sf.n, 'limit': sf.limit,
                         'xlim': sf.xlim, 'zlim': sf.zlim}

    def load_force(self, coilset):
        ff = force_field(coilset, multi_filament=True)
        coilset['Fa'] = ff.Fa  # interaction matrices
        coilset['Fp'] = ff.Fp

    def plot_coils(self, ax=None, current='AT'):
        if ax is None:
            ax = plt.gca()
        self.pf.plot(label=True, current=current, patch=False, ax=ax)
        self.pf.plot(subcoil=True, plasma=True, ax=ax)

    def space_data(self):  # generate interpolators and space timeseries
        if self.date > self.date_switch:
            coordinate_switch = 1
        else:  # old file - correct coordinates
            coordinate_switch = -1
        to = np.copy(self.data['t'])
        dt = np.mean(np.diff(to))
        tmax = np.nanmax(to)
        tmin = np.nanmin(to)
        nt = int(tmax/dt)
        dt = tmax/(nt-1)
        self.t = np.linspace(tmin, tmax, nt)
        self.fun = {}
        for var in self.data:  # interpolate
            if ('I' in var and len(var) <= 5) or ('V' in var):
                self.data[var] *= coordinate_switch
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
                txt, ma='left', ha='center', va='top',
                bbox=dict(facecolor='w', ec='gray', lw=1,
                          boxstyle='round', pad=0.5))
        ax.set_ylabel('$I_{pl}$ MA')

    def plot_turning_points(self, read_txt=False):
        self.load_force_history(n=500, read_txt=read_txt)
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
        print(max_vector[0])
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
        Ipl_lp = lowpass(self.Ipl[:ind], self.dt, dt_window=1)  # current
        dIpldt_lp = np.gradient(Ipl_lp[:ind], self.t[:ind])  # slope
        self.hip = histopeaks(self.t[:ind], dIpldt_lp, nstd=3, nlim=10,
                              nbins=75)  # modes
        opp_index = self.hip.timeseries(Ip=self.Ipl[:ind], plot=plot)
        self.flattop = {}  # plasma current flattop
        self.flattop['index'] = opp_index[0]
        self.flattop['t'] = self.t[opp_index[0]]
        self.flattop['dt'] = np.diff(self.flattop['t'])[0]
        self.flattop['dpsi'] = \
            np.diff(self.data['PSI(axis)'][self.flattop['index']])[0]

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

    def update_DINA(self, t):  # update from DINA interpolators
        self.set_coil_current(t)
        self.set_plasma(t)
        self.update_coilset()
        Fcoil = self.ff.get_force()
        return Fcoil

    def update_coilset(self):  # synchronize coilset currents
        plasma_coil = self.coilset[self.setname]['plasma_coil']
        Ic = self.pf.get_coil_current()
        for setname in self.coilset:
            if setname != self.setname:
                self.coilset[setname]['plasma_coil'] = plasma_coil
                self.update_coil_current(self.coilset[setname], Ic)

    def update_coil_current(self, coilset, Ic):
        for name in Ic:
            if name in coilset['coil']:
                setname_array = [name]
                current_array = [Ic[name]]
            elif name == 'CS1':  # single to pair
                setname_array = ['CS1L', 'CS1U']
                current_array = 0.5 * Ic[name] * np.ones(2)
            else:  # pair to single
                setname_array = ['CS1']
                current_array = [2 * Ic[name]]
            for setname, current in zip(setname_array, current_array):
                coilset['coil'][setname]['Ic'] = current
                Nf = coilset['subcoil'][setname+'_0']['Nf']
                for i in range(Nf):
                    subname = setname+'_{}'.format(i)
                    coilset['subcoil'][subname]['Ic'] = current / Nf

    def update_psi(self, n=None, limit=None, plot=False, ax=None,
                   current='A'):
        if n is not None:
            self.boundary['n'] = n
        if limit is not None:
            self.boundary['limit'] = limit
        x2d, z2d, x, z = geom.grid(self.boundary['n'],
                                   self.boundary['limit'])[:4]
        psi = get_coil_psi(x2d, z2d, self.pf.subcoil, self.pf.plasma_coil)
        eqdsk = {'x': x, 'z': z, 'psi': psi, 'beta': self.plasma['beta'],
                 'li': self.plasma['li'], 'Ipl': self.plasma['Ipl']}
        eqdsk['xlim'] = self.boundary['xlim']
        eqdsk['zlim'] = self.boundary['zlim']
        self.sf.update_eqdsk(eqdsk)
        if plot:
            self.plot_plasma(ax=ax, current=current)
        return eqdsk

    def plot_plasma(self, ax=None, plot_nulls=False, current='A'):
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(8, 10))[1]
        self.sf.contour(Xnorm=True, boundary=True,
                        separatrix='both', ax=ax)
        self.sf.plot_sol(core=True, ax=ax)
        if plot_nulls:
            self.sf.plot_nulls(labels=['X', 'M'])
        self.sf.plot_firstwall(ax=ax)
        self.plot_coils(ax=ax, current=current)
        self.ff.get_force()
        self.ff.plot()
        self.ff.plotCS()

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
        beta = self.fun['BETAp'](t)
        li = self.fun['li(3)'](t)
        self.plasma = {'Ipl': Ipl, 'a': apl, 'kappa': kpl,
                       'beta': beta, 'li': li}
        ###
        # eq update here
        ###
        self.coilset[self.setname]['plasma_coil'].clear()
        if x > 0.0:
            plasma_coil = {'x': x, 'z': z, 'dx': dx, 'dz': dz, 'Ic': Ipl}
            plasma_subcoil = PF.mesh_coil(plasma_coil, 0.25)
            for i, filament in enumerate(plasma_subcoil):
                subname = 'Plasma_{}'.format(i)
                self.coilset[self.setname]['plasma_coil'][subname] = filament
        self.inv.update_plasma_coil()
        self.ff.set_force_field(state='passive')
        for setname in self.coilset:  # set passive_field flag
            update = False if setname == self.setname else True
            self.coilset[setname]['update_passive_field'] = update

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
            Fcoil = self.update_DINA(t)
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
    # scn.load_coils(plot=True)
    '''
