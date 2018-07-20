import numpy as np
from nova.force import force_field
from nep.coil_geom import PFgeom, VSgeom, VVcoils
from nep.DINA.read_tor import read_tor
from collections import OrderedDict
import nova.cross_coil as cc
from nova.streamfunction import SF
from amigo.geom import grid
from nep.DINA.read_plasma import read_plasma
from amigo.pyplot import plt
from amigo.time import clock
from amigo.geom import qrotate
import matplotlib.animation as manimation
from nep.rails import stress_allowable
from nep.DINA.read_dina import dina
import matplotlib.patches as mpatches
import pickle
from os.path import split, join, isfile
import matplotlib.lines as mlines
import os
import nep
from amigo.png_tools import data_load
from amigo.IO import class_dir, pythonIO
from nep.DINA.capacitor_discharge import power_supply


class VDE_force(pythonIO):

    def __init__(self, mode='control', discharge='DINA', Iscale=1,
                 read_txt=False):
        self.Iscale = Iscale
        self.read_txt = read_txt
        self.mode = mode
        self.discharge = discharge
        self.dina = dina('disruptions')
        self.pl = read_plasma('disruptions', Iscale=self.Iscale,
                              read_txt=read_txt)  # load plasma
        self.tor = read_tor('disruptions', Iscale=self.Iscale,
                            read_txt=read_txt)  # load currents
        self.allowable = stress_allowable()  # load allowable interpolators
        pythonIO.__init__(self)  # python read/write

    def load_file(self, folder, frame_index=0, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        self.mode = kwargs.get('mode', self.mode)
        self.discharge = kwargs.get('discharge', self.discharge)
        filepath = self.dina.locate_file('plasma', folder=folder)
        self.name = split(filepath)[-2]
        filepath = join(*split(filepath)[:-1], self.name, 'VDE_force')
        if self.discharge == 'IO':
            filepath += 'IO'
        if read_txt or not isfile(filepath + '.pk'):
            self.read_file(folder)  # read txt file
            self.save_pickle(filepath, ['pf', 'ff', 'vs_rail', 'vs_theta'])
        else:
            self.load_pickle(filepath)
        self.tor.load_file(folder)  # read toroidal strucutres
        self.load_IO(folder)  # load hig-res vv model
        #*************
        # self.load_vs3(folder, discharge=self.discharge)  # load vs3 currents
        #*************

        self.frame_update(frame_index)  # initalize at start of timeseries

        # self.vs3_update(mode=self.mode)  # initalize vs3 current
        # self.force_update()  # update vs3 coil forces
        self.initalize_sf()

    def load_IO(self, scenario, t_pulse=0.3):
        if self.discharge == 'IO':
            self.ps = power_supply(scenario=scenario, vessel=True)
            self.ps.solve(self.tor.t[-1], Io=0, sign=-1, nturn=4,
                          t_pulse=t_pulse, impulse=True, pulse=False, plot=True)

    def read_file(self, folder):
        self.load_active()  # load active coils
        self.load_passive(folder)  # load toroidal strucutres
        self.set_force_field()  # initalise force_field object

    def set_force_field(self):
        active_coils, passive_coils = self.set_coil_type()
        self.ff = force_field(self.pf.index, self.pf.coil, self.pf.sub_coil,
                              self.pf.plasma_coil, multi_filament=True,
                              active_coils=active_coils,
                              passive_coils=passive_coils)

    def set_coil_type(self):
        active_coils = self.pf.index['VS3']['name']
        passive_coils = list(self.pf.coil.keys())
        for coil in active_coils:
            passive_coils.remove(coil)
        passive_coils.append('Plasma')
        return active_coils, passive_coils

    def load_vs3(self, folder, discharge='DINA'):
        # load current interpolator
        self.Ivs3_fun = \
            self.pl.Ivs3_single(folder, discharge=discharge)[-1]

    def load_active(self, dCoil=0.25):
        vs_geom = VSgeom()
        self.vs_rail = vs_geom.rail
        pf_geom = PFgeom(VS=False, dCoil=dCoil)
        self.pf = pf_geom.pf

        self.vs_theta = {}
        for name in vs_geom.geom:
            self.vs_theta[name] = vs_geom.geom[name]['theta']

    def add_IO(self):
        if self.discharge == 'IO':
            vv_cut = [0, 1] + list(np.arange(18, 23)) + \
                list(np.arange(57, 60)) + list(np.arange(91, 96)) +\
                list(np.arange(72, 76)) + list(np.arange(114, 116))
            for i in vv_cut:  # remove DINA coils
                self.tor.vessel_coil.pop('vv_{}'.format(i))
            vv = VVcoils()
            for part in ['VS3', 'trs', 'vv']:  # add high-res set
                coil = OrderedDict()
                for name in vv.pf.index[part]['name']:
                    coil[name] = vv.pf.coil[name]
                self.add_filament(coil, sub_coil=vv.pf.sub_coil, index=part)

    def load_passive(self, folder):
        self.tor.load_file(folder)  # read toroidal strucutres
        self.add_IO()  # add high-res coils + remove DINA
        self.add_filament(self.tor.vessel_coil, index='vv_DINA')
        self.add_filament(self.tor.blanket_coil, index='bb_DINA')

    def add_filament(self, filament, sub_coil=None, index=None):
        nCo = self.pf.nC  # start index
        name = []
        for coil in filament:
            name.append(coil)
            self.pf.coil[coil] = filament[coil]
            if sub_coil:
                Nf = sub_coil[coil+'_0']['Nf']
                for i in range(Nf):
                    subname = coil+'_{}'.format(i)
                    self.pf.sub_coil[subname] = sub_coil[subname]
            else:
                self.pf.sub_coil[coil+'_0'] = filament[coil]
                self.pf.sub_coil[coil+'_0']['Nf'] = 1
        if index:
            self.pf.index[index] = {'name': name,
                                    'index': np.arange(nCo, self.pf.nC)}

    def load_plasma(self, frame_index):
        self.pf.plasma_coil.clear()  # clear
        for name in self.tor.plasma_coil[frame_index]:
            self.pf.plasma_coil[name] = self.tor.plasma_coil[frame_index][name]

    def set_coil_current(self, frame_index):  # PF / Cs
        Ic = dict(zip(self.tor.coil.keys(),
                      self.tor.current['coil'][frame_index]))
        self.pf.update_current(Ic)  # PF / CS coils

    def set_filament_current(self, filament, frame_index):  # vessel / blanket
        Ic = {}  # initalize
        current = self.tor.current['filament'][frame_index]
        for name in filament:
            turn_index = filament[name]['index']
            sign = filament[name]['sign']
            Ic[name] = sign * current[turn_index]
        self.pf.update_current(Ic)

    def frame_update(self, frame_index, vessel=True, blanket=True):
        self.frame_index = frame_index
        self.t = self.tor.t[self.frame_index]
        self.set_coil_current(frame_index)
        if vessel:
            self.set_filament_current(self.tor.vessel_coil, frame_index)
        if blanket:
            self.set_filament_current(self.tor.blanket_coil, frame_index)
        self.load_plasma(frame_index)

    def vs3_update(self, **kwargs):
        print('updating vs3')
        self.mode = kwargs.get('mode', self.mode)
        Ivs3 = self.Ivs3_fun[self.mode](self.t)
        self.set_vs3_current(Ivs3)

    def force_update(self):
        self.ff.set_passive_force_field()  # update plasma force field
        self.ff.set_current()  # update fillament currents
        self.ff.set_force(self.ff.Ic)  # update force calculation

    def set_vs3_current(self, Ivs3):
        self.Ivs3 = float(Ivs3)  # store
        Ic = {'upperVS': -4*Ivs3, 'lowerVS': 4*Ivs3}
        self.pf.update_current(Ic)

    def initalize_sf(self):
        n, limit = 1e4, [1.5, 10, -8.5, 8.5]
        self.x2d, self.z2d, self.x, self.z = grid(n, limit)[:4]
        self.psi = cc.get_coil_psi(self.x2d, self.z2d, self.pf)

        self.sf = SF(eqdsk={'x': self.x, 'z': self.z, 'psi': self.psi,
                            'name': 'DINA_{}'.format(self.tor.name)})

    def contour(self, **kwargs):
        self.psi = cc.get_coil_psi(self.x2d, self.z2d, self.pf)
        self.sf = SF(eqdsk={'x': self.x, 'z': self.z, 'psi': self.psi,
                            'name': 'DINA_{}'.format(self.tor.name)})
        levels = self.sf.contour(51, boundary=False, Xnorm=False, **kwargs)
        return levels

    def get_frames(self, nframe):
        if nframe is None:
            nframe = self.tor.nt
        nframe = int(nframe)
        time = np.linspace(self.tor.t[0], self.tor.t[-2], nframe)
        frames = np.zeros(nframe, dtype=int)
        for i, t in enumerate(time):
            frames[i] = np.argmin(abs(t-self.tor.t))
        return frames, nframe

    def get_stress(self, Fn, Ft, name):
        dx, dy = self.vs_rail[name]['dx'], self.vs_rail[name]['dy']
        dz, n = self.vs_rail[name]['dz'], self.vs_rail[name]['n']
        coil = self.pf.coil[name]
        Ft *= 2*np.pi*coil['x']/n  # force per rail
        Fn *= 2*np.pi*coil['x']/n
        Stress = {}
        Force = {'x': 0, 'y': Ft, 'z': Fn}
        Moment = {'x': Force['y']*dz, 'y': Force['x']*dz, 'z': 0}
        area = dx*dy
        for var in Force:
            Stress[var] = abs(Force[var]) / area
        Ix = dx*dy**3/12  # second moments
        Iy = dy*dx**3/12
        #  stress linerization
        Pm = np.sqrt(Stress['z']**2 + 3*(Stress['x']**2 + Stress['y']**2))
        PmPb = np.sqrt((Stress['z'] +
                        dx*abs(Moment['y'])/(2*Iy) +
                        dy*abs(Moment['x'])/(2*Ix))**2 +
                       3 * ((Stress['x'])**2 + (Stress['y'])**2))
        Sm = self.allowable.get_limit(weld_category=2, load_category=1,
                                      load_type='primary', T=100)
        fPm, fPmPb = 1e-6*Pm/Sm, 1e-6*PmPb/(1.5*Sm)
        return max(fPm, fPmPb)

    def get_field(self, nframe=None):
        B_data = OrderedDict()
        data_dtype = [('Bx', '2float'), ('Bz', '2float'), ('Bmag', '2float'),
                      ('t', float),
                      ('frame_index', int)]

        frames, nframe = self.get_frames(nframe)
        tick = clock(nframe)
        for mode in ['referance', 'control', 'error']:
            B_data[mode] = np.zeros(nframe, dtype=data_dtype)
        for i, frame_index in enumerate(frames):
            self.frame_update(frame_index)
            for mode in B_data:
                self.vs3_update(mode=mode)
                for j, name in enumerate(self.pf.index['VS3']['name']):
                    coil = self.pf.coil[name]
                    point = [coil['x'], coil['z']]
                    B = cc.Bpoint(point, self.pf)
                    B_data[mode][i]['Bx'][j] = B[0]
                    B_data[mode][i]['Bz'][j] = B[1]
                    B_data[mode][i]['Bmag'][j] = np.linalg.norm(B)
                    B_data[mode][i]['frame_index'] = self.frame_index
                    B_data[mode][i]['t'] = self.t
            tick.tock()
        return B_data

    def get_data(self, nframe=None, plot=False, pvar='sigma'):
        vs3_data = OrderedDict()
        data_dtype = [('Fx', '2float'), ('Fz', '2float'), ('Fmag', '2float'),
                      ('Fn', '2float'), ('Ft', '2float'), ('sigma', '2float'),
                      ('t', float), ('I', float),
                      ('frame_index', int),
                      ('Bx', '2float'), ('Bz', '2float'), ('Bmag', '2float')]
        frames, nframe = self.get_frames(nframe)
        tick = clock(nframe)
        for mode in ['referance', 'control', 'error']:
            vs3_data[mode] = np.zeros(nframe, dtype=data_dtype)
        for i, frame_index in enumerate(frames):
            self.frame_update(frame_index)
            for mode in vs3_data:
                self.vs3_update(mode=mode)
                self.force_update()
                for j, name in enumerate(self.pf.index['VS3']['name']):
                    # force
                    F_index = self.ff.active_coils.index(name)
                    coil = self.pf.coil[name]
                    F = 1e6 * self.ff.F[F_index] / (2*np.pi*coil['x'])  # N/m
                    Fxyz = np.array([F[0], 0, F[1]])
                    theta = self.vs_theta[name]
                    Ftn = qrotate(Fxyz, theta=theta, dx=[0, 1, 0])[0]
                    Fmag = np.linalg.norm(F)
                    sigma = self.get_stress(Ftn[-1], Ftn[0], name)
                    vs3_data[mode][i]['frame_index'] = self.frame_index
                    vs3_data[mode][i]['t'] = self.t
                    vs3_data[mode][i]['I'] = self.Ivs3
                    vs3_data[mode][i]['Fx'][j] = F[0]
                    vs3_data[mode][i]['Fz'][j] = F[1]
                    vs3_data[mode][i]['Fn'][j] = Ftn[-1]
                    vs3_data[mode][i]['Ft'][j] = Ftn[0]
                    vs3_data[mode][i]['Fmag'][j] = Fmag
                    vs3_data[mode][i]['sigma'][j] = sigma
                    # centerpoint field
                    centerpoint = [coil['x'], coil['z']]
                    B = cc.Bpoint(centerpoint, self.pf)
                    vs3_data[mode][i]['Bx'][j] = B[0]
                    vs3_data[mode][i]['Bz'][j] = B[1]
                    vs3_data[mode][i]['Bmag'][j] = np.linalg.norm(B)
            tick.tock()
        if plot:
            if 'F' in pvar:
                factor = 1e-3
                ylabel = '$|F|$, kNm$^{-1}$'
            elif pvar == 'sigma':
                factor = 1
                ylabel = '$\sigma^*$'

            ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))[1]
            ax[0].text(0.5, 1, self.tor.name, transform=ax[0].transAxes,
                       ha='center', va='top', weight='bold')
            for mode, color in zip(vs3_data, ['gray', 'C0', 'C3']):
                ax[0].plot(1e3*vs3_data[mode]['t'],
                           1e-3*vs3_data[mode]['I'], '-',
                           color=color, label=mode)
                for i, name in enumerate(self.pf.index['VS3']['name']):
                    ax[i+1].plot(1e3*vs3_data[mode]['t'],
                                 factor*vs3_data[mode][pvar][:, i],
                                 color=color)
                    max_index = np.nanargmax(vs3_data[mode][pvar][:, i])
                    txt = '{}: {:1.1f}'.format(
                        mode, factor*vs3_data[mode][pvar][max_index, i])
                    ax[i+1].plot(
                            1e3*vs3_data[mode]['t'][max_index],
                            factor*vs3_data[mode][pvar][max_index, i],
                            'o', color=color, label=txt)
            plt.despine()
            ax[0].set_ylabel('$I_{vs3}$, kA')
            ax[0].legend(loc=1)
            for i, name in enumerate(self.pf.index['VS3']['name']):
                ax[i+1].set_ylabel(ylabel)
                ax[i+1].text(0.5, 1, name, transform=ax[i+1].transAxes,
                             ha='center', va='top',
                             bbox=dict(facecolor='gray', alpha=0.25))
                ax[i+1].legend()
            for i in range(2):
                plt.setp(ax[i].get_xticklabels(), visible=False)
            ax[2].set_xlabel('$t$, ms')
        return vs3_data

    def movie(self, folder, nframe=None, mode='referance', discharge='DINA'):
        self.read_file(folder, discharge=discharge)
        frames, nframe = self.get_frames(nframe)
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=10, bitrate=-1)
        fig = plt.figure(figsize=(6, 10))
        tick = clock(nframe)
        filename = '../Movies/{}_{}.mp4'.format(self.tor.name, mode)
        levels = self.plot_frame(0, mode=mode)
        with writer.saving(fig, filename, 72):
            for frame_index in frames:
                plt.clf()
                self.plot_frame(frame_index, mode=mode, levels=levels)
                writer.grab_frame()
                tick.tock()

    def plot_frame(self, **kwargs):
        ax = kwargs.get('ax', None)
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(6, 9))[1]
        if 'frame_index' in kwargs:
            self.frame_update(kwargs['frame_index'])
        if 'frame_index' in kwargs or 'mode' in kwargs:
            self.vs3_update(**kwargs)
            self.force_update()
        self.pf.plot(subcoil=True, plasma=True, ax=ax)
        self.ff.plot(coils=['VS3'], scale=3, Fmax=10)
        levels = self.contour(**kwargs)
        return levels

    def plot_coordinates(self):
        for i, name in enumerate(self.pf.index['VS3']['name']):
            theta = self.vs_theta[name]
            coil = self.pf.coil[name]
            xo, zo, dl = coil['x'], coil['z'], 1.5
            dx = np.array([[xo, xo+dl], [0, 0], [zo, zo]]).T
            dz = qrotate(dx, theta=-np.pi/2, xo=[xo, 0, zo], dx=[0, 1, 0])
            # rotate to clamp local coordinate system
            dx_ = qrotate(dx, theta=theta, xo=[xo, 0, zo], dx=[0, 1, 0])
            dz_ = qrotate(dz, theta=theta, xo=[xo, 0, zo], dx=[0, 1, 0])
            plt.plot(dx[:, 0], dx[:, -1], 'C0')
            plt.plot(dz[:, 0], dz[:, -1], 'C1')
            plt.plot(dx_[:, 0], dx_[:, -1], 'C0-.')
            plt.plot(dz_[:, 0], dz_[:, -1], 'C1-.')

    def datafile(self, discharge, nframe):
        filename = join(self.dina.root,
                        'DINA/Data/vs3_{}_{:d}.plk'.format(discharge,
                                                           nframe))
        if self.Iscale != 1:
            filename = filename.replace(
                    '.plk', '_Iscale_{:1.1f}.plk'.format(self.Iscale))
        return filename

    def write_data(self, discharge, nframe=100):
        vs3_data = OrderedDict()
        X = range(self.dina.nfolder)
        for i in X:
            self.read_file(i, discharge=discharge)
            name = self.tor.name
            vs3_data[name] = self.get_data(nframe, plot=False)
        filename = self.datafile(discharge, nframe)
        with open(filename, 'wb') as output:
            pickle.dump(vs3_data, output, -1)
        return vs3_data

    def read_data(self, discharge, nframe=100, forcewrite=False):
        filename = self.datafile(discharge, nframe)
        if not isfile(filename) or forcewrite:
            txt = '\nre-generating data, discharge:{}, '.format(discharge)
            txt += 'nframe:{}'.format(nframe)
            print(txt)
            vs3_data = self.write_data(discharge, nframe)
        else:
            with open(filename, 'rb') as input:
                vs3_data = pickle.load(input)
        return vs3_data

    def plot_Fmax(self, discharge, nframe=100):
        vs3_data = self.read_data(discharge, nframe)  # load data
        ax = plt.subplots(2, 1, sharex=True, sharey=True)[1]
        X = range(self.dina.nfolder)
        Fmax = OrderedDict()
        Fmax_dtype = [('Fmag', '2float'), ('t', '2float'), ('I', '2float'),
                      ('frame_index', '2float')]
        Imax = OrderedDict()
        Imax_dtype = [('I', float), ('t', float), ('frame_index', float)]
        for mode in ['referance', 'control', 'error']:
            Fmax[mode] = np.zeros(len(X), dtype=Fmax_dtype)
            Imax[mode] = np.zeros(len(X), dtype=Imax_dtype)
        for i, name in enumerate(vs3_data):
            vs3 = vs3_data[name]
            for j, (mode, color, width) in enumerate(
                    zip(['referance', 'control', 'error'],
                        ['gray', 'C0', 'C3'], [0.9, 0.7, 0.5])):
                frame_index = np.nanargmax(abs(vs3[mode]['I']))
                Imax[mode][i]['frame_index'] = frame_index
                for var in ['t', 'I']:
                    Imax[mode][i][var] = vs3[mode][var][frame_index]
                for iax, k in enumerate([1, 0]):  # upper / lower VS coils
                    frame_index = np.nanargmax(vs3[mode]['Fmag'][:, k])
                    Fmax[mode][i]['frame_index'][k] = frame_index
                    Fmax[mode][i]['Fmag'][k] =\
                        vs3[mode]['Fmag'][frame_index, k]
                    for var in ['t', 'I']:
                        Fmax[mode][i][var][k] = vs3[mode][var][frame_index]

                    ax[iax].bar(i, 1e-3*Fmax[mode][i]['Fmag'][k], color=color,
                                width=width, label=mode)
        for i, k in enumerate([1, 0]):
            Fm, im = [], []
            for mode in Fmax:
                Fm.extend(Fmax[mode]['Fmag'][:, k])
                im.extend(list(range(len(vs3_data))))
            index = np.argmax(Fm)
            ax[i].text(im[index], 1e-3*Fm[index],
                       '{:1.0f}'.format(1e-3*Fm[index]), weight='bold',
                       va='bottom', ha='center')
        h = []
        for mode, color in zip(['referance', 'control', 'error'],
                               ['gray', 'C0', 'C3']):
            h.append(mpatches.Patch(color=color, label=mode))
        ax[0].legend(handles=h, loc=2)
        plt.xticks(X, self.dina.folders, rotation=70)
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)
        for i, k in enumerate([1, 0]):
            ax[i].set_ylabel('$|F|_{max}$ kNm$^{-1}$')
            name = self.pf.index['VS3']['name'][k]
            ax[i].text(0.5, 1, name, transform=ax[i].transAxes,
                       weight='bold', bbox=dict(facecolor='gray', alpha=0.25),
                       va='top', ha='center')

        self.pl.Ivs3_ensemble()
        ax = plt.subplots(2, 1, sharex=True, sharey=True)[1]
        for index, file in enumerate(vs3_data):
            vs3 = vs3_data[file]
            color = self.pl.get_color(index)[0]
            for i, k in enumerate([1, 0]):
                ax[i].plot(1e3*vs3['control']['t'],
                           1e-3*vs3['control']['Fmag'][:, k],
                           color=color)

        for i, k in enumerate([1, 0]):
            ax[i].set_ylabel('$|F|$ kNm$^{-1}$')
            name = self.pf.index['VS3']['name'][k]
            ax[i].text(1, 0.1, name, transform=ax[i].transAxes,
                       bbox=dict(facecolor='lightgray', alpha=1),
                       va='bottom', ha='right')
            ax[i].set_xlim([0, 1400])
        ax[1].set_xlabel('$t$ ms')
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)
        h = []
        h.append(mlines.Line2D([], [], color='C0', label='MD down'))
        h.append(mlines.Line2D([], [], color='C1', label='MD up'))
        h.append(mlines.Line2D([], [], color='C2', label='VDE down'))
        h.append(mlines.Line2D([], [], color='C3', label='VDE up'))
        ax[0].legend(handles=h, loc=1)

        mode_index = np.nanargmax(
                [max(abs(Imax[mode]['I'][:])) for mode in Imax])
        mode = list(Imax.keys())[mode_index]
        file_index = np.nanargmax(abs(Imax[mode]['I']))
        file = list(vs3_data.keys())[file_index]
        vs3 = vs3_data[file]
        Imax_index = np.nanargmax(abs(vs3[mode]['I']))
        t_max = vs3[mode]['t'][Imax_index]
        I_max = vs3[mode]['I'][Imax_index]
        frame_index_max = vs3[mode]['frame_index'][Imax_index]
        print('Imax', file, frame_index_max, t_max, I_max)
        print('Fmax', 1e-3*vs3[mode]['Fx'][Imax_index],
              1e-3*vs3[mode]['Fz'][Imax_index],
              1e-3*vs3[mode]['Fmag'][Imax_index])

        ax = plt.subplots(2, 1, sharex=True, sharey=True)[1]
        ax_I = plt.subplots(1, 1)[1]
        for i, (k, color_I) in enumerate(zip([1, 0], ['C6', 'C7'])):
            name = self.pf.index['VS3']['name'][k]
            for j, (color, va) in enumerate(zip(['C0', 'C3'],
                                                ['bottom', 'top'])):
                # MD then VDE
                mode_index = np.nanargmax(
                        [max(Fmax[mode]['Fmag'][slice(j*6, (j+1)*6), k])
                         for mode in Fmax])
                mode = list(Fmax.keys())[mode_index]
                file_index = j*6 + np.nanargmax(
                        Fmax[mode]['Fmag'][slice(j*6, (j+1)*6), k])
                file = list(vs3_data.keys())[file_index]
                vs3 = vs3_data[file]
                Fmax_index = np.nanargmax(vs3[mode]['Fmag'][:, k])
                F_max = vs3[mode]['Fmag'][Fmax_index, k]
                t_max = vs3[mode]['t'][Fmax_index]
                I_max = vs3[mode]['I'][Fmax_index]
                frame_index_max = vs3[mode]['frame_index'][Fmax_index]
                ax[i].plot(1e3*vs3[mode]['t'], 1e-3*vs3[mode]['Fmag'][:, k],
                           color=color_I)
                txt = file+'\n'
                txt += '$t=$'+'{:1.1f}ms, '.format(1e3*t_max)
                txt += '$F=$'+'{:1.0f}'.format(1e-3*F_max)
                txt += 'kNm$^{-1}$'
                ax[i].plot(1e3*t_max, 1e-3*F_max, '*', color=0.3*np.ones(3))
                ax[i].text(1e3*t_max+25,
                           1e-3*F_max, txt, va=va, ha='left')

                ax_I.plot(1e3*vs3[mode]['t'], 1e-3*vs3[mode]['I'][:],
                          color=color_I)
                txt = file+'\n'
                txt += '$t=$'+'{:1.1f}ms, '.format(1e3*t_max)
                txt += '$I=$'+'{:1.0f}'.format(1e-3*I_max)
                txt += 'kA'
                ax_I.plot(1e3*t_max, 1e-3*I_max, '*', color=0.3*np.ones(3))
                vaI = va if I_max > 0 else 'top'
                ax_I.text(1e3*t_max+25, 1e-3*I_max,
                          txt, va=vaI, ha='left')
                print(name, file, frame_index_max, t_max, 1e-3*I_max)
                print('Fmax', 1e-3*vs3[mode]['Fx'][Fmax_index],
                      1e-3*vs3[mode]['Fz'][Fmax_index],
                      1e-3*vs3[mode]['Fmag'][Fmax_index])

        for i, k in enumerate([1, 0]):
            ax[i].set_ylabel('$|F|$ kNm$^{-1}$')
            name = self.pf.index['VS3']['name'][k]
            ax[i].text(1, 0.1, name, transform=ax[i].transAxes,
                       bbox=dict(facecolor='lightgray', alpha=1),
                       va='bottom', ha='right')
        ax[1].set_xlabel('$t$ ms')

        plt.ylim([-120, 80])
        plt.despine()
        plt.xlabel('$t$ ms')
        plt.ylabel('$I_{vs3}$ kA')
        plt.sca(ax[0])
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)

        h = []
        h.append(mlines.Line2D([], [], color='C6', label='upperVS'))
        h.append(mlines.Line2D([], [], color='C7', label='lowerVS'))
        ax_I.legend(handles=h, loc=4)

    def plot_line_force(self, vs3, plot_full):
        control_label = self.get_control_label(plot_full)
        ax = plt.subplots(2, 1, sharex=True, sharey=True)[1]
        for i, k in enumerate([1, 0]):
            for mode, color in zip(['referance', 'control', 'error'],
                                   ['gray', 'C0', 'C3']):
                if mode == 'control' or plot_full:
                    ax[i].plot(1e3*vs3[mode]['t'],
                               1e-3*vs3[mode]['Fmag'][:, k],
                               color=color)
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)
        LTC = {}  # load and plot LTC data
        path = os.path.join(class_dir(nep), '../Data/LTC/')
        points = data_load(path, 'VS3_force', date='2018_03_15')[0]
        LTC['lowerVS'] = {'t': points[0]['x'], 'Fmag': points[0]['y']}
        LTC['upperVS'] = {'t': points[1]['x'], 'Fmag': points[1]['y']}
        for i, coil in enumerate(['upperVS', 'lowerVS']):
            ax[i].plot(1e3*LTC[coil]['t'], 1e-3*LTC[coil]['Fmag'],
                       ls='-', color=0.3*np.ones(3))
        ax[0].set_ylabel('upper $|F|$ kNm$^{-1}$')
        ax[1].set_ylabel('lower $|F|$ kNm$^{-1}$')
        ax[1].set_xlabel('$t$ ms')
        h = []
        h.append(mlines.Line2D([], [], ls='-',
                               color=0.3*np.ones(3), label='LTC'))
        if plot_full:
            h.append(mlines.Line2D([], [], color='gray', label='referance'))
        h.append(mlines.Line2D([], [], color='C0', label=control_label))
        if plot_full:
            h.append(mlines.Line2D([], [], color='C3', label='error'))
        ax[0].legend(handles=h, loc=1)

    def plot_current(self, vs3, plot_full, file):
        control_label = self.get_control_label(plot_full)
        ax = plt.subplots(1, 1, sharex=True, sharey=True)[1]
        for mode, color in zip(['referance', 'control', 'error'],
                               ['gray', 'C0', 'C3']):
            if mode == 'control' or plot_full:
                ax.plot(1e3*vs3[mode]['t'], 1e-3*vs3[mode]['I'],
                        color=color)
        plt.despine()
        ax.set_ylabel('$I$ kA')
        ax.set_xlabel('$t$ ms')
        path = os.path.join(class_dir(nep), '../Data/LTC/')
        if file == 'VDE_UP_slow_fast':
            points = data_load(path, 'VS3_current_VDE',
                               date='2018_05_24')[0]
        elif file == 'MD_UP_exp16':
            points = data_load(path, 'VS3_current', date='2018_03_15')[0]
        t = points[0]['x']
        Ic = points[0]['y']
        ax.plot(1e3*t, -1e-3*Ic, ls='-', color=0.3*np.ones(3))
        h = []
        h.append(mlines.Line2D([], [], ls='-',
                               color=0.3*np.ones(3), label='LTC'))
        if plot_full:
            h.append(mlines.Line2D([], [], color='gray', label='referance'))
        h.append(mlines.Line2D([], [], color='C0', label=control_label))
        if plot_full:
            h.append(mlines.Line2D([], [], color='C3', label='error'))
        ax.legend(handles=h, loc=1)

    def plot_Bmag(self, vs3, plot_full):
        control_label = self.get_control_label(plot_full)
        ax = plt.subplots(2, 1, sharex=True, sharey=True)[1]
        for i, k in enumerate([1, 0]):
            for mode, color in zip(['referance', 'control', 'error'],
                                   ['gray', 'C0', 'C3']):
                if mode == 'control' or plot_full:
                    index = abs(vs3[mode]['I']) > 0
                    Bmag = vs3[mode]['Fmag'][index, k]\
                        / (4 * abs(vs3[mode]['I'][index]))
                    ax[i].plot(1e3*vs3[mode]['t'][index], Bmag,
                               '-.', color=color)
                    ax[i].plot(1e3*vs3[mode]['t'][index],
                               vs3[mode]['Bmag'][index, k], '-', color=color)
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)
        path = os.path.join(class_dir(nep), '../Data/LTC/')
        with open(join(path, 'LTC_components'), 'rb') as intput:
            LTC = pickle.load(intput)
        for i, coil in enumerate(['upperVS', 'lowerVS']):
            Bmag = np.sqrt(LTC[coil]['Bx']**2 + LTC[coil]['Bz']**2)
            ax[i].plot(1e3*LTC[coil]['t'], Bmag,  # LTC[coil]['Bmag']
                       ls='-', color=0.3*np.ones(3))
        ax[0].set_ylabel('upper $|B_p|$ T')
        ax[1].set_ylabel('lower $|B_p|$ T')
        ax[1].set_xlabel('$t$ ms')
        h = []
        h.append(mlines.Line2D([], [], ls='-',
                               color=0.3*np.ones(3), label='LTC'))
        if plot_full:
            h.append(mlines.Line2D([], [], color='gray', label='referance'))
        h.append(mlines.Line2D([], [], color='C0', label=control_label))
        if plot_full:
            h.append(mlines.Line2D([], [], color='C3', label='error'))
        # ax.legend(handles=h, loc=1)

    def plot_field_components(self, var, vs3, plot_full):
        control_label = self.get_control_label(plot_full)
        ax = plt.subplots(2, 1, sharex=True, sharey=True)[1]
        for i, k in enumerate([1, 0]):
            for mode, color in zip(['referance', 'control', 'error'],
                                   ['gray', 'C0', 'C3']):
                if mode == 'control' or plot_full:
                    ax[i].plot(1e3*vs3[mode]['t'],
                               vs3[mode]['B{}'.format(var)][:, k],
                               color=color)
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)
        path = os.path.join(class_dir(nep), '../Data/LTC/')
        with open(join(path, 'LTC_components'), 'rb') as intput:
            LTC = pickle.load(intput)
        for i, coil in enumerate(['upperVS', 'lowerVS']):
            ax[i].plot(1e3*LTC[coil]['t'], LTC[coil]['B{}'.format(var)],
                       ls='-', color=0.3*np.ones(3))
        ax[0].set_ylabel(r'upper $B_{}$'.format(var) + ' T')
        ax[1].set_ylabel(r'lower $B_{}$'.format(var) + ' T')
        ax[1].set_xlabel('$t$ ms')
        h = []
        h.append(mlines.Line2D([], [], ls='-',
                               color=0.3*np.ones(3), label='LTC'))
        if plot_full:
            h.append(mlines.Line2D([], [], color='gray', label='referance'))
        h.append(mlines.Line2D([], [], color='C0', label=control_label))
        if plot_full:
            h.append(mlines.Line2D([], [], color='C3', label='error'))
        ax[0].legend(handles=h, loc=1)

    def plot_line_force_components(self, var, vs3, plot_full):
        control_label = self.get_control_label(plot_full)
        ax = plt.subplots(2, 1, sharex=True, sharey=True)[1]
        for i, k in enumerate([1, 0]):
            for mode, color in zip(['referance', 'control', 'error'],
                                   ['gray', 'C0', 'C3']):
                if mode == 'control' or plot_full:
                    ax[i].plot(1e3*vs3[mode]['t'],
                               1e-3*vs3[mode]['F{}'.format(var)][:, k],
                               color=color)
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)
        path = os.path.join(class_dir(nep), '../Data/LTC/')
        with open(join(path, 'LTC_components'), 'rb') as intput:
            LTC = pickle.load(intput)
        for i, coil in enumerate(['upperVS', 'lowerVS']):
            ax[i].plot(1e3*LTC[coil]['t'], 1e-3*LTC[coil]['F{}'.format(var)],
                       ls='-', color=0.3*np.ones(3))
        ax[0].set_ylabel(r'upper $F_{}$'.format(var) + ' kNm$^{-1}$')
        ax[1].set_ylabel(r'lower $F_{}$'.format(var) + ' kNm$^{-1}$')
        ax[1].set_xlabel('$t$ ms')
        h = []
        h.append(mlines.Line2D([], [], ls='-',
                               color=0.3*np.ones(3), label='LTC'))
        if plot_full:
            h.append(mlines.Line2D([], [], color='gray', label='referance'))
        h.append(mlines.Line2D([], [], color='C0', label=control_label))
        if plot_full:
            h.append(mlines.Line2D([], [], color='C3', label='error'))
        ax[0].legend(handles=h, loc=1)

    def get_control_label(self, plot_full):
        control_label = 'control' if plot_full else 'DINA'
        return control_label

    def plot_single(self, file, discharge, nframe=500, plot_full=False):
        vs3_data = self.read_data(discharge, nframe)  # load data
        vs3 = vs3_data[file]
        self.vs3_single = vs3

        self.plot_line_force(vs3, plot_full)
        self.plot_current(vs3, plot_full, file)
        self.plot_Bmag(vs3, plot_full)
        self.plot_field_components('x', vs3, plot_full)
        self.plot_field_components('z', vs3, plot_full)
        self.plot_line_force_components('x', vs3, plot_full)
        self.plot_line_force_components('z', vs3, plot_full)

    def plot(self, **kwargs):
        self.pf.plot(**kwargs)

    '''
    def load_psi(self, folder, plot=False, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        filepath = self.dina.locate_file('plasma', folder=folder)
        self.name = split(filepath)[-2]
        filepath = join(*split(filepath)[:-1], self.name, 'vs3_flux')
        if read_txt or not isfile(filepath + '.pk'):
            self.read_psi(folder, **kwargs)  # read txt file
            self.save_pickle(filepath, ['t', 'flux', 'Vbg', 'dVbg'])
        else:
            self.load_pickle(filepath)
        if plot:
            self.plot_profile()
        vs3_trip = self.pl.Ivs3_single(folder)[0]
        self.t_trip = vs3_trip['t_trip']
    '''


if __name__ == '__main__':

    vde = VDE_force(mode='control', discharge='IO', Iscale=1)

    #folder, frame_index = 3, 100
    #vde.load_file(folder, frame_index=frame_index, read_txt=True)
    vde.plot_frame()

    '''
    vde.plot(subcoil=True)

    ax = plt.gca()
    vvc = VVcoils()
    vvc.plot(ax=ax)

    #plt.figure(figsize=(8, 8))
    vde.plot_frame()
    '''

    # vde.plot_single('MD_UP_exp16', 'DINA', nframe=500)
    # vde.plot_single('VDE_UP_slow_fast', 'DINA', nframe=500)
    # vde.plot_Fmax('DINA', nframe=500)
    # vde.plot_Fmax('LTC', nframe=500)
    # vde.plot_Fmax('ENP', nframe=500)


    # vde.movie(3, nframe=60, mode='control')

    # vde.movie(8, nframe=60, mode='control')

    '''
    mode='control'
    vde.read_file(3, discharge='DINA')
    fig = plt.figure(figsize=(6, 10))

    vde.frame_update(30)
    vde.vs3_update(mode=mode)
    vde.force_update()
    vde.ff.plot(coils=['LVS'], scale=3, Fmax=10)

    vs = VSgeom()
    xo = vs.points[0]
    dx, dz = 0.2, 0.2
    n, limit = 1e3, [xo[0]-dx/2, xo[0]+dx/2, xo[1]-dz/2, xo[1]+dz/2]
    vde.x2d, vde.z2d, vde.x, vde.z = grid(n, limit)[:4]

    levels = vde.contour()
    '''




