import numpy as np
from nova.force import force_field
from nep.coil_geom import VSgeom, VVcoils
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
from amigo.IO import pythonIO
from nep.DINA.capacitor_discharge import power_supply


class coil_force(pythonIO):

    def __init__(self, Iscale=1, read_txt=False, vessel=True, t_pulse=0.3):
        self.Iscale = Iscale
        self.read_txt = read_txt
        self.vessel = vessel
        self.t_pulse = t_pulse
        self.dina = dina('disruptions')
        self.pl = read_plasma('disruptions', Iscale=self.Iscale,
                              read_txt=read_txt)  # load plasma
        self.tor = read_tor('disruptions', Iscale=self.Iscale,
                            read_txt=read_txt)  # load currents
        self.ps = power_supply(nturn=4, read_txt=read_txt)
        self.allowable = stress_allowable()  # load allowable interpolators
        pythonIO.__init__(self)  # python read/write

    def postscript(self):
        if self.vessel:
            postscript = '_vv{:1.2f}'.format(self.t_pulse)
        else:
            postscript = '_novv'
        return postscript

    def load_file(self, scenario, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        self.vessel = kwargs.get('vessel', self.vessel)
        self.t_pulse = kwargs.get('t_pulse', self.t_pulse)
        filepath = self.dina.locate_file('plasma', folder=scenario)
        self.name = split(filepath)[-2]
        filepath = join(*split(filepath)[:-1], self.name, 'coil_force')
        filepath += self.postscript()
        if read_txt or not isfile(filepath + '.pk'):
            self.read_file(scenario)  # read txt file
            self.save_pickle(filepath, ['pf', 'ff', 'vv',
                                        'Ivs3_fun', 'vs_geom'])
        else:
            self.load_pickle(filepath)
        self.tor.load_file(scenario)  # read toroidal strucutres
        self.tor.pf = self.pf  # re-link pf object
        self.ff.index = self.pf.index  # re-link index
        self.ff.pf_coil = self.pf.coil  # re-link pf coil
        self.ff.eq_coil = self.pf.sub_coil  # re-link eq coil
        self.ff.eq_plasma_coi = self.pf.plasma_coil  # relink plasma
        self.initalize_sf()

    def read_file(self, scenario):
        self.tor.load_file(scenario)  # load toroidal scenario
        self.pf = self.tor.pf  # referance to tor pf
        self.t = self.tor.t
        self.vs_geom = VSgeom()  # load vs geometory
        self.vv = self.ps.vv  # link vv coil set
        VS3_coil = {coil: self.vv.pf.coil[coil] for coil in self.vv.pf.coil
                    if 'VS' in coil}
        self.pf.add_coils(VS3_coil, sub_coil=self.vv.pf.sub_coil, label='VS3')
        if self.vessel:  # add vv and trs coils
            vv_coil = {coil: self.vv.pf.coil[coil] for coil in self.vv.pf.coil
                       if 'vv' in coil}
            self.pf.add_coils(vv_coil, sub_coil=self.vv.pf.sub_coil,
                              label='vv')
            trs_coil = {coil: self.vv.pf.coil[coil] for coil in self.vv.pf.coil
                        if 'trs' in coil}
            self.pf.add_coils(trs_coil, sub_coil=self.vv.pf.sub_coil,
                              label='trs')
            vv_remove = [0, 1] + list(np.arange(18, 23)) + \
                list(np.arange(57, 60)) + list(np.arange(91, 96)) +\
                list(np.arange(72, 76)) + list(np.arange(114, 116))
            for vv_index in vv_remove:
                self.pf.remove_coil('vv_{}'.format(vv_index))
        self.set_force_field()  # initalise force_field object
        self.load_ps(scenario)  # load vs currents

    def load_ps(self, scenario):
        self.pl.load_file(scenario)
        trip = self.pl.get_vs3_trip()  # get distuption direction
        self.Ivs3_fun = OrderedDict()  # Ivs3 interpolator
        for mode, sign in zip(['referance', 'control', 'error'], [0, 1, -1]):
            impulse = False if mode == 'referance' else True
            self.Ivs3_fun[mode] = self.ps.solve(
                    self.tor.t[-1], sign=-sign*trip['zdir'],
                    scenario=scenario, t_pulse=self.t_pulse, impulse=impulse,
                    vessel=self.vessel)

    # def load_vs3(self, scenario):  # load DINA current interpolator
    #    self.Ivs3_fun = self.pl.Ivs3_single(folder, discharge='DINA')[-1]

    def initalize_sf(self):
        n, limit = 1e4, [1.5, 10, -8.5, 8.5]
        self.x2d, self.z2d, self.x, self.z = grid(n, limit)[:4]
        self.psi = cc.get_coil_psi(self.x2d, self.z2d, self.pf)
        self.sf = SF(eqdsk={'x': self.x, 'z': self.z, 'psi': self.psi,
                            'name': 'DINA_{}'.format(self.tor.name)})

    def set_force_field(self):
        active_coils, passive_coils = self.set_coil_type()
        self.ff = force_field(self.pf.index, self.pf.coil, self.pf.sub_coil,
                              self.pf.plasma_coil, multi_filament=True,
                              active_coils=active_coils,
                              passive_coils=passive_coils)

    def set_coil_type(self):  # set VS3 coils active
        active_coils = list(self.pf.index['VS3']['name'])  # calculate force
        passive_coils = list(self.pf.coil.keys())
        for coil in active_coils:
            passive_coils.remove(coil)
        passive_coils.append('Plasma')
        return active_coils, passive_coils

    def vs3_update(self, mode='control'):  # update vs3 coil and structure
        Ivs3 = self.Ivs3_fun[mode](self.t_index)  # current vector
        self.set_vs3_current(Ivs3[0])  # vs3 coil current
        if self.vessel:
            Ic = {}
            for i, coil in enumerate(list(self.vv.pf.coil.keys())[2:]):
                Ic[coil] = Ivs3[i+1]
            self.pf.update_current(Ic)  # dissable to remove vv field

    def set_vs3_current(self, Ivs3):
        self.Ivs3 = float(Ivs3)  # store Ivs3 current
        Ic = {'upperVS': -4*Ivs3, 'lowerVS': 4*Ivs3}
        self.pf.update_current(Ic)

    def force_update(self):
        self.ff.set_passive_force_field()  # update plasma force field
        self.ff.set_current()  # update fillament currents
        self.ff.set_force(self.ff.Ic)  # update force calculation

    def frame_update(self, frame_index):
        self.frame_index = frame_index
        self.t_index = self.tor.t[self.frame_index]
        self.tor.set_current(frame_index)  # update coil currents and plasma

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
        dx, dy = self.vs_geom.rail[name]['dx'], self.vs_geom.rail[name]['dy']
        dz, n = self.vs_geom.rail[name]['dz'], self.vs_geom.rail[name]['n']
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
        coil_data = OrderedDict()
        data_dtype = [('Fx', '2float'), ('Fz', '2float'), ('Fmag', '2float'),
                      ('Fn', '2float'), ('Ft', '2float'), ('sigma', '2float'),
                      ('t', float), ('I', float),
                      ('frame_index', int),
                      ('Bx', '2float'), ('Bz', '2float'), ('Bmag', '2float')]
        frames, nframe = self.get_frames(nframe)
        scenario = self.dina.folders[self.ps.scenario]
        tick = clock(nframe, header='loading scenario: {}'.format(scenario))
        for mode in ['referance', 'control', 'error']:
            coil_data[mode] = np.zeros(nframe, dtype=data_dtype)
        for i, frame_index in enumerate(frames):
            self.frame_update(frame_index)
            for mode in coil_data:
                self.vs3_update(mode=mode)
                self.force_update()
                for j, name in enumerate(self.pf.index['VS3']['name']):
                    # force
                    F_index = self.ff.active_coils.index(name)
                    coil = self.pf.coil[name]
                    F = 1e6 * self.ff.F[F_index] / (2*np.pi*coil['x'])  # N/m
                    Fxyz = np.array([F[0], 0, F[1]])
                    theta = self.vs_geom.theta_coil[name]  # theta_coil
                    Ftn = qrotate(Fxyz, theta=theta, dx=[0, 1, 0])[0]
                    Fmag = np.linalg.norm(F)
                    sigma = self.get_stress(Ftn[-1], Ftn[0], name)
                    coil_data[mode][i]['frame_index'] = self.frame_index
                    coil_data[mode][i]['t'] = self.t_index
                    coil_data[mode][i]['I'] = self.Ivs3
                    coil_data[mode][i]['Fx'][j] = F[0]
                    coil_data[mode][i]['Fz'][j] = F[1]
                    coil_data[mode][i]['Fn'][j] = Ftn[-1]
                    coil_data[mode][i]['Ft'][j] = Ftn[0]
                    coil_data[mode][i]['Fmag'][j] = Fmag
                    coil_data[mode][i]['sigma'][j] = sigma
                    # centerpoint field
                    centerpoint = [coil['x'], coil['z']]
                    B = cc.Bpoint(centerpoint, self.pf)
                    coil_data[mode][i]['Bx'][j] = B[0]
                    coil_data[mode][i]['Bz'][j] = B[1]
                    coil_data[mode][i]['Bmag'][j] = np.linalg.norm(B)
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
            for mode, color in zip(coil_data, ['gray', 'C0', 'C3']):
                ax[0].plot(1e3*coil_data[mode]['t'],
                           1e-3*coil_data[mode]['I'], '-',
                           color=color, label=mode)
                for i, name in enumerate(self.pf.index['VS3']['name']):
                    ax[i+1].plot(1e3*coil_data[mode]['t'],
                                 factor*coil_data[mode][pvar][:, i],
                                 color=color)
                    max_index = np.nanargmax(coil_data[mode][pvar][:, i])
                    txt = '{}: {:1.1f}'.format(
                        mode, factor*coil_data[mode][pvar][max_index, i])
                    ax[i+1].plot(
                            1e3*coil_data[mode]['t'][max_index],
                            factor*coil_data[mode][pvar][max_index, i],
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
        return coil_data

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

    def datafile(self, nframe):
        filename = join(self.dina.root, 'DINA/Data/coil_force')
        filename += self.postscript()
        filename += '_{:d}.plk'.format(nframe)
        if self.Iscale != 1:
            filename = filename.replace(
                    '.plk', '_Iscale_{:1.1f}.plk'.format(self.Iscale))
        return filename

    def write_data(self, nframe=100):
        coil_data = OrderedDict()
        X = range(self.dina.nfolder)
        for i in X:
            self.read_file(i)
            name = self.tor.name
            coil_data[name] = self.get_data(nframe, plot=False)
        filename = self.datafile(nframe)
        with open(filename, 'wb') as output:
            pickle.dump(coil_data, output, -1)
        return coil_data

    def read_data(self, nframe=100, forcewrite=False):
        filename = self.datafile(nframe)
        if not isfile(filename) or forcewrite:
            txt = '\nre-generating data:'
            txt += 'nframe:{}'.format(nframe)
            print(txt)
            coil_data = self.write_data(nframe)
        else:
            with open(filename, 'rb') as input:
                coil_data = pickle.load(input)
        return coil_data

    def plot_Fmax(self, nframe=100):
        coil_data = self.read_data(nframe)  # load data
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
        for i, name in enumerate(coil_data):
            vs3 = coil_data[name]
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
                im.extend(list(range(len(coil_data))))
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
        for index, file in enumerate(coil_data):
            vs3 = coil_data[file]
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
        file = list(coil_data.keys())[file_index]
        vs3 = coil_data[file]
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
                file = list(coil_data.keys())[file_index]
                vs3 = coil_data[file]
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

    def plot(self, **kwargs):
        self.pf.plot(**kwargs)



if __name__ == '__main__':

    force = coil_force(t_pulse=0, vessel=True, read_txt=True)

    #force.load_file(3, read_txt=True)

    force.read_data(nframe=2)

    #force.plot_Fmax(nframe=100)
    # new frame_update
    #force.tor.set_current(300)  # update coil currents and plasma

    #force.pf.plot(subcoil=True, plasma=True)
    #force.contour()





    '''
    self.load_IO(folder)  # load hig-res vv model
    #*************
    # self.load_vs3(folder, discharge=self.discharge)  # load vs3 currents
    #*************

    self.frame_update(frame_index)  # initalize at start of timeseries

    # self.vs3_update(mode=self.mode)  # initalize vs3 current
    # self.force_update()  # update vs3 coil forces

    '''




