import numpy as np
from nova.force import force_field
from nep.coil_geom import VSgeom
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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import nep
from amigo.IO import class_dir
from nep.DINA.read_eqdsk import read_eqdsk


class coil_force(pythonIO):

    def __init__(self, Ip_scale=1, read_txt=False, vessel=True, t_pulse=0.3,
                 mode='control', nturn=4):
        self.Ip_scale = Ip_scale
        self.read_txt = read_txt
        self.vessel = vessel
        self.t_pulse = t_pulse
        self.mode = mode
        self.nturn = nturn
        self.dina = dina('disruptions')
        self.pl = read_plasma('disruptions', Ip_scale=self.Ip_scale,
                              read_txt=read_txt)  # load plasma
        self.tor = read_tor('disruptions', Ip_scale=self.Ip_scale,
                            read_txt=read_txt)  # load currents
        self.ps = power_supply(nturn=nturn, Ip_scale=self.Ip_scale,
                               read_txt=read_txt)
        '''
        self.allowable = stress_allowable()  # load allowable interpolators
        pythonIO.__init__(self)  # python read/write
        '''

    def postscript(self):
        if self.vessel:
            postscript = '_vv_tp{:1.2f}s'.format(self.t_pulse)
            postscript = postscript.replace('.', '-')
        else:
            postscript = '_novv'
        if self.nturn != 4:
            postscript += '_{}turn'.format(self.nturn)
        if self.Ip_scale != 1:
            Ip_txt = '_{:1.3f}Ip_scale'.format(self.Ip_scale)
            Ip_txt = Ip_txt.replace('.', '-')
            postscript += Ip_txt
        return postscript

    def load_file(self, scenario, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        self.vessel = kwargs.get('vessel', self.vessel)
        self.t_pulse = kwargs.get('t_pulse', self.t_pulse)
        filepath = self.dina.locate_file('plasma', folder=scenario)
        self.name = split(split(filepath)[0])[-1]
        filepath = join(split(filepath)[0], 'coil_force')
        filepath += self.postscript()
        self.tor.load_file(scenario, read_txt=self.read_txt)
        self.t = self.tor.t
        self.pf = self.tor.pf  # link pf instance
        if read_txt or not isfile(filepath + '.pk'):
            self.read_file(scenario)  # read txt file
            self.save_pickle(filepath, ['Ivs3_fun', 'vs_geom', 'pf', 'ff',
                                        'xlim', 'zlim'])
        else:
            self.load_pickle(filepath)
            self.tor.pf = self.pf  # link pf instance
            self.ff.coilset = self.pf.coilset
        self.grid_sf(n=1e4, limit=[3.5, 10, -8.5, 8.5])

    def read_file(self, scenario):
        self.load_ps(scenario)  # load vs currents
        self.vs_geom = VSgeom()  # load vs geometory
        self.add_vv_coils()  # add local vessel, trs and VS3 coils
        self.set_force_field()  # initalise force_field object
        self.load_first_wall()

    def load_first_wall(self):
        eqdsk = read_eqdsk(file='burn').eqdsk
        self.xlim, self.zlim = eqdsk['xlim'], eqdsk['zlim']

    def add_vv_coils(self):
        coilset = self.ps.vv.pf.coilset
        VS3_coil = {name: coilset['coil'][name] for name in coilset['coil']
                    if 'VS' in name and 'jacket' not in name}
        self.pf.add_coils(VS3_coil, subcoil=coilset['subcoil'], label='VS3')
        if self.vessel:  # add vv and trs coils
            jacket_coil = {name: coilset['coil'][name]
                           for name in coilset['coil']
                           if 'jacket' in name}
            self.pf.add_coils(jacket_coil, subcoil=coilset['subcoil'],
                              label='jacket')
            vv_coil = {name: coilset['coil'][name] for name in coilset['coil']
                       if 'vv' in name}
            self.pf.add_coils(vv_coil, subcoil=coilset['subcoil'],
                              label='vv')
            trs_coil = {name: coilset['coil'][name] for name in coilset['coil']
                        if 'trs' in name}
            self.pf.add_coils(trs_coil, subcoil=coilset['subcoil'],
                              label='trs')
            # remove DINA coils
            vv_remove = [0, 1] + list(np.arange(18, 23)) + \
                list(np.arange(57, 60)) + list(np.arange(91, 96)) +\
                list(np.arange(72, 76)) + list(np.arange(114, 116))
            for vv_index in vv_remove:
                self.pf.remove_coil('vv_{}'.format(vv_index))

    def load_ps(self, scenario, **kwargs):
        if isinstance(scenario, str):
            scenario = self.dina.folders.index(scenario)
        if scenario != -1:
            self.pl.load_file(scenario)
            trip = self.pl.get_vs3_trip()  # get distuption direction
            zdir = trip['zdir']
            t_end = self.tor.t[-1]
        else:
            zdir = 1
            t_end = 100e-3
        self.Ivs3_fun = OrderedDict()  # Ivs3 interpolator
        for mode, sign in zip(['reference', 'control', 'error'], [0, 1, -1]):
            impulse = False if mode == 'reference' else True
            self.Ivs3_fun[mode] = self.ps.solve(
                    t_end, sign=-sign*zdir, scenario=scenario,
                    t_pulse=self.t_pulse, impulse=impulse, vessel=self.vessel,
                    **kwargs)

    def grid_sf(self, n=1e4, limit=[1.5, 10, -8.5, 8.5]):
        self.x2d, self.z2d, self.x, self.z = grid(n, limit)[:4]

    def set_force_field(self):
        active_coils, passive_coils = self.set_coil_type()
        self.ff = force_field(self.pf.coilset, multi_filament=True,
                              active_coils=active_coils,
                              passive_coils=passive_coils)

    def set_coil_type(self):  # set VS3 coils active
        active_coils = list(self.pf.coilset['index']['VS3']['name'])
        passive_coils = list(self.pf.coilset['coil'].keys())
        for coil in active_coils:
            passive_coils.remove(coil)
        passive_coils.append('Plasma')
        return active_coils, passive_coils

    def vs3_update(self, **kwargs):  # update vs3 coil and structure
        self.mode = kwargs.get('mode', self.mode)
        Ivs3 = self.Ivs3_fun[self.mode](self.t_index)  # current vector
        print('Ivs3', Ivs3[0])
        self.set_vs3_current(Ivs3[0])  # vs3 coil current
        coil_list = list(self.ps.vv.pf.coilset['coil'].keys())
        if self.vessel:  # set jacket, vv and trs currents
            Ic = {}  # coil jacket
            for i, coil in enumerate(coil_list[2:6]):
                Ic[coil] = Ivs3[1]  # lower VS jacket
            for i, coil in enumerate(coil_list[6:10]):
                Ic[coil] = Ivs3[2]  # upper VS jacket
            self.pf.update_current(Ic)  # dissable to remove jacket field
            Ic = {}  # vv and trs
            for i, coil in enumerate(coil_list[10:]):
                Ic[coil] = Ivs3[i+3]
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
        self.vs3_update()  # update vs3 coil currents
        # pass to ff object
        self.ff.coilset['plasma'] = self.tor.coilset['PF']['plasma']

    def contour(self, plot=True, ax=None, **kwargs):
        self.psi = cc.get_coil_psi(self.x2d, self.z2d,
                                   self.pf.coilset['subcoil'],
                                   self.pf.coilset['plasma'])
        self.sf = SF(eqdsk={'x': self.x, 'z': self.z, 'psi': self.psi,
                            'fw_limit': False,
                            'xlim': self.xlim, 'zlim': self.zlim})
        if plot:
            if ax is None:
                ax = plt.subplots(1, 1, figsize=(8, 10))[1]
            levels = self.sf.contour(boundary=True, Xnorm=True, ax=ax,
                                     **kwargs)
            self.sf.plot_firstwall()
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
        coil = self.pf.coilset['coil'][name]
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
        for mode in ['reference', 'control', 'error']:
            B_data[mode] = np.zeros(nframe, dtype=data_dtype)
        for i, frame_index in enumerate(frames):
            self.frame_update(frame_index)
            for mode in B_data:
                self.vs3_update(mode=mode)
                for j, name in enumerate(
                        self.pf.coilset['index']['VS3']['name']):
                    coil = self.pf.coilset['coil'][name]
                    point = [coil['x'], coil['z']]
                    B = cc.Bpoint(point, self.pf.coilset)
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
        for mode in ['reference', 'control', 'error']:
            coil_data[mode] = np.zeros(nframe, dtype=data_dtype)
        for i, frame_index in enumerate(frames):
            self.frame_update(frame_index)
            for mode in coil_data:
                self.vs3_update(mode=mode)
                self.force_update()
                for j, name in enumerate(
                        self.pf.coilset['index']['VS3']['name']):
                    # force
                    F_index = self.ff.active_coils.index(name)
                    coil = self.pf.coilset['coil'][name]
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
                    B = cc.Bpoint(centerpoint, self.pf.coilset)
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
                for i, name in enumerate(self.pf.coilset['index']['VS3']['name']):
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
            for i, name in enumerate(self.pf.coilset['index']['VS3']['name']):
                ax[i+1].set_ylabel(ylabel)
                ax[i+1].text(0.5, 1, name, transform=ax[i+1].transAxes,
                             ha='center', va='top',
                             bbox=dict(facecolor='gray', alpha=0.25))
                ax[i+1].legend()
            for i in range(2):
                plt.setp(ax[i].get_xticklabels(), visible=False)
            ax[2].set_xlabel('$t$, ms')
        return coil_data

    def movie(self, folder, nframe=None, mode='reference', discharge='DINA'):
        self.load_file(folder, discharge=discharge)
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
                self.plot_frame(frame_index=frame_index, mode=mode,
                                levels=levels)
                writer.grab_frame()
                tick.tock()

    def plot_frame(self, **kwargs):
        ax = kwargs.get('ax', None)
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(6, 9))[1]
        if 'frame_index' in kwargs:
            self.frame_update(kwargs['frame_index'])
        if 'mode' in kwargs:
            self.vs3_update(**kwargs)
        self.force_update()
        self.pf.plot(subcoil=True, plasma=True, ax=ax)
        self.ff.plot(coils=['VS3'], scale=3, Fmax=10)
        levels = self.contour(**kwargs)
        return levels

    def datafile(self, nframe):
        filename = join(class_dir(nep), 'DINA/Data/coil_force')
        filename += self.postscript()
        filename += '_{:d}.plk'.format(nframe)
        return filename

    def write_data(self, nframe=100):
        coil_data = OrderedDict()
        X = range(self.dina.nfolder)
        for i in X:
            self.load_file(i, read_txt=True)
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
            txt += ' {} frame:{}'.format(filename.split('/')[-1], nframe)
            print(txt)
            coil_data = self.write_data(nframe)
        else:
            with open(filename, 'rb') as input:
                coil_data = pickle.load(input)
        for folder in coil_data:  # rename reference
            newlist = []
            for key in coil_data[folder]:
                newkey = key.replace('referance', 'reference')
                newlist.append((newkey, coil_data[folder][key]))
            coil_data[folder] = OrderedDict(newlist)
        return coil_data

    def plot_Fmax(self, nframe=100):
        self.load_file(0)
        coil_data = self.read_data(nframe)  # load data
        ax = plt.subplots(2, 1, sharex=True, sharey=True)[1]
        X = range(self.dina.nfolder)
        Fmax = OrderedDict()
        Fmax_dtype = [('Fmag', '2float'), ('t', '2float'), ('I', '2float'),
                      ('frame_index', '2float')]
        Imax = OrderedDict()
        Imax_dtype = [('I', float), ('t', float), ('frame_index', float)]
        for mode in ['reference', 'control', 'error']:
            Fmax[mode] = np.zeros(len(X), dtype=Fmax_dtype)
            Imax[mode] = np.zeros(len(X), dtype=Imax_dtype)
        for i, name in enumerate(coil_data):
            vs3 = coil_data[name]
            for j, (mode, color, width) in enumerate(
                    zip(['reference', 'control', 'error'],
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
        for mode, color in zip(['reference', 'control', 'error'],
                               ['gray', 'C0', 'C3']):
            h.append(mpatches.Patch(color=color, label=mode))
        ax[0].legend(handles=h, loc=2)
        plt.xticks(X, self.dina.folders, rotation=70)
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)
        for i, k in enumerate([1, 0]):
            ax[i].set_ylabel('$|F|_{max}$ kNm$^{-1}$')
            name = self.pf.coilset['index']['VS3']['name'][k]
            ax[i].text(0.5, 1, name, transform=ax[i].transAxes,
                       weight='bold', bbox=dict(facecolor='gray', alpha=0.25),
                       va='top', ha='center')

        ax_I = plt.subplots(1, 1)[1]
        modes = ['reference', 'control', 'error']
        colors = ['gray', 'C0', 'C3']
        widths = [0.9, 0.7, 0.5]
        # modes = ['control']
        # colors = ['C0']
        # widths = [0.7]
        for i, name in enumerate(coil_data):
            vs3 = coil_data[name]
            for mode, color, width in zip(modes, colors, widths):
                ax_I.bar(i, 1e-3*Imax[mode][i]['I'], color=color,
                         width=width, label=mode)
        Im, im = [], []
        for mode in Imax:
            Im.extend(Imax[mode]['I'])
            im.extend(list(range(len(coil_data))))
        index = np.argmax(abs(np.array(Im)))
        va = 'bottom' if Im[index] > 0 else 'top'
        ax_I.text(im[index], 1e-3*Im[index],
                  '{:1.0f}'.format(1e-3*Im[index]), weight='bold',
                  va=va, ha='center')
        # plt.xticks(X, self.dina.folders, rotation=70)
        plt.xticks(X, '')
        for x, label in zip(X, self.dina.folders):
            label = label.replace('_', ' ')
            label = label.replace('MD', 'md')
            label = label.replace('VDE', 'vde')
            label = label.replace('UP', 'up')
            label = label.replace('DW', 'down')
            plt.text(x, 60, label + ' ', rotation=90, va='top', ha='center',
                     color='white')
        plt.despine()
        if len(modes) > 1:
            ax_I.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1),
                        ncol=4, bbox_transform=ax_I.transAxes)
        ax_I.set_ylabel('$I$ kA')

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
            name = self.pf.coilset['index']['VS3']['name'][k]
            ax[i].text(1, 0.5, name, transform=ax[i].transAxes,
                       bbox=dict(facecolor='lightgray', alpha=1),
                       va='center', ha='right')
            ax[i].set_xlim([0, 1400])
        ax[1].set_xlabel('$t$ ms')
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)
        h = []
        h.append(mlines.Line2D([], [], color='C0', label='MD down'))
        h.append(mlines.Line2D([], [], color='C1', label='MD up'))
        h.append(mlines.Line2D([], [], color='C2', label='VDE down'))
        h.append(mlines.Line2D([], [], color='C3', label='VDE up'))
        ax[0].legend(handles=h, loc=9, ncol=4)

        ax = plt.subplots(3, 1, sharex=True, sharey=True)[1]
        for index, file in enumerate(coil_data):
            vs3 = coil_data[file]
            color = self.pl.get_color(index)[0]
            for i, mode in enumerate(['reference', 'control', 'error']):
                ax[i].plot(1e3*vs3[mode]['t'],
                           1e-3*vs3[mode]['I'], color=color)
        for i, (mode, fc) in enumerate(zip(['reference', 'control', 'error'],
                                           ['gray', 'C0', 'C3'])):
            ax[i].set_ylabel('$I$ kA')
            ax[i].set_xlim([0, 1400])
            ax[i].text(1, 0.9, mode, transform=ax[i].transAxes,
                       color='lightgray',
                       bbox=dict(facecolor=fc, alpha=1),
                       va='top', ha='right')
        ax[-1].set_xlabel('$t$ ms')
        ax[0].legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1),
                     ncol=4, bbox_transform=ax[0].transAxes)
        plt.despine()
        plt.detick(ax)

        mode_index = np.nanargmax(
                [max(abs(Imax[mode]['I'][:])) for mode in Imax])
        mode = list(Imax.keys())[mode_index]
        file_index = np.nanargmax(abs(Imax[mode]['I']))
        file = list(coil_data.keys())[file_index]
        vs3 = coil_data[file]
        Imax_index = np.nanargmax(abs(vs3[mode]['I']))
        Fmax_index = np.nanargmax(abs(vs3[mode]['I']))
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
            name = self.pf.coilset['index']['VS3']['name'][k]
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
            name = self.pf.coilset['index']['VS3']['name'][k]
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

    def plot(self, insert=False, contour=False, **kwargs):
        if 'ax' not in kwargs:
            kwargs['ax'] = plt.subplots(1, 1, figsize=(7, 10))[1]
        if not self.vessel:
            for coil in ['lower', 'upper']:
                self.pf.coilset['coil'].pop('{}VS'.format(coil))
                name = '{}VS_0'.format(coil)
                for i in range(self.pf.coilset['subcoil'][name]['Nf']):
                    self.pf.coilset['subcoil'].pop('{}VS_{}'.format(coil, i))
        self.pf.plot(**kwargs)
        if contour:
            self.contour(**kwargs)
        if insert:
            ax_main = kwargs['ax']  # Inset image
            vs_geom = VSgeom()
            dx, dz, zoom = 1.75, 1.25, 3
            for coil, loc in zip(vs_geom.geom, [(4, 3, 1), (1, 2, 4)]):
                xo, zo = vs_geom.geom[coil]['x'], vs_geom.geom[coil]['z']
                ax_ins = zoomed_inset_axes(ax_main, zoom, loc=loc[0])
                mark_inset(ax_main, ax_ins, loc1=loc[1], loc2=loc[2], lw=1)
                kwargs['ax'] = ax_ins
                self.pf.plot(**kwargs)
                if contour:
                    self.contour(**kwargs)
                ax_ins.set_xlim(xo + np.array([-dx/2, dx/2]))
                ax_ins.set_ylim(zo + np.array([-dz/2, dz/2]))
                ax_ins.axis('on')
                ax_ins.set_xticks([])
                ax_ins.set_yticks([])


if __name__ == '__main__':

    force = coil_force(vessel=True, t_pulse=0.3, nturn=4, Ip_scale=15/15,
                       read_txt=False)
    '''
    for folder in force.dina.folders:
        print(folder)
        force.load_file(folder, read_txt=True)
    '''

    '''
    force.frame_update(251)
    force.plot(subcoil=True, plasma=False, insert=True, contour=True)
    '''

    '''
    plt.figure(figsize=(7, 10))
    force.pf.initalize_collection()
    force.pf.patch_coil(force.pf.coil)
    force.pf.patch_coil(force.pf.plasma)
    force.pf.plot_patch(c='Jc', clim=[-10, 10])
    plt.axis('equal')
    plt.axis('off')
    '''

    #force.load_file(0)

    # force.read_data(nframe=500, forcewrite=True)
    # plt.set_context('notebook')
    force.plot_Fmax(nframe=500)







