import numpy as np
from nova.force import force_field
from nep.coil_geom import PFgeom, VSgeom
from nep.DINA.read_tor import read_tor
from collections import OrderedDict
import nova.cross_coil as cc
from nova.streamfunction import SF
from amigo.geom import grid
from nep.DINA.read_psi import read_psi
from nep.DINA.read_plasma import read_plasma
from amigo.pyplot import plt
from amigo.time import clock
from amigo.geom import qrotate
import matplotlib.animation as manimation


class VDE_force:

    def __init__(self, folder=0, frame_index=0):
        self.load_vs3(folder)  # load vs3 currents
        self.load_active()  # load active coils
        self.load_passive(folder)  # load toroidal strucutres
        self.ff = force_field(self.pf.index, self.pf.coil, self.pf.sub_coil,
                              self.pf.plasma_coil, multi_filament=True)
        self.frame_update(frame_index)  # initalize at start of timeseries

    def load_vs3(self, folder):
        self.pl = read_plasma('disruptions')  # load plasma
        self.Ivs3_fun = self.pl.Ivs3_single(folder)[-1]  # current interpolator

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

    def frame_update(self, frame_index):  # updates both coil and sub_coil
        self.frame_index = frame_index
        self.t = self.tor.t[self.frame_index]
        self.set_coil_current(frame_index)
        self.set_filament_current(self.tor.vessel_coil, frame_index)
        self.set_filament_current(self.tor.blanket_coil, frame_index)
        self.load_plasma(frame_index)
        self.set_vs3_current(self.Ivs3_fun['control'](self.t))  # default

    def force_update(self, mode):
        self.set_vs3_current(self.Ivs3_fun[mode](self.t))  # VS3 current
        self.ff.set_current()  # update fillament currents
        self.ff.set_force(self.ff.Ic)  # update force calculation

    def set_vs3_current(self, Ivs3):
        self.Ivs3 = Ivs3  # store
        Ic = {'upperVS': -4*Ivs3, 'lowerVS': 4*Ivs3}
        self.pf.update_current(Ic)

    def load_active(self, dCoil=0.25):
        vs_geom = VSgeom()
        pf_geom = PFgeom(VS=True)
        self.pf = pf_geom.pf
        self.pf.mesh_coils(dCoil=dCoil)
        for i, coil in enumerate(vs_geom.pf.coil):
            if i < 4:  # lower
                subcoil = 'lowerVS_{:d}'.format(i)
            else:
                subcoil = 'upperVS_{:d}'.format(i-4)
            self.pf.sub_coil[subcoil] = vs_geom.pf.coil[coil]
            self.pf.sub_coil[subcoil]['Nf'] = 4
        self.vs_theta = {}
        for name in vs_geom.geom:
            self.vs_theta[name+'VS'] = vs_geom.geom[name]['theta']

    def load_passive(self, folder):
        self.tor = read_tor('disruptions')
        self.tor.read_file(folder)  # read toroidal strucutres
        self.add_filament(self.tor.vessel_coil)
        self.add_filament(self.tor.blanket_coil)

    def add_filament(self, filament):
        for coil in filament:
            self.pf.coil[coil] = filament[coil]
            self.pf.sub_coil[coil+'_0'] = filament[coil]
            self.pf.sub_coil[coil+'_0']['Nf'] = 1

    def load_plasma(self, index):
        self.pf.plasma_coil = OrderedDict()  # clear
        for coil in self.tor.plasma_coil[index]:
            self.pf.plasma_coil[coil] = self.tor.plasma_coil[index][coil]
        self.ff.set_passive_force_field()  # update plasma force field

    def contour(self, **kwargs):
        n, limit = 1e4, [1.5, 10, -8.5, 8.5]
        self.x2d, self.z2d, self.x, self.z = grid(n, limit)[:4]
        self.psi = cc.get_coil_psi(self.x2d, self.z2d, self.pf)
        self.sf = SF(eqdsk={'x': self.x, 'z': self.z, 'psi': self.psi,
                            'name': 'DINA_{}'.format(self.tor.name)})
        levels = self.sf.contour(51, boundary=False, **kwargs)
        return levels

    def get_frames(self, nframe):
        if nframe is None:
            nframe = self.tor.nt
        nframe = int(nframe)
        frames = np.linspace(0, self.tor.nt-2, nframe, dtype=int)
        return frames, nframe

    def get_force(self, nframe=None, plot=False):
        self.Fvs3_data = OrderedDict()
        dtype = [('Fx', '2float'), ('Fz', '2float'), ('Fmag', '2float'),
                 ('Fn', '2float'), ('Ft', '2float'),
                 ('t', float), ('I', float)]
        frames, nframe = self.get_frames(nframe)
        tick = clock(nframe)
        for mode in ['referance', 'control', 'error']:
            self.Fvs3_data[mode] = np.zeros(nframe, dtype=dtype)
        for fi, frame_index in enumerate(frames):
            self.frame_update(frame_index)
            for mode in self.Fvs3_data:
                self.force_update(mode)
                for i, (index, name) in \
                        enumerate(zip(self.pf.index['VS3']['index'],
                                      self.pf.index['VS3']['name'])):
                    coil = self.pf.coil[name]
                    F = self.ff.F[index] / (2*np.pi*coil['x'])  # per length
                    Fxyz = np.array([F[0], 0, F[1]])
                    Ftn = qrotate(Fxyz, theta=-self.vs_theta[name],
                                  dx=[0, 1, 0])[0]
                    Fmag = np.linalg.norm(F)
                    self.Fvs3_data[mode][fi]['t'] = self.t
                    self.Fvs3_data[mode][fi]['I'] = self.Ivs3
                    self.Fvs3_data[mode][fi]['Fx'][i] = F[0]
                    self.Fvs3_data[mode][fi]['Fz'][i] = F[1]
                    self.Fvs3_data[mode][fi]['Fn'][i] = Ftn[-1]
                    self.Fvs3_data[mode][fi]['Ft'][i] = Ftn[0]
                    self.Fvs3_data[mode][fi]['Fmag'][i] = Fmag

            tick.tock()

        if plot:
            ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))[1]
            ax[0].text(0.5, 1, self.tor.name, transform=ax[0].transAxes,
                       ha='center', va='top', weight='bold')
            for mode, color in zip(self.Fvs3_data, ['gray', 'C0', 'C3']):
                ax[0].plot(1e3*self.Fvs3_data[mode]['t'],
                           1e-3*self.Fvs3_data[mode]['I'], '-',
                           color=color, label=mode)
                for i, name in enumerate(self.pf.index['VS3']['name']):
                    ax[i+1].plot(1e3*self.Fvs3_data[mode]['t'],
                                 1e3*self.Fvs3_data[mode]['Fmag'][:, i],
                                 color=color)
                    max_index = np.nanargmax(
                            self.Fvs3_data[mode]['Fmag'][:, i])
                    txt = '{}: {:1.1f}'.format(
                        mode, 1e3*self.Fvs3_data[mode]['Fmag'][max_index, i])
                    ax[i+1].plot(
                            1e3*self.Fvs3_data[mode]['t'][max_index],
                            1e3*self.Fvs3_data[mode]['Fmag'][max_index, i],
                            'o', color=color, label=txt)

            plt.despine()
            ax[0].set_ylabel('$I_{vs3}$, kA')
            ax[0].legend()
            for i, name in enumerate(self.pf.index['VS3']['name']):
                ax[i+1].set_ylabel('$|F|$, kNm$^{-1}$')
                ax[i+1].text(0.5, 1, name, transform=ax[i+1].transAxes,
                             ha='center', va='top',
                             bbox=dict(facecolor='gray', alpha=0.25))
                ax[i+1].legend()
            for i in range(2):
                plt.setp(ax[i].get_xticklabels(), visible=False)
            ax[2].set_xlabel('$t$, ms')

    def movie(self, nframe=None, mode='referance'):
        frames, nframe = self.get_frames(nframe)
        #self.frame_update(0)  # get referance contour
        #levels = self.contour()

        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=15, bitrate=-1)
        fig = plt.figure(figsize=(6, 10))
        tick = clock(nframe)

        filename = '../Movies/{}_{}.mp4'.format(self.tor.name, mode)
        with writer.saving(fig, filename, 72):
            for frame_index in frames:
                plt.clf()
                self.frame_update(frame_index)
                self.force_update(mode)
                self.pf.plot(subcoil=True, plasma=True)
                self.ff.plot(coils=['VS3'], scale=2e2)
                self.contour()
                writer.grab_frame()
                tick.tock()


if __name__ == '__main__':
    folder, frame_index = 11, 300

    vde = VDE_force(folder=folder, frame_index=frame_index)

    #vde.get_force(nframe=51, plot=True)

    vde.movie(nframe=50)



    '''
    psi = read_psi('disruptions')
    psi.read_file(folder)
    psi.plot_fw()
    psi.plot(frame_index, levels=levels, color='C1')
    '''

