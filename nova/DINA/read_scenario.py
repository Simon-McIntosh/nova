from nep.DINA.read_dina import dina, lowpass
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


class read_scenario:

    def __init__(self, database_folder='operations'):
        self.dina = dina(database_folder)
        self.load_coils()
        # self.load_plasma()

    def read_file(self, folder, file_type='txt'):
        if file_type == 'txt':
            filename = self.read_txt(folder)
        elif file_type == 'qda':
            filename = self.read_qda(folder)
        self.name = filename.split('\\')[-3]
        if 'time' in self.data:  # rename time field
            self.data['t'] = self.data['time']
            self.data.pop('time')
        self.trim_nan()
        self.space_data()
        self.load_data()
        self.set_noise()  # level of dz/dt diagnostic noise
        self.opperate()  # identify operating modes

    def set_noise(self):
        noise_dict = {'2014-01': 3, '2015-02': 0.6,
                      '2016-01': 0.6, '2017-05': 0.6,
                      '2015-05': 0.6}
        key = self.name.split('DINA')[-1]
        try:
            noise = noise_dict[key]
        except KeyError:
            errtxt = 'simulation {}'.format(self.name)
            errtxt += ' not present in noise_dict'
            raise KeyError(errtxt)
        self.noise = noise

    def read_txt(self, folder, dropnan=True):
        filename = self.dina.locate_file('data2.txt', folder=folder)
        self.data, self.columns = \
            self.dina.read_csv(filename, dropnan=False, split=',')
        return filename

    def read_qda(self, folder):
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

    def load_coils(self, plot=False):
        pf_geom = PFgeom(VS=True)
        self.pf = pf_geom.pf
        if plot:
            self.pf.plot(label=True)

    def load_data(self):
        self.Icoil = {}
        for var in self.data:
            if var == 'Ip':  # plasma
                self.Ip = 1e6*self.data[var]  # MA to A (keep incorrect sign)
            elif var[0] == 'I' and len(var) <= 5:
                # kAturn to -Aturn (change sign)
                self.Icoil[var[1:].upper()] = -1e3*self.data[var]
        self.t = self.data['t']
        self.dt = np.mean(np.diff(self.t))

    def space_data(self):  # generate interpolators and space timeseries
        to = np.copy(self.data['t'])
        dt = np.mean(np.diff(to))
        self.tmax = np.nanmax(to)
        self.nt = int(self.tmax/dt)
        self.dt = self.tmax/(self.nt-1)
        self.t = np.linspace(0, self.tmax, self.nt)
        self.fun = {}
        for var in self.data:  # interpolate
            self.fun[var] = interp1d(to, self.data[var])
            self.data[var] = self.fun[var](self.t)

    def opperate(self, plot=True):  # identify operating modes
        Ip_lp = lowpass(self.Ip, self.dt, dt_window=1.0)  # plasma current
        dIpdt = np.gradient(Ip_lp, self.t)  # calculate gradient
        hip = histopeaks(dIpdt, nstd=3, nlim=6)  # identify operating modes
        self.opp_index = hip.timeseries(self.t, Ip=self.Ip, plot=plot)
        self.flattop_index = self.opp_index[0]  # flattop index

    def get_coil_current(self, ind, VS3=True):  # return dict of coil currents
        Ic = {}
        for coil in self.Icoil:
            if VS3 or 'VS' not in coil:
                Ic[coil] = self.Icoil[coil][ind]
        for coil in ['CS1L', 'CS1U']:  # split
            Ic[coil] = Ic['CS1']
        Ic.pop('CS1')  # remove
        for coil in Ic:  # A to A.turn
            Ic[coil] *= self.pf.coil[coil]['N']
        return Ic

    def set_current(self, ind):
        self.sf.cpasma = -self.Ip[ind]  # set plasma current
        self.eq.get_plasma_coil()  # update plasma coils
        Ic = self.get_coil_current(ind, VS3=False)  # get coil currents
        self.pf.update_current(Ic)

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

    def load_plasma(self, file='burn', plot=False):
        self.eqdsk = read_eqdsk(file=file)  # 'burn', 'inductive'
        self.sf = self.eqdsk.sf
        self.eq = EQ(self.sf, self.pf, n=1e3)  # set plasma coils
        if plot:
            self.plot_plasma()

    def plot_plasma(self, ind=None):
        if ind is not None:  # update currents
            self.set_current(ind)

        plt.figure()
        self.eq.run(update=False)
        self.sf.contour()
        self.pf.plot(plasma=True, label=True, current=True)
        self.pf.plot(subcoil=True, label=False)

    def plot_currents(self):
        index = self.opp_index[0]  # flattop index
        plt.figure()
        for coil in self.Icoil:
            if 'CS' in coil:
                color = 'C3'
                zorder = 10
            elif 'PF' in coil:
                color = 'C0'
                zorder = 5
            elif coil == 'VS3':
                color = 'C9'
                zorder = 2
            plt.plot(self.t, 1e-3*self.Icoil[coil],
                     color=color, zorder=zorder, alpha=0.15)
            plt.plot(self.t[index[0]:index[1]],
                     1e-3*self.Icoil[coil][index[0]:index[1]],
                     color=color, zorder=zorder)
        plt.despine()
        plt.xlabel('$t$ s')
        plt.ylabel('$I$ kA.turn')
        VS3_rms = self.get_VS3_rms()
        txt = '$\sqrt{<dz/dt^2>}=$'
        txt += '{:1.1f}'.format(self.noise)
        txt += 'ms$^{-1}$\n'
        txt += '$\sqrt{<I_{VS3}^2>}=$'+'{:1.2f}kA'.format(1e-3*VS3_rms)
        plt.text(1, 1, txt, transform=plt.gca().transAxes,
                 ha='right', va='bottom',
                 bbox=dict(facecolor='grey', alpha=0.25),
                 fontsize=12)
        plt.text(-0.05, 1.1, self.name, transform=plt.gca().transAxes,
                 ha='left', va='top',
                 bbox=dict(facecolor='grey', alpha=0.25),
                 fontsize=12)

        h_PF = mlines.Line2D([], [], color='C0', label='PF')
        h_CS = mlines.Line2D([], [], color='C3', label='CS')
        h_VS3 = mlines.Line2D([], [], color='C9', label='VS3')
        plt.legend(handles=[h_PF, h_CS, h_VS3], loc=4)

    def get_VS3_rms(self):
        index = self.opp_index[0]  # flattop
        Ivs3 = self.Icoil['VS3'][index[0]:index[1]]  # line current
        VS3_rms = np.std(Ivs3)
        t = self.t[index[0]:index[1]]
        VS3_rms = np.sqrt(np.trapz(Ivs3**2, t)/(t[-1]-t[0]))
        return VS3_rms

    def get_noise(self):
        fs = 1/np.mean(np.diff(scn.t))  # requires equal spaced data
        f, Pxx = periodogram(scn.data['dZ/dt'], fs,
                             'flattop', scaling='spectrum')
        plt.figure()
        plt.semilogy(f, np.sqrt(Pxx))


if __name__ is '__main__':

    scn = read_scenario()
    scn.read_file(folder='15MA DT-DINA2015-05')

    scn.load_plasma()
    scn.load_VS3(n=100, plot=True)

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
