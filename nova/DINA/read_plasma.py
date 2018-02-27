import pandas as pd
from amigo.pyplot import plt
import numpy as np
from amigo.addtext import linelabel
from read_dina import dina, lowpass
from scipy.interpolate import interp1d
from read_dina import timeconstant
import matplotlib.lines as mlines
from scipy.integrate import odeint


class read_plasma:

    def __init__(self, database_folder='disruptions'):
        self.dina = dina(database_folder)

    def read_file(self, folder, dropnan=True):
        filename = self.dina.locate_file('plasma', folder=folder)
        self.name = filename.split('\\')[-2]
        self.data = pd.read_csv(filename, delim_whitespace=True, skiprows=40,
                                na_values='NAN')
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
        data['Ivs_o'] = -1e3*self.data.loc[:, 'I_dw'] / 4  # -kAturn to A

        data['x'] = 1e-2*self.data['Rcur']  # cm - m
        data['z'] = 1e-2*self.data['Zcur']  # cm - m
        dt_min = np.nanmin(np.diff(data['t']))
        t_max = np.nanmax(data['t'])
        self.nt = int(t_max / dt_min)
        self.t = np.linspace(0, t_max, self.nt)  # equal spaced time stencil
        self.dt = t_max/(self.nt-1)
        for var in data:
            if var != 't':  # interpolate
                setattr(self, var, interp1d(data['t'], data[var])(self.t))

    def get_quench(self, plot=False):  # locate current quench
        Ip_lp = lowpass(self.Ip, self.dt, dt_window=0.001)  # plasma current
        dIpdt = np.gradient(Ip_lp, self.t)
        try:
            i_cq = next((i for i, (dIdt, Ip) in enumerate(zip(dIpdt, Ip_lp))
                         if dIdt > 1e8 and Ip > 0.95*Ip_lp[0]))
        except StopIteration:
            i_cq = next((i for i, dIdt in enumerate(dIpdt) if dIdt < -1e8))

        t_cq = self.t[i_cq]  # current quench time
        tc = timeconstant(self.t[i_cq:], Ip_lp[i_cq:], trim_fraction=0.2)
        tdis, ttype, tfit, Ifit = tc.fit(plot=False)  # plasma discharge

        if plot:
            txt = '{} discharge, t={:1.0f}ms'.format(ttype, 1e3*tdis)
            ax = plt.subplots(2, 1, sharex=True)[1]
            ax[0].plot(1e3*self.t, 1e-6*Ip_lp)
            ax[0].plot(1e3*tfit, 1e-6*Ifit, label=txt)
            ax[0].set_ylabel('$Ip$ MA')
            ax[0].legend()

            txt = '$t_{cq}$'+'={:1.1f}ms'.format(1e3*t_cq)
            ax[1].plot(1e3*self.t, 1e-6*dIpdt)
            ax[1].plot(1e3*t_cq, 1e-6*dIpdt[i_cq], 'o',
                       label=txt)
            ax[1].set_xlabel('$t$ ms')
            ax[1].set_ylabel('$dI_p/dt$ MAs$^{-1}$')
            ax[1].legend()
            plt.despine()
            plt.setp(ax[0].get_xticklabels(), visible=False)
            ax[0].text(0.5, 1.05, self.name,
                       transform=ax[0].transAxes,
                       bbox=dict(alpha=0.25), va='bottom', ha='center')

        return i_cq, t_cq, tdis, ttype

    def plot_currents(self, ax=None):
        if ax is None:
            ax = plt.subplots(2, 1, sharex=True)[1]
        ax[0].plot(1e3*self.t, 1e-6*self.Ip, 'C0')
        ax[0].set_ylabel('$I_p$ MA')
        ax[1].plot(1e3*self.t, 1e-3*self.Ivs_o, 'C1')
        ax[1].set_ylabel('$Ivs_o$ kA')
        ax[1].set_xlabel('$t$ ms')
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)

    def load_Ivs3(self, Rvs3=17.66e-3, Lvs3=1.52e-3, plot=False):
        self.Rvs3 = Rvs3
        self.Lvs3 = Lvs3
        self.tau_vs3 = self.Lvs3/self.Rvs3  # vs3 timeconstant
        self.quench = np.zeros(self.dina.nfolder,  # construct data strucutre
                               dtype=[('name', '60U'),
                                      ('t_cq', float),  # current quench
                                      ('i_cq', int),  # quench index
                                      ('I_cq', float),  # plasma current
                                      ('discharge_time', float),
                                      ('discharge_type', '60U'),
                                      ('Icontrol', float),  # control
                                      ('Ivs3_ref', float),  # zero, no spike
                                      ('Ifault', float), # fault
                                      ('Ivs3_o', float)])

        Ivs_dict = {'t': [], 'Icontol': [], 'Icp': [], 'Ifault': [],
                    'to': [], 'Io': []}
        self.Ivs = [Ivs_dict.copy()
                    for _ in range(self.dina.nfolder)]  # Ivs data
        for i in range(self.dina.nfolder):
            self.read_file(i)  # load plasma file
            i_cq, t_cq, tdis, ttype = self.get_quench()
            self.quench[i]['name'] = self.name
            self.quench[i]['t_cq'] = t_cq
            self.quench[i]['i_cq'] = i_cq
            self.quench[i]['I_cq'] = self.Ip[i_cq]
            self.quench[i]['Ivs3_o'] = self.Ivs_o[i_cq]
            self.quench[i]['discharge_time'] = tdis
            self.quench[i]['discharge_type'] = ttype
            max_index = np.argmax(abs(self.Ivs_o[i_cq:]))
            self.quench[i]['Ivs3_ref'] = self.Ivs_o[i_cq:][max_index]

            self.Ivs[i]['t'] = self.t[i_cq:]
            self.Ivs[i]['Icq'] = self.Ivs_o[i_cq:]
            self.Ivs[i]['to'] = self.t[:i_cq]
            self.Ivs[i]['Io'] = self.Ivs_o[:i_cq]
        if plot:
            self.plot_Ivs3()

    def get_color(self, index):
        t_cq = self.quench[index]['t_cq']
        if t_cq < 400e-3:  # MD
            iax = 0
            ic = 0
        else:  # VDE
            iax = 1
            ic = 2
        if 'DW' in self.quench[index]['name']:
            color = 'C{}'.format(ic)
        else:
            color = 'C{}'.format(ic+1)
        return color, iax

    def plot_Ivs3(self):
        ax = plt.subplots(2, 1)[1]
        text, loc = [], 'min'
        for i in range(2):
            text.append(linelabel(postfix='kA', value='1.1f',
                                  loc=loc, ax=ax[i]))
        for i in range(self.dina.nfolder):
            color, iax = self.get_color(i)
            ax[iax].plot(1e3*self.Ivs[i]['t'], 1e-3*self.Ivs[i]['Icq'],
                         color=color)
            text[iax].add('')
            ax[iax].plot(1e3*self.Ivs[i]['to'],
                         1e-3*self.Ivs[i]['Io'], '-',
                         color='gray', lw=0.5)
            ax[iax].plot(1e3*self.Ivs[i]['t'][0],
                         1e-3*self.Ivs[i]['Icq'][0], '*',
                         ms=8, color=color)
        plt.despine()
        for i, vde in enumerate(['MD', 'VDE']):
            h_down = mlines.Line2D([], [], color='C{}'.format(2*i),
                                   label='{} down'.format(vde))
            h_up = mlines.Line2D([], [], color='C{}'.format(2*i+1),
                                 label='{} up'.format(vde))
            ax[i].legend(handles=[h_down, h_up])
            ax[i].set_xlabel('$t$ ms')
            ax[i].set_ylabel('$I_{vs3}$ kA')
            text[i].plot()
        plt.tight_layout()

    def plot_plasma(self):
        ax = plt.subplots(1, 2, figsize=(8, 6), sharex=True, sharey=True)[1]

        text = []
        for i in range(2):
            text.append(linelabel(postfix='', value='', ax=ax[i], Ndiv=20))
        for i in range(self.dina.nfolder):
            self.read_file(i)  # load file
            i_cq = self.quench[i]['i_cq']
            I_cq = self.quench[i]['I_cq']
            color, iax = self.get_color(i)
            ax[iax].plot(self.x, self.z, '-', color=color)
            ax[iax].plot(self.x[0], self.z[0], '*', color=color)
            ax[iax].plot(self.x[i_cq], self.z[i_cq], 's', color=color)
            text[iax].add('$I_{cq}$ '+'{:1.1f}MA'.format(1e-6*I_cq))
        plt.axis('equal')
        plt.despine()
        for i, vde in enumerate(['MD', 'VDE']):
            h_down = mlines.Line2D([], [], color='C{}'.format(2*i),
                                   label='{} down'.format(vde))
            h_up = mlines.Line2D([], [], color='C{}'.format(2*i+1),
                                 label='{} up'.format(vde))
            ax[i].legend(handles=[h_down, h_up])
            text[i].plot()
            ax[i].set_xlabel('$x$ m')
        ax[0].set_ylabel('$y$ m')
        plt.setp(ax[1].get_yticklabels(), visible=False)

    def plot_Ivs_max(self, tau=32.5e-3):
        plt.figure()
        X = range(self.dina.nfolder)
        Imax = np.zeros(self.dina.nfolder)
        for i, x in enumerate(X):
            t = self.Ivs[i]['t']
            Idina = self.Ivs[i]['Icq']
            It = Idina - 60e3*np.exp(-(t-t[0])/tau)
            Imax[i] = It[np.argmax(abs(It))]
            color, iax = self.get_color(i)
            plt.bar(x, 1e-3*Imax[i], color=color, width=0.8)
        # plt.bar(x, Itau, width=0.6,
        #         label=r'$\tau=${:1.0f}ms'.format(plasma.tau))
        max_index = np.argmax(abs(Imax))
        va = 'bottom'
        plt.text(max_index, 1e-3*Imax[max_index],
                 ' {:1.0f}kA'.format(1e-3*Imax[max_index]),
                 va=va, ha='center',
                 color='k', weight='bold', rotation=90)
        plt.xticks(X, self.dina.folders, rotation=70)
        plt.ylabel('$I_{vs3}$ kA')
        # plt.legend()
        plt.despine()

    def trip_vs3(self, eps=0.01):
        trip_current = 0.01*np.max(abs(self.Ivs_o))
        trip_index = next(i for i, Iind in enumerate(self.Ivs_o)
                          if abs(Iind) > trip_current)
        trip_t = self.t[trip_index]  # vs3 circuit open
        return trip_index, trip_t

    def get_vs3(self, Io=60, tau=None, plot=False):
        if tau is None:
            tau = 1.53/(17.66)*1e3  # vs3 discharge time constant
        self.tau = tau
        self.t = np.copy(self.data.loc[:, 't'])
        self.Iind = self.data.loc[:, 'I_dw'] / 4  # per turn
        self.vs_sign = np.sign(self.Iind[np.argmax(abs(self.Iind))])
        Io *= self.vs_sign  # induced current in same direction as pulse
        self.Io = Io*np.ones(len(self.t))  # inital current

        trip_index, trip_t = self.trip_vs3()
        self.Io[trip_index:] = Io*np.exp(-(self.t[trip_index:]-trip_t)/tau)
        self.Ivs = self.Iind+self.Io

        Itau = self.Ivs
        Iinf = self.Iind+Io
        self.Imax = {'tau': Itau[np.argmax(abs(Itau))],
                     'inf': Iinf[np.argmax(abs(Iinf))]}
        if plot:
            loc = 'max' if self.vs_sign > 0 else 'min'
            text = linelabel(loc=loc, postfix='kA')
            plt.plot(self.t, self.Iind+Io, '-', label=r'$\tau=\infty$ms')
            text.add('')
            plt.plot(self.t, self.Ivs, '-',
                     label=r'$\tau=${:1.0f}ms'.format(self.tau))
            text.add('')
            plt.plot(self.t, self.Io, '--', label='$I_{decay}$')
            plt.plot(self.t, self.Iind, '--', label='$I_{VDE}$')
            ax = plt.gca()
            ylim = ax.get_ylim()
            plt.plot(trip_t*np.ones(2), ylim, ':', color='gray')
            plt.ylabel(r'$I_{VS3}$ kA')
            plt.xlabel(r'$t$ ms')
            plt.text(0.5, 1, self.name, transform=plt.gca().transAxes,
                     weight='bold', ha='center')
            text.plot(Ralign=False)
            plt.legend(loc='best')
            plt.despine()

    def plot_power(self):
        index = 3
        Io = 60e3

        t = self.Ivs[index]['t']
        Iin = self.Ivs[index]['Iinput']
        ax = plt.subplots(3, 1, sharex=True)[1]
        for i, s in enumerate([-1, 0, 1]):

            Ivs = lowpass(Iin, self.dt, dt_window=0.005)
            dIvsdt = np.gradient(Ivs, t)
            # vs = self.Rvs3*Ivs  # voltage source
            vs = self.Rvs3*Ivs + self.Lvs3*dIvsdt  # voltage source
            self.vs_fun = interp1d(t, vs, fill_value='extrapolate')

            Iode = odeint(self.dIdt_fun, Ivs[0]+s*Io, t)
            Iode = np.array(Iode).T[0]
            dIodedt = np.gradient(Iode, t)

            # Ivs = Iin + s*Io*np.exp(-(t-self.quench[index]['t_cq'])/self.tau)
            Pvs = self.get_power(Ivs.T, t)
            Pode = self.get_power(Iode.T, t)

            ax[0].plot(t, 1e-3*Ivs, 'C{}-'.format(i))
            ax[0].plot(t, 1e-3*Iode, 'C{}--'.format(i))

            ax[1].plot(t, 1e-3*dIvsdt, 'C{}-'.format(i))
            ax[1].plot(t, 1e-3*dIodedt, 'C{}--'.format(i))

            ax[2].plot(t, 1e-6*Pvs, 'C{}-'.format(i))
            ax[2].plot(t, 1e-6*Pode, 'C{}--'.format(i))
        plt.despine()

    def get_power(self, Ivs, t):
        Ivs = lowpass(Ivs, self.dt, dt_window=0.01)
        dIvsdt = np.gradient(Ivs, t)
        P = Ivs*(self.Lvs3*dIvsdt - Ivs*self.Rvs3)
        # P = Ivs*self.Lvs3*dIvsdt
        return P

    def dIdt_fun(self, I, t):
        # vs_fun = args[0]
        # g = -I*R/L
        g = (self.vs_fun(t) - I*self.Rvs3)/self.Lvs3
        return g



if __name__ == '__main__':

    pl = read_plasma('disruptions')

    pl.load_Ivs3()  # load Ivs3 current waveforms

    pl.plot_Ivs3()

    # pl.plot_power()

    pl.read_file(11)
    pl.get_quench(plot=True)

    pl.plot_plasma()

    pl.plot_Ivs_max()

    '''
    ax = plt.subplots(2, 1, sharex=True)[1]
    for i in range(pl.dina.nfolder):
        pl.read_file(i)
        pl.plot_currents(ax=ax)
    '''

