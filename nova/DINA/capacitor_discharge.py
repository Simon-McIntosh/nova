import numpy as np
from scipy.integrate import ode
from amigo.pyplot import plt
from math import isclose
from nep.coil_geom import VVcoils
from nep.DINA.coupled_inductors import inductance
from nep.DINA.vs3_flux import vs3_flux
from nep.DINA.read_plasma import read_plasma
import operator
from collections import OrderedDict
from matplotlib.lines import Line2D
from numba import jit


class power_supply:

    def __init__(self, **kwargs):
        self.set_defaults(**kwargs)
        self.pl = read_plasma('disruptions')
        self.vs3 = vs3_flux()  # vs3 flux profiles
        self.build_coils()
        self.ode = ode(self.dIdt).set_integrator('dopri5')
        self.initalize(**kwargs)

    def set_defaults(self, **kwargs):
        self.tau_d = kwargs.get('tau_d', 0.05)  # discharge timeconstant
        self.pulse = kwargs.get('pulse', True)
        self.impulse = kwargs.get('impulse', True)
        self.scenario = kwargs.get('scenario', -1)
        self.vessel = kwargs.get('vessel', True)

    def build_coils(self, plot=False):
        self.vvc = VVcoils()
        self.ind = inductance()
        nvs_o = self.ind.nC
        turns = np.append(np.ones(4), -np.ones(4))
        self.ind.add_pf_coil(
                OrderedDict(list(self.vvc.pf.sub_coil.items())[:8]),
                turns=turns)
        for i, index in enumerate(nvs_o+np.arange(1, 8)):  # vs3 loops
            self.ind.add_cp([nvs_o, index])  # link VS coils
        if self.vessel:
            self.ind.add_pf_coil(
                OrderedDict(list(self.vvc.pf.sub_coil.items())[8:]))
        self.ind.reduce()
        self.ncoil = self.ind.nd['nr']  # number of retained coils
        for i in range(self.ncoil):
            index = self.ind.M[i] > self.ind.M[i, i]
            index[0] = False  # maintain large coupling to VS3 coil
            if sum(index) > 0:  # ensure dominant leading diagonal
                self.ind.M[i, index] = 0.9*self.ind.M[i, i]
        if plot:
            self.ind.plot()

    def initalize(self, **kwargs):
        self.initalize_switches()
        self.tpo = 0  # pulse start time
        self.tco = 0  # capacitor discarge start time
        self.tc10 = 0  # capacitor 10% current rise time
        self.data = {'t': [], 'Ivec': [], 'Jvec': [], 'Vcap': [], 'Vps': [],
                     'npulse': 0, 'dt_discharge': [], 'dt_rise': [],
                     'Icap': []}
        self.build_matrices('DINA', **kwargs)
        self.set_constants()
        self.set_time(**kwargs)
        self.load_background_flux(**kwargs)

    def set_constants(self):
        self.sign = 1
        self.Rcap = 0.005  # capacitor charge resistance
        self.Itrip = self.sign * 60e3  # impulse set-point
        self.Vo = self.sign * 2.3e3  # capacitor voltage
        self.Vps_max = self.sign * 1.35e3  # maximum power supply voltage

    def set_time(self, **kwargs):
        self.pulse = kwargs.get('pulse', self.pulse)
        self.impulse = kwargs.get('impulse', self.impulse)
        self.pulse_phase = -0.02  # pulse phase (-ive == lag)
        self.t_pulse = 0.0
        self.pulse_period = 10  # duration between pulses

    def build_matrices(self, code, **kwargs):
        self.tau_d = kwargs.get('tau_d', self.tau_d)  # discharge timeconstant
        if code == 'DINA':
            Lvs3 = 1.32e-3  # total inductance of vs3 loop
            Rvs3 = 12.01e-3  # total vs3 loop resistance
        elif code == 'LTC':
            Lvs3 = 1.38e-3  # total inductance of vs3 loop
            Rvs3 = 17.66e-3  # total vs3 loop resistance
        elif code == 'IO':
            Lvs3 = self.ind.M[0, 0] + 0.2e-3  # add busbar inductance (1.56e-3)
            Rvs3 = 17.66e-3  # total vs3 loop resistance
        self.R = np.copy(self.ind.Rc)  # coil resistance vector
        self.R[0] = Rvs3   # set vs3 loop resistance
        self.M = np.copy(self.ind.M)  # inductance matrix
        self.M[0, 0] = Lvs3   # set total inductance of vs3 loop
        self.Minv = np.linalg.inv(self.M)  # inverse
        self.C = np.inf * np.ones(self.ncoil)
        self.C[0] = self.tau_d/self.R[0]  # impulse capacitor

    def op(self, key, a, b):
        if self.sign < 0:  # flip opperator
            if key[0] == 'g':
                key = 'l'+key[1:]
            elif key[0] == 'l':
                key = 'g'+key[1:]
        return getattr(operator, key)(a, b)

    def zeros(self, t):
        return np.zeros(self.ncoil)

    def load_background_flux(self, **kwargs):
        scenario = kwargs.get('scenario', self.scenario)
        if scenario < 0:  # set zero flux
            self.Vbg = self.zeros
            self.dVbg = self.zeros
        else:
            self.vs3.load_psi(scenario, plot=False, read_txt=False)
            self.Vbg = self.vs3.Vbg
            self.dVbg = self.vs3.dVbg

    def initalize_switches(self):
        self.capacitor_discharge = False
        self.pulse_hold = False

    def get_step(self):
        if self.capacitor_discharge:
            dt = 1e-4
        elif self.pulse_hold:
            dt = 5e-4
        else:
            dt = 1e-3
        return dt

    def dIdt(self, t, I):
        Hcap = I[0]  # capacitor intergral current
        Icap = I[1]  # capacitor charge current
        Ivec = I[2:2+self.ncoil]
        Jvec = I[-self.ncoil:]
        Vbg = self.Vbg(t)
        dVbg = self.dVbg(t)
        if not self.vessel:
            Vbg = Vbg[0]
            dVbg = dVbg[0]
        Vps = np.zeros(self.ncoil)  # power supply voltage
        dVps = np.zeros(self.ncoil)  # power supply voltage rate
        self.Vps = self.dVps = 0  # external power supply
        if self.capacitor_discharge:  # capacitor discharging
            dHcap = -Ivec[0]  # discharge capacitor
            dIcap = 0  # zero charge current
            #dIvec = np.copy(Jvec)
            Hvec = np.zeros(self.ncoil)
            Hvec[0] = -Hcap
            dIvec = np.dot(self.Minv, - self.R*Ivec - Hvec/self.C + Vbg)  #
            dJvec = np.dot(self.Minv, - self.R*Jvec - Ivec/self.C + dVbg)
            #dJvec = np.zeros(self.ncoil)
        else:  # capacitor charging
            dHcap = self.Vo / self.Rcap - Hcap / (self.Rcap * self.C[0])
            dIcap = -Icap / (self.Rcap * self.C[0])
            if self.pulse_hold:
                k = 5e-1  # power supply gain
                Ierr = self.Itrip - Ivec[0]
                self.Vps = self.R[0]*Ivec[0] + k*Ierr  # power supply voltage
                if self.op('gt', self.Vps, self.Vps_max):  # saturated
                    self.Vps = self.Vps_max
                    self.dVps = 0
                else:
                    self.dVps = self.R[0]*Jvec[0] - k*Jvec[0]
                Vps[0] = self.Vps
                dVps[0] = self.dVps
            dIvec = np.dot(self.Minv, - self.R*Ivec + Vps + Vbg)
            dJvec = np.dot(self.Minv, - self.R*Jvec + dVps + dVbg)
        self.Vcap = Hcap/self.C[0]  # capacitor voltage
        dI = np.array([dHcap, dIcap])
        dI = np.append(dI, dIvec)
        dI = np.append(dI, dJvec)
        return dI

    def set_switches(self, t, Iode):
        if self.capacitor_discharge:  # capacitor discharging
            if self.op('ge', Iode[2], 0.1*self.Itrip) \
                    and self.tc10 < self.tco:
                self.tc10 = t  # 10% capacitor current rise
            if self.op('ge', Iode[2], self.Itrip) \
                    or isclose(Iode[2], self.Itrip):#\
                    #or self.op('lt', Iode[-self.ncoil], 0):  # capacitor vide
                print(self.op('ge', Iode[2], self.Itrip),
                      isclose(Iode[2], self.Itrip),
                      self.op('lt', Iode[-self.ncoil], 0))
                self.capacitor_discharge = False  # charge capacitor
                Icap = (self.Vo - Iode[0] / self.C[0]) / self.Rcap
                Iode_o = np.copy(Iode)
                Iode_o[1] = Icap  # initial capacitor charge current
                self.ode.set_initial_value(Iode_o, t=t)
                self.tpo = t  # pulse start time
                self.data['npulse'] += 1  # pulse counter
                self.data['dt_discharge'].append(t - self.tco)
                if self.t_pulse > 0:
                    self.pulse_hold = True
        else:
            npulse = np.floor((t + self.pulse_phase) / self.pulse_period)
            if self.impulse:  # enable capacitor dischage
                if npulse > 0 and self.pulse is False:  # set pulse
                    npulse = 0
            else:
                npulse = -1
            if npulse >= self.data['npulse']:  # start pulse
                self.capacitor_discharge = True  # discharge capacitor
                Hcap = np.zeros(self.ncoil)
                Hcap[0] = Iode[0]  # capacitor intergral current
                Ivec = Iode[2:2+self.ncoil]
                Vbg = self.Vbg(t)  # background voltage
                #Jvec = -np.dot(self.Minv, Vbg - self.R*Ivec - Hcap/self.C)
                Jvec = -np.dot(self.Minv,- self.R*Ivec - Hcap/self.C)
                Jvec = np.zeros(self.ncoil)



                Iode_o = np.copy(Iode)
                Iode_o[-self.ncoil:] = Jvec  # inital current rate in vs3 loop
                self.ode.set_initial_value(Iode_o, t=t)
                self.tco = t  # capacitor discarge start time
        if len(self.data['dt_rise']) < self.data['npulse'] and \
                self.op('ge', Iode[2], 0.9*self.Itrip):  # attained set-point
            self.data['dt_rise'].append(t - self.tc10)  # capacitor risetime
        if self.pulse_hold and t - self.tpo > self.t_pulse:
            self.pulse_hold = False

    def store_data(self, t, Iode):
        self.data['t'].append(t)
        self.data['Ivec'].append(Iode[2:2+self.ncoil])  # current
        self.data['Jvec'].append(Iode[-self.ncoil:])  # current rate
        self.data['Vps'].append(self.Vps)  # power supply voltage
        self.data['Vcap'].append(self.Vcap)  # capacitor voltage
        self.data['Icap'].append(Iode[1])  # capacitor current

    def intergrate(self, tend, Io=0):
        #  [Hcap, Icap, Ivec, Jvec]
        Iode_o = np.zeros(2 + 2*self.ncoil)
        Iode_o[0] = self.C[0] * self.Vo  # capacitor fully charged
        Iode_o[2] = Io  # vs3 circuit inital current
        self.ode.set_initial_value(Iode_o, t=0)
        self.set_switches(0, Iode_o)
        while self.ode.successful() and self.ode.t < tend:
            dt = self.get_step()
            Iode = self.ode.integrate(self.ode.t+dt, step=True)
            self.set_switches(self.ode.t, Iode)
            self.store_data(self.ode.t, Iode)

    def solve(self, tend, Io=0, plot=False, **kwargs):
        self.initalize(**kwargs)
        self.intergrate(tend, Io=Io)
        self.packdata()
        if plot:
            self.plot()
            self.plot_current()

    def packdata(self):
        for var in ['t', 'Ivec', 'Jvec', 'Vps', 'Vcap', 'Icap']:
            self.data[var] = np.array(self.data[var])

    def plot(self):
        ax = plt.subplots(4, 1, sharex=True)[1]
        ax[0].plot(self.data['t'], 1e-3*self.data['Ivec'][:, 0], '-C0')
        ax[0].set_ylabel('$I_{vs3}$ kA')
        ax[1].plot(self.data['t'], 1e-3*self.data['Vps'], '-C1')
        ax[1].set_ylabel('$V_{ps}$ kV')
        ax[2].plot(self.data['t'], 1e-3*self.data['Vcap'], '-C2')
        ax[2].set_ylabel('$V_{capacitor}$ kV')
        ax[3].plot(self.data['t'], 1e-3*self.data['Icap'], '-C3')
        ax[3].set_ylabel('$I_{charge}$ kA')
        plt.despine()
        plt.detick(ax)

    def plot_current(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1, sharex=True)[1]

        if self.vessel:
            colors = OrderedDict()
            colors['lower_vvo'] = 'C1'
            colors['lower_vv1'] = 'C2'
            colors['lower_trs'] = 'C3'
            coils = list(self.ind.pf.coil.keys())[8:]
            for i, coil in enumerate(coils):
                for c in colors:
                    if c in coil:
                        color = colors[c]
                        break
                ax.plot(self.data['t'], 1e-3*self.data['Ivec'][:, i+1],
                        color=color)
            lines, labels = [], []
            for c in colors:
                lines.append(Line2D([0], [0], color=colors[c]))
                labels.append(c)
            ax.legend(lines, labels)

        ax.plot(self.data['t'], 1e-3*self.data['Ivec'][:, 0], '-C0')

        ax.set_ylabel('$I_{vs3}$ kA')
        if self.scenario > 0:
            DINA = self.pl.Ivs3_single(self.scenario)[1]
            ax.plot(DINA['t'], 1e-3*DINA['Ireferance'], 'C3--')

    #def plot_patch(self, index):
    #    for


if __name__ == '__main__':

    ps = power_supply(scenario=-1, vessel=True)

    tau_d = 0.05  # discharge time constant
    ps.solve(0.08, tau_d=tau_d, Io=0, impulse=True, pulse=True, plot=True)

