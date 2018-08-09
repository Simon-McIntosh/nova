import numpy as np
from scipy.integrate import ode
from amigo.pyplot import plt
from math import isclose
from nep.coil_geom import VVcoils
from nep.DINA.coupled_inductors import inductance
from nep.DINA.coil_flux import coil_flux
from nep.DINA.read_plasma import read_plasma
import operator
from collections import OrderedDict
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
import os
import nep
from amigo.png_tools import data_load
from amigo.IO import class_dir


class power_supply:

    def __init__(self, read_txt=False, **kwargs):
        self.set_defaults()
        self.update_defaults(**kwargs)
        self.pl = read_plasma('disruptions', read_txt=read_txt)
        self.cf = coil_flux(read_txt=read_txt)  # initalize flux profiles
        self.build_coils()
        self.ode = ode(self.dIdt).set_integrator('dop853')  # 'dopri5'

    def set_defaults(self):
        self.setup = {'Cd': 2.42, 'pulse': True, 't_pulse': 0.0,
                      'pulse_period': 10, 'impulse': True, 'scenario': -1,
                      'jacket': True, 'vessel': True,
                      'nturn': 4, 'sign': -1, 'code': 'IO',
                      'dt_discharge': 0, 'Io': 0, 'switch_capacitor': True}
        for var in self.setup:
            setattr(self, var, self.setup[var])

    def update_defaults(self, **kwargs):
        for var in kwargs:
            self.setup[var] = kwargs[var]
            setattr(self, var, self.setup[var])

    def build_coils(self, plot=False):
        self.vv = VVcoils()
        self.ind = inductance()
        nvs_o = self.ind.nC
        turns = np.append(np.ones(4), -np.ones(4))
        self.ind.add_pf_coil(
                OrderedDict(list(self.vv.pf.sub_coil.items())[:8]),
                turns=turns)
        for index in nvs_o+np.arange(1, 8):  # link vs turns
            self.ind.add_cp([nvs_o, index])
        if self.jacket:  # add jacket coils
            jacket = list(self.vv.pf.sub_coil.items())[8:16]
            self.ind.add_pf_coil(OrderedDict(jacket))
        if self.vessel:  # add vv coils
            vv_coils = list(self.vv.pf.sub_coil.items())[16:]
            self.ind.add_pf_coil(OrderedDict(vv_coils))

        self.ind.assemble()
        '''
        diag = np.diag(self.ind.Mo)
        diag_m = np.max([diag[:-1], diag[1:]], axis=0)
        diag_1 = np.diag(self.ind.Mo, k=1)
        for i, (dm, d1) in enumerate(zip(diag_m, diag_1)):
            if abs(d1) > 0.5*dm:
                print(d1, dm)
                self.ind.Mo[i, i+1] = 0.5*np.sign(d1)*dm
                self.ind.Mo[i+1, i] = 0.5*np.sign(d1)*dm
        '''
        self.ind.constrain()
        self.ncoil = self.ind.nd['nr']  # number of retained coils

        for i in np.arange(1, self.ncoil):
            index = abs(self.ind.M[i]) > self.ind.M[i, i]
            index[0] = False  # maintain large coupling to VS3 coil
            #index[i] = False
            if sum(index) > 0:  # ensure dominant leading diagonal
                self.ind.M[i, index] = 0.9*self.ind.M[i, i]
        if plot:
            self.ind.plot()

    def initalize(self, **kwargs):
        self.initalize_switches()
        self.data = {'t': [], 'Ivec': [], 'Vcap': [], 'Vps': [],
                     'npulse': 0, 'dt_discharge': [], 'dt_rise': [],
                     'Icap': []}
        self.update_defaults(**kwargs)
        if 'vessel' in kwargs:
            self.build_coils()  # re-build coil-set
        self.build_matrices()
        self.set_constants()
        self.set_time(**kwargs)
        self.load_background_flux()

    def set_constants(self):
        self.Rcap = 0.005  # capacitor charge resistance
        self.Itrip = self.sign * 4/self.nturn * 60e3  # impulse set-point
        self.Vo = self.sign * 2.3e3  # capacitor voltage
        self.Vps_max = self.sign * 1.35e3  # maximum power supply voltage

    def set_time(self, **kwargs):
        if self.scenario != -1 and self.dt_discharge != 0:  # load scenario
            trip = self.pl.Ivs3_single(self.scenario)[0]
            self.t_trip = trip[-1]
        else:
            self.t_trip = 0
        # pulse phase (-ive == lag) = peak coninsident with plasma trip
        if 'pulse_phase' in kwargs:
            self.pulse_phase = kwargs['pulse_phase']
        else:
            self.pulse_phase = self.dt_discharge + self.t_pulse - self.t_trip
        self.tpo = -self.t_pulse + self.t_trip  # pulse start time
        self.tco = -self.pulse_phase + self.t_trip  # discarge start time
        self.tc10 = -self.pulse_phase + self.t_trip  # 10% current rise time

    def build_matrices(self):
        if self.code == 'DINA':
            Lvs3 = 1.32e-3  # total inductance of vs3 loop
            Rvs3 = 12.01e-3  # total vs3 loop resistance
        elif self.code == 'LTC':
            Lvs3 = 1.38e-3  # total inductance of vs3 loop
            Rvs3 = 17.66e-3  # total vs3 loop resistance
        elif self.code == 'IO':
            Lvs3 = self.ind.M[0, 0] + 0.2e-3  # add busbar inductance (1.56e-3)
            Rvs3 = 17.66e-3  # total vs3 loop resistance
        self.R = np.copy(self.ind.Rc)  # coil resistance vector
        self.R[0] = Rvs3   # set vs3 loop resistance
        self.M = np.copy(self.ind.M)  # inductance matrix
        self.M[0, 0] = Lvs3   # set total inductance of vs3 loop
        if self.nturn != 4:  # operation with reduced turn number
            factor = self.nturn / 4
            self.R[0] *= factor
            self.M[0, :] *= factor
            self.M[1:, 0] *= factor
        self.Minv = np.linalg.inv(self.M)  # inverse
        self.C = np.inf * np.ones(self.ncoil)
        self.C[0] = self.Cd  # impulse capacitor

    def op(self, key, a, b):
        if self.sign < 0:  # flip opperator
            if key[0] == 'g':
                key = 'l'+key[1:]
            elif key[0] == 'l':
                key = 'g'+key[1:]
        return getattr(operator, key)(a, b)

    def zeros(self, t):
        return np.zeros(self.ncoil)

    def load_background_flux(self):
        if self.scenario < 0:  # set zero flux
            self.Vbg = self.zeros
            self.dVbg = self.zeros
        else:
            self.cf.load_file(self.scenario, plot=False)
            self.Vbg = self.cf.Vbg
            self.dVbg = self.cf.dVbg

    def initalize_switches(self):
        self.capacitor_empty = False
        self.capacitor_discharge = False
        self.pulse_hold = False

    def get_step(self):
        if self.capacitor_discharge:
            dt = 1e-5
        elif self.pulse_hold:
            dt = 1e-4
        else:
            dt = 1e-4
        return dt

    def dIdt(self, t, I):
        Hcap = I[0]  # capacitor intergral current
        Icap = I[1]  # capacitor charge current
        Ivec = I[2:2+self.ncoil]
        Vbg = self.Vbg(t)
        Vbg[0] *= self.nturn/4
        if not self.vessel:
            Vbg = Vbg[0]
        Vps = np.zeros(self.ncoil)  # power supply voltage
        dVps = np.zeros(self.ncoil)  # power supply voltage rate
        self.Vps = self.dVps = 0  # external power supply
        if self.capacitor_discharge:  # capacitor discharging
            dHcap = -Ivec[0]  # discharge capacitor
            dIcap = 0  # zero charge current
            Hvec = np.zeros(self.ncoil)
            Hvec[0] = -Hcap
            dIvec = np.dot(self.Minv, - self.R*Ivec - Hvec/self.C + Vbg)
        else:  # capacitor charging
            dHcap = self.Vo / self.Rcap - Hcap / (self.Rcap * self.C[0])
            dIcap = -Icap / (self.Rcap * self.C[0])
            if self.pulse_hold:
                k = 0.5  # power supply gain
                Ierr = self.Itrip - Ivec[0]
                self.Vps = self.R[0]*Ivec[0] + k*Ierr  # power supply voltage
                if self.op('gt', self.Vps, self.Vps_max):  # saturated
                    self.Vps = self.Vps_max
                if self.op('lt', self.Vps, -self.Vps_max):  # saturated
                    self.Vps = -self.Vps_max
                Vps[0] = self.Vps
                dVps[0] = self.dVps
            dIvec = np.dot(self.Minv, - self.R*Ivec + Vps + Vbg)
        self.Vcap = Hcap/self.C[0]  # capacitor voltage
        dI = np.array([dHcap, dIcap])
        dI = np.append(dI, dIvec)
        return dI

    def check_setpoint(self, t, Iode):  # attained set-point
        if len(self.data['dt_rise']) <= self.data['npulse'] and \
                self.op('ge', Iode[2], 0.9*self.Itrip):
            self.data['dt_rise'].append(t - self.tc10)

    def set_switches(self, t, Iode):
        if self.capacitor_discharge:  # capacitor discharging
            dI = self.dIdt(t, Iode)
            if self.op('ge', Iode[2], 0.1*self.Itrip) \
                    and self.tc10 <= self.tco:
                self.tc10 = t  # 10% capacitor current rise
            self.check_setpoint(t, Iode)
            if (self.op('ge', Iode[2], self.Itrip) or
                    isclose(Iode[2], self.Itrip) or
                    self.op('lt', dI[2], 0)) and self.switch_capacitor:
                self.capacitor_discharge = False  # charge capacitor
                if self.op('lt', dI[2], 0):
                    self.capacitor_empty = True
                Icap = (self.Vo - Iode[0] / self.C[0]) / self.Rcap
                Iode_o = np.copy(Iode)
                Iode_o[1] = Icap  # initial capacitor charge current
                self.ode.set_initial_value(Iode_o, t=t)
                self.tpo = t  # pulse start time
                self.data['npulse'] += 1  # pulse counter
                if not self.capacitor_empty:
                    self.data['dt_discharge'].append(t - self.tco)
                if self.t_pulse > 0 or self.capacitor_empty:
                    self.pulse_hold = True
        else:
            npulse = np.floor((t + self.pulse_phase) / self.pulse_period)
            if self.impulse:  # enable capacitor dischage
                if npulse > 0 and self.pulse is False:  # limit pulse number
                    npulse = 0
            else:
                npulse = -1
            if npulse >= self.data['npulse']:  # start pulse
                self.capacitor_discharge = True  # discharge capacitor
                Iode_o = np.copy(Iode)
                self.tco = t  # capacitor discarge start time
        if self.capacitor_empty:  # ensure set-point is reached
            self.check_setpoint(t, Iode)
            if isclose(Iode[2], self.Itrip, abs_tol=1e2):
                self.capacitor_empty = False
                self.data['dt_discharge'].append(t - self.tco)
                self.tpo = t  # pulse start time
        elif self.pulse_hold and t - self.tpo > self.t_pulse:
                self.pulse_hold = False
        if self.scenario >= 0 and t > self.t_trip:  # interlock on
            self.pulse_hold = False

    def store_data(self, t, Iode):
        self.data['t'].append(t)
        self.data['Ivec'].append(Iode[2:2+self.ncoil])  # current
        self.data['Vps'].append(self.Vps)  # power supply voltage
        self.data['Vcap'].append(self.Vcap)  # capacitor voltage
        self.data['Icap'].append(Iode[1])  # capacitor current

    def intergrate(self, tend, Io=0):
        #  [Hcap, Icap, Ivec]
        Iode_o = np.zeros(2 + self.ncoil)
        Iode_o[0] = self.C[0] * self.Vo  # capacitor fully charged
        Iode_o[2] = Io  # vs3 circuit inital current
        to = 0 if self.pulse_phase < 0 else -self.pulse_phase
        self.ode.set_initial_value(Iode_o, t=to)
        self.set_switches(to, Iode_o)
        while self.ode.successful() and self.ode.t < tend:
            dt = self.get_step()
            Iode = self.ode.integrate(self.ode.t+dt, step=True)
            self.set_switches(self.ode.t, Iode)
            self.store_data(self.ode.t, Iode)

    def set_dt_discharge(self):
        if self.dt_discharge == 0 and self.impulse:
            scenario = self.scenario
            t_pulse = self.t_pulse
            self.initalize(scenario=-1, t_pulse=0)
            self.intergrate(100e-3)
            self.packdata()
            self.plot()
            self.plot_current()
            dt_discharge = self.data['dt_discharge'][0]
            self.initalize(scenario=scenario, dt_discharge=dt_discharge,
                           t_pulse=t_pulse)

    def solve(self, tend=None, plot=False, **kwargs):
        self.initalize(**kwargs)
        if tend is None:
            try:
                tend = self.cf.t[-1]
            except AttributeError:
                tend = 100e-3
        self.set_dt_discharge()
        self.intergrate(tend)
        self.packdata()
        if plot:
            self.plot()
            self.plot_current()
        return self.Ivec

    def packdata(self):
        for var in ['t', 'Ivec', 'Vps', 'Vcap', 'Icap']:
            self.data[var] = np.array(self.data[var])
        self.Ivec = interp1d(self.data['t'], self.data['Ivec'], axis=0,
                             fill_value=(self.data['Ivec'][0],
                                         self.data['Ivec'][-1]),
                             bounds_error=False)

    def plot(self):
        ax = plt.subplots(4, 1, sharex=True)[1]
        ax[0].plot(1e3*self.data['t'], 1e-3*self.data['Ivec'][:, 0], '-C0')
        ax[0].set_ylabel('$I_{vs3}$ kA')
        ax[1].plot(1e3*self.data['t'], 1e-3*self.data['Vps'], '-C1')
        ax[1].set_ylabel('$V_{ps}$ kV')
        ax[2].plot(1e3*self.data['t'], 1e-3*self.data['Vcap'], '-C2')
        ax[2].set_ylabel('$V_{capacitor}$ kV')
        ax[3].plot(1e3*self.data['t'], 1e-3*self.data['Icap'], '-C3')
        ax[3].set_ylabel('$I_{charge}$ kA')
        plt.despine()
        plt.detick(ax)
        ax[-1].set_xlabel('$t$ ms')

    def plot_current(self, ax=None):
        if ax is None:
            ax = plt.subplots(1, 1, sharex=True)[1]
        if self.vessel:
            colors = OrderedDict()
            colors['conductor'] = 'C0'
            colors['jacket'] = 'C6'
            colors['lower_vvo'] = 'C1'
            colors['lower_vv1'] = 'C2'
            colors['lower_trs'] = 'C3'
            colors['upper_vvo'] = 'C4'
            colors['upper_vv1'] = 'C5'
            coils = list(self.ind.pf.coil.keys())[16:]
            for i, coil in enumerate(coils):
                for c in colors:
                    if c in coil:
                        color = colors[c]
                        break
                ax.plot(1e3*self.data['t'], 1e-3*self.data['Ivec'][:, i+2],
                        color=color)
            lines, labels = [], []
            for c in colors:
                lines.append(Line2D([0], [0], color=colors[c]))
                labels.append(c)
            ax.legend(lines, labels)

        ax.plot(1e3*self.data['t'], 1e-3*self.data['Ivec'][:, :], '-C0')
        #text.plot()

        if self.jacket:
            ax.plot(1e3*self.data['t'], 1e-3*self.data['Ivec'][:, 1], '-C6')
        ax.set_ylabel('$I_{vs3}$ kA')
        ax.set_xlabel('$t$ ms')
        plt.despine()
        imax = np.argmax(abs(self.data['Ivec'][:, 0]))
        Imax = self.data['Ivec'][imax, 0]
        tmax = self.data['t'][imax]
        if Imax > 0:
            va = 'bottom'
        else:
            va = 'top'
        ax.plot(1e3*tmax, 1e-3*Imax, '.C7')
        ax.text(1e3*tmax, 1e-3*Imax, ' {:1.1f}kA'.format(1e-3*Imax),
                ha='left', va=va, color='C7')
        plt.title('t pulse {:1.2f}s'.format(self.t_pulse))

    def benchmark(self):
        if self.data['npulse'] != 1 or self.sign != -1 or self.nturn != 4\
                or self.t_pulse != 0.3 or not self.impulse or not self.vessel:
            if not self.vessel:
                self.update_defaults(vessel=True)
                self.build_coils()
            self.solve(Io=0, sign=-1, nturn=4, t_pulse=0.3, impulse=True)
        path = os.path.join(class_dir(nep), '../Data/LTC/')
        points = data_load(path, 'VS3_current', date='2018_03_15')[0]
        td, Id = points[0]['x'], points[0]['y']  # jacket + vessel

        plt.figure()
        plt.plot(1e3*self.data['t'], 1e-3*self.data['Ivec'][:, 0], '-C0',
                 label='NOVA')
        plt.plot(1e3*td, -1e-3*Id, '--', label='LTC')
        plt.xlim([0, 80])
        plt.legend()
        plt.despine()
        plt.xlabel('$t$ ms')
        plt.ylabel('$I$ kA')



if __name__ == '__main__':

    ps = power_supply(nturn=4, vessel=True, jacket=True,
                      scenario=3, code='IO')

    ps.solve(Io=0, sign=-1, nturn=4, t_pulse=0.0,
             impulse=True, plot=True)
    #ps.benchmark()



