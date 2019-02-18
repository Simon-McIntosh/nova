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
from scipy.optimize import minimize
import nep_data
from os.path import join, isfile
from amigo.IO import pythonIO
from nova.coils import PF


class power_supply(pythonIO):

    def __init__(self, read_txt=False, **kwargs):
        self.read_txt = read_txt
        self.set_defaults()
        self.update_defaults(**kwargs)
        self.pl = read_plasma('disruptions', read_txt=read_txt)
        self.cf = coil_flux(read_txt=read_txt)  # initalize flux profiles
        self.pf = PF()
        self.load_coils(read_txt=read_txt)
        self.ode = ode(self.dIdt).set_integrator('dop853')  # 'dopri5'
        self.initalize(**kwargs)

    def set_defaults(self):
        self.setup = {'Cd': 2.42, 'pulse': True, 't_pulse': 0.0,
                      'pulse_period': 10, 'impulse': True, 'scenario': -1,
                      'vessel': True, 'nturn': 4, 'sign': -1, 'code': 'Nova',
                      'dt_discharge': 0, 'Io': 0, 'origin': 'peak',
                      'trip': 't_trip', 'Ip_scale': 1, 'vessel_model': 'local',
                      'Vo_factor': 1, 'invessel': True, 'Ipulse': 60e3}
        for var in self.setup:
            setattr(self, var, self.setup[var])

    def update_defaults(self, **kwargs):
        for var in kwargs:
            self.setup[var] = kwargs[var]
            setattr(self, var, self.setup[var])

    def get_filename(self, **kwargs):
        vessel = 'vessel' if self.setup['vessel'] else 'no_vessel'
        in_vessel = 'in_vessel' if self.setup['invessel'] else 'ex_vessel'
        vessel_model = self.setup['vessel_model']
        filename = f'Nova_{vessel}_{vessel_model}_{in_vessel}'
        return filename

    def load_coils(self, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        data_dir = join(class_dir(nep_data), 'EM', 'inductance')
        filename = self.get_filename()
        filepath = join(data_dir, filename)
        attributes = ['coilset', 'M', 'R', 'ncoil']
        if read_txt or not isfile(filepath + '.pk'):
            self.build_coils()
            self.save_pickle(filepath, attributes)
        else:
            self.load_pickle(filepath)
            self.pf(self.coilset)  # update local pf instance

    def build_coils(self, plot=False):
        self.vv = VVcoils(model=self.vessel_model, invessel=self.invessel)
        self.ind = inductance()
        nvs_o = self.ind.nC
        self.coilset = self.vv.pf.coilset
        self.pf(self.coilset)
        turns = np.append(np.ones(4), -np.ones(4))
        self.ind.add_pf_coil(
                OrderedDict(list(self.coilset['subcoil'].items())[:8]),
                turns=turns)
        for index in nvs_o+np.arange(1, 8):  # vs3 loops
            self.ind.add_cp([nvs_o, index])  # link VS coils
        nvs_o = self.ind.nC
        jacket = list(self.coilset['coil'].items())[2:10]
        R = np.array([turn[1]['R'] for turn in jacket])
        Rlower = 1 / (np.sum(1/R[:4]))
        Rupper = 1 / (np.sum(1/R[4:]))
        for i in range(8):
            Rt = Rlower/4 if i < 4 else Rupper/4
            jacket[i][1]['R'] = Rt
        # add jacket coils
        turns = 0.25*np.ones(8)
        self.ind.add_pf_coil(OrderedDict(jacket), turns=turns)
        for index in nvs_o+np.arange(1, 4):
            self.ind.add_cp([nvs_o, index])  # lower jacket coils
        for index in nvs_o+4+np.arange(1, 4):
            self.ind.add_cp([nvs_o+4, index])  # upper jacket coils
        if self.vessel:  # add vv coils
            vv_coils = list(self.coilset['coil'].items())[10:]
            self.ind.add_pf_coil(OrderedDict(vv_coils))
        self.ind.reduce()
        self.ncoil = self.ind.nd['nr']  # number of retained coils
        if self.vessel_model == 'local':
            factor = 0.95
        else:
            factor = 0.92
        for i in np.arange(3, self.ncoil):
            index = abs(self.ind.M[i]) > factor * self.ind.M[i, i]
            index[0:3] = False  # maintain large coupling to VS3 coil
            index[i] = False
            if sum(index) > 0:  # ensure dominant leading diagonal
                self.ind.M[i, index] = factor * self.ind.M[i, i]
        self.M = self.ind.M
        self.R = self.ind.Rc
        if plot:
            self.pf.plot()

    def initalize(self, **kwargs):
        self.initalize_switches()
        self.data = {'t': [], 'Ivec': [], 'Vcap': [], 'Vps': [],
                     'npulse': 0, 'dt_discharge': [], 'dt_rise': [],
                     'Icap': []}
        self.update_defaults(**kwargs)
        if 'vessel' in kwargs or 'invessel' in kwargs:
            # self.build_coils()  # re-build coil-set
            self.load_coils()
        build_keys = ['code', 'nturn', 'Cd', 'vessel']
        if np.array([key in kwargs for key in build_keys]).any():
            self.build_matrices()
        self.set_constants()
        self.set_time()
        self.load_background_flux()

    def set_constants(self):
        self.Rcap = 0.005  # capacitor charge resistance
        self.Itrip = self.sign * 4/self.nturn * self.Ipulse  # set-point
        self.Vo = self.sign * self.Vo_factor * 2.3e3  # capacitor voltage
        # maximum power supply voltage
        self.Vps_max = self.sign * self.Vo_factor * 1.35e3

    def set_time(self):
        # self.trip = 't_trip', 't_cq', 't_dz', float
        if isinstance(self.trip, str):
            if self.scenario != -1 and self.dt_discharge != 0:  # load scenario
                trip = self.pl.Ivs3_single(self.scenario)[0]
                self.t_trip = trip[self.trip]  # 't_trip', 't_cq', 't_dz'
            else:
                self.t_trip = 0
        else:
            self.t_trip = self.trip  # trigger time
        # pulse phase (-ive == lag) = peak coninsident with plasma trip
        self.pulse_phase = -self.t_trip
        # self.origin = 'peak', 'start'
        if self.origin == 'peak':  # align with end of pulse peak
            self.pulse_phase += self.dt_discharge + self.t_pulse
        self.tpo = -self.pulse_phase + self.t_trip  # pulse start time
        self.tco = -self.pulse_phase + self.t_trip  # discarge start time
        self.tc10 = -self.pulse_phase + self.t_trip  # 10% current rise time

    def build_matrices(self):
        if self.code == 'DINA':
            Lvs3 = 1.32e-3  # total inductance of vs3 loop
            Rvs3 = 12.01e-3  # total vs3 loop resistance
        elif self.code == 'LTC':
            # Lvs3 = 1.38e-3  # total inductance of vs3 loop
            Lvs3 = 1.395e-3  # total inductance of vs3 loop
            Rvs3 = 17.66e-3  # total vs3 loop resistance
        elif self.code == 'Nova':
            Lvs3 = self.M[0, 0] + 0.2e-3  # add busbar inductance (1.56e-3)
            Rvs3 = 17.66e-3  # total vs3 loop resistance
        self.R[0] = Rvs3   # set vs3 loop resistance
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
            self.cf.load_file(self.scenario, plot=False,
                              vessel_model=self.vessel_model)
            self.Vbg = self.cf.Vbg
            self.dVbg = self.cf.dVbg

    def initalize_switches(self):
        self.capacitor_empty = False
        self.capacitor_discharge = False
        self.pulse_hold = False

    def get_step(self):
        if self.capacitor_discharge:
            dt = 1e-4
        elif self.pulse_hold:
            dt = 5e-4
        else:
            dt = 1e-4
        return dt

    def dIdt(self, t, I):
        Hcap = I[0]  # capacitor intergral current
        Icap = I[1]  # capacitor charge current
        Ivec = I[2:2+self.ncoil]
        Vbg = self.Vbg(t)
        if self.Ip_scale != 1:  # reduced plasma current opperation
            Vbg *= self.Ip_scale
        Vbg[0] *= self.nturn/4
        Vbg[1:3] /= 4  # jacket turns
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
            try:
                dIvec = np.dot(self.Minv, - self.R*Ivec - Hvec/self.C + Vbg)
            except ValueError:
                print(np.shape(self.Minv), np.shape(self.R), np.shape(Ivec),
                      np.shape(Hvec), np.shape(self.C), np.shape(Vbg))

        else:  # capacitor charging
            dHcap = self.Vo / self.Rcap - Hcap / (self.Rcap * self.C[0])
            dIcap = -Icap / (self.Rcap * self.C[0])
            if self.pulse_hold:
                k = 5.0  # power supply gain
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
            if self.op('ge', Iode[2], self.Itrip) \
                    or isclose(Iode[2], self.Itrip) \
                    or self.op('lt', dI[2], 0):  # peak current
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

    def intergrate(self, t_end):
        #  [Hcap, Icap, Ivec]
        Iode_o = np.zeros(2 + self.ncoil)
        Iode_o[0] = self.C[0] * self.Vo  # capacitor fully charged
        Iode_o[2] = self.Io  # vs3 circuit inital current
        to = 0 if self.pulse_phase < 0 else -self.pulse_phase
        self.ode.set_initial_value(Iode_o, t=to)
        self.set_switches(to, Iode_o)
        while self.ode.successful() and self.ode.t < t_end:
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
            dt_discharge = self.data['dt_discharge'][0]
            self.initalize(scenario=scenario,
                           dt_discharge=dt_discharge,
                           t_pulse=t_pulse)

    def solve(self, t_end=None, plot=False, **kwargs):
        self.initalize(**kwargs)
        if t_end is None:
            try:
                t_end = self.cf.t[-1]
            except AttributeError:
                t_end = self.tpo + self.t_pulse + 200e-3
        self.set_dt_discharge()
        self.intergrate(t_end)
        self.packdata()
        if plot:
            self.plot()
            self.plot_current()
        return self.Ivec

    def formatdata(self):
        for var in ['t', 'Ivec', 'Vps', 'Vcap', 'Icap']:
            self.data[var] = np.array(self.data[var])

    def packdata(self):
        self.formatdata()
        self.data['Ivec'][:, 1:3] /= 4  # jacket per turn current
        self.Ivec = interp1d(self.data['t'], self.data['Ivec'], axis=0,
                             fill_value=(self.data['Ivec'][0],
                                         self.data['Ivec'][-1]),
                             bounds_error=False)

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

    def plot_current(self, ax=None, title=True):
        if ax is None:
            ax = plt.subplots(1, 1, sharex=True)[1]
        if self.vessel:
            colors = OrderedDict()
            colors['lower_vvo'] = 'C1'
            colors['lower_vv1'] = 'C2'
            colors['lower_trs'] = 'C3'
            colors['lower_jacket'] = 'C6'
            colors['upper_vvo'] = 'C4'
            colors['upper_vv1'] = 'C5'
            colors['upper_jacket'] = 'C7'
            colors['vv_vvo'] = 'C4'
            colors['vv_vv1'] = 'C5'
            colors['trs_trs'] = 'C3'

            coils = list(self.coilset['coil'].keys())[16:]
            for i, coil in enumerate(coils):
                for c in colors:
                    if c in coil:
                        color = colors[c]
                        break
                index = coils.index(coil)
                try:
                    ax.plot(1e3*self.data['t'],
                            1e-3*self.data['Ivec'][:, index+3], color=color)
                except IndexError:
                    pass
            lines, labels = [], []
            for c in colors:
                lines.append(Line2D([0], [0], color=colors[c]))
                labels.append(c)
            ax.legend(lines, labels)

        ax.plot(1e3*self.data['t'], 1e-3*self.data['Ivec'][:, 0], '-C0')
        try:
            ax.plot(1e3*self.data['t'], 1e-3*self.data['Ivec'][:, 1], '-C6')
            ax.plot(1e3*self.data['t'], 1e-3*self.data['Ivec'][:, 2], '-C7')
        except IndexError:
            pass
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
        if title:
            plt.title('t pulse {:1.2f}s'.format(self.t_pulse))

    def LTC(self):
        path = os.path.join(class_dir(nep), '../Data/LTC/')
        points = data_load(path, 'VS3_discharge_main_report',
                           date='2018_06_25')[0]
        points = data_load(path, 'VS3_current', date='2018_03_15')[0]
        td, Id = points[0]['x'], points[0]['y']  # jacket + vessel
        plt.plot(1e3*td, -1e-3*Id, '--')
        plt.xlim([0, 80])

    def set_referance(self, ncoil=2):
        # store referance profile
        self.referance = {'t': np.copy(self.data['t']),
                          'Ic': np.copy(self.data['Ivec'][:, 0])}
        self.referance['fun'] = interp1d(self.referance['t'],
                                         self.referance['Ic'])
        self.referance['to'] = np.linspace(self.referance['t'][0],
                                           self.referance['t'][-1], 350)
        self.referance['Ico'] = self.referance['fun'](self.referance['to'])

        self.ncoil = ncoil
        self.M = self.M[:self.ncoil, :self.ncoil]
        self.R = self.R[:self.ncoil]
        self.C = self.C[:self.ncoil]

    def reduce(self):
        # self-inductance, mutual, resistance
        if self.ncoil == 2:
            xo = [6.26306679e-05, 0.722, 5.97155991e-04]
        elif self.ncoil == 3:
            xo = [2.49423821e-04, 3.42989379e-05, 5.57050300e-01,
                  1.00965866e-01, 5.72307931e-01, 1.13850869e-03,
                  1.19446487e-03]

        res = minimize(self.fit, xo, method='Nelder-Mead',
                       options={'fatol': 0.01})

        print(res)
        ps.fit(res.x, plot=True)
        '''
        ax = plt.subplots(1, 1)[1]
        ax.plot(1e3*self.referance['t'], 1e-3*self.referance['Ic'], '-C0',
                label='VS3 referance')
        ax.plot(1e3*self.data['t'], 1e-3*self.data['Ivec'][:, 0], '-C3',
                label='VS3 reduced model')
        ax.plot(1e3*self.data['t'], 1e-3*self.data['Ivec'][:, 1:], '-C7',
                label='dummy coil')
        ax.set_xlabel('$t$ ms')
        ax.set_ylabel('$I$ kA')
        plt.despine()
        plt.legend()
        '''

    def fit(self, x, plot=False):
        x = abs(x)  # insure positive input
        if len(x) == 3:
            x[1:2][x[1:2] > 0.99] = 0.99
            self.M[1, 1] = x[0]  # secondary coil self inductance
            # cross-inductance
            self.M[0, 1] = x[1]*np.sqrt(self.M[0, 0] * self.M[1, 1])
            self.M[1, 0] = self.M[0, 1]  # symetric
            self.R[1] = x[2]  # secondary coil resistance
        elif len(x) == 7:
            x[2:5][x[2:5] > 0.95] = 0.95
            self.M[1, 1] = x[0]
            self.M[2, 2] = x[1]
            self.M[0, 1] = x[2]*np.sqrt(self.M[0, 0] * self.M[1, 1])
            self.M[1, 0] = self.M[0, 1]  # symetric
            self.M[1, 2] = x[3]*np.sqrt(self.M[1, 1] * self.M[2, 2])
            self.M[2, 1] = self.M[1, 2]  # symetric
            self.M[0, 2] = x[4]*np.sqrt(self.M[0, 0] * self.M[2, 2])
            self.M[2, 0] = self.M[0, 2]  # symetric
            self.R[1:] = x[5:]  # secondary coil resistance
        self.Minv = np.linalg.inv(self.M)  # inverse
        self.initalize()
        self.intergrate(self.referance['to'][-1])
        self.formatdata()
        self.referance['Ifit'] = interp1d(
                self.data['t'], self.data['Ivec'][:, 0],
                bounds_error=False,
                fill_value=(self.data['Ivec'][0, 0],
                            self.data['Ivec'][-1, 0]))(self.referance['to'])
        err = 1e-3*np.sqrt(np.mean((self.referance['Ifit'] -
                                    self.referance['Ico'])**2))

        if plot:
            plt.figure()
            plt.plot(1e3*self.referance['to'], 1e-3*self.referance['Ico'],
                     label='referance')
            plt.plot(1e3*self.referance['to'],
                     1e-3*self.referance['Ifit'], 'C3', label='fit')
            for i in range(self.ncoil-1):
                plt.plot(1e3*self.data['t'],
                         1e-3*self.data['Ivec'][:, i+1], 'C7',
                         label=f'dummy {i+1}')
            plt.xlabel('$t$ ms')
            plt.ylabel('$I$ kA')
            plt.despine()
            plt.legend()
        return err

    '''
    def reduced_coil_model(inwork)
        first solve model to create fit data

        ps.set_referance(ncoil=2)
        ps.reduce()

        if ps.ncoil == 2:
            # impulse
            x = np.array([5.09661450e-05, 7.28185583e-01, 6.66816036e-04])
            # discharge
            # x = np.array([6.26306679e-05, 7.22409337e-01, 5.97155991e-04])
        elif ps.ncoil == 3:
            x = np.array([ 2.49423821e-04, 3.42989379e-05, 5.57050300e-01,
                          1.00965866e-01, 5.72307931e-01, 1.13850869e-03,
                          1.19446487e-03])

        ps.fit(x, plot=True)

        tc = timeconstant(ps.referance['to'], ps.referance['Ifit'],
                          trim_fraction=0.8)
        tau = tc.nfit(ps.ncoil, trim_fraction=0, plot=True)[1]
        print(tau*1e3)

        R = np.dot(np.ones((ps.ncoil, 1)), ps.R.reshape((1, -1)))
        tau = np.linalg.eigvals(ps.M/R)
        print(tau*1e3)
    '''


if __name__ == '__main__':
    ps = power_supply(nturn=4, vessel=True, scenario=0, code='Nova',
                      Ip_scale=15/15, read_txt=False, vessel_model='full')
    ps.solve(Io=0, sign=-1, t_pulse=0, origin='peak',
             impulse=False, plot=True)









