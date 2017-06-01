import numpy as np
import amigo.geom as geom
from nova.loops import Profile
from nova.shape import Shape
import pylab as pl
from warnings import warn
import datetime
import pickle
from nova.streamfunction import SF
from nova.config import Setup
from collections import OrderedDict
import seaborn as sns


class divertor(object):

    def __init__(self, sf, setup, flux_conformal=False, debug=False):
        self.debug = debug
        self.sf = sf
        self.targets = setup.targets  # target definition (graze, L2D, etc)
        self.setup = setup
        self.div_ex = setup.firstwall['div_ex']
        self.segment = {}
        if flux_conformal:
            self.Xfw, self.Zfw, self.psi_fw = sf.firstwall_loop(
                psi_n=setup.firstwall['psi_n'])
        else:
            self.Xfw, self.Zfw, self.psi_fw = sf.firstwall_loop(
                dx=setup.firstwall['dXfw'])

    def set_target(self, leg, **kwargs):
        if leg not in self.targets:
            self.targets[leg] = {}
        for key in self.targets['default']:
            if key in kwargs:
                self.targets[leg][key] = kwargs[key]  # update
            elif key not in self.targets[leg]:  # prevent overwrite
                self.targets[leg][key] = self.targets['default'][key]

    def select_layer(self, leg, layer_index=[0, -1]):  # choose SOL layer
        sol, Xi = [], []
        for layer in layer_index:
            x, z = self.sf.snip(leg, layer, L2D=self.targets[leg]['L2D'])
            sol.append((x, z))
            Xi.append(self.sf.expansion([x[-1]], [z[-1]]))
        index = np.argmin(Xi)  # min expansion / min graze
        return sol[index][0], sol[index][1]

    def place(self, **kwargs):
        self.sf.sol(update=True, plot=False, debug=self.debug)
        for leg in list(self.sf.legs)[2:]:
            self.set_target(leg)
            Xsol, Zsol = self.select_layer(leg)
            Xo, Zo = Xsol[-1], Zsol[-1]
            Flip = [-1, 1]
            Direction = [1, -1]
            Theta_end = [0, np.pi]
            theta_sign = 1
            if self.sf.Xloc == 'upper':
                theta_sign *= -1
                Direction = Direction[::-1]
            if 'inner' in leg:
                psi_plasma = self.psi_fw[1]
            else:
                psi_plasma = self.psi_fw[0]
            dpsi = self.psi_fw[1] - self.sf.Xpsi
            dpsi = self.psi_fw[1] - self.sf.Xpsi
            Phi_target = [psi_plasma, self.sf.Xpsi - self.div_ex * dpsi]
            if leg is 'inner1' or leg is 'outer2':
                Phi_target[0] = self.sf.Xpsi + self.div_ex * dpsi
            if self.targets[leg]['open']:
                theta_sign *= -1
                Direction = Direction[::-1]
                Theta_end = Theta_end[::-1]
            if 'outer' in leg:
                Direction = Direction[::-1]
                theta_sign *= -1
            if leg is 'inner1' or leg is 'outer2':
                Theta_end = Theta_end[::-1]

            self.targets[leg]['X'] = np.array([])
            self.targets[leg]['Z'] = np.array([])
            dPlate = self.targets[leg]['dPlate']
            for flip, direction, theta_end, psi_target\
                    in zip(Flip, Direction, Theta_end, Phi_target):
                X, Z = self.match_psi(Xo, Zo, direction, theta_end, theta_sign,
                                      psi_target, self.targets[leg]['graze'],
                                      dPlate, leg, debug=self.debug)
                self.targets[leg]['X'] = np.append(self.targets[leg]['X'],
                                                   X[::flip])
                self.targets[leg]['Z'] = np.append(self.targets[leg]['Z'],
                                                   Z[::flip])
            if leg is 'outer':
                self.targets[leg]['X'] = self.targets[leg]['X'][::-1]
                self.targets[leg]['Z'] = self.targets[leg]['Z'][::-1]

        Xb, Zb = np.array([]), np.array([])
        if self.sf.nleg == 6:  # SF
            Xb = np.append(Xb, self.targets['inner2']['X'][1:])
            Zb = np.append(Zb, self.targets['inner2']['Z'][1:])
            x, z = self.connect(self.sf.Xpsi -
                                self.div_ex * dpsi,
                                ['inner2', 'inner1'], [-1, -1])
            Xb, Zb = self.append(Xb, Zb, x, z)
            Xb = np.append(Xb, self.targets['inner1']['X'][::-1])
            Zb = np.append(Zb, self.targets['inner1']['Z'][::-1])
            x, z = self.connect(self.sf.Xpsi +
                                self.div_ex * dpsi,
                                ['inner1', 'outer2'], [0, 0])
            Xb, Zb = self.append(Xb, Zb, x, z)
            Xb = np.append(Xb, self.targets['outer2']['X'][1:])
            Zb = np.append(Zb, self.targets['outer2']['Z'][1:])
            x, z = self.connect(self.sf.Xpsi -
                                self.div_ex * dpsi,
                                ['outer2', 'outer1'], [-1, -1])
            Xb, Zb = self.append(Xb, Zb, x, z)
            Xb = np.append(Xb, self.targets['outer1']['X'][::-1])
            Zb = np.append(Zb, self.targets['outer1']['Z'][::-1])
            self.segment['divertor'] = {'x': Xb, 'z': Zb}  # store diveror
        else:
            Xb = np.append(Xb, self.targets['inner']['X'][1:])
            Zb = np.append(Zb, self.targets['inner']['Z'][1:])
            x, z = self.connect(self.sf.Xpsi -
                                self.div_ex * dpsi,
                                ['inner', 'outer'], [-1, 0])
            Xb, Zb = self.append(Xb, Zb, x, z)
            self.segment['divertor'] = {'x': Xb, 'z': Zb}  # store diveror
            Xb = np.append(Xb, self.targets['outer']['X'][1:])
            Zb = np.append(Zb, self.targets['outer']['Z'][1:])
            self.segment['divertor'] = {'x': Xb, 'z': Zb}  # store diveror

    def intersect(self, x, z, xd, zd, offset=0, s=0):
        xd, zd = np.copy(xd), np.copy(zd)
        xd, zd = geom.inloop(x, z, xd, zd, side='out')  # external to fw
        xd, zd = geom.cut(xd, zd)
        if offset != 0:
            xd, zd = geom.rzInterp(xd, zd, npoints=100)
            xd, zd = geom.offset(xd, zd, offset, min_steps=10, s=s)

        istart = np.argmin((x - xd[-1])**2 + (z - zd[-1])**2)  # connect to fw
        iend = np.argmin((x - xd[0])**2 + (z - zd[0])**2)
        index = {'istart': istart, 'iend': iend}
        xd = np.append(np.append(x[iend], xd), x[istart])
        zd = np.append(np.append(z[iend], zd), z[istart])
        if offset != 0:
            xd, zd = geom.rzInterp(xd, zd, npoints=100)
        return xd, zd, index

    def join(self, main_chamber):
        x, z = main_chamber.draw(npoints=1000)
        istart = np.argmin(
            (x - self.sf.Xpoint[0])**2 + (z - self.sf.Xpoint[1])**2)
        x = np.append(x[istart:], x[:istart + 1])
        z = np.append(z[istart:], z[:istart + 1])
        x, z = geom.unique(x, z)[:2]
        self.segment['inner_loop'] = {'x': x, 'z': z}
        xd, zd = self.segment['divertor']['x'], self.segment['divertor']['z']
        xd, zd = geom.unique(xd, zd)[:2]
        if self.sf.Xloc == 'lower':
            zindex = zd <= self.sf.Xpoint[1] + 0.5 * (self.sf.mo[1] -
                                                      self.sf.Xpoint[1])
        else:
            zindex = zd >= self.sf.Xpoint[1] + 0.5 * (self.sf.mo[1] -
                                                      self.sf.Xpoint[1])
        xd, zd = xd[zindex], zd[zindex]  # remove upper points
        xd, zd = geom.rzInterp(xd, zd)  # resample

        # divertor inner wall
        xd, zd, i_in = self.intersect(x, z, xd, zd, offset=0)
        self.segment['divertor_inner'] = {'x': xd, 'z': zd}
        self.segment['first_wall'] = self.join_loops(x, z, xd, zd, i_in)[1]

        # divertor outer wall
        xd, zd, i_out = self.intersect(x, z, xd, zd, s=0.075,
                                       offset=self.setup.firstwall['dx_div'])
        self.segment['divertor_outer'] = {'x': xd, 'z': zd}

        xloop = np.append(self.segment['divertor_inner']['x'],
                          x[i_in['istart']:i_out['istart']])
        zloop = np.append(self.segment['divertor_inner']['z'],
                          z[i_in['istart']:i_out['istart']])
        xloop = np.append(xloop, self.segment['divertor_outer']['x'][::-1])
        zloop = np.append(zloop, self.segment['divertor_outer']['z'][::-1])
        xloop = np.append(xloop, x[i_out['iend']:i_in['iend']])
        zloop = np.append(zloop, z[i_out['iend']:i_in['iend']])
        xloop = np.append(xloop, xloop[0])
        zloop = np.append(zloop, zloop[0])
        self.segment['divertor'] = {'x': xloop, 'z': zloop}

        # gap to blanket
        xd, zd, i_gap = self.intersect(x, z, xd, zd,
                                       offset=self.setup.firstwall['bb_gap'])
        self.segment['divertor_gap'] = {'x': xd, 'z': zd}
        chamber, loop = self.join_loops(x, z, xd, zd, i_gap)
        self.segment['blanket_inner'] = chamber
        self.segment['vessel_gap'] = loop
        return self.segment

    def join_loops(self, x, z, xd, zd, index):
        if self.sf.Xloc == 'lower':
            xc = x[index['istart']:index['iend']]
            zc = z[index['istart']:index['iend']]
        else:
            xc = x[index['iend']:index['istart']][::-1]
            zc = z[index['iend']:index['istart']][::-1]
        xloop = np.append(np.append(xd, xc), xd[0])
        zloop = np.append(np.append(zd, zc), zd[0])
        chamber = {'x': xc, 'z': zc}
        loop = {'x': xloop, 'z': zloop}
        return chamber, loop

    def connect(self, psi, target_pair, ends, loop=[]):
        gap = []
        if loop:
            x, z = loop
        else:
            psi_line = self.sf.get_contour([psi])[0]
            for line in psi_line:
                x, z = line[:, 0], line[:, 1]
                gap.append(np.min((self.targets[target_pair[0]]['X'][ends[0]] -
                                   x)**2 +
                                  (self.targets[target_pair[0]]['Z'][ends[0]] -
                                  z)**2))
            select = np.argmin(gap)
            line = psi_line[select]
            x, z = line[:, 0], line[:, 1]
        index = np.zeros(2, dtype=int)
        index[0] = np.argmin((self.targets[target_pair[0]]['X'][ends[0]] -
                             x)**2 +
                             (self.targets[target_pair[0]]['Z'][ends[0]] -
                             z)**2)
        index[1] = np.argmin((self.targets[target_pair[1]]['X'][ends[1]] -
                             x)**2 +
                             (self.targets[target_pair[1]]['Z'][ends[1]] -
                             z)**2)
        if index[0] > index[1]:
            index = index[::-1]
        x, z = x[index[0]:index[1] + 1], z[index[0]:index[1] + 1]
        return x, z

    def match_psi(self, Xo, Zo, direction, theta_end, theta_sign, phi_target,
                  graze, dPlate, leg, debug=False):
        color = sns.color_palette('Set2', 2)
        gain = 0.25  # 0.25
        Nmax = 500
        Lo = [5.0, 0.0015]  # [blend,turn]  5,0.015
        x2m = [-1, -1]  # ramp to step (+ive-lead, -ive-lag ramp==1, step==inf)
        Nplate = 1  # number of target plate divisions (1==flat)
        L = Lo[0] if theta_end == 0 else Lo[1]
        Lsead = L
        flag = 0
        for i in range(Nmax):
            X, Z, phi = self.blend_target(Xo, Zo, dPlate, L, direction,
                                          theta_end, theta_sign, graze, x2m,
                                          Nplate)
            L -= gain * (phi_target - phi)
            if debug:
                pl.plot(X, Z, color=color[0], lw=1)
            if np.abs(phi_target - phi) < 1e-4:
                if debug:
                    pl.plot(X, Z, 'x', color=color[1], lw=3)
                break
            if L < 0:
                L = 1
                gain *= -1
            if i == Nmax - 1 or L > 15:
                print(leg, 'dir', direction, 'phi target convergence error')
                print('traget', phi_target, 'phi', phi)
                print('Nmax', i + 1, 'L', L, 'Lo', Lsead)
                if flag == 0:
                    break
                    gain *= -1  # reverse gain
                    flag = 1
                else:
                    break
        return X, Z

    def blend_target(self, Xo, Zo, dPlate, L, direction, theta_end, theta_sign,
                     graze, x2m, Nplate):
        x2s = x2m[0] if theta_end == 0 else x2m[1]
        dL = 0.1 if theta_end == 0 else 0.05  # 0.005,0.005
        X, Z = np.array([Xo]), np.array([Zo])
        X, Z = self.extend_target(X, Z, dPlate / (2 * Nplate), Nplate, x2s,
                                  theta_end, theta_sign,
                                  direction, graze, False,
                                  target=True)  # constant graze
        Ninterp = int(dPlate / (2 * dL))
        if Ninterp < 2:
            Ninterp = 2
        X, Z = geom.rzInterp(X, Z, Ninterp)
        # update graze
        graze = self.sf.get_graze([X[-1], Z[-1]], [X[-1] - X[-2],
                                  Z[-1] - Z[-2]])
        N = np.int(L / dL + 1)
        if N < 30:
            N = 30
        dL = L / (N - 1)
        target_angle = np.arctan2((Z[-1] - Z[-2]), (X[-1] - X[-2]))
        Xi = self.sf.expansion([X[-1]], [Z[-1]])
        theta = self.sf.strike_point(Xi, graze)
        B = direction * self.sf.Bpoint((X[-1], Z[-1]))
        Bhat = geom.rotate_vector2D(B, theta_sign * theta)
        trans_angle = np.arctan2(Bhat[1], Bhat[0])
        if abs(target_angle - trans_angle) > 0.01 * np.pi:
            accute = True
        else:
            accute = False
        X, Z = self.extend_target(X, Z, dL, N, x2s, theta_end, theta_sign,
                                  direction, graze, accute)  # transition graze
        phi = self.sf.Ppoint([X[-1], Z[-1]])
        return X, Z, phi

    def extend_target(self, X, Z, dL, N, x2s, theta_end, theta_sign, direction,
                      graze, accute, target=False):
        for i in range(N):
            if target:
                Lhat = 0
            else:
                Lhat = i / (N - 1)
                if x2s < 0:  # delayed transtion
                    Lhat = Lhat**np.abs(x2s)
                else:  # expedient transition
                    Lhat = Lhat**(1 / x2s)
            Xi = self.sf.expansion([X[-1]], [Z[-1]])
            theta = self.sf.strike_point(Xi, graze)
            if accute:
                theta = np.pi - theta
            theta = Lhat * theta_end + (1 - Lhat) * theta
            B = direction * self.sf.Bpoint((X[-1], Z[-1]))
            Bhat = geom.rotate_vector2D(B, theta_sign * theta)
            X = np.append(X, X[-1] + dL * Bhat[0])
            Z = np.append(Z, Z[-1] + dL * Bhat[1])
        return X, Z

    def append(self, X, Z, x, z):
        dx = np.zeros(2)
        for i, end in enumerate([0, -1]):
            dx[i] = (X[-1] - x[end])**2 + (Z[-1] - z[end])**2
        if dx[1] < dx[0]:
            x, z = x[::-1], z[::-1]
        return np.append(X, x[1:-1]), np.append(Z, z[1:-1])


class main_chamber(object):

    def __init__(self, name, **kwargs):
        self.name = name
        self.set_filename(**kwargs)
        self.initalise_loop()

    def initalise_loop(self):
        self.profile = Profile(self.filename, family='S', part='chamber',
                               npoints=200)
        self.shp = Shape(self.profile, objective='L')
        self.set_bounds()

    def set_bounds(self):
        self.shp.loop.adjust_xo('upper', lb=0.7)
        self.shp.loop.adjust_xo('top', lb=0.05, ub=0.75)
        self.shp.loop.adjust_xo('lower', lb=0.7)
        self.shp.loop.adjust_xo('bottom', lb=0.05, ub=0.75)
        self.shp.loop.adjust_xo('l', lb=0.8, ub=1.5)
        self.shp.loop.adjust_xo('tilt', lb=-25, ub=25)
        # self.shp.loop.remove_oppvar('flat')
        # self.shp.loop.remove_oppvar('tilt')

    def date(self, verbose=True):
        today = datetime.date.today().strftime('%Y_%m_%d')
        if verbose:
            print(today)
        return today

    def set_filename(self, update=False, **kwargs):
        today = self.date(verbose=False)
        if update:  # use today's date
            date_str = today
        else:
            date_str = kwargs.get('date', today)
        self.filename = '{}_{}'.format(date_str, self.name)  # chamber name

    def generate(self, eq_names, dx=0.225, psi_n=1.07,
                 flux_fit=False, symetric=False, plot=False,
                 plot_bounds=False, verbose=False):
        self.set_filename(update=True)  # update date in filename
        self.profile.loop.reset_oppvar(symetric)  # reset loop oppvar
        self.set_bounds()
        self.config = {'dx': dx, 'psi_n': psi_n,
                       'flux_fit': flux_fit, 'Nsub': 100}
        self.config['eqdsk'] = []
        sf_list = self.load_sf(eq_names)
        for sf in sf_list:  # convert input to list
            self.add_bound(sf)
        # self.shp.add_interior(r_gap=0.001)  # add interior bound
        self.shp.minimise(verbose=verbose)
        self.write()  # append config data to loop pickle
        if plot:
            self.plot_chamber()
        if plot_bounds:
            self.shp.plot_bounds()

    def load_sf(self, eq_names):
        sf_dict, sf_list = OrderedDict(), []
        for configuration in eq_names:
            setup = Setup(configuration)
            sf = SF(setup.filename)
            sf_dict[configuration] = sf.filename.split('/')[-1]
            sf_list.append(sf)
        self.config['eqdsk'] = sf_dict
        return sf_list

    def write(self):  # overwrite loop_dict + add extra chamber fields
        with open(self.profile.dataname, 'wb') as output:
            pickle.dump(self.profile.loop_dict, output, -1)
            pickle.dump(self.config, output, -1)
            pickle.dump(self.shp.bound, output, -1)  # boundary points
            pickle.dump(self.shp.bindex, output, -1)  # boundary index

    def load_data(self, plot=False):
        try:
            with open(self.profile.dataname, 'rb') as input:
                self.profile.loop_dict = pickle.load(input)
                self.config = pickle.load(input)
                self.shp.bound = pickle.load(input)
                self.shp.bindex = pickle.load(input)
        except:
            print(self.profile.dataname)
            errtxt = 'boundary information not found'
            raise ValueError(errtxt)
        if plot:
            self.plot_chamber()

    def plot_chamber(self):
        self.shp.loop.plot()
        self.shp.plot_bounds()
        x, z = self.draw()
        pl.plot(x, z)

    def add_bound(self, sf):
        xpl, zpl = sf.get_offset(self.config['dx'], Nsub=self.config['Nsub'])
        # vessel inner bounds
        self.shp.add_bound({'x': xpl, 'z': zpl}, 'internal')
        Xpoint = sf.Xpoint_array[:, 0]  # select lower
        self.shp.add_bound({'x': Xpoint[0] + 0.12 * sf.shape['a'],
                            'z': Xpoint[1]}, 'internal')
        self.shp.add_bound({'x': Xpoint[0],
                            'z': Xpoint[1] - 0.01 * sf.shape['a']}, 'internal')

        if self.config['flux_fit']:  # add flux boundary points
            sf.get_LFP()  # get low feild point
            rflux, zflux = sf.first_wall_psi(psi_n=self.config['psi_n'],
                                             trim=False)[:2]

            rflux, zflux = sf.midplane_loop(rflux, zflux)
            rflux, zflux = geom.order(rflux, zflux)
            istop = next((i for i in range(len(zflux))
                          if zflux[i] < sf.LFPz), -1)
            rflux, zflux = rflux[:istop], zflux[:istop]
            dL = np.diff(geom.length(rflux, zflux))
            if np.max(dL) > 3 * np.median(dL) or \
                    np.argmax(zflux) == len(zflux) - 1:
                wtxt = '\n\nOpen feild line detected\n'
                wtxt += 'disabling flux fit for '
                wtxt += '{:1.1f}% psi_n \n'.format(1e2 * self.config['psi_n'])
                wtxt += 'configuration: ' + sf.filename + '\n'
                warn(wtxt)
            else:  # add flux_fit bounds
                rflux, zflux = geom.rzSLine(rflux, zflux,
                                            int(self.config['Nsub'] / 2))
                self.shp.add_bound({'x': rflux, 'z': zflux}, 'internal')

    def draw(self, npoints=250):
        x = self.profile.loop.draw(npoints=npoints)
        x, z = x['x'], x['z']
        x, z = geom.order(x, z, anti=True)
        return x, z

if __name__ == '__main__':

    print('usage example in nova.radial_build')
