import numpy as np
import pylab as pl
import seaborn as sns
import amigo.geom as geom
from nova.loops import Profile
from nova.shape import Shape
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.optimize import minimize
import collections
from amigo.IO import trim_dir
import json
from nova.firstwall import divertor
colors = sns.color_palette('Paired', 12)


class RB(object):
    def __init__(self, sf, setup, npoints=500):
        self.setup = setup
        self.sf = sf
        self.npoints = npoints
        self.dataname = setup.configuration
        self.datadir = trim_dir('../../../Data/')
        self.segment = {}  # store section segments (divertox,fw,blanket...)

    def generate(self, mc, mode='calc', color=0.6 * np.ones(3),
                 plot=True, debug=False, symetric=False, DN=False):
        self.main_chamber = mc.filename
        if mode == 'eqdsk':  # read first wall from eqdsk
            self.Xb, self.Zb = self.sf.xlim, self.sf.ylim
        elif mode == 'calc':
            div = divertor(self.sf, self.setup, debug=debug)
            if DN:  # double null
                self.sf.get_Xpsi(select='upper')  # upper X-point
                div.place(debug=debug)
                self.segment = div.join(mc)
                xu = self.segment['first_wall']['x']
                zu = self.segment['first_wall']['z']
                blanket_u = self.place_blanket(select='upper', store=False)[0]
                self.sf.get_Xpsi(select='lower')  # lower X-point
                div.place(debug=debug)
                self.segment = div.join(mc)
                xl = self.segment['first_wall']['x']
                zl = self.segment['first_wall']['z']
                blanket_l = self.place_blanket(select='lower', store=False)[0]
                x, z = self.upper_lower(
                    xu, zu, xl, zl, Zjoin=self.sf.Mpoint[1])
                self.segment['first_wall'] = {'x': x, 'z': z}
                xfw, zfw = self.upper_lower(blanket_u[0]['x'][::-1],
                                            blanket_u[0]['z'][::-1],
                                            blanket_l[0]['x'],
                                            blanket_l[0]['z'],
                                            Zjoin=self.sf.Mpoint[1])
                self.blanket = wrap({'x': x, 'z': z}, {'x': xfw, 'z': zfw})
                for loop in ['inner', 'outer']:
                    self.blanket.sort_z(loop, select='lower')
                bfill = self.blanket.fill(plot=False)
                self.segment['blanket_fw'] = bfill[0]
                self.segment['blanket'] = bfill[1]
            else:
                self.sf.get_Xpsi(select='lower')  # lower X-point
                div.place(debug=debug)
                self.segment = div.join(mc)
                self.blanket = self.place_blanket(select='lower')[-1]
            self.vessel = self.vessel_fill()
            self.Xb = self.segment['first_wall']['x']
            self.Zb = self.segment['first_wall']['z']
            xbl, zbl = self.blanket.get_segment('outer')
            self.segment['blanket_outer'] = {'x': xbl, 'z': zbl}
        else:
            errtxt = 'set input mode \'eqdsk\',\'calc\''
            raise ValueError(errtxt)

        if mode != 'eqdsk':  # update eqdsk
            self.sf.xlim = self.Xb
            self.sf.ylim = self.Zb
            self.sf.nlim = len(self.sf.xlim)

        if plot:
            pl.plot(self.segment['first_wall']['x'],
                    self.segment['first_wall']['z'],
                    lw=1.75, color=0.5 * np.ones(3))
            self.blanket.fill(plot=True, color=colors[0])
            self.vessel.fill(plot=True, color=colors[1])
            pl.axis('equal')
            pl.axis('off')

    def upper_lower(self, Xu, Zu, Xl, Zl, Zjoin=0):
        u_index = np.arange(len(Xu), dtype='int')
        iu_lower = u_index[Zu < Zjoin]
        iu_out, iu_in = iu_lower[0] - 1, iu_lower[-1] + 1
        l_index = np.arange(len(Xl), dtype='int')
        il_upper = l_index[Zl > Zjoin]
        il_out, il_in = il_upper[0] - 1, il_upper[-1] + 1
        X = np.append(Xl[:il_out], Xu[:iu_out][::-1])
        X = np.append(X, Xu[iu_in:][::-1])
        X = np.append(X, Xl[il_in:])
        Z = np.append(Zl[:il_out], Zu[:iu_out][::-1])
        Z = np.append(Z, Zu[iu_in:][::-1])
        Z = np.append(Z, Zl[il_in:])
        return X, Z

    def place_blanket(self, select='upper', store=True, plot=False):
        blanket = wrap(self.segment['first_wall'], self.segment['inner_loop'])
        blanket.sort_z('inner', select=select)
        blanket.offset('outer', self.setup.build['BB'],
                       ref_o=3 / 10 * np.pi, dref=np.pi / 3)
        bfill = blanket.fill(plot=plot, color=colors[0])
        if store:
            self.segment['blanket_fw'] = bfill[0]
            self.segment['blanket'] = bfill[1]
        return bfill, blanket

    def vessel_fill(self, gap=True):
        x, z = self.segment['blanket_fw']['x'], self.segment['blanket_fw']['z']
        loop = Loop(x, z)
        x, z = loop.fill(dt=0.05)
        xb = np.append(x, x[0])
        zb = np.append(z, z[0])
        profile = Profile(self.setup.configuration, family='S', part='vv',
                          npoints=400, read_write=False)
        shp = Shape(profile, objective='L')
        shp.loop.adjust_xo('upper', lb=0.6)
        shp.loop.adjust_xo('lower', lb=0.6)
        shp.loop.adjust_xo('l', lb=0.6)
        # shp.loop.remove_oppvar('flat')
        x, z = geom.rzSLine(xb, zb, 200)  # sub-sample
        xup, zup = x[z > self.sf.Xpoint[1]], z[z > self.sf.Xpoint[1]]
        shp.add_bound({'x': xup, 'z': zup}, 'internal')  # vessel inner bounds
        xd, zd = x[z < self.sf.Xpoint[1]], z[z < self.sf.Xpoint[1]]
        xo, zo = geom.offset(xd, zd, 0.1)  # divertor offset
        shp.add_bound({'x': xd, 'z': zd}, 'internal')  # vessel inner bounds
        shp.add_bound({'x': xd, 'z': zd - 0.25},
                      'internal')  # gap below divertor
        # shp.plot_bounds()
        shp.minimise()
        # shp.loop.plot()
        x = profile.loop.draw()
        xin, zin = x['x'], x['z']
        loop = Loop(xin, zin)
        x, z = loop.fill(
            dt=self.setup.build['VV'], ref_o=2 / 8 * np.pi, dref=np.pi / 6)
        shp.clear_bound()
        shp.add_bound({'x': x, 'z': z}, 'internal')  # vessel outer bounds
        shp.minimise()
        x = profile.loop.draw()
        x, z = x['x'], x['z']
        if 'SX' in self.setup.configuration or gap is True:
            vv = wrap({'x': xin, 'z': zin}, {'x': x, 'z': z})
        else:
            vv = wrap({'x': xb, 'z': zb}, {'x': x, 'z': z})
        vv.sort_z('inner', select=self.sf.Xloc)
        vv.sort_z('outer', select=self.sf.Xloc)

        self.segment['vessel_inner'] = {'x': xin, 'z': zin}
        self.segment['vessel_outer'] = {'x': x, 'z': z}
        self.segment['vessel'] = vv.fill()[1]
        return vv

    def get_sol(self, plot=False):
        self.trim_sol(plot=plot)
        for leg in list(self.sf.legs)[2:]:
            L2D, L3D, Xsol, Zsol = self.sf.connection(leg, 0)
            Xo, Zo = Xsol[-1], Zsol[-1]
            L2Dedge, L3Dedge = self.sf.connection(leg, -1)[:2]
            if leg not in self.setup.targets:
                self.setup.targets[leg] = {}
            Xi = self.sf.expansion([Xo], [Zo])
            graze, theta = np.zeros(self.sf.Nsol), np.zeros(self.sf.Nsol)
            pannel = self.sf.legs[leg]['pannel']
            for i in range(self.sf.Nsol):
                xo = self.sf.legs[leg]['X'][i][-1]
                zo = self.sf.legs[leg]['Z'][i][-1]
                graze[i] = self.sf.get_graze((xo, zo), pannel[i])
                theta[i] = self.sf.strike_point(Xi, graze[i])
            self.setup.targets[leg]['graze_deg'] = graze * 180 / np.pi
            self.setup.targets[leg]['theta_deg'] = theta * 180 / np.pi
            self.setup.targets[leg]['L2Do'] = L2D
            self.setup.targets[leg]['L3Do'] = L3D
            self.setup.targets[leg]['L2Dedge'] = L2Dedge
            self.setup.targets[leg]['L3Dedge'] = L3Dedge
            self.setup.targets[leg]['Xo'] = Xo
            self.setup.targets[leg]['Zo'] = Zo
            self.setup.targets[leg]['Xsol'] = Xsol
            self.setup.targets[leg]['Zsol'] = Zsol

    def trim_sol(self, color='k', plot=True):
        self.sf.sol()
        color = sns.color_palette('Set2', self.sf.nleg + 5)
        # color = 'k'
        for c, leg in enumerate(self.sf.legs.keys()):
            if 'core' not in leg:
                Xsol = self.sf.legs[leg]['X']
                Zsol = self.sf.legs[leg]['Z']
                self.sf.legs[leg]['pannel'] = [[] for i in range(self.sf.Nsol)]
                for i in range(self.sf.Nsol):
                    if len(Xsol[i]) > 0:
                        X, Z = Xsol[i], Zsol[i]
                        for j in range(2):  # predict - correct
                            X, Z, pannel = self.trim(self.Xb, self.Zb, X, Z)
                        self.sf.legs[leg]['X'][i] = X  # update sf
                        self.sf.legs[leg]['Z'][i] = Z
                        self.sf.legs[leg]['pannel'][i] = pannel
                        if plot:
                            if color != 'k' and i > 0:
                                pl.plot(X, Z, '-', color=0.7 *
                                        np.ones(3))  # color[c+3]
                                # pl.plot(X,Z,'-',color=color[c+3])
                            elif color == 'k':
                                pl.plot(X, Z, '-', color='k', alpha=0.15)
                            else:
                                # pl.plot(X,Z,color=color[c])
                                pl.plot(X, Z, '--', color=[0.5, 0.5, 0.5])

    def crossed_lines(self, Xo, Zo, X1, Z1):
        index = np.zeros(2)
        dl = np.zeros(len(Xo))
        for i, (xo, zo) in enumerate(zip(Xo, Zo)):
            dl[i] = np.min((X1 - xo)**2 + (Z1 - zo)**2)
        index[0] = np.argmin(dl)
        index[1] = np.argmin((X1 - Xo[index[0]])**2 + (Z1 - Zo[index[0]])**2)
        return index

    def trim(self, Xloop, Zloop, X, Z):
        Xloop, Zloop = geom.order(Xloop, Zloop)
        L = geom.length(X, Z)
        index = np.append(np.diff(L) != 0, True)
        X, Z = X[index], Z[index]  # remove duplicates
        nXloop, nZloop, Xloop, Zloop = geom.normal(Xloop, Zloop)
        Xin, Zin = np.array([]), np.array([])
        for x, z in zip(X, Z):
            i = np.argmin((x - Xloop)**2 + (z - Zloop)**2)
            dx = [Xloop[i] - x, Zloop[i] - z]
            dn = [nXloop[i], nZloop[i]]
            if np.dot(dx, dn) > 0:
                Xin, Zin = np.append(Xin, x), np.append(Zin, z)
        i = np.argmin((Xin[-1] - X)**2 + (Zin[-1] - Z)**2)
        # extend past target
        Xin, Zin = X[:i + 2], Z[:i + 2]
        # sol crossing bndry
        i = np.argmin((Xin[-1] - X)**2 + (Zin[-1] - Z) ** 2)
        jo = np.argmin((X[i] - Xloop)**2 + (Z[i] - Zloop)**2)
        j = np.array([jo, jo + 1])
        s = np.array([X[i], Z[i]])
        ds = np.array([X[i] - X[i - 1], Z[i] - Z[i - 1]])
        b = np.array([Xloop[j[0]], Zloop[j[0]]])
        bstep, db = self.get_bstep(s, ds, b, Xloop, Zloop, j)
        if bstep < 0:
            j = np.array([jo - 1, jo])  # switch target pannel
            bstep, db = self.get_bstep(s, ds, b, Xloop, Zloop, j)
        step = np.cross(b - s, db) / np.cross(ds, db)
        intersect = s + step * ds  # predict - correct
        if step < 0:  # step back
            Xin, Zin = Xin[:-1], Zin[:-1]
        Xin, Zin = np.append(Xin, intersect[0]), np.append(Zin, intersect[1])
        return Xin, Zin, db

    def get_bstep(self, s, ds, b, Xloop, Zloop, j):
        db = np.array([Xloop[j[1]] - Xloop[j[0]], Zloop[j[1]] - Zloop[j[0]]])
        step = np.cross(b - s, ds) / np.cross(ds, db)
        return step, db

    def write_json(self, **kwargs):
        data = {}
        data['eqdsk'] = self.sf.eqdsk['name']
        data['main_chamber'] = self.main_chamber
        data['configuration'] = self.setup.configuration
        data['targets'] = {}
        for leg in self.setup.targets:  # store target data
            data['targets'][leg] = {}
            for name in self.setup.targets[leg]:
                packet = self.setup.targets[leg][name]
                if isinstance(packet, collections.Iterable):
                    if not isinstance(packet, list):
                        packet = packet.tolist()
                data['targets'][leg][name] = packet
        for loop in ['first_wall', 'divertor', 'blanket_inner',
                     'blanket_outer', 'vessel_inner', 'vessel_outer']:
            data[loop] = {}
            for var in self.segment[loop]:
                data[loop][var] = self.segment[loop][var].tolist()
        if 'tf' in kwargs:  # add tf profile
            tf = kwargs.get('tf')
            for loop, label in zip(['in', 'out'], ['TF_inner', 'TF_outer']):
                data[label] = {}
                for var in tf.x[loop]:
                    data[label][var] = tf.x[loop][var].tolist()

        with open(self.datadir + '{}.json'.format(self.dataname), 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True)


class wrap(object):

    def __init__(self, inner_points, outer_points):
        self.loops = collections.OrderedDict()
        self.loops['inner'] = {'points': inner_points}
        self.loops['outer'] = {'points': outer_points}

    def get_segment(self, loop):
        segment = self.loops[loop]['points']
        return segment['x'], segment['z']

    def set_segment(self, loop, x, z):
        self.loops[loop]['points'] = {'x': x, 'z': z}

    def sort_z(self, loop, select='lower'):
        x, z = self.get_segment(loop)
        x, z = geom.order(x, z, anti=True)  # order points
        if select == 'lower':
            imin = np.argmin(z)  # locate minimum
            x = np.append(x[imin:], x[:imin])  # sort
            z = np.append(z[imin:], z[:imin])
        else:
            imax = np.argmax(z)  # locate minimum
            x = np.append(x[imax:], x[:imax])  # sort
            z = np.append(z[imax:], z[:imax])
        self.set_segment(loop, x, z)

    def offset(self, loop, dt, **kwargs):
        x, z = self.get_segment(loop)
        gloop = Loop(x, z)
        x, z = gloop.fill(dt=dt, **kwargs)
        self.set_segment(loop, x, z)

    def interpolate(self):
        for loop in self.loops:
            x, z = self.get_segment(loop)
            x, z, l = geom.unique(x, z)
            interpolant = {'x': IUS(l, x), 'z': IUS(l, z)}
            self.loops[loop]['fun'] = interpolant

    def interp(self, loop, l):
        interpolant = self.loops[loop]['fun']
        return interpolant['x'](l), interpolant['z'](l)

    def sead(self, dl, N=500):  # course search
        l = np.linspace(dl[0], dl[1], N)
        x, z = np.zeros((N, 2)), np.zeros((N, 2))
        for i, loop in enumerate(self.loops):
            x[:, i], z[:, i] = self.interp(loop, l)
        dx_min, i_in, i_out = np.max(x[:, 1]), 0, 0
        for i, (xin, zin) in enumerate(zip(x[:, 0], z[:, 0])):
            dX = np.sqrt((x[:, 1] - xin)**2 + (z[:, 1] - zin)**2)
            dx = np.min(dX)
            if dx < dx_min:
                dx_min = dx
                i_in = i
                i_out = np.argmin(dX)
        return l[i_in], l[i_out]

    def cross(self, L):
        x, z = np.zeros(2), np.zeros(2)
        for i, (loop, l) in enumerate(zip(self.loops, L)):
            x[i], z[i] = self.interp(loop, l)
        err = (x[0] - x[1])**2 + (z[0] - z[1])**2
        return err

    def index(self, loop, l):
        rp, zp = self.interp(loop, l)  # point
        x, z = self.get_segment(loop)
        i = np.argmin((x - rp)**2 + (z - zp)**2)
        return i

    def close_loop(self):
        for loop in self.loops:
            x, z = self.get_segment(loop)
            if (x[0] - x[-1])**2 + (z[0] - z[-1])**2 != 0:
                x = np.append(x, x[0])
                z = np.append(z, z[0])
                self.set_segment(loop, x, z)

    def concentric(self, xin, zin, xout, zout):
        points = geom.inloop(xout, zout, xin, zin, side='out')
        if np.shape(points)[1] == 0:
            return True
        else:
            return False

    def fill(self, plot=False, color=colors[0]):  # minimization focused search
        xin, zin = self.get_segment('inner')
        xout, zout = self.get_segment('outer')
        concentric = self.concentric(xin, zin, xout, zout)
        if concentric:
            self.close_loop()
            xin, zin = self.get_segment('inner')
            xout, zout = self.get_segment('outer')
        self.interpolate()  # construct interpolators
        self.indx = {'inner': np.array([0, len(xin)], dtype=int),
                     'outer': np.array([0, len(xout)], dtype=int)}
        if not concentric:
            self.indx = {'inner': np.zeros(2, dtype=int),
                         'outer': np.zeros(2, dtype=int)}
            # low feild / high feild
            for i, dl in enumerate([[0, 0.5], [0.5, 1]]):
                lo = self.sead(dl)
                L = minimize(self.cross, lo, method='L-BFGS-B',
                             bounds=([0, 1], [0, 1])).x
                for loop, l in zip(self.loops, L):
                    self.indx[loop][i] = self.index(loop, l)

        x = np.append(xout[self.indx['outer'][0]:self.indx['outer'][1]],
                      xin[self.indx['inner'][0]:self.indx['inner'][1]][::-1])
        z = np.append(zout[self.indx['outer'][0]:self.indx['outer'][1]],
                      zin[self.indx['inner'][0]:self.indx['inner'][1]][::-1])
        self.patch = {'x': x, 'z': z}
        x = np.append(xin[:self.indx['inner'][0]],
                      xout[self.indx['outer'][0]:self.indx['outer'][1]])
        x = np.append(x, xin[self.indx['inner'][1]:])
        z = np.append(zin[:self.indx['inner'][0]],
                      zout[self.indx['outer'][0]:self.indx['outer'][1]])
        z = np.append(z, zin[self.indx['inner'][1]:])
        self.segment = {'x': x, 'z': z}
        if plot:
            self.plot(color=color)
        return self.segment, self.patch

    def plot(self, plot_loops=False, color=colors[0]):
        geom.polyfill(self.patch['x'], self.patch['z'], color=color)
        if plot_loops:
            pl.plot(self.segment['x'], self.segment['z'],
                    color=0.75 * np.ones(3))
            for loop in self.loops:
                x, z = self.get_segment(loop)
                pl.plot(x, z, '-')


class Loop(object):

    def __init__(self, X, Z, **kwargs):
        self.X = X
        self.Z = Z
        self.xo = kwargs.get('xo', (np.mean(X), np.mean(Z)))

    def rzPut(self):
        self.Xstore, self.Zstore = self.X, self.Z

    def rzGet(self):
        self.X, self.Z = self.Xstore, self.Zstore

    def fill(self, trim=None, dX=0, dt=0, ref_o=4 / 8 * np.pi, dref=np.pi / 4,
             edge=True, ends=True, color='k', label=None, alpha=0.8,
             referance='theta', part_fill=True, loop=False, s=0, gap=0,
             plot=False):
        dt_max = 0.1  # 2.5
        if not part_fill:
            dt_max = dt
        if isinstance(dt, list):
            dt = self.blend(dt, ref_o=ref_o, dref=dref, referance=referance,
                            gap=gap)
        dt, nt = geom.max_steps(dt, dt_max)
        Xin, Zin = geom.offset(self.X, self.Z, dX)  # gap offset
        for i in range(nt):
            self.part_fill(trim=trim, dt=dt, ref_o=ref_o, dref=dref,
                           edge=edge, ends=ends, color=color, label=label,
                           alpha=alpha, referance=referance, loop=loop, s=s,
                           plot=False)
        Xout, Zout = self.X, self.Z
        if plot:
            geom.polyparrot({'x': Xin, 'z': Zin}, {'x': Xout, 'z': Zout},
                            color=color, alpha=1)  # fill
        return Xout, Zout

    def part_fill(self, trim=None, dt=0, ref_o=4 / 8 * np.pi, dref=np.pi / 4,
                  edge=True, ends=True,
                  color='k', label=None, alpha=0.8, referance='theta',
                  loop=False, s=0, plot=False):
        Xin, Zin = self.X, self.Z
        if loop:
            Napp = 5  # Nappend
            X = np.append(self.X, self.X[:Napp])
            X = np.append(self.X[-Napp:], X)
            Z = np.append(self.Z, self.Z[:Napp])
            Z = np.append(self.Z[-Napp:], Z)
            X, Z = geom.rzSLine(X, Z, npoints=len(X), s=s)
            if isinstance(dt, (np.ndarray, list)):
                dt = np.append(dt, dt[:Napp])
                dt = np.append(dt[-Napp:], dt)
            Xout, Zout = geom.offset(X, Z, dt)
            print('part fill')
            Xout, Zout = Xout[Napp:-Napp], Zout[Napp:-Napp]
            Xout[-1], Zout[-1] = Xout[0], Zout[0]
        else:
            X, Z = geom.rzSLine(self.X, self.Z, npoints=len(self.X), s=s)
            Xout, Zout = geom.offset(X, Z, dt)
        self.X, self.Z = Xout, Zout  # update
        if trim is None:
            Lindex = [0, len(Xin)]
        else:
            Lindex = self.trim(trim)
        if plot:
            flag = 0
            for i in np.arange(Lindex[0], Lindex[1] - 1):
                Xfill = np.array([Xin[i], Xout[i], Xout[i + 1], Xin[i + 1]])
                Zfill = np.array([Zin[i], Zout[i], Zout[i + 1], Zin[i + 1]])
                if flag is 0 and label is not None:
                    flag = 1
                    pl.fill(Xfill, Zfill, facecolor=color, alpha=alpha,
                            edgecolor='none', label=label)
                else:
                    pl.fill(Xfill, Zfill, facecolor=color, alpha=alpha,
                            edgecolor='none')

    def blend(self, dt, ref_o=4 / 8 * np.pi, dref=np.pi / 4, gap=0,
              referance='theta'):
        if referance is 'theta':
            theta = np.arctan2(self.Z - self.xo[1], self.X - self.xo[0]) - gap
            theta[theta > np.pi] = theta[theta > np.pi] - 2 * np.pi
            tblend = dt[0] * np.ones(len(theta))  # inner
            tblend[(theta > -ref_o) & (theta < ref_o)] = dt[1]  # outer
            if dref > 0:
                for updown in [-1, 1]:
                    blend_index = (updown * theta >= ref_o) &\
                        (updown * theta < ref_o + dref)
                    tblend[blend_index] = dt[1] + (dt[0] - dt[1]) / dref *\
                        (updown * theta[blend_index] - ref_o)
        else:
            L = geom.length(self.X, self.Z)
            tblend = dt[0] * np.ones(len(L))  # start
            tblend[L > ref_o] = dt[1]  # end
            if dref > 0:
                blend_index = (L >= ref_o) & (L < ref_o + dref)
                tblend[blend_index] = dt[0] + (dt[1] - dt[0]) /\
                    dref * (L[blend_index] - ref_o)
        return tblend

    def trim(self, trim, X, Z):
        L = geom.length(X, Z, norm=True)
        index = []
        for t in trim:
            index.append(np.argmin(np.abs(L - t)))
        return index
