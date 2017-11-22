import numpy as np
from amigo.pyplot import plt
from nova.config import select
from nova.properties import second_moment
from nova.coils import PF, TF
from nova.loops import Profile
from nova.streamfunction import SF
from nova.inverse import INV
from amigo import geom
from amigo.addtext import linelabel
from matplotlib import gridspec
from scipy.optimize import minimize_scalar, minimize
from warnings import warn
from amigo.geom import polyline


class architect(object):
    '''
    calculate the sectional properties of cross-sections used in the
    coil cage structural solver, TF / intercoil supports / gravity supports
    places structural elements around TF coil
    '''
    def __init__(self, tf, pf, plot=False):
        self.tf, self.pf = tf, pf
        self.loop, self.nTF = self.tf.profile.loop, self.tf.nTF
        self.cross_sections()
        self.connect(plot=plot)

    def cross_sections(self):
        self.initalise_cs()
        self.tf_cs()
        self.gs_cs()

    def connect(self, plot=False):
        self.set_loop_interpolators()  # tf loop interpolators
        self.tf.split_loop(plot=True)
        self.CS_support()  # calculate CS support seats
        self.PF_support()  # calculate PF support seats
        self.G_support(Xo=13, width=0.75)  # gravity support
        self.OIC_support(thickness=0.15)  # outer intercoil support
        self.get_TF_nodes()  # TF node index for connecting parts
        if plot:
            self.plot_connections()

    def set_loop_interpolators(self):
        self.tf.loop_interpolators(offset=0)
        self.tf_fun = self.tf.fun.copy()  # copy dict of functions
        self.tf.loop_interpolators(offset=-0.15)
        self.tf_fun_offset = self.tf.fun.copy()  # copy dict of functions

    def plot_connections(self, *args):
        if len(args) == 0:
            plt.figure(figsize=plt.figaspect(1.5))
        self.tf.fill()
        self.pf.plot(subcoil=False, plasma=True, current=True, label=True)
        self.pf.plot(subcoil=True)
        for section in ['nose', 'trans_lower', 'loop', 'trans_upper']:
            plt.plot(self.tf.p[section]['x'], self.tf.p[section]['z'], 'o-')
        for name in self.PFsupport:
            nodes = np.array(self.PFsupport[name]['nodes'])
            p = self.PFsupport[name]['p']
            geom.polyfill(nodes[:, 0], nodes[:, 1], color=0.4 * np.ones(3))
            plt.plot(p['x'], p['z'], '-o')
        nodes = np.array(self.Gsupport['base'])
        geom.polyfill(nodes[:, 0], nodes[:, 1], color=0.4 * np.ones(3))
        plt.plot(self.Gsupport['Xo'] * np.ones(2),
                 [self.Gsupport['zbase'], self.Gsupport['zfloor']], 'o-',
                 color=0.4 * np.ones(3), lw=4)
        geom.polyfill(self.loop['cs']['x'], self.loop['cs']['z'],
                      color=0.4 * np.ones(3))
        for name in self.loop['OIS']:
            nodes = np.array(self.loop['OIS'][name]['nodes'])
            geom.polyfill(nodes[:, 0], nodes[:, 1], color=0.4 * np.ones(3))
        plt.axis('off')
        plt.axis('equal')

    def OIS_placment(L, TFloop, point):
        err = (point[0] - TFloop['x'](L))**2 + (point[1] - TFloop['z'](L))**2
        return err

    def draw_OIS(self, L, width, thickness, TFloop):
        dl = width / TFloop['L']
        xcl = np.array([TFloop['x'](L - dl / 2), TFloop['x'](L + dl / 2)])
        zcl = np.array([TFloop['z'](L - dl / 2), TFloop['z'](L + dl / 2)])
        xo, zo = np.mean(xcl), np.mean(zcl)
        L = minimize_scalar(architect.OIS_placment, method='bounded',
                            args=(TFloop, (xo, zo)), bounds=[0, 1]).x
        dr, dz = TFloop['x'](L) - xo, TFloop['z'](L) - zo
        xcl += dr / 2
        zcl += dz / 2
        dt = np.array([xcl[1] - xcl[0], 0, zcl[1] - zcl[0]])
        dt /= np.linalg.norm(dt)
        dn = np.cross(dt, np.array([0, 1, 0]))

        xcl = np.append(xcl + dn[0] * thickness / 2,
                        xcl[::-1] - dn[0] * thickness / 2)
        zcl = np.append(zcl + dn[2] * thickness / 2,
                        zcl[::-1] - dn[2] * thickness / 2)
        nodes = [[xcl[i], zcl[i]] for i in range(4)]
        return nodes

    def OIC_support(self, thickness=0.15):  # outer intercoil support
        self.loop['OIS'] = {}
        self.OICsupport = {}
        for i, (L, width) in enumerate(zip([0.4, 0.66], [4.5, 3.75])):
            name = 'ois{:d}'.format(i)
            self.loop['OIS'][name] = {'L': L, 'width': width,
                                      'thickness': thickness}
            nodes = self.draw_OIS(L, width, thickness, self.tf_fun['cl'])
            self.loop['OIS'][name]['nodes'] = nodes
            x = np.array([nodes[0][0], nodes[1][0]])
            z = np.array([nodes[0][1], nodes[1][1]])
            xo, zo = x.mean(), z.mean()
            Lmid = minimize_scalar(architect.OIS_placment, method='bounded',
                                   args=(self.tf_fun['cl'], (xo, zo)),
                                   bounds=[0, 1]).x

            p = {'x': np.array([x[0], self.tf_fun['cl']['x'](Lmid), x[-1]]),
                 'z': np.array([z[0], self.tf_fun['cl']['z'](Lmid), z[-1]])}
            el_dy = np.array([x[-1], 0, z[-1]]) - np.array([x[0], 0, z[0]])
            el_dy /= np.linalg.norm(el_dy)
            self.OICsupport[name] = {'p': p, 'thickness': thickness,
                                     'width': width, 'el_dy': el_dy}
            self.adjust_TFnode(p['x'][0], p['z'][0])
            self.adjust_TFnode(p['x'][1], p['z'][1])

    def CS_support(self):
        zo = self.tf.profile.loop.cs['z']
        L = minimize_scalar(architect.cs_top, method='bounded',
                            args=(self.loop['cs']['xwp'], self.tf_fun['out']),
                            bounds=[0.5, 1]).x
        ztop = float(self.tf_fun['out']['z'](L))
        self.loop['cs']['z'] = [zo, zo, ztop, ztop]
        self.loop['cs']['ztop'] = ztop
        self.loop['cs']['zo'] = zo

    def cs_top(L, xwp, TFloop):
        err = abs(TFloop['x'](L) - xwp)
        return err

    def update_TF(self, xo):  # stripped ordered dict
        self.tf.get_loops(self.loop.draw(x=xo))  # update tf
        self.tf.split_loop()

    def support_arm(L, coil, TFloop):
        dl = np.sqrt((coil['x'] - TFloop['x'](L))**2 +
                     (coil['z'] - TFloop['z'](L))**2)
        return dl

    def intersect(x, xc, nhat, TFloop):
        L, s = x  # unpack
        rTF, zTF = TFloop['x'](L), TFloop['z'](L)
        rs, zs = s * nhat + xc
        err = np.sqrt((rTF - rs)**2 + (zTF - zs)**2)
        return err

    def connect_loop(self, coil, loop, edge=0.15, hover=0.1, argmin=60):
        L = minimize_scalar(architect.support_arm, method='bounded',
                            args=(coil, loop), bounds=[0, 1]).x
        rTF, zTF = loop['x'](L), loop['z'](L)
        nhat = np.array([rTF - coil['x'], zTF - coil['z']])
        ndir = 180 / np.pi * np.arctan(abs(nhat[1] / nhat[0]))  # angle, deg
        if ndir < argmin:  # limit absolute support angle
            nhat = np.array([np.sign(nhat[0]),
                             np.tan(argmin * np.pi / 180) * np.sign(nhat[1])])
            ndir = 180 / np.pi * np.arctan(abs(nhat[1] / nhat[0]))  # update
        nhat /= np.linalg.norm(nhat)
        above = np.sign(np.dot(nhat, [0, 1]))
        zc = coil['z'] + above * (coil['dz'] / 2 + hover)
        nodes = [[] for _ in range(4)]
        for i, sign in enumerate([-1, 1]):  # inboard / outboard
            rc = coil['x'] + sign * (coil['dx'] / 2 + edge)
            nodes[i] = [rc, zc]
            xc = np.array([rc, zc])
            xo = np.array([L, 0.5])
            res = minimize(architect.intersect, xo, method='L-BFGS-B',
                           bounds=([0, 1], [0, 15]), args=(xc, nhat, loop))
            rs, zs = res.x[1] * nhat + xc
            nodes[3 - i] = [rs, zs]

        nd = {'x': np.zeros(3), 'z': np.zeros(3)}  # cl, outboard, pf
        for i in range(2):  # cl, pf [0,2]
            nd['x'][2*i] = np.mean([nodes[2*i][0], nodes[2*i+1][0]])
            nd['z'][2*i] = np.mean([nodes[2*i][1], nodes[2*i+1][1]])
        rout, zout = polyline(self.tf.p['out'], nd)  # TFout intersect
        nd['x'][1] = rout  # outboard TF node [1]
        nd['z'][1] = zout

        xc = [nd['x'][-1], nd['z'][-1]]
        res = minimize(architect.intersect, xo, method='L-BFGS-B',
                       bounds=([0, 1], [0, 15]),
                       args=(xc, nhat, self.tf_fun['cl']))
        # nd['x'][-1], nd['z'][-1] = res.x[1] * nhat + xc
        x, z = res.x[1] * nhat + xc
        nd['x'][-1], nd['z'][-1] = self.cl_snap(x, z)
        return nodes, nd, ndir, L

    def cl_snap(self, x, z):
        L = minimize_scalar(architect.support_arm, method='bounded',
                            args=({'x': x, 'z': z}, self.tf_fun['cl']),
                            bounds=[0, 1]).x
        return self.tf_fun['cl']['x'](L), self.tf_fun['cl']['z'](L)

    def PF_support(self, n=5):
        dt = self.loop['cs']['dt']
        space = self.cs['wp']['d']+2*self.cs['case']['side']-dt
        self.cs['pfs'] = {'dt': dt, 'n': n, 'space': space}
        self.PFsupport = {}
        for name in self.pf.index['PF']['name']:
            coil = self.pf.coil[name]
            nodes, p, ndir, L = self.connect_loop(
                    coil, self.tf_fun_offset['out'],
                    edge=0.15, hover=0.1, argmin=35)
            width = coil['dx']*np.sin(ndir*np.pi/180)  # support width
            self.PFsupport[name] = {'nodes': nodes, 'p': p, 'width': width}
            self.adjust_TFnode(p['x'][-1], p['z'][-1])

    def get_TF_nodes(self):
        tf_x, tf_z = self.tf.p['cl_fe']['x'], self.tf.p['cl_fe']['z']
        for name in self.PFsupport:
            self.PFsupport[name]['nd_tf'] = \
                np.argmin((tf_x-self.PFsupport[name]['p']['x'][-1])**2 +
                          (tf_z-self.PFsupport[name]['p']['z'][-1])**2)
        self.Gsupport['nd_tf'] = np.argmin((tf_x-self.Gsupport['Xo'])**2 +
                                           (tf_z-self.Gsupport['zbase'])**2)
        for name in self.OICsupport:
            p = self.OICsupport[name]['p']
            self.OICsupport[name]['nd_tf'] = np.zeros(len(p['x']), dtype=int)
            for i, (x, z) in enumerate(zip(p['x'], p['z'])):
                self.OICsupport[name]['nd_tf'][i] = np.argmin((tf_x-x)**2 +
                                                              (tf_z-z)**2)

    def GS_placement(L, Xo, TFloop):
        return abs(Xo - TFloop['x'](L))

    def G_support(self, Xo=13, dZo=-1, width=0.75):  # gravity support
        L = minimize_scalar(architect.GS_placement, method='bounded',
                            args=(Xo - width / 2, self.tf_fun['out']),
                            bounds=[0, 0.5]).x
        coil = {'x': self.tf_fun['out']['x'](L) + width / 2,
                'z': self.tf_fun['out']['z'](L) - width / 2,
                'dx': width, 'dz': width}
        nodes, p = self.connect_loop(coil, self.tf_fun_offset['out'],
                                     edge=0, hover=0, argmin=90)[:2]
        self.adjust_TFnode(p['x'][-1], p['z'][-1])  # add / adjust

        self.Gsupport = {}
        self.Gsupport['base'] = nodes
        z = [[self.pf.coil[name]['z'] - self.pf.coil[name]['dz'] / 2]
             for name in self.pf.coil]
        floor = np.min(z) + dZo
        self.Gsupport['zbase'] = float(self.tf_fun['out']['z'](L))
        self.Gsupport['zfloor'] = floor
        self.Gsupport['Xo'] = Xo
        self.Gsupport['rtube'] = width / 3  # tube radius
        self.Gsupport['ttube'] = width / 9  # tube thickness
        self.Gsupport['width'] = width

        theta = np.pi / self.nTF  # half angle
        alpha = np.arctan((Xo * np.tan(theta) - width) /
                          (self.Gsupport['zbase'] - self.Gsupport['zfloor'] -
                           width / 2))
        pin2pin = np.sqrt((Xo * np.tan(theta) - width)**2 +
                          (self.Gsupport['zbase'] - self.Gsupport['zfloor'] -
                           width / 2)**2)

        self.Gsupport['alpha'] = alpha
        self.Gsupport['pin2pin'] = pin2pin
        self.Gsupport['spread'] = pin2pin * np.sin(alpha) + 2 * width
        self.Gsupport['zground'] = pin2pin * (1-np.cos(alpha)) +\
            nodes[1][1]-1.5*width-pin2pin
        self.Gsupport['yfloor'] = pin2pin * np.sin(alpha)

    def adjust_TFnode(self, x, z):
        x, z = self.cl_snap(x, z)  # snap node to centreline
        i = np.argmin((self.tf.p['cl_fe']['x']-x)**2 +
                      (self.tf.p['cl_fe']['z']-z)**2)
        dl = np.sqrt((self.tf.p['cl_fe']['x'][i+1] -
                      self.tf.p['cl_fe']['x'][i-1])**2 +
                     (self.tf.p['cl_fe']['z'][i+1] -
                      self.tf.p['cl_fe']['z'][i-1])**2)/2
        dx = np.sqrt((self.tf.p['cl_fe']['x'][i] - x)**2 +
                     (self.tf.p['cl_fe']['z'][i] - z)**2)
        if dx > 0.5*dl:  # add node (0.2*)
            Ln = minimize_scalar(architect.OIS_placment, method='bounded',
                                 args=(self.tf_fun['out'], (x, z)),
                                 bounds=[0, 1]).x
            Li = minimize_scalar(architect.OIS_placment, method='bounded',
                                 args=(self.tf_fun['out'],
                                       (self.tf.p['cl_fe']['x'][i],
                                        self.tf.p['cl_fe']['z'][i])),
                                 bounds=[0, 1]).x

            j = 0 if Ln < Li else 1
            # self.tf.p['cl_fe']['x'] = np.insert(self.tf.p['cl_fe']['x'], i+j, x)
            # self.tf.p['cl_fe']['z'] = np.insert(self.tf.p['cl_fe']['z'], i+j, z)
        else:  # adjust node
            a = 1
            # self.tf.p['cl_fe']['x'][i], self.tf.p['cl_fe']['z'][i] = x, z

    def initalise_cs(self):
        '''
        initalise cross section parameters
        tf:    TF coil
            side == side wall thickness
            lf_in == inner tf surface wall thickness on low feild side
            lf_out == outer tf surface wall thickness on low feild side
            hf_in == inner tf surface wall thickness on high feild side
            hf_out == outer tf surface wall thickness on high feild side (nose)
        wp:    winding pack
            d == winding pack depth, in toroidal direction
            w == winding pack width, in poloidal direction
        '''
        self.cs = {}  # cross section parameter dict
        self.cs['case'] = {'side': self.tf.section['case']['side'],
                           'lf_in': self.tf.section['case']['outboard'],
                           'lf_out': self.tf.section['case']['external'],
                           'hf_in': self.tf.section['case']['inboard'],
                           'hf_out': self.tf.section['case']['nose']}
        self.cs['wp'] = {'d': self.tf.section['winding_pack']['depth'],
                         'w': self.tf.section['winding_pack']['width'],
                         'xo': self.tf.profile.loop.cs['x']}
        self.cs['gs'] = {'x': 0.1, 't': 0.05}

    def update_cs(self, **kwargs):
        for var in kwargs:
            set_var = False
            for name in self.cs:
                cs = self.cs[name]
                if var in cs:  # update
                    cs[var] = kwargs[var]
                    set_var = True
            if not set_var:
                warn('update failed\n{} not found in cs'.format(var))
        self.tf_cs()  # update tf cross-section
        self.gs_cs()  # update gravity support parameters

    def tf_cs(self, plot=False):
        wp, case = self.cs['wp'], self.cs['case']  # load parameters
        theta = np.pi / self.nTF
        rsep = (wp['d'] / 2 + case['side']) / np.tan(theta)
        xnose = wp['xo'] - (wp['w'] + case['hf_out'] + case['hf_in'])
        if rsep <= xnose:
            ynose = wp['d'] / 2 + case['side']
        else:
            ynose = xnose * np.tan(theta)
        xwp = wp['xo']-(wp['w']+case['hf_in'])
        if rsep <= xwp:
            ywp = wp['d']/2+case['side']
        else:
            ywp = xwp*np.tan(theta)
        self.loop = {}  # cross-section loops
        self.loop['cs'] = {'xnose': xnose, 'ynose': ynose, 'dt': case['side'],
                           'xwp': xwp, 'ywp': ywp}
        self.loop['cs']['x'] = [xnose, xwp, xwp, xnose]
        self.loop['wp'] = {'z': [-wp['w'] / 2, wp['w'] / 2],
                           'y': wp['d'] / 2 * np.ones(2)}
        self.loop['case_in'] = {'z': np.array([-(case['hf_out'] + wp['w'] / 2),
                                               rsep - (wp['xo'] - wp['w'] / 2 -
                                                       case['hf_in']),
                                               wp['w'] / 2 + case['hf_in']]),
                                'y': np.array([ynose,
                                               wp['d'] / 2 + case['side'],
                                               wp['d'] / 2 + case['side']])}
        self.loop['case_out'] = {'z': np.array([-wp['w'] / 2 - case['lf_out'],
                                                np.mean([case['lf_out'],
                                                         case['lf_in']]),
                                                wp['w'] / 2 + case['lf_in']]),
                                 'y': wp['d'] / 2 + case['side'] * np.ones(3)}
        for name in ['wp', 'case_in', 'case_out']:
            self.get_pnt(name)  # get loop points
        self.case_tc = torsion()  # initalise torsional constant object
        self.case_tc.add_loop(self.loop['wp']['pnt'], 'in')  # add winding pack
        if plot:
            self.plot_tf_section()

    def plot_tf_section(self):

        fig, ax = plt.subplots(1, 2, sharex=True)
        plt.sca(ax[0])
        plt.axis('equal')
        plt.axis('off')
        plt.xlim([-1, 0.5])
        plt.plot(self.loop['case_in']['pnt'][0],
                 self.loop['case_in']['pnt'][1])
        geom.polyfill(self.loop['case_in']['pnt'][0],
                      self.loop['case_in']['pnt'][1],
                      color='C0')
        geom.polyfill(self.loop['wp']['pnt'][0], self.loop['wp']['pnt'][1],
                      color='C1')
        plt.sca(ax[1])
        plt.axis('equal')
        plt.axis('off')
        geom.polyfill(self.loop['case_out']['pnt'][0],
                      self.loop['case_out']['pnt'][1],
                      color='C0')
        geom.polyfill(self.loop['wp']['pnt'][0], self.loop['wp']['pnt'][1],
                      color='C1')

    def get_pnt(self, name):
        y, z = self.loop[name]['y'], self.loop[name]['z']
        if len(y) != len(z):
            raise ValueError('unequal point number in {} dict'.format(name))
        elif len(y) == 2:  # wp or outer case
            self.loop[name]['pnt'] = [[y[0], y[-1], -y[-1], -y[0]],
                                      [z[0], z[-1], z[-1], z[0]]]
        elif y[0] == y[1]:  # outer case
            self.loop[name]['pnt'] = [[y[0], y[-1], -y[-1], -y[0]],
                                      [z[0], z[-1], z[-1], z[0]]]
        elif len(y) == 3:  # inner case
            self.loop[name]['pnt'] = [[y[0], y[1], y[2], -y[2], -y[1],
                                      -y[0]],
                                      [z[0], z[1], z[2], z[2], z[1], z[0]]]

    def trans(self, frac):  # set transitional points
        '''
        frac == 0 returns inner loop
        fract == 1 returns outer loop
        frac > 0 & frac < 1 returns blended loop (transitional)
        '''
        self.loop['case'] = {'y': np.zeros(3), 'z': np.zeros(3)}
        for x in ['y', 'z']:
            self.loop['case'][x] = self.loop['case_in'][x] + \
                frac * (self.loop['case_out'][x] - self.loop['case_in'][x])
        self.get_pnt('case')

    def case(self, frac=0, plot=False, centroid=True):
        '''
        build case cross-sectional properties
        frac==0, inboard
        frac==1, outboard
        frac>0 & frac<1 transitional
        '''
        self.trans(frac)  # load case profile
        y, z = self.loop['case']['pnt'][0], self.loop['case']['pnt'][1]
        y_wp, z_wp = self.loop['wp']['pnt'][0], self.loop['wp']['pnt'][1]
        sm = second_moment()
        if frac < 1:  # inner or transitional
            sm.add_shape('tri', b=y[1] - y[0], h=z[1] - z[0],
                         flip_z=True, flip_y=False, dz=z[1], dy=y[0])
            sm.add_shape('tri', b=y[-1] - y[-2], h=z[-2] - z[-1],
                         flip_z=True, flip_y=True, dz=z[-2], dy=y[-1])
            sm.add_shape('rect', b=y[0] - y[-1], h=z[1] - z[0],
                         dz=z[1] - (z[1] - z[0]) / 2)
            if z[2] > z[1]:
                sm.add_shape('rect', b=y[1] - y[-2], h=z[2] - z[1],
                             dz=z[1] + (z[2] - z[1]) / 2)
            else:
                sm.remove_shape('rect', b=y[1] - y[-2], h=z[1] - z[2],
                                dz=z[1] + (z[2] - z[1]) / 2)
        else:
            sm.add_shape('rect', b=y[1] - y[2], h=z[1] - z[0],
                         dz=np.mean([z[0], z[1]]))
        sm.remove_shape('rect', b=y_wp[1] - y_wp[2], h=z_wp[1] - z_wp[0])
        C, I, A = sm.report()
        self.case_tc.add_loop(self.loop['case']['pnt'], 'out')
        J = self.case_tc.solve()
        if plot:
            sm.plot(centroid=centroid)
        section = {'C': C, 'I': I, 'A': A, 'J': J,
                   'pnt': np.copy(self.loop['case']['pnt'])}
        return section

    def winding_pack(self, plot=False):  # winding pack sectional properties
        sm = second_moment()
        y, z = self.loop['wp']['pnt'][0], self.loop['wp']['pnt'][1]
        sm.add_shape('rect', b=y[1]-y[2], h=z[1]-z[0])
        C, I, A = sm.report()
        J = 1/3*abs(y[1]-y[2])*abs(z[1]-z[0])**3
        if plot:
            sm.plot()
        section = {'C': C, 'I': I, 'A': A, 'J': J,
                   'pnt': np.copy(self.loop['wp']['pnt'])}
        return section

    def gs_cs(self):
        rm, t2 = self.cs['gs']['x'], self.cs['gs']['t']/2
        if t2 > rm:  # half thickness mean radius
            errtxt = 'gravity support thickness greater than mean radius'
            raise ValueError(errtxt)
        self.loop['gs'] = {'ro': rm-t2, 'r': rm+t2}

    def gravity_support(self, plot=False, **kwargs):  # set t, or x
        update = False
        for var in ['x', 't']:
            if var in kwargs:  # t==pipe thickness, x==mean radius
                update = True
                self.cs['gs'][var] = kwargs[var]
        if update:
            self.gs_cs()
        sm = second_moment()
        ro, r = self.loop['gs']['ro'], self.loop['gs']['r']
        sm.add_shape('circ', r=r, ro=ro)
        pnt = sm.get_pnt()
        C, I, A = sm.report()
        J = I['xx']
        if plot:
            sm.plot()
        section = {'C': C, 'I': I, 'A': A, 'J': J, 'pnt': pnt}
        return section

    def intercoil_support(self, thickness, width, plot=False):
        # outer intercoil support
        sm = second_moment()
        sm.add_shape('rect', b=width, h=thickness)
        C, I, A = sm.report()
        pnt = sm.get_pnt()
        J = 1/3*width*thickness**3
        if plot:
            sm.plot()
        section = {'C': C, 'I': I, 'A': A, 'J': J, 'pnt': pnt}
        return section

    def coil_support(self, width=0.5, plot=False):  # PF coil support
        sm = second_moment()
        dt = self.cs['pfs']['dt']
        n = self.cs['pfs']['n']
        space = self.cs['pfs']['space']
        for dy in np.linspace(-space/2, space/2, n):
            sm.add_shape('rect', b=dt, h=width, dy=dy)
        C, I, A = sm.report()
        pnt = sm.get_pnt()
        J = n*1/3*width*dt**3
        if plot:
            sm.plot()

        section = {'C': C, 'I': I, 'A': A, 'J': J, 'pnt': pnt}
        return section

    def plot_transition(self):
        fig, ax = plt.subplots(1, 1)
        gs = gridspec.GridSpec(3, 3, wspace=0.0, hspace=0.0)
        for i, (fact, txt) in enumerate(zip([0, 0.5, 1],
                                            ['inboard', 'transition',
                                             'outboard'])):
            ax = plt.subplot(gs[i])
            ax.axis('equal')
            ax.axis('off')
            plt.text(0, 0, txt, ha='center', va='center')
            self.case(fact, plot=True, centroid=False)
        ax = plt.subplot(gs[3:6])
        plt.sca(ax)
        N = 100
        Im = np.zeros((3, N))
        J = np.zeros(N)
        fact = np.linspace(0, 1, N)
        for i, f in enumerate(fact):
            section = atec.case(f)
            J[i], Isec = section['J'], section['I']
            Im[:, i] = Isec['xx'], Isec['yy'], Isec['zz']
        text = linelabel(value='1.2f')
        plt.plot(fact, Im[1, :])
        text.add('Iyy')
        plt.plot(fact, Im[2, :])
        text.add('Izz')
        plt.plot(fact, J)
        text.add('J')
        text.plot()
        plt.despine()
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['inboard', 'transition', 'outboard'])
        plt.ylabel(r'sectional properties, m$^2$')
        plt.xlabel('cross-section position')


class torsion(object):

    def __init__(self, npoints=500):
        self.npoints = npoints
        self.L = {}

    def add_loop(self, pnt, name):
        if name not in ['in', 'out']:
            raise ValueError('loop name required to be \'in\' or \'out\'')
        y, z = pnt[0], pnt[1]
        y, z = np.append(y[-1], y), np.append(z[-1], z)
        y, z = geom.rzInterp(y, z, npoints=self.npoints)
        self.L[name] = [y, z]

    def solve(self, plot=False):
        if 'in' not in self.L or 'out' not in self.L:
            raise ValueError('set both loops, \'in\' and \'out\'')
        yin, zin = self.L['in'][0], self.L['in'][1]
        yout, zout = self.L['out'][0], self.L['out'][1]
        ycl, zcl = np.zeros(self.npoints), np.zeros(self.npoints)
        t = np.zeros(self.npoints)
        for i, (x, z) in enumerate(zip(yout, zout)):
            j = np.argmin((x - yin)**2 + (z - zin)**2)
            ycl[i], zcl[i] = np.mean([x, yin[j]]), np.mean([z, zin[j]])
            t[i] = np.sqrt((x - yin[j])**2 + (z - zin[j])**2)
        A = geom.polyarea(ycl, zcl)
        dL = np.diff(geom.length(ycl, zcl, norm=False))
        J = 4 * A**2 / np.sum(dL / t[:-1])
        if plot:
            plt.plot(ycl, zcl)
            plt.plot(yin, zin)
            plt.plot(yout, zout)
        return J


if __name__ is '__main__':

    nTF = 16
    base = {'TF': 'demo', 'eq': 'DEMO_SN_SOF'}
    config, setup = select(base, nTF=nTF, update=False)

    profile = Profile(config['TF_base'], family='S', load=True,
                      part='TF', nTF=nTF, obj='L', npoints=500)
    sf = SF(setup.filename)
    pf = PF(sf.eqdsk)
    tf = TF(profile=profile, sf=sf)

    inv = INV(pf, tf, dCoil=2.5, offset=0.3)
    inv.colocate(sf, n=1e3, expand=0.5, centre=0, width=363/(2*np.pi))
    inv.set_limit(FPFz=80)
    inv.wrap_PF(solve=False)
    inv.eq.run(update=False)

    fig = plt.figure(figsize=plt.figaspect(1.2))
    sf.contour()
    inv.plot_fix(tails=True)
    atec = architect(tf, pf)
    atec.plot_connections(fig)
    atec.plot_transition()

    atec.plot_tf_section()


