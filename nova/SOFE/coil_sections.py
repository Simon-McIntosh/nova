from nova.OCC_gen import OCC
from nova.config import select
from nova.properties import second_moment
from nova.coils import PF, TF
from nova.loops import Profile
from nova.streamfunction import SF
from amigo import geom
from amigo.addtext import linelabel
import pylab as pl
import numpy as np
import seaborn as sns
rc = {'figure.figsize': [10 * 12 / 16, 10], 'savefig.dpi': 350,
      'savefig.jpeg_quality': 100, 'savefig.pad_inches': 0.1,
      'lines.linewidth': 2}
sns.set(context='talk', style='white', font='sans-serif', palette='Set2',
        font_scale=7 / 8, rc=rc)
color = sns.color_palette('Set2', 12)  # Ne touchez pas!

nTF = 18
base = {'TF': 'demo_nTF', 'eq': 'DEMO_SN_SOF'}
config, setup = select(base, nTF=nTF, update=True)


class architect(object):
    '''
    calculate the sectional properties of cross-sections used in the
    coil cage structural solver, TF / intercoil supports / gravity supports
    places structural elements around TF coil
    '''

    def __init__(self, config, setup, family='S', nTF=16, obj='L'):
        self.nTF = nTF
        self.config = config

        self.profile = Profile(config['TF'], family=family, load=True,
                               part='TF', nTF=nTF, obj=obj, npoints=250)
        self.sf = SF(setup.filename)
        self.tf = TF(profile=self.profile, sf=self.sf)
        self.pf = PF(self.sf.eqdsk)

        self.tf_section(plot=False)

        '''
        self.PF_support()  # calculate PF support seats
        self.CS_support()  # calculate CS support seats
        self.Gravity_support()
        '''
        
    '''
    rwp = ro - (width + inboard)
    if rsep <= rwp:
        ywp = depth / 2 + side
    else:
        ywp = rwp * np.tan(theta)
    self.cs['support'] = {'rnose': rnose, 'ynose': ynose, 'dt': side,
                      'rwp': rwp, 'ywp': ywp}
    '''

    def tf_section(self, plot=False):
        side = self.tf.section['case']['side']
        inboard = self.tf.section['case']['inboard']
        outboard = self.tf.section['case']['outboard']
        external = self.tf.section['case']['external']
        nose = self.tf.section['case']['nose']
        depth = self.tf.section['winding_pack']['depth']
        width = self.tf.section['winding_pack']['width']

        ro = self.profile.loop.p[0]['p0']['r']
        theta = np.pi / self.nTF
        rsep = (depth / 2 + side) / np.tan(theta)
        rnose = ro - (width + inboard + nose)
        if rsep <= rnose:
            ynose = depth / 2 + side
        else:
            ynose = rnose * np.tan(theta)
        
        self.cs = {}  # cross-sections

        self.cs['wp'] = {'z': [-width / 2, width / 2],
                         'y': depth / 2 * np.ones(2)}
        self.cs['case_in'] = {'z': np.array([-(nose + width / 2),
                                             rsep - (ro - width / 2 - inboard),
                                             width / 2 + inboard]),
                              'y': np.array([ynose, depth / 2 + side,
                                             depth / 2 + side])}
        self.cs['case_out'] = {'z': np.array([-width / 2 - external,
                                              np.mean([external, outboard]),
                                              width / 2 + outboard]),
                               'y': depth / 2 + side * np.ones(3)}
        for name in ['wp', 'case_in', 'case_out']:
            self.get_pnt(name)  # get loop points
        self.case_tc = torsion()  # initalise torsional constant object
        self.case_tc.add_loop(self.cs['wp']['pnt'], 'in')

        if plot:
            fig, ax = pl.subplots(1, 2, sharex=True, figsize=(6, 4))
            pl.sca(ax[0])
            pl.axis('equal')
            pl.axis('off')
            pl.xlim([-1, 0.5])
            pl.plot(self.cs['case_in']['pnt'][0],
                    self.cs['case_in']['pnt'][1])
            geom.polyfill(self.cs['case_in']['pnt'][0],
                          self.cs['case_in']['pnt'][1],
                          color=color[0])
            geom.polyfill(self.cs['wp']['pnt'][0], self.cs['wp']['pnt'][1],
                          color=color[1])
            pl.sca(ax[1])
            pl.axis('equal')
            pl.axis('off')
            geom.polyfill(self.cs['case_out']['pnt'][0],
                          self.cs['case_out']['pnt'][1],
                          color=color[0])
            geom.polyfill(self.cs['wp']['pnt'][0], self.cs['wp']['pnt'][1],
                          color=color[1])

    def get_pnt(self, name):
        y, z = self.cs[name]['y'], self.cs[name]['z']
        if len(y) != len(z):
            raise ValueError('unequal point number in {} dict'.format(name))
        elif len(y) == 2:  # wp or outer case
            self.cs[name]['pnt'] = [[y[0], y[-1], -y[-1], -y[0]],
                                    [z[0], z[-1], z[-1], z[0]]]
        elif y[0] == y[1]:  # outer case
            self.cs[name]['pnt'] = [[y[0], y[-1], -y[-1], -y[0]],
                                    [z[0], z[-1], z[-1], z[0]]]
        elif len(y) == 3:  # inner case
            self.cs[name]['pnt'] = [[y[0], y[1], y[2], -y[2], -y[1],
                                     -y[0]],
                                    [z[0], z[1], z[2], z[2], z[1], z[0]]]

    def trans(self, frac):  # set transitional points
        '''
        frac == 0 returns inner loop
        fract == 1 returns outer loop
        frac > 0 & frac < 1 returns blended loop (transitional)
        '''
        self.cs['case'] = {'y': np.zeros(3), 'z': np.zeros(3)}
        for x in ['y', 'z']:
            self.cs['case'][x] = self.cs['case_in'][x] + \
                frac * (self.cs['case_out'][x] - self.cs['case_in'][x])
        self.get_pnt('case')

    def case(self, frac=0, plot=False):
        '''
        build case cross-sectional properties
        frac==0, inboard
        frac==1, outboard
        frac>0 & frac<1 transitional
        '''
        self.trans(frac)  # load case profile
        y, z = self.cs['case']['pnt'][0], self.cs['case']['pnt'][1]
        y_wp, z_wp = self.cs['wp']['pnt'][0], self.cs['wp']['pnt'][1]
        sm = second_moment()
        if frac < 1:  # inner or transitional
            sm.add_shape('tri', b=y[1] - y[0], h=z[1] - z[0],
                         flip_z=True, flip_y=False, dz=z[1], dy=y[0])
            sm.add_shape('tri', b=y[-1] - y[-2], h=z[-2] - z[-1],
                         flip_z=True, flip_y=True, dz=z[-2], dy=y[-1])
            sm.add_shape('rect', b=y[0] - y[-1], h=z[1] - z[0],
                         dz=z[1] - (z[1] - z[0]) / 2)
            sm.add_shape('rect', b=y[1] - y[-2], h=z[2] - z[1],
                         dz=z[1] + (z[2] - z[1]) / 2)
        else:
            sm.add_shape('rect', b=y[1] - y[2], h=z[1] - z[0],
                         dz=np.mean([z[0], z[1]]))
        sm.remove_shape('rect', b=y_wp[1] - y_wp[2], h=z_wp[1] - z_wp[0])
        C, I, A = sm.report()
        self.case_tc.add_loop(self.cs['case']['pnt'], 'out')
        J = self.case_tc.solve()
        if plot:
            sm.plot()
        return C, I, A, J

    def winding_pack(self, plot=False):  # winding pack sectional properties
        sm = second_moment()
        y, z = self.cs['wp']['pnt'][0], self.cs['wp']['pnt'][1]
        sm.add_shape('rect', b=y[1] - y[2], h=z[1] - z[0])
        C, I, A = sm.report()
        J = 0
        if plot:
            sm.plot()
        return C, I, A, J
    
    def gravity_support(self, plot=False):
        sm = second_moment()
        r, ro = 0.6, 0.59
        sm.add_shape('circ',r=r,ro=ro)
        C, I, A = sm.report()
        
        theta = np.linspace(0,2*np.pi,50)
        Lin = [ro*np.cos(theta), ro*np.sin(theta)]
        Lout = [r*np.cos(theta), r*np.sin(theta)]
        tc = torsion()  # initalise torsional constant object
        tc.add_loop(Lin, 'in')
        tc.add_loop(Lout, 'out')
        J = tc.solve(plot=True)
        print('J', J, I['xx'])
        sm.plot()

    def OIS(self):  # outer intercoil support
        sm = second_moment()
        sm.add_shape('rect', b=occ.OISsupport['OIS0']['thickness'],
                     h=occ.OISsupport['OIS0']['width'])
        C, I, A = sm.report()
        J = 0
        if plot:
            sm.plot()
        return C, I, A, J 

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
        for i, (r, z) in enumerate(zip(yout, zout)):
            j = np.argmin((r - yin)**2 + (z - zin)**2)
            ycl[i], zcl[i] = np.mean([r, yin[j]]), np.mean([z, zin[j]])
            t[i] = np.sqrt((r - yin[j])**2 + (z - zin[j])**2)
        A = geom.polyarea(ycl, zcl)
        dL = np.diff(geom.length(ycl, zcl, norm=False))
        J = 4 * A**2 / np.sum(dL / t[:-1])
        if plot:
            pl.plot(ycl, zcl)
            pl.plot(yin, zin)
            pl.plot(yout, zout)
        return J
  


atec = architect(config, setup, nTF=nTF)
# atec.winding_pack()
atec.case(1,plot=True)

atec.gravity_support(plot=True)

N = 100
Im = np.zeros((3, N))
J = np.zeros(N)

for i, f in enumerate(np.linspace(0, 1, N)):
    C, I, A, J[i] = atec.case(f)
    Im[:, i] = I['xx'], I['yy'], I['zz']

pl.figure(figsize=(10, 10 * 12 / 16))
text = linelabel(value = '1.2f')
pl.plot(Im[0, :])
text.add('Ixx')
pl.plot(Im[1, :])
text.add('Iyy')
pl.plot(Im[2, :])
text.add('Izz')
pl.plot(J)
text.add('J')
text.plot()
sns.despine()