from nova.OCC_gen import OCC
from nova.config import select
from nova.properties import second_moment
from nova.coils import PF, TF
from nova.loops import Profile
from nova.streamfunction import SF
from amigo import geom
from amigo.addtext import linelabel
from matplotlib import gridspec
import pylab as pl
import numpy as np
from warnings import warn
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
        self.initalise_cs()
        self.tf_cs()
        self.gs_cs()
        self.initalize_mat()
        self.define_materials()
        self.pf = PF(self.sf.eqdsk)

        '''
        self.PF_support()  # calculate PF support seats
        self.CS_support()  # calculate CS support seats
        self.Gravity_support()
        '''

    def initalize_mat(self, nmat_max=20, nsec_max=5):
        self.nmat_max = nmat_max
        self.nsec_max = nsec_max
        self.nmat = 0
        self.mtype = np.dtype({'names': ['name', 'E', 'G', 'rho',
                                         'J', 'A', 'Iyy', 'Izz'],
                               'formats': ['S24', 'float', 'float', 'float',
                                           'float', 'float', 'float',
                                           'float']})
        self.mat = np.zeros((nmat_max), dtype=[('ID', 'int'), ('name', 'S24'),
                                               ('nsection', 'int'),
                                               ('mat_o', self.mtype),
                                               ('mat_array',
                                               (self.mtype, self.nsec_max))])

    def define_materials(self):
        # forged == inner leg, cast == outer + supports
        self.mat_data = {}
        self.mat_data['wp'] = {'E': 95e9, 'rho': 8940, 'v': 0.33}
        self.mat_data['steel_forged'] = {'E': 205e9, 'rho': 7850, 'v': 0.29}
        self.mat_data['steel_cast'] = {'E': 190e9, 'rho': 7850, 'v': 0.29}
        self.update_shear_modulus()

    def update_shear_modulus(self):
        for name in self.mat_data:
            E, v = self.mat_data[name]['E'], self.mat_data[name]['v']
            self.mat_data[name]['G'] = E/(2*(1+v))

    '''
    rwp = ro - (width + inboard)
    if rsep <= rwp:
        ywp = depth / 2 + side
    else:
        ywp = rwp * np.tan(theta)
    self.cs['support'] = {'rnose': rnose, 'ynose': ynose, 'dt': side,
                      'rwp': rwp, 'ywp': ywp}
    '''

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
                         'ro': self.profile.loop.p[0]['p0']['r']}
        self.cs['gs'] = {'r': 0.1, 't': 0.05}

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
        rnose = wp['ro'] - (wp['w'] + case['hf_in'] + case['hf_out'])
        if rsep <= rnose:
            ynose = wp['d'] / 2 + case['side']
        else:
            ynose = rnose * np.tan(theta)
        self.loop = {}  # cross-section loops
        self.loop['wp'] = {'z': [-wp['w'] / 2, wp['w'] / 2],
                           'y': wp['d'] / 2 * np.ones(2)}
        self.loop['case_in'] = {'z': np.array([-(case['hf_out'] + wp['w'] / 2),
                                               rsep - (wp['ro'] - wp['w'] / 2 -
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
        fig, ax = pl.subplots(1, 2, sharex=True, figsize=(6, 4))
        pl.sca(ax[0])
        pl.axis('equal')
        pl.axis('off')
        pl.xlim([-1, 0.5])
        pl.plot(self.loop['case_in']['pnt'][0],
                self.loop['case_in']['pnt'][1])
        geom.polyfill(self.loop['case_in']['pnt'][0],
                      self.loop['case_in']['pnt'][1],
                      color=color[0])
        geom.polyfill(self.loop['wp']['pnt'][0], self.loop['wp']['pnt'][1],
                      color=color[1])
        pl.sca(ax[1])
        pl.axis('equal')
        pl.axis('off')
        geom.polyfill(self.loop['case_out']['pnt'][0],
                      self.loop['case_out']['pnt'][1],
                      color=color[0])
        geom.polyfill(self.loop['wp']['pnt'][0], self.loop['wp']['pnt'][1],
                      color=color[1])

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
        section = {'C': C, 'I': I, 'A': A, 'J': J}
        return section

    def winding_pack(self, plot=False):  # winding pack sectional properties
        sm = second_moment()
        y, z = self.loop['wp']['pnt'][0], self.loop['wp']['pnt'][1]
        sm.add_shape('rect', b=y[1] - y[2], h=z[1] - z[0])
        C, I, A = sm.report()
        J = 0
        if plot:
            sm.plot()
        section = {'C': C, 'I': I, 'A': A, 'J': J}
        return section

    def gs_cs(self):
        rm, t2 = self.cs['gs']['r'], self.cs['gs']['t']/2
        if t2 > rm:  # half thickness mean radius
            errtxt = 'gravity support thickness greater than mean radius'
            raise ValueError(errtxt)
        self.loop['gs'] = {'ro': rm-t2, 'r': rm+t2}

    def gravity_support(self, plot=False, **kwargs):  # set t, or r
        update = False
        for var in ['r', 't']:
            if var in kwargs:  # t==pipe thickness, r==mean radius
                update = True
                self.cs['gs'][var] = kwargs[var]
        if update:
            self.gs_cs()
        sm = second_moment()
        ro, r = self.loop['gs']['ro'], self.loop['gs']['r']
        sm.add_shape('circ', r=r, ro=ro)
        C, I, A = sm.report()
        J = I['xx']
        if plot:
            sm.plot()
        section = {'C': C, 'I': I, 'A': A, 'J': J}
        return section

    def intercoil_support(self, plot=False):  # outer intercoil support
        sm = second_moment()
        thickness = 0.1
        width = 2
        sm.add_shape('rect', b=thickness, h=width)
        C, I, A = sm.report()
        pnt = {'y': [thickness/2, thickness/2, -thickness/2, -thickness/2],
               'z': [-width/2, width/2, width/2, -width/2]}
        J = 0

        if plot:
            sm.plot()
        section = self.get_section(C, I, A, J, pnt)
        return section
    
    def get_section(self, C, I, A, J, pnt):
        section = {'C': C, 'I': I, 'A': A, 'J': J, 'pnt': pnt}
        return section

    def plot_transition(self):
        fig, ax = pl.subplots(1, 1)
        gs = gridspec.GridSpec(3, 3, wspace=0.0, hspace=0.0)
        for i, (fact, txt) in enumerate(zip([0, 0.5, 1],
                                            ['inboard', 'transition',
                                             'outboard'])):
            ax = pl.subplot(gs[i])
            ax.axis('equal')
            ax.axis('off')
            pl.text(0, 0, txt, ha='center', va='center')
            self.case(fact, plot=True, centroid=False)
        ax = pl.subplot(gs[3:6])
        pl.sca(ax)
        N = 100
        Im = np.zeros((3, N))
        J = np.zeros(N)
        fact = np.linspace(0, 1, N)
        for i, f in enumerate(fact):
            section = atec.case(f)
            J[i], I = section['J'], section['I']
            Im[:, i] = I['xx'], I['yy'], I['zz']
        text = linelabel(value='1.2f')
        pl.plot(fact, Im[1, :])
        text.add('Iyy')
        pl.plot(fact, Im[2, :])
        text.add('Izz')
        pl.plot(fact, J)
        text.add('J')
        text.plot()
        sns.despine()
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(['inboard', 'transition', 'outboard'])
        pl.ylabel(r'sectional properties, m$^2$')
        pl.xlabel('cross-section position')

    def get_mat(self, material_name, section):
        material = self.mat_data[material_name]
        E, G, rho = material['E'], material['G'], material['rho']
        mat = {'E': E, 'G': G, 'rho': rho,
               'J': section['J'], 'A': section['A'], 'C': section['C'],
               'Iyy': section['I']['yy'], 'Izz': section['I']['zz']}
        return mat

    #def add_mat(self):


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

if __name__ is '__main__':

    atec = architect(config, setup, nTF=nTF)
    # atec.winding_pack()

    atec.update_cs()  # side=0.1

    atec.case(0.5, plot=True)

    atec.gravity_support(plot=True)
    
    mat = atec.get_mat('wp', atec.winding_pack())
    
    print(mat)
    # atec.plot_transition()
