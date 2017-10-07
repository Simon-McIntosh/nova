import numpy as np
from amigo.pyplot import plt
from nova.finite_element import FE
from nova.config import select
from nova.coils import PF
from nova.inverse import INV
from nova.coils import TF
from nova.loops import Profile
from nova.structure import architect
from nova.streamfunction import SF
from amigo.time import clock
from amigo import geom
import matplotlib.animation as manimation
from mpl_toolkits.mplot3d import Axes3D
from blueprint.CAD.buildCAD import buildCAD
from blueprint.CAD import coilCAD


class SS:  # structural solver
    '''
    Un structural solver incroyable
    Le passé est l'avenir!
    '''
    def __init__(self, sf, pf, tf):
        self.sf = sf
        self.pf = pf
        self.tf = tf

        self.inv = INV(pf, tf, dCoil=2.5, offset=0.3)
        self.inv.colocate(sf, n=1e3, expand=0.5, centre=0, width=363/(2*np.pi))
        self.inv.wrap_PF(solve=False)

        self.atec = architect(tf, pf, plot=False)  # load sectional properties
        self.fe = FE(frame='3D')  # initalise FE solver
        self.add_mat()  # pass sectional properties to fe solver

        self.build_tf()  # build TF coil
        if True:
        # self.fe.add_bc(['ny'], [0], part='trans_lower', ends=0)
        # self.fe.add_bc(['ny'], [-1], part='trans_upper', ends=1)
            self.fe.add_bc(['fix'], [-1], part='nose', ends=2)
            self.fe.add_bc(['fix'], [0], part='nose', ends=0)
        else:
            self.gravity_support()  # add gravity support to TF loop
            self.outer_intercoil_supports()  # add outer intercoil supports
            self.connect_pf()  # connect PF coils

    def add_mat(self, ntrans=30):
        self.fe.add_mat('nose', ['wp', 'steel_forged'],  # high field TF
                        [self.atec.winding_pack(), self.atec.case(0)])
        self.fe.add_mat('loop', ['wp', 'steel_cast'],  # low field TF
                        [self.atec.winding_pack(), self.atec.case(1)])
        trans = []  # nose-loop transitions
        for i, l_frac in enumerate(np.linspace(0, 1, ntrans)):
            trans.append(self.fe.add_mat('trans_{}'.format(i),
                                         ['wp', 'steel_cast'],
                                         [self.atec.winding_pack(),
                                          self.atec.case(l_frac)]))
        self.fe.mat_index['trans_lower'] = [trans[-1]]  # trans
        self.fe.mat_index['trans_upper'] = [trans[-1]]  # trans[::-1]
        self.fe.add_mat('GS', ['steel_cast'], [self.atec.gravity_support()])
        for name in self.atec.OICsupport:  # outer intercoil supports
            mat_name = name
            thickness = self.atec.OICsupport[name]['thickness']
            width = self.atec.OICsupport[name]['width']
            self.fe.add_mat(mat_name, ['steel_cast'],
                            [self.atec.intercoil_support(thickness, width)])
        for name in self.atec.PFsupport:  # PF coil connection sections
            mat_name = 'PFS_{}'.format(name)
            self.fe.add_mat(mat_name, ['steel_cast'],
                            [self.atec.coil_support(
                                    width=self.atec.PFsupport[name]['width'])])

    def build_tf(self):
        if self.fe.nnd != 0:
            errtxt = 'pre-exsisting nodes found in fe object\n'
            errtxt = 'TF connections (OIC + GS) assume clean entry\n'
            errtxt = 'clear mesh with fe.clfe()'
            raise ValueError(errtxt)
        P = np.zeros((len(self.tf.p['cl_fe']['x']), 3))
        P[:, 0], P[:, 2] = self.tf.p['cl_fe']['x'], self.tf.p['cl_fe']['z']
        self.fe.add_nodes(P)  # all TF nodes
        self.TFparts = ['nose', 'trans_lower', 'loop', 'trans_upper']  #
        for part in self.TFparts:  # hookup elements
            self.fe.add_elements(n=tf.p[part]['nd'], part_name=part, nmat='loop')  # , nmat=part)
        # constrain TF nose - free translation in z
        self.fe.add_bc('nw', 'all', part='nose')

    def connect_pf(self):
        for name in self.atec.PFsupport:  # PF coil connections
            nd_tf = self.atec.PFsupport[name]['nd_tf']
            p = self.atec.PFsupport[name]['p']  # connect PF to TF
            self.fe.add_nodes([p['x'][1], 0, p['z'][1]])
            self.fe.add_nodes([p['x'][0], 0, p['z'][0]])
            self.atec.PFsupport[name]['nd'] = self.fe.nnd-1  # store node index
            mat_name = 'PFS_{}'.format(name)
            # self.fe.add_elements(
            #         n=[nd_tf, self.fe.nnd-1], part_name=name, nmat=mat_name)
            self.fe.add_elements(
                    n=[self.fe.nnd-2,  self.fe.nnd-1], part_name=name,
                    nmat=mat_name)
            self.fe.add_cp([nd_tf,  self.fe.nnd-2], dof='fix', rotate=False)

    def add_pf_load(self):
        self.inv.ff.get_force()
        for name in self.atec.PFsupport:  # PF coil connections
            Fcoil = self.inv.ff.Fcoil[name]
            self.fe.add_nodal_load(
                    self.atec.PFsupport[name]['nd'], 'fz', 1e6*Fcoil['fz'])

    def gravity_support(self, nGS=7):
        yGS = np.linspace(self.atec.Gsupport['yfloor'], 0, nGS)
        zGS = np.linspace(self.atec.Gsupport['zfloor'],
                          self.atec.Gsupport['zbase'], nGS)
        for ygs, zgs in zip(yGS, zGS):
            self.fe.add_nodes([self.atec.Gsupport['Xo'], ygs, zgs])
        yGS = np.linspace(0, -self.atec.Gsupport['yfloor'], nGS)
        zGS = np.linspace(self.atec.Gsupport['zbase'],
                          self.atec.Gsupport['zfloor'], nGS)
        for ygs, zgs in zip(yGS[1:], zGS[1:]):
            self.fe.add_nodes([self.atec.Gsupport['Xo'], ygs, zgs])
        self.fe.add_elements(
                n=range(self.fe.nnd-2*nGS+1, self.fe.nnd),
                part_name='GS', nmat='GS')

        # couple GS to TF loop
        self.fe.add_cp([self.atec.Gsupport['nd_tf'], self.fe.nnd-nGS],
                       dof='nrx', rotate=False)
        self.fe.add_bc(['nry'], [0], part='GS', ends=0)  # free rotation in y
        self.fe.add_bc(['nry'], [-1], part='GS', ends=1)

    def outer_intercoil_supports(self, nIC=3):
        for name in self.atec.OICsupport:
            nd_tf = self.atec.OICsupport[name]['nd_tf']
            el_dy = self.atec.OICsupport[name]['el_dy']
            Xm = geom.qrotate(self.fe.X[nd_tf[1]], -np.pi/self.tf.nTF,  # minus
                              xo=[0, 0, 0], dx=[0, 0, 1])
            Xp = geom.qrotate(self.fe.X[nd_tf[1]], np.pi/self.tf.nTF,  # plus
                              xo=[0, 0, 0], dx=[0, 0, 1])
            Xrotate = np.zeros((2*nIC, 3))
            for i in range(3):
                Xrotate[:nIC, i] = np.linspace(
                        Xm[0, i], self.fe.X[nd_tf[1]][i],
                        nIC+1)[:-1]
                Xrotate[nIC:, i] = np.linspace(
                        self.fe.X[nd_tf[1]][i], Xp[0, i],
                        nIC+1)[1:]
            self.fe.add_nodes(Xrotate)
            nodes = list(range(self.fe.nndo, self.fe.nndo+nIC)) + [nd_tf[1]]
            nodes += list(range(self.fe.nnd-nIC, self.fe.nnd))
            self.fe.add_elements(
                n=nodes,
                part_name=name, nmat=name, el_dy=el_dy)
            self.fe.add_cp([nodes[0], nodes[-1]],
                           dof='fix', rotate=True, axis='z')

    def solve(self):
        self.fe.clf()  # clear forces
        #self.add_pf_load()
        #self.fe.add_weight()  # add weight to all elements
        self.fe.add_tf_load(self.sf, self.inv.ff, self.tf,
                            self.inv.eq.Bpoint, parts=self.TFparts,
                            method='function')
        self.fe.solve()

    def plot3D(self, ax=None):
        self.fe.plot_3D(ax=ax, nTF=self.tf.nTF)

    def movie(self, filename, scale=15, nscale=20, nloop=3):
        single = np.linspace(0, scale, nscale)
        cycle = np.append(single, single[::-1][1:])
        loop = np.array([])
        for i in range(nloop):
            loop = np.append(loop, cycle[1:])
        moviename = '../Movies/{}'.format(filename)
        moviename += '.mp4'
        FFMpegWriter = manimation.writers['ffmpeg']
        writer = FFMpegWriter(fps=20, bitrate=5000, codec='libx264',
                              extra_args=['-pix_fmt', 'yuv420p'])
        timer = clock(len(loop))

        # fig = plt.figure()
        fig = plt.figure(figsize=(8, 8))
        ax = Axes3D(fig)
        with writer.saving(fig, moviename, 100):
            for scale in loop:
                ax.clear()
                self.fe.deform(scale)
                self.plot3D(ax=ax)
                writer.grab_frame()
                timer.ticktoc()


if __name__ == '__main__':

    nTF = 16
    base = {'TF': 'demo', 'eq': 'SN'}
    config, setup = select(base, nTF=nTF, update=False)
    profile = Profile(config['TF_base'], family='D', load=True,
                      part='TF', nTF=nTF, obj='L', npoints=501)

    sf = SF(setup.filename)
    pf = PF(sf.eqdsk)
    tf = TF(profile=profile, sf=sf)

    ss = SS(sf, pf, tf)  # structural solver
    ss.solve()
    ss.fe.plot()

    ss.fe.plot_moment()

    '''
    TFcoil = coilCAD.TFcoilCAD(ss.atec)
    model = buildCAD(ss.tf.nTF)
    model.add_part(TFcoil)
    model.display(1, output='web')
    '''

    # ss.plot3D()
    # ss.fe.plot_sections()
    # ss.movie('structural_solver')
    # ss.fe.plot_curvature()
    # ss.fe.plot_twin()

    '''
    plt.figure(figsize=(16, 16))
    plt.pcolor(abs(ss.fe.K), cmap=plt.cm.gray, vmax=1)
    plt.axis('equal')
    plt.axis('off')
    '''
