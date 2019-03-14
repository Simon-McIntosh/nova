from nova.coils import PF
from nep.coil_geom import PFgeom, VVcoils, VSgeom
from amigo.pyplot import plt
from nova.finite_inductance import inductance
from nep.DINA.scenario import scenario
from amigo import geom
import nova.cross_coil as cc
from nova.streamfunction import SF
from nova.coilclass import coilclass


class transient(inductance):

    def __init__(self):
        super().__init__()
        self.scn = scenario()  # DINA database instance
        self.load_scenario('17MA DT-DINA2019-01')

    def load_scenario(self, folder):
        self.scn.load_file(folder)

    def build_coils(self, dCoil=0.75):
        nCo = self.nC
        pf_coilset = PFgeom(VS=False, dCoil=dCoil).pf.coilset
        pf_coilset = PF.update_coil_key(pf_coilset, T=22, Ic=0)
        self.add_coilset(pf_coilset)
        self.add_cp([nCo+8, nCo+9], 'CS1')  # link VS coils

    def build_vessel(self, ncluster=1):
        vv_coilset = VVcoils(model='full').pf.coilset
        vv_coilset = PF.update_coil_key(vv_coilset, T=100, Ic=0)
        if ncluster > 1:  # cluster vessel
            vv_coilset = PF.cluster_coilset(vv_coilset, ncluster=ncluster)
        self.add_coilset(vv_coilset)

    def build_vs_coils(self):
        nCo = self.nC
        vs_coilset = VSgeom(jacket=True).pf.coilset
        vs_coilset = PF.update_coil_key(vs_coilset, T=100, Ic=0)
        self.add_coilset(vs_coilset)
        self.add_cp([nCo, nCo+1], 'VS3')  # link VS coils

    def build_plasma(self, t, tau=16e-3, dx=2.5, dz=2.5):
        pp = self.scn.extract_plasma(t)
        R = pp['lp'] / tau
        plasma_coil = {'x': pp['xcur'], 'z': pp['zcur'],
                       'It': pp['Ipl'], 'dx': dx, 'dz': dz, 'L': pp['lp'],
                       'cross_section': 'circle'}
        pl_coilset = PF.initalize_coilset()
        pl_coilset['coil']['Plasma'] = plasma_coil
        pl_coilset['subcoil']['Plasma_0'] = PF.mesh_coil(plasma_coil)[0]
        pl_coilset = PF.update_coil_key(pl_coilset, T=None, Ic=pp['Ipl'],
                                        R=R, material=None, m=None)
        pl_coilset['index']['plasma'] = {
                'index': [0], 'name': 'Plasma', 'n': 1}
        self.add_coilset(pl_coilset)

    def build(self, t):
        self.build_plasma(t)
        self.build_coils()
        # self.build_vs_coils()
        # self.build_vessel()
        self.pf.update_current(self.scn.get_coil_current(t))

    def plot_M(self):
        plt.figure()
        plt.pcolormesh(abs(self.M.iloc[:20, :20]), cmap=plt.cm.gray,
                       edgecolors='k')
        plt.axis('square')
        plt.axis('off')

    def get_eqdsk(self):
        eqdsk = geom.grid(5e3, [3.5, 8.6, -4.5, 4.5], eqdsk=True)
        eqdsk['psi'] = cc.get_coil_psi(eqdsk['x2d'], eqdsk['z2d'],
                                       self.pf.coilset['subcoil'],
                                       self.pf.coilset['plasma'])
        return eqdsk



if __name__ is '__main__':

    trans = transient()

    trans.build(200)
    trans.reduce()

    # trans.plot_M()
    # print(trans.M)

    eqdsk = trans.get_eqdsk()
    sf = SF(eqdsk=eqdsk)

    trans.plot(subcoil=True)
    sf.contour()


    bs = cc.biot_savart(coilset=trans.pf.coilset)
    #cc.green_((eqdsk['x2d'], eqdsk['z2d']), trans.pf.coilset['subcoil'])



    #


    '''
    scn.update_scenario(t=690)
    scn.update_psi(plot=True, current='AT', n=5e3)
    '''
