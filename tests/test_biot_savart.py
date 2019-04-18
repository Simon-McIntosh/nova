from nova.coil_class import CoilClass
from nova.biot_savart import biot_savart
from numpy import allclose


def test_inductance(plot=False):
    '''
    test inductance calculation against DDD values for 2 CS and 1 PF coil
    baseline (old) CS geometory used
    '''
    cc = CoilClass(dCoil=0.25)
    cc.add_coil(3.9431, 7.5641, 0.9590, 0.9841, Nt=248.64,
                name='PF1', part='PF')
    cc.add_coil(1.722, 5.313, 0.719, 2.075, Nt=554, name='CS3U', part='CS')
    cc.add_coil(1.722, 3.188, 0.719, 2.075, Nt=554, name='CS2U', part='CS')
    biot_savart(coilset=cc.coilset, mutual=False).inductance()
    Mc_bs = cc.coilset.matrix['inductance']['Mc'].values  # calculated
    Mc_ddd = [[7.076E-01, 1.348E-01, 6.021E-02],
              [1.348E-01, 7.954E-01, 2.471E-01],
              [6.021E-02, 2.471E-01, 7.954E-01]]
    assert allclose(Mc_ddd, Mc_bs, atol=5e-3)
    if plot:
        cc.plot()


if __name__ is '__main__':

    test_inductance(plot=True)
