from nep.coil_geom import PFgeom
from nova.elliptic import EQ
from nep.DINA.read_eqdsk import read_eqdsk
from amigo.pyplot import plt
from nova.biot_savart import biot_savart


cc = PFgeom(dCoil=0.15).cc
# cc.add_coil(6.5, 1, 0.5, 0.5, dCoil=0.25, name='new')
# cc.add_mpc(['PF1', 'new'], 1)  # link new to 'CS1L'


cc.scenario_filename = -2
cc.scenario = 'EOF'


eqdsk = read_eqdsk(file='burn').eqdsk  # 'burn', 'inductive'
eqdsk['Ip'] = cc.d2.Ip  # update plasma current
eq = EQ(cc.coilset, eqdsk, n=3e3)  # set plasma coils

# 721ms per loop
eq.run(update=False)
'''


plt.figure(figsize=(8, 10))
cc.plot(label=True, current=True, unit='A', plasma=True)
eq.sf.contour()

biot_savart(cc.coilset).inductance()
print(cc.coilset.matrix['inductance']['Mc'])
'''

