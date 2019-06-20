from nep.coil_geom import PFgeom
from nova.elliptic import EQ
from nep.DINA.read_eqdsk import read_eqdsk
from amigo.pyplot import plt
from nova.biot_savart import biot_savart
from nova.coil_class import CoilClass

cc = PFgeom(dCoil=0.25).cc

# cc = CoilClass(dCoil=0.25)
# cc.add_coil(5.5, 1, 2.5, 2.5, dCoil=0.1, name='PF1')
# cc.add_mpc(['PF1', 'new'], 1)  # link new to 'CS1L'

cc.plot()

# cc.calculate_inductance(coil_index='Plasma')


cc.scenario_filename = -1
cc.scenario = 'SOF'


'''
cc.calculate_inductance(coil_index='Plasma')

cc.scenario = 'EOF'
'''



# cc.calculate_flux_interaction(n=5e3, limit=[4, 8.25, -3, 4])

'''
eqdsk = read_eqdsk(file='burn').eqdsk  # 'burn', 'inductive'
eqdsk['Ip'] = cc.d2.Ip  # update plasma current
eq = EQ(cc.coilset, eqdsk, n=3e3)  # set plasma coils
'''

#cc.calculate_inductance(coil_index='Plasma')
#print(cc.inductance['Mt'].loc['Plasma', 'Plasma'])

# 721ms per loop
# 236ms with patch on-demand
# eq.run(update=False)

'''
cc.scenario_filename = -1
cc.scenario = 'XPF'
cc.drop_coil(index='Plasma')

ax = plt.subplots(1, 1, figsize=(7, 9))[1]
cc.plot()
cc.solve_grid(plot=True, nlevels=31, limit=[1.5, 8.5, -2, 3], n=1e3)
ax.axis('equal')
ax.axis('off')

'''


'''


plt.figure(figsize=(8, 10))
cc.plot(label=True, current=True, unit='A', plasma=True)
eq.sf.contour()

biot_savart(cc.coilset).inductance()
print(cc.coilset.matrix['inductance']['Mc'])
'''

