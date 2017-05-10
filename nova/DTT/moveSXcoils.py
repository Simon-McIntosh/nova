from nova.streamfunction import SF
from nova.elliptic import EQ
from nova.inverse import INV
from nova.config import Setup, select
from itertools import cycle
import numpy as np
from nova.radial_build import RB
from nova.shelf import PKL
from nova.coils import PF, TF
from nova import loops
from nova.loops import Profile
from nova.inverse import SWING
from nova.firstwall import main_chamber
from nova.shape import Shape

import seaborn as sns
sns.set(context='talk', style='white', font='sans-serif', font_scale=7 / 8)
Color = cycle(sns.color_palette('Set2'))

pkl = PKL('SXdev', directory='../../Movies/')

name = 'SN'

nTF, ripple = 18, True
base = {'TF': 'dtt', 'eq': 'DTT_SN', 'name': name}
config, setup = select(base, nTF=nTF, update=False)

sf = SF(setup.filename)
pf = PF(sf.eqdsk)


for coil in pf.index['CS']['name']:
    pf.coil[coil]['r'] = 2.7
    pf.coil[coil]['dr'] = 0.8

profile = Profile(config['TF'], family='S', part='TF', nTF=nTF, obj='L')
tf = TF(profile=profile, sf=sf)


inv = INV(pf, tf, dCoil=2.5)
inv.colocate(sf, n=1e3, expand=0.5, centre=0, width=363/(2*np.pi))

#inv.set_limit(FPFz=50)

inv.grid_PF(nPF=4)


#inv.wrap_PF(solve=True)

inv.limit['L']['Coil8'] = [0.98,0.99]
inv.optimize()
#

#pf.coil['Coil5']['I'] *= 1.3



pf.plot(label=True, current=True)
pf.plot(subcoil=True, plasma=True)
tf.fill()

inv.eq.run(update=False)
sf.contour()

inv.ff.plot()
inv.plot_fix()
inv.plot_coils()

#loops.plot_variables(inv.Io, scale=1, postfix='MA')
#loops.plot_variables(inv.Lo, scale=1)


'''
Lpf = inv.grid_PF(nPF=4)
Lcs = inv.grid_CS(nCS=5,Zbound=[-10,8],gap=0.1,Ro=2.7,dr=0.8)
L = np.append(Lpf,Lcs)
inv.set_Lo(L)  # set position bounds

Lnorm = loops.normalize_variables(inv.Lo)


inv.set_swing(centre=0, width=10, array=np.linspace(-0.5, 0.5, 3))
inv.set_force_feild()
inv.update_position(Lnorm, update_area=True)
#inv.optimize(Lnorm)


#swing = SWING(inv,sf,plot=True)
eq.run(update=False)
sf.contour()
pf.plot(subcoil=False, current=True, label=True)
pf.plot(subcoil=True)

inv.ff.plot()

mc = main_chamber('dtt', date='2017_03_08')
mc.load_data(plot=True)  # load from file
mc.shp.plot_bounds()


rb = RB(sf, Setup(name))
rb.generate(mc, debug=False)
#profile = Profile(config['TF'],family='S',part='TF',nTF=nTF,obj='L')
shp = Shape(tf.profile, eqconf=config['eq_base'], ny=3)
shp.add_vessel(rb.segment['vessel_outer'])
# shp.minimise(ripple=ripple,verbose=True)
shp.tf.fill()
'''

'''
inv.Lnorm = Lnorm
sw = SWING(inv,sf)
sw.flat_top()
sw.output()

for ends,name in zip([0,-1],['SOF','EOF']):
    inv.solve_slsqp(inv.swing['flux'][ends])
    eq.run(update=False)
    sf.eqwrite(pf,config='SXex_{}'.format(name))
'''

'''
rb.write_json(tf=shp.tf)

loops.plot_variables(inv.Io, scale=1, postfix='MA')
loops.plot_variables(inv.Lo, scale=1)

sf.eqwrite(pf, config='SXex')
pkl.write(data={'sf': sf, 'eq': eq, 'inv': inv})  # pickle data
'''

'''
eq.gen_opp()

#rb.firstwall(calc=False,plot=True,debug=False)


inv = INV(sf,eq,tf)


Lpf = inv.grid_PF(nPF=5)
Lcs = inv.grid_CS(nCS=3,Zbound=[-12,8],gap=0.1)
Lo = np.append(Lpf,Lcs)
inv.update_coils()

inv.fit_PF(offset=0.3)  # fit PF coils to TF
inv.fix_boundary_psi(N=31,alpha=1-1e-4,factor=1)  # add boundary points
inv.fix_boundary_feild(N=31,alpha=1-1e-4,factor=1)  # add boundary points
inv.add_null(factor=3,point=sf.Xpoint)

Rex,arg = 1.5,40
R = sf.Xpoint[0]*(Rex-1)/np.sin(arg*np.pi/180)
target = (R,arg)
inv.add_alpha(1,factor=1,polar=target)  # X-point psi
inv.add_B(0,[-15],factor=1,polar=target)  # X-point feild

inv.set_swing()
inv.update_limits(LCS=[-12,14])
Lo = inv.optimize(Lo)

inv.fix_flux(inv.swing['flux'][0])
inv.solve_slsqp()

eq = EQ(sf,pf,dCoil=2,sigma=0,boundary=tf.get_loop(expand=0),n=2e4)

eq.get_Vcoil()
eq.gen_opp()
sf.contour()

tf.fill()
pf.plot(coils=pf.coil,label=True,plasma=True,current=True)
inv.plot_fix(tails=True)

loops.plot_variables(inv.Io,scale=1,postfix='MA')
loops.plot_variables(inv.Lo,scale=1)

#
'''

'''

#inv.write_swing()
#sf.eqwrite(config='SXex')

for swing in np.linspace(-20,80,5):
    pl.figure()
    pl.axis('equal')
    pl.axis('off')

    inv.swing_fix(swing)
    inv.solve()

    inv.update_coils(plot=True)
    sf.plot_coils(Color,coils=sf.coil,label=False,plasma=False,current=True)
    sf.plot_coils(Color,coils=eq.coil,label=False,plasma=False)

    eq.run()

    sf.contour()
    eq.plasma()
    #eq.plotb()
    #sf.eqwrite(config='SXex')
    pl.plot(sf.rbdry,sf.zbdry,'--')
    inv.plot_fix()

print('L3D',inv.rb.sol.connection('outer',0)[-1][-1])
print('R',Rsol[-1])
print('R/X',Rsol[-1]/sf.Xpoint[0])
print('Itotal',inv.Itotal*1e-6,'MA')
print('R',rb.targets['outer']['Rsol'][-1],'Z',
      rb.targets['outer']['Zsol'][-1])
'''
