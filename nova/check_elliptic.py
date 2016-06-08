import pylab as pl
import numpy as np
from streamfunction import SF
from elliptic import grid
import cross_coil as cc
from eqConfig import Config
from itertools import cycle
import seaborn as sns
sns.set_context('talk')
sns.set_style('white')
Color = cycle(sns.color_palette('Set2'))

config = 'SN'  # SN,X,SX,SF
conf = Config(config)
sf = SF(conf)
pl.axis('equal')
sf.contour()



sf.coil['plasma'] = {'I':20.254e6,'r':sf.rmagx,'z':sf.zmagx,'dr':3,'dz':3}
                
#del sf.coil['plasma']
#sf.coil['Coil10']['I']*=2

psi = grid([4,12],[-8,8],1500,sf)
psi.coils(delta=0.5)
psi.rbdry,psi.zbdry = np.copy(sf.rbdry),np.copy(sf.zbdry)
#psi.Xpsi = sf.get_Xpsi()[0]

psi.coil_currents()


'''
for i in range(10):
    psi.coil_currents()
    psi.run()
    print(psi.Xpsi)
'''
'''
pl.plot(psi.sf.rbdry,psi.sf.zbdry)
pl.plot(psi.rbdry,psi.zbdry)


psi.coils(delta=0.25)
psi.run()
'''

'''
psi.coil_currents()
psi.coils(delta=0.25)
'''

'''
psi.run()
psi.psi *= 2*np.pi  # scale
psi.sf.update_plasma({'r':psi.r,'z':psi.z,'psi':psi.psi})
psi.psi -= psi.sf.get_Xpsi()[0]  # shift
psi.sf.update_plasma({'r':psi.r,'z':psi.z,'psi':psi.psi})
psi.plot(levels=sf.cs.levels)  #
sf.plot_coils(Color,coils=psi.coil,label=False)


pl.savefig('../Figs/Coil_forces.png',dpi=200)
'''
#psi.check_psi()
'''
pl.figure()
psi.plot(levels=sf.cs.levels) 


#pl.figure()
for i in range(150):
    psi.coreBC()
    psi.edgeBC()
    psi.solve()
    pl.plot(psi.sf.rbdry,psi.sf.zbdry)

#pl.figure()
psi.psi *= 2*np.pi  # scale
psi.sf.update_plasma({'r':psi.r,'z':psi.z,'psi':psi.psi})
psi.psi -= psi.sf.get_Xpsi()[0]  # shift
psi.sf.update_plasma({'r':psi.r,'z':psi.z,'psi':psi.psi})

psi.plot(levels=sf.cs.levels)
pl.plot(psi.sf.rbdry,psi.sf.zbdry)

sf.plot_coils(Color,coils=psi.coil)
pl.axis('equal')
pl.xlim([3,13])
pl.ylim([-6,6])

sf.get_boundary()  # update boundary
R,Z = psi.inloop(psi.sf.rbdry,psi.sf.zbdry,
             psi.r2d.flatten(),psi.z2d.flatten())

#pl.figure()
pl.plot(psi.sf.rbdry,psi.sf.zbdry)
#pl.plot(R,Z,'bd')



pl.figure()
#psi.b[psi.indx(0,np.arange(psi.nz))] = 0
#psi.b[psi.indx(np.arange(psi.nr),0)] = 0
#psi.b[psi.indx(psi.nr-1, np.arange(psi.nz))] = 0
#psi.b[psi.indx(np.arange(psi.nr),psi.nz-1)] = 0
pl.contourf(psi.r,psi.z,np.reshape(psi.b,(psi.nr,psi.nz)).T,cmap=pl.cm.RdBu)
pl.plot(psi.sf.rbdry,psi.sf.zbdry)
pl.colorbar()





pl.figure()
pl.contourf(psi.r,psi.z,np.reshape(psi.b,(psi.nr,psi.nz)).T,cmap=pl.cm.RdBu)
pl.colorbar()



    psi.solve()

    pl.figure()
    psi.plot(levels=sf.cs.levels) 



#sf.plot_coils(Color,coils=psi.coil)

pl.axis('equal')
pl.xlim([3,13])
pl.ylim([-6,6])

'''

