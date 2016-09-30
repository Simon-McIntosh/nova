from radial_build import RB
from streamfunction import SF
from eqConfig import Config
import pylab as pl
import numpy as np
import seaborn as sns
rc = {'figure.figsize':[3.14*8/12,3.14],'savefig.dpi':300,
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1}
sns.set(context='paper',style='white',font='serif',palette='Set2',
        font_scale=7/8,rc=rc)
        
pl.figure()
pl.axis('equal')

config = 'SN'  # SN,X,SX,SF,SX2

conf = Config(config,inside=False)
sf = SF(conf,sample=2)
sf.contour()
rb = RB(conf,sf,Np=90)
rb.sf.sol(plot=False)

radius,theta = np.array([]),np.array([])
for N in range(rb.sf.Nsol):
    #pl.plot(rb.sf.Rsol[N],rb.sf.Zsol[N])
    pl.plot(sf.Xpoint[0],sf.Xpoint[1],'o',markersize=1)


        
c = sns.color_palette('Set2',4)
if rb.sf.nleg == 6:  # snow flake
    leglist = ['inner1','inner2','outer1','outer2']
else:
    leglist = ['inner','outer']
for N in range(rb.sf.Nsol):
    for i,leg in enumerate(leglist):
        pl.plot(rb.sf.legs[leg]['R'][N],rb.sf.legs[leg]['Z'][N],color=c[i])
        

