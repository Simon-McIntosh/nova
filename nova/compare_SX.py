import pylab as pl
import numpy as np
import matplotlib as mpl
from itertools import cycle
from eqConfig import Config
from radial_build import RB
from streamfunction import SF

import seaborn as sns
rc = {'figure.figsize':[2*3.14*12/16,2*3.14],'savefig.dpi':150, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.3}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
Color = cycle(sns.color_palette('Set2',3))

figheight = 8.0
mpl.rcParams['figure.figsize'] = [figheight,figheight*5/9]
xlim,ylim = [8,12],[-8,-7]

pl.figure()
pl.axis('equal')
pl.axis('off')
pl.xlim(xlim)
pl.ylim(ylim)



conf = Config('SX7')
sf = SF(conf,sample=1)
conf.TF(sf)
rb = RB(conf,sf,Np=1e3)
rb.divertor_outline(True,color=next(Color))
Rt,Zt = rb.targets['outer']['R'],rb.targets['outer']['Z']
Ro,Zo = rb.targets['outer']['Rsol'][-1],rb.targets['outer']['Zsol'][-1]
index = np.argmin((Rt-Ro)**2+(Zt-Zo)**2)
R,Z = Rt[:index+1],Zt[:index+1]
graze = sf.get_graze([Ro,Zo],[R[-2]-R[-1],Z[-2]-Z[-1]])
print(graze*180/np.pi,rb.targets['outer']['theta']*180/np.pi)

conf = Config('SX8')
sf = SF(conf,sample=1)
conf.TF(sf)
rb = RB(conf,sf,Np=1e3)
rb.divertor_outline(True,color=next(Color))
Rt,Zt = rb.targets['outer']['R'],rb.targets['outer']['Z']
Ro,Zo = rb.targets['outer']['Rsol'][-1],rb.targets['outer']['Zsol'][-1]
index = np.argmin((Rt-Ro)**2+(Zt-Zo)**2)
R,Z = Rt[:index+1],Zt[:index+1]
graze = sf.get_graze([Ro,Zo],[R[-1]-R[-2],Z[-1]-Z[-2]])
print(graze*180/np.pi,rb.targets['outer']['theta']*180/np.pi)

filename = '2015_First_Wall_and_Divertor_CCFE_2MKUGH_v1_0.txt'
#filename = 'SX8_bdry.txt'
with open('../Data/'+filename) as f:
    for i in range(2):    
        f.readline()
    if len(f.readline().split()) > 0:
        f.readline()    
    f.readline()
    if f.readline().split()[0] == 'inner1':
        nskip = 3
    else:
        nskip = 1
    for i in range(nskip):    
        f.readline()
    for i in range(3):    
        f.readline()
    n = int(f.readline())
    r,z = np.zeros(n),np.zeros(n)
    for i,line in enumerate(f):
        data = line.split()
        if len(data) == 2:
            r[i],z[i] = data[0],data[1]
        else:
            r[i],z[i] = data[0],data[2]
pl.plot(r,z,'.-',color=next(Color),markersize=5)

pl.legend(['SX7','SX8','IDM'])