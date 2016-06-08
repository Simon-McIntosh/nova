import pylab as pl
import numpy as np
from itertools import cycle
import seaborn as sns
sns.set_context('poster')
sns.set_style('ticks')
sns.set_palette('Set1')
Color = cycle(sns.color_palette('Set1'))
from eqConfig import Config
from radial_build import RB
from streamfunction import SF
from elliptic import EQ
from addtext import linelabel
import pickle as pk
from scipy.interpolate import interp1d as interp1

path = '../Figs/datamine/'

import seaborn as sns
rc = {'figure.figsize':[8,6],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set1',
        font_scale=7/8,rc=rc)

pl.figure()
text = linelabel(Ndiv=30,value=' 1.0f',postfix='m',loc='end')

for config in ['SN','SFp','SFm','X','SX8']:#['SN','SFp','SFm','X','SX8','SXex']:#,'SFm','SFp','X','SX8','SXex']: # 
    color = next(Color)
    conf = Config(config)
    sf = SF(conf,sample=1)

    rb = RB(conf,sf,Np=250)
    rb.divertor_outline(False,plot=False,debug=False)
    '''
    sf.eq['ncoil'] = 0
    eq = EQ(sf,boundary={'R':rb.Rb,'Z':rb.Zb,'expand':0.5},n=5e4)  #'expand':0.5,'zmax':sf.Xpoint[1]+sf.rcirc+sf.drcirc}
   
    if 'ex' not in config: 
        eq.run()
    #eq.set_eq_psi() 
    '''
    sf.sol(Nsol=51,update=True,plot=False)
    rb.trim_sol(plot=False)  # trim legs to targets
    sf.add_core()  # referance low field midplane
    '''
    for i in range(sf.Nsol):
        for target in ['outer1']:
            pl.plot(sf.legs[target]['R'][i],sf.legs[target]['Z'][i],'k--',alpha=0.15)
    pl.axis('equal')
    pl.ylim([-6,-5])
    pl.xlim([7,9])
    '''

    if 'F' in config:
        target = 'outer1'
    else:
        target = 'outer'
    Dsol,connect = np.array([]),np.array([])
    for i in range(sf.Nsol):
        L2D,L3D,r,z = sf.connection(target,i)

        if L3D[-1] > 0:
            Dsol = np.append(Dsol,sf.Dsol[i])
            connect = np.append(connect,L3D[-1])

    imax = np.argmax(connect)
    pl.plot(1e3*Dsol[1:],connect[1:],color=color)
    label = config
    label = label.replace('m','-')
    label = label.replace('p','+')
    label = label.replace('8','')
    text.add(label)
    '''
    if config in ['SN','SFm','SFp']:
        file = 'Louter_vs_rus_2015'
        with open(path+file+'_'+config+'.dat', 'rb') as f: 
            seg = pk.load(f)
        Dsol_hr,connect_hr = seg[:,0],seg[:,1]
        connect = interp1(Dsol_hr,connect_hr)(Dsol[1:])
        #pl.plot(1e3*Dsol_hr,connect_hr,'--',color=color)
        pl.plot(1e3*Dsol[1:],connect,'--',color=color)
        text.add(label+' hr')
    '''
 

#pl.xlim(-0.1,1e3*Dsol[-1])
pl.xticks(np.linspace(0,1e3*Dsol[-1],6))
pl.yscale('log')
pl.ylim([1e2,1e3])
text.plot(yscale='log10')
sns.despine(trim=False)
pl.ylabel(r'Connection, $L3D$ [m]')
pl.xlabel(r'LFS offset, $\Delta r$ [mm]') 

pl.savefig('../Figs/connection_profiles.eps',dpi=200)
