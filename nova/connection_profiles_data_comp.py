import pylab as pl
import numpy as np
from itertools import cycle
import seaborn as sns
sns.set_context('poster')
sns.set_style('ticks')
sns.set_palette('Set2')
Color = cycle(sns.color_palette('Set2'))
from eqConfig import Config
from radial_build import RB
from streamfunction import SF
from elliptic import EQ
from addtext import linelabel
import pickle as pk
from scipy.interpolate import interp1d as interp1

path = '../Figs/datamine/'

import seaborn as sns
rc = {'figure.figsize':[6,8],'savefig.dpi':100, # *8/15
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)

pl.figure()
text = linelabel(Ndiv=42,value=' 1.0f',postfix='m',loc='end')

for config in ['SN','SFp','SFm']:#,'SFm','SFp','X','SX8','SXex']: # 
    color = next(Color)
    conf = Config(config)
    sf = SF(conf,sample=1)
    
    rb = RB(conf,sf,Np=250)
    rb.divertor_outline(False,plot=False,debug=False)
    '''
    sf.eq['ncoil'] = 0
    eq = EQ(sf,boundary={'R':rb.Rb,'Z':rb.Zb,'expand':0.5},n=1e4)  #'expand':0.5,'zmax':sf.Xpoint[1]+sf.rcirc+sf.drcirc}
   
    if 'ex' not in config: 
        eq.run()
    eq.set_eq_psi() 
    '''
    sf.sol(Nsol=21,update=True,plot=False)
    rb.trim_sol(plot=False)  # trim legs to targets
    sf.add_core()  # referance low field midplane
    '''
    for i in range(sf.Nsol):
        pl.plot(sf.legs['outer1']['R'][i],sf.legs['outer1']['Z'][i],'k--',alpha=0.15)
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
    config_add = config.replace('m','-')
    config_add = config_add.replace('p','+')
    text.add(config_add+' sm')
    
    if config in ['SN','SFm','SFp']:
        file = 'Louter_vs_rus_2015'
        with open(path+file+'_'+config+'.dat', 'rb') as f: 
            seg = pk.load(f)
        Dsol_hr,connect_hr = seg[:,0],seg[:,1]
        connect = interp1(Dsol_hr,connect_hr)(Dsol[1:])
        #pl.plot(1e3*Dsol_hr,connect_hr,'--',color=color)
        pl.plot(1e3*Dsol[1:],connect,'--',color=color)
        text.add(config_add+' hr')
        
        file = 'CREATE'
        with open(path+file+'_'+config+'.dat', 'rb') as f:  # load segment geometory
            seg = pk.load(f)
        Dsol_vp,connect_vp = seg[:,0],seg[:,1]    
        b = np.log10(10e3/10)/(10e3-10)
        a = 10e3/10**(b*10e3)
        connect_vp = a*10**(b*connect_vp)
        connect = interp1(Dsol_vp,connect_vp)(Dsol[1:])
        pl.plot(1e3*Dsol[1:],connect,':',color=color)
        text.add(config_add+' vp')    


    

#pl.xlim(-0.1,1e3*Dsol[-1])
pl.xticks(np.linspace(0,1e3*Dsol[-1],6))
pl.yscale('log')
pl.ylim([5e1,3e3])
text.plot(yscale='log10')
sns.despine(trim=False)
pl.ylabel(r'Connection, $L3D$ [m]')
pl.xlabel(r'LFS offset, $\Delta r$ [mm]') 

  

#pl.savefig('../Figs/connection_profiles.png',dpi=200)
