import pylab as pl
import numpy as np
from matplotlib.animation import FuncAnimation
from shelf import PKL
import pickle
import seaborn as sns
rc = {'figure.figsize':[8,8*4/12],'savefig.dpi':125, #*12/16
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':1.75}
sns.set(context='poster',style='white',font='sans-serif',palette='Set2',
        font_scale=1,rc=rc)
color = sns.color_palette('Set2')
from addtext import linelabel
from itertools import cycle
Color = cycle(sns.color_palette('Set2'))
from radial_build import RB

pkl = PKL('moveSX')
sf,eq,inv = pkl.fetch(['sf','eq','inv'])

Swing =  np.linspace(np.max(inv.Swing),np.min(inv.Swing),50)#-10
nSwing = len(Swing)
nPF = len(inv.PF_coils)
F = np.zeros((nPF,nSwing,2))
I = np.zeros((nPF,nSwing))
Fcs = np.zeros((nSwing,2))

for j,swing in enumerate(Swing):
    inv.swing_fix(swing)
    inv.solve_slsqp()
    print(inv.Fsep*1e-6)
    for i,name in enumerate(inv.PF_coils):
        F[i,j,0] = inv.coil['active'][name]['Fr_sum']
        F[i,j,1] = inv.coil['active'][name]['Fz_sum']
        I[i,j] = inv.coil['active'][name]['I_sum']
    Fcs[j,0] = inv.Fsep
    Fcs[j,1] = inv.FzCS
    
pl.subplots()
text = linelabel(Ndiv=7,value='',postfix='') 
for i,name in enumerate(inv.PF_coils):
    print(name, np.max(abs(F[i,:,1])*1e-6))
    pl.plot(-2*np.pi*(Swing-Swing[0]),abs(F[i,:,1])*1e-6)
    text.add(name)
pl.plot(-2*np.pi*(Swing-Swing[0]),Fcs[:,0]*1e-6)
text.add('Fsep')
pl.plot(-2*np.pi*(Swing-Swing[0]),abs(Fcs[:,1])*1e-6)
text.add('FzCS')
pl.ylabel(r'$|Fz|$ MN')
pl.ylim([0,350])
pl.xlabel(r'Swing Wb')
pl.tight_layout()
#pl.xlabel(r'Swing Wb')
sns.despine()
text.plot()
pl.tight_layout(rect=[0,0,0.95,1])
pl.savefig('../Figs/SX_forces.png',dpi=300)

pl.subplots()
text = linelabel(Ndiv=7,value='',postfix='') 
for i,name in enumerate(inv.PF_coils):
    pl.plot(-2*np.pi*(Swing-Swing[0]),abs(I[i,:])*1e-6)
    text.add(name)
pl.ylabel(r'$|I|$ MA')
pl.xlabel(r'Swing Wb')
pl.tight_layout()
sns.despine()
text.plot()
pl.tight_layout(rect=[0,0,0.95,1])
pl.savefig('../Figs/SX_currents.png',dpi=300)