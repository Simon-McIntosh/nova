import numpy as np
from cross_coil import Bcoil, mu_o, loop, plasma_coils, Bcoil_b
import pylab as pl
from eqlib import eq
import matplotlib
font = {'family': 'serif', 'serif': 'Times', 'weight': 'normal', 'size': 12}
matplotlib.rc('font', **font)
fig_path = '../Figs/'

plasma_on = 1
config = 'DD'
swing = 4

path = './coil_data/'
file = 'Double_Decker_Coils'

eqlib = eq(config)
eqlib.swing(swing)
coil = eqlib.coil

I = 16e6  # plasma current [A]
delta = 0.2  # node spaceing [m]
plasma = loop(path,file,'plasma')
if plasma_on == 1: 
    plasma_coils(plasma, I, delta, coil)
    Fid = 'pFz'
else:
    Fid = 'npFz'
        
ansys_file = 'DDforces.txt'
f = open(path+ansys_file, 'w')  # open file
for name in coil.keys(): 
    B = Bcoil(mu_o, coil, [coil[name]['r'],coil[name]['z']])
    B = np.array([B[0],0,B[1]])
    #B = Bcoil_b(mu_o, coil, [coil[name]['r'],0,coil[name]['z']])

    coil[name]['Fv'] = 2*np.pi*coil[name]['r']*np.cross(B,[0,coil[name]['I'],0])
    
    if name not in 'plasma':
        f.write(name+'_Fx = '+'{:1.3e}'.format(coil[name]['Fv'][0])+'\n')
        f.write(name+'_Fz = '+'{:1.3e}'.format(coil[name]['Fv'][2])+'\n')
        f.write('\n')
f.close()

print(coil['P6']['Fv'])
    
 # create bar plot
Fsimon, Fdave, xlab = [], [], []
for name in sorted(coil.keys()):
    if 'plasma' not in name:
        Fsimon.append(coil[name]['Fv'][2])
        Fdave.append(coil[name][Fid])
        xlab.append(name)
        
ind = np.arange(len(xlab))  
width = 0.4  # bar width
fig = pl.figure(figsize=(9,8))
ax = fig.add_subplot(111)
ax.bar(ind, Fsimon, width, color='r', label='simon')
ax.bar(ind+width, Fdave, width, color='b', label='dave')
ax.set_ylabel('Vertical force [N]')
ax.set_xlabel('Coil')
ax.set_xticks(ind+width)
ax.set_xticklabels(xlab, rotation='vertical')
ax.legend(loc=0, numpoints=1)
pl.ylim([-8e8,12.5e8])
pl.grid()
pl.tight_layout()



if I > 0:
    plasma_dir = 'pos'
else:
    plasma_dir = 'neg'

if plasma_on == 1:
    figname = fig_path+config+'_force_'+plasma_dir+'_swing_'+str(swing)+'.png'
else:
    figname = fig_path+config+'_force_'+'plasma_off'+'.png'    
pl.savefig(figname, dpi=300)


