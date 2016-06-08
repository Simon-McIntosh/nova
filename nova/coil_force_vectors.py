import pylab as pl
import numpy as np
from config import layout
from extract import extract_geom
from streamfunction import SF
from cross_coil import Fcoil
import pickle
from radial_build import RB
from feild_calc import solCalc
import matplotlib
font = {'family': 'serif', 'serif': 'Arial', 'weight': 'normal', 'size': 16}
matplotlib.rc('font', **font)
fig_path = '../Figs/'

config = 'DD3'  # DD1/DD3


with open('./plot_data/'+config+'_sol.pkl', 'rb') as input:
    sf = pickle.load(input)
    plot = pickle.load(input)
    geom = pickle.load(input)
    


sol = solCalc(sf,geom,config)
rb = RB(geom,sf,config,Np=400)
divertor_coils = ['P6','P5','PS2','P6B','P5B','PS2','PS5']


for name in divertor_coils: 
    geom.coil[name]['Fr'],geom.coil[name]['Fz'] = Fcoil(geom.coil, name)

fig = pl.figure(figsize=(10,6))
plot = layout(geom)
plot.coil_keys = divertor_coils

#plot.sol()  # scrape-off layer
#plot.plates()  # divertor plates
#plot.TF()  # TF coil
#plot.FW()
#plot.plasma()
plot.coil_fill()
plot.coil_current([0,20],[-11,-2])
plot.coil_force([0,20],[-11,-2])
#plot.coil_sheild()

sf.sol()


Fmax = 100e6
scale = 1
coil = geom.coil
for name in divertor_coils: 
    F = (coil[name]['Fr']**2+coil[name]['Fz'])**0.5
    print(name, ' Fr:', '{:1.2f}'.format(coil[name]['Fr']*1e-6/18), 'MN,',
          ' Fz:', '{:1.2f}'.format(coil[name]['Fz']*1e-6/18), 'MN')
Fmax = Fmax/scale

for name in divertor_coils:
    pl.arrow(coil[name]['r'], coil[name]['z'], 
             coil[name]['Fr']/Fmax, coil[name]['Fz']/Fmax, fc="k", ec="k",
             head_width=0.15, head_length=0.2)
    pl.arrow(coil[name]['r'], coil[name]['z'], 
             coil[name]['Fr']/Fmax, 0, fc="k", ec="k",alpha=0.5,
             head_width=0.15, head_length=0.2)
    pl.arrow(coil[name]['r'], coil[name]['z'], 
             0, coil[name]['Fz']/Fmax, fc="k", ec="k",alpha=0.5,
             head_width=0.15, head_length=0.2)

pl.axis('equal')
pl.xlim([3,14])
pl.ylim([-10,-3])
pl.xlabel('radial coordinate, r [m]')
pl.ylabel('vertical coordinate, z [m]')
#pl.grid()
pl.tight_layout()

pl.axis('off')
pl.savefig(fig_path+'Force_vectors_'+config+'.png',bbox_inches='tight', dpi=600)

ansys_file = 'DDforces.txt'
f = open('coil_data/'+ansys_file, 'w')  # open file
for name in divertor_coils: 
    f.write(name+'_r = '+'{:1.3e}'.format(coil[name]['r'])+'\n')
    f.write(name+'_z = '+'{:1.3e}'.format(coil[name]['z']+15)+'\n')
    f.write(name+'_Fr = '+'{:1.3e}'.format(coil[name]['Fr'])+'\n')
    f.write(name+'_Fz = '+'{:1.3e}'.format(coil[name]['Fz'])+'\n')
    f.write(name+'_I = '+'{:1.3e}'.format(1e-6*coil[name]['I'])+'\n')
    f.write('\n')
f.close()
        
        
        
        
        
        

