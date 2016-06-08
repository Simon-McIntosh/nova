import numpy as np
from cross_coil import Bcoil, mu_o, loop, plasma_coils
from eqlib import eq

path = './coil_data/'
file = 'Double_Decker_Coils'
config = 'DD'
I = 16e6  # plasma current [A]
delta = 0.2  # node spaceing [m]
plasma = loop(path,file,'plasma')

for swing in range(1,5):
    ansys_file = 'DDswing_'+str(swing)+'.txt'
    
    eqlib = eq(config)
    eqlib.swing(swing)
    coil = eqlib.coil
    plasma_coils(plasma, I, delta, coil)
    
    f = open(path+ansys_file, 'w')  # open file
    for name in coil.keys(): 
        B = Bcoil(mu_o, coil, [coil[name]['r'],0,coil[name]['z']])
        coil[name]['Fv'] = 2*np.pi*coil[name]['r']*np.cross(B,[0,coil[name]['I'],0])
        
        if 'plasma' not in name:
            f.write(name+'_Fx = '+'{:1.3e}'.format(coil[name]['Fv'][0])+'\n')
            f.write(name+'_Fz = '+'{:1.3e}'.format(coil[name]['Fv'][2])+'\n')
            f.write('\n')
    f.close()