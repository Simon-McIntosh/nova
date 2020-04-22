from nep.DINA.VDE_force import VDE_force
from nep.coil_geom import VSgeom
import nova.cross_coil as cc
from amigo.pyplot import plt
import numpy as np


folder = 3
vde = VDE_force(mode='control', discharge='DINA', Iscale=1)
vde.load_file(folder, frame_index=0)

vs3 = VSgeom()
psi = {}
for coil in vs3.geom:
    x, z = vs3.geom[coil]['x'], vs3.geom[coil]['z']
    psi[coil] = {'x': x, 'z': z, 'psi': np.zeros(vde.tor.nt)}

for frame_index in range(vde.tor.nt):
    vde.frame_update(frame_index)  # update coil currents and plasma position
    for coil in psi:
        x, z = psi[coil]['x'], psi[coil]['z']
        psi[coil]['psi'][frame_index] = cc.get_coil_psi(x, z, vde.pf)[0]

for coil in psi:
    plt.plot(vde.tor.t, psi[coil]['psi'])

