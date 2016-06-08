import numpy as np

def grid():# contor grid
    res = 0.2
    rlim, zlim = ([2,15], [-11,6])
    dr, dz = (rlim[1]-rlim[0], zlim[1]-zlim[0])
    nr, nz = (int(dr/res), int(dz/res))

    r = np.linspace(rlim[0], rlim[1], nr)
    z = np.linspace(zlim[0], zlim[1], nz)
    rm, zm = np.meshgrid(r, z)
    phi = np.zeros((nz,nr))
    for i in range(nz):
        for j in range(nr):
            phi[i,j] = Pcoil(mu_o, coil, [rm[i,j],zm[i,j]])
    
    level = [np.mean(phi)-1.5*np.std(phi), np.mean(phi)+1.5*np.std(phi)]
    pl.contour(rm, zm, phi, levels=np.linspace(level[0],level[1],30), colors='k')
    pl.axis('equal')
    pl.xlim(rlim)
    pl.ylim(zlim)
