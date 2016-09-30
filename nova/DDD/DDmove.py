import pylab as pl
import numpy as np
from config import layout
from extract import extract_geom
from streamfunction import SF
from cross_coil import Pcoil, Pcalc
import matplotlib
font = {'family': 'serif', 'serif': 'Times', 'weight': 'normal', 'size': 14}
matplotlib.rc('font', **font)
fig_path = '../Figs/'

config = 'DD3'  # 'DD1' 
geom = extract_geom(config)
I = 16e6  # plasma current [MA]
delta = 1.0  # node spaceing [m] (0.4)

#geom.plasma_coils(I, delta)

plasma_name = 'plasma-1'
geom.coil[plasma_name] = {}
geom.coil[plasma_name]['I'] = I
geom.coil[plasma_name]['r'] = geom.plasma['rc']
geom.coil[plasma_name]['z'] = geom.plasma['zc']
geom.coil[plasma_name]['rc'] = delta 
          
sf = SF(geom.coil, res=0.1, rlim=[5,14], zlim=[-12,6])  # res=0.2
    
fig = pl.figure(figsize=(12,20))
sf.sol(color='b')
Xpsi,Xpoint = sf.get_Xpsi([8,-4])  # Xpoint

'''
lower_coils = ['CS3L','CS3LB','PS4','PS3','P6C','PS1']
for lowc in lower_coils:  # lift lower coils
    geom.coil[lowc]['z'] = geom.coil[lowc]['z'] + 0.5
'''    
#geom.coil['P6']['r'] = geom.coil['P6']['r'] + 0.75
#geom.coil['P6']['z'] = geom.coil['P6']['z'] + 0.75

#geom.coil['P5']['r'] = 6
#geom.coil['P5']['z'] = -4

#geom.coil['P5']['r'] = geom.coil['P5']['r'] + 2
#geom.coil['P5']['z'] = geom.coil['P5']['z'] + 0.2

dX,theta,dXcoil = 0.4,0.5,2.5  # theta=0.5
geom.coil['P5']['r'] = Xpoint[0]+dXcoil*np.cos(np.pi/4-theta)
geom.coil['P5']['z'] = Xpoint[1]-dXcoil*np.sin(np.pi/4-theta)
geom.coil['P6']['r'] = Xpoint[0]-dXcoil*np.sin(np.pi/4-theta)
geom.coil['P6']['z'] = Xpoint[1]-dXcoil*np.cos(np.pi/4-theta)

C = np.zeros((1,2))  # Coallation points
C[0,:] = Xpoint
#C = np.append(C, [[6.5,0]],0)
#C = np.append(C, [[11.2,0]],0)
#C = np.append(C, [[8.3,3.5]],0)

for i in range(4):
    C = np.append(C, [[Xpoint[0]+dX*np.cos(theta+i*np.pi/2),
                       Xpoint[1]+dX*np.sin(theta+i*np.pi/2)]],0)
C = np.append(C, [[5.75,-6.3]],0)

C = np.append(C, [[11,-6.0]],0)
C = np.append(C, [[8.5,-8.5]],0)

#C = np.append(C, [[12,-6]],0)
#C = np.append(C, [[(geom.coil['P6']['r']+geom.coil['P5']['r'])/2,
#                   (geom.coil['P6']['z']+geom.coil['P5']['z'])/2]],0)
#C = np.append(C, [[geom.coil['P6']['r']-1.3,
#                   geom.coil['P6']['z']]],0)
#C = np.append(C, [[geom.coil['P5']['r'],geom.coil['P5']['z']-1.52]],0)

#C = np.append(C, [[10,-7.5]],0)

nC = np.shape(C)[0] 

geom.coil['PS2']['z'] = geom.coil['P5']['z']
geom.coil['PS5']['z'] = geom.coil['P5']['z']

geom.coil['P6B']['z'] = geom.coil['P6']['z']-0.5
geom.coil['P5B']['z'] = geom.coil['P6']['z']-1

geom.coil['PS1']['z'] = -7.5  # geom.coil['PS1']['z']+2
geom.coil['PS1']['r'] = 13  # geom.coil['PS1']['r']+2

zero_current = ['P5','P5B','P6B','PS2','PS5','CS3LB','PS4','PS3','P6C']  # 'P6B','P5B','PS5','PS2'
adj_coils = ['P6','P5','CS3L','PS1','CS2L']  # ,'P6B','P5B' ,'PS5'
#adj_coils = np.append(adj_coils,['PS3','P6C','PS1','PS4','CS3L','CS3LB'])  # 
#adj_coils = np.append(adj_coils,['CS3L','CS3LB','PS4','PS3','P6C','PS1'])
#adj_coils = np.append(adj_coils,['CS3L','CS2L','CS1','CS2U','CS3U',
#                                 'P1','P2','P3','P4'])


C[0,:] = Xpoint
    
for loop in np.append(zero_current,adj_coils):
    geom.coil[loop]['I'] = 0
    
G = np.zeros((np.shape(C)[0],np.shape(adj_coils)[0]))  # [G][I] = [Tpsi]
    
BGpsi = np.zeros((nC,1))
for i in range(nC):
    BGpsi[i] = Pcoil(geom.coil,C[i,:])  # background Xpoint potential
Tpsi = Xpsi*np.ones((nC,1))-BGpsi  # target potential

#Tpsi[-1] = 1.1*Xpsi  # high potential

    
for i,c in enumerate(C):  # coallation points
    for j,name in enumerate(adj_coils):
        G[i,j] = Pcalc(c[0],c[1],geom.coil[name]['r'],geom.coil[name]['z'],1)
    
I = np.linalg.lstsq(G,Tpsi)[0]  # solve
for i,name in enumerate(adj_coils):
    geom.coil[name]['I'] = I[i,0]
        
#geom.coil['PS1']['I'] = 1.2*geom.coil['PS1']['I']
    
sf.potential()
sf.sol()
pl.contour(sf.rm,sf.zm,sf.psi,levels=[Xpsi],color='k')
sf.contour() 
#print(Tpsi,np.dot(G,I))

for i in range(nC):
    pl.plot(C[i,0],C[i,1],'co')
    
plot = layout(geom)
plot.coil_fill()
plot.coil_label(['P6','P5','P6B','P5B','PS2','PS5'])
plot.coil_label(['CS3LB','PS4','PS3','P6C','PS1'])
plot.coil_label()
plot.coil_sheild()
plot.TF()  # TF coil

#plot.sol()  # scrape-off layer
#plot.plates()  # divertor plates
#
#plot.FW()
#plot.plasma()
#plot.coil_fill()
#plot.coil_label(['P6','P5','P6B','P5B','PS2','PS5'])
#plot.coil_sheild()

pl.grid()
pl.axis('equal')
pl.xlim([3,16])
pl.ylim([-11.5,8])
#pl.xlim([6,10])
#pl.ylim([-5,-3])

pl.xlabel('radial coordinate, r [m]')
pl.ylabel('vertical coordinate, z [m]')
pl.tight_layout()
pl.savefig(fig_path+'DD_wide.png', dpi=300)
