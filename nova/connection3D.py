import numpy as np
import pylab as pl
import pickle
from radial_build import RB
from feild_calc import solCalc
from matplotlib.collections import LineCollection
import matplotlib
font = {'family': 'serif', 'serif': 'Arial', 'weight': 'normal', 'size': 18}
matplotlib.rc('font', **font)

def cline(x,y,z,clim,cmap):  # color line
    norm = pl.Normalize(clim[0],clim[1])
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)    
    lc = LineCollection(segments,cmap=cmap,norm=norm)
    if not hasattr(z, "__iter__"):  # single value check
        z = np.array([z])
    lc.set_array(z)
    lc.set_linewidth(1)
    pl.gca().add_collection(lc)

config = 'DD1'  # DD1/DD3

graze = 1.5*np.pi/180  # toroidal grazing angle
dPlate = 1.2  # target plate lenght
dCap = 0.12

with open('./plot_data/'+config+'_sol.pkl', 'rb') as input:
    sf = pickle.load(input)
    plot = pickle.load(input)
    geom = pickle.load(input)

rb = RB(geom,sf,config,Np=400)
sol = solCalc(sf,geom,config)

pl.figure(figsize=(14,10))
pl.axis('equal')

sol.strike_arc(config,plot=True)   
for leg in ['inner','outer']:
    Rsol,Zsol = sol.legs(leg,2,plot=True)
    delta_sol = np.arctan2(-np.diff(Zsol[-2:]), -np.diff(Rsol[-2:]))
    Xi = sol.expansion([Rsol[-1]],[Zsol[-1]])
    L2D,L3D = sol.connection(leg,5)
    theta = sol.strike_point(Xi,graze)
    
    print(leg, 'theta:',theta*180/np.pi, 'Xi',Xi, 'R',
          Rsol[-1],'1/R2',Rsol[-1]**-2,'Bphi',6*10.5/Rsol[-1])
    print('L3D', L3D[-1])


'''
pl.figure(figsize=0.8*np.array([6.5,11]))

clim = [3,11]  # colormap limits
max2D = []
max3D = []

#pl.xlim
sol = solCalc(sf,geom,config)
sol.strike_arc(config,plot=False)

for leg,board in zip(['inner','outer'],['inboard','outboard']):
    for i in np.arange(3,11):
        L2D,L3D = sol.connection(leg,i)
        max2D.append(np.max(L2D))
        max3D.append(np.max(L3D))
        
        if i == 3:
            ax = pl.gca()
            bbox_props = dict(boxstyle="round", fc="white", ec="k", lw=2)
            ax.text(L2D[-1]-0.8,1.01*L3D[-1],board+' leg ',
                        horizontalalignment='right',verticalalignment='bottom',
                        bbox=bbox_props)
        
        if leg is 'inner':
            cm = pl.get_cmap('jet_r')
        elif leg is 'outer':
            cm = pl.get_cmap('jet_r')
        cline(L2D,L3D,i,clim,cm)
        
        
pl.xlim([0,np.max(max2D)])
pl.ylim([0,1.1*np.max(max3D)])       
pl.xlabel('Poloidal distance from Xpoint, L2D [m]')
pl.ylabel('Connection length from Xpoint, L3D [m]')    
    
pl.savefig('../Figs/FEC_L3D'+config+'.png', dpi=600,
           bbox_inches='tight',pad_inches=0.2)
    
'''
