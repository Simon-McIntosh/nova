import pylab as pl
import numpy as np
from itertools import cycle
from config import Setup

from radial_build import RB
from streamfunction import SF
from elliptic import EQ
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from addtext import linelabel
import time
from inverse import INV


#from elliptic import grid
#import cross_coil as cc
import seaborn as sns
rc = {'figure.figsize':[10,10*12/16],'savefig.dpi':100, # 
      'savefig.jpeg_quality':100,'savefig.pad_inches':0.1,
      'lines.linewidth':2}
sns.set(context='talk',style='white',font='sans-serif',palette='Set2',
        font_scale=7/8,rc=rc)
color = sns.color_palette('Set2')
Color = cycle(color)

config = 'SX8'  # SN,X,SX,SX8,SFm,SX2

pl.figure(figsize=(10*14/16,10))
pl.axis('equal')
pl.axis('off')
#pl.ylim([-12.5,11])
#pl.xlim([7,8])

setup = Setup('SN')
sf = SF(setup.filename,sample=4,Nova=False)
rb = RB(setup,sf,Np=250)


#conf = Config(config)
#sf = SF(conf,sample=4,Nova=False)
sf.contour()
#eq = EQ(sf,sigma=0,limit=[5.5,12,-8,5],n=5e4)  # resample
#eq.plotj()

#
#conf.TF(sf)
#rb = RB(conf,sf,Np=250)

rb.firstwall(calc=True,plot=True,debug=False)
rb.trim_sol(plot=True)

'''

leg = 'outer'
if config == 'SN':
    filename = '2015_First_Wall_and_Divertor_CCFE_2MGVE7_v1_0.txt'  # SN
elif config[:2] == 'SF':
    filename = '2015_First_Wall_and_Divertor_CCFE_2MDBJ9_v1_0.txt'  # SF
    leg += '1'
elif config[:2] == 'SX':
    filename = '2015_First_Wall_and_Divertor_CCFE_2MKUGH_v1_0.txt'
    filename = 'SX7_bdry_19_10.txt'
    #filename = 'SX8_bdry_18_11.txt'
    #filename = 'SX8_bdry_final_email.txt'

filename = config+'_old_bdry.txt'
#filename = '2015_First_Wall_and_Divertor_CCFE_2MKUGH_v1_0.txt'
with open('../Data/'+filename) as f:
    for i in range(2):    
        f.readline()
    if len(f.readline().split()) > 0:
        f.readline()    
    f.readline()
    if f.readline().split()[0] == 'inner1':
        nskip = 3
    else:
        nskip = 1
    for i in range(nskip):    
        f.readline()

    for i in range(3):    
        f.readline()
    n = int(f.readline())
    r,z = np.zeros(n),np.zeros(n)
    for i,line in enumerate(f):
        data = line.split()
        if len(data) == 2:
            r[i],z[i] = data[0],data[1]
        else:
            r[i],z[i] = data[0],data[2]
#pl.plot(r,z,'-')

l = rb.length(r,z)
index = np.append(np.diff(l)!=0,True)
r,z = r[index],z[index]

rb.Rb,rb.Zb = r,z



rb.trim_sol(plot=True)


Rt,Zt = rb.Rb,rb.Zb  # rb.targets[leg]['R'], rb.targets[leg]['Z']
n = rb.normal(Rt,Zt)
T = np.dot(np.matrix([[0,-1],[1,0]]),n)


graze = np.ones(sf.Nsol)
for i in range(sf.Nsol):
    ro,zo = sf.legs[leg]['R'][i][-1],sf.legs[leg]['Z'][i][-1]
    index = np.argmin((Rt-ro)**2+(Zt-zo)**2)

    graze[i] = 180/np.pi*sf.get_graze([ro,zo],T[:,index])
    print(i,index,graze[i])
    #pl.plot(ro,zo,'o')

pl.figure()
pl.plot(sf.Dsol,graze)



#sf.set_plasma({'r':eq.r,'z':eq.z,'psi':eq.psi},contour=True)
#sf.write_flux()
#sf.write_coils()

#inv = INV(sf,eq,configTF='SN',config='SN')
#inv.plot_coils()

'''
inv.fix_boundary_psi(N=15,alpha=0.99,factor=1)  # add boundary points
inv.set_plasma()
inv.solve()
'''

#inv.set_force_feild(multi_filament=True)

#inv.get_force() 
#inv.plot_force()

#sf.sol(plot=True)

#eq.plotb()
#

'''
sf.plot_coils(next(Color),coils=sf.coil,label=True,plasma=False,current=True) 

#sf.plot_coils(next(Color),coils=eq.coil,label=False,plasma=False) 

rb.FWfill(dt=conf.tfw,loop=True,alpha=0.7,color=next(Color))
rb.fill(dt=conf.BB[::-1],alpha=0.7,ref_o=0.3,dref=0.2,
        referance='length',color=next(Color))
rb.fill(dt=conf.tBBsupport,alpha=0.7,color=next(Color))
rb.BBsheild_fill(dt=conf.sheild,ref_o=0.35*np.pi,dref=0.2*np.pi,
                 offset=1/10*np.pi,alpha=0.7,color=next(Color))
rb.VVfill(dt=conf.VV,ref_o=0.25*np.pi,dref=0.25*np.pi,offset=0.5/10*np.pi,
          alpha=0.7,loop=True,color=next(Color))  # ref_o=0.385

conf.TFopp = 'L'
rb.set_TFbound()  # TF boundary conditions
rb.TFbound['ro_min'] -= 0.5
#rb.plot_TFbounds()          
rb.TFopp(False,objF=conf.TFopp)  # L==length, V==volume
rb.TFfill()
'''

'''
Rfw,Zfw = [],[]
with open('../Data/X_bdry.txt') as f:
    for i in range(10):
        f.readline()
    for line in f:
        data = line.split()
        Rfw.append(float(data[0]))
        Zfw.append(float(data[2]))
Rfw.append(Rfw[0])
Zfw.append(Zfw[0])
pl.plot(Rfw,Zfw)

Rp,Zp = rb.theta_sort(rb.Rp,rb.Zp,origin='top')
L = rb.length(Rp,Zp)
index = (L>0.02) & (L<0.98)
Rp,Zp = Rp[index],Zp[index]

dfw,theta = np.zeros(len(Rp)),np.zeros(len(Rp))
Rn,Zn = rb.normal(Rp,Zp)
for i,(r,z,rn,zn) in enumerate(zip(Rp,Zp,Rn,Zn)):
    ifw = np.argmin(np.sqrt((r-Rfw)**2+(z-Zfw)**2))
    dr,dz = Rfw[ifw]-r,Zfw[ifw]-z
    dfw[i] = np.dot(np.array([dr,dz]),np.array([rn,zn]))
    theta[i] = np.arctan2(sf.Mpoint[0]-r,z-sf.Mpoint[1])

theta = np.unwrap(theta)
pl.plot(Rp,Zp)
pl.plot(Rfw,Zfw)

pl.figure()
pl.plot(theta,dfw*1e3)
pl.plot(theta,250*np.ones(len(theta)),'k--',alpha=0.25)
pl.xlabel('poloidal angle')
pl.ylabel(r'first wall gap mm')
sns.despine()
'''


'''

#eq.run()
#eq.update_coil_psi()
#rb.trim_sol(plot=True)
#eq.gen(Nmax=50)
#sf.sol(plot=True)
#sf.eqwrite()


figname = conf.dataname+'_shoot'
pl.tight_layout(rect=[0,-0.1,0.95,1.1])
pl.savefig('../Figs/'+figname+'.png',dpi=300)


#eq.get_plasma_coil()  
#eq.update_coil_psi()  # set psi


eq.get_Xpsi()  # high-res Xpsi
eq.get_Mpsi()  # high-res Mpsi

print(sf.Xpsi,sf.Mpsi)

sf.sol(Nsol=15,update=True)  # store sol psi






#rb.trim_sol(Nsol=7,plot=True)


pl.figure(figsize=(5,3))

rb.trim_sol(Nsol=7,plot=False)

for leg in rb.targets.keys():
    if 'core' not in leg:
        for i in range(sf.Nsol):
            L2D,L3D = rb.fc.connection(leg,i)
            pl.plot(L2D,L3D)
        
#sf = SF(conf,sample=1)
#sf.plot_coils(Color,label=False,current=False)


Color = cycle(sns.color_palette('Set2'))
for leg in conf.targets.keys():
    R,Z = rb.sol.legs(leg)
    pl.plot(R,Z,color=next(Color))
'''

'''
Cmove = []
if config is 'SF2':  
    Cmove = {'external':[10,11,12,13,14,15],'internal':[]}
if config is 'SN2':  
    Cmove = {'external':[5,6,7,8,9,10],'internal':[]}
if config is 'X2':  
    Cmove = {'external':[5,6,7,8,9,10],'internal':[]}
if config is 'X':  
    Cmove = {'external':range(10,16),'internal':[]}
if config is 'SXm':  
    Cmove = {'external':[5,6,7,8,9,10],'internal':[]}
if config is 'SX22':  
    Cmove = {'external':[10,11,12,13,9,8],'internal':[]}
if config is 'SX3':  
    Cmove = {'external':[10,11,12,13,14,15],'internal':[]}
if config == 'SX4' or config == 'SX5' or config == 'SX6':  
    Cmove = {'external':[10,11,12,13,14,15],'internal':[]}
if Cmove:    
    coils = rb.fit_coils(Cmove,dLo=[0.1,0.5])
    next(Color)
    sf.plot_coils(Color,coils=coils,label=False)    
'''

'''
C = next(Color)
for leg in conf.targets.keys():
    target = rb.targets[leg]
    for L2D in np.arange(1,target['L2D'][-1],1):
        i = np.argmin(abs(target['L2D']-L2D))
        pl.plot(target['Rsol'][i],target['Zsol'][i],'ko',
                markersize=4,color=C,alpha=0.8)
        pl.text(target['Rsol'][i],target['Zsol'][i],
                ' {:1.0f}'.format(L2D),ha='left',fontsize=10,alpha=1,color=C)
pl.axis('off')


pl.tight_layout()
'''

'''
col_labels=['volume','length']
row_labels=['TF','LCFS','ratio']
table_vals=[[r'{:1.0f}m$^3$'.format(rb.TFvol),r'{:1.1f}m'.format(rb.TFlength)],
            [r'{:1.0f}m$^3$'.format(rb.Pvol),r'{:1.1f}m'.format(rb.Plength)],
            ['{:1.2f}'.format(rb.Rvol),r'{:1.2f}'.format(rb.Rlength)]]        
cell_colours = np.chararray(np.shape(table_vals))
for i in range(np.shape(table_vals)[0]):
    for j in range(np.shape(table_vals)[1]):
        cell_colours[i,j] = 'w'
pl.table(cellText=table_vals,colWidths=[0.2]*3,rowLabels=row_labels,
         colLabels=col_labels,loc='bottom right',
         alpha=1,bbox=[0.6, 0.05, 0.4, 0.15],fontsize=24,
         cellColours=cell_colours.decode())
''' 
      
'''
a = pl.axes([0.18, 0.1, .5, .14])
Color = cycle(sns.color_palette('Set2'))
for leg,ha,va in zip(conf.targets.keys(),['right','left'],['bottom','bottom']):
    color = next(Color)
    target = rb.targets[leg]

    if 'L3D' in target.keys():
        pl.plot(target['L2D'],target['L3D'],color=color)
        pl.text(target['L2D'][-1],target['L3D'][-1],
                ' '+leg+' {:1.1f}m'.format(target['L3D'][-1]),
                ha=ha,va=va,color=color)
        pl.plot(target['L2Dedge'],target['L3Dedge'],'--',color=color)

        pl.text(target['L2Dedge'][-1],target['L3Dedge'][-1],
                ' {:1.1f}m'.format(target['L3Dedge'][-1]),
                ha=ha,va=va,color=color)
                
        for L2D in np.arange(1,target['L2D'][-1],1):
            i = np.argmin(abs(target['L2D']-L2D))
            pl.plot(target['L2D'][i],target['L3D'][i],'ko',markersize=3)
            pl.text(target['L2D'][i],target['L3D'][i],
                    ' {:1.0f}'.format(L2D),ha='left',fontsize=10,alpha=0.5)

pl.xlabel(r'$L2D$ [m]')
pl.ylabel(r'$L3D$ [m]')
sns.despine()
'''

'''
#psi,X=sf.get_Xpsi(xo=[rb.targets['outer']['Ro'],rb.targets['outer']['Zo']])
#pl.plot(X[0],X[1],'*')

eq.grid(boundary={'R':rb.Rb,'Z':rb.Zb,'expand':0.25,
                  'zmax':sf.Xpoint[1]+sf.rcirc+sf.drcirc},n=5e4)
eq.set_sf_psi()  # set psi
sf.set_plasma({'r':eq.r,'z':eq.z,'psi':eq.psi},contour=True)
sf.sol(Nsol=21,plot=True)
'''

'''
figname = conf.dataname+'_radial_build'
pl.tight_layout(rect=[0,-0.1,0.95,1.1])
pl.savefig('../Figs/'+figname+'.pdf',dpi=300)
'''



#rb.trim_sol(Nsol=35,update=True,plot=True)


'''
pl.figure(figsize=(5,3))
color = sns.color_palette('Set2',sf.nleg)
for c,leg in enumerate(rb.targets.keys()):
    if 'core' not in leg:
        for i in range(sf.Nsol):
            L2D,L3D = sf.connection(leg,i)[:2]
            pl.plot(L2D,L3D,color=color[c+1])
        L2D,L3D_in = sf.connection(leg,0)[:2]
        L2D_out,L3D_out = sf.connection(leg,-1)[:2]
        L3D_out = IUS(L2D_out,L3D_out)(L2D)
        pl.fill_between(L2D,L3D_in,y2=L3D_out)
        pl.text(L2D[-1],L3D_in[-1],leg)
'''

'''
pl.figure(figsize=(8,4))    
color = sns.color_palette('Set2',sf.nleg)
text = linelabel(Ndiv=15,value='',postfix='')    
for c,leg in enumerate(rb.targets.keys()):
    if 'core' not in leg:
        L3D = np.zeros(sf.Nsol)
        for i in range(sf.Nsol):
            L3D[i] = sf.connection(leg,i)[1][-1]
        pl.plot(sf.Dsol[1:]*1e3,L3D[1:],color=color[c])
        text.add(leg)    
text.plot()
sns.despine()
pl.xlabel(r'LFS offset mm')
pl.ylabel(r'connection length m')
figname = conf.dataname+'_connection'
pl.tight_layout(rect=[0.05,0.0,0.95,0.95])
pl.savefig('../Figs/'+figname+'.png',dpi=300)
'''


'''
sf.coil['plasma'] = {'I':20.254e6,'r':sf.rmagx,'z':sf.zmagx,'dr':3,'dz':3}

                
#del sf.coil['plasmsa']
#sf.coil['Coil10']['I']*=2


psi = grid([4,12],[-8,8],1500,sf)

psi.rbdry,psi.zbdry = np.copy(sf.rbdry),np.copy(sf.zbdry)
psi.Xpsi = sf.get_Xpsi()[0]


psi.coils(delta=0.5)
FR,FZ = np.array([]),np.array([])
for coil in psi.coil.keys():
    ro,zo = psi.coil[coil]['r'],psi.coil[coil]['z']
    Fr,Fz = cc.Fcoil(psi.coil, coil)
    FR,FZ = np.append(FR,Fr),np.append(FZ,Fz)
Fmax = np.max(np.sqrt(FR**2+FR**2))
color = next(Color)
for Fr,Fz,coil in zip(FR,FZ,list(psi.coil.keys())):  
    pl.arrow(psi.coil[coil]['r'][0], psi.coil[coil]['z'][0], 
    Fr/Fmax, Fz/Fmax, fc=color, ec=color,
    head_width=0.1, head_length=0.15)
sf.plot_coils(Color,coils=psi.coil,label=False)
     
'''  
'''          
pl.figure()
connect = []
for i in range(rb.sf.Nsol):
    L2D,L3D = rb.sol.connection('outer',i)
    connect.append(L3D[-1])
pl.plot(rb.sf.Dsol,connect)
'''   

'''
pl.figure(figsize=(10,10*8/16))
text = linelabel(Ndiv=15,value='1.2f',postfix='')   
for j,leg in enumerate(['inner','outer']):
    R,Z = sf.legs[leg]['R'][0],sf.legs[leg]['Z'][0]
    graze = np.zeros(len(R))
    for i,(r,z) in enumerate(zip(R,Z)):
        graze[i] = sf.get_max_graze(r,z)
    L = sf.length(R,Z,norm=False)
    index = L<conf.targets[leg]['L2D'][0]
    pl.plot(L[index],graze[index]*180/np.pi,color=color[j])
    text.add(leg) 
pl.plot(L[index],1.5*np.ones(len(L[index])),'k--',alpha=0.25)
pl.xlabel(r'$L2D$ outer m')
pl.ylabel(r'max graze deg')
text.plot()
sns.despine()
'''
