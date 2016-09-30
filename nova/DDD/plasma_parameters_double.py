import numpy as np
import pylab as pl
import pickle
from radial_build import RB
from feild_calc import solCalc

TFopp = False # optimise TF coil placment
config = 'DD3'  # DD1/DD3
gap = 0.175
tfw = 0.1
BB = [0.56,0.8]  # breeding blanket thickness (in/out)
tBBsupport = 0.1  # blanket support
VV = [0.34,.75]  # vacumn vessel thickness (in/out)
sheild = [0.85,0.1]  # neutron sheilding

Icoil = 17.5e6  # coil amp-turns
Jmax = 15e6  # max current density
Acoil = Icoil/Jmax
dRcoil = np.sqrt(Acoil)
dRsteel = 0.15*dRcoil
flip = 1
graze = 1.5*np.pi/180

with open('./plot_data/'+config+'_sol.pkl', 'rb') as input:
    sf = pickle.load(input)
    plot = pickle.load(input)
    geom = pickle.load(input)

sol = solCalc(sf,geom,config,flip=flip)
rb = RB(geom,sf,config,Np=400,flip=flip)
#rb.internal_coils(['P6','P5','PS2','P6B','P5B','PS2','PS5'])

pl.figure(figsize=(18,12))
pl.axis('equal')
pl.xlim([-15.5,15.5]),pl.ylim([-15,9])
#pl.xlim([6,9]),pl.ylim([-5,-3])
#pl.xlim([6,14]),pl.ylim([-10,-6])
#pl.xlim([2,17.5]),pl.ylim([-11,5])
                              
rb.fill(dR=gap,dt=tfw,alpha=0.7,color='r',label='first wall')
rb.fill(dt=BB,alpha=0.7,color='b',label='blanket')
rb.fill(dt=tBBsupport,alpha=0.4,color='g',label='blanket support')

P6support = {}
P6support['top'] ={}
P6support['bottom'] = {}
P5support = {}
P5support['top'] ={}
P5support['bottom'] = {}

if config is 'DD3': 
    rb.VVfill(connect=[0.36,0.802],r=[13.56,13.5,10,6.8,5.2],
              z=[-1.5,-5,-10,-9.5,-3.5],w=[0.9,2,2,3.5,2],dt=VV,
              ref_o=3/8*np.pi,alpha=0.3,color='b',label='vacuum vessel')
    rb.TFopp(TFopp)        
              
    rb.rzGet()  # VV ports
    rb.fill(trim=[0.795,0.795+0.04],dt=VV,ref_o=3/8*np.pi,alpha=1,color='r')
    rb.rzGet()
    rb.fill(trim=[0.745,0.745+0.04],dt=VV,ref_o=3/8*np.pi,alpha=1,color='r',
            label='ports')
    
    rs,zs = rb.support_fill(coils=['P6','P6B','P5B'],r=[12,13.2],z=[-8,-8.7],
                    dR=0.05,alpha=1,color='g',label='coil supports',
                    referance='length',ref_o=0.5,dref=0.5,dt=[0.3,0.6])
                    
    P6support['top']['r'] = rs
    P6support['bottom']['r'] = rb.R
    P6support['top']['z'] = zs
    P6support['bottom']['z'] = rb.Z
    
    rsI,zsI = rb.offset(rs,zs,0.05)
    RI,ZI = rb.offset(rb.R,rb.Z,-0.05)
    
    P6support['top']['rI'] = rsI
    P6support['bottom']['rI'] = RI
    P6support['top']['zI'] = zsI
    P6support['bottom']['zI'] = ZI
               
    rb.sheild_fill(dR=0.05,dt=sheild,referance='length',ref_o=0.5,dref=0.3,
                    alpha=0.3,color='k',label='neutron shields')
    
    rs,zs = rb.support_fill(coils=['P5','PS2','PS5'],r=[13.5,14.8],z=[-5.5,-5.5],
                    dR=0.1,alpha=1,color='g',
                    referance='length',ref_o=0.6,dref=0.4,dt=[0.3,0.6])
                    
    P5support['top']['r'] = rs
    P5support['bottom']['r'] = rb.R
    P5support['top']['z'] = zs
    P5support['bottom']['z'] = rb.Z
    
    rsI,zsI = rb.offset(rs,zs,0.05)
    RI,ZI = rb.offset(rb.R,rb.Z,-0.05)
    
    P5support['top']['rI'] = rsI
    P5support['bottom']['rI'] = RI
    P5support['top']['zI'] = zsI
    P5support['bottom']['zI'] = ZI
    
    rb.sheild_fill(dR=0.1,dt=sheild,referance='length',ref_o=0.5,dref=0.3,
                    alpha=0.3,color='k')  
                    
    rb.TFfill(dRsteel,dRcoil)
    rb.TFsupport(dt=4*dRsteel+dRcoil,trim=[0.353,0.353+0.02],color='k',alpha=1)
    rb.TFsupport(dt=4*dRsteel+dRcoil,trim=[0.428,0.428+0.02],color='k',alpha=1)
       

    rb.plot.coil_fill()
    rb.plot.coil_sheild()
    #rb.plot.coil_label() 
    
    with open('./plot_data/'+config+'_Supports.pkl', 'wb') as output:
        pickle.dump(P6support,output,-1)
        pickle.dump(P5support,output,-1)

else:
    rb.VVfill(connect=[0.1,0.9],r=[9.5,8,6.3,5.8],plot=False,
              z=[-5.1,-5.9,-5.3,-3.5],w=[1,1,1,2],dt=VV,
              ref_o=3/8*np.pi,alpha=0.3,color='b',label='vacuum vessel')
    rb.TFopp(TFopp)
    rb.TFfill(dRsteel,dRcoil)


sol.strike_arc(config,plot=False)   
for leg in ['inner','outer']:
    Rsol,Zsol = sol.legs(leg,5,plot=False)
    delta_sol = np.arctan2(-np.diff(Zsol[-2:]), -np.diff(Rsol[-2:]))
    Xi = sol.expansion([Rsol[-1]],[Zsol[-1]])
    #L2D,L3D = sol.connection(leg,1)
    theta = sol.strike_point(Xi,graze)
    
    print(leg, 'theta:',theta*180/np.pi, 'Xi',Xi, 'R',
          Rsol[-1],'1/R2',Rsol[-1]**-2,'Bphi',6*10.5/Rsol[-1])
    print('L3D', L3D[-1])
    dPlate = 1.2
    switch,closed,side = -1,-1,-1
    label = None
    
    if leg is 'outer':
        label = 'targets'
        switch *= -1
    if config is 'DD3':
        closed *= -1
        side *= -1

    tilt = delta_sol+closed*switch*theta
    Lplate = np.linspace(side*switch*dPlate/2,-side*switch*dPlate/2,15)
    
    Rplate = Rsol[-1]+Lplate*np.cos(tilt)
    Zplate = Zsol[-1]+Lplate*np.sin(tilt)
    
    rb.R,rb.Z = Rplate,Zplate
    rb.fill(dt=0.1,alpha=0.7,color='c',trim=None,label=label)
    #pl.plot(Rplate,Zplate)
    
    for layer_index in np.arange(1,11):
        sol.legs(leg,layer_index,plot=True)
        

pl.legend(loc=3,ncol=3)
pl.xlabel('Radial coordinate, R [m]')
pl.ylabel('Vertical coordinate, Z [m]')

TFopp = False # optimise TF coil placment
config = 'DD1'  # DD1/DD3
gap = 0.175
tfw = 0.1
BB = [0.56,0.8]  # breeding blanket thickness (in/out)
tBBsupport = 0.1  # blanket support
VV = [0.34,.75]  # vacumn vessel thickness (in/out)
sheild = [0.85,0.1]  # neutron sheilding

Icoil = 17.5e6  # coil amp-turns
Jmax = 15e6  # max current density
Acoil = Icoil/Jmax
dRcoil = np.sqrt(Acoil)
dRsteel = 0.15*dRcoil
flip = -1
graze = 1.5*np.pi/180

with open('./plot_data/'+config+'_sol.pkl', 'rb') as input:
    sf = pickle.load(input)
    plot = pickle.load(input)
    geom = pickle.load(input)

sol = solCalc(sf,geom,flip=flip)
rb = RB(geom,sf,config,Np=400,flip=flip)
rb.internal_coils(['P6','P5','PS2','P6B','P5B','PS2','PS5'])

#pl.figure(figsize=(18,12))
#pl.axis('equal')
#pl.xlim([-15.5,15.5]),pl.ylim([-15,9])
#pl.xlim([6,9]),pl.ylim([-5,-3])
#pl.xlim([6,14]),pl.ylim([-10,-6])
#pl.xlim([2,17.5]),pl.ylim([-11,5])
                              
rb.fill(dR=gap,dt=tfw,alpha=0.7,color='r',label='first wall')
rb.fill(dt=BB,alpha=0.7,color='b',label='blanket')
rb.fill(dt=tBBsupport,alpha=0.4,color='g',label='blanket support')

P6support = {}
P6support['top'] ={}
P6support['bottom'] = {}
P5support = {}
P5support['top'] ={}
P5support['bottom'] = {}

if config is 'DD3': 
    rb.VVfill(connect=[0.36,0.802],r=[13.56,13.5,10,6.8,5.2],
              z=[-1.5,-5,-10,-9.5,-3.5],w=[0.9,2,2,3.5,2],dt=VV,
              ref_o=3/8*np.pi,alpha=0.3,color='b',label='vacuum vessel')
    rb.TFopp(TFopp)        
              
    rb.rzGet()  # VV ports
    rb.fill(trim=[0.795,0.795+0.04],dt=VV,ref_o=3/8*np.pi,alpha=1,color='r')
    rb.rzGet()
    rb.fill(trim=[0.745,0.745+0.04],dt=VV,ref_o=3/8*np.pi,alpha=1,color='r',
            label='ports')
    
    rs,zs = rb.support_fill(coils=['P6','P6B','P5B'],r=[12,13.2],z=[-8,-8.7],
                    dR=0.05,alpha=1,color='g',label='coil supports',
                    referance='length',ref_o=0.5,dref=0.5,dt=[0.3,0.6])
                    
    P6support['top']['r'] = rs
    P6support['bottom']['r'] = rb.R
    P6support['top']['z'] = zs
    P6support['bottom']['z'] = rb.Z
    
    rsI,zsI = rb.offset(rs,zs,0.05)
    RI,ZI = rb.offset(rb.R,rb.Z,-0.05)
    
    P6support['top']['rI'] = rsI
    P6support['bottom']['rI'] = RI
    P6support['top']['zI'] = zsI
    P6support['bottom']['zI'] = ZI
               
    rb.sheild_fill(dR=0.05,dt=sheild,referance='length',ref_o=0.5,dref=0.3,
                    alpha=0.3,color='k',label='neutron shields')
    
    rs,zs = rb.support_fill(coils=['P5','PS2','PS5'],r=[13.5,14.8],z=[-5.5,-5.5],
                    dR=0.1,alpha=1,color='g',
                    referance='length',ref_o=0.6,dref=0.4,dt=[0.3,0.6])
                    
    P5support['top']['r'] = rs
    P5support['bottom']['r'] = rb.R
    P5support['top']['z'] = zs
    P5support['bottom']['z'] = rb.Z
    
    rsI,zsI = rb.offset(rs,zs,0.05)
    RI,ZI = rb.offset(rb.R,rb.Z,-0.05)
    
    P5support['top']['rI'] = rsI
    P5support['bottom']['rI'] = RI
    P5support['top']['zI'] = zsI
    P5support['bottom']['zI'] = ZI
    
    rb.sheild_fill(dR=0.1,dt=sheild,referance='length',ref_o=0.5,dref=0.3,
                    alpha=0.3,color='k')  
                    
    rb.TFfill(dRsteel,dRcoil)
    rb.TFsupport(dt=4*dRsteel+dRcoil,trim=[0.353,0.353+0.02],color='k',alpha=1)
    rb.TFsupport(dt=4*dRsteel+dRcoil,trim=[0.428,0.428+0.02],color='k',alpha=1)
       

    rb.plot.coil_fill()
    rb.plot.coil_sheild()
    #rb.plot.coil_label() 
    
    with open('./plot_data/'+config+'_Supports.pkl', 'wb') as output:
        pickle.dump(P6support,output,-1)
        pickle.dump(P5support,output,-1)

else:
    rb.VVfill(connect=[0.1,0.9],r=[9.5,8,6.3,5.8],plot=False,
              z=[-5.1,-5.9,-5.3,-3.5],w=[1,1,1,2],dt=VV,
              ref_o=3/8*np.pi,alpha=0.3,color='b',label='vacuum vessel')
    rb.TFopp(TFopp)
    rb.TFfill(dRsteel,dRcoil)


sol.strike_arc(config,plot=False)   
for leg in ['inner','outer']:
    Rsol,Zsol = sol.legs(leg,5,plot=False)
    delta_sol = np.arctan2(-np.diff(Zsol[-2:]), -np.diff(Rsol[-2:]))
    Xi = sol.expansion([Rsol[-1]],[Zsol[-1]])
    #L2D,L3D = sol.connection(leg,1)
    theta = sol.strike_point(Xi,graze)
    
    print(leg, 'theta:',theta*180/np.pi, 'Xi',Xi, 'R',
          Rsol[-1],'1/R2',Rsol[-1]**-2,'Bphi',6*10.5/Rsol[-1])
    #print('L3D', L3D[-1])
    dPlate = 1.2
    switch,closed,side = -1,-1,-1
    label = None
    
    if leg is 'outer':
        label = 'targets'
        switch *= -1
    if config is 'DD3':
        closed *= -1
        side *= -1

    tilt = delta_sol+closed*switch*theta
    Lplate = np.linspace(side*switch*dPlate/2,-side*switch*dPlate/2,15)
    
    Rplate = Rsol[-1]+Lplate*np.cos(tilt)
    Zplate = Zsol[-1]+Lplate*np.sin(tilt)
    
    rb.R,rb.Z = Rplate,Zplate
    rb.fill(dt=0.1,alpha=0.7,color='c',trim=None,label=label)
    #pl.plot(Rplate,Zplate)
    
    for layer_index in np.arange(1,11):
        sol.legs(leg,layer_index,plot=True)
        
line, = pl.plot([0,0],[-10,8],'r-.', linewidth=3)
line.set_dashes(1.5*np.array([16, 8, 4, 8])) 
pl.savefig('../Figs/FEC_vessel_double.png', dpi=800)


