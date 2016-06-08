import numpy as np
import pylab as pl
import pickle
from scipy.interpolate import interp1d as interp1
import scipy.optimize as op
from itertools import count

class PF(object):
    
    def __init__(self):
        a = 1
        
    def set_coils(self,eq):
        self.coil = {}
        if eq['ncoil'] > 0: 
            nC = count(0)
            for i,(r,z,dr,dz,I) in enumerate(zip(eq['rc'],eq['zc'],eq['drc'],
                                                     eq['dzc'],eq['Ic'])):
                name = 'Coil{:1.0f}'.format(next(nC))
                self.coil[name] = {'r':r,'z':z,'dr':dr,'dz':dz,'I':I,
                                   'rc':np.sqrt(dr**2+dz**2)/2}
                '''
                if self.config is 'SN':
                    if i==10: break
                if self.config is 'SF2':
                    if i==15: break
                '''

class TF(object):
    
    def __init__(self):
        a = 1

    def drawTF(self,xCoil,Nspace=100):
        zCoil = xCoil[0]
        fone = xCoil[1]  # r0 == fone*zCoil
        ftwo = xCoil[2]  # r1 == ftwo*r0
        ro = xCoil[3]
        zo = xCoil[4]
        tf = {}  # TF coil dict
        tf['origin'] = [ro,zo]
        tf['z'] = zCoil
        #tf['r']= [fone*zCoil,ftwo*zCoil]
        tf['r']= [fone,ftwo]
        tf['alpha'] = np.array([40,75])*np.pi/180  #40,75
        Rtf = np.array([tf['origin'][0],tf['origin'][0]])  # straight section
        Ztf = np.array([tf['origin'][1],tf['origin'][1]+tf['z']])
        theta = np.linspace(0,tf['alpha'][0],
                            round(0.5*Nspace*tf['alpha'][0]/np.pi))  # arc 0
        Rtf = np.append(Rtf, Rtf[-1]+tf['r'][0]*(1-np.cos(theta)))
        Ztf = np.append(Ztf, Ztf[-1]+tf['r'][0]*np.sin(theta))
        theta = np.linspace(theta[-1],tf['alpha'][0]+tf['alpha'][1],
                            round(0.5*Nspace*tf['alpha'][1]/np.pi))  # arc 1
        Rtf = np.append(Rtf, Rtf[-1]+
                        tf['r'][1]*(np.cos(tf['alpha'][0])-np.cos(theta)))
        Ztf = np.append(Ztf, Ztf[-1]+
                        tf['r'][1]*(np.sin(theta)-np.sin(tf['alpha'][0])))
        tf['r'].append((Ztf[-1]-tf['origin'][1])/np.sin(np.pi-sum(tf['alpha'])))                
        theta = np.linspace(theta[-1],np.pi,60)  # arc 2
        Rtf = np.append(Rtf, Rtf[-1]+
                        tf['r'][2]*(np.cos(np.pi-theta)-
                                    np.cos(np.pi-sum(tf['alpha']))))
        Ztf = np.append(Ztf, Ztf[-1]-
                        tf['r'][2]*(np.sin(sum(tf['alpha']))-np.sin(np.pi-theta)))
                        
        Rtf = np.append(Rtf, Rtf[::-1])[::-1]
        Ztf = np.append(Ztf, -Ztf[::-1]+2*tf['origin'][1])[::-1] 
        
        length = np.cumsum(np.sqrt(np.diff(Rtf)**2+np.diff(Ztf)**2))
        length = np.append(0,length)
        L = length[-1]
        length = length/L
        Rtf = interp1(length,Rtf)(np.linspace(0,1,Nspace))
        Ztf = interp1(length,Ztf)(np.linspace(0,1,Nspace))
        return (Rtf,Ztf,L)
        
    def TFcoil(self,calc=True,plot_bounds=False):
        self.set_TFbound()  # TF boundary conditions
        self.TFbound['ro_min'] -= 0.5  # offset minimum radius
        if plot_bounds:
            self.plot_TFbounds()          
        self.TFopp(calc)
        self.TFfill()

    def TFopp(self,TFopp,**kwargs):
        if 'objF' not in kwargs:
            self.TFobj = self.setup.TF['opp']
        if TFopp:
            #xCoilo = [5.12950249,1.89223928,4.13904361,4.52853726,-1.08407415]  
            xCoilo = [ 6.42769145,1.26028077,4.51906176,4.53712734,-2.22330594] 
            #xCoilo = [ 4.36308341,3.00229706,3.66983675, 4.56925003,-0.15107878]
            # [zCoil,r1,r2,ro,zo]  
            #xCoilo = [ 4.77411362,0.70016516,1.15501718,4.66075124,-1.03606014]
            #xCoilo = [ 3,0.71389467,1.46209824,4.42292996,-1.04186448]
            constraint = {'type':'ineq','fun':self.dotTF}
            limit = 1e2
            bounds = [(0.1,limit),(1,limit),(1,limit),  # (0.7,None),(0.1,None)
                      (self.TFbound['ro_min'],limit),(-limit,limit)]  # 
            self.xCoil = op.minimize(self.fitTF,xCoilo,bounds=bounds,
                                     options={'disp':True,'maxiter':500},
                                     method='SLSQP',constraints=constraint,
                                     tol=1e-2).x
            
            with open('../Data/'+self.dataname+'_TF.pkl', 'wb') as output:
                pickle.dump(self.xCoil,output,-1)
        else:
            with open('../Data/'+self.dataname+'_TF.pkl', 'rb') as input:
                self.xCoil = pickle.load(input)
        self.volume_ratio()
        self.length_ratio()
        
    def TFwidth(self,sf):
        '''
        Icoil = 2*np.pi*sf.rcentr*\
        np.abs(sf.bcentr)/(4*np.pi*1e-7)/self.nTF # coil amp-turns
        Acoil = Icoil/self.Jmax
        self.dRcoil = np.sqrt(Acoil)
        '''
        self.dRcoil = 0.489
        self.dRsteel = 0.15*self.dRcoil
 
    def TFfill(self,part_fill=True):
        self.fitTF(self.xCoil)
        self.R,self.Z,L = self.drawTF(self.xCoil, Nspace=self.Np)  
        self.TFinR,self.TFinZ = self.R,self.Z
        self.rzPut()
        self.fill(dR=0,dt=self.dRsteel,trim=None,color='k',alpha=0.6,
                  part_fill=False,loop=True)
        self.fill(dt=self.dRcoil,trim=None,color='k',alpha=0.3,
                  label='TF coil',part_fill=False,loop=True)
        self.fill(dt=self.dRsteel,trim=None,color='k',alpha=0.6,
                  label='TF support',part_fill=False,loop=True)
        self.TFoutR,self.TFoutZ = self.R,self.Z
        
        with open('../Data/'+self.dataname+'_TFcoil.txt','w') as f:
            f.write('Rin m\t\tZin m\t\tRout m\t\tZout m\n')
            for rin,zin,rout,zout in zip(self.TFinR,self.TFinZ,
                                         self.TFoutR,self.TFoutZ):
                f.write('{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n'.format(\
                rin,zin,rout,zout))            
        
    def TFsupport(self,**kwargs):
        self.rzGet()
        self.fill(**kwargs)
        
    def vol_calc(self,R,Z):
        dR = np.diff(R)
        dZ = np.diff(Z)
        V = 0
        for r,dr,dz in zip(R[:-1],dR,dZ):
            V += np.abs((r+dr/2)**2*dz)
        V *= np.pi
        return V
    
    def volume_ratio(self):
        #pl.plot(self.Rp,self.Zp,'rx-')
        self.Pvol = self.loop_vol(self.Rp,self.Zp,plot=False)
        Rtf,Ztf,L = self.drawTF(self.xCoil, Nspace=300)
        self.TFvol = self.loop_vol(Rtf,Ztf,plot=False)
        self.Rvol = self.TFvol/self.Pvol
        
    def length_ratio(self):
        self.Plength = self.length(self.Rp,self.Zp,norm=False)[-1]
        Rtf,Ztf,L = self.drawTF(self.xCoil, Nspace=300)
        R,Z = self.offset(Rtf,Ztf,self.dRsteel+self.dRcoil/2)
        self.TFlength = self.length(R,Z,norm=False)[-1]
        self.Rlength = self.TFlength/self.Plength
    
    def loop_vol(self,R,Z,plot=False):
        imin,imax = np.argmin(Z),np.argmax(Z)
        Rin = np.append(R[::-1][:imin+1][::-1],R[:imin+1])
        Zin = np.append(Z[::-1][:imin+1][::-1],Z[:imin+1])
        Rout = R[imin:imax+1]
        Zout = Z[imin:imax+1]
        if plot:
            pl.plot(R[0],Z[0],'bo')
            pl.plot(R[20],Z[20],'bd')
            pl.plot(Rin,Zin,'bx-')
            pl.plot(Rout,Zout,'gx-')
        return self.vol_calc(Rout,Zout)-self.vol_calc(Rin,Zin)
    
    def fitTF(self,xCoil):
        Rtf,Ztf,L = self.drawTF(xCoil, Nspace=500)
        if self.TFobj is 'L':  # coil length
            objF = L
        else:  # coil volume
            objF = self.loop_vol(Rtf,Ztf)
        return objF
        
    def dot_diffrence(self,xCoil,Side):
        Rloop,Zloop,L = self.drawTF(xCoil, Nspace=500)
        switch = 1 if Side is 'in' else -1
        if Side is 'out':
            Rloop,Zloop = self.offset(Rloop,Zloop,self.dRcoil+
                                      2*self.dRsteel)
        nRloop,nZloop = self.normal(Rloop,Zloop)
        R,Z = self.TFbound[Side]['R'],self.TFbound[Side]['Z']
        dsum = 0
        for r,z in zip(R,Z):
            i = np.argmin((r-Rloop)**2+(z-Zloop)**2)
            dr = [Rloop[i]-r,Zloop[i]-z]  
            dn = [nRloop[i],nZloop[i]]
            dot = switch*np.dot(dr,dn)
            if dot < 0:
                dsum -= (dr[0]**2+dr[1]**2)
        return dsum
        
    def dotTF(self,xCoil):
        dsum = 0
        for Side in ['in','out']:
            dsum += self.dot_diffrence(xCoil,Side) 
        return dsum
        
    def set_TFbound(self):
        Rin,Zin = self.coil_corners(self.setup.coils['internal'])  # internal    
        Rex,Zex = self.coil_corners(self.setup.coils['external'])  # external 
        ro_min = 4.35  # minimum TF radius
        internal = {'R':np.append(self.R,Rin),'Z':np.append(self.Z,Zin)}
        external = {'R':Rex,'Z':Zex}
        self.TFbound = {'in':internal,'out':external,'ro_min':ro_min}
        
    def plot_TFbounds(self):
        pl.plot(self.TFbound['in']['R'],self.TFbound['in']['Z'],'d',
                markersize=3)
        pl.plot(self.TFbound['out']['R'],self.TFbound['out']['Z'],'o',
                markersize=3)
        pl.plot(self.TFbound['ro_min'],0,'*',markersize=3)