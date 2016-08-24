import numpy as np
import pylab as pl
import pickle
from scipy.interpolate import interp1d as interp1
import scipy.optimize as op
from itertools import cycle,count
import seaborn as sns
import matplotlib
import collections
from amigo.geom import Loop
import amigo.geom as geom

def loop_vol(R,Z,plot=False):
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
    return vol_calc(Rout,Zout)-vol_calc(Rin,Zin)
    
def vol_calc(R,Z):
    dR = np.diff(R)
    dZ = np.diff(Z)
    V = 0
    for r,dr,dz in zip(R[:-1],dR,dZ):
        V += np.abs((r+dr/2)**2*dz)
    V *= np.pi
    return V
    
class PF(object):
    def __init__(self,eqdsk):
        self.set_coils(eqdsk)
        
    def set_coils(self,eqdsk):
        self.coil = {}
        if eqdsk['ncoil'] > 0: 
            nC = count(0)
            for i,(r,z,dr,dz,I) in enumerate(zip(eqdsk['rc'],eqdsk['zc'],
                                                 eqdsk['drc'],eqdsk['dzc'],
                                                 eqdsk['Ic'])):
                name = 'Coil{:1.0f}'.format(next(nC))
                self.coil[name] = {'r':r,'z':z,'dr':dr,'dz':dz,'I':I,
                                   'rc':np.sqrt(dr**2+dz**2)/2}
                #if i>=10:
                #    print('exit set_coil loop - coils')
                #    break
         
    def unpack_coils(self):
        nc = len(self.coil.keys())
        Ic = np.zeros(nc)
        rc,zc,drc,dzc = np.zeros(nc),np.zeros(nc),np.zeros(nc),np.zeros(nc)
        names = []
        for i,name in enumerate(self.coil.keys()):
            rc[i] = self.coil[name]['r']
            zc[i] = self.coil[name]['z']
            drc[i] = self.coil[name]['dr']
            dzc[i] = self.coil[name]['dz']
            Ic[i] = self.coil[name]['I']
            names.append(name)
        return nc,rc,zc,drc,dzc,Ic,names
        
    def plot_coil(self,coils,label=False,current=False,coil_color=None,fs=12):
        if coil_color==None:
            color = cycle(sns.color_palette('Set2',len(coils.keys())))
        else:
            color = coil_color  # color itterator
        
        color = sns.color_palette('Set3',300)
        for i,name in enumerate(coils.keys()):
            coil = coils[name]
            if label and current:
                zshift = coil['dz']/4
            else:
                zshift = 0
            r,z,dr,dz = coil['r'],coil['z'],coil['dr'],coil['dz']
            Rfill = [r+dr/2,r+dr/2,r-dr/2,r-dr/2]
            Zfill = [z-dz/2,z+dz/2,z+dz/2,z-dz/2]
            if coil['I'] != 0:
                edgecolor = 'k'
            else:
                edgecolor = 'r'  
            #coil_color = next(color)
            try:
                ic = int(name.split('_')[0].split('Coil')[-1])
            except:
                ic = 0  # plasma coils
            coil_color = color[ic]   
            pl.fill(Rfill,Zfill,facecolor=coil_color,alpha=0.75,
                    edgecolor=edgecolor)
            if label: 
                pl.text(r,z+zshift,name.replace('Coil',''),fontsize=fs*0.6,
                        ha='center',va='center',color=[0.25,0.25,0.25])
            if current: 
                if r<self.Xpoint[0]:
                    drs = -2/3*dr
                    ha = 'right'
                else:
                    drs = 2/3*dr
                    ha = 'left'
                pl.text(r+drs,z-zshift,'{:1.1f}MA'.format(coil['I']*1e-6),
                        fontsize=fs*0.6,ha=ha,va='center',
                        color=[0.25,0.25,0.25])
                                  
    def plot(self,color=None,coils=None,label=False,plasma=False,
                   current=False):
        fs = matplotlib.rcParams['legend.fontsize']
        if coils is None:
            coils = self.coil
        self.plot_coil(coils,label=label,current=current,fs=fs,
                       coil_color=color)
        if plasma:
            coils = self.plasma_coil                
            self.plot_coil(coils,coil_color=color)
            
    def coil_corners(self,coils):
        R,Z = np.array([]),np.array([])
        Nc = len(coils['id'])
        dR,dZ = np.zeros(Nc),np.zeros(Nc)
        if len(coils['dR'])>0: 
            dR[Nc-len(coils['dR']):]=coils['dR']
        if len(coils['dZ'])>0: 
            dZ[Nc-len(coils['dZ']):]=coils['dZ']
        for Cid,Cdr,Cdz in zip(coils['id'],dR,dZ):
            r = self.coil['Coil'+str(Cid)]['r']
            z = self.coil['Coil'+str(Cid)]['z']
            dr = self.coil['Coil'+str(Cid)]['dr']
            dz = self.coil['Coil'+str(Cid)]['dz']
            if Cdr==0 and Cdz==0:
                R = np.append(R,[r+dr/2,r+dr/2,r-dr/2,r-dr/2])
                Z = np.append(Z,[z+dz/2,z-dz/2,z+dz/2,z-dz/2])
            else:
                R = np.append(R,r+Cdr)
                Z = np.append(Z,z+Cdz)
        return R,Z
              
    def Cshift(self,coil,side,dL):
        if 'in' in side:
            R,Z = self.TFinR,self.TFinZ
        else:
            R,Z = self.TFoutR,self.TFoutZ
        rc,zc = coil['r'],coil['z']
        drc,dzc = coil['dr'],coil['dz']
        rp = rc+np.array([-drc,drc,drc,-drc])/2
        zp = zc+np.array([-dzc,-dzc,dzc,dzc])/2
        nR,nZ = self.normal(R,Z)       
        mag = np.sqrt(nR**2+nZ**2)
        nR /= mag
        nZ /= mag
        i = []
        L = np.empty(len(rp))
        dn = np.empty((2,len(rp)))
        for j,(r,z) in enumerate(zip(rp,zp)):
            i.append(np.argmin((R-r)**2+(Z-z)**2))
            dr = [r-R[i[-1]],z-Z[i[-1]]]  
            dn[:,j] = [nR[i[-1]],nZ[i[-1]]]
            L[j] = np.dot(dr,dn[:,j])
            if 'in' in side: L[j] *= -1
        jc = np.argmin(L)
        fact = dL-L[jc]
        if 'in' in side: fact *= -1
        delta = fact*dn[:,jc]
        return delta[0],delta[1]
    
    def fit_coils(self,Cmove,dLo=0.1):
        coils = collections.OrderedDict()
        for side in Cmove.keys():
            if isinstance(dLo,(list,tuple)):
                if not 'in' in side:
                    dL = dLo[0]
                else:
                    dL = dLo[-1]
            else:
                dL = dLo
            for index in Cmove[side]:
                coil = 'Coil'+str(index)
                dr,dz = self.Cshift(self.sf.coil[coil],side,dL)
                coils[coil] = self.sf.coil[coil]
                coils[coil]['r'] = coils[coil]['r']+dr
                coils[coil]['z'] += dz
                coils[coil]['shiftr'] = dr
                coils[coil]['shiftz'] = dz
                
        with open('../Data/'+self.conf.config+'_coil_shift.txt','w') as f:
            f.write('Ncoils = {:1.0f}\n\n'.format(len(coils.keys())))
            f.write('Name\tR[m]\t\tZ[m]\t\tshiftR[m]\tshiftZ[m]\n')
            index = sorted(map((lambda s: int(s.strip('Coil'))),coils.keys()))
            for i in index:
                coil = 'Coil'+str(i)
                f.write(coil+'\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n'.format(\
                coils[coil]['r'],coils[coil]['z'],\
                coils[coil]['shiftr'],coils[coil]['shiftz']))                                   
        return coils

class TF(object):
    
    def __init__(self,setup,fit=False,Np=200,objective='V',**kwargs):
        self.objective = objective
        self.Np = Np
        self.setup = setup
        self.dataname = setup.dataname
        self.width()
        self.calc(fit,**kwargs)
        self.fill(write=False)

    def draw(self,xCoil,Nspace=100):
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
        L = np.cumsum(np.sqrt(np.diff(Rtf)**2+np.diff(Ztf)**2))
        L = np.append(0,L)
        return (Rtf,Ztf,L[-1])

    def calc(self,fit,plot_bounds=False,**kwargs):
        #if 'objF' not in kwargs:
        #    self.TFobj = self.setup.TF['opp']
        if fit: 
            self.set_bound(**kwargs)  # TF boundary conditions
            self.bound['ro_min'] -= 0.5  # offset minimum radius
            if plot_bounds:
                self.plot_bounds() 
            xCoilo = [ 6.42769145,1.26028077,4.51906176,4.53712734,-2.22330594] 
            constraint = {'type':'ineq','fun':self.dot}
            limit = 1e2
            bounds = [(0.1,limit),(1,limit),(1,limit),  # (0.7,None),(0.1,None)
                      (self.bound['ro_min'],limit),(-limit,limit)]  # 
            self.xCoil = op.minimize(self.fit,xCoilo,bounds=bounds,
                                     options={'disp':True,'maxiter':500},
                                     method='SLSQP',constraints=constraint,
                                     tol=1e-2).x
            with open('../Data/'+self.dataname+'_TF.pkl', 'wb') as output:
                pickle.dump(self.xCoil,output,-1)
        else:
            with open('../Data/'+self.dataname+'_TF.pkl', 'rb') as input:
                self.xCoil = pickle.load(input)
        
    def width(self):
        self.dRcoil = 0.489
        self.dRsteel = 0.15*self.dRcoil
 
    def fill(self,write=True,plot=True):
        #self.fitTF(self.xCoil)
        R,Z,L = self.draw(self.xCoil,Nspace=self.Np) 
        loop = Loop(R,Z,[np.mean(R),np.mean(Z)])
        self.Rin,self.Zin = loop.R,loop.Z
        
        loop.rzPut()
        loop.fill(dR=0,dt=self.dRsteel,trim=None,color='k',alpha=0.6,
                  part_fill=False,loop=True,plot=plot)
        loop.fill(dt=self.dRcoil/2,trim=None,color='k',alpha=0.3,
                  label='TF coil',part_fill=False,loop=True,plot=plot)
        self.Rmid,self.Zmid = loop.R,loop.Z  
        loop.fill(dt=self.dRcoil/2,trim=None,color='k',alpha=0.3,
                  label='TF coil',part_fill=False,loop=True,plot=plot)
        loop.fill(dt=self.dRsteel,trim=None,color='k',alpha=0.6,
                  label='TF support',part_fill=False,loop=True,plot=plot)
        self.Rout,self.Zout = loop.R,loop.Z
        
        if write:
            with open('../Data/'+self.dataname+'_TFcoil.txt','w') as f:
                f.write('Rin m\t\tZin m\t\tRout m\t\tZout m\n')
                for rin,zin,rout,zout in zip(self.TFinR,self.TFinZ,
                                             self.TFoutR,self.TFoutZ):
                    f.write('{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n'.format(\
                    rin,zin,rout,zout))            
        
    def support(self,**kwargs):
        self.rzGet()
        self.fill(**kwargs)
        
    def volume_ratio(self,Rp,Zp):
        #pl.plot(self.Rp,self.Zp,'rx-')
        self.Pvol = loop_vol(Rp,Zp,plot=False)
        Rtf,Ztf,L = self.drawTF(self.xCoil, Nspace=300)
        self.TFvol = loop_vol(Rtf,Ztf,plot=False)
        self.Rvol = self.TFvol/self.Pvol
        
    def length_ratio(self):
        self.Plength = self.length(self.Rp,self.Zp,norm=False)[-1]
        Rtf,Ztf,L = self.drawTF(self.xCoil, Nspace=300)
        R,Z = geom.offset(Rtf,Ztf,self.dRsteel+self.dRcoil/2)
        self.TFlength = self.length(R,Z,norm=False)[-1]
        self.Rlength = self.TFlength/self.Plength
    
    def fit(self,xCoil):
        Rtf,Ztf,L = self.draw(xCoil, Nspace=500)
        if self.objective is 'L':  # coil length
            objF = L
        else:  # coil volume
            objF = loop_vol(Rtf,Ztf)
        return objF
        
    def dot_diffrence(self,xCoil,Side):
        Rloop,Zloop,L = self.draw(xCoil, Nspace=500)
        switch = 1 if Side is 'in' else -1
        if Side is 'out':
            Rloop,Zloop = geom.offset(Rloop,Zloop,self.dRcoil+
                                      2*self.dRsteel)
        nRloop,nZloop = geom.normal(Rloop,Zloop)
        R,Z = self.bound[Side]['R'],self.bound[Side]['Z']
        dsum = 0
        for r,z in zip(R,Z):
            i = np.argmin((r-Rloop)**2+(z-Zloop)**2)
            dr = [Rloop[i]-r,Zloop[i]-z]  
            dn = [nRloop[i],nZloop[i]]
            dot = switch*np.dot(dr,dn)
            if dot < 0:
                dsum -= (dr[0]**2+dr[1]**2)
        return dsum
        
    def dot(self,xCoil):
        dsum = 0
        for Side in ['in','out']:
            dsum += self.dot_diffrence(xCoil,Side) 
        return dsum
        
    def set_bound(self,**kwargs):
        pf = kwargs.get('pf')  # requre loop,pf
        loop = kwargs.get('loop')
        Rin,Zin = pf.coil_corners(self.setup.coils['internal'])  # internal    
        Rex,Zex = pf.coil_corners(self.setup.coils['external'])  # external 
        ro_min = 4.35  # minimum TF radius
        internal = {'R':np.append(loop.R,Rin),'Z':np.append(loop.Z,Zin)}
        external = {'R':Rex,'Z':Zex}
        self.bound = {'in':internal,'out':external,'ro_min':ro_min}
        
    def plot_bounds(self):
        pl.plot(self.bound['in']['R'],self.TFbound['in']['Z'],'d',
                markersize=3)
        pl.plot(self.bound['out']['R'],self.TFbound['out']['Z'],'o',
                markersize=3)
        pl.plot(self.bound['ro_min'],0,'*',markersize=3)
        
