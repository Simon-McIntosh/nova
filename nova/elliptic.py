import pylab as pl
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg.dsolve.linsolve import spsolve
import cross_coil as cc
import scipy.optimize as op
from copy import deepcopy
from scipy.optimize import minimize
from itertools import cycle
import seaborn as sns
Color = cycle(sns.color_palette('Set2'))
from numpy.random import uniform
from matplotlib.colors import Normalize
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.linalg import lstsq

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

class EQ(object):
    def __init__(self,sf,pf,sigma=0,dCoil=0.5,**kwargs):
        self.mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]
        self.sf = sf
        self.pf = pf
        self.coils(dCoil=dCoil)  # multi-filiment coils 
        self.get_Vcoil()  # identify vertical stability coils
        self.resample(sigma=sigma,**kwargs)
        
    def resample(self,sigma=0,**kwargs):  # resample current density
        self.r,self.z = self.sf.r,self.sf.z 
        self.r2d,self.z2d = self.sf.r2d,self.sf.z2d    
        self.dr,self.dz,self.dA = self.sf.dr,self.sf.dz,self.sf.dr*self.sf.dz
        self.psi = self.sf.psi  # copy psi
        self.GSoper()  # apply GS opperator to psi
        GSspline = RBS(self.r,self.z,self.GS)  # construct spline interpolator
        self.grid(**kwargs)  # update solution grid
        self.GS = GSspline.ev(self.r2d,self.z2d)  # interpolate b
        self.GS = self.convolve(self.GS,sigma=sigma)  # post interpolation filter
        self.b = np.copy(np.reshape(self.GS,self.nr*self.nz))  # set core
        self.edgeBC(update=False)  # set edge
        self.psi = self.solve()  # re-solve
        self.set_eq_psi()  # pass grid and psi back to sf

    def limits(self,boundary):
        R,Z = boundary.get('R'),boundary.get('Z')
        for key in ['rmin','rmax','zmin','zmax']:
            if key in boundary.keys():
                if 'r' in key:
                    var = R
                else:
                    var = Z
                if 'min' in key:
                    index = var>boundary.get(key)
                else:
                    index = var<boundary.get(key)
                R,Z = R[index],Z[index]
        lim = np.array([R.min(),R.max(),Z.min(),Z.max()])
        if 'expand' in boundary.keys():
            expand = boundary.get('expand')
        else:
            expand = 0.5
        for i,direction in enumerate([-1,1,-1,1]):
            lim[i] += direction*expand      
        return lim

    def grid(self,**kwargs):
        if 'boundary' in kwargs:
            limit = self.limits(kwargs.get('boundary'))
        elif 'limit' in kwargs:
            limit = kwargs.get('limit')
        elif 'delta' in kwargs:
            delta = kwargs.get('delta')
            limit = np.array([self.sf.r[0]+delta,self.sf.r[-1]-delta,
                              self.sf.z[0]+delta,self.sf.z[-1]-delta])
        else:
            limit = np.array([self.sf.r[0],self.sf.r[-1],
                              self.sf.z[0],self.sf.z[-1]])
        if 'n' in kwargs.keys():
            n = kwargs.get('n')
        else:
            n = self.sf.nr*self.sf.nz
        self.limit,self.n = limit,n
        ro,zo = limit[:2],limit[2:]
        dro,dzo = (ro[-1]-ro[0]),(zo[-1]-zo[0])
        ar = dro/dzo
        self.nz = int(np.sqrt(n/ar))
        self.nr = int(n/self.nz)
        self.r = np.linspace(ro[0],ro[1],self.nr)
        self.z = np.linspace(zo[0],zo[1],self.nz)
        self.r2d,self.z2d = np.meshgrid(self.r,self.z,indexing='ij')
        self.N = self.nr*self.nz
        self.b = np.zeros(self.N)
        self.bpl = np.zeros(self.N)
        self.dr = (self.r[-1]-self.r[0])/(self.nr-1)
        self.dz = (self.z[-1]-self.z[0])/(self.nz-1)    
        self.rc = np.linspace(ro[0]-self.dr/2,ro[1]+self.dr/2,self.nr+1)
        self.zc = np.linspace(zo[0]-self.dz/2,zo[1]+self.dz/2,self.nz+1)
        self.r2dc,self.z2dc = np.meshgrid(self.rc,self.zc,indexing='ij')
        self.dA = self.dr*self.dz
        self.A = self.matrix()
        self.boundary()
        self.edge()
        self.core_index()
        
    def core_index(self):
        self.core_indx = np.ones(self.N,dtype=bool)
        self.core_indx[self.indx(0,np.arange(self.nz))] = 0
        self.core_indx[self.indx(np.arange(self.nr),0)] = 0
        self.core_indx[self.indx(self.nr-1, np.arange(self.nz))] = 0
        self.core_indx[self.indx(np.arange(self.nr),self.nz-1)] = 0
        
    def coils(self,dCoil=-1):
        if dCoil < 0:  # dCoil not set, use stored value
            if not hasattr(self,'dCoil'):
                self.dCoil = 0
        else:
            self.dCoil = dCoil
        coil = self.pf.coil
        if self.dCoil==0:
            self.coil = coil
            for name in coil.keys():
                self.coil[name]['rc'] = np.sqrt(coil[name]['dr']**2+
                                                coil[name]['dr']**2)
        else:
            self.coil = {}
            for name in coil.keys():
                self.size_coil(coil,name,self.dCoil)
                        
    def size_coil(self,coil,name,dCoil):
        rc,zc = coil[name]['r'],coil[name]['z']
        Dr,Dz = coil[name]['dr'],coil[name]['dz']
        if coil[name]['I'] != 0:
            if Dr>0 and Dz>0 and 'plasma' not in name:
                nr,nz = np.ceil(Dr/dCoil),np.ceil(Dz/dCoil)
                dr,dz = Dr/nr,Dz/nz
                r = rc+np.linspace(dr/2,Dr-dr/2,nr)-Dr/2
                z = zc+np.linspace(dz/2,Dz-dz/2,nz)-Dz/2
                R,Z = np.meshgrid(r,z,indexing='ij')
                R,Z = np.reshape(R,(-1,1)),np.reshape(Z,(-1,1))
                Nf = len(R)  # filament number
                I = coil[name]['I']/Nf
                bundle = {'r':np.zeros(Nf),'z':np.zeros(Nf),
                          'dr':dr*np.ones(Nf),'dz':dz*np.ones(Nf),
                          'I':I*np.ones(Nf),'sub_name':np.array([]),
                          'Nf':0}
                for i,(r,z) in enumerate(zip(R,Z)):
                    sub_name = name+'_{:1.0f}'.format(i)
                    self.coil[sub_name] = {'r':r,'z':z,'dr':dr,'dz':dz,
                                           'I':I,'Nf':Nf,
                                           'rc':np.sqrt(dr**2+dz**2)/2}
                    bundle['r'][i],bundle['z'][i] = r,z
                    bundle['sub_name'] = np.append(bundle['sub_name'],sub_name)
                bundle['Nf'] = i+1
                bundle['Ro'] = np.mean(bundle['r'])
                bundle['Zo'] = np.mean(bundle['z'])
            else:
                print(name)
                self.coil[name] = coil[name]
                bundle = coil[name]
        return bundle
                          
    def get_Vcoil(self):
        ex_coil = []
        for name in self.pf.coil.keys():
            if 'plasma' not in name:  
                ex_coil.append(name)
        Nex = len(ex_coil)
        self.Vcoil = np.zeros((Nex,),dtype=[('name','S10'),('value','float'),
                                            ('Io','float'),('Ip','float'),
                                            ('Ii','float'),('Nf','int')])
        for i,name in enumerate(ex_coil):
            self.Vcoil['name'][i] = name
            self.Vcoil['Io'][i] = self.pf.coil[name]['I']
            self.Vcoil['Nf'][i] = self.coil[name+'_0']['Nf']  
            Mpoint = self.sf.Mpoint
            r,z = self.pf.coil[name]['r'],self.pf.coil[name]['z']
            I = self.pf.coil[name]['I']
            self.Vcoil['value'][i] = np.sign(I)*cc.green_feild(Mpoint[0],Mpoint[1],
                                                      r,z)[0]  
        self.Vcoil = np.sort(self.Vcoil,order='value')
        self.Vcoil['value'] *= abs(self.Vcoil['Io'])
        
    def indx(self,i,j):
        return i*self.nz+j
        
    def ij(self,indx):
        j = np.mod(indx,self.nz)
        i = int((indx-j)/self.nz)
        return i,j
        
    def matrix(self):
        A = lil_matrix((self.N,self.N))
        A.setdiag(np.ones(self.N))
        for i in range(1,self.nr-1):
            for j in range(1,self.nz-1):
                rp = 0.5*(self.r[i+1] + self.r[i]) # r_{i+1/2}
                rm = 0.5*(self.r[i] + self.r[i-1]) # r_{i-1/2}
                ind = self.indx(i,j)
                A[ind, ind] = -(self.r[i]/self.dr**2)*\
                (1/rp + 1/rm)-2/self.dz**2
                A[ind, self.indx(i+1,j)] = (self.r[i]/self.dr**2)/rp
                A[ind, self.indx(i-1,j)] = (self.r[i]/self.dr**2)/rm
                A[ind, self.indx(i,j+1)] = 1/self.dz**2
                A[ind, self.indx(i,j-1)] = 1/self.dz**2
        return A.tocsr()
        
    def resetBC(self):
        self.b[self.core_indx] *= 0
        self.b *= 0
        
    def ingrid(self,r,z):
        if r>self.r[0]+self.dr and r<self.r[-1]-self.dr\
        and z>self.z[0]+self.dz and z<self.z[-1]-self.dz:
            return True
        else:
            return False
            
    def normal(self,R,Z):
        dR,dZ = np.gradient(R),np.gradient(Z)
        mag = np.sqrt(dR**2+dZ**2)
        index = mag>0
        dR,dZ,mag = dR[index],dZ[index],mag[index]  # clear duplicates
        R,Z = R[index],Z[index]
        t = np.zeros((len(dR),3))
        t[:,0],t[:,1] = dR/mag,dZ/mag
        n = np.cross(t, [0,0,1])
        nR,nZ = n[:,0],n[:,1]
        return R,Z,nR,nZ
        
    def inloop(self,Rloop,Zloop,R,Z):
        Rloop,Zloop = self.orientate(Rloop,Zloop)
        Rloop,Zloop,nRloop,nZloop = self.normal(Rloop,Zloop)
        Rin,Zin = np.array([]),np.array([])
        for r,z in zip(R,Z):
            i = np.argmin((r-Rloop)**2+(z-Zloop)**2)
            dr = [Rloop[i]-r,Zloop[i]-z]  
            dn = [nRloop[i],nZloop[i]]
            if np.dot(dr,dn) > 0:
                Rin,Zin = np.append(Rin,r),np.append(Zin,z)
        return Rin,Zin
            
    def psi_ex(self):
        psi = np.zeros(self.Ne)
        for name in self.coil.keys():
            r,z = self.coil[name]['r'],self.coil[name]['z']
            I = self.coil[name]['I']
            if not self.ingrid(r,z):
                psi += self.mu_o*I*cc.green(self.Re,self.Ze,r,z)
        self.psi_external = psi
        return psi   
    
    def psi_pl(self):
        psi_core = self.solve()  # solve with zero edgeBC
        dgdn = self.boundary_normal(psi_core)
        psi = np.zeros(self.Ne)
        for i,(r,z) in enumerate(zip(self.Re,self.Ze)):
            circ = -cc.green(r,z,self.Rb,self.Zb)*dgdn/self.Rb
            psi[i] = np.trapz(circ,self.Lb)
        self.psi_e_plasma = psi
        return psi
        
    def psi_edge(self):
        psi = np.zeros(self.Ne)
        for i,(r,z) in enumerate(zip(self.Re,self.Ze)):
            psi[i] = self.sf.Pcoil((r,z))
        return psi

    def coil_core(self):
        for name in self.coil.keys():
            r,z = self.coil[name]['r'],self.coil[name]['z']
            I = self.coil[name]['I']
            if self.ingrid(r,z):
                i = np.argmin(np.abs(r-self.r))
                j = np.argmin(np.abs(z-self.z))
                self.b[self.indx(i,j)] += -self.mu_o*I*self.r[i]/(self.dA)
    
    def orientate(self,r,z):
        theta = np.arctan2(z-self.sf.Mpoint[1],r-self.sf.Mpoint[0])
        index = np.argsort(theta)
        r,z = r[index],z[index]
        return r,z
                
    def plasma_core(self,update=True):
        if update:  # calculate plasma contribution
            rbdry,zbdry = self.sf.get_boundary(alpha=1-1e-3)
            #rbdry,zbdry = self.sf.get_boundary(alpha=0.8)  # !!!!!!!! # for vde
            R,Z = self.inloop(rbdry,zbdry,
                              self.r2d.flatten(),self.z2d.flatten())
            self.Ip,self.Nplasma = 0,0
            self.plasma_index = np.zeros(self.N,dtype=int)
            for r,z in zip(R,Z):
                i = np.argmin(np.abs(r-self.r))
                j = np.argmin(np.abs(z-self.z))
                indx = self.indx(i,j)
                psi = (self.psi[i,j]-self.sf.Mpsi)/(self.sf.Xpsi-self.sf.Mpsi)
                if psi<1:
                    self.Nplasma += 1
                    self.plasma_index[self.Nplasma-1] = indx
                    self.bpl[indx] = -self.mu_o*r**2*self.sf.Pprime(psi)\
                    -self.sf.FFprime(psi)
                    self.Ip -= self.dA*self.bpl[indx]/(self.mu_o*r)
            scale_plasma = self.sf.cpasma/self.Ip
            #scale_plasma = 1
            #print(scale_plasma)
            self.sf.b_scale = scale_plasma
            for i,indx in zip(range(self.Nplasma),self.plasma_index):
                self.bpl[indx] *= scale_plasma
                self.b[indx] = self.bpl[indx]
        
    def set_plasma_coil(self,delta=0):
        self.plasma_coil = {} 
        if delta > 0:
            rbdry,zbdry = self.sf.get_boundary()
            lbdry = self.sf.length(rbdry,zbdry)
            rc = np.mean([rbdry.min(),rbdry.max()])
            zc = np.mean([zbdry.min(),zbdry.max()])
            radius = interp1(lbdry,((rbdry-rc)**2+(zbdry-zc)**2)**0.5)
            theta = interp1(lbdry,np.arctan2(zbdry-zc, rbdry-rc))
            Length = np.linspace(lbdry[0],lbdry[-1],len(lbdry))
            Radius = radius(Length)
            Theta = theta(Length)
            nr = np.ceil(Radius.max()/delta)
            R,Z = np.array([]),np.array([])
            for rfact in np.linspace(1/(2*nr),1-1/(2*nr),nr):
                r,z = rfact*Radius*np.cos(Theta),rfact*Radius*np.sin(Theta)
                L = self.sf.length(r,z,norm=False)[-1]
                nloop = np.ceil(L/delta)
                length = np.linspace(1/nloop,1,nloop)
                rinterp,tinterp = rfact*radius(length),theta(length)
                R = np.append(R,rc+rinterp*np.cos(tinterp))
                Z = np.append(Z,zc+rinterp*np.sin(tinterp))
            Rp,Zp = R.flatten(),Z.flatten()
            Ip,Np = np.zeros(len(R)),np.zeros(len(R))
        else:
            Rp,Zp = np.zeros(self.Nplasma),np.zeros(self.Nplasma)
            Ip,Np = np.zeros(self.Nplasma),np.zeros(self.Nplasma)
        for indx in range(self.Nplasma):
            i,j = self.ij(self.plasma_index[indx])
            r,z = self.r[i],self.z[j]
            I = -self.dA*self.b[self.plasma_index[indx]]/(self.mu_o*r)
            if delta > 0:
                index = np.argmin((Rp-r)**2+(Zp-z)**2)
            else:
                index = indx
                Rp[index],Zp[index] = r,z
            Ip[index] += I
            Np[index] += 1
        indx = -1
        for r,z,I,n in zip(Rp,Zp,Ip,Np):
            if n > 0:
                indx += 1
                self.plasma_coil['Plasma_{:1.0f}'.format(indx)] = {'r':r,'z':z,\
                'dr':self.dr*np.sqrt(n),'dz':self.dz*np.sqrt(n),
                'rc':np.sqrt(n*self.dr**2+n*self.dz**2)/2,
                'I':I,'indx':indx}
        self.sf.plasma_coil = self.plasma_coil
       
    def get_plasma_coil(self,delta=0):
        self.b *= 0
        self.plasma_core()
        self.set_plasma_coil(delta=delta)
        
    def coreBC(self,update=True):
        self.plasma_core(update=update)
        self.coil_core()
        
    def edgeBC(self,update=True,external_coils=True):
        if not update or self.sf.eq['ncoil'] == 0:   # edge BC from sf
            psi = self.psi_edge()
        else:
            psi = self.psi_pl()  # plasma component
            if external_coils: 
                psi += self.psi_ex()  # coils external to grid           
        self.b[self.indx(0,np.arange(self.nz))] = \
        psi[2*self.nr+self.nz:2*self.nr+2*self.nz][::-1]
        self.b[self.indx(np.arange(self.nr),0)] = psi[:self.nr]
        self.b[self.indx(self.nr-1, np.arange(self.nz))] = \
        psi[self.nr:self.nr+self.nz]
        self.b[self.indx(np.arange(self.nr),self.nz-1)] = \
        psi[self.nr+self.nz:2*self.nr+self.nz][::-1]

    def solve(self):
        psi = spsolve(self.A,self.b)
        return np.reshape(psi,(self.nr,self.nz))
        
    def damp_psi(self,alpha=0.95):
        if hasattr(self,'psio'):
            self.psio = alpha*self.psi + (1-alpha)*self.psio 
        else:
            self.psio = self.psi
        self.psi = self.psio 
          
    def run(self,update=True):
        self.resetBC()
        self.coreBC(update=update)
        self.edgeBC()
        self.psi = self.solve()
        print('run')
        self.set_eq_psi()
            
    def set_eq_psi(self):  # set psi from eq
        eqdsk = {'r':self.r,'z':self.z,'psi':self.psi}
        try:
            self.sf.update_plasma(eqdsk)  # include boundary update
        except:
            print('boundary update failed')
            self.sf.set_plasma(eqdsk,contour=True)
    
    def plasma(self):
        self.resetBC()
        self.plasma_core()
        self.edgeBC(external_coils=False)
        self.psi_plasma = self.solve()
        self.set_plasma_coil()

    def gen(self,kp=2.5,ki=0.25,Nmax=100,Verr=1e-4,**kwargs): 
        self.get_Vcoil()
        if 'Vtarget' in kwargs.keys():
            self.Vtarget = kwargs['Vtarget']
        else:
            self.Vtarget = self.sf.Mpoint[1]
        M,Mflag = np.zeros(Nmax),False
        for i in range(Nmax):
            self.run()
            Merr = self.sf.Mpoint[1]-self.Vtarget
            M[i] = Merr
            if abs(Merr) <= Verr:
                if Mflag:
                    print('Vtarget {:1.5f} i={:1.0f} Merr={:1.6e}'.format(\
                    self.Vtarget,i,Merr))
                    break
                Mflag = True
            else:
                Mflag = False
            for index in [-1]:  # vertical stability coil pair [0,-1]
                self.Vcoil['Ip'][index] = np.sign(self.Vcoil['value'][index])*\
                kp*self.Vcoil['Io'][index]*Merr  
                dIi = np.sign(self.Vcoil['value'][index])*ki*self.Vcoil['Io'][index]*Merr 
                if np.sign(self.Vcoil['Ii'][index])*np.sign(self.Vcoil['Ii'][index]+dIi)<0:
                    self.Vcoil['Ii'][index] = 0
                else:
                    self.Vcoil['Ii'][index] += dIi
                for subcoil in range(self.Vcoil['Nf'][index]):
                    subname = self.Vcoil['name'][index].decode()
                    subname += '_{:1.0f}'.format(subcoil)
                    self.coil[subname]['I'] = (self.Vcoil['Io'][index]+\
                    self.Vcoil['Ip'][index]+self.Vcoil['Ii'][index])/self.Vcoil['Nf'][index]
        self.set_plasma_coil()
        
    def fit(self,inv,N=5):
        for i in range(N):
            self.plasma()  # without coils
            inv.solve_slsqp()
            inv.set_background()
            inv.get_weight()
            inv.set_force_feild(state='both')
            self.run()  # with coils
            print(self.sf.Mpoint[1])
            
    def GSoper(self):  # apply GS operator
        edge_order = 2
        dpsi = np.gradient(self.psi,edge_order=edge_order)
        dpsi_r,dpsi_z = dpsi[0]/self.dr,dpsi[1]/self.dz
        GSr = self.r2d*np.gradient(dpsi_r/self.r2d,edge_order=edge_order)[0]/self.dr
        GSz = np.gradient(dpsi_z,edge_order=edge_order)[1]/self.dz
        self.GS = GSr+GSz
        
    def convolve(self,var,sigma=0):
        if sigma > 0:
            var = gaussian_filter(var,sigma*self.dA**-0.5)  # convolution filter
        return var
            
    def getj(self,sigma=0):
        j = -self.GS/(self.mu_o*self.r2d)  # current density from psi
        return self.convolve(j,sigma=sigma)
        
    def sparseBC(self,sigma=0):  # set sparse BC 
        b = self.convolve(self.GS,sigma=sigma)
        self.b = np.reshape(b,self.nr*self.nz)
        
    def set_plasma_current(self):  # plasma current from sf line intergral
        Ip = 0
        R,Z = self.sf.get_boundary()
        #R,Z = self.sf.get_boundary(0.8)  # !!!!!!! # for vde
        tR,tZ,R,Z = cc.tangent(R,Z,norm=False)
        for r,z,tr,tz in zip(R,Z,tR,tZ):
            B = self.sf.Bcoil((r,z))
            t = np.array([tr,tz])
            Ip += np.dot(B,t)
        Ip /= cc.mu_o 
        self.sf.cpasma = Ip
        
    def fluxfunctions(self,npsi=100,sigma=0):
        FFp,Pp = np.ones(npsi),np.ones(npsi)  # flux functions
        psi_norm = np.linspace(1e-3,1-1e-3,npsi)
        j = self.getj(sigma=sigma)  # current denstiy
        bspline = RBS(self.r,self.z,j) 
        for i,psi in enumerate(psi_norm):
            rb,zb = self.sf.get_boundary(alpha=psi)
            Lb = self.sf.length(rb,zb)
            L,dL = np.linspace(Lb.min(),Lb.max(),len(Lb),retstep=True)
            r,z = np.interp(L,Lb,rb),np.interp(L,Lb,zb)  # even spacing
            js = bspline.ev(r,z)          
            A = np.matrix(np.ones((len(r),2)))
            A[:,0] = np.reshape(1/(self.mu_o*r),(-1,1))
            A[:,1] = np.reshape(r,(-1,1))
            FFp[i],Pp[i] = lstsq(A,js)[0]  # fit flux functions
            js_lstsq = r*Pp[i]+FFp[i]/(self.mu_o*r)  # evaluate
            jfact = sum(js)/sum(js_lstsq)
            FFp[i] *= jfact  # normalize
            Pp[i] *= jfact  # normalize
        return FFp,Pp,psi_norm
        
    def set_fluxfunctions(self,update=True,sigma=0):  # sigma==smoothing sd [m]
        self.FFp,self.Pp,self.psi_norm = self.fluxfunctions(sigma=sigma) 
        self.psi_norm = np.append(0,self.psi_norm)  # bounds 0-1
        self.psi_norm[-1] = 1 
        self.FFp = np.append(self.FFp[0],self.FFp)   
        self.Pp = np.append(self.Pp[0],self.Pp)   
        self.sf.FFprime_o = self.sf.FFprime  # store previous
        self.sf.Pprime_o = self.sf.Pprime
        if update:  # replace with new functions
            self.sf.FFprime = interp1d(self.psi_norm,self.FFp)
            self.sf.Pprime = interp1d(self.psi_norm,self.Pp)  
        else:  # retain old function shapes and scale to fit
            FFscale = sum(self.FFp)/sum(self.sf.FFprime_o(self.psi_norm))
            Pscale = sum(self.Pp)/sum(self.sf.Pprime_o(self.psi_norm))
            self.sf.FFprime = interp1d(self.psi_norm,FFscale*
                                       self.sf.FFprime_o(self.psi_norm))
            self.sf.Pprime = interp1d(self.psi_norm,Pscale*
                                      self.sf.Pprime_o(self.psi_norm))  

    def edge(self):
        R = np.zeros(2*(self.nr+self.nz))
        Z = np.zeros(2*(self.nr+self.nz))
        R[:self.nr] = self.r
        R[self.nr:self.nr+self.nz] = self.r[-1]
        R[self.nr+self.nz:2*self.nr+self.nz] = self.r[::-1]
        R[2*self.nr+self.nz:2*self.nr+2*self.nz] = self.r[0]
        Z[:self.nr] = self.z[0]
        Z[self.nr:self.nr+self.nz] = self.z
        Z[self.nr+self.nz:2*self.nr+self.nz] = self.z[-1]
        Z[2*self.nr+self.nz:2*self.nr+2*self.nz] = self.z[::-1]
        self.Re,self.Ze,self.Ne = R,Z,len(R)
        
    def boundary(self):
        R = np.zeros(2*(self.nr-1)+2*(self.nz-1))
        Z = np.zeros(2*(self.nr-1)+2*(self.nz-1))
        dL = np.zeros(2*(self.nr-1)+2*(self.nz-1))
        L = np.zeros(2*(self.nr-1)+2*(self.nz-1))
        R[:self.nr-1] = self.r[:-1]+self.dr/2
        R[self.nr-1:self.nr+self.nz-2] = self.r[-1]
        R[self.nr+self.nz-2:2*self.nr+self.nz-3] = self.r[:-1][::-1]+self.dr/2
        R[2*self.nr+self.nz-3:2*self.nr+2*self.nz-4] = self.r[0]
        Z[:self.nr-1] = self.z[0]
        Z[self.nr-1:self.nr+self.nz-2] = self.z[:-1]+self.dz/2
        Z[self.nr+self.nz-2:2*self.nr+self.nz-3] = self.z[-1]
        Z[2*self.nr+self.nz-3:2*self.nr+2*self.nz-4] = \
        self.z[:-1][::-1]+self.dz/2
        dL[:self.nr-1] = self.dr
        dL[self.nr-1:self.nr+self.nz-2] = self.dz
        dL[self.nr+self.nz-2:2*self.nr+self.nz-3] = self.dr
        dL[2*self.nr+self.nz-3:2*self.nr+2*self.nz-4] = self.dz
        L[:self.nr-1] = self.dr/2+np.cumsum(self.dr*np.ones(self.nr-1))-self.dr
        L[self.nr-1:self.nr+self.nz-2] = L[self.nr-2]+(self.dr+self.dz)/2+\
        np.cumsum(self.dz*np.ones(self.nz-1))-self.dz
        L[self.nr+self.nz-2:2*self.nr+self.nz-3] = \
        L[self.nr+self.nz-3]+(self.dz+self.dr)/2+\
        np.cumsum(self.dr*np.ones(self.nr-1))-self.dr
        L[2*self.nr+self.nz-3:2*self.nr+2*self.nz-4] = \
        L[2*self.nr+self.nz-4]+(self.dr+self.dz)/2+\
        np.cumsum(self.dz*np.ones(self.nz-1))-self.dz
        L[-1] += self.dz/2
        self.Rb,self.Zb,self.dL,self.Lb = R,Z,dL,L
        
    def boundary_normal(self,psi):
        dgdn = np.zeros(2*(self.nr-1)+2*(self.nz-1))
        dgdn[:self.nr-1] = -(psi[1:,1]+psi[:-1,1])/(2*self.dz)
        dgdn[self.nr-1:self.nr+self.nz-2] = \
        -(psi[-2,1:]+psi[-2,:-1])/(2*self.dr)
        dgdn[self.nr+self.nz-2:2*self.nr+self.nz-3] = \
        -(psi[1:,-2][::-1]+psi[:-1,-2][::-1])/(2*self.dz)
        dgdn[2*self.nr+self.nz-3:2*self.nr+2*self.nz-4] = \
        -(psi[1,1:][::-1]+psi[1,:-1][::-1])/(2*self.dr)
        return dgdn
        
    def plotb(self,trim=True,alpha=0.5):
        b = self.b.reshape((self.nr,self.nz)).T
        if trim:  # trim edge
            b = b[1:-1,1:-1]
            rlim,zlim = [self.r[1],self.r[-2]],[self.z[1],self.z[-2]]
        else:
            rlim,zlim = [self.r[0],self.r[-1]],[self.z[0],self.z[-1]]
        cmap = pl.get_cmap('Purples_r')
        cmap = pl.get_cmap('RdBu')
        #cmap._init() # create the _lut array, with rgba values
        #cmap._lut[-4,-1] = 1.0  # set zero value clear
        #cmap._lut[0,:-1] = 1  # set zero value clear

        norm = MidpointNormalize(midpoint=0)
        pl.imshow(b, cmap=cmap, norm=norm,#vmin=b.min(),vmax=b.max()
                  extent=[rlim[0],rlim[-1],zlim[0],zlim[-1]],
                  interpolation='nearest', origin='lower',alpha=alpha) 
        c = pl.colorbar(orientation='horizontal')
        c.set_xlabel(r'$j$ MAm$^{-1}$')
        
    def plotj(self,sigma=0):
        #self.GSoper()
        j = self.getj(sigma=sigma)
        self.plot_matrix(j,scale=1e-6,trim=False)
        
        
    def plot_matrix(self,m,midpoint=0,scale=1,trim=True):
        if trim:  # trim edge
            m = m[1:-1,1:-1]
            rlim,zlim = [self.r[1],self.r[-2]],[self.z[1],self.z[-2]]
        else:
            rlim,zlim = [self.r[0],self.r[-1]],[self.z[0],self.z[-1]]
        cmap = pl.get_cmap('RdBu_r')
        caxis = np.round([scale*m.min(),scale*m.max()],decimals=3)
        norm = MidpointNormalize(midpoint=midpoint,vmin=caxis[0],vmax=caxis[1])
        pl.imshow(scale*m.T,cmap=cmap,norm=norm,
                  extent=[rlim[0],rlim[-1],zlim[0],zlim[-1]],
                  interpolation='nearest',origin='lower',alpha=1) 
        c = pl.colorbar(orientation='horizontal',shrink=.6, pad=0.025, aspect=15)
        c.ax.set_xlabel(r'$J$ MAm$^{-2}$')
        c.set_ticks([caxis[0],0,caxis[1]])

          
    def plot(self,levels=[]):
        if not list(levels):
            Nstd = 2.5
            level,n = [np.mean(self.psi)-Nstd*np.std(self.psi), 
                     np.mean(self.psi)+Nstd*np.std(self.psi)],15
            levels = np.linspace(level[0],level[1],n)
        self.cs = pl.contourf(self.r2d,self.z2d,self.psi,
                    levels=levels,cmap=pl.cm.RdBu)
        pl.colorbar()
        cs = pl.contour(self.r2d,self.z2d,self.psi,levels=self.cs.levels)
        pl.clabel(cs, inline=1, fontsize=10)
  
    def set_sf_psi(self):
        self.psi = np.zeros(np.shape(self.r2d))
        for i in range(self.nr):
            for j in range(self.nz):
                self.psi[i,j] = self.sf.Pcoil((self.r[i],self.z[j]))
 
    def add_Pcoil(self,r,z,coil):
        rc,zc,I = coil['r'],coil['z'],coil['I']
        return self.mu_o*I*cc.green(r,z,rc,zc)
        
    def add_Bcoil(self,r,z,coil):
        rc,zc,I = coil['r'],coil['z'],coil['I']
        return self.mu_o*I*cc.green_feild(r,z,rc,zc)
        
    def get_coil_psi(self):
        self.psi = np.zeros(np.shape(self.r2d))
        for name in self.coil.keys():
            self.psi += self.add_Pcoil(self.r2d,self.z2d,self.coil[name])
        for name in self.plasma_coil.keys():
            self.psi += self.add_Pcoil(self.r2d,self.z2d,
                                       self.plasma_coil[name])
        return {'r':self.r,'z':self.z,'psi':self.psi}
        
    def set_coil_psi(self):
        eq = self.get_coil_psi()
        self.sf.set_plasma(eq)
        self.get_Xpsi()  # high-res Xpsi

    def update_coil_psi(self):
        eq = self.get_coil_psi()
        self.sf.update_plasma(eq)
        self.get_Xpsi()  # high-res Xpsi

    def set_plasma_psi(self,plot=False,**kwargs):
        self.psi = np.zeros(np.shape(self.r2d))
        for name in self.plasma_coil.keys():
            self.psi += self.add_coil_psi(self.r2d,self.z2d,self.plasma_coil[name])
        self.sf.update_plasma({'r':self.r,'z':self.z,'psi':self.psi})
            
    def plot_psi(self,**kwargs):
        if 'levels' in kwargs:
            CS = pl.contour(self.r2d,self.z2d,self.psi,
                            levels=kwargs['levels'],colors=[[0.5,0.5,0.5]])
        elif hasattr(self,'cs'):
            levels = self.cs.levels
            CS = pl.contour(self.r2d,self.z2d,self.psi,levels=levels)
        else:
            CS = pl.contour(self.r2d,self.z2d,self.psi,31,
                            colors=[[0.5,0.5,0.5]])
        for cs in CS.collections:
            cs.set_linestyle('solid')
            cs.set_alpha(0.5)
            
    def Pcoil(self,point):
        psi = 0
        for name in self.coil.keys():
            psi += self.add_Pcoil(point[0],point[1],self.coil[name])
        for name in self.plasma_coil.keys():
            psi += self.add_Pcoil(point[0],point[1],self.plasma_coil[name])
        return psi
        
    def Bfeild(self,point):
        feild = np.zeros(2)
        for name in self.coil.keys():
            feild += self.add_Bcoil(point[0],point[1],self.coil[name])
        for name in self.plasma_coil.keys():
            feild += self.add_Bcoil(point[0],point[1],self.plasma_coil[name])
        return feild
        
    def Bmag(self,point):
        feild = self.Bfeild(point)
        B = np.sqrt(feild[0]**2+feild[1]**2)
        return B
        
    def get_Xpsi(self):
        self.sf.Xpoint = minimize(self.Bmag,np.array(self.sf.Xpoint),
                                  method='nelder-mead',
                                  options={'xtol':1e-4,'disp':False}).x
        self.sf.Xpsi = self.Pcoil(self.sf.Xpoint)

    def get_Mpsi(self):
        self.sf.Mpoint = minimize(self.Bmag,np.array(self.sf.Mpoint),
                                  method='nelder-mead',
                                  options={'xtol':1e-4,'disp':False}).x
        self.sf.Mpsi = self.Pcoil(self.sf.Mpoint)
        