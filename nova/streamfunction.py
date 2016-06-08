import numpy as np
import pylab as pl
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import UnivariateSpline as spline
from scipy.interpolate import InterpolatedUnivariateSpline as sinterp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from nova import geqdsk
import seaborn as sns
from matplotlib._cntr import Cntr as cntr
from collections import OrderedDict

class SF(object):
    
    def __init__(self,filename,upsample=1,**kwargs):
        self.filename = filename
        self.set_kwargs(kwargs)
        eqdsk = geqdsk.read('../eqdsk/'+self.filename)
        self.eqdsk = eqdsk
        self.normalise()  # unit normalisation
        self.set_plasma(eqdsk,contour=False)
        self.set_boundary(eqdsk['rbdry'],eqdsk['zbdry'])
        #self.set_coils(eqdsk)
        self.set_flux(eqdsk)  # calculate flux profiles
        self.set_TF(eqdsk)  
        self.set_current(eqdsk)
        xo_arg = np.argmin(self.zbdry)
        self.xo = [self.rbdry[xo_arg],self.zbdry[xo_arg]]
        self.mo = [eqdsk['rmagx'],eqdsk['zmagx']]
        self.upsample(upsample)
        self.get_Xpsi()
        self.get_Mpsi()
        self.set_contour()  # set cfeild
        self.get_sol_psi(dSOL=3e-3,Nsol=15,verbose=False)
        self.rcirc = 0.2*(self.Mpoint[1]-self.Xpoint[1])  # leg search radius
        self.drcirc = 0.1*self.rcirc  # leg search width
        self.xlim,self.ylim,self.nlim = eqdsk['xlim'],eqdsk['ylim'],eqdsk['nlim']

    def set_kwargs(self,kwargs):
        for key in kwargs:
            setattr(self,key,kwargs[key])
            
    def read_config(self,conf):
        self.Nsol = conf.Nsol
        self.dSOL = conf.dSOL
        self.filename = conf.filename
        self.dataname = conf.dataname
              
    def normalise(self):
        if 'Fiesta' in self.eqdsk['name'] or 'Nova' in self.eqdsk['name']\
        or 'disr' in self.eqdsk['name']:
            self.norm = 1
        else:  # CREATE
            self.eqdsk['cpasma'] *= -1
            self.norm = 2*np.pi
            for key in ['psi','simagx','sibdry']:
                self.eqdsk[key] /= self.norm  # per radian
            for key in ['ffprim','pprime']:
                self.eqdsk[key] *= self.norm  # per Webber/radian  
        self.b_scale = 1  # flux function scaling

    def trim_r(self,rmin=1.5):
        if self.r[0] == 0:  # trim zero radius entries
            i = np.argmin(abs(self.r-rmin))
            self.r = self.r[i:]
            self.psi = self.psi[i:,:]
     
    def eqwrite(self,prefix='Nova',config=''):
        if len(config) > 0: 
            name = prefix+'_'+config
        else:
            name = prefix
        nc,rc,zc,drc,dzc,Ic = self.unpack_coils()[:-1]
        psi_ff = np.linspace(0,1,self.nr)
        pad = np.zeros(self.nr)
        eq = {'name':name,
              'nx':self.nr, 'ny':self.nz,  # Number of horizontal and vertical points
              'r':self.r, 'z':self.z,  # Location of the grid-points
              'rdim':self.r[-1]-self.r[0],  # Size of the domain in meters
              'zdim':self.z[-1]-self.z[0],  # Size of the domain in meters
              'rcentr':self.eqdsk['rcentr'],  # Reference vacuum toroidal field (m, T)
              'bcentr':self.eqdsk['bcentr'],  # Reference vacuum toroidal field (m, T)
              'rgrid1':self.r[0],  # R of left side of domain
              'zmid':self.z[0]+(self.z[-1]-self.z[0])/2,  # Z at the middle of the domain
              'rmagx':self.Mpoint[0],  # Location of magnetic axis
              'zmagx':self.Mpoint[1],  # Location of magnetic axis
              'simagx':self.Mpsi,  # Poloidal flux at the axis (Weber / rad)
              'sibdry':self.Xpsi,  # Poloidal flux at plasma boundary (Weber / rad)
              'cpasma':self.eqdsk['cpasma'], 
              'psi':np.transpose(self.psi).reshape((-1,)),  # Poloidal flux in Weber/rad on grid points
              'fpol':self.Fpsi(psi_ff),  # Poloidal current function on uniform flux grid
              'ffprim':self.b_scale*self.FFprime(psi_ff),  # "FF'(psi) in (mT)^2/(Weber/rad) on uniform flux grid"
              'pprime':self.b_scale*self.Pprime(psi_ff),  # "P'(psi) in (N/m2)/(Weber/rad) on uniform flux grid"
              'pressure':pad,  # Plasma pressure in N/m^2 on uniform flux grid
              'qpsi':pad,  # q values on uniform flux grid
              'nbdry':self.nbdry,'rbdry':self.rbdry, 'zbdry':self.zbdry, # Plasma boundary
              'nlim':self.nlim,'xlim':self.xlim,'ylim':self.ylim,  # first wall
              'ncoil':nc,'rc':rc,'zc':zc,'drc':drc,'dzc':dzc,'Ic':Ic} # coils
        print('writing eqdsk','./plot_data/'+config+'.eqdsk')
        geqdsk.write('./plot_data/'+config+'.eqdsk',eq)
        
    def write_flux(self):
        psi_norm = np.linspace(0,1,self.nr)
        pprime = self.b_scale*self.Pprime(psi_norm)
        FFprime = self.b_scale*self.FFprime(psi_norm)
        with open('../Data/'+self.dataname+'_flux.txt','w') as f:
            f.write('psi_norm\tp\' [Pa/(Weber/rad)]\tFF\' [(mT)^2/(Weber/rad)]\n')
            for psi,p_,FF_ in zip(psi_norm,pprime,FFprime):
                f.write('{:1.4f}\t\t{:1.4f}\t\t{:1.4f}\n'.format(psi,p_,FF_))

    def set_flux(self,eqdsk):
        F_ff = eqdsk['fpol']
        P_ff = eqdsk['pressure']
        n = len(F_ff)
        psi_ff = np.linspace(0,1,n)
        #F_ff = spline(psi_ff,F_ff,s=1e-7)(psi_ff)
        #P_ff = spline(psi_ff,P_ff,s=1e6)(psi_ff)
        
        F_ff = interp1d(psi_ff,F_ff)(psi_ff)
        P_ff = interp1d(psi_ff,P_ff)(psi_ff)
        dF_ff = np.gradient(F_ff,1/(n-1))
        dP_ff = np.gradient(P_ff,1/(n-1))
        self.Fpsi = interp1d(psi_ff,F_ff)
        self.dFpsi = interp1d(psi_ff,dP_ff)
        self.dPpsi = interp1d(psi_ff,dP_ff)
        FFp = spline(psi_ff,eqdsk['ffprim'],s=1e-5)(psi_ff)
        Pp = spline(psi_ff,eqdsk['pprime'],s=1e2)(psi_ff) # s=1e5
        
        #self.FFprime = interp1d(psi_ff,F_ff*dF_ff,fill_value=0,bounds_error=False)
        #self.Pprime = interp1d(psi_ff,dP_ff,fill_value=0,bounds_error=False)
        
        self.FFprime = interp1d(psi_ff,FFp,fill_value=0,bounds_error=False)
        self.Pprime = interp1d(psi_ff,Pp,fill_value=0,bounds_error=False)

    def set_TF(self,eqdsk):
        for key in ['rcentr','bcentr']: 
            setattr(self,key,eqdsk[key])
  
    def length(self,R,Z,norm=True):
        L = np.append(0,np.cumsum(np.sqrt(np.diff(R)**2+np.diff(Z)**2)))
        if norm: L = L/L[-1]
        return L
        
    def set_boundary(self,r,z,n=0): 
        L = self.length(r,z)
        if n==0:
            self.nbdry = 5e2
        else:
            self.nbdry = n
        Lsp = np.linspace(L[0],L[-1],self.nbdry)
        self.rbdry = spline(L,r,s=1e-3)(Lsp)
        self.zbdry = spline(L,z,s=1e-3)(Lsp)
        #self.rbdry[0] = self.rbdry[-1]
        #self.zbdry[0] = self.zbdry[-1]
         
    def set_current(self,eqdsk):
        for key in ['cpasma']: 
            setattr(self,key,eqdsk[key])
            
    def update_plasma(self,eq):  # update requres full separatrix
        for attr in ['Bspline','Pspline','Xpsi','Mpsi','Br','Bz']:
            if hasattr(self, attr):
                delattr(self,attr)
        self.set_plasma(eq)
        self.get_Xpsi()
        self.get_Mpsi()
        self.set_contour()  # calculate cfeild
        r,z = self.get_boundary()
        self.set_boundary(r,z)
        #self.get_sol_psi()  # re-calculate sol_psi
        
    def set_plasma(self,eq,contour=False):
        for key in ['r','z','psi']: 
            if key in eq.keys():
                setattr(self,key,eq[key])
        self.trim_r()
        self.space()       
        self.Bfeild()
        if contour:
            self.get_Xpsi()
            self.set_contour()  # calculate cfeild
     
    def upsample(self,sample):
        if sample>1:
            '''
            EQ(self,n=sample*self.n)
            self.space()
            '''
            from scipy.interpolate import RectBivariateSpline as rbs
            sample = np.int(np.float(sample))
            interp_psi = rbs(self.r,self.z,self.psi)
            self.nr,self.nz = sample*self.nr,sample*self.nz
            self.r = np.linspace(self.r[0],self.r[-1],self.nr)
            self.z = np.linspace(self.z[0],self.z[-1],self.nz)
            self.psi = interp_psi(self.r,self.z,dx=0,dy=0)
            self.space()
        
    def space(self):
        self.nr = len(self.r)
        self.nz = len(self.z)
        self.n = self.nr*self.nz
        self.dr = (self.r[-1]-self.r[0])/(self.nr-1)  
        self.dz = (self.z[-1]-self.z[0])/(self.nz-1)
        self.r2d,self.z2d = np.meshgrid(self.r,self.z,indexing='ij')
        
    def Bfeild(self):
        psi_r,psi_z = np.gradient(self.psi,self.dr,self.dz)
        rm = np.array(np.matrix(self.r).T*np.ones([1,self.nz]))
        rm[rm==0] = 1e-34
        self.Br = -psi_z/rm
        self.Bz = psi_r/rm
        
    def Bcoil(self,point):  # magnetic feild at point
        feild = np.zeros(2)
        if not hasattr(self, 'Bspline'):
            self.Bspline = [[],[]]
            self.Bspline[0] = RectBivariateSpline(self.r,self.z,self.Br)
            self.Bspline[1] = RectBivariateSpline(self.r,self.z,self.Bz)
        for i in range(2):
            feild[i] = self.Bspline[i](point[0],point[1])[0]
        return feild
        
    def minimum_feild(self,radius,theta):
        R = radius*np.sin(theta)+self.Xpoint[0]
        Z = radius*np.cos(theta)+self.Xpoint[1]
        B = np.zeros(len(R))
        for i,(r,z) in enumerate(zip(R,Z)):
            feild = self.Bcoil((r,z))
            B[i] = np.sqrt(feild[0]**2+feild[1]**2)
        return np.argmin(B)
    
    def Pcoil(self,point):
        if not hasattr(self, 'Pspline'):
            self.Pspline = RectBivariateSpline(self.r,self.z,self.psi)    
        psi = self.Pspline.ev(point[0],point[1])
        return psi
        
    def contour(self,Nstd=1.5,Nlevel=31,Xnorm=True,lw=1,**kwargs):
        alpha,lw = np.array([1,0.5]),lw*np.array([2.25,1.75])   
        if not hasattr(self,'Xpsi'):
            self.get_Xpsi()
        if not hasattr(self,'Mpsi'):
            self.get_Mpsi()
        if 'levels' not in kwargs.keys():
            dpsi = 0.01*(self.Xpsi-self.Mpsi)
            level,n = [self.Mpsi+dpsi,self.Xpsi-dpsi],17
            level,n = [np.mean(self.psi)-Nstd*np.std(self.psi), 
                       np.mean(self.psi)+Nstd*np.std(self.psi)],15
            level,n = [-Nstd*np.std(self.psi), 
                       Nstd*np.std(self.psi)],Nlevel
            if Nstd*np.std(self.psi) < self.Mpsi-self.Xpsi and \
            self.z.max() > self.Mpoint[1]: 
                Nstd = (self.Mpsi-self.Xpsi)/np.std(self.psi)
                level,n = [-Nstd*np.std(self.psi), 
                           Nstd*np.std(self.psi)],Nlevel
            levels = np.linspace(level[0],level[1],n)
            linetype = '-'
        else:
            levels = kwargs['levels']
            linetype = '-'
        if 'color' in kwargs.keys():
            color = kwargs['color']
        else:
            color = 'k'
        if 'linetype' in kwargs.keys():
            linetype = kwargs['linetype']
        if color == 'k': alpha *= 0.25
        if Xnorm: levels = levels+self.Xpsi
        contours = self.get_contour(levels)
        for psi_line in contours:
            for line in psi_line:
                r,z = line[:,0],line[:,1]
                pindex=0 if self.inPlasma(r,z) else 1
                pl.plot(r,z,linetype,linewidth=lw[pindex],
                        color=color,alpha=alpha[pindex])
        pl.axis('equal')
        pl.axis('off')
        return levels
        
    def inPlasma(self,R,Z,delta=0.2):
        return R.min()>=self.rbdry.min()-delta and \
        R.max()<=self.rbdry.max()+delta and \
        Z.min()>=self.zbdry.min()-delta and \
        Z.max()<=self.zbdry.max()+delta
            
    def plot_cs(self,cs,norm,Plasma=False,color='k',
                pcolor='w',linetype='-'):
        alpha = np.array([1,0.5])
        lw = 0.75
        if not Plasma: norm = 0
        if color == 'k': alpha *= 0.25
        
        for p in cs.get_paths():
            v = p.vertices
            R,Z,delta = v[:,0][:],v[:,1][:],0.5
            inPlasma = R.min()>=self.rbdry.min()-delta and \
            R.max()<=self.rbdry.max()+delta and \
            Z.min()>=self.zbdry.min()-delta and \
            Z.max()<=self.zbdry.max()+delta
            if inPlasma:
                pl.plot(R,Z,linetype,linewidth=1.25*lw,
                        color=norm*np.array([1,1,1]),alpha=alpha[0])  
            else:
                pl.plot(R,Z,linetype,linewidth=lw,color=color,alpha=alpha[1])   
    
    def Bcontour(self, axis, Nstd=1.5, color='r'):
        var = 'B'+axis
        if not hasattr(self, var):
            self.Bfeild()  
        B = getattr(self, var)
        level = [np.mean(B)-Nstd*np.std(B), 
                 np.mean(B)+Nstd*np.std(B)]
        CS = pl.contour(self.r, self.z, B, 
                   levels=np.linspace(level[0],level[1],30), colors=color) 
        for cs in CS.collections:
            cs.set_linestyle('solid')
            
    def Bquiver(self):
        if not hasattr(self, 'Br'):
            self.Bfeild()  
        pl.quiver(self.r,self.z,self.Br.T,self.Bz.T) 
                       
    def Bsf(self):
        if not hasattr(self, 'Br'):
            self.Bfeild() 
        pl.streamplot(self.r,self.z,self.Br.T,self.Bz.T,
                             color=self.Br.T,cmap=pl.cm.RdBu)
        pl.clim([-1.5,1.5])
        #pl.colorbar(strm.lines)
        
    def getX(self,xo=None):
        def feild(x):
            B = self.Bcoil(x)
            return sum(B*B)**0.5
        res = minimize(feild, np.array(xo), method='nelder-mead', 
                       options={'xtol': 1e-7, 'disp': False})  
        return res.x
            
    def get_Xpsi(self,xo=None):
        if xo is None:
            if hasattr(self,'xo'):
                xo = self.xo
            else:
                xo_arg = np.argmin(self.eqdsk['zbdry'])
                xo = [self.eqdsk['rbdry'][xo_arg],self.eqdsk['zbdry'][xo_arg]]
        Xpoint = np.zeros((2,2))
        Xpsi = np.zeros(2)
        for i,flip in enumerate([1,-1]):
            xo[1] *= flip
            Xpoint[:,i] = self.getX(xo=xo)
            Xpsi[i] = self.Pcoil(Xpoint[:,i])
        i = np.argmax(Xpsi)
        self.Xpsi = Xpsi[i]
        self.Xpoint = Xpoint[:,i]
        if i == 0: xo[1] *= -1  # re-flip
        return (self.Xpsi,self.Xpoint)
   
    def getM(self,mo=None):
        if mo is None:
            mo = self.mo
        def psi(m):
            return -self.Pcoil(m)
        res = minimize(psi, np.array(mo), method='nelder-mead', 
                       options={'xtol': 1e-7, 'disp': False})  
        return res.x
        
    def get_Mpsi(self, mo=None):
        self.Mpoint = self.getM(mo=mo)
        self.Mpsi = self.Pcoil(self.Mpoint)
        return (self.Mpsi,self.Mpoint)
        
    def remove_contour(self):
        for key in ['cfeild','cfeild_bndry']:
            if hasattr(self,key):
                delattr(self,key)
        
    def set_contour(self):
        psi_boundary = 1.1*(self.Xpsi-self.Mpsi)+self.Mpsi
        try:
            psi_bndry = np.pad(self.psi[:,1:-1],((0,0),(1,1)),
                       mode='constant',constant_values=psi_boundary)
        except:
            psi_boundary = ((psi_boundary,psi_boundary),
                            (psi_boundary,psi_boundary))
            psi_bndry = np.pad(self.psi[:,1:-1],((0,0),(1,1)),
                       mode='constant',constant_values=psi_boundary)
        self.cfeild = cntr(self.r2d,self.z2d,self.psi)
        self.cfeild_bndry = cntr(self.r2d,self.z2d,psi_bndry)
        
    def get_contour(self,levels,boundary=False):
        if boundary:
            cfeild = lambda level: self.cfeild_bndry.trace(level,level,0)
        else:
            cfeild = lambda level: self.cfeild.trace(level,level,0)
        lines = []
        for level in levels:
            psi_line = cfeild(level)
            psi_line = psi_line[:len(psi_line)//2]
            lines.append(psi_line)
        return lines
        
    def get_boundary(self,alpha=1,delta_loop=1):
        self.Spsi = alpha*(self.Xpsi-self.Mpsi)+self.Mpsi
        psi_line = self.get_contour([self.Spsi],boundary=True)[0]
        R,Z = np.array([]),np.array([])
        for line in psi_line:
            r,z = line[:,0],line[:,1]
            index = z >= self.Xpoint[1]
            if sum(index) > 0:
                r,z = r[index],z[index]
                loop = np.sqrt((r[0]-r[-1])**2+(z[0]-z[-1])**2) < delta_loop
                if (z>self.Mpoint[1]).any() and (z<self.Mpoint[1]).any() and loop:
                    R,Z = np.append(R,r),np.append(Z,z)    
        R,Z = self.clock(R,Z)
        return R,Z
        
    def clock(self,R,Z,anti=False):
        rc,zc = self.Mpoint  
        radius = ((R-rc)**2+(Z-zc)**2)**0.5
        theta = np.arctan2(Z-zc, R-rc)
        index = theta.argsort()
        index = index[::-1]
        radius,theta = radius[index],theta[index] 
        R,Z = rc+radius*np.cos(theta),zc+radius*np.sin(theta)
        R,Z = np.append(R,R[0]),np.append(Z,Z[0])
        if anti:
            R,Z = R[::-1],Z[::-1]
        return R,Z
        
    def get_midplane(self,r,z):
        def psi_err(r,*args):
            z = args[0]
            psi = self.Pcoil((r,z))
            return abs(psi-self.Xpsi)
        res = minimize(psi_err,np.array(r),method='nelder-mead', 
                       args=(z),options={'xtol': 1e-7,'disp': False})  
        return res.x[0]
        
    def get_LFP(self,xo=None,alpha=1-1e-3):
        r,z = self.get_boundary(alpha=alpha)
        if self.Xpoint[1] < self.Mpoint[1]:
            index = z>self.Xpoint[1]
        else:  # alowance for upper Xpoint
            index = z<self.Xpoint[1]
        r_loop,z_loop = (r[index], z[index])
        rc,zc = self.Mpoint  
        radius = ((r_loop-rc)**2+(z_loop-zc)**2)**0.5
        theta = np.arctan2(z_loop-zc, r_loop-rc)
        index = theta.argsort()
        radius,theta = radius[index],theta[index] 
        theta = np.append(theta[-1]-2*np.pi, theta)
        radius = np.append(radius[-1], radius)
        r = rc+radius*np.cos(theta)
        z = zc+radius*np.sin(theta)
        fLFSr = interp1d(theta,r) 
        fLFSz = interp1d(theta,z)
        self.LFPr,self.LFPz = fLFSr(0),fLFSz(0)
        self.LFPr = self.get_midplane(self.LFPr,self.LFPz)
        self.HFPr,self.HFPz = fLFSr(-np.pi),fLFSz(-np.pi)
        self.HFPr = self.get_midplane(self.HFPr,self.HFPz)
        return (self.LFPr,self.LFPz,self.HFPr,self.HFPz)
        
    def get_sol_psi(self,verbose=True,**kwargs):
        for var in ['dSOL','Nsol']:
            if var in kwargs:
                setattr(self,var,kwargs[var])
        if verbose:
            print('calculating sol psi',self.Nsol,self.dSOL) 
        self.get_LFP()
        self.Dsol = np.linspace(0,self.dSOL,self.Nsol)
        r = self.LFPr+self.Dsol
        z = self.LFPz*np.ones(len(r))
        self.sol_psi = np.zeros(len(r))
        for i,(rp,zp) in enumerate(zip(r,z)):
            self.sol_psi[i] = self.Pcoil([rp,zp])
    
    def upsample_sol(self,nmult=10):
        k = 1  # smoothing factor
        for i,(r,z) in enumerate(zip(self.Rsol,self.Zsol)):
            l = self.length(r,z)
            L = np.linspace(0,1,nmult*len(l))
            self.Rsol[i] = sinterp(l,r,k=k)(L)
            self.Zsol[i] = sinterp(l,z,k=k)(L)
        
        
    def sol(self,dr=0,Nsol=0,plot=False,update=False,debug=False):  # dr [m]
        if update or not hasattr(self,'sol_psi') or dr > self.dSOL\
        or Nsol > self.Nsol: 
            self.get_sol_psi(dr=dr,Nsol=Nsol)  # re-calculcate LFP
        elif (Nsol>0 and Nsol != self.Nsol) or \
        (dr>0 and dr != self.dSOL):  # update  
            if dr>0: self.dSOL = dr
            if Nsol>0: self.Nsol = Nsol
            Dsol = np.linspace(0,self.dSOL,self.Nsol)
            self.sol_psi = interp1d(self.Dsol,self.sol_psi)(Dsol)
            self.Dsol = Dsol
        contours = self.get_contour(self.sol_psi)    
        self.Rsol,self.Zsol = self.pick_contour(contours,Xpoint=True,
                                                Midplane=False,Plasma=False)
        self.upsample_sol(nmult=4)  # upsamle
        self.get_legs(debug=debug)
        if plot:
            color = sns.color_palette('Set2',6)
            for c,leg in enumerate(self.legs):
                for i in np.arange(self.legs[leg]['i'])[::-1]:
                    r,z = self.snip(leg,i)
                    r,z = self.legs[leg]['R'][i],self.legs[leg]['Z'][i]
                    pl.plot(r,z,color=color[c],linewidth=0.5)
                    
                    
    def add_core(self):  # refarance from low-feild midplane
        for i in range(self.Nsol):
            for leg in ['inner','inner1','inner2','outer','outer1','outer2']:
                if leg in self.legs:
                    if 'inner' in leg:
                        core = 'core1'
                    else:
                        core = 'core2'
                    Rc = self.legs[core]['R'][i][:-1]
                    Zc = self.legs[core]['Z'][i][:-1]
                    self.legs[leg]['R'][i] = np.append(Rc,self.legs[leg]['R'][i])
                    self.legs[leg]['Z'][i] = np.append(Zc,self.legs[leg]['Z'][i])
             
    def orientate(self,R,Z):
        if R[-1] > R[0]:  # counter clockwise
            R=R[::-1]
            Z=Z[::-1]
        return R,Z
  
    def pick_contour(self,contours,Xpoint=False,Midplane=True,Plasma=False):
        Rs = []
        Zs = []
        Xp,Mid,Pl = True,True,True
        for psi_line in contours:
            for line in psi_line:
                R,Z = line[:,0],line[:,1]
                if Xpoint:  # check Xpoint proximity
                    rX = np.sqrt((R-self.Xpoint[0])**2+(Z-self.Xpoint[1])**2)
                    if (min(rX) < self.rcirc):
                        Xp = True
                    else:
                        Xp = False
                if Midplane:  # check lf midplane crossing
                    if (np.max(Z) > self.LFPz) and (np.min(Z) < self.LFPz):  
                        Mid = True
                    else:
                        Mid = False   
                if Plasma:
                    if (np.max(R) < np.max(self.rbdry)) and\
                    (np.min(R) > np.min(self.rbdry)) and\
                    (np.max(Z) < np.max(self.zbdry)) and\
                    (np.min(Z) > np.min(self.zbdry)):
                        Pl = True
                    else:
                        Pl = False          
                if Xp and Mid and Pl:
                    R,Z = self.orientate(R,Z)
                    Rs.append(R)
                    Zs.append(Z)
        return Rs,Zs
                    
    def topolar(self,R,Z):
        x,y = R,Z
        r = np.sqrt((x-self.Xpoint[0])**2+(y-self.Xpoint[1])**2)
        t = np.arctan2(x-self.Xpoint[0],y-self.Xpoint[1])
        return r,t
        
    def store_leg(self,rloop,tloop):
        if np.argmin(rloop) > len(rloop)/2:  # point legs out
            rloop,tloop = rloop[::-1],tloop[::-1]
        ncirc = np.argmin(abs(rloop-self.rcirc))
        tID = np.argmin(abs(tloop[ncirc]-self.tleg))
        legID = self.tID[tID]
        if self.nleg == 6:
            if legID <= 1:
                label = 'inner'+str(legID+1)
            elif legID >= 4:
                label = 'outer'+str(legID-3)
            elif legID == 2:
                label = 'core1'
            elif legID == 3:
                label = 'core2'
            else:
                label = ''
        else:
            if legID == 0:
                label = 'inner'
            elif legID == 3:
                label = 'outer'
            elif legID == 1:
                label = 'core1'
            elif legID == 2:
                label = 'core2'
            else:
                label = ''
        if label:
            i = self.legs[label]['i']
            R = rloop*np.sin(tloop)+self.Xpoint[0]
            Z = rloop*np.cos(tloop)+self.Xpoint[1] 
            if i > 0:
                if R[0]**2+Z[0]**2 == (self.legs[label]['R'][i-1][0]**2+
                                       self.legs[label]['Z'][i-1][0]**2):
                    i -= 1
            if 'core' in label:
                R,Z = R[::-1],Z[::-1]
            self.legs[label]['R'][i] = R
            self.legs[label]['Z'][i] = Z
            self.legs[label]['i'] = i+1

    def min_L2D(self,targets):
        L2D = np.zeros(len(targets.keys()))
        for i,target in enumerate(targets.keys()):
            L2D[i] = targets[target]['L2D'][0]
        return L2D.min()
        
    def check_legs(self):
        if self.sf.z.min() > self.sf.Xpoint[1]-self.sf.rcirc:
            print('grid out of bounds')

    
    def get_legs(self,debug=False):
        if debug:
            theta = np.linspace(-np.pi,np.pi,100)
            r = (self.rcirc-self.drcirc/2)*np.cos(theta)
            z = (self.rcirc-self.drcirc/2)*np.sin(theta)
            pl.plot(r+self.Xpoint[0],z+self.Xpoint[1],'k--',alpha=0.5)
            r = (self.rcirc+self.drcirc/2)*np.cos(theta)
            z = (self.rcirc+self.drcirc/2)*np.sin(theta)
            pl.plot(r+self.Xpoint[0],z+self.Xpoint[1],'k--',alpha=0.5)

        self.tleg = np.array([])
        for N in range(len(self.Rsol)):
            r,t = self.topolar(self.Rsol[N],self.Zsol[N])
            index = (r>self.rcirc-self.drcirc/2) & (r<self.rcirc+self.drcirc/2)
            self.tleg = np.append(self.tleg,t[index])
        nbin = 50
        nhist,bins = np.histogram(self.tleg,bins=nbin)
        flag,self.nleg,self.tleg = 0,0,np.array([])
        for i in range(len(nhist)):
            if nhist[i] > 0:
                if flag == 0:
                    tstart = bins[i]
                    tend = bins[i]
                    flag = 1
                if flag == 1:
                    tend = bins[i]
            elif flag == 1:
                self.tleg = np.append(self.tleg,(tstart+tend)/2)
                self.nleg += 1
                flag = 0
            else:
                flag = 0
        if nhist[-1] > 0:
            tend = bins[-1]
            self.tleg = np.append(self.tleg,(tstart+tend)/2)
            self.nleg += 1

        if self.nleg == 6:  # snow flake
            self.legs = {\
            'inner1':{'R':[[] for i in range(self.Nsol)],
            'Z':[[] for i in range(self.Nsol)],'i':0},
            'inner2':{'R':[[] for i in range(self.Nsol)],
            'Z':[[] for i in range(self.Nsol)],'i':0},
            'outer1':{'R':[[] for i in range(self.Nsol)],
            'Z':[[] for i in range(self.Nsol)],'i':0},
            'outer2':{'R':[[] for i in range(self.Nsol)],
            'Z':[[] for i in range(self.Nsol)],'i':0},
            'core1':{'R':[[] for i in range(self.Nsol)],
            'Z':[[] for i in range(self.Nsol)],'i':0},
            'core2':{'R':[[] for i in range(self.Nsol)],
            'Z':[[] for i in range(self.Nsol)],'i':0}}
        else:
            self.legs = {\
            'inner':{'R':[[] for i in range(self.Nsol)],
            'Z':[[] for i in range(self.Nsol)],'i':0},
            'outer':{'R':[[] for i in range(self.Nsol)],
            'Z':[[] for i in range(self.Nsol)],'i':0},
            'core1':{'R':[[] for i in range(self.Nsol)],
            'Z':[[] for i in range(self.Nsol)],'i':0},
            'core2':{'R':[[] for i in range(self.Nsol)],
            'Z':[[] for i in range(self.Nsol)],'i':0}}
        self.legs = OrderedDict(sorted(self.legs.items(), key=lambda t: t[0]))
        
        self.tID = np.arange(self.nleg)
        self.tID = np.append(self.nleg-1,self.tID)
        self.tID = np.append(self.tID,0)
        self.tleg = np.append(-np.pi-(np.pi-self.tleg[-1]),self.tleg)
        self.tleg = np.append(self.tleg,np.pi+(np.pi+self.tleg[1]))

        for N in range(len(self.Rsol)):  
            ends,ro = [0,-1],np.zeros(2)
            for i in ends:
                ro[i] = np.sqrt(self.Rsol[N][i]**2+self.Zsol[N][i]**2)
            r,t = self.topolar(self.Rsol[N],self.Zsol[N])
            post = False
            rpost,tpost = 0,0
            if ro[0] == ro[-1]:  # cut loops
                if np.min(r*np.cos(t)) > self.drcirc-self.rcirc:
                    nmax = np.argmax(r*np.sin(t))  # LF 
                else:
                    nmax = np.argmin(r*np.cos(t))  # minimum z
                r = np.append(r[nmax:],r[:nmax])
                t = np.append(t[nmax:],t[:nmax])
            while len(r) > 0:
                if r[0] > self.rcirc:
                    if np.min(r) < self.rcirc:
                        ncut = np.arange(len(r))[r<self.rcirc][0]
                        rloop,tloop = r[:ncut],t[:ncut]      
                        loop = False
                    else:
                        ncut = -1
                        rloop,tloop = r,t
                        loop = True
                    if post:
                        rloop,tloop = np.append(rpost,rloop),np.append(tpost,tloop)
                else:
                    ncut = np.arange(len(r))[r>self.rcirc][0]
                    rin,tin = r[:ncut],t[:ncut] 
                    nx = self.minimum_feild(rin,tin)  # minimum feild
                    rpre,tpre = rin[:nx+1],tin[:nx+1]
                    rpost,tpost = rin[nx:],tin[nx:]
                    loop = True
                    post = True
                    rloop,tloop = np.append(rloop,rpre),np.append(tloop,tpre)
                if loop:                     
                    if rloop[0] < self.rcirc and rloop[-1] < self.rcirc:
                        if np.min(rloop*np.cos(tloop)) > self.drcirc-self.rcirc:
                            nmax = np.argmax(rloop*np.sin(tloop))  # LF 
                        else:
                            nmax = np.argmax(rloop)   
                        self.store_leg(rloop[:nmax],tloop[:nmax])
                        self.store_leg(rloop[nmax:],tloop[nmax:])
                    else:
                        self.store_leg(rloop,tloop)  
                if ncut == -1:
                    r,t = [],[]
                else:
                    r,t = r[ncut:],t[ncut:]
                    
    def strike_point(self,Xi,graze):
        ratio = np.sin(graze)*np.sqrt(Xi[-1]**2+1)
        if np.abs(ratio) > 1:
            theta = np.sign(ratio)*np.pi
        else:
            theta = np.arcsin(ratio)
        return theta

    def snip(self,leg,layer_index=0,L2D=0):
        if not hasattr(self,'Rsol'):
            self.sol()
        Rsol = self.legs[leg]['R'][layer_index]
        Zsol = self.legs[leg]['Z'][layer_index] 
        Lsol = self.length(Rsol,Zsol,norm=False)
        if L2D == 0:
            L2D = Lsol[-1]
        if layer_index != 0:
            Rsolo = self.legs[leg]['R'][0]
            Zsolo = self.legs[leg]['Z'][0] 
            Lsolo = self.length(Rsolo,Zsolo,norm=False)
            indexo = np.argmin(np.abs(Lsolo-L2D))
            index = np.argmin((Rsol-Rsolo[indexo])**2+
                          (Zsol-Zsolo[indexo])**2)
            L2D = Lsol[index]
        else:
            index = np.argmin(np.abs(Lsol-L2D))
        if Lsol[index] > L2D: 
            index -= 1
        Rend,Zend = interp1d(Lsol,Rsol)(L2D),interp1d(Lsol,Zsol)(L2D)
        Rsol,Zsol = Rsol[:index],Zsol[:index]  # trim to strike point
        Rsol,Zsol = np.append(Rsol,Rend),np.append(Zsol,Zend)
        return (Rsol,Zsol)

    def pick_leg(self,leg,layer_index):
        R = self.legs[leg]['R'][layer_index]
        Z = self.legs[leg]['Z'][layer_index]
        return R,Z
        
    def Xtrim(self,Rsol,Zsol):
        Xindex = np.argmin((self.Xpoint[0]-Rsol)**2+
                           (self.Xpoint[1]-Zsol)**2)
        if (Rsol[-1]-Rsol[Xindex])**2+(Zsol[-1]-Zsol[Xindex])**2 <\
        (Rsol[0]-Rsol[Xindex])**2+(Zsol[0]-Zsol[Xindex])**2:
            Rsol = Rsol[:Xindex]  # trim to Xpoints
            Zsol = Zsol[:Xindex]
            Rsol = Rsol[::-1]
            Zsol = Zsol[::-1]
        else:
            Rsol = Rsol[Xindex:]  # trim to Xpoints
            Zsol = Zsol[Xindex:]
        return (Rsol,Zsol)
        
    def get_graze(self,point,target):
        T = target / np.sqrt(target[0]**2+target[1]**2)  # target vector
        B = self.Bcoil([point[0],point[1]])
        B /= np.sqrt(B[0]**2+B[1]**2)  # poloidal feild line vector
        theta = np.arccos(np.dot(B,T))
        if theta > np.pi/2: theta = np.pi-theta
        Xi = self.expansion([point[0]],[point[1]])
        graze = np.arcsin(np.sin(theta)*(Xi[-1]**2+1)**-0.5)
        return graze
        
    def get_max_graze(self,r,z):
        theta = np.pi/2  # normal target, maximum grazing angle
        Xi = self.expansion([r],[z])
        graze = np.arcsin(np.sin(theta)*(Xi[-1]**2+1)**-0.5)
        return graze
            
    def expansion(self,Rsol,Zsol):
        Xi = np.array([])
        Bm = np.abs(self.bcentr*self.rcentr)
        for r,z in zip(Rsol,Zsol):
            B = self.Bcoil([r,z])
            Bp = np.sqrt(B[0]**2+B[1]**2)  # polodial feild
            Bphi = Bm/r  # torodal field
            Xi = np.append(Xi,Bphi/Bp)  # feild expansion
        return Xi
        
    def connection(self,leg,layer_index,L2D=0):
        if L2D > 0:  # trim targets to L2D
            Rsol,Zsol = self.snip(leg,layer_index,L2D)  
        else:  # rb.trim_sol to trim to targets
            Rsol = self.legs[leg]['R'][layer_index]
            Zsol = self.legs[leg]['Z'][layer_index] 
        Lsol = self.length(Rsol,Zsol)
        index = np.append(np.diff(Lsol)!=0,True)
        Rsol,Zsol = Rsol[index],Zsol[index]  # remove duplicates
        if len(Rsol) < 2:
            L2D,L3D = [0],[0]
        else:
            dRsol = np.diff(Rsol)
            dZsol = np.diff(Zsol)
            L2D = np.append(0,np.cumsum(np.sqrt(dRsol**2+dZsol**2)))
            dTsol = np.array([])
            Xi = self.expansion(Rsol,Zsol)
            for r,dr,dz,xi in zip(Rsol[1:],dRsol,dZsol,Xi):
                dLp = np.sqrt(dr**2+dz**2)
                dLphi = xi*dLp
                dTsol = np.append(dTsol,dLphi/(r+dr/2))
            L3D = np.append(0,np.cumsum(dTsol*np.sqrt((dRsol/dTsol)**2+
                                        (dZsol/dTsol)**2+(Rsol[:-1])**2)))
        return L2D,L3D,Rsol,Zsol
