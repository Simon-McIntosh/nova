import numpy as np
import shelve
import load_eqlib

class extract_geom(object):
    
    def __init__(self, config):
        self.path = './config_data/'+config+'/'
        self.config = config
        self.Jmax = 2e7  # [A/m^2]
        
        #eqlib = load_eqlib.PEX(self.path,self.config,self.Jmax)
        eqlib = load_eqlib.DTT('ddd',self.Jmax)
        self.coil = eqlib.coil
        self.plasma = eqlib.plasma
        self.Name = eqlib.Name
        self.set_plasma()  # define plasma in radial cordinates
        self.get_TF() 
        self.get_FW() 
        

    def set_current(self, name, I):
        self.coil[name]['I'] = I
        self.coil[name]['rc'] = (abs(self.coil[name]['I'])/(self.Jmax*np.pi))**0.5

    def set_plasma(self):
        # plasma edge (LCFS)
        self.rp = self.plasma['LCFS']['r']
        self.zp = self.plasma['LCFS']['z']
        self.rlim = [min(self.rp),max(self.rp)]
        self.zlim = [min(self.zp),max(self.zp)]
        self.dr = self.rlim[1]-self.rlim[0]
        self.dz = self.zlim[1]-self.zlim[0]
        self.N = len(self.rp)
        self.n = np.zeros((self.N-1,3))
        for i in range(self.N-1):
            r = np.array([self.rp[i+1]-self.rp[i],self.zp[i+1]-self.zp[i],0])
            self.n[i]  = np.cross(r,np.array([0,0,1]))
        
        # plasma centre
        self.rc = self.plasma['rc']
        self.zc = self.plasma['zc']

        # unwrap theta around plasma edge
        self.theta = np.unwrap(np.arctan2(self.zp-self.zc, self.rp-self.rc))
        self.radius = ((self.zp-self.zc)**2 + (self.rp-self.rc)**2)**0.5
        for i,theta in enumerate(self.theta):
            self.theta[i] = (theta+2*np.pi)%(2*np.pi)
        index = self.theta.argsort()
        self.theta = self.theta[index]
        self.radius = self.radius[index]
        self.theta = np.append(self.theta, self.theta[0]+2*np.pi)
        self.theta = np.append(self.theta[-2]-2*np.pi, self.theta)
        self.radius = np.append(self.radius, self.radius[0])
        self.radius = np.append(self.radius[-2], self.radius)
        
    def min_r(self,xo):
        zo = xo[0]
        ro = xo[1]  # major radius
        A = xo[2]  # aspect ratio
        k = xo[3]  # elongation
        tau = xo[4]  # triangularity
        R,Z = self.elipsoid(zo,ro,A,k,tau)
        err = 0
        for r,z in zip(self.plasma['LCFS']['r'],self.plasma['LCFS']['z']):
            err += np.min((R-r)**2+(Z-z)**2)
        return err
    
    def normal(self,R,Z):
        dR,dZ = np.gradient(R),np.gradient(Z)
        mag = np.sqrt(dR**2+dZ**2)
        t = np.zeros((len(R),3))
        t[:,0],t[:,1] = dR/mag,dZ/mag
        n = np.cross(t, [0,0,1])
        nR,nZ = n[:,0],n[:,1]
        return (nR,nZ)
    
    def dot_r(self,xo):
        zo = xo[0]
        ro = xo[1]  # major radius
        A = xo[2]  # aspect ratio
        k = xo[3]  # elongation
        tau = xo[4]  # triangularity
        R,Z = self.elipsoid(zo,ro,A,k,tau)
        nR,nZ = self.normal(R,Z)
        dsum = 0
        for r,z in zip(self.plasma['LCFS']['r'],self.plasma['LCFS']['z']):
            i = np.argmin((r-R)**2+(z-Z)**2)
            dr = [R[i]-r,Z[i]-z]  
            dn = [nR[i],nZ[i]]
            dot = np.dot(dr,dn)
            if dot < 0:
                dsum -= (dr[0]**2+dr[1]**2)
        return dsum
        
    def elipsoid(self,zo,ro,A,k,tau):
        R = ro+ro/A*np.cos(self.theta_elip+tau*np.sin(self.theta_elip))
        Z = zo+ro/A*k*np.sin(self.theta_elip)
        return (R,Z)    
    
    def plasma_coils(self,I,N):
        import numpy as np
        from scipy.interpolate import interp1d as interp1
        import scipy.optimize as op
        from itertools import count
        import pylab as pl
        index = count(0)
        delta = self.dr/N
    
        grid = 'elip'  # 'sq || rad
        
        if grid is 'elip':  # elipsoid
            zo=0
            ro=9.08
            A=3.89
            k=1.69
            tau=0.44
            f = 0.6  # edge current fraction Iedge = fIo
            self.xo = [zo,ro,A,k,tau]
            self.theta_elip = np.linspace(-np.pi,np.pi,100)
            #constraint = {'type':'ineq','fun':self.dot_r}
            par = op.minimize(self.min_r,self.xo,
                              options={'disp':True,'maxiter':500},
                              method='SLSQP',tol=1e-2)
            self.xo = par.x
            zo=self.xo[0]
            ro=self.xo[1]
            A=self.xo[2]
            k=self.xo[3]
            tau=self.xo[4]
            dcoil = 1.5*ro/(N*A)
            rshift = np.linspace(0.45,0,N)
            rmax = ro/A
            rspace = np.linspace(0,rmax,N)

            rp,zp,nc = np.array([]),np.array([]),np.array([])
            for shift,r in zip(rshift,rspace):
                if r  == 0.0:
                    R,Z = ro+shift,zo
                    rp,zp = np.append(rp,R),np.append(zp,Z)
                    nc = np.append(nc,1)
                else:
                    R,Z = self.elipsoid(zo,ro+shift,ro/r,k,tau)
                    L = np.cumsum(np.sqrt(np.gradient(R)**2+np.gradient(Z)**2))
                    Linterp = np.linspace(L[0],L[-1],np.round(L[-1]/dcoil))
                    rp = np.append(rp,interp1(L,R)(Linterp))
                    zp = np.append(zp,interp1(L,Z)(Linterp))
                    nc = np.append(nc,len(interp1(L,R)(Linterp)))
                    pl.plot(R,Z,'b-')

            Ci = (1-(1-f)*(np.arange(N)/(N-1))**2)*nc  # I = Io*sum(Ci) = Io*sum(nci(1-xi/(N-1)))
            Io = I/np.sum(Ci)
            Ip = np.zeros(np.shape(rp))
            ncOld = 0
            for i in range(len(nc)):
                ncSum = sum(nc[0:i+1])
                Ip[ncOld:ncSum] = Io*(1-(1-f)*(i/(N-1))**2)
                ncOld = ncSum

            delta = dcoil
            print(sum(Ip))
            
        elif grid == 'sq':
            nr, nz = (int(self.dr/delta), int(self.dz/delta))
            r,dr = np.linspace(self.rlim[0]-self.dr/2,
                               self.rlim[1]+self.dr/2, nr+1, retstep='true')
            z,dz = np.linspace(self.zlim[0]-self.dz/2,
                               self.zlim[1]+self.dz/2, nz+1, retstep='true')
            rm, zm = np.meshgrid(r, z)
            
            rp,zp = [],[]
            
            for i in range(nz):
                for j in range(nr):
                    if self.check([rm[i,j],zm[i,j]]):
                        rp.append(rm[i,j])
                        zp.append(zm[i,j])
        elif grid == 'rad':
            nr = np.int(np.max(self.radius)/delta)
            r_line = np.linspace(0,np.max(self.radius),nr)
    
            for rL in r_line:
                if rL == 0:
                    rm = np.array([self.rc])
                    zm = np.array([self.zc])
                    radius = np.zeros(1)
                    theta = np.zeros(1)
                else:
                    n_theta = np.int(2*np.pi*rL/delta)
                    thetaL = np.linspace(0,2*np.pi,n_theta,endpoint=False)
                    rm = np.append(rm,self.rc+rL*np.cos(thetaL))
                    zm = np.append(zm,self.zc+rL*np.sin(thetaL))
                    theta = np.append(theta, thetaL)
                    radius = np.append(radius, rL*np.ones(n_theta))
                  
            rp,zp,theta_p,radius_p = [],[],[],[]
            for r,z,rad,th in zip(rm,zm,radius,theta):
                if self.check_plasma([r,z]):
                    rp.append(r)
                    zp.append(z)
                    theta_p.append(th)
                    radius_p.append(rad)
                    
        plasma_name = []
        print('lenrp',len(rp))
        for r,z in zip(rp,zp):
            plasma_name.append('plasma-'+str(next(index)))
            self.coil[plasma_name[-1]] = {}
            self.coil[plasma_name[-1]]['r'] = r
            self.coil[plasma_name[-1]]['z'] = z
            self.coil[plasma_name[-1]]['rc'] = delta 
            
        if grid is 'elip':
            for name,ip in zip(plasma_name,Ip):
                self.coil[name]['I'] = ip
        else:
            self.current_profile(I,rp,radius_p,theta_p,plasma_name)
    
    def current_profile(self,I,rp,radius_p,theta_p,plasma_name):
        from scipy.interpolate import interp1d
        from itertools import count
        itt = count(1)
        Imax = 2*I/len(rp)
        gain = 0.01
        while True:
            Isum = 0
            for r,t,name in zip(radius_p, theta_p, plasma_name):
                fre = interp1d(self.theta, self.radius, kind='cubic')
                re = fre(t)
                a = Imax/re**2
                self.coil[name]['I'] = Imax-a*r**2
                Isum += self.coil[name]['I']   
            err = I-Isum
            Imax = gain*err+Imax
    
            if abs(err) < 1e1: 
                break
            
            if next(itt) > 100:
                print('plasma shape convergence error')
                break
        
    def check_plasma(self, point):
        for i in range(self.N-1):
            p = np.array([point[0]-self.rp[i], point[1]-self.zp[i], 0])
            inside = True
            if np.dot(p,self.n[i]) < 0:
                inside = False
                break
        return inside
        
    def get_TF(self):
        file = 'structure'
        data = shelve.open(self.path+self.config+'_'+file)
        surface = ['TFin','TFout','divertorL', 'divertorU']
        self.structure = {}
        for index,name in zip(range(1,len(data.keys())+1), surface):
            key = 'data-'+str(index)  
            self.structure[name] = {}
            self.structure[name]['r'] = data[key][:,0]
            self.structure[name]['z'] = data[key][:,1]
        data.close()
        
    def get_FW(self):
        data = shelve.open('./config_data/SDX/SDX_first_wall')
        self.fw = {} 
        self.fw['r'] = data['data-1'][:,0]
        self.fw['z'] = data['data-1'][:,1]
        data.close()    
