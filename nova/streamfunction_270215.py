import numpy as np
import pylab as pl

class SF(object):
    
    def __init__(self, file, res=0.4, rlim=[2,15], zlim=[-11,6]):
        if isinstance(file, str):  # eqdsk filename
            from tokamak.formats import geqdsk
            from scipy.interpolate import interpn
            self.type = 'eqdsk'
            eq = geqdsk.read(file)
            self.nr = eq['nx']
            self.nz = eq['ny']
            self.rm = eq['r']
            self.zm = eq['z']
            self.psi = eq['psi']
            self.dr = np.mean(np.diff(self.rm,axis=1))
            self.dz = np.mean(np.diff(self.zm,axis=0))
            
            def Bfeild():
                print(self.dr,self.dz,type(self.psi))
                psi_z,psi_r = np.gradient(self.psi,self.dz,self.dr)
                self.Br = -psi_z
                self.Bz = psi_r  
                
                pl.streamplot(self.rm[0,:], self.zm[:,0], self.Br, self.Bz)
                '''
                self.Bz = np.zeros((self.nz,self.nr))
                for i in range(self.nz):
                    for j in range(self.nr):
                        B = self.Bcoil(self.coil, [self.rm[i,j],self.zm[i,j]])
                        self.Br[i,j] = B[0]
                        self.Bz[i,j] = B[1]
                '''
            self.Bfeild = Bfeild
            def Bcoil():  # self,coil,point
                feild = np.zeros(2)
                print(5, self.type)
                #interpn(points, values, xi, method='linear')
                return feild
            self.Bcoil = Bcoil
        else:  # coil structure
            from cross_coil import Pcoil,Bcoil
            self.type = 'coil'
            self.Pcoil = Pcoil
            self.Bcoil = Bcoil
            self.coil = file
            self.res = res
            self.rlim = rlim
            self.zlim = zlim
            def grid(**kw):  # contor grid
                for key in kw.keys():
                    setattr(self, key, kw[key])
                dr, dz = (self.rlim[1]-self.rlim[0], self.zlim[1]-self.zlim[0])
                self.nr, self.nz = (int(dr/self.res), int(dz/self.res))
                r = np.linspace(self.rlim[0], self.rlim[1], self.nr)
                z = np.linspace(self.zlim[0], self.zlim[1], self.nz)
                self.rm, self.zm = np.meshgrid(r, z)
            def potential():
                self.psi = np.zeros((self.nz,self.nr))
                for i in range(self.nz):
                    for j in range(self.nr):
                        self.psi[i,j] = self.Pcoil(self.coil, 
                                                   [self.rm[i,j],self.zm[i,j]])
            def Bfeild():
                self.Br = np.zeros((self.nz,self.nr))
                self.Bz = np.zeros((self.nz,self.nr))
                for i in range(self.nz):
                    for j in range(self.nr):
                        B = self.Bcoil(self.coil, [self.rm[i,j],self.zm[i,j]])
                        self.Br[i,j] = B[0]
                        self.Bz[i,j] = B[1]
            self.grid()
       
    def contour(self, Nstd=1.5, label=False):
        if not hasattr(self, 'psi'):
            self.potential()
        level = [np.mean(self.psi)-Nstd*np.std(self.psi), 
                 np.mean(self.psi)+Nstd*np.std(self.psi)]
        CS = pl.contour(self.rm, self.zm, self.psi, 
                   levels=np.linspace(level[0],level[1],30), colors='k')
        for cs in CS.collections:
            cs.set_linestyle('solid')
        if label:
            pl.clabel(CS, CS.levels[::2], inline=1, fmt='%1.2e')
                   

    def Bcontor(self, axis, Nstd=1.5, color='r'):
        var = 'B'+axis
        if not hasattr(self, var):
            self.Bfeild()  
        B = getattr(self, var)
        level = [np.mean(B)-Nstd*np.std(B), 
                 np.mean(B)+Nstd*np.std(B)]
        level = [-5, 5]
        pl.contour(self.rm, self.zm, B, 
                   levels=np.linspace(level[0],level[1],30), colors=color) 
                       
    def Bstreamfunction(self):
        if not hasattr(self, 'Br'):
            self.Bfeild() 
        pl.streamplot(self.rm, self.zm, self.Br, self.Bz)
        
    def getX(self,xo):
        from scipy.optimize import minimize
        def feild(x):
            B = self.Bcoil(self.coil, x)
            return sum(B*B)**0.5
        res = minimize(feild, np.array(xo), method='nelder-mead', 
                       options={'xtol': 1e-3, 'disp': False})  
        return res.x
            
    def get_Xpsi(self, xo):
        self.Xpoint = self.getX(xo)
        self.Xpsi = self.Pcoil(self.coil, self.Xpoint)
        #pl.plot(Xpoint[0], Xpoint[1], 'co')
        #pl.contour(self.rm, self.zm, self.psi, levels=[Xpsi], colors='k')
        return (self.Xpsi, self.Xpoint)

    def midplane_contour(self, C):
        p = C.collections[0].get_paths()  # plasma contour
        N = len(p)
        for i in range(N):
            v = p[i].vertices
            z = v[:,1]
            if (z>0).any() and (z<0).any():
                LCFS = i
        v = p[LCFS].vertices
        r,z = (v[:,0],v[:,1])
        return (r,z)
        
    def get_LFP(self, xo=[7,-4.5]):  # aproximate location of first field null
        from scipy.interpolate import interp1d

        Xpsi,Xpoint = self.get_Xpsi(xo)
        pl.figure()  # dummy figure
        C=pl.contour(self.rm, self.zm, self.psi, levels=[Xpsi])
        pl.close()
        
        r,z = self.midplane_contour(C)
        index = z>Xpoint[1]
        r_loop,z_loop = (r[index], z[index])
        
        rc,zc = np.mean([np.min(r_loop), np.max(r_loop)]),0
        radius = ((r_loop-rc)**2+(z_loop-zc)**2)**0.5
        theta = np.arctan2(z_loop-zc, r_loop-rc)
        index = theta.argsort()
        radius,theta = radius[index],theta[index] 
        theta = np.append(theta[-1]-2*np.pi, theta)
        radius = np.append(radius[-1], radius)
        
        r = rc+radius*np.cos(theta)
        z = zc+radius*np.sin(theta)
        
        fLFSr = interp1d(theta, r) 
        fLFSz = interp1d(theta, z)
        self.LFPr,self.LFPz = fLFSr(0),fLFSz(0)
        
        return (self.LFPr,self.LFPz)
        
    def sol(self, color=None):
        if not hasattr(self, 'psi'):
            self.potential()
            
        if color == None:
            LFPr,LFPz = self.get_LFP()
            r = np.linspace(LFPr-1e-3, LFPr+10e-3, 11)
            z = LFPz*np.ones(len(r))
            psi = []
            for rp,zp in zip(r,z):
                psi.append(self.Pcoil(self.coil, [rp,zp]))
            cs = pl.contour(self.rm, self.zm, self.psi, levels=psi)
        else:
            Xpsi,Xpoint = self.get_Xpsi([7,-4.5])
            cs = pl.contour(self.rm, self.zm, self.psi, levels=[Xpsi], colors=color)
        self.pick_contour(cs)
        
    def pick_contour(self,cs,Xpoint=False,Midplane=True):
        self.Rsol = []
        self.Zsol = []
        Xp,Mid = True,True
        for i in range(len(cs.collections)):
            for j in range(len(cs.collections[i].get_paths())):
                p = cs.collections[i].get_paths()[j]
                v = p.vertices
                R = v[:,0][:]
                Z = v[:,1][:]
                
                if Xpoint:  # check Xpoint crossing
                    if (np.max(Z) > self.Xpoint[1]) and (np.min(Z) < self.Xpoint[1]):
                        Xp = True
                    else:
                        Xp = False
                if Midplane:  # check lf midplane crossing
                    if (np.max(Z) > self.LFPz) and (np.min(Z) < self.LFPz):  
                        Mid = True
                    else:
                        Mid = False
                if Xp and Mid:
                    self.Rsol.append(R)
                    self.Zsol.append(Z)
            
            
            
