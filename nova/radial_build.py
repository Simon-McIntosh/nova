import numpy as np
import pylab as pl
import pickle
from scipy.interpolate import interp1d as interp1
from scipy.interpolate import UnivariateSpline as spline
import seaborn as sns
from itertools import cycle

class RB(object):
    
    def __init__(self,setup,sf,Np=500):
        self.setup = setup
        self.sf = sf
        self.Np = Np
        self.dataname = setup.configuration+'_dev'
        #self.TFwidth(sf)

        if not hasattr (sf,'Xpoint'):
            sf.get_Xpsi()  
        self.xo = [sf.Xpoint[0],self.sf.eqdsk['zmagx']]
        r,z = sf.get_boundary()
        sf.set_boundary(r,z,n=Np)
        self.Rp,self.Zp = self.sf.rbdry,self.sf.zbdry
        self.Lp = self.length(self.Rp,self.Zp,norm=False)[-1]
        self.Rfw,self.Zfw,self.psi_fw = self.first_wall(self.setup.firstwall['dRfw']) 
        self.R,self.Z = self.Rfw,self.Zfw

        
    def set_target(self,leg,**kwargs):
        if leg not in self.setup.targets:
            self.setup.targets[leg] = {}
        for key in self.setup.targets['default']:
            if key in kwargs:
                self.setup.targets[leg][key] = kwargs[key]  # update
            elif key not in self.setup.targets[leg]:  #  prevent overwrite
                self.setup.targets[leg][key] = self.setup.targets['default'][key]  # default
   
    def theta_sort(self,R,Z,origin='lfs'):
        if origin == 'lfs':
            theta = np.arctan2(Z-self.xo[1],R-self.xo[0])
        elif origin == 'top':
            theta = np.arctan2(self.xo[0]-R,Z-self.xo[1])
        index = np.argsort(theta)
        R,Z = R[index],Z[index]
        return R,Z

    def rt(self,R,Z,**kwargs):
        if 'point' in kwargs.keys():
            point = kwargs.get('point')
        else:
            point = self.xo
        theta = np.unwrap(np.arctan2(Z-point[1],R-point[0]))
        radius = np.sqrt((Z-point[1])**2+(R-point[0])**2)
        index = np.argsort(theta)
        radius,theta = radius[index],theta[index]
        return radius,theta
        
    def rz(self,radius,theta,point):
        if not point: point = self.xo
        R = point[0]+radius*np.cos(theta)
        Z = point[1]+radius*np.sin(theta)
        return R,Z
        
    def rzSpline(self,R,Z,Np,w=None,s=0.005):
        radius,theta = self.rt(R,Z)
        Ts = np.linspace(theta[0],theta[-1],Np)
        if w is None:
            radius = spline(theta,radius,s=s)(Ts)
        else:
            radius = spline(theta,radius,w=w,s=s)(Ts)
        Rs,Zs = self.rz(radius,Ts)
        return Rs,Zs,Ts
        
    def rzSLine(self,R,Z,Np=0,s=0,Hres=False):
        if Np == 0: Np=self.Np
        L = self.length(R,Z)
        if Hres: Np *= 10
        Linterp = np.linspace(0,1,Np)
        if s == 0:
            R = interp1(L,R)(Linterp)
            Z = interp1(L,Z)(Linterp)
        else:
            R = spline(L,R,s=s)(Linterp)
            Z = spline(L,Z,s=s)(Linterp)
        return R,Z
        
    def rzInterp(self,R,Z,Np=0,ends=True):
        if Np == 0: Np=self.Np
        L = self.length(R,Z)
        Linterp = np.linspace(0,1,Np)
        R = interp1(L,R)(Linterp)
        Z = interp1(L,Z)(Linterp)
        if not ends:
            R,Z = R[:-1],Z[:-1]
        return R,Z
        
    def rzCirc(self,R,Z):
        radius,theta = self.rt(R,Z)
        R,Z = self.rz(radius,theta)
        return R,Z
        
    def rzPut(self):
        self.Rstore,self.Zstore = self.R,self.Z
        
    def rzGet(self):
        self.R,self.Z = self.Rstore,self.Zstore

    def first_wall_psi(self,req,zeq):
        psi = self.sf.Pcoil([req,zeq])  # low feild
        contours = self.sf.get_contour([psi])    
        R,Z = self.sf.pick_contour(contours)
        min_contour = np.empty(len(R))
        for i in range(len(R)):
            min_contour[i] = np.min((R[i]-req)**2+(Z[i]-zeq)**2)
        imin = np.argmin(min_contour)
        r,z = R[imin],Z[imin]
        r,z = r[z<=zeq],z[z<=zeq]
        if req > self.sf.Xpoint[0]:
            r,z = r[r>self.sf.Xpoint[0]],z[r>self.sf.Xpoint[0]]
        else:
            r,z = r[r<self.sf.Xpoint[0]],z[r<self.sf.Xpoint[0]]
        istart = np.argmin((r-req)**2+(z-zeq)**2)
        r = np.append(r[istart+1:],r[:istart])
        z = np.append(z[istart+1:],z[:istart])
        istart = np.argmin((r-req)**2+(z-zeq)**2)
        if istart > 0:
            r,z = r[::-1],z[::-1]
        return r,z,psi
        
    def first_wall(self,dr):  # dr [m]
        if not hasattr(self.sf,'LFPr'):
            self.sf.get_LFP()
        self.sf.LFfwr,self.sf.LFfwz = self.sf.LFPr+dr,self.sf.LFPz    
        self.sf.HFfwr,self.sf.HFfwz = self.sf.HFPr-dr,self.sf.HFPz
        r_lfs,z_lfs,psi_lfs = self.first_wall_psi(self.sf.LFfwr,self.sf.LFfwz)
        r_hfs,z_hfs,psi_hfs = self.first_wall_psi(self.sf.HFfwr,self.sf.HFfwz)
        r_top,z_top = self.offset(self.sf.rbdry,self.sf.zbdry,-dr)
        r_top,z_top = self.theta_sort(r_top,z_top,origin='top')
        index = z_top>=self.sf.LFPz
        r_top,z_top = r_top[index],z_top[index]
        istart = np.argmin((r_top-self.sf.HFfwr)**2+(z_top-self.sf.HFfwz)**2)
        if istart > 0:
            r_top,z_top = r_top[::-1],z_top[::-1]
        r = np.append(r_hfs[::-1],r_top)
        r = np.append(r,r_lfs)
        z = np.append(z_hfs[::-1],z_top)
        z = np.append(z,z_lfs)
        return r[::-1],z[::-1],(psi_lfs,psi_hfs)  # r[::-1],z[::-1]
  
    def elipsoid(self,theta,ro,zo,A,k,delta):
        R = ro+ro/A*np.cos(theta+delta*np.sin(theta))
        Z = zo+ro/A*k*np.sin(theta)
        return (R,Z)
        
    def normal(self,R,Z):
        dR,dZ = np.gradient(R),np.gradient(Z)
        mag = np.sqrt(dR**2+dZ**2)
        index = mag>0
        dR,dZ,mag = dR[index],dZ[index],mag[index]  # clear duplicates
        R,Z = R[index],Z[index]
        t = np.zeros((len(R),3))
        t[:,0],t[:,1] = dR/mag,dZ/mag
        n = np.cross(t, [0,0,1])
        nR,nZ = n[:,0],n[:,1]
        return (nR,nZ)
        
    def length(self,R,Z,norm=True):
        L = np.append(0,np.cumsum(np.sqrt(np.diff(R)**2+np.diff(Z)**2)))
        if norm: L = L/L[-1]
        return L

    def inflate(self,R,Z,dR):
        s = 1e-3
        L = self.length(R,Z)
        nR,nZ = self.normal(R,Z)
        Rdr,Zdr = R+dR*nR,Z+dR*nZ  # radial offset
        Rloop,Zloop = self.reorder(Rdr,Zdr)  # spline loop
        Lloop = self.length(Rloop,Zloop)
        Rloop = interp1(Lloop,Rloop)(L)  # respace
        Zloop = interp1(Lloop,Zloop)(L)
        Lend = np.zeros(2)
        for i in [0,-1]:
            Lend[i] = L[np.argmin((Rloop-Rdr[i])**2+(Zloop-Zdr[i])**2)]
        Linterp = np.linspace(Lend[0],Lend[1],len(R))
        R = spline(L,Rloop,s=s)(Linterp) 
        Z = spline(L,Zloop,s=s)(Linterp)
        return (R,Z)
        
    def reorder(self,R,Z):
        s = 1e-2
        theta = np.unwrap(np.arctan2(Z-self.xo[1],R-self.xo[0]))
        radius = np.sqrt((Z-self.xo[1])**2+(R-self.xo[0])**2)
        radius = spline(theta,radius,s=s)(self.theta)
        R = self.xo[0]+radius*np.cos(self.theta)
        Z = self.xo[1]+radius*np.sin(self.theta)
        return (R,Z)
    
    def max_steps(self,dR,dr_max):
        dRbar = np.mean(dR)
        nr=int(np.ceil(dRbar/dr_max))
        if nr < 3: nr = 3
        dr=dR/nr
        return dr,nr
        
    def offset(self,R,Z,dR):
        dr_max = 0.02  # maximum step size
        if np.mean(dR) != 0:
            dr,nr = self.max_steps(dR,dr_max)
            for i in range(nr):
                nR,nZ = self.normal(R,Z)
                R = R+dr*nR    
                Z = Z+dr*nZ 
        return R,Z   
        
    def trim(self,trim,R,Z):
        L = self.length(R,Z,norm=True)
        index = []
        for t in trim:
            index.append(np.argmin(np.abs(L-t)))
        return index
        
    def fill(self,trim=None,dR=0,dt=0,ref_o=4/8*np.pi,dref=np.pi/4,
             edge=True,ends=True,
             color='k',label=None,alpha=0.8,referance='theta',part_fill=True,
             loop=False,s=0,offset=0):

        dt_max = 0.5
        if not part_fill:
            dt_max = dt
        if isinstance(dt,list):
            dt = self.blend(dt,ref_o=ref_o,dref=dref,referance=referance,
                            offset=offset)
        dt,nt = self.max_steps(dt,dt_max)
        Rin,Zin = self.offset(self.R,self.Z,dR)  # gap offset
        for i in range(nt):
            self.part_fill(trim=trim,dt=dt,ref_o=ref_o,dref=dref,
             edge=edge,ends=ends,color=color,label=label,alpha=alpha,
             referance=referance,loop=loop,s=s)
             
    def coil_sheild(self,coils=[],dt=0.8):
        self.rzPut()
        for coil in coils:
            name = 'Coil'+str(coil)
            r,dr = self.sf.coil[name]['r'],self.sf.coil[name]['dr']
            z,dz = self.sf.coil[name]['z'],self.sf.coil[name]['dz']
            r = [r-dr/2,r-dr/2,r+dr/2,r+dr/2,r-dr/2]
            z = [z-dz/2,z+dz/2,z+dz/2,z-dz/2,z-dz/2]
            self.R,self.Z = self.rzInterp(r,z,Np=100)
            self.R,self.Z = self.R[::-1],self.Z[::-1]
            self.fill(dt=dt,loop=True,alpha=0.1,color='k',s=5e-3)
        self.rzGet()

    def part_fill(self,trim=None,dt=0,ref_o=4/8*np.pi,dref=np.pi/4,
             edge=True,ends=True,
             color='k',label=None,alpha=0.8,referance='theta',loop=False,
             s=0):
        Rin,Zin = self.R,self.Z
        if loop:
            Napp = 5  # Nappend
            R = np.append(self.R,self.R[:Napp])
            R = np.append(self.R[-Napp:],R)
            Z = np.append(self.Z,self.Z[:Napp])
            Z = np.append(self.Z[-Napp:],Z)
            R,Z = self.rzSLine(R,Z,Np=len(R),s=s)
            if isinstance(dt,(np.ndarray,list)):
                dt = np.append(dt,dt[:Napp])
                dt = np.append(dt[-Napp:],dt)
            Rout,Zout = self.offset(R,Z,dt)
            Rout,Zout = Rout[Napp:-Napp],Zout[Napp:-Napp]
            Rout[-1],Zout[-1] = Rout[0],Zout[0]
        else:
            R,Z = self.rzSLine(self.R,self.Z,Np=len(self.R),s=s)
            Rout,Zout = self.offset(R,Z,dt)
        self.R,self.Z = Rout,Zout  # update
        if trim is None:
            Lindex = [0,len(Rin)]
        else:
            Lindex = self.trim(trim)
        flag = 0
        for i in np.arange(Lindex[0],Lindex[1]-1):
            Rfill = np.array([Rin[i],Rout[i],Rout[i+1],Rin[i+1]])
            Zfill = np.array([Zin[i],Zout[i],Zout[i+1],Zin[i+1]])
            if flag is 0 and label is not None:
                flag = 1
                pl.fill(Rfill,Zfill,facecolor=color,alpha=alpha,
                        edgecolor='none',label=label)
            else:
                pl.fill(Rfill,Zfill,facecolor=color,alpha=alpha,
                        edgecolor='none')

    def rotateB(self,B,theta):
        Bmag = np.sqrt(B[0]**2+B[1]**2)
        R = np.matrix([[np.cos(theta),-np.sin(theta)],
                        [np.sin(theta),np.cos(theta)]])
        Bhat = (R*np.matrix([[B[0]],[B[1]]])).getA1()/Bmag
        return Bhat
        
    def firstwall(self,calc=True,color=[0.5,0.5,0.5],plot=True,debug=False):
        if calc:
            self.Rb,self.Zb = self.target_fill(debug=debug)
            with open('../Data/'+self.dataname+'_FW.pkl', 'wb') as output:
                pickle.dump(self.setup.targets,output,-1)
                pickle.dump(self.Rb,output,-1)
                pickle.dump(self.Zb,output,-1)
        else:
            with open('../Data/'+self.dataname+'_FW.pkl', 'rb') as input:
                self.setup.targets = pickle.load(input)
                self.Rb = pickle.load(input)
                self.Zb = pickle.load(input)
        self.R,self.Z = self.trim_contour(self.Rb,self.Zb) # update fw
        self.sf.nlim = len(self.Rb)  # update sf
        self.sf.xlim,self.sf.ylim = self.Rb,self.Zb
        if plot:
            index = (self.Rb <= self.sf.r.max()) & (self.Rb >= self.sf.r.min()) &\
            (self.Zb <= self.sf.z.max()) & (self.Zb >= self.sf.z.min())
            pl.plot(self.Rb[index],self.Zb[index],
                    '-',color=color,alpha=0.75,linewidth=1.25)
            
    def get_fw(self,expand=0):  # generate boundary dict for elliptic
        self.firstwall(calc=False,plot=False,debug=False)
        boundary = {'R':self.Rb,'Z':self.Zb,'expand':expand}
        return boundary
        
    def connect(self,psi,target_pair,ends,loop=[]):
        gap = []
        if loop:
            r,z = loop
        else:
            psi_line = self.sf.get_contour([psi])[0]
            for line in psi_line:
                r,z = line[:,0],line[:,1]
                gap.append(np.min((self.setup.targets[target_pair[0]]['R'][ends[0]]-r)**2+
                          (self.setup.targets[target_pair[0]]['Z'][ends[0]]-z)**2))
            select = np.argmin(gap)
            line = psi_line[select]
            r,z = line[:,0],line[:,1]
        index = np.zeros(2)
        index[0] = np.argmin((self.setup.targets[target_pair[0]]['R'][ends[0]]-r)**2+
                 (self.setup.targets[target_pair[0]]['Z'][ends[0]]-z)**2)
        index[1] = np.argmin((self.setup.targets[target_pair[1]]['R'][ends[1]]-r)**2+
                 (self.setup.targets[target_pair[1]]['Z'][ends[1]]-z)**2)
        if index[0]>index[1]:
            index = index[::-1]
        r,z = r[index[0]:index[1]+1],z[index[0]:index[1]+1] 
        return r,z

    def target_fill(self,debug=False,**kwargs):
        layer_index = 0  # construct from first open flux surface
        self.sf.sol()
        for leg in list(self.sf.legs)[2:]:
            self.set_target(leg)
            Rsol,Zsol = self.sf.snip(leg,layer_index,L2D=self.setup.targets[leg]['L2D'])
            Ro,Zo = Rsol[-1],Zsol[-1]
            
            Flip = [-1,1]
            Direction = [1,-1]
            Theta_end = [0,np.pi]
            if 'inner' in leg:
                psi_plasma = self.psi_fw[1]
            else:
                psi_plasma = self.psi_fw[0]

            self.dpsi = self.psi_fw[1]-self.sf.Xpsi
            Phi_target = [psi_plasma,self.sf.Xpsi-self.setup.firstwall['div_ex']*self.dpsi]  
            if leg is 'inner1' or leg is 'outer2':
                Phi_target[0] = self.sf.Xpsi+self.setup.firstwall['div_ex']*self.dpsi

            if self.setup.targets[leg]['open']:
                theta_sign = -1
                Direction = Direction[::-1]
                Theta_end = Theta_end[::-1]
            else:
                theta_sign = 1
            if 'outer' in leg:
                Direction = Direction[::-1]
                theta_sign *= -1
            if leg is 'inner1' or leg is 'outer2':
                Theta_end = Theta_end[::-1]

            self.setup.targets[leg]['R'] = np.array([])
            self.setup.targets[leg]['Z'] = np.array([])
            dPlate = self.setup.targets[leg]['dPlate']
            for flip,direction,theta_end,psi_target\
            in zip(Flip,Direction,Theta_end,Phi_target):
                R,Z = self.match_psi(Ro,Zo,direction,theta_end,theta_sign,
                                     psi_target,self.setup.targets[leg]['graze'],
                                     dPlate,leg,debug=debug)
                self.setup.targets[leg]['R'] = np.append(self.setup.targets[leg]['R'],R[::flip])
                self.setup.targets[leg]['Z'] = np.append(self.setup.targets[leg]['Z'],Z[::flip])
            if leg is 'outer':
                self.setup.targets[leg]['R'] = self.setup.targets[leg]['R'][::-1]
                self.setup.targets[leg]['Z'] = self.setup.targets[leg]['Z'][::-1]
        Rb,Zb = np.array([]),np.array([])

        if self.sf.nleg == 6:  # SF
            Rb = np.append(Rb,self.setup.targets['inner2']['R'][1:])
            Zb = np.append(Zb,self.setup.targets['inner2']['Z'][1:])
            r,z = self.connect(self.sf.Xpsi-self.setup.firstwall['div_ex']*self.dpsi,
                               ['inner2','inner1'],[-1,-1])
            Rb,Zb = self.append(Rb,Zb,r,z)
            Rb = np.append(Rb,self.setup.targets['inner1']['R'][::-1])
            Zb = np.append(Zb,self.setup.targets['inner1']['Z'][::-1])
            r,z = self.connect(self.sf.Xpsi+self.setup.firstwall['div_ex']*self.dpsi,
                               ['inner1','outer2'],[0,0])            
            Rb,Zb = self.append(Rb,Zb,r,z)
            Rb = np.append(Rb,self.setup.targets['outer2']['R'][1:])
            Zb = np.append(Zb,self.setup.targets['outer2']['Z'][1:])
            r,z = self.connect(self.sf.Xpsi-self.setup.firstwall['div_ex']*self.dpsi,
                               ['outer2','outer1'],[-1,-1])            
            Rb,Zb = self.append(Rb,Zb,r,z)
            Rb = np.append(Rb,self.setup.targets['outer1']['R'][::-1])
            Zb = np.append(Zb,self.setup.targets['outer1']['Z'][::-1])
            r,z = self.connect(self.psi_fw[1],['outer1','inner2'],[0,0],
                               loop=(self.R[::-1],self.Z[::-1]))
            Rb,Zb = self.append(Rb,Zb,r,z)
        else:
            Rb = np.append(Rb,self.setup.targets['inner']['R'][1:])
            Zb = np.append(Zb,self.setup.targets['inner']['Z'][1:])
            r,z = self.connect(self.sf.Xpsi-self.setup.firstwall['div_ex']*self.dpsi,
                               ['inner','outer'],[-1,0])                
            Rb,Zb = self.append(Rb,Zb,r,z)
            Rb = np.append(Rb,self.setup.targets['outer']['R'][1:])
            Zb = np.append(Zb,self.setup.targets['outer']['Z'][1:])
            r,z = self.connect(self.psi_fw[1],['outer','inner'],[-1,0],
                               loop=(self.R,self.Z))
            Rb,Zb = self.append(Rb,Zb,r,z)
        self.Rb,self.Zb = self.midplane_loop(Rb,Zb)
        self.get_sol()  # trim sol to targets and store values
        self.write_boundary(self.Rb,self.Zb)
        return self.Rb,self.Zb
        
    def append(self,R,Z,r,z):
        dr = np.zeros(2)
        for i,end in enumerate([0,-1]):
            dr[i] = (R[-1]-r[end])**2+(Z[-1]-z[end])**2
        if dr[1] < dr[0]:
            r,z = r[::-1],z[::-1]
        return np.append(R,r[1:-1]),np.append(Z,z[1:-1]) 
        
    def get_sol(self):
        self.trim_sol(plot=False)
        for leg in list(self.sf.legs)[2:]:
            L2D,L3D,Rsol,Zsol = self.sf.connection(leg,0)
            Ro,Zo = Rsol[-1],Zsol[-1]
            L2Dedge,L3Dedge = self.sf.connection(leg,-1)[:2]
            Xi = self.sf.expansion([Ro],[Zo])
            theta = self.sf.strike_point(Xi,self.setup.targets[leg]['graze'])
            self.setup.targets[leg]['theta'] = theta
            self.setup.targets[leg]['L2Do'] = L2D
            self.setup.targets[leg]['L3Do'] = L3D
            self.setup.targets[leg]['L2Dedge'] = L2Dedge
            self.setup.targets[leg]['L3Dedge'] = L3Dedge
            self.setup.targets[leg]['Ro'] = Ro
            self.setup.targets[leg]['Zo'] = Zo 
            self.setup.targets[leg]['Rsol'] = Rsol
            self.setup.targets[leg]['Zsol'] = Zsol 
        
    def trim_sol(self,dr=0,Nsol=0,update=False,color='k',plot=True):
        self.sf.sol(dr=dr,Nsol=Nsol,update=update)
        color = sns.color_palette('Set2',self.sf.nleg+5)
        #color = 'k'
        for c,leg in enumerate(self.sf.legs.keys()):
            if 'core' not in leg:
                Rsol = self.sf.legs[leg]['R']
                Zsol = self.sf.legs[leg]['Z']
                for i in range(self.sf.Nsol):
                    if len(Rsol[i]) > 0:
                        R,Z = Rsol[i],Zsol[i]
                        for j in range(2):  # predict - correct
                            R,Z = self.inloop(self.Rb[::-1],self.Zb[::-1],R,Z)
                        self.sf.legs[leg]['R'][i] = R  # update sf
                        self.sf.legs[leg]['Z'][i] = Z
                        if plot:
                            if color != 'k' and i > 0:
                                pl.plot(R,Z,'-',color=color[c+3])
                            elif color == 'k':
                                pl.plot(R,Z,'-',color='k',alpha=0.15)
                            else:
                                #pl.plot(R,Z,color=color[c])
                                pl.plot(R,Z,'--',color=[0.5,0.5,0.5])

    def trim_contour(self,Rb,Zb):
        Rt,Zt = np.array([]),np.array([])  # trim contour for BB
        Lnorm = self.length(self.R,self.Z,norm=False)[-1]
        for flip,trim in zip([1,-1],[0.275*self.setup.firstwall['trim'][1],
                              0.725*self.setup.firstwall['trim'][0]]):
            L = self.length(Rb[::flip],Zb[::flip],norm=False)
            i = np.argmin(np.abs(L/Lnorm-trim))
            Rt = np.append(Rt,Rb[::flip][:i][::-flip])
            Zt = np.append(Zt,Zb[::flip][:i][::-flip])
        Rt,Zt = self.rzSLine(Rt,Zt,self.Np)
        return Rt,Zt
        
    def midplane_loop(self,Rb,Zb):
        index = np.argmin((Rb-self.sf.LFfwr)**2+(Zb-self.sf.LFfwz)**2)
        Rb = np.append(Rb[:index+1][::-1],Rb[index:][::-1])
        Zb = np.append(Zb[:index+1][::-1],Zb[index:][::-1])
        Lb = self.length(Rb,Zb)
        index = np.append(np.diff(Lb)!=0,True)
        Rb,Zb = Rb[index],Zb[index]  # remove duplicates
        return Rb,Zb
        
    def write_boundary_feild(self,Rb,Zb):
        Rb,Zb = Rb[::-1],Zb[::-1]  # flip boundary (clockwise)
        with open('../Data/'+self.dataname+'_bdry_feild.txt','w') as f:
            f.write(self.conf.filename.split('/')[-1]+'\n')
            f.write('boundary clockwise from outboard midplane\n')
            f.write('r[m]\tz[m]\tBp[T]\tBphi[T]\n')
            f.write('{:1.0f}\n'.format(len(Rb)-1))
            Bm = np.abs(self.sf.bcentr*self.sf.rcentr)
            for r,z in zip(Rb,Zb):
                B = self.sf.Bcoil([r,z])
                Bp = np.sqrt(B[0]**2+B[1]**2)  # polodial feild
                Bphi = Bm/r  # torodal field
                f.write('{:1.4f}\t{:1.4f}\t'.format(r,z))
                f.write('{:1.4f}\t{:1.4f}\n'.format(Bp,Bphi))
        
    def write_boundary(self,Rb,Zb):
        with open('../Data/'+self.dataname+'_bdry.txt','w') as f:
            f.write(self.sf.filename.split('/')[-1]+'\n')
            f.write('boundary placed {:1.0f}'\
            .format(1e3*self.setup.firstwall['dRfw']))
            f.write('mm from LCFS at LFS, upper==constant offset, lower==feild line conformal\n')
            f.write('connection calculated from Xpoint at ')
            f.write('{:1.2f}mm from LFS\n\n'.format(1e3*self.sf.Dsol[-1]))
            f.write('target\tgraze[deg]\topen\tRo[m]\t\tZo[m]\t\tTo[deg]\tL2D[m]\tL3D[m]\n')
            for leg in list(self.sf.legs)[2:]:
                target = self.setup.targets[leg]
                f.write(leg+'\t')
                f.write('{:1.2f}\t\t'.format(target['graze']*180/np.pi))
                if target['open']:
                    f.write('open\t')
                else:
                    f.write('closed\t')
                f.write('{:1.6f}\t'.format(target['Ro']))
                f.write('{:1.6f}\t'.format(target['Zo']))
                f.write('{:1.2f}\t'.format(target['theta']*180/np.pi))
                f.write('{:1.3f}\t'.format(target['L2Dedge'][-1]))
                f.write('{:1.3f}\t\n'.format(target['L3Dedge'][-1]))
            f.write('\n')
            f.write('boundary segments clockwise from outboard midplane\n')
            f.write('r[m]\t\tz[m]\n')
            f.write('{:1.0f}\n'.format(len(Rb)))
            for i in range(len(Rb)):
                f.write('{:1.6f}\t{:1.6f}\n'.format(Rb[i],Zb[i]))  # dp
    
    def crossed_lines(self,Ro,Zo,R1,Z1):
        index = np.zeros(2)
        dl = np.zeros(len(Ro))
        for i,(ro,zo) in enumerate(zip(Ro,Zo)):
            dl[i] = np.min((R1-ro)**2+(Z1-zo)**2)
        index[0] = np.argmin(dl)
        index[1] = np.argmin((R1-Ro[index[0]])**2+(Z1-Zo[index[0]])**2)
        return index
                
    def inloop(self,Rloop,Zloop,R,Z):
        L = self.length(R,Z)
        index = np.append(np.diff(L)!=0,True)
        R,Z = R[index],Z[index]  # remove duplicates
        nRloop,nZloop = self.normal(Rloop,Zloop)
        Rin,Zin = np.array([]),np.array([])

        for r,z in zip(R,Z):
            i = np.argmin((r-Rloop)**2+(z-Zloop)**2)
            dr = [Rloop[i]-r,Zloop[i]-z]  
            dn = [nRloop[i],nZloop[i]]
            if np.dot(dr,dn) > 0:
                Rin,Zin = np.append(Rin,r),np.append(Zin,z)
        i = np.argmin((Rin[-1]-R)**2+(Zin[-1]-Z)**2)
        Rin,Zin = R[:i+2],Z[:i+2] # extend past target
        i = np.argmin((Rin[-1]-R)**2+(Zin[-1]-Z)**2)  # sol crossing bndry
        jo = np.argmin((R[i]-Rloop)**2+(Z[i]-Zloop)**2)
        j = np.array([jo,jo+1])
        s = np.array([R[i],Z[i]])
        ds = np.array([R[i]-R[i-1],Z[i]-Z[i-1]])
        b = np.array([Rloop[j[0]],Zloop[j[0]]])
        bstep,db = self.get_bstep(s,ds,b,Rloop,Zloop,j)
        if bstep < 0:
            j = np.array([jo-1,jo])  # switch target pannel
            bstep,db = self.get_bstep(s,ds,b,Rloop,Zloop,j)
        step = np.cross(b-s,db)/np.cross(ds,db)
        intersect = s+step*ds  # predict - correct
        if step < 0:  # step back
            Rin,Zin = Rin[:-1],Zin[:-1]
        Rin,Zin = np.append(Rin,intersect[0]),np.append(Zin,intersect[1])
        return Rin,Zin
        
    def get_bstep(self,s,ds,b,Rloop,Zloop,j):
        db = np.array([Rloop[j[1]]-Rloop[j[0]],Zloop[j[1]]-Zloop[j[0]]])
        step = np.cross(b-s,ds)/np.cross(ds,db)
        return step,db

    def match_psi(self,Ro,Zo,direction,theta_end,theta_sign,phi_target,graze,
                  dPlate,leg,debug=False):        
        color = sns.color_palette('Set2',2)
        gain = 0.15  # 0.25
        Nmax = 500
        Lo = [5,0.015]  # [blend,turn]  5,0.015
        r2m = [-1.5,-1]  # ramp to step (+ive-lead, -ive-lag ramp==1, step==inf)
        Nplate = 1 # number of target plate divisions (1==flat)
        L = Lo[0] if theta_end == 0 else Lo[1]
        Lsead = L
        flag = 0
        for i in range(Nmax):
            R,Z,phi = self.blend_target(Ro,Zo,dPlate,L,direction,theta_end,
                                        theta_sign,graze,r2m,Nplate)
            L -= gain*(phi_target-phi)
            if debug: pl.plot(R,Z,color=color[0])
            if np.abs(phi_target-phi) < 1e-4:
                if debug: 
                    pl.plot(R,Z,'r',color=color[1])
                    print('rz',Ro,Zo,'N',i)
                break
            if L < 0:
                L = 1
                gain *= -1
            if i == Nmax-1 or L>15:
                print(leg,'dir',direction,'phi target convergence error')
                print('Nmax',i+1,'L',L,'Lo',Lsead)
                if flag == 0:
                    break
                    gain *= -1  # reverse gain
                    flag = 1
                else:
                    break
        return R,Z
                        
    def blend_target(self,Ro,Zo,dPlate,L,direction,theta_end,theta_sign,graze,
                     r2m,Nplate):
            r2s = r2m[0] if theta_end == 0 else r2m[1]
            dL = 0.1 if theta_end == 0 else 0.05  # 0.005,0.005
            R,Z = np.array([Ro]),np.array([Zo])
            R,Z = self.extend_target(R,Z,dPlate/(2*Nplate),Nplate,r2s,
                                     theta_end,theta_sign,
                                     direction,graze,False,
                                     target=True)  # constant graze 
            Ninterp = int(dPlate/(2*dL))
            if Ninterp < 2: Ninterp = 2
            R,Z = self.rzInterp(R,Z,Ninterp)
            graze = self.sf.get_graze([R[-1],Z[-1]],
                                      [R[-1]-R[-2],Z[-1]-Z[-2]])  # update graze
            N = np.int(L/dL+1)
            if N<30: N=30
            dL = L/(N-1)
            target_angle = np.arctan2((Z[-1]-Z[-2]),(R[-1]-R[-2]))
            Xi = self.sf.expansion([R[-1]],[Z[-1]])
            theta = self.sf.strike_point(Xi,graze)
            B = direction*self.sf.Bcoil((R[-1],Z[-1]))
            Bhat = self.rotateB(B,theta_sign*theta)
            trans_angle = np.arctan2(Bhat[1],Bhat[0])
            if abs(target_angle-trans_angle) > 0.01*np.pi:
                accute = True
            else:
                accute = False
            R,Z = self.extend_target(R,Z,dL,N,r2s,theta_end,theta_sign,
                                direction,graze,accute)  # transition graze
            phi = self.sf.Pcoil([R[-1],Z[-1]])
            return R,Z,phi
            
    def extend_target(self,R,Z,dL,N,r2s,theta_end,theta_sign,direction,graze,
                      accute,target=False):
        for i in range(N):
            if target:
                Lhat = 0
            else:
                Lhat = i/(N-1)
                if r2s < 0:  # delayed transtion
                    Lhat = Lhat**np.abs(r2s)
                else:  # expedient transition
                    Lhat = Lhat**(1/r2s)
            Xi = self.sf.expansion([R[-1]],[Z[-1]])
            theta = self.sf.strike_point(Xi,graze)
            if accute: theta = np.pi-theta
            theta = Lhat*theta_end+(1-Lhat)*theta
            B = direction*self.sf.Bcoil((R[-1],Z[-1]))
            Bhat = self.rotateB(B,theta_sign*theta)
            R = np.append(R,R[-1]+dL*Bhat[0])
            Z = np.append(Z,Z[-1]+dL*Bhat[1])
        return R,Z
        
    def vessel(self):  # radial build from firstwall
        Color = cycle(sns.color_palette('Set2',12)[1:])
        self.FWfill(dt=self.setup.build['tfw'],color=next(Color))  # first wall 
        self.fill(dt=self.setup.build['BB'][::-1],alpha=0.7,ref_o=0.3,dref=0.2,
                  referance='length',color=next(Color))  # blanket+sheild
        self.fill(dt=self.setup.build['tBBsupport'],alpha=0.7,
                  color=next(Color))  # blanket support
        self.BBsheild_fill(dt=self.setup.build['sheild'],
                           ref_o=0.35*np.pi,dref=0.2*np.pi,
                           offset=1/10*np.pi,alpha=0.7,color=next(Color))
        self.VVfill(dt=self.setup.build['VV'],ref_o=0.25*np.pi,  # vessel
                    dref=0.25*np.pi,offset=0.5/10*np.pi,alpha=0.7,
                    loop=True,color=next(Color))  # ref_o=0.385
        pl.axis('equal')
        
    def FWfill(self,**kwargs):
        self.R,self.Z = self.rzSLine(self.Rb[::-1],self.Zb[::-1],Hres=False)
        self.fill(loop=True,alpha=0.7,**kwargs)
        self.R,self.Z = self.rzSLine(self.R,self.Z)
        self.R,self.Z = self.trim_contour(self.R[::-1],self.Z[::-1]) # update fw

    def BBsheild_fill(self,**kwargs):
        if 'sheild_connect' in self.setup.build:
            connect = self.setup.build['sheild_connect']
        else:
            connect=[0,1]
        index = self.trim(connect,self.R,self.Z)
        rVV = self.R[index[0]:index[1]+1]
        zVV = self.Z[index[0]:index[1]+1]
        ni = 10  #  number of trim nodes
        rp,zp = rVV[-ni:],zVV[-ni:] 
        if self.setup.build['Dsheild']:
            Rleg,Zleg = [],[]
            Rd = self.Rb[self.Zb<self.sf.Xpoint[1]][::-1]
            Zd = self.Zb[self.Zb<self.sf.Xpoint[1]][::-1]
            Ld = self.length(Rd,Zd)
            index = (Ld>self.conf.Dsheild[0]) & (Ld<self.conf.Dsheild[1])
            Rleg.append(Rd[index])
            Zleg.append(Zd[index])
        else:
            Nsp = len(self.setup.targets.keys())-1  # ignore default
            if self.setup.build['sheild_base'] >= 0: Nsp+=1
            sp = np.zeros((Nsp,),  # spline points
                               dtype=[('Rleg','float'),('Zleg','float'),
                                      ('Tleg','float'),('dR','float')])
            zmin = np.zeros((1,),  # spline points
                               dtype=[('Rleg','float'),('Zleg','float'),
                                      ('Tleg','float'),('dR','float')])
            for i,leg in enumerate(list(self.setup.targets)[1:]):
                Rt,Zt = self.setup.targets[leg]['R'],self.setup.targets[leg]['Z']
                R,Z = self.offset(Rt,Zt,self.setup.build['tfw'])
                dRt = np.mean((Rt-self.sf.Xpoint[0])**2+
                              (Zt-self.sf.Xpoint[1])**2)
                dR = np.mean((R-self.sf.Xpoint[0])**2+
                              (Z-self.sf.Xpoint[1])**2)
                if  dR<dRt:
                    Rt,Zt = self.offset(Rt,Zt,-self.setup.build['tfw'])
                else:
                    Rt,Zt = R,Z
                if self.setup.targets[leg]['Zo'] > self.sf.Xpoint[1]:
                    index = Zt<self.setup.targets[leg]['Zo']
                else:
                    Zmax = self.setup.targets[leg]['Zo']+\
                    (self.sf.Xpoint[1]-self.setup.targets[leg]['Zo'])/2
                    index = Zt<Zmax  
                Rt,Zt = Rt[index],Zt[index]              
                radius = np.sqrt((Rt-self.sf.Xpoint[0])**2+
                                 (Zt-self.sf.Xpoint[1])**2)
                argmax = np.argmax(radius)
                r = radius[argmax]+self.setup.targets[leg]['dR']
                theta = np.arctan2(Rt[argmax]-self.sf.Xpoint[0],
                                   self.sf.Xpoint[1]-Zt[argmax])
                sp['Rleg'][i] = self.sf.Xpoint[0]+r*np.sin(theta)
                sp['Zleg'][i] = self.sf.Xpoint[1]-r*np.cos(theta)
                sp['Tleg'][i] = theta
                sp['dR'][i] = self.setup.targets[leg]['dR']
            
            L = self.length(self.Rb,self.Zb,norm=False)[-1]
            Rt,Zt = self.offset(self.Rb,self.Zb,self.setup.build['tfw'])
            Lt = self.length(Rt,Zt,norm=False)[-1]
            if Lt<L:
               Rt,Zt = self.offset(self.Rb,self.Zb,-self.setup.build['tfw']) 
            argmin = np.argmin(Zt)
            radius = np.sqrt((Rt-self.sf.Xpoint[0])**2+
                                 (Zt-self.sf.Xpoint[1])**2)
            if self.setup.build['sheild_base'] >= 0:
                r = radius[argmin]+self.setup.build['sheild_base']
                theta = np.arctan2(Rt[argmin]-self.sf.Xpoint[0],
                                   self.sf.Xpoint[1]-Zt[argmin])
                zmin['Rleg'][0] = self.sf.Xpoint[0]+r*np.sin(theta)
                zmin['Zleg'][0] = self.sf.Xpoint[1]-r*np.cos(theta)
                zmin['Tleg'][0] = theta
                zmin['dR'][0] = self.setup.build['sheild_base']
                zmin = np.sort(zmin,order='Zleg')  
                sp[-1] = zmin[0]
            sp = np.sort(sp,order='Tleg')[::-1]
            Rleg,Zleg = [],[]
            for i,dr in enumerate(sp['dR']):  # remove -ive dR
                if dr >= 0:
                    Rleg.append(sp['Rleg'][i])
                    Zleg.append(sp['Zleg'][i])
        rp,zp = np.append(rp,Rleg[::-1]),np.append(zp,Zleg[::-1])
        rp,zp = np.append(rp,rVV[:ni]),np.append(zp,zVV[:ni])
        rs,zs = self.rzSLine(rp,zp,Np=260,s=5e-6)
        R,Z = np.append(rVV[ni+1:-ni],rs),np.append(zVV[ni+1:-ni],zs)
        R,Z = np.append(R,R[0]),np.append(Z,Z[0])
        self.R,self.Z = self.rzSLine(R,Z,len(R),s=5e-4)
        self.fill(trim=None,loop=True,s=2e-4,**kwargs)
 
    def support_fill(self,coils=[],r=[],z=[],**kwargs):
        k = 3
        rp = np.array([])
        zp = np.array([])
        for name in coils:
            rp = np.append(rp,self.geom.coil[name]['r'])
            zp = np.append(zp,self.geom.coil[name]['z']-
                           self.geom.coil[name]['rc'])
        
        zp[0] += 1.5*self.geom.coil[coils[0]]['rc']
        rp[0] += 0.5*self.geom.coil[coils[0]]['rc']
        rp = np.append(rp,r)
        zp = np.append(zp,z)
        wp = np.ones(len(rp))
        wp[0] = 3
        L = np.append(0,np.cumsum(np.sqrt(np.diff(rp)**2+np.diff(zp)**2)))
        L = L/L[-1]
        Ls = np.linspace(0,1,20)
        rs = spline(L,rp,w=wp,k=k)(Ls)
        zs = spline(L,zp,w=wp,k=k)(Ls)
        
        self.R,self.Z = rs,zs
        self.rzPut()
        self.fill(trim=None,**kwargs)
        
        rs,zs = self.offset(rs,zs,kwargs['dR'])
        return rs,zs
   
    def sheild_fill(self,dR=0.05,**kwargs):
        rs,zs = self.offset(self.Rstore,self.Zstore,dR)
        dr_end = (rs[0]-self.R[0])/2
        dz_end = (zs[0]-self.Z[0])/2
        radius_end = np.sqrt(dr_end**2+dz_end**2)
        ro_end = self.R[0]+dr_end
        zo_end = self.Z[0]+dz_end
        theta_o = np.arctan2(zs[0]-zo_end,rs[0]-ro_end)
        theta_end = np.linspace(theta_o,theta_o+np.pi,20)
        r_end = ro_end+radius_end*np.cos(theta_end)
        z_end = zo_end+radius_end*np.sin(theta_end)
        r_sheild = np.append(rs[::-1],r_end[1:-1])
        r_sheild = np.append(r_sheild,self.R)
        z_sheild = np.append(zs[::-1],z_end[1:-1])
        z_sheild = np.append(z_sheild,self.Z)
        self.R,self.Z = r_sheild,z_sheild
        self.fill(trim=None,**kwargs)
        
    def VVfill(self,**kwargs):
        VVinR,VVinZ = self.R,self.Z
        self.fill(**kwargs)
        VVoutR,VVoutZ = self.R,self.Z
        with open('../Data/'+self.setup.configuration+'_VV.txt','w') as f:
            f.write('Rin m\t\tZin m\t\tRout m\t\tZout m\n')
            for rin,zin,rout,zout in zip(VVinR,VVinZ,VVoutR,VVoutZ):
                f.write('{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n'.format(\
                rin,zin,rout,zout))
        
    def blend(self,dt,ref_o=4/8*np.pi,dref=np.pi/4,offset=0,referance='theta'):
        if referance is 'theta':
            theta = np.arctan2(self.Z-self.xo[1],self.R-self.xo[0])-offset
            theta[theta>np.pi] = theta[theta>np.pi]-2*np.pi
            tblend = dt[0]*np.ones(len(theta))  # inner 
            tblend[(theta>-ref_o) & (theta<ref_o)] = dt[1]  # outer 
            if dref > 0:
                for updown in [-1,1]:
                    blend_index = (updown*theta>=ref_o) &\
                                    (updown*theta<ref_o+dref)
                    tblend[blend_index] = dt[1]+(dt[0]-dt[1])/dref*\
                                        (updown*theta[blend_index]-ref_o)
        else:
            L = self.length(self.R,self.Z)
            tblend = dt[0]*np.ones(len(L))  # start
            tblend[L>ref_o] = dt[1]  # end
            if dref > 0:
                blend_index = (L>=ref_o) & (L<ref_o+dref)
                tblend[blend_index] = dt[0]+(dt[1]-dt[0])/dref*(L[blend_index]-
                                                                ref_o)
        return tblend
        
    def wblend(self,N):
        L = np.linspace(-0.5,0.5,N)
        w = 0.5+np.sin(np.pi*L)/2
        return w
        
    def window(self,N,start,end,trans):
        L = np.linspace(0,1,N)
        w = np.ones(N)
        w[(L<start) | (L>end)] = 0
        index = (L>=start) & (L<start+trans)
        w[index] = 0.5+np.sin(np.pi*(L[index]-start)/trans-np.pi/2)/2
        index = (L<=end) & (L>end-trans)
        w[index] = 0.5+np.sin(np.pi*(end-L[index])/trans-np.pi/2)/2
        return w
        
    def wall_offset(self,Rbo,Zbo,R,Z,flip,Lblend=1.5):
        ib = np.argmin(Zbo)
        if flip == 1:
            Rb,Zb = Rbo[ib:],Zbo[ib:]
        else:
            Rb,Zb = Rbo[:ib],Zbo[:ib]
        L = []
        for r,z in zip(R,Z):
            L.append(np.min(np.sqrt((Rb-r)**2+(Zb-z)**2)))
        i = np.argmin(L)
        j = np.argmin((Rb-R[i])**2+(Zb-Z[i])**2)  
        if flip == 1:
            R,Z = R[:i][::-1],Z[:i][::-1]
            Rb,Zb = Rb[j:],Zb[j:]
            Rt,Zt = Rbo[ib:][:j],Zbo[ib:][:j]
        else:
            R,Z = R[i:],Z[i:]
            Rb,Zb = Rb[:j][::-1],Zb[:j][::-1]
            Rt,Zt = Rbo[:ib][j:][::-1],Zbo[:ib][j:][::-1]
        Lb = self.length(Rb,Zb,norm=False)
        L = self.length(R,Z,norm=False)
        N = np.argmin(abs(Lb-Lblend))
        w = self.wblend(N)
        Rc = (1-w)*Rb[:N]+w*interp1(L,R)(Lb[:N])
        Zc = (1-w)*Zb[:N]+w*interp1(L,Z)(Lb[:N])   
        N = np.argmin(abs(L-Lblend))
        Rc = np.append(Rc,R[N:])
        Zc = np.append(Zc,Z[N:])
        return Rc,Zc,Rt,Zt
    
    def fw_offset(self,Rb,Zb,dr=0.66):
        Rbo,Zbo = Rb[::-1],Zb[::-1]
        R,Z = self.offset(self.Rp,self.Zp,dr)
        R,Z = self.rzSLine(R,Z)
        R,Z,Rtout,Ztout = self.wall_offset(Rbo,Zbo,R,Z,-1)
        R,Z,Rtin,Ztin = self.wall_offset(Rbo,Zbo,R,Z,1)
        R,Z = np.append(Rtin,R),np.append(Ztin,Z)
        R,Z = np.append(R,Rtout[::-1]),np.append(Z,Ztout[::-1])   
        return R[::-1],Z[::-1]

        