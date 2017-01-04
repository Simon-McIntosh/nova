import numpy as np
import pylab as pl
import pickle
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline as spline
import seaborn as sns
colors = sns.color_palette('Paired',12)
from itertools import cycle
from amigo.geom import Loop
import amigo.geom as geom
from nova.loops import Profile,plot_oppvar
from nova.shape import Shape
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.optimize import minimize
from collections import OrderedDict


class RB(object):
    def __init__(self,setup,sf,npoints=500):
        self.setup = setup
        self.sf = sf
        self.npoints = npoints
        self.dataname = setup.configuration
        self.segment = {}  # store section segments (divertor,fw,blanket...)
        
    def update_sf(self):
        if not hasattr (self.sf,'Xpoint'):
            self.sf.get_Xpsi()  
        self.xo = [self.sf.Xpoint[0],self.sf.eqdsk['zmagx']]
        self.Rp,self.Zp = self.sf.get_boundary()
        self.Lp = geom.length(self.Rp,self.Zp,norm=False)[-1]
        self.Rfw,self.Zfw,self.psi_fw = self.firstwall_loop(self.setup.firstwall['dRfw']) 
        self.loop = Loop(self.Rfw,self.Zfw,xo=self.xo)  # first wall contour

    def set_firstwall(self,r,z):
        r,z = geom.order(r,z,anti=False)
        self.Rb,self.Zb = r,z
        
    def set_target(self,leg,**kwargs):
        if leg not in self.setup.targets:
            self.setup.targets[leg] = {}
        for key in self.setup.targets['default']:
            if key in kwargs:
                self.setup.targets[leg][key] = kwargs[key]  # update
            elif key not in self.setup.targets[leg]:  #  prevent overwrite
                self.setup.targets[leg][key] = self.setup.targets['default'][key]  # default
   
    def first_wall_psi(self,trim=True,**kwargs):
        if 'point' in kwargs:
            req,zeq = kwargs.get('point')
            psi = self.sf.Ppoint([req,zeq])
        else:
            req,zeq = self.sf.LFPr,self.sf.LFPz 
            if 'psi_n' in kwargs:  # normalized psi
                psi_n = kwargs.get('psi_n')
                psi = psi_n*(self.sf.Xpsi-self.sf.Mpsi)+self.sf.Mpsi
            elif 'psi' in kwargs:
                psi = kwargs.get('psi')
            else:
                raise ValueError('set point=(r,z) or psi in kwargs')
        contours = self.sf.get_contour([psi])    
        R,Z = self.sf.pick_contour(contours)
        min_contour = np.empty(len(R))
        for i in range(len(R)):
            min_contour[i] = np.min((R[i]-req)**2+(Z[i]-zeq)**2)
        imin = np.argmin(min_contour)
        r,z = R[imin],Z[imin]
        if trim:
            if self.sf.Xloc == 'lower':
                r,z = r[z<=zeq],z[z<=zeq]
            elif self.sf.Xloc == 'upper':
                r,z = r[z>=zeq],z[z>=zeq]
            else:
                raise ValueError('Xloc not set (get_Xpsi)')
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
  
    def firstwall_loop(self,dr):  # dr [m]
        if not hasattr(self.sf,'LFPr'):
            self.sf.get_LFP()
         
        r,z,psi = self.first_wall_psi(psi_n=1.07,trim=False)
        psi_lfs = psi_hfs = psi
        self.sf.LFfwr,self.sf.LFfwz = self.sf.LFPr,self.sf.LFPz
        self.sf.HFfwr,self.sf.HFfwz = self.sf.HFPr,self.sf.HFPz
        '''
        self.sf.LFfwr,self.sf.LFfwz = self.sf.LFPr+dr,self.sf.LFPz    
        self.sf.HFfwr,self.sf.HFfwz = self.sf.HFPr-dr,self.sf.HFPz
        r_lfs,z_lfs,psi_lfs = self.first_wall_psi(point=(self.sf.LFfwr,
                                                         self.sf.LFfwz))
        r_hfs,z_hfs,psi_hfs = self.first_wall_psi(point=(self.sf.HFfwr,
                                                         self.sf.HFfwz))
        r_top,z_top = geom.offset(self.Rp,self.Zp,dr)
        if self.sf.Xloc == 'lower':
            r_top,z_top = geom.theta_sort(r_top,z_top,xo=self.xo,origin='top')
            index = z_top>=self.sf.LFPz
        else:
            r_top,z_top = geom.theta_sort(r_top,z_top,xo=self.xo,origin='bottom')
            index = z_top<=self.sf.LFPz
        r_top,z_top = r_top[index],z_top[index]
        istart = np.argmin((r_top-self.sf.HFfwr)**2+(z_top-self.sf.HFfwz)**2)
        if istart > 0:
            r_top,z_top = r_top[::-1],z_top[::-1]
        r = np.append(r_hfs[::-1],r_top)
        r = np.append(r,r_lfs)
        z = np.append(z_hfs[::-1],z_top)
        z = np.append(z,z_lfs)
        '''
        pl.plot(r[::-1],z[::-1],'r')
        return r[::-1],z[::-1],(psi_lfs,psi_hfs) 
  
    def elipsoid(self,theta,ro,zo,A,k,delta):
        R = ro+ro/A*np.cos(theta+delta*np.sin(theta))
        Z = zo+ro/A*k*np.sin(theta)
        return (R,Z)

    def inflate(self,R,Z,dR):
        s = 1e-3
        L = geom.length(R,Z)
        nR,nZ = geom.normal(R,Z)
        Rdr,Zdr = R+dR*nR,Z+dR*nZ  # radial offset
        Rloop,Zloop = self.reorder(Rdr,Zdr)  # spline loop
        Lloop = geom.length(Rloop,Zloop)
        Rloop = interp1d(Lloop,Rloop)(L)  # respace
        Zloop = interp1d(Lloop,Zloop)(L)
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
         
    def coil_sheild(self,coils=[],dt=0.8):
        self.rzPut()
        for coil in coils:
            name = 'Coil'+str(coil)
            r,dr = self.sf.coil[name]['r'],self.sf.coil[name]['dr']
            z,dz = self.sf.coil[name]['z'],self.sf.coil[name]['dz']
            r = [r-dr/2,r-dr/2,r+dr/2,r+dr/2,r-dr/2]
            z = [z-dz/2,z+dz/2,z+dz/2,z-dz/2,z-dz/2]
            self.loop.R,self.loop.Z = self.rzInterp(r,z,npoints=100)
            self.loop.R,self.loop.Z = self.loop.R[::-1],self.loop.Z[::-1]
            self.loop.fill(dt=dt,loop=True,alpha=0.1,color='k',s=5e-3)
        self.rzGet()

    def rotateB(self,B,theta):
        Bmag = np.sqrt(B[0]**2+B[1]**2)
        R = np.matrix([[np.cos(theta),-np.sin(theta)],
                        [np.sin(theta),np.cos(theta)]])
        Bhat = (R*np.matrix([[B[0]],[B[1]]])).getA1()/Bmag
        return Bhat
   
    def firstwall(self,mode='calc',color=0.6*np.ones(3),plot=True,debug=False):
        self.update_sf()  # update streamfunction
        if mode == 'eqdsk':  # read first wall from eqdsk
            self.Rb,self.Zb = self.sf.xlim,self.sf.ylim
        elif mode == 'calc':
            self.Rb,self.Zb = self.target_fill(debug=debug)
            with open('../../Data/'+self.dataname+'_FW.pkl', 'wb') as output:
                pickle.dump(self.setup.targets,output,-1)
                pickle.dump(self.Rb,output,-1)
                pickle.dump(self.Zb,output,-1)
        elif mode == 'read':
            with open('../../Data/'+self.dataname+'_FW.pkl', 'rb') as input:
                self.setup.targets = pickle.load(input)
                self.Rb = pickle.load(input)
                self.Zb = pickle.load(input)
        else:
            errtxt = 'set input mode \'eqdsk\',\'calc\',\'read\''
            raise ValueError(errtxt)
        if mode != 'eqdsk':  # update eqdsk
            self.sf.nlim = len(self.Rb)  # update sf
            self.sf.xlim,self.sf.ylim = self.Rb,self.Zb
        self.loop.R,self.loop.Z = self.trim_contour(self.Rb,self.Zb) # update fw
        if plot:
            index = (self.Rb <= self.sf.r.max()) & (self.Rb >= self.sf.r.min()) &\
            (self.Zb <= self.sf.z.max()) & (self.Zb >= self.sf.z.min())
            pl.plot(self.Rb[index],self.Zb[index],
                    '-',color=color,linewidth=1.75)
       
    def get_fw(self,expand=0):  # generate boundary dict for elliptic
        if not hasattr(self,'Rb'):  # first wall not set
            self.firstwall(mode='eqdsk',plot=False,debug=False)
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
        index = np.zeros(2,dtype=int)
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
        self.sf.sol(update=True,plot=False)
        for leg in list(self.sf.legs)[2:]:
            self.set_target(leg)
            Rsol,Zsol = self.sf.snip(leg,layer_index,L2D=self.setup.targets[leg]['L2D'])
            Ro,Zo = Rsol[-1],Zsol[-1]
            Flip = [-1,1]
            Direction = [1,-1]
            Theta_end = [0,np.pi]
            theta_sign = 1
            
            if self.sf.Xloc == 'upper':
                theta_sign *= -1
                Direction = Direction[::-1]
            if 'inner' in leg:
                psi_plasma = self.psi_fw[1]
            else:
                psi_plasma = self.psi_fw[0]
            self.dpsi = self.psi_fw[1]-self.sf.Xpsi
            Phi_target = [psi_plasma,self.sf.Xpsi-self.setup.firstwall['div_ex']*self.dpsi]  
            if leg is 'inner1' or leg is 'outer2':
                Phi_target[0] = self.sf.Xpsi+self.setup.firstwall['div_ex']*self.dpsi
            if self.setup.targets[leg]['open']:
                theta_sign *= -1
                Direction = Direction[::-1]
                Theta_end = Theta_end[::-1]
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
            self.segment['divertor'] = {'r':Rb,'z':Zb}  # store diveror
            r,z = self.connect(self.psi_fw[1],['outer1','inner2'],[0,0],
                               loop=(self.loop.R[::-1],self.loop.Z[::-1]))
            Rb,Zb = self.append(Rb,Zb,r,z)
        else:
            Rb = np.append(Rb,self.setup.targets['inner']['R'][1:])
            Zb = np.append(Zb,self.setup.targets['inner']['Z'][1:])
            r,z = self.connect(self.sf.Xpsi-self.setup.firstwall['div_ex']*self.dpsi,
                               ['inner','outer'],[-1,0])                
            Rb,Zb = self.append(Rb,Zb,r,z)
            self.segment['divertor'] = {'r':Rb,'z':Zb}  # store diveror
            Rb = np.append(Rb,self.setup.targets['outer']['R'][1:])
            Zb = np.append(Zb,self.setup.targets['outer']['Z'][1:])
            self.segment['divertor'] = {'r':Rb,'z':Zb}  # store diveror
            r,z = self.connect(self.psi_fw[1],['outer','inner'],[-1,0],
                               loop=(self.loop.R,self.loop.Z))
            Rb,Zb = self.append(Rb,Zb,r,z)
        
        self.Rb,self.Zb = self.midplane_loop(Rb,Zb)

        pl.plot(self.Rb,self.Zb,'r')
        self.Rb,self.Zb = self.first_wall_fit(self.setup.firstwall['dRfw'])
        #self.blanket_fill()
        #self.vessel_fill()
        #self.get_sol()  # trim sol to targets and store values
        #self.write_boundary(self.Rb,self.Zb)
        return self.Rb,self.Zb
        
    def first_wall_fit(self,dr):
        profile = Profile(self.setup.configuration,family='S',part='fwf',
                          npoints=200)
        shp = Shape(profile,objective='L')
        #shp.loop.oppvar.remove('z2')
        shp.loop.oppvar.remove('flat')
        #shp.loop.set_symetric(True)
        shp.loop.set_l({'value':0.5,'lb':0.65,'ub':1.5})  # 1/tesion)
        shp.loop.xo['upper'] = {'value':0.33,'lb':0.75,'ub':1}  
        shp.loop.xo['lower'] = {'value':0.33,'lb':0.75,'ub':1}
        shp.loop.xo['top'] = {'value':0.33,'lb':0.05,'ub':0.5}  # horizontal shift
        shp.loop.xo['bottom'] = {'value':0.33,'lb':0.05,'ub':0.5}
        rfw,zfw = geom.offset(self.Rp,self.Zp,dr)  # offset from sep
        rfw,zfw = geom.rzSLine(rfw,zfw,100)  # sub-sample
        if self.sf.Xloc == 'lower':
            index = zfw > self.sf.Xpoint[1]+0.2  # remove x-point points
        else:
            index = zfw < self.sf.Xpoint[1]-0.2
        #rfw,zfw = rfw[index],zfw[index]
        shp.add_bound({'r':rfw,'z':zfw},'internal')  # vessel inner bounds
        shp.add_bound({'r':np.min(rfw)-0.001,'z':0},'interior')  # rmin
        
        index = self.Zb > self.sf.LFfwz
        rflux,zflux = self.Rb[index],self.Zb[index]
        rflux,zflux = geom.rzSLine(rflux,zflux,80)  # sub-sample
        shp.add_bound({'r':rflux,'z':zflux},'internal')  # flux inner bounds

        shp.minimise()
        #shp.loop.plot()
        #shp.plot_bounds()
        x = profile.loop.draw()
        r,z = geom.order(x['r'],x['z'],anti=True)
        istart = np.argmin((r-self.sf.Xpoint[0])**2+(z-self.sf.Xpoint[1])**2)
        r = np.append(r[istart:],r[:istart+1])
        z = np.append(z[istart:],z[:istart+1])
        r,z,l = geom.unique(r,z)
        self.segment['inner_loop'] = {'r':r,'z':z}
        rd,zd = self.segment['divertor']['r'],self.segment['divertor']['z']
        rd,zd = geom.unique(rd,zd)[:2]
        if self.sf.Xloc == 'lower':
            zindex = zd <= self.sf.Xpoint[1]+0.5*(self.sf.mo[1]-
                                                  self.sf.Xpoint[1])
        else:
            zindex = zd >= self.sf.Xpoint[1]+0.5*(self.sf.mo[1]-
                                                  self.sf.Xpoint[1])        
        rd,zd = rd[zindex],zd[zindex]  # remove upper points
        
        pl.plot(r,z,'b')
        pl.plot(rd,zd,'k--')
        print(np.shape(rd),np.shape(zd))
        rd,zd = geom.rzInterp(rd,zd)  # resample
        rd,zd = geom.inloop(r,z,rd,zd,side='out')  # divertor external to fw
        istart = np.argmin((r-rd[-1])**2+(z-zd[-1])**2)  # connect to fw
        iend = np.argmin((r-rd[0])**2+(z-zd[0])**2)
        r = np.append(np.append(rd,r[istart:iend]),rd[0])
        z = np.append(np.append(zd,z[istart:iend]),zd[0])
        self.segment['first_wall'] = {'r':r,'z':z} 
        return r,z
        
    def blanket_fill(self):
        blanket = wrap(self.segment['first_wall'],self.segment['inner_loop'])
        blanket.sort_zmin('inner')
        blanket.offset('outer',self.setup.build['BB'],
                       ref_o=3/10*np.pi,dref=np.pi/3)
        bfill = blanket.fill(color=colors[0])
        self.segment['blanket_fw']  = bfill[0]
        self.segment['blanket']  = bfill[1]

    def vessel_fill(self):
        r,z = self.segment['blanket_fw']['r'],self.segment['blanket_fw']['z']
        loop = geom.Loop(r,z)
        r,z = loop.fill(dt=0.05)
        rb = np.append(r,r[0])
        zb = np.append(z,z[0])
        profile = Profile(self.setup.configuration,family='S',part='vv',
                          npoints=400)
        shp = Shape(profile,objective='L')
        shp.loop.set_l({'value':0.5,'lb':0.6,'ub':1.5})  # 1/tesion)
        shp.loop.xo['upper'] = {'value':0.33,'lb':0.5,'ub':1}  
        shp.loop.xo['lower'] = {'value':0.33,'lb':0.5,'ub':1}
        #shp.loop.oppvar.remove('flat')
        r,z = geom.rzSLine(rb,zb,200)  # sub-sample
        rup,zup = r[z>self.sf.Xpoint[1]],z[z>self.sf.Xpoint[1]]
        shp.add_bound({'r':rup,'z':zup},'internal')  # vessel inner bounds
        rd,zd = r[z<self.sf.Xpoint[1]],z[z<self.sf.Xpoint[1]]
        ro,zo = geom.offset(rd,zd,0.1)  # divertor offset
        shp.add_bound({'r':rd,'z':zd},'internal')  # vessel inner bounds
        shp.add_bound({'r':rd,'z':zd-0.25},'internal')  # gap below divertor
        #shp.plot_bounds()
        shp.minimise()
        #shp.loop.plot()
        x = profile.loop.draw()
        rin,zin = x['r'],x['z']
        loop = geom.Loop(rin,zin)
        r,z = loop.fill(dt=self.setup.build['VV'],ref_o=2/8*np.pi,dref=np.pi/6)
        shp.clear_bound()
        shp.add_bound({'r':r,'z':z},'internal')  # vessel outer bounds
        shp.minimise()
        x = profile.loop.draw()
        r,z = x['r'],x['z']
        #vv = wrap({'r':rb,'z':zb},{'r':r,'z':z})
        vv = wrap({'r':rin,'z':zin},{'r':r,'z':z})
        vv.sort_zmin('inner')
        vv.sort_zmin('outer')
        vv.fill(color=colors[1])
        self.segment['vessel_inner'] = {'r':rin,'z':zin}
        self.segment['vessel_outer'] = {'r':r,'z':z}

    def append(self,R,Z,r,z):
        dr = np.zeros(2)
        for i,end in enumerate([0,-1]):
            dr[i] = (R[-1]-r[end])**2+(Z[-1]-z[end])**2
        if dr[1] < dr[0]:
            r,z = r[::-1],z[::-1]
        return np.append(R,r[1:-1]),np.append(Z,z[1:-1]) 
        
    def get_sol(self,plot=False):
        self.trim_sol(plot=plot)
        for leg in list(self.sf.legs)[2:]:
            L2D,L3D,Rsol,Zsol = self.sf.connection(leg,0)
            Ro,Zo = Rsol[-1],Zsol[-1]
            L2Dedge,L3Dedge = self.sf.connection(leg,-1)[:2]
            if leg not in self.setup.targets:
                self.setup.targets[leg] = {}
            Xi = self.sf.expansion([Ro],[Zo])
            graze,theta = np.zeros(self.sf.Nsol),np.zeros(self.sf.Nsol)
            pannel = self.sf.legs[leg]['pannel']
            for i in range(self.sf.Nsol):
                ro = self.sf.legs[leg]['R'][i][-1]
                zo = self.sf.legs[leg]['Z'][i][-1]
                graze[i] = self.sf.get_graze((ro,zo),pannel[i])
                theta[i] = self.sf.strike_point(Xi,graze[i])                             
            self.setup.targets[leg]['graze_deg'] = graze*180/np.pi
            self.setup.targets[leg]['theta_deg'] = theta*180/np.pi
            self.setup.targets[leg]['L2Do'] = L2D
            self.setup.targets[leg]['L3Do'] = L3D
            self.setup.targets[leg]['L2Dedge'] = L2Dedge
            self.setup.targets[leg]['L3Dedge'] = L3Dedge
            self.setup.targets[leg]['Ro'] = Ro
            self.setup.targets[leg]['Zo'] = Zo 
            self.setup.targets[leg]['Rsol'] = Rsol
            self.setup.targets[leg]['Zsol'] = Zsol 
        
    def trim_sol(self,color='k',plot=True):
        self.sf.sol()
        color = sns.color_palette('Set2',self.sf.nleg+5)
        #color = 'k'
        for c,leg in enumerate(self.sf.legs.keys()):
            if 'core' not in leg:
                Rsol = self.sf.legs[leg]['R']
                Zsol = self.sf.legs[leg]['Z']
                self.sf.legs[leg]['pannel'] = [[] for i in range(self.sf.Nsol)]
                for i in range(self.sf.Nsol):
                    if len(Rsol[i]) > 0:
                        R,Z = Rsol[i],Zsol[i]
                        for j in range(2):  # predict - correct
                            R,Z,pannel = self.inloop(self.Rb,self.Zb,R,Z)
                        self.sf.legs[leg]['R'][i] = R  # update sf
                        self.sf.legs[leg]['Z'][i] = Z
                        self.sf.legs[leg]['pannel'][i] = pannel
                        if plot:
                            if color != 'k' and i > 0:
                                pl.plot(R,Z,'-',color=0.7*np.ones(3))  # color[c+3]
                            elif color == 'k':
                                pl.plot(R,Z,'-',color='k',alpha=0.15)
                            else:
                                #pl.plot(R,Z,color=color[c])
                                pl.plot(R,Z,'--',color=[0.5,0.5,0.5])

    def trim_contour(self,Rb,Zb):
        Rt,Zt = np.array([]),np.array([])  # trim contour for BB
        Lnorm = geom.length(self.loop.R,self.loop.Z,norm=False)[-1]
        for flip,trim in zip([1,-1],[0.275*self.setup.firstwall['trim'][1],
                              0.725*self.setup.firstwall['trim'][0]]):
            L = geom.length(Rb[::flip],Zb[::flip],norm=False)
            i = np.argmin(np.abs(L/Lnorm-trim))
            Rt = np.append(Rt,Rb[::flip][:i][::-flip])
            Zt = np.append(Zt,Zb[::flip][:i][::-flip])
        Rt,Zt = geom.rzSLine(Rt,Zt,npoints=self.npoints)
        return Rt,Zt

    def midplane_loop(self,Rb,Zb):
        index = np.argmin((Rb-self.sf.LFfwr)**2+(Zb-self.sf.LFfwz)**2)
        Rb = np.append(Rb[:index+1][::-1],Rb[index:][::-1])
        Zb = np.append(Zb[:index+1][::-1],Zb[index:][::-1])
        Lb = geom.length(Rb,Zb)
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
        with open('../../Data/'+self.dataname+'_bdry.txt','w') as f:
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
                f.write('{:1.2f}\t\t'.format(target['graze_deg'][0]))
                if target['open']:
                    f.write('open\t')
                else:
                    f.write('closed\t')
                f.write('{:1.6f}\t'.format(target['Ro']))
                f.write('{:1.6f}\t'.format(target['Zo']))
                f.write('{:1.2f}\t'.format(target['theta_deg'][0]))
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
        Rloop,Zloop = geom.order(Rloop,Zloop)
        L = geom.length(R,Z)
        index = np.append(np.diff(L)!=0,True)
        R,Z = R[index],Z[index]  # remove duplicates
        nRloop,nZloop,Rloop,Zloop = geom.normal(Rloop,Zloop)
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
        return Rin,Zin,db
        
    def get_bstep(self,s,ds,b,Rloop,Zloop,j):
        db = np.array([Rloop[j[1]]-Rloop[j[0]],Zloop[j[1]]-Zloop[j[0]]])
        step = np.cross(b-s,ds)/np.cross(ds,db)
        return step,db

    def match_psi(self,Ro,Zo,direction,theta_end,theta_sign,phi_target,graze,
                  dPlate,leg,debug=False):        
        color = sns.color_palette('Set2',2)
        gain = 0.15  # 0.25
        Nmax = 500
        Lo = [0.5,0.005]  # [blend,turn]  5,0.015
        r2m = [-1.5,-1]  # ramp to step (+ive-lead, -ive-lag ramp==1, step==inf)
        Nplate = 1 # number of target plate divisions (1==flat)
        L = Lo[0] if theta_end == 0 else Lo[1]
        Lsead = L
        flag = 0
        for i in range(Nmax):
            R,Z,phi = self.blend_target(Ro,Zo,dPlate,L,direction,theta_end,
                                        theta_sign,graze,r2m,Nplate)
            L -= gain*(phi_target-phi)
            if debug: pl.plot(R,Z,color=color[0],lw=1)
            if np.abs(phi_target-phi) < 1e-4:
                if debug: 
                    pl.plot(R,Z,'r',color=color[1],lw=3)
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
            R,Z = geom.rzInterp(R,Z,Ninterp)
            graze = self.sf.get_graze([R[-1],Z[-1]],
                                      [R[-1]-R[-2],Z[-1]-Z[-2]])  # update graze
            N = np.int(L/dL+1)
            if N<30: N=30
            dL = L/(N-1)
            target_angle = np.arctan2((Z[-1]-Z[-2]),(R[-1]-R[-2]))
            Xi = self.sf.expansion([R[-1]],[Z[-1]])
            theta = self.sf.strike_point(Xi,graze)
            B = direction*self.sf.Bpoint((R[-1],Z[-1]))
            Bhat = self.rotateB(B,theta_sign*theta)
            trans_angle = np.arctan2(Bhat[1],Bhat[0])
            if abs(target_angle-trans_angle) > 0.01*np.pi:
                accute = True
            else:
                accute = False
            R,Z = self.extend_target(R,Z,dL,N,r2s,theta_end,theta_sign,
                                direction,graze,accute)  # transition graze
            phi = self.sf.Ppoint([R[-1],Z[-1]])
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
            B = direction*self.sf.Bpoint((R[-1],Z[-1]))
            Bhat = self.rotateB(B,theta_sign*theta)
            R = np.append(R,R[-1]+dL*Bhat[0])
            Z = np.append(Z,Z[-1]+dL*Bhat[1])
        return R,Z
        
    def vessel_fit(self):
        profile = Profile(self.setup.configuration,family='S',part='VV')
        shp = Shape(profile,objective='L')
        #shp.loop.set_l({'value':0.8,'lb':0.5,'ub':0.9})
        shp.loop.oppvar.remove('z2')
        #shp.loop.oppvar.remove('tilt')
        shp.loop.set_symetric(True)
        rvv,zvv = geom.rzSLine(self.loop.R,self.loop.Z,30)
        rvv,zvv = geom.offset(rvv,zvv,0.05)  # offset from blanket
        shp.add_bound({'r':rvv,'z':zvv},'internal')  # vessel inner bounds
        shp.minimise()
        x = profile.loop.draw()
        self.loop.R,self.loop.Z = x['r'],x['z']
        
    def vessel(self):  # radial build from firstwall
        Color = cycle(sns.color_palette('Set2',12)[1:])
        self.FWfill(dt=self.setup.build['tfw'],color=next(Color))  # first wall 
        self.loop.fill(dt=self.setup.build['BB'][::-1],alpha=0.7,ref_o=0.3,dref=0.2,
                  referance='geom.length',color=next(Color))  # blanket+sheild
        self.loop.fill(dt=self.setup.build['tBBsupport'],alpha=0.7,
                  color=next(Color))  # blanket support
        self.sheild_fill(dt=self.setup.build['sheild'],
                           ref_o=0.35*np.pi,dref=0.2*np.pi,
                           gap=1/10*np.pi,alpha=0.7,color=next(Color))
        self.vessel_fit()
        self.VVfill(dt=self.setup.build['VV'],ref_o=0.25*np.pi,  # vessel
                    dref=0.25*np.pi,gap=0.5/10*np.pi,alpha=0.7,
                    loop=True,color=next(Color))  # ref_o=0.385
        pl.axis('equal')
        
    def FWfill(self,**kwargs):
        if not hasattr(self,'Rb'):  # first wall not set
            self.firstwall(mode='eqdsk',plot=False,debug=False)  # load fw
        self.loop.R,self.loop.Z = geom.rzSLine(self.Rb[::-1],self.Zb[::-1],
                                     npoints=self.npoints,Hres=False)
        self.loop.fill(loop=True,alpha=0.7,**kwargs)
        self.loop.R,self.loop.Z = geom.rzSLine(self.loop.R,self.loop.Z,npoints=self.npoints)
        self.loop.R,self.loop.Z = self.trim_contour(self.loop.R[::-1],self.loop.Z[::-1]) # update 

    def sheild_fill(self,**kwargs):  # 
        if 'sheild_connect' in self.setup.build:
            connect = self.setup.build['sheild_connect']
        else:
            connect=[0,1]
        index = self.loop.trim(connect,self.loop.R,self.loop.Z)
        rVV = self.loop.R[index[0]:index[1]+1]
        zVV = self.loop.Z[index[0]:index[1]+1]
        ni = 10  #  number of trim nodes
        rp,zp = rVV[-ni:],zVV[-ni:] 
        if self.setup.build['Dsheild']:
            Rleg,Zleg = [],[]
            Rd = self.Rb[self.Zb<self.sf.Xpoint[1]][::-1]
            Zd = self.Zb[self.Zb<self.sf.Xpoint[1]][::-1]
            Ld = geom.length(Rd,Zd)
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
                R,Z = geom.offset(Rt,Zt,self.setup.build['tfw'])
                dRt = np.mean((Rt-self.sf.Xpoint[0])**2+
                              (Zt-self.sf.Xpoint[1])**2)
                dR = np.mean((R-self.sf.Xpoint[0])**2+
                              (Z-self.sf.Xpoint[1])**2)
                if  dR<dRt:
                    Rt,Zt = geom.offset(Rt,Zt,-self.setup.build['tfw'])
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
            
            L = geom.length(self.Rb,self.Zb,norm=False)[-1]
            Rt,Zt = geom.offset(self.Rb,self.Zb,self.setup.build['tfw'])
            Lt = geom.length(Rt,Zt,norm=False)[-1]
            if Lt<L:
               Rt,Zt = geom.offset(self.Rb,self.Zb,-self.setup.build['tfw']) 
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
        rs,zs = geom.rzSLine(rp,zp,npoints=260,s=5e-6)
        R,Z = np.append(rVV[ni+1:-ni],rs),np.append(zVV[ni+1:-ni],zs)
        R,Z = np.append(R,R[0]),np.append(Z,Z[0])
        self.loop.R,self.loop.Z = geom.rzSLine(R,Z,len(R),s=5e-4)
        #self.loop.fill(trim=None,loop=True,s=2e-4,**kwargs)
 
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
        
        self.loop.R,self.loop.Z = rs,zs
        self.rzPut()
        self.loop.fill(trim=None,**kwargs)
        
        rs,zs = geom.offset(rs,zs,kwargs['dR'])
        return rs,zs

    def VVfill(self,**kwargs):
        VVinR,VVinZ = self.loop.R,self.loop.Z
        self.loop.fill(**kwargs)
        VVoutR,VVoutZ = self.loop.R,self.loop.Z
        with open('../../Data/'+self.setup.configuration+'_VV.txt','w') as f:
            f.write('Rin m\t\tZin m\t\tRout m\t\tZout m\n')
            for rin,zin,rout,zout in zip(VVinR,VVinZ,VVoutR,VVoutZ):
                f.write('{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n'.format(\
                rin,zin,rout,zout))
 
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
        Lb = geom.length(Rb,Zb,norm=False)
        L = geom.length(R,Z,norm=False)
        N = np.argmin(abs(Lb-Lblend))
        w = self.wblend(N)
        Rc = (1-w)*Rb[:N]+w*interp1d(L,R)(Lb[:N])
        Zc = (1-w)*Zb[:N]+w*interp1d(L,Z)(Lb[:N])   
        N = np.argmin(abs(L-Lblend))
        Rc = np.append(Rc,R[N:])
        Zc = np.append(Zc,Z[N:])
        return Rc,Zc,Rt,Zt
    
    def fw_offset(self,Rb,Zb,dr=0.66):
        Rbo,Zbo = Rb[::-1],Zb[::-1]
        R,Z = geom.offset(self.Rp,self.Zp,dr)
        R,Z = geom.rzSLine(R,Z)
        R,Z,Rtout,Ztout = self.wall_offset(Rbo,Zbo,R,Z,-1)
        R,Z,Rtin,Ztin = self.wall_offset(Rbo,Zbo,R,Z,1)
        R,Z = np.append(Rtin,R),np.append(Ztin,Z)
        R,Z = np.append(R,Rtout[::-1]),np.append(Z,Ztout[::-1])   
        return R[::-1],Z[::-1]


class wrap(object):
    
    def __init__(self,inner_points,outer_points):
        self.loops = OrderedDict()
        self.loops['inner'] = {'points':inner_points}
        self.loops['outer'] = {'points':outer_points}
        
    def get_segment(self,loop):
        segment = self.loops[loop]['points']
        return segment['r'],segment['z']
        
    def set_segment(self,loop,r,z):
        self.loops[loop]['points'] = {'r':r,'z':z}
        
    def sort_zmin(self,loop):
        r,z = self.get_segment(loop)
        r,z = geom.order(r,z,anti=True)  # order points
        imin = np.argmin(z)  # locate minimum
        r = np.append(r[imin:],r[:imin])  # sort
        z = np.append(z[imin:],z[:imin])
        self.set_segment(loop,r,z)

    def offset(self,loop,dt,**kwargs):
        r,z = self.get_segment(loop)
        gloop = geom.Loop(r,z)
        r,z = gloop.fill(dt=dt,**kwargs)
        self.set_segment(loop,r,z)
           
    def interpolate(self):
        for loop in self.loops:
            r,z = self.get_segment(loop)
            r,z,l = geom.unique(r,z)
            interpolant = {'r':IUS(l,r),'z':IUS(l,z)}
            self.loops[loop]['fun'] = interpolant  

    def interp(self,loop,l):
        interpolant = self.loops[loop]['fun']
        return interpolant['r'](l),interpolant['z'](l)

    def sead(self,dl,N=50): # course search
        l = np.linspace(dl[0],dl[1],N)
        r,z = np.zeros((N,2)),np.zeros((N,2))
        for i,loop in enumerate(self.loops):
            r[:,i],z[:,i] = self.interp(loop,l)
        dr_min,i_in,i_out = np.max(r[:,1]),0,0
        for i,(rin,zin) in enumerate(zip(r[:,0],z[:,0])):
            dR = np.sqrt((r[:,1]-rin)**2+(z[:,1]-zin)**2)
            dr = np.min(dR)
            if dr < dr_min:
                dr_min = dr
                i_in = i
                i_out = np.argmin(dR)
        return l[i_in],l[i_out]
    
    def cross(self,L):
        r,z = np.zeros(2),np.zeros(2)
        for i,(loop,l) in enumerate(zip(self.loops,L)):
            r[i],z[i] = self.interp(loop,l)
        err = (r[0]-r[1])**2+(z[0]-z[1])**2
        return err
        
    def index(self,loop,l):
        rp,zp = self.interp(loop,l)  # point
        r,z = self.get_segment(loop)
        i = np.argmin((r-rp)**2+(z-zp)**2)
        return i
        
    def close_loop(self):
        for loop in self.loops:
            r,z = self.get_segment(loop)
            if (r[0]-r[-1])**2+(z[0]-z[-1])**2 != 0:
                r = np.append(r,r[0])
                z = np.append(z,z[0])
                self.set_segment(loop,r,z)
        
    def concentric(self,rin,zin,rout,zout):
        points = geom.inloop(rout,zout,rin,zin,side='out')
        if np.shape(points)[1] == 0:
            return True
        else:
            return False
        
    def fill(self,plot=True,color=colors[0]):  # minimization focused search
        rin,zin = self.get_segment('inner')
        rout,zout = self.get_segment('outer')        
        concentric = self.concentric(rin,zin,rout,zout)
        if concentric:
            self.close_loop()
            rin,zin = self.get_segment('inner')
            rout,zout = self.get_segment('outer') 
        self.interpolate()  # construct interpolators
        self.indx = {'inner':np.array([0,len(rin)],dtype=int),
                     'outer':np.array([0,len(rout)],dtype=int)}
        if not concentric:
            self.indx = {'inner':np.zeros(2,dtype=int),
                         'outer':np.zeros(2,dtype=int)}
            for i,dl in enumerate([[0,0.5],[0.5,1]]):  # low feild / high feild
                lo = self.sead(dl)
                L = minimize(self.cross,lo,method='L-BFGS-B',
                             bounds=([0,1],[0,1])).x 
                for loop,l in zip(self.loops,L):
                    self.indx[loop][i] = self.index(loop,l)

        r = np.append(rout[self.indx['outer'][0]:self.indx['outer'][1]],
                      rin[self.indx['inner'][0]:self.indx['inner'][1]][::-1])
        z = np.append(zout[self.indx['outer'][0]:self.indx['outer'][1]],
                      zin[self.indx['inner'][0]:self.indx['inner'][1]][::-1])
        self.fill = {'r':r,'z':z}
        r = np.append(np.append(rin[:self.indx['inner'][0]],
                      rout[self.indx['outer'][0]:self.indx['outer'][1]]),
                      rin[self.indx['inner'][1]:])
        z = np.append(np.append(zin[:self.indx['inner'][0]],
                      zout[self.indx['outer'][0]:self.indx['outer'][1]]),
                      zin[self.indx['inner'][1]:])
        self.segment = {'r':r,'z':z}
        if plot:
            self.plot(color=color)
        return self.segment,self.fill
            
    def plot(self,plot_loops=False,color=colors[0]):
        geom.polyfill(self.fill['r'],self.fill['z'],color=color)
        if plot_loops:
            pl.plot(self.segment['r'],self.segment['z'],color=0.75*np.ones(3))
            for loop in self.loops:
                r,z = self.get_segment(loop)
                pl.plot(r,z,'-')
