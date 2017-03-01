import numpy as np
from amigo.geom import Loop
import amigo.geom as geom
from nova.loops import Profile
from nova.shape import Shape
import pylab as pl
from warnings import warn


class targets(object):
    
    def __init__(self,sf,targets,**kwargs):
        self.sf = sf
        self.targets = targets  # target definition (graze, L2D, etc)
        self.dRfw = kwargs.get('dRfw',0.225)
        self.div_ex = kwargs.get('div_ex',0.25)
        
    def set_target(self,leg,**kwargs):
        if leg not in self.targets:
            self.targets[leg] = {}
        for key in self.targets['default']:
            if key in kwargs:
                self.targets[leg][key] = kwargs[key]  # update
            elif key not in self.targets[leg]:  #  prevent overwrite
                self.targets[leg][key] = self.targets['default'][key]
    
    def place(self,debug=False,**kwargs):
        layer_index = 0  # construct from first open flux surface
        symetric = kwargs.get('symetric',False)
        self.sf.sol(update=True,plot=False,debug=debug)
        for leg in list(self.sf.legs)[2:]:
            self.set_target(leg)
            Rsol,Zsol = self.sf.snip(leg,layer_index,
                                     L2D=self.targets[leg]['L2D'])
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
            dpsi = self.psi_fw[1]-self.sf.Xpsi
            dpsi = self.psi_fw[1]-self.sf.Xpsi                  
            Phi_target = [psi_plasma,self.sf.Xpsi-self.div_ex*dpsi]  
            if leg is 'inner1' or leg is 'outer2':
                Phi_target[0] = self.sf.Xpsi+self.div_ex*dpsi
            if self.targets[leg]['open']:
                theta_sign *= -1
                Direction = Direction[::-1]
                Theta_end = Theta_end[::-1]
            if 'outer' in leg:
                Direction = Direction[::-1]
                theta_sign *= -1
            if leg is 'inner1' or leg is 'outer2':
                Theta_end = Theta_end[::-1]

            self.targets[leg]['R'] = np.array([])
            self.targets[leg]['Z'] = np.array([])
            dPlate = self.targets[leg]['dPlate']
            for flip,direction,theta_end,psi_target\
            in zip(Flip,Direction,Theta_end,Phi_target):
                R,Z = self.match_psi(Ro,Zo,direction,theta_end,theta_sign,
                                     psi_target,self.targets[leg]['graze'],
                                     dPlate,leg,debug=debug)
                self.targets[leg]['R'] = np.append(self.targets[leg]['R'],
                                                   R[::flip])
                self.targets[leg]['Z'] = np.append(self.targets[leg]['Z'],
                                                   Z[::flip])
            if leg is 'outer':
                self.targets[leg]['R'] = self.targets[leg]['R'][::-1]
                self.targets[leg]['Z'] = self.targets[leg]['Z'][::-1]
        Rb,Zb = np.array([]),np.array([])

        if self.sf.nleg == 6:  # SF
            Rb = np.append(Rb,self.targets['inner2']['R'][1:])
            Zb = np.append(Zb,self.targets['inner2']['Z'][1:])
            r,z = self.connect(self.sf.Xpsi-\
            self.div_ex*dpsi,['inner2','inner1'],[-1,-1])
            Rb,Zb = self.append(Rb,Zb,r,z)
            Rb = np.append(Rb,self.targets['inner1']['R'][::-1])
            Zb = np.append(Zb,self.targets['inner1']['Z'][::-1])
            r,z = self.connect(self.sf.Xpsi+\
            self.div_ex*dpsi,['inner1','outer2'],[0,0])            
            Rb,Zb = self.append(Rb,Zb,r,z)
            Rb = np.append(Rb,self.targets['outer2']['R'][1:])
            Zb = np.append(Zb,self.targets['outer2']['Z'][1:])
            r,z = self.connect(self.sf.Xpsi-\
            self.div_ex*dpsi,['outer2','outer1'],[-1,-1])            
            Rb,Zb = self.append(Rb,Zb,r,z)
            Rb = np.append(Rb,self.targets['outer1']['R'][::-1])
            Zb = np.append(Zb,self.targets['outer1']['Z'][::-1])
            self.segment['divertor'] = {'r':Rb,'z':Zb}  # store diveror
            r,z = self.connect(self.psi_fw[1],['outer1','inner2'],[0,0],
                               loop=(self.loop.R[::-1],self.loop.Z[::-1]))
            Rb,Zb = self.append(Rb,Zb,r,z)
        else:
            Rb = np.append(Rb,self.targets['inner']['R'][1:])
            Zb = np.append(Zb,self.targets['inner']['Z'][1:])
            r,z = self.connect(self.sf.Xpsi-\
            self.div_ex*dpsi,['inner','outer'],[-1,0])                
            Rb,Zb = self.append(Rb,Zb,r,z)
            self.segment['divertor'] = {'r':Rb,'z':Zb}  # store diveror
            Rb = np.append(Rb,self.targets['outer']['R'][1:])
            Zb = np.append(Zb,self.targets['outer']['Z'][1:])
            self.segment['divertor'] = {'r':Rb,'z':Zb}  # store diveror
            r,z = self.connect(self.psi_fw[1],['outer','inner'],[-1,0],
                               loop=(self.loop.R,self.loop.Z))
            Rb,Zb = self.append(Rb,Zb,r,z)
        self.Rb,self.Zb = self.midplane_loop(Rb,Zb)
        self.Rb,self.Zb = self.first_wall_fit(self.dRfw,symetric=symetric,
                                              debug=debug)
        #self.get_sol()  # trim sol to targets and store values
        #self.write_boundary(self.Rb,self.Zb)
        return self.Rb,self.Zb
 

class firstwall(object):
    
    def __init__(self,name,sf,dr=0.225,psi_n=1.07,conformal=False,
                 flux_fit=False,debug=False,symetric=False):
        self.name = name  # first wall chamber name
        sf = self.set_sf(sf)
        self.dr = dr  # geometric offset
        self.flux_fit = flux_fit
        self.psi_n = psi_n
        self.debug = debug
        self.Nsub = 100  # loop sub-sample point number
        
        self.initalise_loop(symetric=symetric)
        for sf in self.set_sf(sf):  # convert input to list
            self.add_bound(sf)
        self.shp.add_internal(r_gap=0.001)  # add internal bound
        
        self.fit(debug)
        
    def set_sf(self,sf):
        if not isinstance(sf,list):  # convert sf input to list
            sf = [sf]
        return sf
    
    def initalise_loop(self,symetric=False):
        self.profile = Profile(self.name,family='S',part='fw',
                               npoints=200,symetric=symetric)
        self.shp = Shape(self.profile,objective='L')
        # adjust parameter limits for spline loop
        self.shp.loop.adjust_xo('upper',lb=0.7)
        self.shp.loop.adjust_xo('top',lb=0.05,ub=0.75)
        self.shp.loop.adjust_xo('lower',lb=0.7)
        self.shp.loop.adjust_xo('bottom',lb=0.05,ub=0.75)
        self.shp.loop.adjust_xo('l',lb=0.8,ub=1.5)

    def add_bound(self,sf): 
        rpl,zpl = sf.get_boundary()  # boundary points
        rpl,zpl = geom.offset(rpl,zpl,self.dr)  # offset from sep
        index = zpl > sf.Xpoint[1]  # trim to Xpoint
        rpl,zpl = rpl[index],zpl[index]
        rpl,zpl = geom.rzSLine(rpl,zpl,self.Nsub)  # sub-sample
        self.shp.add_bound({'r':rpl,'z':zpl},'internal')  # vessel inner bounds

        if self.flux_fit:  # add flux boundary points
            sf.get_LFP()  # get low feild point
            rflux,zflux = self.first_wall_psi(sf,psi_n=self.psi_n,trim=False)[:2]
            rflux,zflux = self.midplane_loop(sf,rflux,zflux)
            rflux,zflux = geom.order(rflux,zflux)
            istop = next((i for i in range(len(zflux)) if zflux[i]<sf.LFPz),-1)
            rflux,zflux = rflux[:istop],zflux[:istop]
            dL = np.diff(geom.length(rflux,zflux))
            if np.max(dL) > 3*np.median(dL) or \
            np.argmax(zflux) == len(zflux)-1:
                wtxt = '\n\nOpen feild line detected\n'
                wtxt += 'disabling flux fit for '
                wtxt += '{:1.1f}% psi_n \n'.format(1e2*self.psi_n)
                wtxt += 'configuration: '+sf.filename+'\n'
                warn(wtxt)
            else:  # add flux_fit bounds
                rflux,zflux = geom.rzSLine(rflux,zflux,int(self.Nsub/2))  
                self.shp.add_bound({'r':rflux,'z':zflux},'internal')  
  
    def midplane_loop(self,sf,r,z):
        index = np.argmin((r-sf.LFPr)**2+(z-sf.LFPz)**2)
        if z[index] <= sf.LFPz:
            index -= 1
        r = np.append(r[:index+1][::-1],r[index:][::-1])
        z = np.append(z[:index+1][::-1],z[index:][::-1])
        L = geom.length(r,z)
        index = np.append(np.diff(L)!=0,True)
        r,z = r[index],z[index]  # remove duplicates
        return r,z

    def fit(self,debug):
        self.shp.minimise()
        if debug:
            self.shp.loop.plot()
            self.shp.plot_bounds()
        x = self.profile.loop.draw()
        r,z = x['r'],x['z']
        r,z = geom.order(r,z,anti=True)
            
        pl.plot(r,z)
        
    def first_wall_psi(self,sf,trim=True,single_contour=False,**kwargs):
        if 'point' in kwargs:
            req,zeq = kwargs.get('point')
            psi = sf.Ppoint([req,zeq])
        else:
            req,zeq = sf.LFPr,sf.LFPz 
            if 'psi_n' in kwargs:  # normalized psi
                psi_n = kwargs.get('psi_n')
                psi = psi_n*(sf.Xpsi-sf.Mpsi)+sf.Mpsi
            elif 'psi' in kwargs:
                psi = kwargs.get('psi')
            else:
                raise ValueError('set point=(r,z) or psi in kwargs')
        contours = sf.get_contour([psi])    
        R,Z = sf.pick_contour(contours,Xpoint=True)
        if single_contour:
            min_contour = np.empty(len(R))
            for i in range(len(R)):
                min_contour[i] = np.min((R[i]-req)**2+(Z[i]-zeq)**2)
            imin = np.argmin(min_contour)
            r,z = R[imin],Z[imin]
        else:
            r,z = np.array([]),np.array([])
            for i in range(len(R)):
                r = np.append(r,R[i])
                z = np.append(z,Z[i])
        if trim:
            if sf.Xloc == 'lower':
                r,z = r[z<=zeq],z[z<=zeq]
            elif sf.Xloc == 'upper':
                r,z = r[z>=zeq],z[z>=zeq]
            else:
                raise ValueError('Xloc not set (get_Xpsi)')
            if req > sf.Xpoint[0]:
                r,z = r[r>sf.Xpoint[0]],z[r>sf.Xpoint[0]]
            else:
                r,z = r[r<sf.Xpoint[0]],z[r<sf.Xpoint[0]]
            istart = np.argmin((r-req)**2+(z-zeq)**2)
            r = np.append(r[istart+1:],r[:istart])
            z = np.append(z[istart+1:],z[:istart])
        istart = np.argmin((r-req)**2+(z-zeq)**2)
        if istart > 0:
            r,z = r[::-1],z[::-1]
        return r,z,psi
            
    '''
    def firstwall_loop(self,sf): 
        if not hasattr(sf,'LFPr'):
            sf.get_LFP()
        if self.flux_fit:
            r,z,psi = self.first_wall_psi(sf,psi_n=self.psi_n,trim=False)
            psi_lfs = psi_hfs = psi
            sf.LFfwr,sf.LFfwz = self.sf.LFPr,self.sf.LFPz
            sf.HFfwr,sf.HFfwz = self.sf.HFPr,self.sf.HFPz
            #pl.plot(r[::-1],z[::-1],'--',color=0.75*np.ones(3))
        else:
            dr = self.setup.firstwall['dRfw'] # dr [m]
            self.sf.LFfwr,self.sf.LFfwz = self.sf.LFPr+dr,self.sf.LFPz    
            self.sf.HFfwr,self.sf.HFfwz = self.sf.HFPr-dr,self.sf.HFPz
            r_lfs,z_lfs,psi_lfs = self.first_wall_psi(point=(self.sf.LFfwr,
                                                             self.sf.LFfwz))
            r_hfs,z_hfs,psi_hfs = self.first_wall_psi(point=(self.sf.HFfwr,
                                                             self.sf.HFfwz))
            r_top,z_top = geom.offset(self.Rp,self.Zp,dr)
            if self.sf.Xloc == 'lower':
                r_top,z_top = geom.theta_sort(r_top,z_top,xo=self.xo,
                                              origin='top')
                index = z_top>=self.sf.LFPz
            else:
                r_top,z_top = geom.theta_sort(r_top,z_top,xo=self.xo,
                                              origin='bottom')
                index = z_top<=self.sf.LFPz
            r_top,z_top = r_top[index],z_top[index]
            istart = np.argmin((r_top-self.sf.HFfwr)**2+
                               (z_top-self.sf.HFfwz)**2)
            if istart > 0:
                r_top,z_top = r_top[::-1],z_top[::-1]
            r = np.append(r_hfs[::-1],r_top)
            r = np.append(r,r_lfs)
            z = np.append(z_hfs[::-1],z_top)
            z = np.append(z,z_lfs)
        return r[::-1],z[::-1],(psi_lfs,psi_hfs)         
    '''
'''
    def update_sf(self):
        if not hasattr (self.sf,'Xpoint'):
            self.sf.get_Xpsi()  
        self.xo = [self.sf.Xpoint[0],self.sf.eqdsk['zmagx']]
        self.Rp,self.Zp = self.sf.get_boundary()
        self.Lp = geom.length(self.Rp,self.Zp,norm=False)[-1]
        self.Rfw,self.Zfw,self.psi_fw = self.firstwall_loop() 
        self.loop = Loop(self.Rfw,self.Zfw,xo=self.xo)  # first wall contour
        

        
    def add_bound(r,z):
        
        
    def add_divertor():
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
        rd,zd = geom.rzInterp(rd,zd)  # resample
        rd,zd = geom.inloop(r,z,rd,zd,side='out')  # divertor external to fw
        istart = np.argmin((r-rd[-1])**2+(z-zd[-1])**2)  # connect to fw
        iend = np.argmin((r-rd[0])**2+(z-zd[0])**2)
        
        if self.sf.Xloc == 'lower':
            r = np.append(np.append(rd,r[istart:iend]),rd[0])
            z = np.append(np.append(zd,z[istart:iend]),zd[0])
        else:
            r = np.append(np.append(rd,r[iend:istart][::-1]),rd[0])
            z = np.append(np.append(zd,z[iend:istart][::-1]),zd[0])
        self.segment['first_wall'] = {'r':r,'z':z} 
        return r,z
    
'''    



