import numpy as np
import amigo.geom as geom
from nova.loops import Profile
from nova.shape import Shape
import pylab as pl
from warnings import warn
import datetime
import pickle
from nova.streamfunction import SF
from nova.config import Setup
from collections import OrderedDict
import seaborn as sns

class divertor(object):
    
    def __init__(self,sf,setup,flux_conformal=False,debug=False):
        self.debug = debug
        self.sf = sf
        self.targets = setup.targets  # target definition (graze, L2D, etc)
        self.div_ex = setup.firstwall['div_ex']
        self.segment = {}
        if flux_conformal:
            self.Rfw,self.Zfw,self.psi_fw = sf.firstwall_loop(\
            psi_n=setup.firstwall['psi_n'])
        else:
            self.Rfw,self.Zfw,self.psi_fw = sf.firstwall_loop(\
            dr=setup.firstwall['dRfw'])
        
    def set_target(self,leg,**kwargs):
        if leg not in self.targets:
            self.targets[leg] = {}
        for key in self.targets['default']:
            if key in kwargs:
                self.targets[leg][key] = kwargs[key]  # update
            elif key not in self.targets[leg]:  #  prevent overwrite
                self.targets[leg][key] = self.targets['default'][key]
    
    def place(self,**kwargs):
        layer_index = 0  # construct from first open flux surface
        self.sf.sol(update=True,plot=False,debug=self.debug)
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
                                     dPlate,leg,debug=self.debug)
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
        
    def join(self,main_chamber):
        r,z = main_chamber.draw(npoints=500)
        istart = np.argmin((r-self.sf.Xpoint[0])**2+(z-self.sf.Xpoint[1])**2)
        r = np.append(r[istart:],r[:istart+1])
        z = np.append(z[istart:],z[:istart+1])
        r,z = geom.unique(r,z)[:2]
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
        return self.segment
    
    def connect(self,psi,target_pair,ends,loop=[]):
        gap = []
        if loop:
            r,z = loop
        else:
            psi_line = self.sf.get_contour([psi])[0]
            for line in psi_line:
                r,z = line[:,0],line[:,1]
                gap.append(np.min((self.targets[target_pair[0]]['R'][ends[0]]-r)**2+
                          (self.targets[target_pair[0]]['Z'][ends[0]]-z)**2))
            select = np.argmin(gap)
            line = psi_line[select]
            r,z = line[:,0],line[:,1]
        index = np.zeros(2,dtype=int)
        index[0] = np.argmin((self.targets[target_pair[0]]['R'][ends[0]]-r)**2+
                 (self.targets[target_pair[0]]['Z'][ends[0]]-z)**2)
        index[1] = np.argmin((self.targets[target_pair[1]]['R'][ends[1]]-r)**2+
                 (self.targets[target_pair[1]]['Z'][ends[1]]-z)**2)
        if index[0]>index[1]:
            index = index[::-1]
        r,z = r[index[0]:index[1]+1],z[index[0]:index[1]+1] 
        return r,z
    
    def match_psi(self,Ro,Zo,direction,theta_end,theta_sign,phi_target,graze,
                  dPlate,leg,debug=False): 
        color = sns.color_palette('Set2',2)
        gain = 0.15  # 0.25
        Nmax = 500
        Lo = [1.0,0.0015]  # [blend,turn]  5,0.015
        r2m = [-1.25,-1]  # ramp to step (+ive-lead, -ive-lag ramp==1, step==inf)
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
            Bhat = geom.rotate_vector2D(B,theta_sign*theta)
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
            Bhat = geom.rotate_vector2D(B,theta_sign*theta)
            R = np.append(R,R[-1]+dL*Bhat[0])
            Z = np.append(Z,Z[-1]+dL*Bhat[1])
        return R,Z
    
    def append(self,R,Z,r,z):
        dr = np.zeros(2)
        for i,end in enumerate([0,-1]):
            dr[i] = (R[-1]-r[end])**2+(Z[-1]-z[end])**2
        if dr[1] < dr[0]:
            r,z = r[::-1],z[::-1]
        return np.append(R,r[1:-1]),np.append(Z,z[1:-1]) 


class main_chamber(object):
    
    def __init__(self,name,**kwargs):
        self.name = name
        self.set_filename(**kwargs)
        self.initalise_loop()
        
    def initalise_loop(self):
        self.profile = Profile(self.filename,family='S',part='chamber',
                               npoints=200)
        self.shp = Shape(self.profile,objective='L')
        self.set_bounds()
        
    def set_bounds(self):
        self.shp.loop.adjust_xo('upper',lb=0.7)
        self.shp.loop.adjust_xo('top',lb=0.05,ub=0.75)
        self.shp.loop.adjust_xo('lower',lb=0.7)
        self.shp.loop.adjust_xo('bottom',lb=0.05,ub=0.75)
        self.shp.loop.adjust_xo('l',lb=0.8,ub=1.5)
        
    def date(self,verbose=True):
        today = datetime.date.today().strftime('%Y_%m_%d')
        if verbose:
            print(today)
        return today
    
    def set_filename(self,update=False,**kwargs):
        today = self.date(verbose=False)
        if update:  # use today's date
            date_str = today
        else:
            date_str = kwargs.get('date',today)
        self.filename = '{}_{}'.format(date_str,self.name)  # chamber name
        
    def generate(self,eq_names,dr=0.225,psi_n=1.07,
                 flux_fit=False,symetric=False,plot=False):
        self.set_filename(update=True)  # update date in filename
        self.profile.loop.reset_oppvar(symetric)  # reset loop oppvar
        self.config = {'dr':dr,'psi_n':psi_n,'flux_fit':flux_fit,'Nsub':100}
        self.config['eqdsk'] = []  
        sf_list = self.load_sf(eq_names)
        for sf in sf_list:  # convert input to list
            self.add_bound(sf)
        self.shp.add_internal(r_gap=0.001)  # add internal bound
        self.shp.minimise()
        self.write()  # append config data to loop pickle
        if plot:
            self.plot_chamber()
        
    def load_sf(self,eq_names):
        sf_dict,sf_list = OrderedDict(),[]
        for configuration in eq_names:
            sf = SF(Setup(configuration).filename)
            sf_dict[configuration] = sf.filename.split('/')[-1]
            sf_list.append(sf)
        self.config['eqdsk'] = sf_dict
        return sf_list
        
    def write(self):  # overwrite loop_dict + add extra chamber fields
        with open(self.profile.dataname, 'wb') as output:
            pickle.dump(self.profile.loop_dict,output,-1)
            pickle.dump(self.config,output,-1)
            pickle.dump(self.shp.bound,output,-1)  # boundary points
            pickle.dump(self.shp.bindex,output,-1)  # boundary index

    def load_data(self,plot=False):
        try:
            with open(self.profile.dataname, 'rb') as input:
                self.profile.loop_dict = pickle.load(input)
                self.config = pickle.load(input) 
                self.shp.bound = pickle.load(input) 
                self.shp.bindex = pickle.load(input) 
        except:
            errtxt = 'boundary information not found'
            raise ValueError(errtxt)
        if plot:
            self.plot_chamber()

    def plot_chamber(self):
        self.shp.loop.plot()
        self.shp.plot_bounds()
        r,z = self.draw()
        pl.plot(r,z)
      
    def add_bound(self,sf): 
        rpl,zpl = sf.get_offset(self.config['dr'],Nsub=self.config['Nsub'])
        self.shp.add_bound({'r':rpl,'z':zpl},'internal')  # vessel inner bounds

        if self.config['flux_fit']:  # add flux boundary points
            sf.get_LFP()  # get low feild point
            rflux,zflux = sf.first_wall_psi(psi_n=self.config['psi_n'],
                                            trim=False)[:2]
            rflux,zflux = sf.midplane_loop(rflux,zflux)
            rflux,zflux = geom.order(rflux,zflux)
            istop = next((i for i in range(len(zflux)) if zflux[i]<sf.LFPz),-1)
            rflux,zflux = rflux[:istop],zflux[:istop]
            dL = np.diff(geom.length(rflux,zflux))
            if np.max(dL) > 3*np.median(dL) or \
            np.argmax(zflux) == len(zflux)-1:
                wtxt = '\n\nOpen feild line detected\n'
                wtxt += 'disabling flux fit for '
                wtxt += '{:1.1f}% psi_n \n'.format(1e2*self.config['psi_n'])
                wtxt += 'configuration: '+sf.filename+'\n'
                warn(wtxt)
            else:  # add flux_fit bounds
                rflux,zflux = geom.rzSLine(rflux,zflux,
                                           int(self.config['Nsub']/2))
                self.shp.add_bound({'r':rflux,'z':zflux},'internal')  

    def draw(self,npoints=250):
        x = self.profile.loop.draw(npoints=npoints)
        r,z = x['r'],x['z']
        r,z = geom.order(r,z,anti=True)
        return r,z

