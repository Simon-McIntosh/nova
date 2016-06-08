import numpy as np
import pylab as pl
import cross_coil as cc
from collections import OrderedDict as od
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d as interp1
import seaborn as sns
from radial_build import RB
import scipy.optimize as op
from itertools import cycle
import copy
import time
import multiprocessing

mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]

class INV(object):
    
    def __init__(self,sf,eq,configTF='SN',config='SN'):
        self.ncpu = multiprocessing.cpu_count()
        self.sf = sf
        self.eq = eq
        self.fix = {'r':np.array([]),'z':np.array([]),'BC':np.array([]),
                    'value':np.array([]),'Bdir':np.array([[],[]]).T,
                    'factor':np.array([])}  
        self.cnstr = {'r':np.array([]),'z':np.array([]),'BC':np.array([]),
                     'value':np.array([]),'condition':np.array([])}  
        self.add_active([],empty=True)  # all coils active
        self.update_coils()
        self.initalise_current()
        conf = Config(config)
        conf.TF(self.sf)
        self.rb = RB(conf,self.sf,Np=150)  # load radial build
        self.interpTF(config=configTF,Ltrim=[0.15,0.85])  # TF interpolators [0.08,0.85]
        self.log_scalar = ['Fr_max','Fz_max','F_max','FzCS','Fsep','rms',
                           'rms_bndry','rmsID','Isum','Imax','z_offset',
                           'FrPF','FzPF','dTpol']
        self.log_array = ['Lo','Iswing']
        self.log_iter = ['current_iter','plasma_iter','position_iter']
        self.log_plasma = ['plasma_coil']
        self.CS_Lnorm,self.CS_Znorm = 2,1
        self.TFoffset = 0.1
        
    def Cname(self,coil):
        return 'Coil{:1.0f}'.format(coil)
          
    def add_active(self,Clist,Ctype=None,empty=False):  # list of coil names
        if empty: 
            self.adjust_coils = []
            self.PF_coils = []
            self.CS_coils = []
        if Clist:
            for coil in Clist:
                name = self.Cname(coil)
                if name not in self.adjust_coils:
                    self.adjust_coils.append(name)
                if Ctype == 'PF' and name not in self.PF_coils:
                    self.PF_coils.append(name)
                elif Ctype == 'CS' and name not in self.CS_coils:
                    self.CS_coils.append(name)
        else:
            self.adjust_coils = list(self.sf.coil.keys())  # add all
            if Ctype == 'PF': 
                self.PF_coils = list(self.sf.coil.keys())
            elif Ctype == 'CS': 
                self.CS_coils = list(self.sf.coil.keys())    
        self.update_coils()  # update

    def remove_active(self,Clist,Ctype=None,full=False):  # list of coil names
        if full: 
            self.adjust_coils = list(self.sf.coil.keys())  # add all
            if Ctype == 'PF':
                self.PF_coils = list(self.sf.coil.keys())
            elif Ctype == 'CS':
                self.CS_coils = list(self.sf.coil.keys())  
        if Clist:
            for coil in Clist:
                name = self.Cname(coil)
                if name in self.adjust_coils:
                    self.adjust_coils.pop(self.adjust_coils.index(name))
                if (Ctype == 'PF' or Ctype == 'all') and name in self.PF_coils:
                    self.PF_coils.pop(self.PF_coils.index(name))
                if (Ctype == 'CS' or Ctype == 'all') and name in self.CS_coils:
                    self.CS_coils.pop(self.CS_coils.index(name))
        else:
            self.adjust_coils = []  # remove all
            self.PF_coils = []
            self.CS_coils = []
        self.update_coils()  # update
        
    def delete_active(self):
        self.update_coils()  # update
        Clist = []
        for name in self.coil['active'].keys():
            Clist.append(int(name.split('Coil')[-1]))
        self.remove_active(Clist,Ctype='all')
        self.remove_coil(Clist)
        self.update_coils()  # update
            
    def regenVcoil(self,Clist):
        regenV = False
        for index in [0,-1]:
            Vnum = int(self.eq.Vcoil[index]['name'].decode().split('Coil')[-1])
            if Vnum in Clist:
                regenV = True
        if regenV:
            self.eq.get_Vcoil()
    
    def remove_coil(self,Clist):
        for coil in Clist:
            name = self.Cname(coil)
            if name in self.sf.coil.keys():
                del self.sf.coil[name]
                for i in range(self.eq.coil[name+'_0']['Nf']):
                    del self.eq.coil[name+'_{:1.0f}'.format(i)]
        self.regenVcoil(Clist)
        self.update_coils()  # update
   
    def get_point(self,**kwargs):
        keys = kwargs.keys()
        if 'point' in keys:
            r,z = kwargs['point']
        elif 'polar' in keys:
            mag,arg = kwargs['polar']
            r = self.sf.Xpoint[0]+mag*np.sin(arg*np.pi/180)
            z = self.sf.Xpoint[1]-mag*np.cos(arg*np.pi/180)
        elif 'Lout' in keys:
            L = kwargs['Lout']
            r,z = self.TFoutR(L),self.TFoutZ(L)
            dr,dz = self.TFoutdR(L),self.TFoutdZ(L)
        elif 'Lin' in keys:
            L = kwargs['Lin']
            r,z = self.TFinR(L),self.TFinZ(L)
            dr,dz = self.TFindR(L),self.TFindZ(L)
        if 'norm' in keys and 'point' not in keys:        
            delta = kwargs['norm']*np.array([dr,dz])/np.sqrt(dr**2+dz**2)
            r += delta[1]
            z -= delta[0]
        return r,z
        
    def add_coil(self,Ctype=None,**kwargs):
        r,z = self.get_point(**kwargs)
        index,i = np.zeros(len(self.sf.coil.keys())),-1
        for i,name in enumerate(self.sf.coil.keys()):
            index[i] = name.split('Coil')[-1]
        try:
            Cid = index.max()+1
        except:
            Cid = 0
        name = 'Coil{:1.0f}'.format(Cid)
        self.add_active([Cid],Ctype=Ctype)
        delta = np.ones(2)
        for i,dx in enumerate(['dr','dz']):
            if dx in kwargs.keys():
                delta[i] = kwargs.get(dx)
        self.sf.coil[name] = {'r':r,'z':z,'dr':delta[0],'dz':delta[1],'I':1e6,
                              'rc':np.sqrt(delta[0]**2+delta[1]**2)/2}

    def append_coil(self,coils):
        for sub_name in coils.keys():
            name = sub_name.split('_')[0]
            if name in self.adjust_coils:
                state = 'active'
            else:
                state = 'passive'
            if name not in self.coil[state].keys():  # initalise
                self.coil[state][name] = {'r':np.array([]),'z':np.array([]),
                                          'dr':np.array([]),'dz':np.array([]),
                                          'sub_name':np.array([]),
                                          'I':np.array([]),
                                          'Fr':np.array([]),'Fz':np.array([]),
                                          'Fr_sum':0,'Fz_sum':0,'I_sum':0,
                                          'Ro':0,'Zo':0,'Nf':0}
            coil = coils[sub_name]
            for key,var in zip(['r','z','I','dr','dz','sub_name'],
                               [coil['r'],coil['z'],coil['I'],
                                coil['dr'],coil['dz'],sub_name]):
                self.coil[state][name][key] = \
                np.append(self.coil[state][name][key],var)
            self.coil[state][name]['Ro'] = np.mean(self.coil[state][name]['r'])
            self.coil[state][name]['Zo'] = np.mean(self.coil[state][name]['z'])
            self.coil[state][name]['Nf'] = len(self.coil[state][name]['z'])
         
    def plot_coils(self):
        Color = cycle(sns.color_palette('Set2',12))
        for state,marker in zip(['passive','active'],['o','d']):
            for name in self.coil[state].keys():
                if name != 'Plasma':
                    color = next(Color)
                    for r,z in zip(self.coil[state][name]['r'],
                                   self.coil[state][name]['z']):
                        pl.plot(r,z,marker,color=color,markersize=3)
                        if state == 'passive':  # empty marker
                            pl.plot(r,z,marker,color='w',markersize=1.5)
                            
    def add_cnstr(self,r,z,BC,condition,value):
        var = {'r':r,'z':z,'value':value,'condition':condition,'BC':BC}
        nvar = len(r)
        for name in ['value','condition','BC']:
            if np.shape(var[name])[0] != nvar:
                var[name] = np.array([var[name]]*nvar)
        for name in var.keys(): 
                self.cnstr[name] = np.append(self.cnstr[name],var[name])                        
            
    def add_fix(self,r,z,value,Bdir,BC,factor):
        var = {'r':r,'z':z,'value':value,'Bdir':Bdir,'BC':BC,'factor':factor}
        nvar = len(r)
        for name in ['value','Bdir','BC','factor']:
            if np.shape(var[name])[0] != nvar:
                var[name] = np.array([var[name]]*nvar)
        for name in var.keys(): 
            if name == 'Bdir':
                for i in range(nvar):
                    norm = np.sqrt(var[name][i][0]**2+var[name][i][1]**2)
                    if norm != 0:
                        var[name][i] /= norm  # normalise tangent vectors
                self.fix[name] = np.append(self.fix[name],var[name],axis=0)
            else:
                self.fix[name] = np.append(self.fix[name],var[name])
                
    def swing_fix(self,swing):
        if not hasattr(self,'fix_o'):
            self.fix_o = copy.deepcopy(self.fix)
        for i,(bc,value) in enumerate(zip(self.fix_o['BC'],self.fix_o['value'])):
            if 'psi' in bc:
                self.fix['value'][i] = value+swing
        
    def set_boundary(self,N=21,alpha=0.995):
        r,z = self.sf.get_boundary(alpha=alpha)
        self.eq_spline()
        nr = -self.psi.ev(r,z,dx=1,dy=0)
        nz = -self.psi.ev(r,z,dx=0,dy=1)
        L = self.sf.length(r,z)
        Lc = np.linspace(0,1,N+1)[:-1]
        rb,zb = interp1(L,r)(Lc),interp1(L,z)(Lc)
        Bdir = np.array([interp1(L,nr)(Lc),interp1(L,nz)(Lc)]).T
        return rb,zb,Bdir
        
    def get_psi(self,alpha):
        Xpsi = self.sf.Xpsi  # self.eq.Pcoil(self.sf.Xpoint)
        Mpsi = self.sf.Mpsi  # self.eq.Pcoil(self.sf.Mpoint)
        psi = Mpsi+alpha*(Xpsi-Mpsi)
        return psi
        
    def fix_boundary_psi(self,N=21,alpha=0.995,factor=1):
        r,z,Bdir = self.set_boundary(N=N,alpha=alpha)
        psi = self.get_psi(alpha)*np.ones(N)
        self.add_fix(r,z,psi,Bdir,['psi_bndry'],[factor])

    def fix_boundary_feild(self,N=21,alpha=0.995,factor=1):
        r,z,Bdir = self.set_boundary(N=N,alpha=alpha)
        self.add_fix(r,z,[0.0],Bdir,['Bdir'],[factor])
        
    def add_null(self,factor=1,**kwargs):
        r,z = self.get_point(**kwargs)
        self.add_fix([r],[z],[0.0],np.array([[1.0],[0.0]]).T,['Br'],[factor])
        self.add_fix([r],[z],[0.0],np.array([[0.0],[1.0]]).T,['Bz'],[factor])
        
    def add_Bro(self,factor=1,**kwargs):
        r,z = self.get_point(**kwargs)   
        self.add_fix([r],[z],[0.0],np.array([[1.0],[0.0]]).T,['Br'],[factor])
        
    def add_Bzo(self,factor=1,**kwargs):
        r,z = self.get_point(**kwargs)    
        self.add_fix([r],[z],[0.0],np.array([[0.0],[1.0]]).T,['Bz'],[factor])
        
    def add_B(self,B,Bdir,factor=1,zero_norm=False,**kwargs):
        r,z = self.get_point(**kwargs)
        if len(Bdir) == 1:  # normal angle from horizontal in degrees
            arg = Bdir[0]
            Bdir = [-np.sin(arg*np.pi/180),np.cos(arg*np.pi/180)]
        Bdir /= np.sqrt(Bdir[0]**2+Bdir[1]**2)
        self.add_fix([r],[z],[B],np.array([[Bdir[0]],[Bdir[1]]]).T,
                     ['Bdir'],[factor])
        if zero_norm:
            self.add_fix([r],[z],[0],np.array([[-Bdir[1]],[Bdir[0]]]).T,
                     ['Bdir'],[factor])
                     
    def add_theta(self,theta,factor=1,graze=1.5,**kwargs):
        r,z = self.get_point(**kwargs)
        Bm = np.abs(self.sf.bcentr*self.sf.rcentr)  # toroidal moment
        Bphi = Bm/r  # torodal field
        Bp = Bphi/np.sqrt((np.sin(theta*np.pi/180)/
                           np.sin(graze*np.pi/180))**2)
        self.add_fix([r],[z],[Bp],np.array([[0],[0]]).T,['Bp'],[factor])

    def add_psi(self,psi,factor=1,**kwargs):
        r,z = self.get_point(**kwargs)
        self.add_fix([r],[z],[psi],np.array([[0],[0]]).T,['psi'],[factor])
        
    def add_alpha(self,alpha,factor=1,**kwargs):
        psi = self.get_psi(alpha)
        self.add_psi(psi,factor=factor,**kwargs)
        
    def add_Bcon(self,B,**kwargs):
        r,z = self.get_point(**kwargs)
        self.add_cnstr([r],[z],['B'],['gt'],[B])
        
    def plot_fix(self):
        if hasattr(self,'weight'):
            weight = self.weight/np.mean(self.weight)
        else:
            weight = np.ones(len(self.fix['BC']))
        Color = sns.color_palette('Set2',8)
        psi,Bdir,Brz = [],[],[]
        tail_length = 0.75
        for bc,w in zip(self.fix['BC'],weight):
            if 'psi' in bc:
                psi.append(w)
            elif bc == 'Bdir':
                Bdir.append(w)
            elif bc == 'Br' or bc == 'Bz':
                Brz.append(w)
        if len(psi) > 0: psi_norm = tail_length/np.mean(psi)
        if len(Bdir) > 0: Bdir_norm = tail_length/np.mean(Bdir)
        if len(Brz) > 0: Brz_norm = tail_length/np.mean(Brz)
        for r,z,bc,bdir,w in zip(self.fix['r'],self.fix['z'],
                                 self.fix['BC'],self.fix['Bdir'],weight):
            if bdir[0]**2+bdir[1]**2 > 0:  # tails    
                    direction = bdir
            else:
                d_dr,d_dz = self.get_gradients(bc,r,z)
                direction = np.array([d_dr,d_dz])/np.sqrt(d_dr**2+d_dz**2)
            if 'psi' in bc:
                norm = psi_norm
                marker,size,color = 'o',10.0,Color[0]
                pl.plot(r,z,marker,color=color,markersize=size)
                pl.plot(r,z,marker,color=[1,1,1],markersize=0.5*size)
                #pl.plot([r,r+direction[0]*norm*w],[z,z+direction[1]*norm*w],
                #        color=color,linewidth=2)
            else:
                if bc == 'Bdir':
                    norm = Bdir_norm
                    marker,size,color,mew = 'o',4,Color[1],0.0
                elif bc == 'null':
                    norm = 1
                    marker,size,color,mew = 'o',2,Color[2],0.0
                elif bc == 'Br' or bc == 'Bz':
                    norm = Brz_norm
                    marker,size,color,mew = 'x',8,Color[2],1
                pl.plot(r,z,marker,color=color,markersize=size,
                        markeredgewidth=mew)

    def set_background(self):
        self.BG = np.zeros(len(self.fix['BC']))  # background 
        self.eq_spline()  # generate equlibrium interpolators
        self.plasma_spline()  # generate plasma interpolators
        self.add_value('passive')
        
    def set_T(self):
        self.T = (self.fix['value']-self.BG).reshape((len(self.BG),1)) 
                    
    def set_foreground(self):
        self.G = np.zeros((len(self.fix['BC']),
                           len(self.coil['active'].keys())))  # [G][I] = [T] 
        self.add_value('active')
                    
    def add_value(self,state):
        Rf,Zf,BC,Bdir = self.unpack_fix()
        for i,(rf,zf,bc,bdir) in enumerate(zip(Rf,Zf,BC,Bdir)):
            for j,name in enumerate(self.coil[state].keys()):
                if name == 'Plasma' and self.eq.ingrid(rf,zf):  # only passive
                    self.BG[i] += self.add_value_plasma(bc,rf,zf,bdir)
                else:
                    coil = self.coil[state][name]
                    R,Z,current = coil['r'],coil['z'],coil['I']
                    if state == 'active': current = np.ones(np.shape(current))
                    for r,z,I in zip(R,Z,current):
                        value = self.add_value_coil(bc,rf,zf,r,z,I,bdir)
                        if state == 'active':
                            self.G[i,j] += value
                        else:
                            self.BG[i] += value
        
    def add_value_coil(self,bc,rf,zf,r,z,I,bdir):
        if 'psi' in bc:
            value = mu_o*I*cc.green(rf,zf,r,z)
        else:
            B = mu_o*I*cc.green_feild(rf,zf,r,z)
            value = self.Bfeild(bc,B[0],B[1],bdir)
        return value
        
    def get_force(self):
        self.Fr_max,self.Fz_max,self.F_max = 0,0,0
        F_sum = np.zeros((self.Nadj,2))
        for i in range(2):
            F_sum[:,i] = (self.I*(np.dot(self.Fa[:,:,i],self.I)+
                                 self.Fp[:,i].reshape(-1,1)))[:,0]
        for i,name in enumerate(self.coil['active'].keys()):
            Fr,Fz = F_sum[i,0],F_sum[i,1]
            Fmag = np.sqrt(Fr**2+Fz**2)
            if np.abs(Fr) > np.abs(self.Fr_max): self.Fr_max = Fr
            if np.abs(Fz) > np.abs(self.Fz_max): self.Fz_max = Fz
            if Fmag > self.F_max: self.F_max = Fmag
            self.coil['active'][name]['Fr_sum'] = Fr
            self.coil['active'][name]['Fz_sum'] = Fz
        nPF = len(self.PF_coils) 
        FrPF,FzPF = np.zeros(nPF),np.zeros(nPF)
        for i,name in enumerate(self.PF_coils):
            FrPF[i] = self.coil['active'][name]['Fr_sum']
            FzPF[i] = self.coil['active'][name]['Fz_sum']
        self.FrPF = np.max(abs(FrPF))
        self.FzPF = np.max(abs(FzPF))
        nCS = len(self.CS_coils)
        FzCS,Fsep = np.zeros(nCS),np.zeros(nCS-1)
        for i,name in enumerate(self.CS_coils[::-1]):  # top-bottom
            FzCS[i] = self.coil['active'][name]['Fz_sum']
        for j in range(nCS-1):  # evaluate each gap
            Fsep[j] = np.sum(FzCS[:j+1])-np.sum(FzCS[j+1:])
        self.Fsep = Fsep.max()
        self.FzCS = np.sum(FzCS)
            
            
    def plot_force(self,scale=3):
        coil = self.coil['active']
        color = sns.color_palette('Set2',2)
        for name in coil.keys():
            Ro,Zo = coil[name]['Ro'],coil[name]['Zo']
            pl.arrow(Ro,Zo,0,scale*coil[name]['Fz_sum']/self.F_max,
                     linewidth=1,head_width=0.1,head_length=0.1,
                     color=color[0])
            pl.arrow(Ro,Zo,scale*coil[name]['Fr_sum']/self.F_max,0,
                     linewidth=1,head_width=0.1,head_length=0.1,
                     color=color[1])
        
    def set_force_feild(self,state='both',multi_filament=False):
        # [I]T([Fa][I]+[Fp]) = F
        active_coils = self.coil['active'].keys()
        passive_coils = self.coil['passive'].keys()
        self.Nadj = len(active_coils)  # number of active coils
        if state == 'both' or state == 'active':
            self.set_active_force_feild(active_coils,
                                        multi_filament=multi_filament)
        if state == 'both' or state == 'passive':
            self.set_passive_force_feild(active_coils,passive_coils)

    def set_active_force_feild(self,active_coils,multi_filament=False):  
        self.Fa = np.zeros((self.Nadj,self.Nadj,2))  # active
        for i,sink in enumerate(active_coils):
            for j,source in enumerate(active_coils):
                rG = cc.Gtorque(self.eq.coil,self.sf.coil,
                                source,sink,multi_filament)
                self.Fa[i,j,0] = 2*np.pi*mu_o*rG[1]  #  cross product
                self.Fa[i,j,1] = -2*np.pi*mu_o*rG[0]
        
    def set_passive_force_feild(self,active_coils,passive_coils):
        self.Fp = np.zeros((self.Nadj,2))  # passive
        for i,sink in enumerate(active_coils):
            rB = cc.Btorque(self.eq,passive_coils,sink)
            self.Fp[i,0] = 2*np.pi*mu_o*rB[1]  #  cross product
            self.Fp[i,1] = -2*np.pi*mu_o*rB[0]  
  
    def add_value_plasma(self,bc,rf,zf,bdir):
        if 'psi' in bc:
            value = self.psi_plasma.ev(rf,zf)  # interpolate from GS
        else:
            Br,Bz = self.B_plasma[0].ev(rf,zf),self.B_plasma[1].ev(rf,zf)
            value = self.Bfeild(bc,Br,Bz,bdir)
        return value

    def Bfeild(self,bc,Br,Bz,Bdir):
        if bc == 'Br':
            value = Br
        elif bc == 'Bz':
            value = Bz 
        elif bc == 'null':
            value = np.sqrt(Bz**2+Bz**2)  
        elif bc == 'Bdir':
            nhat = Bdir/np.sqrt(Bdir[0]**2+Bdir[1]**2)
            value = np.dot([Br,Bz],nhat)
        elif bc == 'Bp':
            value = Br-Bz/2
        return value
        
    def get_mesh(self):
        self.rm = np.array(np.matrix(self.eq.r).T*np.ones([1,self.eq.nz]))
        self.rm[self.rm==0] = 1e-34
        self.zm = np.array(np.ones([self.eq.nr,1])*np.matrix(self.eq.z))
        
    def eq_spline(self):
        self.psi = RBS(self.eq.r,self.eq.z,self.eq.psi)
        self.get_mesh()
        psi_r = self.psi.ev(self.rm,self.zm,dx=1,dy=0)
        psi_z = self.psi.ev(self.rm,self.zm,dx=0,dy=1)
        Br,Bz = -psi_z/self.rm,psi_r/self.rm
        B = np.sqrt(Br**2+Bz**2)
        self.B = RBS(self.eq.r,self.eq.z,B)
        
    def plasma_spline(self):
        self.B_plasma = [[],[]]
        self.psi_plasma = RBS(self.eq.r,self.eq.z,self.eq.psi_plasma)
        self.get_mesh()
        psi_plasma_r = self.psi_plasma.ev(self.rm,self.zm,dx=1,dy=0)
        psi_plasma_z = self.psi_plasma.ev(self.rm,self.zm,dx=0,dy=1)
        Bplasma_r,Bplasma_z = -psi_plasma_z/self.rm,psi_plasma_r/self.rm
        self.B_plasma[0] = RBS(self.eq.r,self.eq.z,Bplasma_r)
        self.B_plasma[1] = RBS(self.eq.r,self.eq.z,Bplasma_z)
        
    def unpack_fix(self):
        Rf,Zf = self.fix['r'],self.fix['z']
        BC,Bdir = self.fix['BC'],self.fix['Bdir']
        return Rf,Zf,BC,Bdir
        
    def get_gradients(self,bc,rf,zf):
        if 'psi' in bc:
            d_dr = self.psi.ev(rf,zf,dx=1,dy=0)
            d_dz = self.psi.ev(rf,zf,dx=0,dy=1)
        else:
            d_dr = self.B.ev(rf,zf,dx=1,dy=0)
            d_dz = self.B.ev(rf,zf,dx=0,dy=1)
        return d_dr,d_dz
        
    def get_weight(self):
        Rf,Zf,BC,Bdir = self.unpack_fix()
        self.weight = np.ones(len(BC))
        '''
        for i,(rf,zf,bc,bdir,factor) in enumerate(zip(Rf,Zf,BC,Bdir,
                                                      self.fix['factor'])):
            d_dr,d_dz = self.get_gradients(bc,rf,zf)
            if bdir[0]**2+bdir[1]**2 == 0:# or bc != 'psi':
                self.weight[i] = 1/abs(np.sqrt(d_dr**2+d_dz**2))
            else:
                self.weight[i] = 1/abs(np.dot([d_dr,d_dz],bdir))
        '''
        
    def gain_weight(self):
        factor = np.reshape(self.fix['factor'],(-1,1))
        weight = np.reshape(self.weight,(-1,1))
        self.wG = np.sqrt(factor*weight)*self.G
        self.wT = np.sqrt(factor*weight)*self.T
            
    def solve(self):
        self.set_T()
        self.set_foreground()
        self.gain_weight()
        self.I = np.linalg.lstsq(self.wG,self.wT)[0]  # solve
        self.get_rms()
        self.update_current()
     
    def read_TF(self,config):
        file = '../Data/'+config+'_TFcoil.txt'
        nL = sum(1 for line in open(file))-1  # file data length
        Rin,Zin,Rout,Zout = np.zeros(nL),np.zeros(nL),np.zeros(nL),np.zeros(nL)
        with open(file,'r') as f:
            f.readline()  # header
            for i,line in enumerate(f):
                line = line.split('\t')
                Rin[i] = float(line[0])
                Zin[i] = float(line[1])
                Rout[i] = float(line[2])
                Zout[i] = float(line[3])
        return Rout,Zout
        
    def TFinterpolator(self,R,Z,dr=0,Ltrim=[0.1,0.9]):
        R,Z = self.rb.offset(R,Z,dr)
        L = self.rb.length(R,Z)
        Lt = np.linspace(Ltrim[0],Ltrim[1],int((1-np.diff(Ltrim))*len(L)))
        R,Z = interp1(L,R)(Lt),interp1(L,Z)(Lt)
        L = np.linspace(0,1,len(R))
        return IUS(L,R),IUS(L,Z)
        
    def interpTF(self,config='SN',Ltrim=[0.1,0.9]):
        self.rb.TFopp(False,config=config)  #  load TF outline
        self.rb.TFfill()  # construct TF geometory
        self.TFinR,self.TFinZ =  self.TFinterpolator(self.rb.TFinR,
                                                     self.rb.TFinZ,Ltrim=Ltrim)
        self.TFindR = self.TFinR.derivative(n=1) 
        self.TFindZ = self.TFinZ.derivative(n=1)                                             
        self.TFoutR,self.TFoutZ =  self.TFinterpolator(self.rb.TFoutR,
                                                       self.rb.TFoutZ,dr=0.75,
                                                       Ltrim=Ltrim)
        self.TFoutdR = self.TFoutR.derivative(n=1) 
        self.TFoutdZ = self.TFoutZ.derivative(n=1) 
        
    def arange_CS(self,Z,L,dL):
        Z *= self.CS_Znorm
        L *= self.CS_Lnorm
        dL = abs(dL)**2.5
        dL /= np.sum(dL)  # normalize
        zbase = Z-L/2
        dZ = L*dL
        Z = np.zeros(len(dZ))
        for i,dz in enumerate(L*dL):
            Z[i] = zbase+dz/2
            zbase += dz
        return Z,dZ
        
    def grid_CS(self,nCS=3,L=19,Ro=3.2,Zo=-1,dr=1.0):  #dr=1.25
        Zo /= self.CS_Znorm
        L /= self.CS_Lnorm
        #dL = np.sin(np.pi/nCS*(np.arange(nCS)+0.5))  # sinusodal spacing
        dL = np.ones(nCS)/(nCS)  # linear spacing
        Z,dZ = self.arange_CS(Zo,L,dL)
        for z,dz in zip(Z,dZ):
            self.add_coil(point=(Ro,z),dr=dr,dz=dz,Ctype='CS')
        return np.append(np.array([Zo,L]),dL)
    
    def grid_PF(self,nPF=5):
        dL = 1/nPF
        Lo = np.linspace(dL/2,1-dL/2,nPF)
        self.delete_active()
        for L in Lo:
            self.add_coil(Lout=L,Ctype='PF')    
        return Lo
            
    def get_rms(self):
        err = np.dot(self.wG,self.I)-self.wT
        self.rms = np.sqrt(np.mean(err**2))
        return err
    
    def get_rms_bndry(self):
        self.eq.run(update=False)
        psi_o = self.fix['value'][0]
        psi_line = self.sf.get_contour([psi_o])[0]
        dx_bndry,dx_min = np.array([]),0 
        dz_bndry,dz_min = np.array([]),0 
        Rf,Zf,BC = self.unpack_fix()[:3]
        for rf,zf,bc,psi in zip(Rf,Zf,BC,self.fix['value']):
            if bc == 'psi_bndry':
                if psi != psi_o:  # update contour
                    psi_o = psi
                    psi_line = self.sf.get_contour([psi_o])[0]
                for j,line in enumerate(psi_line):
                    r,z = line[:,0],line[:,1]
                    dx = np.sqrt((rf-r)**2+(zf-z)**2)
                    xmin_index = np.argmin(dx)
                    if j==0 or dx[xmin_index] < dx_min:  # update boundary error
                        dx_min = dx[xmin_index]
                    dz = zf-z
                    zmin_index = np.argmin(np.abs(dz))
                    if j==0 or np.abs(dz[zmin_index]) < np.abs(dz_min):  # update boundary error
                        dz_min = dz[zmin_index]    
                        
                dx_bndry = np.append(dx_bndry,dx_min)
                dz_bndry = np.append(dz_bndry,dz_min)
        rms_bndry = np.sqrt(np.mean(dx_bndry**2))  # calculate rms
        z_offset = np.mean(dz_bndry)
        return rms_bndry,z_offset
        
    def move_PF(self,coil,AR=0,**kwargs):
        name = self.Cname(coil)
        if 'point' in kwargs.keys() or 'Lout' in kwargs.keys()\
        or 'Lin' in kwargs.keys():
            r,z = self.get_point(**kwargs)
        elif 'delta' in kwargs.keys():
            ref,dr,dz = kwargs['delta']
            coil = self.sf.coil[self.Cname(ref)]
            rc,zc = coil['r'],coil['z']
            r,z = rc+dr,zc+dz
        elif 'L' in kwargs.keys():
            ref,dL = kwargs['dL']
            coil = self.sf.coil[self.Cname(ref)]
            rc,zc = coil['r'],coil['z'] 
        ARmax = 3
        if abs(AR) > ARmax: AR = ARmax*np.sign(AR)
        if AR > 0: 
            AR = 1+AR
        else:
            AR = 1/(1-AR)
        dA = self.sf.coil[name]['dr']*self.sf.coil[name]['dz']
        self.sf.coil[name]['dr'] = np.sqrt(AR*dA)
        self.sf.coil[name]['dz'] = self.sf.coil[name]['dr']/AR
        dr = r-self.sf.coil[name]['r']
        dz = z-self.sf.coil[name]['z']
        self.sf.coil[name]['r'] += dr
        self.sf.coil[name]['z'] += dz
        drTF,dzTF = self.rb.Cshift(self.sf.coil[name],'out',self.TFoffset)  # fit-TF
        self.sf.coil[name]['r'] += drTF
        self.sf.coil[name]['z'] += dzTF
        self.coil['active'][name]['r'] += dr+drTF
        self.coil['active'][name]['z'] += dz+dzTF
        for i in range(self.eq.coil[name+'_0']['Nf']):
            sub_name = name+'_{:1.0f}'.format(i)
            self.eq.coil[sub_name]['r'] += dr+drTF
            self.eq.coil[sub_name]['z'] += dz+dzTF
        self.update_bundle(name)
                
    def move_CS(self,name,z,dz):
        self.sf.coil[name]['z'] = z
        self.sf.coil[name]['dz'] = dz
        self.update_bundle(name)
        
    def tails(self,x,dx=0.1):  # contiuous PF position limit
        if x > (1-dx):
            y = (1-dx) + dx*(1-np.exp(-(x-(1-dx))))
        elif x < dx:
            y = dx - dx*(1-np.exp(-(dx-x)))
        else:
            y = x
        return y

    def update_position(self,Lo):
        nPF = len(self.PF_coils)
        Lpf = Lo[:nPF]
        for name,L in zip(self.PF_coils,Lpf):
            L = self.tails(L)
            #if L<0: L=0
            #if L>1: L=1
            r,z = self.TFoutR(L),self.TFoutZ(L)
            ref = int(name.split('Coil')[-1])
            self.move_PF(ref,point=(r,z))
        if len(Lo) > nPF:
            Z,L = Lo[nPF],Lo[nPF+1]
            self.Zcs = Z*self.CS_Znorm
            self.Lcs = L*self.CS_Lnorm
            dL = Lo[nPF+2:]
            Z,dZ = self.arange_CS(Z,L,dL)
            for name,z,dz in zip(self.CS_coils,Z,dZ):
                self.move_CS(name,z,dz)
        self.set_force_feild(state='active')  # update force feild
        
    def reset_current(self):
        for j,name in enumerate(self.coil['active'].keys()):
            self.sf.coil[name]['I'] = 0
        
    def update_area(self,relax=0.1,max_factor=0.2):
        for name in self.PF_coils:
            Ic = abs(self.sf.coil[name]['I'])
            if Ic > 150e6: Ic = 150e6  # coil area upper bound
            dA = Ic/12.5e6  # apply current density limit
            factor = dA/(self.sf.coil[name]['dr']*self.sf.coil[name]['dz'])-1
            if factor > max_factor: factor=max_factor
            scale = 1+relax*factor
            self.sf.coil[name]['dr'] *= scale
            self.sf.coil[name]['dz'] *= scale
            drTF,dzTF = self.rb.Cshift(self.sf.coil[name],'out',self.TFoffset)  # fit-TF
            self.sf.coil[name]['r'] += drTF
            self.sf.coil[name]['z'] += dzTF
            self.update_bundle(name)
                    
    def update_bundle(self,name):
        Nold = self.eq.coil[name+'_0']['Nf']
        bundle = self.eq.size_coil(self.sf.coil,name,self.eq.dCoil)
        N = self.eq.coil[name+'_0']['Nf']
        self.coil['active'][name] = {}
        for key in bundle.keys():
            self.coil['active'][name][key] = bundle[key]
        if Nold > N:
            for i in range(N,Nold):
                del self.eq.coil[name+'_{:1.0f}'.format(i)]

    def return_rms(self,I,*args):
        norm = args[0]
        self.I = I.reshape(-1,1)/norm  # solve
        self.get_rms()
        return self.rms
  
    def Ilimit(self,I,*args):
        norm,Imax,Ics,FzPF = args[0],args[1],args[2],args[3]
        FzCS,Fsep = args[4],args[5]
        ieqcon = np.zeros(len(I)+len(self.PF_coils))
        self.I = I.reshape(-1,1)/norm
        self.update_current()
        self.get_rms()  
        self.get_force()
        for i,name, in enumerate(self.coil['active'].keys()):
            if name in self.PF_coils:  # PF current limit
                ieqcon[i] = Imax-abs(self.Icoil[i])
            if name in self.CS_coils:  # CS current density limit
                dA = self.sf.coil[name]['dr']*self.sf.coil[name]['dz']
                ImaxCS = 12.5e6*dA
                ieqcon[i] = ImaxCS-abs(self.Icoil[i])
        for j,name in enumerate(self.PF_coils):  # PF vertical force limit
            if name in self.coil['active'].keys():
                Fz = abs(self.coil['active'][name]['Fz_sum'])
                ieqcon[i+j+1] = 1-Fz/FzPF
            else:
                ieqcon[i+j+1] = 0
        ieqcon = np.append(ieqcon,1-self.Ics/Ics)  # CS Isum    
        ieqcon = np.append(ieqcon,1-self.Fsep/Fsep)  # CS seperation force
        ieqcon = np.append(ieqcon,1-np.abs(self.FzCS)/FzCS)  # CS total force
        return ieqcon
        
    def solve_slsqp(self):
        norm,Imax,Ics,FzPF,FzCS,Fsep=5e-8,30e6,500e6,350e6,350e6,300e6  # 450e6
        #norm,Imax,Ics,FzPF,FzCS,Fsep=5e-8,30e6,500e6,250e6,250e6,250e6  # 450e6
        self.solve()  # sead 
        I,niter = op.fmin_slsqp(self.return_rms,self.I*norm,full_output=True,
                               args=(norm,Imax,Ics,FzPF,FzCS,Fsep),
                               f_ieqcons=self.Ilimit,
                               disp=0,iter=5e2)[:3:2]
        self.I = I.reshape(-1,1)/norm
        self.update_current()
        self.get_rms()
        return niter
        
    def copy_coil(self,coil_read):
        coil_copy = {}
        for strand in coil_read.keys():
            coil_copy[strand] = {}
            coil_copy[strand] = coil_read[strand].copy()
        return coil_copy
            
    def store_update(self,extent='full'):
        if extent=='full':
            for var in self.log_scalar:
                self.log[var].append(getattr(self,var))
            for var in self.log_array:
                self.log[var].append(getattr(self,var).copy())
            for label in ['current','plasma']:
                self.log[label+'_iter'].append(self.iter[label])
        else:
            self.log['plasma_coil'].append(self.copy_coil(self.eq.plasma_coil).copy())
            self.log['position_iter'].append(self.iter['position'])

    def position_coils(self,Lo,*args):
        get_fit = args[0]
        self.Lo = Lo
        self.iter['current'] == 0
        self.iter['position'] += 1
        self.update_position(Lo) 
        self.Swing = [-37.5,90.11] # [-25,105]  # -20,80
        
        Scentre = np.mean([-37.5,90.11])
        self.Swing = Scentre+np.array([-0.5,0.5])*363/(2*np.pi)
        nC,nS = len(self.adjust_coils),len(self.Swing)
        self.Iswing,rms = np.zeros((nS,nC)),np.zeros(nS)
        FrPF,FzPF = np.zeros(nS),np.zeros(nS)
        Tpol = np.zeros(nS)
        rms_bndry,z_offset = np.zeros(nS),np.zeros(nS)
        for i,swing in enumerate(self.Swing):
            self.swing_fix(swing)  
            niter = self.solve_slsqp()
            self.iter['current'] += niter
            Tpol[i] = self.poloidal_angle()
            self.Iswing[i],rms[i] = self.Icoil,self.rms
            FrPF[i],FzPF[i] = self.FrPF,self.FzPF
            if get_fit: rms_bndry[i],z_offset[i] = self.get_rms_bndry()
        self.dTpol = Tpol[-1]-Tpol[0]
        self.FrPF,self.FzPF = np.max(abs(FrPF)),np.max(abs(FzPF))
        Icoil = self.Iswing[np.argmax(abs(self.Iswing),axis=0),range(nC)]
        self.Isum,self.Imax = np.sum(abs(Icoil)),max(abs(Icoil))
        self.rmsID = np.argmax(rms)
        self.rms = rms[self.rmsID]
        self.rms_bndry = np.max(rms_bndry)
        self.z_offset = np.mean(z_offset)
        for name,I in zip(self.coil['active'].keys(),Icoil):
            if name in self.PF_coils: 
                self.sf.coil[name]['I'] = I # set maximum current (for area)
        self.update_area()  
        self.store_update()
        return self.rms
        
    def poloidal_angle(self):  # calculate poloidal feild line angle
        if 'Bdir' in self.fix['BC']:
            index = self.fix['BC']=='Bdir'  # angle for last Bdir co-location
            B = self.eq.Bfeild([self.fix['r'][index][-1],self.fix['z'][index][-1]])
            Tpol = 180*np.arctan2(B[1],B[0])/np.pi  # angle in degrees
        else:
            Tpol = 0
        return Tpol
            
    def PFspace(self,Lo,*args):
        Lpf = Lo[:len(self.PF_coils)]
        for i,L in enumerate(Lpf):
            Lpf[i] = self.tails(L)
        dL = np.zeros(len(self.PF_coils))
        dLmin = 2e-2
        for i,l in enumerate(Lpf):
            L = np.append(Lpf[:i],Lpf[i+1:])
            dL[i] = np.min(abs(l-L))
        fun = np.min(dL)-dLmin
        return fun
        
    def CSlength(self,Lo,*args):
        nPF = len(self.PF_coils)
        L = Lo[nPF+1]
        fun = 30-self.CS_Lnorm*L
        return fun
        
    def CSzmax(self,Lo,*args):
        zmax = 9
        nPF = len(self.PF_coils)
        Z = self.CS_Znorm*Lo[nPF]
        L = self.CS_Lnorm*Lo[nPF+1]
        Zmax = Z+L/2
        fun = zmax-Zmax
        return fun
        
    def CSzmin(self,Lo,*args):
        zmin = -14
        nPF = len(self.PF_coils)
        Z = self.CS_Znorm*Lo[nPF]
        L = self.CS_Lnorm*Lo[nPF+1]
        Zmin = Z-L/2
        fun = Zmin-zmin
        return fun
        
    def minimize(self,Lo,rhobeg=0.1,rhoend=1e-4):  # rhobeg=0.1,rhoend=1e-4
        self.iter['position'] == 0 
        Lo = op.fmin_cobyla(self.position_coils,Lo,
                            [self.PFspace,self.CSzmax,self.CSzmin],args=([False]),
                            rhobeg=rhobeg,rhoend=rhoend)  # ,disp=3,maxfun=30
        rms = self.position_coils(Lo,True)
        text = 'Isum {:1.0f} Ics {:1.0f}'.format(self.Isum*1e-6,self.Ics*1e-6)
        text += ' rms {:1.4f} rmsID {:1.0f}'.format(self.rms,self.rmsID)
        text += ' rms_bndry {:1.3f}'.format(self.rms_bndry)
        text += ' z_offset {:1.3f}'.format(self.z_offset)
        text += ' dTpol {:1.3f}'.format(self.dTpol)
        #text += ' Lcs {:1.2f} rb {:1.3e} re {:1.3e}'.format(self.Lcs,rhobeg,
        #                                                     rhoend)
        print(text)
        #Powell,Nelder-Mead,COBYLA,SLSQP
        return Lo,rms
        
    def tick(self):
        self.tloop = time.time()
        self.tloop_cpu = time.process_time()
        
    def tock(self):
        tw,tcpu = time.time(),time.process_time()
        dt = tw-self.tloop
        self.ttotal = tw-self.tstart
        dt_cpu = tcpu-self.tloop_cpu
        self.ttotal_cpu = tcpu-self.tstart_cpu
        dfcpu,self.fcpu = dt_cpu/dt,self.ttotal_cpu/self.ttotal
        print('dt {:1.0f}s speedup {:1.0f}%'.format(dt,100*dfcpu/self.ncpu))
        print('total {:1.0f}s speedup {:1.0f}%'.format(self.ttotal,
                                                     100*self.fcpu/self.ncpu))
        print('')
        
    def initialize_log(self):
        self.log = {}
        for var in self.log_scalar+self.log_array+self.log_iter+self.log_plasma:
            self.log[var] = []
        self.iter = {'plasma':0,'position':0,'current':0}
            
    def optimize(self,Lo,Nmax=20):
        fun = 0
        rhobeg = [0.4,0.2]  # start,min
        rhoend = [5e-2,1e-4]  # start,min
        self.initialize_log()
        self.Vtarget = self.sf.Mpoint[1]
        self.initalise_plasma = True
        self.ttotal,self.ttotal_cpu = 0,0
        self.tstart,self.tstart_cpu = time.time(),time.process_time()
        for _ in range(Nmax):
            self.iter['plasma'] += 1
            self.tick()
            fun_o = fun
            self.set_plasma()
            Lo,fun = self.minimize(Lo,rhobeg=rhobeg[0],rhoend=rhoend[0])
            self.store_update(extent='position')
            self.tock()
            rhobeg = self.rhobound(rhobeg)
            rhoend = self.rhobound(rhoend)
            err = abs(fun-fun_o) 
            if err < 1e-3:
                break
        self.Lo = Lo
        return fun,Lo
        
    def rhobound(self,rho):
        if rho[0] > rho[1]:
            rho[0] *= 0.75
        else:
            rho[0] = rho[1]
        return rho

    def optimize_rms(self,Lo):
        self.set_plasma()
        Lo = self.mix_coils(Lo,-0.75)[0]  # inital fit

    def set_plasma(self):
        if self.initalise_plasma:
            self.eq.plasma()
            self.initalise_plasma = False
        else:
            self.swing_fix(np.mean(self.Swing))
            self.solve_slsqp()
            self.eq.get_Vcoil()  # select vertical stability coil
            self.eq.gen(Vtarget=self.Vtarget)
        self.set_background()
        self.get_weight()
        self.set_force_feild(state='both')

    def update_current(self):
        self.Isum,self.Ics = 0,0
        self.Icoil = np.zeros(len(self.coil['active'].keys()))
        for j,name in enumerate(self.coil['active'].keys()):
            Nfilament = self.eq.coil[name+'_0']['Nf']
            self.sf.coil[name]['I'] = self.I[j][0]*Nfilament  # update sf 
            self.coil['active'][name]['I_sum'] = self.I[j][0]*Nfilament  # inv
            self.Icoil[j] = self.sf.coil[name]['I']  # store current
            for i in range(Nfilament):
                sub_name = name+'_{:1.0f}'.format(i)
                self.eq.coil[sub_name]['I'] = self.I[j][0]  # update eq
                self.coil['active'][name]['I'][i] = self.I[j][0]  # update inv
            self.Isum += abs(self.Icoil[j])  # sum absolute current
            if name in self.CS_coils:
                self.Ics += abs(self.Icoil[j])
        self.Imax = self.Icoil.max()
        
    def initalise_current(self):
        adjust_coils = self.coil['active'].keys()
        self.I = np.zeros((len(adjust_coils),1))
        for i,name in enumerate(adjust_coils):
            self.I[i,0] = self.eq.coil[name+'_0']['I']

    def update_coils(self,plot=False):
        self.coil = {'active':od(),'passive':od()}
        self.append_coil(self.eq.coil)
        self.append_coil(self.eq.plasma_coil)
        if plot:
            self.plot_coils()
            
    def write_swing(self):
        nC,nS = len(self.adjust_coils),len(self.Swing)
        Icoil = np.zeros((nS,nC))
        for i,swing in enumerate(self.Swing[::-1]):
            self.swing_fix(swing)  
            self.solve_slsqp()
            Icoil[i] = self.Icoil
        with open('../Data/'+self.sf.conf.dataname+'_currents.txt','w') as f:
            f.write('Coil\tr [m]\tz [m]\tdr [m]\tdz [m]')
            f.write('\tI SOF [A]\tI EOF [A]\n')
            for j,name in enumerate(self.coil['active'].keys()):
                r = float(self.sf.coil[name]['r'])
                z = float(self.sf.coil[name]['z'])
                dr = self.sf.coil[name]['dr']
                dz = self.sf.coil[name]['dz']
                position = '\t{:1.3f}\t{:1.3f}'.format(r,z)
                size = '\t{:1.3f}\t{:1.3f}'.format(dr,dz)
                current = '\t{:1.3f}\t{:1.3f}\n'.format(Icoil[0,j],Icoil[1,j])
                f.write(name+position+size+current)