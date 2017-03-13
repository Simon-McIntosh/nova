import numpy as np
import pylab as pl
import nova.cross_coil as cc
from collections import OrderedDict
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import interp1d
import seaborn as sns
import scipy.optimize as op
from itertools import cycle
import copy
import time
import multiprocessing
from nova import loops
import nlopt
from nova.force import force_feild
from scipy.optimize import minimize_scalar
from copy import deepcopy
from warnings import warn
import sys
from scipy.optimize import brentq
from amigo import geom
from nova.elliptic import EQ

class INV(object):
    
    def __init__(self,pf,tf=None,Jmax=12.5,TFoffset=0.3,svd=False,
                 Iscale=1e6,dCoil=1.5):
        self.wsqrt = 1
        self.svd = svd  # coil current singular value decomposition flag
        self.Iscale = Iscale  # set current units to MA
        self.ncpu = multiprocessing.cpu_count()
        self.pf = pf
        self.dCoil = dCoil
        self.pf.grid_coils(self.dCoil)
        self.tf = tf
        self.TFoffset = TFoffset  # offset coils from outer TF loop
        if tf is not None:
            self.tf.loop_interpolators(offset=self.TFoffset)
        self.Jmax = Jmax  # MAm-2
        self.fix = {'r':np.array([]),'z':np.array([]),'BC':np.array([]),
                    'value':np.array([]),'Bdir':np.array([[],[]]).T,
                    'factor':np.array([]),'n':0}  
        self.cnstr = {'r':np.array([]),'z':np.array([]),'BC':np.array([]),
                     'value':np.array([]),'condition':np.array([])}  
        self.add_active([],empty=True)  # all coils active
        self.update_coils()
        self.log_scalar = ['rms','rmsID']  
        #'Fr_max','Fz_max','F_max','FzCS','Fsep','rms_bndry','Isum',
        #'Imax','z_offset','FrPF','FzPF','dTpol'
        self.log_array = ['Lo']  # ,'Iswing'
        self.log_iter = ['current_iter','plasma_iter','position_iter']
        self.log_plasma = ['plasma_coil']
        self.CS_Lnorm,self.CS_Znorm = 2,1
        self.force_feild_active = False
    
    def load_equlibrium(self,sf,expand=1.05,n=2.5e3):
        self.eq = EQ(sf,self.pf,dCoil=self.dCoil,
                boundary=sf.get_sep(expand=expand),n=n) 
        
    def set_force_feild(self):
        self.ff = force_feild(self.pf.index,self.pf.coil,
                              self.pf.sub_coil,self.pf.plasma_coil,
                              multi_filament=True)
        
    def set_swing(self,centre=0,width=363/(2*np.pi),array=[-0.5,0,0.5]): 
        flux = centre-np.array(array)*width  # Webber/rad 
        nC,nS = self.nC,len(flux)
        self.swing = {'flux':flux,'I':np.zeros((nS,nC)),'rms':np.zeros(nS),
                      'FrPF':np.zeros(nS),'FzPF':np.zeros(nS),
                      'FsepCS':np.zeros(nS),'FzCS':np.zeros(nS),   
                      'Isum':np.zeros(nS),
                      'IsumCS':np.zeros(nS),'IsumPF':np.zeros(nS),
                      'Imax':np.zeros(nS),
                      'Tpol':np.zeros(nS),'rms_bndry':np.zeros(nS),
                      'z_offset':np.zeros(nS),'dTpol':0,
                      'Fcoil':[[] for _ in range(nS)]}      
                      
    def initalise_limits(self):
        self.limit = {'I':{},'L':[],'F':[]}
        self.limit['I'] = {'PF':50,'CS_sum':500}  # MA 
        self.limit['L'] = {'CS':[-15,15],'PF':[0.05,0.95]}  # m / fraction
        self.limit['F'] = {'PFz':450,'CSz_sum':350,'CSz_sep':300}  # MN
        self.set_PF_limit()
        
    def set_PF_limit(self):
        for coil in self.PF_coils: 
            self.limit['L'][coil] = self.limit['L']['PF']
            
    def update_limits(self,**kwargs):  # set as ICS_sum for [I][CS_sum] etc...
        for key in kwargs:
            variable = key[0]
            if key[1:] in self.limit[variable]:
                self.limit[variable][key[1:]] = kwargs[key]
        if 'LPF' in kwargs:
            self.set_PF_limit()
                        
    def Cname(self,coil):
        return 'Coil{:1.0f}'.format(coil)
        
    def initalise_coil(self):
        self.nC = len(self.adjust_coils)  # number of active coils
        self.nPF = len(self.PF_coils)
        self.nCS = len(self.CS_coils)
        self.coil = {'active':OrderedDict(),'passive':OrderedDict()}
        names = self.pf.coil.keys()
        index = []
        for name in names:
            index.append(int(name.split('Coil')[-1]))
        self.all_coils = ['Coil{:1.0f}'.format(i) for i in np.sort(index)]
        names = np.append(self.all_coils,'Plasma')
        for name in names:
            if name in self.adjust_coils:
                state = 'active'
            else:
                state = 'passive'
            self.coil[state][name] = {'r':np.array([]),'z':np.array([]),\
            'dr':np.array([]),'dz':np.array([]),'sub_name':np.array([]),\
            'I':np.array([]),'Fr':np.array([]),'Fz':np.array([]),\
            'Fr_sum':0,'Fz_sum':0,'I_sum':0,'Ro':0,'Zo':0,'Nf':0}
            
    def initalise_current(self):
        adjust_coils = self.coil['active'].keys()
        self.I = np.zeros((len(adjust_coils)))  
        for i,name in enumerate(adjust_coils):
            self.I[i] = self.pf.sub_coil[name+'_0']['I']/self.Iscale
        self.alpha = np.zeros(self.nC)  # initalise alpha
        
    def update_coils(self,plot=False):  # full coil update
        self.initalise_coil()
        self.append_coil(self.pf.sub_coil)
        self.append_coil(self.pf.plasma_coil)
        for coil in self.all_coils:
            self.update_bundle(coil)  # re-grid elliptic coils
        self.pf.categorize_coils()  # index pf coils
        self.initalise_current()
        self.initalise_limits()
        self.set_swing()
        if plot:
            self.plot_coils()
            
    def append_coil(self,coils):
        for sub_name in coils.keys():
            name = sub_name.split('_')[0]
            if name in self.adjust_coils:
                state = 'active'
            else:
                state = 'passive'
            coil = coils[sub_name]
            for key,var in zip(['r','z','I','dr','dz','sub_name'],
                               [coil['r'],coil['z'],coil['I'],
                                coil['dr'],coil['dz'],sub_name]):
                self.coil[state][name][key] = \
                np.append(self.coil[state][name][key],var)
            self.coil[state][name]['Isum'] = np.sum(self.coil[state][name]['I'])
            self.coil[state][name]['Ro'] = np.mean(self.coil[state][name]['r'])
            self.coil[state][name]['Zo'] = np.mean(self.coil[state][name]['z'])
            self.coil[state][name]['Nf'] = len(self.coil[state][name]['z'])
          
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
            self.adjust_coils = list(self.pf.coil.keys())  # add all
            self.PF_coils = list(self.pf.index['PF']['name'])
            self.CS_coils = list(self.pf.index['CS']['name'])

    def remove_active(self,Clist=[],Ctype='all',full=False):  # list of coil names
        if full: 
            self.adjust_coils = list(self.pf.coil.keys())  # add all
            if Ctype == 'PF':
                self.PF_coils = list(self.pf.coil.keys())
            elif Ctype == 'CS':
                self.CS_coils = list(self.pf.coil.keys())  
        if len(Clist)>0:
            for name in Clist.copy():
                if isinstance(name,int):
                    name = self.Cname(name)  # prepend 'Coil'
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
        
    def delete_active(self):
        Clist = []
        for name in self.coil['active'].keys():
            Clist.append(int(name.split('Coil')[-1]))
        self.remove_active(Clist,Ctype='all')
        self.remove_coil(Clist)

    def remove_coil(self,Clist):
        for coil in Clist:
            name = self.Cname(coil)
            if name in self.pf.coil.keys():
                del self.pf.coil[name]
                for i in range(self.pf.sub_coil[name+'_0']['Nf']):
                    del self.pf.sub_coil[name+'_{:1.0f}'.format(i)]
        self.remove_active(Clist)
        self.update_coils()
   
    def get_point(self,**kwargs):
        keys = kwargs.keys()
        if 'point' in keys:
            r,z = kwargs['point']
        elif 'polar' in keys:
            mag,arg = kwargs['polar']
            r = self.eq.sf.Xpoint[0]+mag*np.sin(arg*np.pi/180)
            z = self.eq.sf.Xpoint[1]-mag*np.cos(arg*np.pi/180)
        elif 'Lout' in keys:
            L = kwargs['Lout']
            r,z = self.tf.fun['out']['r'](L),self.tf.fun['out']['z'](L)
            dr,dz = self.tf.fun['out']['dr'](L),self.tf.fun['out']['dz'](L)
        elif 'Lin' in keys:
            L = kwargs['Lin']
            r,z = self.tf.fun['in']['r'](L),self.tf.fun['in']['z'](L)
            dr,dz = self.tf.fun['in']['dr'](L),self.tf.fun['in']['dz'](L)
        if 'norm' in keys and 'point' not in keys:        
            delta = kwargs['norm']*np.array([dr,dz])/np.sqrt(dr**2+dz**2)
            r += delta[1]
            z -= delta[0]
        return r,z

    def plot_coils(self):
        Color = cycle(sns.color_palette('Set2',12))
        for state,marker in zip(['passive','active'],['o','d']):
            for name in self.coil[state].keys():
                if name != 'Plasma':
                    color = next(Color)
                    for r,z in zip(self.coil[state][name]['r'],
                                   self.coil[state][name]['z']):
                        pl.plot(r,z,marker,color=color,markersize=4)
                        if state == 'passive':  # empty marker
                            pl.plot(r,z,marker,color='w',markersize=2)
                        
            
    def add_fix(self,r,z,value,Bdir,BC,factor):
        var = {'r':r,'z':z,'value':value,'Bdir':Bdir,'BC':BC,'factor':factor}
        nvar = len(r)
        self.fix['n'] += nvar
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
                
    def fix_flux(self,flux):
        if not hasattr(self,'fix_o'):
            self.fix_o = copy.deepcopy(self.fix)
        for i,(bc,value) in enumerate(zip(self.fix_o['BC'],
                                          self.fix_o['value'])):
            if 'psi' in bc:
                self.fix['value'][i] = value+flux
        self.set_target()  # adjust target flux
        
    def set_boundary(self,N=21,alpha=0.995):
        r,z = self.eq.sf.get_boundary(alpha=alpha)
        self.eq_spline()
        nr = -self.psi.ev(r,z,dx=1,dy=0)
        nz = -self.psi.ev(r,z,dx=0,dy=1)
        L = geom.length(r,z)
        Lc = np.linspace(0,1,N+1)[:-1]
        rb,zb = interp1d(L,r)(Lc),interp1d(L,z)(Lc)
        Bdir = np.array([interp1d(L,nr)(Lc),interp1d(L,nz)(Lc)]).T
        return rb,zb,Bdir
        
    def get_psi(self,alpha):
        Xpsi = self.eq.sf.Xpsi  
        Mpsi = self.eq.sf.Mpsi  
        psi = Mpsi+alpha*(Xpsi-Mpsi)
        return psi
        
    def fix_boundary_psi(self,N=21,alpha=0.995,factor=1):
        r,z,Bdir = self.set_boundary(N=N,alpha=alpha)
        psi = self.get_psi(alpha)*np.ones(N)
        psi -= self.eq.sf.Xpsi  # normalise
        self.add_fix(r,z,psi,Bdir,['psi_bndry'],[factor])

    def fix_boundary_feild(self,N=21,alpha=0.995,factor=1,**kwargs):
        r,z,Bdir = self.set_boundary(N=N,alpha=alpha)
        #Bdir = kwargs.get('Bdir',Bdir)
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
        Bm = np.abs(self.eq.sf.bcentr*self.eq.sf.rcentr)  # toroidal moment
        Bphi = Bm/r  # torodal field
        Bp = Bphi/np.sqrt((np.sin(theta*np.pi/180)/
                           np.sin(graze*np.pi/180))**2)
        self.add_fix([r],[z],[Bp],np.array([[0],[0]]).T,['Bp'],[factor])

    def add_psi(self,psi,factor=1,**kwargs):
        r,z = self.get_point(**kwargs)
        label = kwargs.get('label','psi')
        self.add_fix([r],[z],[psi],np.array([[0],[0]]).T,[label],[factor])
        
    def add_alpha(self,alpha,factor=1,**kwargs):
        psi = self.get_psi(alpha)
        psi -= self.eq.sf.Xpsi  # normalise
        self.add_psi(psi,factor=factor,**kwargs)
        
    def add_Bcon(self,B,**kwargs):
        r,z = self.get_point(**kwargs)
        self.add_cnstr([r],[z],['B'],['gt'],[B])
        
    def plot_fix(self,tails=True):
        self.get_weight()
        if hasattr(self,'wsqrt'):
            weight = self.wsqrt/np.mean(self.wsqrt)
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
            if bdir[0]**2+bdir[1]**2 == 0:  # tails  
                direction = [0,-1]
            else:
                direction = bdir
            #else:
            #    d_dr,d_dz = self.get_gradients(bc,r,z)
            #    direction = np.array([d_dr,d_dz])/np.sqrt(d_dr**2+d_dz**2)
            if 'psi' in bc:
                norm = psi_norm
                marker,size,color = 'o',7.5,Color[0]
                pl.plot(r,z,marker,color=color,markersize=size)
                pl.plot(r,z,marker,color=[1,1,1],markersize=0.3*size)
            else:
                if bc == 'Bdir':
                    norm = Bdir_norm
                    marker,size,color,mew = 'o',4,Color[1],0.0
                elif bc == 'null':
                    norm = Brz_norm
                    marker,size,color,mew = 'o',2,Color[2],0.0
                elif bc == 'Br':
                    norm = Brz_norm
                    marker,size,color,mew = '_',10,Color[2],1.75
                elif bc == 'Bz':
                    norm = Brz_norm
                    marker,size,color,mew = '|',10,Color[2],1.75
                pl.plot(r,z,marker,color=color,markersize=size,
                        markeredgewidth=mew)
            if tails:
                pl.plot([r,r+direction[0]*norm*w],[z,z+direction[1]*norm*w],
                        color=color,linewidth=2)

    def set_target(self):
        self.T = (self.fix['value']-self.BG).reshape((len(self.BG),1)) 
        self.wT = self.wsqrt*self.T

    def set_background(self):
        self.BG = np.zeros(len(self.fix['BC']))  # background 
        self.eq_spline()  # generate equlibrium interpolators
        self.plasma_spline()  # generate plasma interpolators
        self.add_value('passive')
                 
    def set_foreground(self):
        self.G = np.zeros((len(self.fix['BC']),
                           len(self.coil['active'].keys())))  # [G][I] = [T] 
        self.add_value('active')
        self.wG = self.wsqrt*self.G
        if self.svd:  # singular value dec.
            self.U,self.S,self.V = np.linalg.svd(self.wG)
            self.wG = np.dot(self.wG,self.V)  # solve in terms of alpha

    def add_value(self,state):
        Rf,Zf,BC,Bdir = self.unpack_fix()
        for i,(rf,zf,bc,bdir) in enumerate(zip(Rf,Zf,BC,Bdir)):
            for j,name in enumerate(self.coil[state].keys()):
                #if name == 'Plasma' and False:
                #    self.BG[i] += self.add_value_plasma(bc,rf,zf,bdir)
                #else:
                coil = self.coil[state][name]
                R,Z,Ic = coil['r'],coil['z'],coil['I']
                dR,dZ = coil['dr'],coil['dz']
                for r,z,ic,dr,dz in zip(R,Z,Ic,dR,dZ):
                    value = self.add_value_coil(bc,rf,zf,r,z,bdir,dr,dz)
                    if state == 'active':
                        self.G[i,j] += value
                    elif state == 'passive':
                        self.BG[i] += ic*value/self.Iscale
                    else:
                        errtxt = 'specify coil state'
                        errtxt += '\'active\', \'passive\'\n'
                        raise ValueError(errtxt)
                            
    def add_value_coil(self,bc,rf,zf,r,z,bdir,dr,dz):
        if 'psi' in bc:
            value = self.Iscale*cc.mu_o*cc.green(rf,zf,r,z,dRc=dr,dZc=dz)  
        else:
            B = self.Iscale*cc.mu_o*cc.green_feild(rf,zf,r,z)
            value = self.Bfeild(bc,B[0],B[1],bdir)
        return value
        
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
        self.eq.plasma()  # evaluate scalar potential with plasma only
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
        try:
            if 'psi' in bc:
                d_dr = self.psi.ev(rf,zf,dx=1,dy=0)
                d_dz = self.psi.ev(rf,zf,dx=0,dy=1)
            else:
                d_dr = self.B.ev(rf,zf,dx=1,dy=0)
                d_dz = self.B.ev(rf,zf,dx=0,dy=1)
        except:
            warn('gradient evaluation failed')
            d_dr,d_dz = np.ones(len(bc)),np.ones(len(bc))
        return d_dr,d_dz
        
    def get_weight(self):
        Rf,Zf,BC,Bdir = self.unpack_fix()
        weight = np.zeros(len(BC))
        for i,(rf,zf,bc,bdir,factor) in enumerate(zip(Rf,Zf,BC,Bdir,
                                                      self.fix['factor'])):
            d_dr,d_dz = self.get_gradients(bc,rf,zf)
            if 'psi' not in bc:  # (Br,Bz)
                weight[i] = 1/abs(np.sqrt(d_dr**2+d_dz**2))
            elif bc == 'psi_bndry':
                weight[i] = 1/abs(np.dot([d_dr,d_dz],bdir))
        if 'psi_bndry' in self.fix['BC']:
            wbar = np.mean([weight[i] for i,bc in enumerate(self.fix['BC']) \
                                    if bc == 'psi_bndry'])
        else:
            wbar = np.mean(weight)
        for i,bc in enumerate(BC):
            if bc == 'psi_x' or bc == 'psi':  # psi point weights
                weight[i] = wbar  # mean boundary weight
        if (weight==0).any():
            warn('fix weight entry not set')

        factor = np.reshape(self.fix['factor'],(-1,1))
        weight = np.reshape(weight,(-1,1))
        self.wsqrt = np.sqrt(factor*weight)
        
    def arange_CS(self,Z):  # ,dZmin=0.01
        Z = np.sort(Z)
        dZ = abs(np.diff(Z)-self.gap)
        Zc = np.zeros(self.nCS)
        for i in range(self.nCS):
            Zc[i] = Z[i]+self.gap/2+dZ[i]/2
        return Zc,dZ
        
    def add_coil(self,Ctype=None,**kwargs):
        r,z = self.get_point(**kwargs)
        index,i = np.zeros(len(self.pf.coil.keys())),-1
        for i,name in enumerate(self.pf.coil.keys()):
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
        I = kwargs.get('I',1e6)
        self.pf.coil[name] = {'r':r,'z':z,
                                 'dr':delta[0],'dz':delta[1],'I':I,
                                 'rc':np.sqrt(delta[0]**2+delta[1]**2)/2}
        
    def grid_CS(self,nCS=3,Ro=2.9,Zbound=[-10,9],dr=0.818,gap=0.1,fdr=1):  #dr=1.0, dr=1.25,Ro=3.2,dr=0.818
        self.gap = gap
        dr *= fdr  # thicken CS
        Ro -= dr*(fdr-1)/2  # shift centre inwards
        dz = (np.diff(Zbound)-gap*(nCS-1))/nCS  # coil height
        Zc = np.linspace(Zbound[0]+dz/2,Zbound[-1]-dz/2,nCS)  # coil centres
        for zc in Zc:
            self.add_coil(point=(Ro,zc),dr=dr,dz=dz,Ctype='CS')
        Ze = np.linspace(Zbound[0]-gap/2,Zbound[-1]+gap/2,nCS+1)  # coil edges    
        return Ze

    def grid_PF(self,nPF=5):
        dL = 1/nPF
        Lo = np.linspace(dL/2,1-dL/2,nPF)
        self.delete_active()
        for L in Lo:
            self.add_coil(Lout=L,Ctype='PF',norm=self.TFoffset)    
        return Lo
        
    def grid_coils(self,nCS=None,Zbound=None,gap=0.1,offset=0.3):  # fit PF and CS coils to updated TF coil
        self.gap = gap
        coil = deepcopy(self.pf.coil)  # copy pf coilset
        index = deepcopy(self.pf.index)
        self.delete_active()
        TFloop = self.tf.fun['out']
        def norm(L,loop,point):
            return (loop['r'](L)-point[0])**2+(loop['z'](L)-point[1])**2
        Lpf = np.zeros(index['PF']['n'])
        for i,name in enumerate(index['PF']['name']):
            c = coil[name]
            Lpf[i] = minimize_scalar(norm,method='bounded',
                        args=(TFloop,(c['r'],c['z'])),bounds=[0,1]).x
            self.add_coil(Lout=Lpf[i],Ctype='PF',norm=self.TFoffset,
                          dr=c['dr'],dz=c['dz'],I=c['I']) 
        '''
        if Zbound == None:
            zmin = [coil[name]['z']-coil[name]['dz']/2 
                    for name in self.pf.index['CS']['name']]
            zmax = [coil[name]['z']+coil[name]['dz']/2 
                    for name in self.pf.index['CS']['name']]
            Zbound = [np.min(zmin)-gap/2,np.max(zmax)+gap/2]
            self.update_limits(LCS=Zbound)   
        '''
        if nCS == None:
            Lcs = np.zeros(index['CS']['n']+1)
            for i,name in enumerate(index['CS']['name']):
                c = coil[name]
                self.add_coil(point=(c['r'],c['z']),Ctype='CS',
                              dr=c['dr'],dz=c['dz'],I=c['I'])
                if i == 0:
                    Lcs[0] = c['z']-c['dz']/2-gap/2
                Lcs[i+1] = c['z']+c['dz']/2+gap/2       
        else:
            nCS = index['CS']['n']
            Lcs = self.grid_CS(nCS=nCS,Zbound=Zbound,gap=gap)
        self.update_coils()
        self.fit_PF(offset=offset)
        return np.append(Lpf,Lcs)
                                
    def get_rms_bndry(self):
        self.eq.run(update=False)
        psi_o = self.fix['value'][0]
        psi_line = self.eq.sf.get_contour([psi_o])[0]
        dx_bndry,dx_min = np.array([]),0 
        dz_bndry,dz_min = np.array([]),0 
        Rf,Zf,BC = self.unpack_fix()[:3]
        for rf,zf,bc,psi in zip(Rf,Zf,BC,self.fix['value']):
            if bc == 'psi_bndry':
                if psi != psi_o:  # update contour
                    psi_o = psi
                    psi_line = self.eq.sf.get_contour([psi_o])[0]
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
            coil = self.pf.coil[self.Cname(ref)]
            rc,zc = coil['r'],coil['z']
            r,z = rc+dr,zc+dz
        elif 'L' in kwargs.keys():
            ref,dL = kwargs['dL']
            coil = self.pf.coil[self.Cname(ref)]
            rc,zc = coil['r'],coil['z'] 
        ARmax = 3
        if abs(AR) > ARmax: AR = ARmax*np.sign(AR)
        if AR > 0: 
            AR = 1+AR
        else:
            AR = 1/(1-AR)
        dA = self.pf.coil[name]['dr']*self.pf.coil[name]['dz']
        self.pf.coil[name]['dr'] = np.sqrt(AR*dA)
        self.pf.coil[name]['dz'] = self.pf.coil[name]['dr']/AR
        dr = r-self.pf.coil[name]['r']
        dz = z-self.pf.coil[name]['z']
        self.shift_coil(name,dr,dz)
    
    def fit_PF(self,**kwargs):  # offset PF coils from TF
        offset = kwargs.get('offset',self.TFoffset)
        dl = 0
        for name in self.PF_coils:
            dr,dz = self.tf.Cshift(self.pf.coil[name],'out',offset)
            dl += dr**2+dz**2
            self.shift_coil(name,dr,dz)
        return np.sqrt(dl/self.nPF)
        
    def shift_coil(self,name,dr,dz):
        self.pf.coil[name]['r'] += dr
        self.pf.coil[name]['z'] += dz
        self.coil['active'][name]['r'] += dr
        self.coil['active'][name]['z'] += dz
        for i in range(self.pf.sub_coil[name+'_0']['Nf']):
            sub_name = name+'_{:1.0f}'.format(i)
            self.pf.sub_coil[sub_name]['r'] += dr
            self.pf.sub_coil[sub_name]['z'] += dz
        self.update_bundle(name) 
        
    def move_CS(self,name,z,dz):
        self.pf.coil[name]['z'] = z
        self.pf.coil[name]['dz'] = dz
        self.update_bundle(name)

    def reset_current(self):
        for j,name in enumerate(self.coil['active'].keys()):
            self.pf.coil[name]['I'] = 0
        
    def update_bundle(self,name):
        try:
            Nold = self.pf.sub_coil[name+'_0']['Nf']
        except:
            Nold = 0
        bundle = self.pf.size_coil(name,self.dCoil)
        N = self.pf.sub_coil[name+'_0']['Nf']
        self.coil['active'][name] = {}
        for key in bundle.keys():
            self.coil['active'][name][key] = bundle[key]
        if Nold > N:
            for i in range(N,Nold):
                del self.pf.sub_coil[name+'_{:1.0f}'.format(i)]
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

    def poloidal_angle(self):  # calculate poloidal feild line angle
        if 'Bdir' in self.fix['BC']:
            index = self.fix['BC']=='Bdir'  # angle for last Bdir co-location
            B = self.eq.Bpoint([self.fix['r'][index][-1],self.fix['z'][index][-1]])
            Tpol = 180*np.arctan2(B[1],B[0])/np.pi  # angle in degrees
        else:
            Tpol = 0
        return Tpol
                  
    def PFspace(self,Lo,*args):
        Lpf = Lo[:self.nPF]
        dL = np.zeros(self.nPF)
        dLmin = 2e-2
        for i,l in enumerate(Lpf):
            L = np.append(Lpf[:i],Lpf[i+1:])
            dL[i] = np.min(abs(l-L))
        fun = np.min(dL)-dLmin
        return fun
        
    def get_rms(self,vector,error=False):
        err = np.dot(self.wG,vector.reshape(-1,1))-self.wT
        rms = np.sqrt(np.mean(err**2))
        if error:  # return error
            return rms,err
        else:
            return rms
        
    def solve(self):
        vector = np.linalg.lstsq(self.wG,self.wT)[0].reshape(-1)
        self.rms = self.get_rms(vector)
        if self.svd:
            self.I = np.dot(self.V,self.alpha)
        else:
            self.I = vector
        self.update_current()

    def frms(self,vector,grad):
        self.iter['current'] += 1
        rms,err = self.get_rms(vector,error=True)
        if grad.size>0:
            jac = np.ones(self.nC)/(rms*self.fix['n'])
            for i in range(self.nC):
                jac[i] *= np.dot(self.wG[:,i],err)
            grad[:] = jac
        return rms
    
    def Ilimit(self,constraint,alpha,grad):  # only for svd==True
        if grad.size > 0:
            grad[:self.nC] = -self.V  # lower bound
            grad[self.nC:] = self.V  # upper bound
        I = np.dot(self.V,alpha)
        constraint[:self.nC] = self.Io['lb']-I  # lower bound
        constraint[self.nC:] = I-self.Io['ub']  # upper bound

    def set_Io(self):  # set lower/upper bounds on coil currents (Jmax)
        self.Io = {'name':self.adjust_coils,'value':np.zeros(self.nC),
                   'lb':np.zeros(self.nC),'ub':np.zeros(self.nC)}
        for name in self.adjust_coils:  # limits in MA
            i = int(name.replace('Coil',''))
            coil = self.pf.coil[name]
            Nf = self.pf.sub_coil[name+'_0']['Nf']  # fillament number
            self.Io['value'][i] = coil['I']/(Nf*self.Iscale)
            if name in self.PF_coils:
                self.Io['lb'][i] = -self.limit['I']['PF']/Nf
                self.Io['ub'][i] = self.limit['I']['PF']/Nf
            if name in self.CS_coils:
                Ilim = coil['dr']*coil['dz']*self.Jmax
                self.Io['lb'][i] = -Ilim/Nf
                self.Io['ub'][i] = Ilim/Nf
        lindex = self.Io['value']<self.Io['lb']
        self.Io['value'][lindex] = self.Io['lb'][lindex]
        uindex = self.Io['value']>self.Io['ub']
        self.Io['value'][uindex] = self.Io['ub'][uindex]

    def Flimit(self,constraint,vector,grad):
        if self.svd:  # convert eigenvalues to current vector
            I = np.dot(self.V,vector)
        else:
            I = vector
        F,dF = self.ff.set_force(I)  # set coil force and jacobian
        if grad.size > 0:  # calculate constraint jacobian
            grad[:self.nPF] = -dF[:self.nPF,:,1]  # PFz lower bound
            grad[self.nPF:2*self.nPF] = dF[:self.nPF,:,1]  # PFz upper bound
            grad[2*self.nPF] = -np.sum(dF[self.nPF:,:,1],axis=0)  # CSsum lower
            grad[2*self.nPF+1] = np.sum(dF[self.nPF:,:,1],axis=0)  # CS_ upper
            for j in range(self.nCS-1):  # evaluate each gap in CS stack
                grad[2*self.nPF+2+j] = np.sum(dF[self.nPF+j+1:,:,1],axis=0)-\
                np.sum(dF[self.nPF:self.nPF+j+1,:,1],axis=0)
        PFz = F[:self.nPF,1]  # vertical force on PF coils
        PFz_limit = self.limit['F']['PFz']  # limit force
        constraint[:self.nPF] = -PFz_limit-PFz  # PFz lower bound
        constraint[self.nPF:2*self.nPF] = PFz-PFz_limit  # PFz upper bound
        FzCS = F[self.nPF:,1]  # vertical force on CS coils
        CSz_sum = np.sum(FzCS)  # vertical force on CS stack 
        CSsum_limit = self.limit['F']['CSz_sum']
        constraint[2*self.nPF] = -CSsum_limit-CSz_sum  # CSsum lower bound
        constraint[2*self.nPF+1] = CSz_sum-CSsum_limit  # CSsum upper bound
        CSsep_limit = self.limit['F']['CSz_sep']
        for j in range(self.nCS-1):  # evaluate each gap
            Fsep = np.sum(FzCS[j+1:])-np.sum(FzCS[:j+1])
            constraint[2*self.nPF+2+j] = Fsep-CSsep_limit  # CS seperation
        
    def solve_slsqp(self,flux):  # solve for constrained current vector 
        self.fix_flux(flux)  # swing
        self.set_Io()  # set coil current and bounds
        opt = nlopt.opt(nlopt.LD_SLSQP,self.nC)
        opt.set_min_objective(self.frms)
        opt.set_ftol_abs(1e-12)  # 1e-12
        tol = 1e-3*np.ones(2*self.nPF+self.nCS+1)
        opt.add_inequality_mconstraint(self.Flimit,tol)
        if self.svd:  # coil current eigen-decomposition
            opt.add_inequality_mconstraint(self.Ilimit,1e-3*np.ones(2*self.nC))
            self.alpha = opt.optimize(self.alpha)
            self.I = np.dot(self.V,self.alpha)
        else:
            opt.set_lower_bounds(self.Io['lb'])
            opt.set_upper_bounds(self.Io['ub'])
            self.I = opt.optimize(self.Io['value'])
        self.Io['value'] = self.I.reshape(-1)
        self.rms = opt.last_optimum_value()
        self.update_current()
        return self.rms
   
    def update_area(self,relax=1,margin=0):
        I = self.swing['I'][np.argmax(abs(self.swing['I']),axis=0),
                            range(self.nC)]    
        for name in self.PF_coils:
            if name in self.adjust_coils:
                Ic = I[list(self.adjust_coils).index(name)]
            else:
                Ic = self.pf.coil[name]['I']
            if Ic != 0:
                Ic = abs(Ic)                
                if Ic > self.limit['I']['PF']: 
                    Ic = self.limit['I']['PF']  # coil area upper bound
                dA_target = Ic/self.Jmax  # apply current density limit
                dA = self.pf.coil[name]['dr']*self.pf.coil[name]['dz']
                ratio = dA_target/dA
                scale = (ratio*(1+margin))**0.5
                self.pf.coil[name]['dr'] *= relax*(scale-1)+1
                self.pf.coil[name]['dz'] *= relax*(scale-1)+1
        dl = self.fit_PF()  # fit re-sized coils to TF
        return dl
            
    def update_position_vector(self,Lnorm,grad):
        rms = self.update_position(Lnorm,update_area=True,store_update=True)
        if grad.size > 0:
            grad[:] = op.approx_fprime(Lnorm,self.update_position,1e-7)
        rms_str = '\r{:d}) rms {:1.2f}mm '.format(self.iter['position'],1e3*rms)
        rms_str += 'time {:1.0f}s'.format(time.time()-self.tstart)
        rms_str += '\t\t\t'  # white space
        sys.stdout.write(rms_str)
        sys.stdout.flush()
        return rms
        
    def update_position_scipy(self,Lnorm):
        jac = op.approx_fprime(Lnorm,self.update_position,5e-4)
        rms = self.update_position(Lnorm,store_update=True)
        return rms,jac
                        
    def update_position(self,Lnorm,update_area=False,store_update=False):
        self.iter['current'] == 0
        self.Lo['norm'] = np.copy(Lnorm)
        L = loops.denormalize_variables(Lnorm,self.Lo)
        Lpf = L[:self.nPF]
        for name,lpf in zip(self.PF_coils,Lpf):
            r,z = self.tf.fun['out']['r'](lpf),self.tf.fun['out']['z'](lpf)
            ref = int(name.split('Coil')[-1])
            self.move_PF(ref,point=(r,z))
        if len(L) > self.nPF:
            Lcs = L[self.nPF:]
            Z,dZ = self.arange_CS(Lcs)
            for name,z,dz in zip(self.CS_coils,Z,dZ):
                self.move_CS(name,z,dz)
        if update_area:  # converge PF coil areas
            dl_o = 0
            imax,err,relax = 15,1e-3,1
            dL = []
            for i in range(imax):
                dl = self.update_area(relax=relax)
                dL.append(dl)
                self.ff.set_force_feild(state='active')  
                self.set_foreground()
                rms = self.swing_flux()
                if abs(dl-dl_o) < err:  # coil areas converged
                    break
                else:
                    dl_o = dl
                if i > 1:
                    relax *= 0.5
                if i==imax-1:
                    print('warning: coil areas not converged')
                    print(dL)
        else:
            self.fit_PF()  # fit coils to TF
            self.ff.set_force_feild(state='active')  
            self.set_foreground()
            rms = self.swing_flux()
        if store_update:
            self.iter['position'] += 1
            self.store_update()
        return rms

    def swing_flux(self,bndry=False):
        for i,flux in enumerate(self.swing['flux']):
            self.swing['rms'][i] = self.solve_slsqp(flux)
            Fcoil = self.ff.get_force()
            self.swing['FrPF'][i] = Fcoil['PF']['r']
            self.swing['FzPF'][i] = Fcoil['PF']['z']
            self.swing['FsepCS'][i] = Fcoil['CS']['sep']
            self.swing['FzCS'][i] = Fcoil['CS']['zsum']
            self.swing['Fcoil'][i] = {'F':Fcoil['F'],'dF':Fcoil['dF']}
            self.swing['Tpol'][i] = self.poloidal_angle()
            self.swing['I'][i] = self.Icoil
            self.swing['Isum'][i] = self.Isum
            self.swing['IsumCS'][i] = self.IsumCS
            self.swing['IsumPF'][i] = self.IsumPF
            self.swing['Imax'][i] = self.Imax
        self.rmsID = np.argmax(self.swing['rms'])
        self.rms = self.swing['rms'][self.rmsID]
        return self.rms
   
    def set_Lo(self,L):  # set lower/upper bounds on coil positions
        self.nL = len(L)  # length of coil position vector
        self.Lo = {'name':[[] for _ in range(self.nL)],
                   'value':np.zeros(self.nL),
                   'lb':np.zeros(self.nL),'ub':np.zeros(self.nL)}
        for i,l in enumerate(L[:self.nPF]):  # PF coils
            name = 'Lpf{:1.0f}'.format(i)
            coil = 'Coil{:1.0f}'.format(i)
            lb,ub = self.limit['L'][coil]
            loops.add_value(self.Lo,i,name,l,lb,ub)
        for i,l in enumerate(L[self.nPF:]):  # CS coils
            name = 'Zcs{:1.0f}'.format(i)
            loops.add_value(self.Lo,i+self.nPF,name,l,
                            self.limit['L']['CS'][0],self.limit['L']['CS'][1])

    def Llimit(self,constraint,L,grad):  
        PFdL = 1e-4  # minimum PF inter-coil spacing
        CSdL = 1e-4  # minimum CS coil height
        if grad.size > 0:
            grad[:] = np.zeros((self.nPF-1+self.nCS,len(L)))  # initalise
            for i in range(self.nPF-1):  # PF
                grad[i,i] = 1
                grad[i,i+1] = -1
            for i in range(self.nCS):  # CS
                grad[i+self.nPF-1,i+self.nPF] = 1
                grad[i+self.nPF-1,i+self.nPF+1] = -1
        constraint[:self.nPF-1] = L[:self.nPF-1]-L[1:self.nPF]+PFdL  # PF 
        constraint[self.nPF-1:] = L[self.nPF:-1]-L[self.nPF+1:]+CSdL  # CS 

    def minimize(self,L,method='ls'):
        self.iter['position'] == 0 
        self.set_Lo(L)  # set position bounds
        Lnorm = loops.normalize_variables(self.Lo)
        
        if method == 'bh':  # basinhopping
            #minimizer = {'method':'SLSQP','jac':True,#'args':True,'options':{'eps':1e-3}, #'jac':True,
            #             'bounds':[[0,1] for _ in range(self.nL)]}
            minimizer = {'method':'L-BFGS-B','jac':True,
                         'bounds':[[0,1] for _ in range(self.nL)]}             
                         
            res = op.basinhopping(self.update_position_scipy,Lnorm,niter=1e4,
                                    T=1e-3,stepsize=0.05,disp=True,
                                    minimizer_kwargs=minimizer,
                                    interval=50,niter_success=100)
            Lnorm,self.rms = res.x,res.fun
        elif method == 'de':  # differential_evolution
            bounds = [[0,1] for _ in range(self.nL)]
            res = op.differential_evolution(self.update_position,bounds,\
            args=(False,),strategy='best1bin',maxiter=100,popsize=15,tol=0.01,\
            mutation=(0.5,1),recombination=0.7,polish=True,disp=True)
            Lnorm,self.rms = res.x,res.fun
        elif method == 'ls':  # sequential least squares
            print('Optimising configuration:')
            #opt = nlopt.opt(nlopt.LD_SLSQP,self.nL)
            opt = nlopt.opt(nlopt.LD_MMA,self.nL)
            opt.set_ftol_abs(5e-4)
            #opt.set_ftol_rel(1e-2)
            #opt.set_stopval(30e-3)  # <x [m]
            opt.set_min_objective(self.update_position_vector)
            opt.set_lower_bounds([0 for _ in range(self.nL)])
            opt.set_upper_bounds([1 for _ in range(self.nL)])
            tol = 1e-3*np.ones(self.nPF-1+self.nCS)
            opt.add_inequality_mconstraint(self.Llimit,tol)
            self.rms = opt.last_optimum_value()
            Lnorm = opt.optimize(Lnorm)
        loops.denormalize_variables(Lnorm,self.Lo)
        
        '''
        #if bndry: rms_bndry[i],z_offset[i] = self.get_rms_bndry()
        self.rms_bndry = np.max(rms_bndry)
        self.z_offset = np.mean(z_offset)
        rms = self.position_coils(Lo,True)
        text = 'Isum {:1.0f} Ics {:1.0f}'.format(self.Isum*1e-6,self.Ics*1e-6)
        text += ' rms {:1.4f} rmsID {:1.0f}'.format(self.rms,self.rmsID)
        text += ' rms_bndry {:1.3f}'.format(self.rms_bndry)
        text += ' z_offset {:1.3f}'.format(self.z_offset)
        text += ' dTpol {:1.3f}'.format(self.dTpol)
        print(text)
        '''
        print('\nrms {:1.2f}mm'.format(1e3*self.rms))
        return self.Lo['value'],self.rms
        
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
            
    def optimize(self,Lo):
        self.initialize_log()
        self.ztarget = self.eq.sf.Mpoint[1]
        self.ttotal,self.ttotal_cpu = 0,0
        self.tstart,self.tstart_cpu = time.time(),time.process_time()
        self.iter['plasma'] += 1
        self.tick()
        self.add_plasma()
        Lo,fun = self.minimize(Lo)
        self.store_update(extent='position')
        self.tock()
        return Lo

    def add_plasma(self):
        self.set_background()
        self.get_weight()
        
    def snap_coils(self,offset=0.3):
        L = self.grid_coils(offset=0.3)
        self.set_Lo(L)  # set position bounds
        Lnorm = loops.normalize_variables(self.Lo)
        return Lnorm

    def update_current(self):
        self.Isum,self.IsumCS,self.IsumPF = 0,0,0
        self.ff.I = self.I  # pass current to force feild
        self.Icoil = np.zeros(len(self.coil['active'].keys()))
        for j,name in enumerate(self.coil['active']):
            #Nfilament = self.pf.sub_coil[name+'_0']['Nf']
            Nfilament = self.coil['active'][name]['Nf']
            self.Icoil[j] = self.I[j]*Nfilament  # store current
            self.pf.coil[name]['I'] = self.Icoil[j]*self.Iscale  
            self.coil['active'][name]['I_sum'] = self.Icoil[j]*self.Iscale
            
            for i in range(Nfilament):
                sub_name = name+'_{:1.0f}'.format(i)
                self.pf.sub_coil[sub_name]['I'] = self.I[j]*self.Iscale  
                self.coil['active'][name]['I'][i] = self.I[j]*self.Iscale  
            self.Isum += abs(self.Icoil[j])  # sum absolute current
            if name in self.CS_coils:
                self.IsumCS += abs(self.Icoil[j])
            elif name in self.PF_coils:
                self.IsumPF += abs(self.Icoil[j])
        imax = np.argmax(abs(self.Icoil))
        self.Imax = self.Icoil[imax]
     
    def write_swing(self):
        nC,nS = len(self.adjust_coils),len(self.Swing)
        Icoil = np.zeros((nS,nC))
        for i,flux in enumerate(self.Swing[::-1]):
            self.solve_slsqp(flux)
            Icoil[i] = self.Icoil
        with open('../Data/'+
                  self.tf.dataname.replace('TF.pkl','_currents.txt'),'w') as f:
            f.write('Coil\tr [m]\tz [m]\tdr [m]\tdz [m]')
            f.write('\tI SOF [A]\tI EOF [A]\n')
            for j,name in enumerate(self.coil['active'].keys()):
                r = float(self.pf.coil[name]['r'])
                z = float(self.pf.coil[name]['z'])
                dr = self.pf.coil[name]['dr']
                dz = self.pf.coil[name]['dz']
                position = '\t{:1.3f}\t{:1.3f}'.format(r,z)
                size = '\t{:1.3f}\t{:1.3f}'.format(dr,dz)
                current = '\t{:1.3f}\t{:1.3f}\n'.format(Icoil[0,j],Icoil[1,j])
                f.write(name+position+size+current)
                
    def fix_boundary(self,plot=False):
        self.fix_boundary_psi(N=25,alpha=1-1e-4,factor=1)  # add boundary points
        self.fix_boundary_feild(N=25,alpha=1-1e-4,factor=1)  # add boundary points
        self.add_null(factor=1,point=self.eq.sf.Xpoint)
        self.initialize_log()
        if plot:
            self.plot_fix(tails=True)
                
class scenario(object):

    def __init__(self,inv,sf,rms_limit=0.05,wref=25,plot=True):
        self.inv = inv
        self.rms_limit = rms_limit
        self.wref = wref
        self.inv.load_equlibrium(sf)
        self.inv.fix_boundary(plot=plot)
        self.inv.add_plasma()
        self.Lnorm = inv.snap_coils()
        self.inv.set_force_feild()
        
    def get_rms(self,centre):
        self.inv.set_swing(centre=centre,width=self.wref,
                           array=np.linspace(-0.5,0.5,3))
        rms = self.inv.update_position(self.Lnorm,update_area=True)
        return float(self.rms_limit-rms)
        
    def find_root(self,slim):
        swing = brentq(self.get_rms,slim[0],slim[1],xtol=0.1,maxiter=500,
                       disp=True)
        return swing
        
    def flat_top(self):
        SOF = self.find_root([-60,0])-self.wref/2
        EOF = self.find_root([0,60])+self.wref/2                      
        
        self.width = 0.95*(EOF-SOF)
        print('swing width {:1.0f}Vs'.format(2*np.pi*self.width))
        self.inv.set_swing(centre=np.mean([SOF,EOF]),width=self.width,
                           array=np.linspace(-0.5,0.5,3))
        self.inv.update_position(self.Lnorm,update_area=True)
        
    def energy(self):
        self.inv.pf.inductance(1)
        E = np.zeros(len(self.inv.swing['flux']))
        for i,I in enumerate(self.inv.swing['I']):
            I *= self.inv.Iscale
            E[i] = 0.5*np.dot(np.dot(I,self.inv.pf.M),I)
        print(E*1e-9)
        
    def output(self):
        PFvol = 0
        for name in self.inv.pf.coil:
            coil = self.inv.pf.coil
            r,dr,dz = coil['r'],coil['dr'],coil['dz']
            PFvol += 2*np.pi*r*dr*dz
            
        
        
        