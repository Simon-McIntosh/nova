import numpy as np
import pylab as pl
import nova.cross_coil as cc
from collections import OrderedDict
from scipy.interpolate import RectBivariateSpline as RBS
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d
import seaborn as sns
import scipy.optimize as op
from itertools import cycle
import copy
import time
import multiprocessing
from amigo import geom
from nova import loops
import nlopt

slsqp = copy.deepcopy(op.fmin_slsqp)

class INV(object):
    
    def __init__(self,sf,eq,tf,Jmax=12.5e6,TFoffset=0.3):
        self.ncpu = multiprocessing.cpu_count()
        self.sf = sf
        self.eq = eq
        self.tf = tf
        self.tf.loop_interpolators()
        self.Jmax = Jmax
        self.fix = {'r':np.array([]),'z':np.array([]),'BC':np.array([]),
                    'value':np.array([]),'Bdir':np.array([[],[]]).T,
                    'factor':np.array([]),'n':0}  
        self.cnstr = {'r':np.array([]),'z':np.array([]),'BC':np.array([]),
                     'value':np.array([]),'condition':np.array([])}  
        self.add_active([],empty=True)  # all coils active
        self.update_coils()
        self.initalise_current()
        self.log_scalar = ['Fr_max','Fz_max','F_max','FzCS','Fsep','rms',
                           'rms_bndry','rmsID','Isum','Imax','z_offset',
                           'FrPF','FzPF','dTpol']
        self.log_array = ['Lo','Iswing']
        self.log_iter = ['current_iter','plasma_iter','position_iter']
        self.log_plasma = ['plasma_coil']
        self.CS_Lnorm,self.CS_Znorm = 2,1
        self.TFoffset = TFoffset  # offset coils from outer TF loop
        self.force_feild_active = False
     
    def set_swing(self,centre=25,width=363):
        flux = centre+np.array([-0.5,0.5])*width/(2*np.pi)  # Webber/rad
        
        nC,nS = self.nC,len(flux)
        self.swing = {'flux':flux,'I':np.zeros((nS,nC)),'rms':np.zeros(nS),
                      'FrPF':np.zeros(nS),'FzPF':np.zeros(nS),
                      'Tpol':np.zeros(nS),'rms_bndry':np.zeros(nS),
                      'z_offset':np.zeros(nS),'dTpol':0,
                      'Fcoil':[[] for _ in range(nS)]}
        
    def Cname(self,coil):
        return 'Coil{:1.0f}'.format(coil)
        
    def initalise_coil(self):
        self.nC = len(self.adjust_coils)  # number of active coils
        self.nPF = len(self.PF_coils)
        self.nCS = len(self.CS_coils)
        self.coil = {'active':OrderedDict(),'passive':OrderedDict()}
        names = self.eq.pf.coil.keys()
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
        
    def update_coils(self,plot=False):  # full coil update
        self.initalise_coil()
        self.append_coil(self.eq.coil)
        if not hasattr(self.eq,'plasma_coil'):
            self.eq.get_plasma_coil()
        self.append_coil(self.eq.plasma_coil)
        for coil in self.all_coils:
            self.update_bundle(coil)  # re-grid elliptic coils
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
            self.adjust_coils = list(self.eq.pf.coil.keys())  # add all
            if Ctype == 'PF': 
                self.PF_coils = list(self.eq.pf.coil.keys())
            elif Ctype == 'CS': 
                self.CS_coils = list(self.eq.pf.coil.keys())    

    def remove_active(self,Clist,Ctype='all',full=False):  # list of coil names
        if full: 
            self.adjust_coils = list(self.eq.pf.coil.keys())  # add all
            if Ctype == 'PF':
                self.PF_coils = list(self.eq.pf.coil.keys())
            elif Ctype == 'CS':
                self.CS_coils = list(self.eq.pf.coil.keys())  
        if Clist:
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
        #self.update_coils()
        
    def delete_active(self):
        Clist = []
        for name in self.coil['active'].keys():
            Clist.append(int(name.split('Coil')[-1]))
        self.remove_active(Clist,Ctype='all')
        self.remove_coil(Clist)

    def remove_coil(self,Clist):
        for coil in Clist:
            name = self.Cname(coil)
            if name in self.eq.pf.coil.keys():
                del self.eq.pf.coil[name]
                for i in range(self.eq.coil[name+'_0']['Nf']):
                    del self.eq.coil[name+'_{:1.0f}'.format(i)]
   
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
        
    def add_coil(self,Ctype=None,**kwargs):
        r,z = self.get_point(**kwargs)
        index,i = np.zeros(len(self.eq.pf.coil.keys())),-1
        for i,name in enumerate(self.eq.pf.coil.keys()):
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
        self.eq.pf.coil[name] = {'r':r,'z':z,'dr':delta[0],'dz':delta[1],'I':1e6,
                              'rc':np.sqrt(delta[0]**2+delta[1]**2)/2}
     
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
        r,z = self.sf.get_boundary(alpha=alpha)
        self.eq_spline()
        nr = -self.psi.ev(r,z,dx=1,dy=0)
        nz = -self.psi.ev(r,z,dx=0,dy=1)
        L = self.sf.length(r,z)
        Lc = np.linspace(0,1,N+1)[:-1]
        rb,zb = interp1d(L,r)(Lc),interp1d(L,z)(Lc)
        Bdir = np.array([interp1d(L,nr)(Lc),interp1d(L,nz)(Lc)]).T
        return rb,zb,Bdir
        
    def get_psi(self,alpha):
        Xpsi = self.sf.Xpsi  
        Mpsi = self.sf.Mpsi  
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
        
    def plot_fix(self,tails=True):
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
            if bdir[0]**2+bdir[1]**2 > 0:  # tails    
                direction = bdir
            else:
                d_dr,d_dz = self.get_gradients(bc,r,z)
                direction = np.array([d_dr,d_dz])/np.sqrt(d_dr**2+d_dz**2)
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
                        elif state == 'passive':
                            self.BG[i] += value
                        else:
                            errtxt = 'specify coil state'
                            errtxt += '\'active\', \'passive\'\n'
                            raise ValueError(errtxt)
                            
    def add_value_coil(self,bc,rf,zf,r,z,I,bdir):
        if 'psi' in bc:
            value = cc.mu_o*I*cc.green(rf,zf,r,z)
        else:
            B = cc.mu_o*I*cc.green_feild(rf,zf,r,z)
            value = self.Bfeild(bc,B[0],B[1],bdir)
        return value
        
    def add_value_plasma(self,bc,rf,zf,bdir):
        if 'psi' in bc:
            value = self.psi_plasma.ev(rf,zf)  # interpolate from GS
        else:
            Br,Bz = self.B_plasma[0].ev(rf,zf),self.B_plasma[1].ev(rf,zf)
            value = self.Bfeild(bc,Br,Bz,bdir)
        return value
        
    def check_force_feild(self):
        if not self.force_feild_active:
            errtxt = 'no force feild\n'
            errtxt += 'set_force_feild\n'
            raise ValueError(errtxt)
        if self.nPF == 0:
            errtxt = 'PF_coils empty\n'
            raise ValueError(errtxt)
        if self.nCS == 0:
            errtxt = 'CS_coils empty\n'
            raise ValueError(errtxt)
        
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
        self.force_feild_active = True
        active_coils = self.coil['active'].keys()
        passive_coils = self.coil['passive'].keys()
        if state == 'both' or state == 'active':
            self.set_active_force_feild(active_coils,
                                        multi_filament=multi_filament)
        if state == 'both' or state == 'passive':
            self.set_passive_force_feild(active_coils,passive_coils)

    def set_active_force_feild(self,active_coils,multi_filament=False):  
        self.Fa = np.zeros((self.nC,self.nC,2))  # active
        for i,sink in enumerate(active_coils):
            for j,source in enumerate(active_coils):
                rG = cc.Gtorque(self.eq.coil,self.eq.pf.coil,
                                source,sink,multi_filament)
                self.Fa[i,j,0] = 2*np.pi*cc.mu_o*rG[1]  #  cross product
                self.Fa[i,j,1] = -2*np.pi*cc.mu_o*rG[0]
        
    def set_passive_force_feild(self,active_coils,passive_coils):
        self.Fp = np.zeros((self.nC,2))  # passive
        for i,sink in enumerate(active_coils):
            rB = cc.Btorque(self.eq.coil,self.eq.plasma_coil,passive_coils,sink)
            self.Fp[i,0] = 2*np.pi*cc.mu_o*rB[1]  #  cross product
            self.Fp[i,1] = -2*np.pi*cc.mu_o*rB[0]  

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
        if 'psi' in bc:
            d_dr = self.psi.ev(rf,zf,dx=1,dy=0)
            d_dz = self.psi.ev(rf,zf,dx=0,dy=1)
        else:
            d_dr = self.B.ev(rf,zf,dx=1,dy=0)
            d_dz = self.B.ev(rf,zf,dx=0,dy=1)
        return d_dr,d_dz
        
    def get_weight(self):
        Rf,Zf,BC,Bdir = self.unpack_fix()
        weight = np.ones(len(BC))
        for i,(rf,zf,bc,bdir,factor) in enumerate(zip(Rf,Zf,BC,Bdir,
                                                      self.fix['factor'])):
            d_dr,d_dz = self.get_gradients(bc,rf,zf)
            if bdir[0]**2+bdir[1]**2 == 0 or 'psi' not in bc:
                weight[i] = 1/abs(np.sqrt(d_dr**2+d_dz**2))
            else:
                weight[i] = 1/abs(np.dot([d_dr,d_dz],bdir))
        factor = np.reshape(self.fix['factor'],(-1,1))
        weight = np.reshape(weight,(-1,1))
        self.wsqrt = np.sqrt(factor*weight)
        
    def arange_CS(self,Z,dZmin=0.2):
        Z = np.sort(Z)
        L = Z[-1]-Z[0]-self.gap*self.nCS
        dZ = np.diff(Z)-self.gap
        i_min = dZ<dZmin
        nmin = np.sum(i_min)
        if nmin>0:
            Lsmall = np.sum(dZ[i_min])
            dZ[i_min] = dZmin
            scale = (L-nmin*dZmin)/(L-Lsmall)
            dZ[np.invert(i_min)] *= scale
   
        Zc = np.zeros(self.nCS)
        Zc[0] = Z[0]+self.gap/2+dZ[0]/2
        for i in range(self.nCS-1):
            Zc[i+1] = Zc[i]+dZ[i]/2+dZ[i+1]/2+self.gap
        return Zc,dZ
        
    def grid_CS(self,nCS=3,Ro=2.9,Zbound=[-10,9],dr=0.818,gap=0.1):  #dr=1.0, dr=1.25,Ro=3.2,dr=0.818
        self.gap = gap
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
            self.add_coil(Lout=L,Ctype='PF')    
        return Lo
  
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
            coil = self.eq.pf.coil[self.Cname(ref)]
            rc,zc = coil['r'],coil['z']
            r,z = rc+dr,zc+dz
        elif 'L' in kwargs.keys():
            ref,dL = kwargs['dL']
            coil = self.eq.pf.coil[self.Cname(ref)]
            rc,zc = coil['r'],coil['z'] 
        ARmax = 3
        if abs(AR) > ARmax: AR = ARmax*np.sign(AR)
        if AR > 0: 
            AR = 1+AR
        else:
            AR = 1/(1-AR)
        dA = self.eq.pf.coil[name]['dr']*self.eq.pf.coil[name]['dz']
        self.eq.pf.coil[name]['dr'] = np.sqrt(AR*dA)
        self.eq.pf.coil[name]['dz'] = self.eq.pf.coil[name]['dr']/AR
        dr = r-self.eq.pf.coil[name]['r']
        dz = z-self.eq.pf.coil[name]['z']
        self.shift_coil(name,dr,dz)
    
    def fit_PF(self,**kwargs):  # offset PF coils from TF
        offset = kwargs.get('offset',self.TFoffset)
        for name in self.PF_coils:
            dr,dz = self.tf.Cshift(self.eq.pf.coil[name],'out',offset)
            self.shift_coil(name,dr,dz)
        
    def shift_coil(self,name,dr,dz):
        self.eq.pf.coil[name]['r'] += dr
        self.eq.pf.coil[name]['z'] += dz
        self.coil['active'][name]['r'] += dr
        self.coil['active'][name]['z'] += dz
        for i in range(self.eq.coil[name+'_0']['Nf']):
            sub_name = name+'_{:1.0f}'.format(i)
            self.eq.coil[sub_name]['r'] += dr
            self.eq.coil[sub_name]['z'] += dz
        self.update_bundle(name) 
        
    def move_CS(self,name,z,dz):
        self.eq.pf.coil[name]['z'] = z
        self.eq.pf.coil[name]['dz'] = dz
        self.update_bundle(name)
   
    def tails(self,x,dx=0.1):  # contiuous PF position limit
        if x > (1-dx):
            y = (1-dx)+dx*(1-np.exp(-(x-(1-dx))))
        elif x < dx:
            y = dx - dx*(1-np.exp(-(dx-x)))
        else:
            y = x
        return y

    def reset_current(self):
        for j,name in enumerate(self.coil['active'].keys()):
            self.eq.pf.coil[name]['I'] = 0
        
    def update_bundle(self,name):
        try:
            Nold = self.eq.coil[name+'_0']['Nf']
        except:
            Nold = 0
        bundle = self.eq.size_coil(self.eq.pf.coil,name,self.eq.dCoil)
        N = self.eq.coil[name+'_0']['Nf']
        self.coil['active'][name] = {}
        for key in bundle.keys():
            self.coil['active'][name][key] = bundle[key]
        if Nold > N:
            for i in range(N,Nold):
                del self.eq.coil[name+'_{:1.0f}'.format(i)]

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
        for i,L in enumerate(Lpf):
            Lpf[i] = self.tails(L)
        dL = np.zeros(self.nPF)
        dLmin = 2e-2
        for i,l in enumerate(Lpf):
            L = np.append(Lpf[:i],Lpf[i+1:])
            dL[i] = np.min(abs(l-L))
        fun = np.min(dL)-dLmin
        return fun
        
    def CSlength(self,Lo,*args):
        L = Lo[self.nPF+1]
        #fun = 30-self.CS_Lnorm*L
        fun = 60-self.CS_Lnorm*L
        return fun
        
    def CSzmax(self,Lo,*args):
        zmax = 20#9
        Z = self.CS_Znorm*Lo[self.nPF]
        L = self.CS_Lnorm*Lo[self.nPF+1]
        Zmax = Z+L/2
        fun = zmax-Zmax
        return fun
        
    def CSzmin(self,Lo,*args):
        zmin = -20#-14
        Z = self.CS_Znorm*Lo[self.nPF]
        L = self.CS_Lnorm*Lo[self.nPF+1]
        Zmin = Z-L/2
        fun = Zmin-zmin
        return fun
            
    def get_rms(self):
        err = np.dot(self.wG,self.I)-self.wT
        self.rms = np.sqrt(np.mean(err**2))
        return err

    def r_rms(self,Inorm):
        I = loops.get_oppvar(self.Io,self.adjust_coils,Inorm)
        self.I = I.reshape(-1,1)  # set current vector
        self.get_rms()  # solve
        return self.rms
        
    def solve(self):
        self.I = np.linalg.lstsq(self.wG,self.wT)[0]  # least squares solve
        self.get_rms()
        self.update_current()
             
    def set_Io(self):  # set lower/upper bounds on coil currents (Jmax)
        self.Io = OrderedDict()
        for name in self.all_coils:  # all coils
            coil = self.eq.pf.coil[name]
            Imax = coil['dr']*coil['dz']*self.Jmax
            Imin = -Imax
            Nf = self.eq.coil[name+'_0']['Nf']  # fillament number
            self.Io[name] = {'value':coil['I']/Nf,'lb':Imin/Nf,'ub':Imax/Nf}

    def frms(self,Inorm,grad):
        I = loops.get_oppvar(self.Io,self.adjust_coils,Inorm)
        d = np.append(self.wG,-self.wT,axis=1)  # delta
        ms = np.zeros((self.nC+1,self.nC+1))  # mean square
        Is = np.dot(np.append(I,1).reshape(-1,1),
                    np.append(I,1).reshape(1,-1))  # I square
        for i in range(self.fix['n']):
            ms += np.dot(d[i,:].reshape((-1,1)),d[i,:].reshape((1,-1)))
        ms /= self.fix['n']  # mean square
        rms = np.sum(ms*Is)**0.5  # check rms
        if grad.size>0:
            drms = 0.5*np.sum(ms*Is)**-0.5
            II = np.diagonal(ms)
            np.fill_diagonal(ms,0)
            jac = drms*np.ones(self.nC)
            for i in range(self.nC):
                jac[i] *= (2*II[i]*I[i]+2*(np.dot(ms[i,:-1].reshape((1,-1)),
                                                  I.reshape((-1,1)))+ms[i,-1]))
            grad[:] = jac
        return rms
        
    ''''    
    def frms(self,I):

        #I = loops.get_oppvar(self.Io,self.adjust_coils,Inorm)
        
        d = np.append(self.wG,-self.wT,axis=1)  # delta
        ms = np.zeros((self.nC+1,self.nC+1))  # mean square
        Is = np.dot(np.append(I,1).reshape(-1,1),
                    np.append(I,1).reshape(1,-1))  # I square
        for i in range(self.fix['n']):
            ms += np.dot(d[i,:].reshape((-1,1)),d[i,:].reshape((1,-1)))
        ms /= self.fix['n']  # mean square
        rms = np.sum(ms*Is)**0.5  # check rms
        return rms
    '''

        
    def solve_slsqp(self):
        self.solve()  # sead coil currents
        
        

        opt = nlopt.opt(nlopt.LD_SLSQP,1)
        
        opt.set_min_objective(self.frms)
        
        self.set_Io()  # set coil current and bounds
        Inorm,bounds = loops.set_oppvar(self.Io,self.adjust_coils)
        
        opt.set_lower_bounds(bounds[:,0])
        opt.set_upper_bounds(bounds[:,1])
        opt.set_ftol_rel(1e-2)
        
        Inorm = opt.optimize([3])
        
        I = loops.get_oppvar(self.Io,self.adjust_coils,Inorm)
        
        print(xopt)
        

        '''
        rms = self.rms


        '''
        #slsqp(INV.fx2,[1,2])
        #op.fmin_slsqp(INV.fx2,[1,2])
        #self.Ilimit(Inorm,Imax,Ics,FzPF,FzCS,Fsep)
        #Inorm,self.rms,niter,imode = op.fmin_slsqp(self.return_rms,Inorm,
        #                                           #fprime=self.fprime,
        #                                           full_output=True,disp=0,
        #                                           )[:4]

        return self.rms
   
    def update_area(self,Imax=50e6,margin=0.1):
        I = self.swing['I'][np.argmax(abs(self.swing['I']),axis=0),
                            range(self.nC)]    
        for name in self.PF_coils:
            if name in self.adjust_coils:
                Ic = I[list(self.adjust_coils).index(name)]
            else:
                Ic = self.eq.pf.coil[name]['I']
            if Ic != 0:
                Ic = abs(Ic)                
                if Ic > Imax: 
                    Ic = Imax  # coil area upper bound
                dA_target = Ic/self.Jmax  # apply current density limit
                dA = self.eq.pf.coil[name]['dr']*self.eq.pf.coil[name]['dz']
                ratio = dA_target/dA
                scale = (ratio*(1+margin))**0.5
                self.eq.pf.coil[name]['dr'] *= scale
                self.eq.pf.coil[name]['dz'] *= scale
        self.fit_PF()  # fit re-sized coils to TF
            
    def update_position(self,Lnorm):
        self.iter['current'] == 0
        self.iter['position'] += 1
        L = loops.get_oppvar(self.Lo,self.position_coils,Lnorm)
        Lpf = L[:self.nPF]
        for name,lpf in zip(self.PF_coils,Lpf):
            lpf = self.tails(lpf)
            r,z = self.tf.fun['out']['r'](lpf),self.tf.fun['out']['z'](lpf)
            ref = int(name.split('Coil')[-1])
            self.move_PF(ref,point=(r,z))
        if len(L) > self.nPF:
            Lcs = L[self.nPF:]
            Z,dZ = self.arange_CS(Lcs)
            for name,z,dz in zip(self.CS_coils,Z,dZ):
                self.move_CS(name,z,dz)
        self.update_area()
        self.set_force_feild(state='active')  # update active force feild
        self.set_foreground()

        rms_o = self.swing_flux()
        imax,err = 15,1e-6
        for i in range(imax):
            rms = self.swing_flux()
            if abs(rms-rms_o) < err:
                break
            else:
                rms_o = rms
        if i==imax-1:
            print('warning coil areas not converged')
        return rms

    def swing_flux(self,bndry=False):
        for i,flux in enumerate(self.swing['flux']):
            self.fix_flux(flux)  # swing
            #self.iter['current'] += self.solve_slsqp()
            self.swing['rms'][i] = self.solve_slsqp()
            self.swing['Fcoil'][i] = self.get_force()
            self.swing['Tpol'][i] = self.poloidal_angle()
            self.swing['I'][i] = self.Icoil
            #self.swing['rms'][i] = self.rms
        self.update_area()
        self.set_force_feild(state='active') 
        self.set_foreground()
        self.rmsID = np.argmax(self.swing['rms'])
        self.rms = self.swing['rms'][self.rmsID]
        
        #self.store_update()
        return self.rms
        
    def get_force(self):
        self.check_force_feild()
        Fcoil = {'PF':{'r':0,'z':0},'CS':{'sep':0,'zsum':0},'max':0}
        Fvector = np.zeros((self.nC,2))
        for i in range(2):  # coil force (bundle of eq elements)
            Fvector[:,i] = (self.I*(np.dot(self.Fa[:,:,i],self.I)+
                            self.Fp[:,i].reshape(-1,1)))[:,0]
        Fvector *= 1e-6  # MA
        Fcoil['max'] = np.max(np.sqrt(Fvector[:,0]**2+Fvector[:,1]**2))    
        Fcoil['PF']['r'] = np.max(abs(Fvector[:self.nPF,0]))
        Fcoil['PF']['z'] = np.max(abs(Fvector[:self.nPF,1]))
        FzCS = Fvector[self.nPF::-1]  # top-bottom
        for j in range(self.nCS-1):  # evaluate each gap
            Fcoil['CS']['sep'] = np.max([Fcoil['CS']['sep'],
                                         np.sum(FzCS[:j+1])-np.sum(FzCS[j+1:])])
        Fcoil['CS']['zsum'] = np.sum(FzCS)
        return Fcoil
        
    def set_limits(self,Imax=30e6,Ics=500e6,FzPF=350e6,FzCS=350e6,Fsep=300e6):
        self.limits = {'Imax':Imax,'Ics':Ics,'FzPF':FzPF,'FzCS':FzCS,
                       'Fsep':Fsep}
        
    def get_limits(self,Inorm,*args):
        Imax,Ics,FzPF = args[0],args[1],args[2]
        FzCS,Fsep = args[3],args[4]
        
        I = loops.get_oppvar(self.Io,self.adjust_coils,Inorm)
        ieqcon = np.zeros(len(I)+self.nPF)
        self.I = I.reshape(-1,1)
        self.update_current()
        self.get_rms()  
        self.get_force()
        for i,name, in enumerate(self.coil['active'].keys()):
            if name in self.PF_coils:  # PF current limit
                ieqcon[i] = Imax-abs(self.Icoil[i])
            if name in self.CS_coils:  # CS current density limit
                dA = self.eq.pf.coil[name]['dr']*self.eq.pf.coil[name]['dz']
                ImaxCS = self.Jmax*dA
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
        
    def set_Lo(self,L):  # set lower/upper bounds on coil positions
        self.Lo = OrderedDict()
        self.position_coils = []
        for name,l in zip(self.PF_coils,L[:self.nPF]):  # PF coils
            self.position_coils.append(name)
            self.Lo[name] = {'value':l,'lb':0.1,'ub':0.9}
        for i,l in enumerate(L[self.nPF:]):  # PF coils
            name = 'Zcs{:1.0f}'.format(i)
            self.position_coils.append(name)
            self.Lo[name] = {'value':l,'lb':-10,'ub':8}    
        
    def minimize(self,L,rhobeg=0.1,rhoend=1e-4):  # rhobeg=0.1,rhoend=1e-4
        self.iter['position'] == 0 
        
        self.set_Lo(L)  # set position bounds
        Lnorm,bounds = loops.set_oppvar(self.Lo,self.position_coils)

        '''
        Lo,nitter = op.fmin_slsqp(self.position_coils,Lo,disp=3,
                                  bounds=[[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9],[0.1,0.9]],
                                  full_output=True)[:3:2]  
                           #,ieqcons=[self.PFspace,self.CSzmax,self.CSzmin]
        '''                          
        res = op.minimize(self.update_position,Lnorm,method='SLSQP',
                          bounds=bounds,
                          options={'disp':3})
                          #bounds=[(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.9)])
        Lo = res.x
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
        rms=0
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
            
    def optimize(self,Lo,Nmax=1):
        fun = 0
        rhobeg = [1,1]  # start,min
        rhoend = [5e-5,5e-5]  # start,min
        self.initialize_log()
        self.ztarget = self.sf.Mpoint[1]
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
            #rhobeg = self.rhobound(rhobeg)
            #rhoend = self.rhobound(rhoend)
            err = abs(fun-fun_o) 
            if err < 1e-3:
                break
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
            self.initalise_plasma = False
            self.set_background()
            self.get_weight()
            self.set_force_feild(state='both')
        #else:
            #self.swing_fix(np.mean(self.Swing))
            #self.solve_slsqp()
            #self.eq.get_Vcoil()  # select vertical stability coil
            #self.eq.gen(self.ztarget)
        #self.set_background()
        #self.get_weight()
        #self.set_force_feild(state='both')

    def update_current(self):
        self.Isum,self.Ics = 0,0
        self.Icoil = np.zeros(len(self.coil['active'].keys()))
        for j,name in enumerate(self.coil['active']):
            Nfilament = self.eq.coil[name+'_0']['Nf']
            self.eq.pf.coil[name]['I'] = self.I[j,0]*Nfilament  # update sf 
            self.coil['active'][name]['I_sum'] = self.I[j]*Nfilament  # inv
            self.Icoil[j] = self.eq.pf.coil[name]['I']  # store current
            for i in range(Nfilament):
                sub_name = name+'_{:1.0f}'.format(i)
                self.eq.coil[sub_name]['I'] = self.I[j,0]  # update eq
                self.coil['active'][name]['I'][i] = self.I[j,0]  # update inv
            self.Isum += abs(self.Icoil[j])  # sum absolute current
            if name in self.CS_coils:
                self.Ics += abs(self.Icoil[j])
        self.Imax = self.Icoil.max()
        
    def initalise_current(self):
        adjust_coils = self.coil['active'].keys()
        self.I = np.zeros((len(adjust_coils)))  # ,1
        for i,name in enumerate(adjust_coils):
            self.I[i] = self.eq.coil[name+'_0']['I']
           
    def write_swing(self):
        nC,nS = len(self.adjust_coils),len(self.Swing)
        Icoil = np.zeros((nS,nC))
        for i,swing in enumerate(self.Swing[::-1]):
            self.swing_fix(swing)  
            self.solve_slsqp()
            Icoil[i] = self.Icoil
        with open('../Data/'+
                  self.tf.dataname.replace('TF.pkl','_currents.txt'),'w') as f:
            f.write('Coil\tr [m]\tz [m]\tdr [m]\tdz [m]')
            f.write('\tI SOF [A]\tI EOF [A]\n')
            for j,name in enumerate(self.coil['active'].keys()):
                r = float(self.eq.pf.coil[name]['r'])
                z = float(self.eq.pf.coil[name]['z'])
                dr = self.eq.pf.coil[name]['dr']
                dz = self.eq.pf.coil[name]['dz']
                position = '\t{:1.3f}\t{:1.3f}'.format(r,z)
                size = '\t{:1.3f}\t{:1.3f}'.format(dr,dz)
                current = '\t{:1.3f}\t{:1.3f}\n'.format(Icoil[0,j],Icoil[1,j])
                f.write(name+position+size+current)