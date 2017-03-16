import numpy as np
import pylab as pl
from scipy.interpolate import interp1d
from itertools import count
import seaborn as sns
import matplotlib
import collections
import amigo.geom as geom
from nova.loops import Profile
from nova.config import Setup
from nova.streamfunction import SF
import nova.cross_coil as cc
colors = sns.color_palette('Paired',12)
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from nova.DEMOxlsx import DEMO
from warnings import warn
from nova.inverse import INV
from copy import deepcopy
  
class PF(object):
    def __init__(self,eqdsk):
        self.nC = count(0)
        self.set_coils(eqdsk)
        self.plasma_coil = collections.OrderedDict()
        
    def set_coils(self,eqdsk):
        self.coil = collections.OrderedDict()
        if eqdsk['ncoil'] > 0: 
            CSindex = np.argmin(eqdsk['rc'])  # CS radius and width
            self.rCS,self.drCS = eqdsk['rc'][CSindex],eqdsk['drc'][CSindex]
            for i,(r,z,dr,dz,I) in enumerate(zip(eqdsk['rc'],eqdsk['zc'],
                                                 eqdsk['drc'],eqdsk['dzc'],
                                                 eqdsk['Ic'])):
                self.add_coil(r,z,dr,dz,I,categorize=False)
                if eqdsk['ncoil'] > 100 and i>=eqdsk['ncoil']-101:
                    print('exit set_coil loop - coils')
                    break
        self.categorize_coils()
        
    def categorize_coils(self):
        catogory = np.zeros(len(self.coil),dtype=[('r','float'),('z','float'),
                                                ('index','int'),('name','object')])
        for i,name in enumerate(self.coil):
            catogory[i]['r'] = self.coil[name]['r']
            catogory[i]['z'] = self.coil[name]['z']
            catogory[i]['index'] = i
            catogory[i]['name'] = name
        CSsort = np.sort(catogory,order=['r','z'])  # sort CS, r then z
        CSsort = CSsort[CSsort['r'] < self.rCS+self.drCS]
        PFsort = np.sort(catogory,order='z')  # sort PF,  z
        PFsort = PFsort[PFsort['r'] > self.rCS+self.drCS]
        self.index = {'PF':{'n':len(PFsort['index']),'index':PFsort['index'],
                      'name':PFsort['name']},
                      'CS':{'n':len(CSsort['index']),'index':CSsort['index'],
                      'name':CSsort['name']}}
    
    def add_coil(self,r,z,dr,dz,I,categorize=True):
        name = 'Coil{:1.0f}'.format(next(self.nC))
        self.coil[name] = {'r':r,'z':z,'dr':dr,'dz':dz,'I':I,
                           'rc':np.sqrt(dr**2+dz**2)/2} 
        if categorize:
            self.categorize_coils()
                
    def remove_coil(self,Clist):
        for c in Clist:
            coil = 'Coil{:1.0f}'.format(c)
            self.coil.pop(coil)
        self.categorize_coils()
    
    def unpack_coils(self):
        nc = len(self.coil.keys())
        Ic = np.zeros(nc)
        rc,zc,drc,dzc = np.zeros(nc),np.zeros(nc),np.zeros(nc),np.zeros(nc)
        names = []
        for i,name in enumerate(self.coil.keys()):
            rc[i] = self.coil[name]['r']
            zc[i] = self.coil[name]['z']
            drc[i] = self.coil[name]['dr']
            dzc[i] = self.coil[name]['dz']
            Ic[i] = self.coil[name]['I']
            names.append(name)
        return nc,rc,zc,drc,dzc,Ic,names
    
    def grid_coils(self,dCoil=-1):
        if dCoil < 0:  # dCoil not set, use stored value
            if not hasattr(self,'dCoil'):
                self.dCoil = 0
        else:
            self.dCoil = dCoil
        if self.dCoil==0:
            self.sub_coil = self.coil
            for name in self.coil.keys():
                self.sub_coil[name]['rc'] = np.sqrt(self.coil[name]['dr']**2+
                                                    self.coil[name]['dr']**2)
        else:
            self.sub_coil = {}
            for name in self.coil.keys():
                self.size_coil(name,self.dCoil)
                
    def size_coil(self,name,dCoil):
        rc,zc = self.coil[name]['r'],self.coil[name]['z']
        Dr,Dz = self.coil[name]['dr'],self.coil[name]['dz']
        Dr = abs(Dr)
        Dz = abs(Dz)
        #self.coil_o[name] = {}
        if self.coil[name]['I'] != 0:
            if Dr>0 and Dz>0 and 'plasma' not in name:
                nr,nz = np.ceil(Dr/dCoil),np.ceil(Dz/dCoil)
                dr,dz = Dr/nr,Dz/nz
                r = rc+np.linspace(dr/2,Dr-dr/2,nr)-Dr/2
                z = zc+np.linspace(dz/2,Dz-dz/2,nz)-Dz/2
                R,Z = np.meshgrid(r,z,indexing='ij')
                R,Z = np.reshape(R,(-1,1)),np.reshape(Z,(-1,1))
                Nf = len(R)  # filament number
                #self.coil_o[name]['Nf'] = Nf
                #self.coil_o[name]['Io'] = self.pf.pf_coil[name]['I']
                I = self.coil[name]['I']/Nf
                bundle = {'r':np.zeros(Nf),'z':np.zeros(Nf),
                          'dr':dr*np.ones(Nf),'dz':dz*np.ones(Nf),
                          'I':I*np.ones(Nf),'sub_name':np.array([]),
                          'Nf':0}
                for i,(r,z) in enumerate(zip(R,Z)):
                    sub_name = name+'_{:1.0f}'.format(i)
                    self.sub_coil[sub_name] = {'r':r,'z':z,'dr':dr,'dz':dz,
                                               'I':I,'Nf':Nf,
                                               'rc':np.sqrt(dr**2+dz**2)/2}
                    bundle['r'][i],bundle['z'][i] = r,z
                    bundle['sub_name'] = np.append(bundle['sub_name'],sub_name)
                bundle['Nf'] = i+1
                bundle['Ro'] = np.mean(bundle['r'])
                bundle['Zo'] = np.mean(bundle['z'])
            else:
                print('coil bundle not found',name)
                self.sub_coil[name] = self.coil[name]
                bundle = self.coil[name]
        return bundle
        
    def plot_coil(self,coils,label=False,current=False,coil_color=None,fs=12,
                  alpha=1):
        if coil_color is None:
            color = colors
        else:
            color = coil_color  # color itterator
        if len(np.shape(color)) == 1:
            color = color*np.ones((6,1))
             
        for i,name in enumerate(coils.keys()):
            coil = coils[name]
            r,z,dr,dz = coil['r'],coil['z'],coil['dr'],coil['dz']
            Rfill = [r+dr/2,r+dr/2,r-dr/2,r-dr/2]
            Zfill = [z-dz/2,z+dz/2,z+dz/2,z-dz/2]
            if coil['I'] != 0:
                edgecolor = 'k'
            else:
                edgecolor = 'r' 
            coil_color = color[4] 
            if name.split('_')[0] in self.index['CS']['name']:
                drs = -2.5/3*dr
                ha = 'right'
                coil_color = color[5]
            elif name.split('_')[0] in self.index['PF']['name']:
                drs = 2.5/3*dr
                ha = 'left'
                coil_color = color[4]    
            pl.fill(Rfill,Zfill,facecolor=coil_color,alpha=alpha,
                    edgecolor=edgecolor)
            if label and current:
                zshift = max([coil['dz']/4,0.4])
            else:
                zshift = 0
            if label: 
                pl.text(r+drs,z+zshift,name,fontsize=fs*1.1,
                        ha=ha,va='center',color=0.2*np.ones(3))
            if current: 
                pl.text(r+drs,z-zshift,'{:1.1f}MA'.format(coil['I']*1e-6),
                        fontsize=fs*1.1,ha=ha,va='center',
                        color=0.2*np.ones(3))
                                            
    def plot(self,color=None,subcoil=False,label=False,plasma=False,
                   current=False,alpha=1):
        fs = matplotlib.rcParams['legend.fontsize']
        if subcoil:
            coils = self.sub_coil
        else:
            coils = self.coil
        self.plot_coil(coils,label=label,current=current,fs=fs,
                       coil_color=color,alpha=alpha)
        if plasma:
            coils = self.plasma_coil                
            self.plot_coil(coils,coil_color=color,alpha=alpha)

    def inductance(self,dCoil=0.5,Iscale=1):
        pf = deepcopy(self)
        inv = INV(pf,Iscale=Iscale,dCoil=dCoil)
        Nf = np.array([1/inv.coil['active'][coil]['Nf']
                       for coil in inv.coil['active']])
        for i,coil in enumerate(inv.adjust_coils):
            r,z = inv.pf.coil[coil]['r'],inv.pf.coil[coil]['z']
            inv.add_psi(1,point=(r,z))
        inv.set_foreground()
        fillaments = np.dot(np.ones((len(Nf),1)),Nf.reshape(1,-1))
        self.M = 2*np.pi*inv.G*fillaments  # PF/CS inductance matrix
         
    def coil_corners(self,coils):
        R,Z = np.array([]),np.array([])
        Nc = len(coils['id'])
        dR,dZ = np.zeros(Nc),np.zeros(Nc)
        if len(coils['dR'])>0: 
            dR[Nc-len(coils['dR']):]=coils['dR']
        if len(coils['dZ'])>0: 
            dZ[Nc-len(coils['dZ']):]=coils['dZ']
        for Cid,Cdr,Cdz in zip(coils['id'],dR,dZ):
            r = self.coil['Coil'+str(Cid)]['r']
            z = self.coil['Coil'+str(Cid)]['z']
            dr = self.coil['Coil'+str(Cid)]['dr']
            dz = self.coil['Coil'+str(Cid)]['dz']
            if Cdr==0 and Cdz==0:
                R = np.append(R,[r+dr/2,r+dr/2,r-dr/2,r-dr/2])
                Z = np.append(Z,[z+dz/2,z-dz/2,z+dz/2,z-dz/2])
            else:
                R = np.append(R,r+Cdr)
                Z = np.append(Z,z+Cdz)
        return R,Z

    def fit_coils(self,Cmove,dLo=0.1):
        coils = collections.OrderedDict()
        for side in Cmove.keys():
            if isinstance(dLo,(list,tuple)):
                if not 'in' in side:
                    dL = dLo[0]
                else:
                    dL = dLo[-1]
            else:
                dL = dLo
            for index in Cmove[side]:
                coil = 'Coil'+str(index)
                dr,dz = self.Cshift(self.sf.coil[coil],side,dL)
                coils[coil] = self.sf.coil[coil]
                coils[coil]['r'] = coils[coil]['r']+dr
                coils[coil]['z'] += dz
                coils[coil]['shiftr'] = dr
                coils[coil]['shiftz'] = dz
                
        with open('../Data/'+self.conf.config+'_coil_shift.txt','w') as f:
            f.write('Ncoils = {:1.0f}\n\n'.format(len(coils.keys())))
            f.write('Name\tR[m]\t\tZ[m]\t\tshiftR[m]\tshiftZ[m]\n')
            index = sorted(map((lambda s: int(s.strip('Coil'))),coils.keys()))
            for i in index:
                coil = 'Coil'+str(i)
                f.write(coil+'\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n'.format(\
                coils[coil]['r'],coils[coil]['z'],\
                coils[coil]['shiftr'],coils[coil]['shiftz']))                                   
        return coils

class TF(object):
    
    def __init__(self,**kwargs):
        self.initalise_loops()  # initalise loop family
        if 'sf' in kwargs:
            self.sf = kwargs['sf']
        if 'profile' in kwargs:
            self.profile = kwargs['profile']
            self.update_profile()
        elif 'x_in' in kwargs and 'nTF' in kwargs:
            self.x_in = kwargs['x_in']
            self.nTF = kwargs['nTF']
        else:
            err_txt = 'insurficent key word inputs\n'
            err_txt += 'set \'profile\' or \'x_in\' and \'nTF\''
            raise ValueError(err_txt)
        self.set_inner_loop()
        
    def set_inner_loop(self):
        self.ro = np.min(self.x_in['r'])
        self.cross_section()  # coil cross-sections
        self.get_loops(self.x_in)
        
    def update_profile(self):
        self.x_in = self.profile.loop.draw()  # inner loop profile
        if hasattr(self.profile,'nTF'):
            self.nTF = self.profile.nTF
        else:
            self.nTF = 18
            warn('using default nTF: {1.0f}'.format(self.nTF))
        
    def adjust_xo(self,name,**kwargs):
        self.profile.loop.adjust_xo(name,**kwargs)
        self.update_profile()
        self.set_inner_loop()
        
    def cross_section(self,J=18.25,twall=0.045):  # MA/m2 TF current density
        self.section = {}
        self.section['case'] = {'side':0.1,'nose':0.51,'inboard':0.04,
                                'outboard':0.19,'external':0.225}
        if hasattr(self,'sf'):  # TF object initalised with sf
            BR = self.sf.eqdsk['bcentr']*self.sf.eqdsk['rcentr']
            Iturn = 1e-6*abs(2*np.pi*BR/(self.nTF*cc.mu_o))
            Acs = Iturn/J
            rwp1 = self.ro-self.section['case']['inboard']
            theta = np.pi/self.nTF
            rwall = twall/np.sin(theta)
            depth = np.tan(theta)*(rwp1-rwall+\
            np.sqrt((rwall-rwp1)**2-4*Acs/(2*np.tan(theta))))
            width = Acs/depth
            self.section['winding_pack'] = {'width':width,'depth':depth}
        else:
            warn('using default winding pack dimensions')
            self.section['winding_pack'] = {'width':0.625,'depth':1.243}
        self.rc = self.section['winding_pack']['width']/2

    def initalise_loops(self):
        self.x = {}
        for loop in ['in','wp_in','cl','wp_out','out','nose','loop']:
            self.x[loop] = {'r':[],'z':[]}

    def transition_index(self,r_in,z_in,eps=1e-4):
        npoints = len(r_in)
        r_cl = r_in[0]+eps
        upper = npoints-next((i for i,r_in_ in enumerate(r_in) if r_in_>r_cl))#+1
        lower = next((i for i,r_in_ in enumerate(r_in) if r_in_>r_cl))
        top,bottom = np.argmax(z_in), np.argmin(z_in)
        index = {'upper':upper,'lower':lower,'top':top,'bottom':bottom}
        return index
        
    def loop_dt(self,r,z,dt_in,dt_out,index):
        l = geom.length(r,z)
        L = np.array([0,l[index['lower']],l[index['bottom']],
                      l[index['top']],l[index['upper']],1])
        dR = np.array([dt_in,dt_in,dt_out,dt_out,dt_in,dt_in])
        dt = interp1d(L,dR)(l)
        return dt
 
    def get_loops(self,x):
        r,z = x['r'],x['z']
        wp = self.section['winding_pack']
        case = self.section['case']
        inboard_dt = [case['inboard'],wp['width']/2,wp['width']/2,case['nose']]
        outboard_dt = [case['outboard'],wp['width']/2,wp['width']/2,
                       case['external']]
        loops = ['wp_in','cl','wp_out','out']
        self.x['in']['r'],self.x['in']['z'] = r,z
        index = self.transition_index(self.x['in']['r'],self.x['in']['z'])
        for loop,dt_in,dt_out in zip(loops,inboard_dt,outboard_dt):
            dt = self.loop_dt(r,z,dt_in,dt_out,index)
            r,z = geom.offset(r,z,dt,close_loop=True)
            self.x[loop]['r'],self.x[loop]['z'] = r,z
        return self.x
        
    def split_loop(self):  # split inboard/outboard for fe model
        r,z = self.x['cl']['r'],self.x['cl']['z']
        index = self.transition_index(r,z)
        upper,lower = index['upper'],index['lower']
        self.x['nose']['r'] = np.append(r[upper:],r[1:lower+1])
        self.x['nose']['z'] = np.append(z[upper:],z[1:lower+1])
        self.x['loop']['r'] = r[lower+1:upper]
        self.x['loop']['z'] = z[lower+1:upper]

    def loop_interpolators(self,trim=[0,1],offset=0.75,full=False):  # outer loop coordinate interpolators
        r,z = self.x['cl']['r'],self.x['cl']['z']        
        self.fun = {'in':{},'out':{}}
        for side,sign in zip(['in','out','cl'],[-1,1,1]):  # inner/outer loop offset
            r,z = self.x[side]['r'],self.x[side]['z']
            index = self.transition_index(r,z)
            r = r[index['lower']+1:index['upper']]
            z = z[index['lower']+1:index['upper']]
            r,z = geom.offset(r,z,sign*offset)
            if full:  # full loop (including nose)
                rmid,zmid = np.mean([r[0],r[-1]]),np.mean([z[0],z[-1]])
                r = np.append(rmid,r)
                r = np.append(r,rmid)
                z = np.append(zmid,z)
                z = np.append(z,zmid)
            l = geom.length(r,z)
            lt = np.linspace(trim[0],trim[1],int(np.diff(trim)*len(l)))
            r,z = interp1d(l,r)(lt),interp1d(l,z)(lt)
            l = np.linspace(0,1,len(r))
            self.fun[side] = {'r':IUS(l,r),'z':IUS(l,z)}
            self.fun[side]['L'] = geom.length(r,z,norm=False)[-1]
            self.fun[side]['dr'] = self.fun[side]['r'].derivative()
            self.fun[side]['dz'] = self.fun[side]['z'].derivative()
     
    def Cshift(self,coil,side,dL):  # shift pf coils to tf track
        if 'in' in side:
            R,Z = self.x['in']['r'],self.x['in']['z']
        else:
            R,Z = self.x['out']['r'],self.x['out']['z']
        rc,zc = coil['r'],coil['z']
        drc,dzc = coil['dr'],coil['dz']
        rp = rc+np.array([-drc,drc,drc,-drc])/2
        zp = zc+np.array([-dzc,-dzc,dzc,dzc])/2
        nR,nZ = geom.normal(R,Z)[:2]       
        mag = np.sqrt(nR**2+nZ**2)
        nR /= mag
        nZ /= mag
        i = []
        L = np.empty(len(rp))
        dn = np.empty((2,len(rp)))
        for j,(r,z) in enumerate(zip(rp,zp)):
            i.append(np.argmin((R-r)**2+(Z-z)**2))
            dr = [r-R[i[-1]],z-Z[i[-1]]]  
            dn[:,j] = [nR[i[-1]],nZ[i[-1]]]
            L[j] = np.dot(dr,dn[:,j])
            if 'in' in side: L[j] *= -1
        jc = np.argmin(L)
        fact = dL-L[jc]
        if 'in' in side: fact *= -1
        delta = fact*dn[:,jc]
        return delta[0],delta[1]
        
    def get_loop(self,expand=0):  # generate boundary dict for elliptic
        R,Z = self.x['cl']['r'],self.x['cl']['z']
        boundary = {'R':R,'Z':Z,'expand':expand}
        return boundary

    def fill(self,write=False,plot=True,alpha=1,plot_cl=False):
        geom.polyparrot(self.x['in'],self.x['wp_in'],
                        color=0.4*np.ones(3),alpha=alpha)
        geom.polyparrot(self.x['wp_in'],self.x['wp_out'],
                        color=0.6*np.ones(3),alpha=alpha)
        geom.polyparrot(self.x['wp_out'],self.x['out'],
                        color=0.4*np.ones(3),alpha=alpha)
        if plot_cl:  # plot winding pack centre line
            pl.plot(self.x['cl']['r'],self.x['cl']['z'],
                    '-.',color=0.5*np.ones(3))
        pl.axis('equal')
        pl.axis('off')

    def support(self,**kwargs):
        self.rzGet()
        self.fill(**kwargs)
   
if __name__ is '__main__':  # test functions

    nPF,nTF = 6,16
    config = {'TF':'SN','eq':'SN_{:d}PF_{:d}TF'.format(nPF,nTF)}
    setup = Setup(config['eq'])
    sf = SF(setup.filename)
    profile = Profile(config['TF'],family='S',part='TF',nTF=nTF,obj='L',
                      load=True)
    #profile.loop.plot()
    tf = TF(profile=profile,sf=sf)
    tf.fill()

    demo = DEMO()
    demo.fill_part('Vessel')
    demo.fill_part('Blanket')
    demo.plot_ports()
  

    rp,zp = demo.port['P0']['right']['r'],demo.port['P0']['right']['z']                  
    pl.plot(rp,zp,'k',lw=3)
    '''
    xc = [rp[0],zp[0]]
    nhat = np.array([rp[1]-rp[0],zp[1]-zp[0]])
    sal.tf.loop_interpolators(offset=0)  # construct TF interpolators
    loop = sal.tf.fun['out']

    Lo = minimize_scalar(SALOME.OIS_placment,method='bounded',
                         args=(loop,(rp[0],zp[0])),bounds=[0,1]).x
    xo = [Lo,0]             
    L = minimize(SALOME.intersect,xo,method='L-BFGS-B', 
                 bounds=([0.1,0.9],[0,15]),args=(xc,nhat,loop)).x
    pl.plot(loop['r'](L[0]),loop['z'](L[0]),'o')
    '''


    
