import numpy as np
import pylab as pl
import scipy as sp
import pickle
import scipy.optimize as op
from scipy.optimize import fmin_slsqp
from itertools import cycle,count
import seaborn as sns
import matplotlib
import collections
import pandas as pd
from amigo.geom import Loop
import amigo.geom as geom
import nova.geqdsk
from nova.inductance import neumann
from nova.TF.TFcoil import Acoil,Dcoil,Scoil
from nova.TF.ripple import ripple
import time
from nova.config import Setup
colors = sns.color_palette('Set2',12)
    
class PF(object):
    def __init__(self,eqdsk):
        self.set_coils(eqdsk)
        
    def set_coils(self,eqdsk):
        self.coil = {}
        if eqdsk['ncoil'] > 0: 
            nC = count(0)
            for i,(r,z,dr,dz,I) in enumerate(zip(eqdsk['rc'],eqdsk['zc'],
                                                 eqdsk['drc'],eqdsk['dzc'],
                                                 eqdsk['Ic'])):
                name = 'Coil{:1.0f}'.format(next(nC))
                self.coil[name] = {'r':r,'z':z,'dr':dr,'dz':dz,'I':I,
                                   'rc':np.sqrt(dr**2+dz**2)/2}
                if i>=10:
                    print('exit set_coil loop - coils')
                    break
         
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
        
    def plot_coil(self,coils,label=False,current=False,coil_color=None,fs=12):
        if coil_color==None:
            color = cycle(sns.color_palette('Set2',len(coils.keys())))
        else:
            color = coil_color  # color itterator
        
        color = sns.color_palette('Set3',300)
        r_med = sp.median([coils[coil]['r'] for coil in coils])
        for i,name in enumerate(coils.keys()):
            coil = coils[name]
            if label and current:
                zshift = coil['dz']/4
            else:
                zshift = 0
            r,z,dr,dz = coil['r'],coil['z'],coil['dr'],coil['dz']
            Rfill = [r+dr/2,r+dr/2,r-dr/2,r-dr/2]
            Zfill = [z-dz/2,z+dz/2,z+dz/2,z-dz/2]
            if coil['I'] != 0:
                edgecolor = 'k'
            else:
                edgecolor = 'r'  
            #coil_color = next(color)
            try:
                ic = int(name.split('_')[0].split('Coil')[-1])
            except:
                ic = 0  # plasma coils
            coil_color = color[ic] 
            coil_color = color[8]
            pl.fill(Rfill,Zfill,facecolor=coil_color,alpha=1,
                    edgecolor=edgecolor)
            if label: 
                pl.text(r,z+zshift,name.replace('Coil',''),fontsize=fs*0.6,
                        ha='center',va='center',color=[0.25,0.25,0.25])
            if current: 
                if r < r_med:
                    drs = -2/3*dr
                    ha = 'right'
                else:
                    drs = 2/3*dr
                    ha = 'left'
                pl.text(r+drs,z-zshift,'{:1.1f}MA'.format(coil['I']*1e-6),
                        fontsize=fs*0.6,ha=ha,va='center',
                        color=[0.25,0.25,0.25])
                                  
    def plot(self,color=None,coils=None,label=False,plasma=False,
                   current=False):
        fs = matplotlib.rcParams['legend.fontsize']
        if coils is None:
            coils = self.coil
        self.plot_coil(coils,label=label,current=current,fs=fs,
                       coil_color=color)
        if plasma:
            coils = self.plasma_coil                
            self.plot_coil(coils,coil_color=color)
            
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
              
    def Cshift(self,coil,side,dL):
        if 'in' in side:
            R,Z = self.TFinR,self.TFinZ
        else:
            R,Z = self.TFoutR,self.TFoutZ
        rc,zc = coil['r'],coil['z']
        drc,dzc = coil['dr'],coil['dz']
        rp = rc+np.array([-drc,drc,drc,-drc])/2
        zp = zc+np.array([-dzc,-dzc,dzc,dzc])/2
        nR,nZ = self.normal(R,Z)       
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
    
    def __init__(self,npoints=100,objective='L',nTF=18,shape={}):
        self.nTF = nTF
        self.setargs(shape)
        self.objective = objective
        self.npoints = npoints
        self.width()
        #self.amp_turns()  # calculate amp turns (only with eqdsk filename)
        self.generate()

    def setargs(self,shape):  # set input data based on shape contents
        fit = shape.get('fit',True)  # fit coil if parameters avalible
        self.coil_type = shape.get('coil_type','A')  # arc type default    
        self.config = shape.get('config','tmp')  # save file prefix
        
        
        '''
        if 'config' in shape:

        else:
            for name in ['dataname','filename']:
                if name in shape:
                    setattr(self,name,shape.get(name))
        '''
        
        self.rp = ripple(nTF=self.nTF,plasma={'sf':shape.get('sf',None)})
            
        if 'R' in shape and 'Z' in shape:  # define coil by R,Z arrays
            self.datatype = 'array'
            self.Rcl,self.Zcl = shape['R'],shape['Z']  # coil centreline
            
        elif 'vessel' in shape and 'pf' in shape and fit:
            self.datatype = 'fit'  # fit coil to internal incoil + pf coils
            self.set_bound(shape)  # TF boundary conditions
            #self.bound['ro_min'] -= 0.5  # offset minimum radius
            if 'plot' in shape:
                if shape['plot']:
                    self.plot_bounds() 
        elif 'config' in shape and 'coil_type' in shape:
            self.datatype = 'file'  # load coil from file
            self.setup = Setup(shape.get('config'))
            self.setup.coils['external']['id']=[]
            self.dataname = self.setup.dataname  # TF coil geometory
            self.filename = self.setup.filename  # eqdsk file
        else:
            errtxt = '\n'
            errtxt += 'unable to load TF coil\n'
            raise ValueError(errtxt)
     
    def load_coil(self):
        if self.coil_type == 'A':  # tripple arc (5-7 parameter)
            self.coil = Acoil()
        elif self.coil_type == 'D':  # princton D (3 parameter)
            self.coil = Dcoil()
        elif self.coil_type == 'S':  # polybezier (8-16 parameter)
            self.coil = Scoil(symetric=True,nl=1)
        else:
            errtxt = '\n'
            errtxt += 'coil type \''+self.coil_type+'\'\n'
            errtxt += 'select from [A,D,S]\n'
            raise ValueError(errtxt)
         
    def set_bound(self,shape):
        self.bound = {}
        for side in ['internal','external']:
            R,Z = np.array([]),np.array([])
            if side == 'internal' and 'vessel' in shape:
                vv = shape['vessel']  # add vessel loop
                R,Z = np.append(R,vv.R),np.append(Z,vv.Z)
            if 'pf' in shape and hasattr(self,'setup'):
                Rpf,Zpf = shape['pf'].coil_corners(self.setup.coils[side])
                R,Z = np.append(R,Rpf),np.append(Z,Zpf)
            self.bound[side] = {'R':R,'Z':Z}
        #self.bound['ro_min'] = 4.35  # minimum TF radius
        if len(self.bound) == 0:
            errtxt = '\n'
            errtxt += 'Require TF bounds input,'
            errtxt += 'shape[\'vessel\'] and or shape[\'pf\'] + self.setup:\n'
            raise ValueError(errtxt)
  
    def set_oppvar(self):  # set optimization bounds and normalize
        nopp = len(self.coil.oppvar)
        xo,self.bounds = np.zeros(nopp),np.zeros((nopp,2))
        xnorm,bnorm = np.zeros(nopp),np.zeros((nopp,2))
        for i,var in enumerate(self.coil.oppvar):
            xo[i] = self.coil.xo[var]['value']
            self.bounds[i,:] = (self.coil.xo[var]['lb'],
                                self.coil.xo[var]['ub'])
            xnorm[i] = (xo[i]-self.bounds[i,0])/(self.bounds[i,1]-
                                                 self.bounds[i,0])
            bnorm[i,:] = [0,1]
        return xnorm,bnorm
        
    def get_oppvar(self,xnorm):
        xo = np.copy(xnorm)
        for i in range(len(xnorm)):
            xo[i] = xo[i]*(self.bounds[i,1]-self.bounds[i,0])+self.bounds[i,0]
        return xo
    
    def get_coil_loops(self,R,Z,profile='cl'):
        width = self.dRcoil+2*self.dRsteel
        self.R,self.Z = geom.rzSLine(R,Z,npoints=self.npoints)
        if profile == 'cl':
            self.Rcl,self.Zcl =  R,Z
            self.Rin,self.Zin = geom.offset(R,Z,-width/2)
            self.Rout,self.Zout = geom.offset(R,Z,width/2)
        elif profile == 'in':
            self.Rin,self.Zin =  R,Z
            self.Rcl,self.Zcl = geom.offset(R,Z,width/2)
            self.Rout,self.Zout = geom.offset(R,Z,width)
        elif profile == 'out':
            self.Rout,self.Zout =  R,Z
            self.Rcl,self.Zcl = geom.offset(R,Z,-width/2)
            self.Rin,self.Zin = geom.offset(R,Z,-width)
        
    def generate(self):
        if self.datatype == 'array':  # from input array
            self.get_coil_loops(self.Rcl,self.Zcl,profile='cl')
        else:
            self.load_coil()  # load coil object
            if self.datatype == 'fit': 
                tic = time.time()
                xnorm,bnorm = self.set_oppvar()  # normalized inputs
                '''
                constraints = [{'type':'ineq','fun':self.dot},
                               {'type':'ineq','fun':self.ripple}]
                
                xo = op.minimize(self.fit,xo,  #,bounds=bounds
                                 options={'disp':True,'maxiter':500,
                                          'catol':1e-6,'rhobeg':1e-3},
                                 method='COBYLA',constraints=constraints).x
                print('optimisation time {:1.1f}s'.format(time.time()-tic))
                
                xnorm = op.minimize(self.fit,xnorm,bounds=bnorm,
                                 options={'disp':True,'maxiter':500},
                                 method='SLSQP',constraints=constraints,
                                 tol=1e-2).x
                '''                    
                xnorm = fmin_slsqp(self.fit,xnorm,
                                   f_ieqcons=self.constraint_array,
                                   bounds=bnorm,acc=0.02)

                xo = self.get_oppvar(xnorm)
                self.coil.set_input(xo=xo)  # TFcoil inner loop
                
                print('optimisation time {:1.1f}s'.format(time.time()-tic))
                print('noppvar {:1.0f}'.format(len(self.coil.oppvar)))
                dataname = '../../Data/'+self.config+'_'+\
                self.coil_type+'TF.pkl'
                with open(dataname, 'wb') as output:
                    pickle.dump(self.coil.xo,output,-1)
                    pickle.dump(self.coil.oppvar,output,-1)
            elif self.datatype == 'file':
                dataname = '../../Data/'+self.config+'_'+\
                self.coil_type+'TF.pkl'
                try:
                    with open(dataname, 'rb') as input:
                        self.coil.xo = pickle.load(input)
                        self.coil.oppvar = pickle.load(input)
                except:
                    errtxt = '\n'
                    errtxt += 'file \''+dataname+'\' not found\n'
                    errtxt += 'regenerate coil profile, fit=True\n'
                    raise ValueError(errtxt)
                    
            x = self.coil.draw()
            self.get_coil_loops(x['r'],x['z'],profile='in')

            self.rp.set_TFcoil(Rcl=self.Rcl,Zcl=self.Zcl)
            print('ripple',self.rp.get_ripple())
            
    def plot_oppvar(self,eps=1e-2):
        xnorm,bnorm = self.set_oppvar()
        for var in self.coil.xo:
            self.coil.xo[var]['xnorm'] = (self.coil.xo[var]['value']-
                                          self.coil.xo[var]['lb'])/\
                                         (self.coil.xo[var]['ub']-
                                          self.coil.xo[var]['lb'])
        data = pd.DataFrame(self.coil.xo).T
        data.reset_index(level=0,inplace=True)
        pl.figure(figsize=(8,4))
        sns.set_color_codes("muted")
        sns.barplot(x='xnorm',y='index',data=data,color="b")
        sns.despine(bottom=True)
        pl.ylabel('')
        ax = pl.gca()
        ax.get_xaxis().set_visible(False)
        patch = ax.patches
        values = [self.coil.xo[var]['value'] for var in self.coil.xo]
        xnorms = [self.coil.xo[var]['xnorm'] for var in self.coil.xo]
        for p,value,xnorm,var in zip(patch,values,xnorms,self.coil.xo):
            x = p.get_width()
            y = p.get_y() 
            if xnorm < eps or xnorm > 1-eps:
                color = 'r'
            else:
                color = 'k'
            text =' {:1.3f}'.format(value)
            if var not in self.coil.oppvar:
                 text += '*'
            ax.text(x,y,text,ha='left',va='top',
                    size='small',color=color)
        pl.plot(0.5*np.ones(2),np.sort(ax.get_ylim()),'--',color=0.5*np.ones(3),
                zorder=0,lw=1)
        pl.plot(np.ones(2),np.sort(ax.get_ylim()),'-',color=0.5*np.ones(3),
                zorder=0,lw=2)
        pl.xlim([0,1])

    def constraint_array(self,xnorm,ripple_limit=0.6):
        xo = self.get_oppvar(xnorm)  # de-normalize
        x = self.coil.draw(xo=xo) 
        dot = np.array([])
        for side in ['internal','external']:
            dot = np.append(dot,self.dot_diffrence(x,side))
        self.get_coil_loops(x['r'],x['z'],profile='in')
        self.rp.set_TFcoil(Rcl=self.Rcl,Zcl=self.Zcl)
        max_ripple = self.rp.get_ripple()
        dot = np.append(dot,ripple_limit-max_ripple)
        return dot
        
    def fit(self,xnorm):
        xo = self.get_oppvar(xnorm)  # de-normalize
        x = self.coil.draw(xo=xo)
        if self.objective is 'L':  # coil length
            objF = geom.length(x['r'],x['z'],norm=False)[-1]
        else:  # coil volume
            objF = geom.loop_vol(x['r'],x['z'])
        return objF
        
    def ripple(self,xnorm,ripple_limit=0.6):
        dsum = 0
        xo = self.get_oppvar(xnorm)  # de-normalize
        x = self.coil.draw(xo=xo) 
        self.get_coil_loops(x['r'],x['z'],profile='in')
        self.rp.set_TFcoil(Rcl=self.Rcl,Zcl=self.Zcl,nTF=self.nTF)
        max_ripple = self.rp.get_ripple()
        if max_ripple > ripple_limit:
            dsum -= max_ripple-ripple_limit
        return dsum
        
    def dot(self,xnorm):
        dsum = 0
        xo = self.get_oppvar(xnorm)  # de-normalize
        x = self.coil.draw(xo=xo) 
        for side in ['internal','external']:
            dsum += self.dot_diffrence(x,side) 
        return dsum
        
    def dot_diffrence(self,x,side):
        Rloop,Zloop = x['r'],x['z']  # inside coil loop
        switch = 1 if side is 'internal' else -1
        if side is 'external':
            Rloop,Zloop = geom.offset(Rloop,Zloop,self.dRcoil+2*self.dRsteel)
        nRloop,nZloop,Rloop,Zloop = geom.normal(Rloop,Zloop)
        R,Z = self.bound[side]['R'],self.bound[side]['Z']
        #dsum = 0
        dot = np.zeros(len(R))
        for j,(r,z) in enumerate(zip(R,Z)):
            i = np.argmin((r-Rloop)**2+(z-Zloop)**2)
            dr = [Rloop[i]-r,Zloop[i]-z]  
            dn = [nRloop[i],nZloop[i]]
            dot[j] = switch*np.dot(dr,dn)
            #if dot < 0:
            #    dsum -= (dr[0]**2+dr[1]**2)
        return dot
        
    def width(self):
        self.dRcoil = 0.489
        self.dRsteel = 0.15*self.dRcoil
 
    def fill(self,write=False,plot=True,text=True):
        loop = Loop(self.Rin,self.Zin,xo=[np.mean(self.Rin),np.mean(self.Zin)])
        self.Rin,self.Zin = loop.R,loop.Z
        self.Rmid,self.Zmid = geom.offset(self.Rin,self.Zin,
                                          self.dRsteel+self.dRcoil/2)
        self.Rout,self.Zout = geom.offset(self.Rmid,self.Zmid,
                                          self.dRsteel+self.dRcoil/2)
        
        x = geom.polyloop({'in':{'r':self.Rin,'z':self.Zin},
                           'out':{'r':self.Rout,'z':self.Zout}})
        geom.polyfill(x['r'],x['z'])
        xt = [x['r'][np.argmax(x['z'])],np.mean(x['z'])]
        pl.text(xt[0],xt[1],'nTF={:1.0f}'.format(self.nTF),
                ha='center',va='center',size='large',color=0.5*np.ones(3))
        if write:
            with open('../Data/'+self.dataname+'_TFcoil.txt','w') as f:
                f.write('Rin m\t\tZin m\t\tRout m\t\tZout m\n')
                for rin,zin,rout,zout in zip(self.TFinR,self.TFinZ,
                                             self.TFoutR,self.TFoutZ):
                    f.write('{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n'.format(\
                    rin,zin,rout,zout))            
        
    def support(self,**kwargs):
        self.rzGet()
        self.fill(**kwargs)
        
    def volume_ratio(self,Rp,Zp):
        self.Pvol = loop_vol(Rp,Zp,plot=False)
        Rtf,Ztf,L = self.drawTF(self.xCoil, npoints=300)
        self.TFvol = loop_vol(Rtf,Ztf,plot=False)
        self.Rvol = self.TFvol/self.Pvol
        
    def length_ratio(self):
        self.Plength = self.length(self.Rp,self.Zp,norm=False)[-1]
        Rtf,Ztf,L = self.drawTF(self.xCoil, npoints=300)
        R,Z = geom.offset(Rtf,Ztf,self.dRsteel+self.dRcoil/2)
        self.TFlength = self.length(R,Z,norm=False)[-1]
        self.Rlength = self.TFlength/self.Plength

    def plot_bounds(self):
        for side,marker in zip(['internal','external'],['.','x']):
            pl.plot(self.bound[side]['R'],self.bound[side]['Z'],
                    marker,markersize=6,color=colors[9])
    
    '''    
    def amp_turns(self):
        if hasattr(self,'filename'):
            eqdsk = nova.geqdsk.read(self.filename)
            mu_o = 4*np.pi*1e-7  # magnetic constant [Vs/Am]
            self.Iturn = 2*np.pi*eqdsk['rcentr']*np.abs(eqdsk['bcentr'])/\
            (mu_o*self.nTF) 
        else:
            self.Iturn = 0
    
    def energy(self,plot=False,Jmax=7.2e7,Nseg=100,**kwargs):
        if 'Iturn' in kwargs:
            self.Iturn = kwargs['Iturn']
        elif 'nTF' in kwargs:
            self.nTF = kwargs['nTF']
            self.amp_turns()
        else:
            self.amp_turns()

        self.Acs = self.Iturn/Jmax
        R,Z = geom.rzSLine(self.Rcl,self.Zcl,npoints=Nseg)  # re-space
        Xo = np.zeros((Nseg,3))
        Xo[:,0],Xo[:,2] = R,Z
        theta = np.linspace(0,2*np.pi,self.nTF,endpoint=False)
        r = np.sqrt(self.Acs)
        nturn = 1
        neu = neumann()
        M = np.zeros((self.nTF))
        if plot:
            fig = pl.figure()
            ax = fig.gca(projection='3d')
        for i in np.arange(0,self.nTF):
            X = np.dot(Xo,geom.rotate(theta[i]))
            neu.setX(X)
            neu.setr(r)
            neu.setX_(Xo)
            M[i] = nturn**2*neu.calculate()
            if plot:
                ax.plot(np.append(X[:,0],X[0,0]),np.append(X[:,1],X[0,1]),
                        np.append(X[:,2],X[0,2]))
        self.Ecage = 0.5*self.Iturn**2*self.nTF*np.sum(M)
    '''

