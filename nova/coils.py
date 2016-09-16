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
        if text:
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