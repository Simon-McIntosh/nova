import numpy as np
import pylab as pl
import pickle
import scipy.optimize as op
from itertools import cycle,count
import seaborn as sns
import matplotlib
import collections
from amigo.geom import Loop
import amigo.geom as geom
import nova.geqdsk
from nova.inductance import neumann
from nova.TF.TFcoil import Acoil,Dcoil,Scoil
from nova.TF.ripple import ripple
import time
from nova.config import Setup

def loop_vol(R,Z,plot=False):
    imin,imax = np.argmin(Z),np.argmax(Z)
    Rin = np.append(R[::-1][:imin+1][::-1],R[:imin+1])
    Zin = np.append(Z[::-1][:imin+1][::-1],Z[:imin+1])
    Rout = R[imin:imax+1]
    Zout = Z[imin:imax+1]
    if plot:
        pl.plot(R[0],Z[0],'bo')
        pl.plot(R[20],Z[20],'bd')
        pl.plot(Rin,Zin,'bx-')
        pl.plot(Rout,Zout,'gx-')
    return vol_calc(Rout,Zout)-vol_calc(Rin,Zin)
    
def vol_calc(R,Z):
    dR = np.diff(R)
    dZ = np.diff(Z)
    V = 0
    for r,dr,dz in zip(R[:-1],dR,dZ):
        V += np.abs((r+dr/2)**2*dz)
    V *= np.pi
    return V
    
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
                #if i>=10:
                #    print('exit set_coil loop - coils')
                #    break
         
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
                if r<self.Xpoint[0]:
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
    
    def __init__(self,npoints=200,objective='L',nTF=18,shape={}):
        self.setargs(shape)
        self.objective = objective
        self.npoints = npoints
        self.width()
        self.nTF = nTF
        self.amp_turns()  # calculate amp turns (only with eqdsk filename)
        self.generate()

        
    def setargs(self,shape):  # set input data based on shape contents
        fit = shape.get('fit',True)  # fit coil if parameters avalible
        self.coil_type = shape.get('coil_type','unset')    
        self.config = shape.get('config','tmp')
        if 'config' in shape:
            self.setup = Setup(shape.get('config'))
            self.dataname = self.setup.dataname
            self.filename = self.setup.filename
        else:
            for name in ['dataname','filename']:
                if name in shape:
                    setattr(self,name,shape.get(name))
        if 'R' in shape and 'Z' in shape:  # define coil by R,Z arrays
            self.datatype = 'array'
            self.Rcl,self.Zcl = shape['R'],shape['Z']  # coil centreline
        elif 'vessel' in shape and 'pf' in shape and fit:
            self.datatype = 'fit'  # fit coil to internal incoil + pf coils
            self.set_bound(shape)  # TF boundary conditions
            self.bound['ro_min'] -= 0.5  # offset minimum radius
            if 'plot' in shape:
                if shape['plot']:
                    self.plot_bounds() 
        elif 'config' in shape and 'coil_type' in shape:
            self.datatype = 'file'  # load coil from file
        else:
            errtxt = '\n'
            errtxt += 'unable to load TF coil\n'
            raise ValueError(errtxt)
     
    def load_coil(self):
        if self.coil_type == 'A':  # tripple arc (5-7 parameter)
            self.coil = Acoil()
        elif self.coil_type == 'D':  # princton D (3 parameter)
            self.coil = Dcoil()
        elif self.coil_type == 'S':  # polybezier (8-15 parameter)
            self.coil = Scoil(symetric=True,nl=1)
        else:
            errtxt = '\n'
            errtxt += 'coil type \''+self.coil_type+'\'\n'
            errtxt += 'select from [A,D,S]\n'
            raise ValueError(errtxt)
         
    def set_bound(self,shape):
        if 'pf' in shape and 'vessel' in shape:  # requre vessel,pf
            pf = shape['pf']
            vv = shape['vessel']
            Rvv,Zvv = geom.rzSLine(vv.R,vv.Z,npoints=30)
        else:
            errtxt = '\n'
            errtxt += 'Require following input,'
            errtxt += 'bounds={\'pf\':pf,\'vessel\':rb.loop}:\n'
            raise ValueError(errtxt)
        Rin,Zin = pf.coil_corners(self.setup.coils['internal'])  # internal    
        Rex,Zex = pf.coil_corners(self.setup.coils['external'])  # external 
        ro_min = 4.35  # minimum TF radius
        internal = {'R':np.append(Rvv,Rin),'Z':np.append(Zvv,Zin)}
        external = {'R':Rex,'Z':Zex}
        self.bound = {'in':internal,'out':external,'ro_min':ro_min}
        
    def get_oppvar(self):
        xo,bounds = [],[]
        for var in self.coil.oppvar:
            xo.append(self.coil.xo[var]['value'])
            bounds.append((self.coil.xo[var]['lb'],
                           self.coil.xo[var]['ub'])) 
        return xo,bounds
    
    def get_coil_loops(self,R,Z,profile='cl'):
        width = self.dRcoil+2*self.dRsteel
        self.R,self.Z = geom.rzSLine(R,Z,npoints=self.npoints)
        if profile == 'cl':
            self.Rcl,self.Zcl =  R,Z
            self.Rin,self.Zin = geom.offset(R,Z,-width/2)
            #self.Rout,self.Zout = geom.offset(R,Z,width/2)
        elif profile == 'in':
            self.Rin,self.Zin =  R,Z
            self.Rcl,self.Zcl = geom.offset(R,Z,width/2)
            #self.Rout,self.Zout = geom.offset(R,Z,width)
        elif profile == 'out':
            #self.Rout,self.Zout =  R,Z
            self.Rcl,self.Zcl = geom.offset(R,Z,-width/2)
            self.Rin,self.Zin = geom.offset(R,Z,-width)
        
    def generate(self):
        if self.datatype == 'array':  # from input array
            self.get_coil_loops(self.Rcl,self.Zcl,profile='cl')
        else:
            self.load_coil()  # load coil object
            if self.datatype == 'fit': 
                tic = time.time()
                xo,bounds = self.get_oppvar()
                constraints = {'type':'ineq','fun':self.dot}
                xo = op.minimize(self.fit,xo,bounds=bounds,
                                 options={'disp':True,'maxiter':500},
                                 method='SLSQP',constraints=constraints,
                                 tol=1e-2).x
                self.coil.set_input(xo=xo)  # coil centerline
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
            print('nTF',self.nTF,'Iturn',self.Iturn)
            coil = {'Rcl':self.Rcl,'Zcl':self.Zcl,
                    'nTF':self.nTF,'Iturn':self.Iturn}
            rp = ripple(plasma={'config':'SN'},coil=coil)
            print('ripple',rp.get_ripple())
            rp.plot_loops()
            #self.fill()
            '''
            referance = Dcoil()
            x = referance.draw()
            referance.plot()
            self.get_coil_loops(x['r'],x['z'],profile='in')
            coil = {'Rcl':self.Rcl,'Zcl':self.Zcl,
                    'nTF':self.nTF,'Iturn':self.Iturn}
            rp = ripple(plasma={'config':'SN'},coil=coil)
            print('ref ripple',rp.get_ripple())
            '''
            

    def fit(self,xo):
        x = self.coil.draw(xo=xo)
        if self.objective is 'L':  # coil length
            objF = geom.length(x['r'],x['z'],norm=False)[-1]
        else:  # coil volume
            objF = loop_vol(x['r'],x['z'])
        return objF
        
    def dot(self,xo):
        dsum = 0
        x = self.coil.draw(xo=xo) 
        self.get_coil_loops(x['r'],x['z'],profile='in')
        coil = {'Rcl':self.Rcl,'Zcl':self.Zcl,
                'nTF':self.nTF,'Iturn':self.Iturn}
        rp = ripple(plasma={'config':'SN'},coil=coil)
        max_ripple = rp.get_ripple()
        if max_ripple > 2.5:
            dsum -= max_ripple-2.5
            
        for side in ['in','out']:
            dsum += self.dot_diffrence(x,side) 
        return dsum
        
    def dot_diffrence(self,x,side):
        Rloop,Zloop = x['r'],x['z']  # inside coil loop
        switch = 1 if side is 'in' else -1
        if side is 'out':
            Rloop,Zloop = geom.offset(Rloop,Zloop,self.dRcoil+2*self.dRsteel)
        nRloop,nZloop = geom.normal(Rloop,Zloop)
        R,Z = self.bound[side]['R'],self.bound[side]['Z']
        dsum = 0
        for r,z in zip(R,Z):
            i = np.argmin((r-Rloop)**2+(z-Zloop)**2)
            dr = [Rloop[i]-r,Zloop[i]-z]  
            dn = [nRloop[i],nZloop[i]]
            dot = switch*np.dot(dr,dn)
            if dot < 0:
                dsum -= (dr[0]**2+dr[1]**2)
        return dsum
        
    def width(self):
        self.dRcoil = 0.489
        self.dRsteel = 0.15*self.dRcoil
 
    def fill(self,write=False,plot=True):
        loop = Loop(self.Rin,self.Zin,[np.mean(self.Rin),np.mean(self.Zin)])
        self.Rin,self.Zin = loop.R,loop.Z
        
        loop.rzPut()
        loop.fill(dR=0,dt=self.dRsteel,trim=None,color=0.4*np.ones(3),alpha=1,
                  part_fill=False,loop=True,plot=plot)
        loop.fill(dt=self.dRcoil/2,trim=None,color=0.7*np.ones(3),alpha=1,
                  label='TF coil',part_fill=False,loop=True,plot=plot)
        self.Rmid,self.Zmid = loop.R,loop.Z  
        loop.fill(dt=self.dRcoil/2,trim=None,color=0.7*np.ones(3),alpha=1,
                  label='TF coil',part_fill=False,loop=True,plot=plot)
        loop.fill(dt=self.dRsteel,trim=None,color=0.4*np.ones(3),alpha=1,
                  label='TF support',part_fill=False,loop=True,plot=plot)
        self.Rout,self.Zout = loop.R,loop.Z
        
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
        #pl.plot(self.Rp,self.Zp,'rx-')
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
        pl.plot(self.bound['in']['R'],self.bound['in']['Z'],'d',
                markersize=3)
        pl.plot(self.bound['out']['R'],self.bound['out']['Z'],'o',
                markersize=3)
        pl.plot(self.bound['ro_min'],0,'*',markersize=3)
        
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
        R,Z = geom.rzSLine(self.Rmid,self.Zmid,npoints=Nseg)  # re-space
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
        

