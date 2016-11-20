import pylab as pl
import numpy as np
from amigo import geom
import seaborn as sns
from scipy.optimize import fmin_slsqp
import time
from nova.TF.DEMOxlsx import DEMO
from nova.coils import TF
from nova.loops import set_oppvar,get_oppvar,plot_oppvar,remove_oppvar,Profile
from nova.coil_cage import coil_cage
from nova.config import Setup
from nova.streamfunction import SF

class Shape(object):
    
    def __init__(self,profile,obj='L',nTF='unset',eqconf='unset',**kwargs):
        self.profile = profile
        self.loop = self.profile.loop
        self.nTF = nTF
        self.obj = obj
        self.bound = {}  # initalise bounds
        for side in ['internal','interior','external']:
            self.bound[side] = {'r':[],'z':[]}
            if side in kwargs:
                self.add_bound(kwargs[side],side)
        if nTF is not 'unset' and eqconf is not 'unset':
            self.tf = TF(self.profile)      
            self.cage = coil_cage(nTF=nTF,rc=self.tf.rc,
                                  plasma={'config':eqconf},ny=1)
            x = self.tf.get_loops(self.loop.draw())
            self.cage.set_TFcoil(x['cl'])
        else:
            if self.obj is 'E':
                errtxt = 'nTF and SFconfig keywords not set\n'
                errtxt += 'unable to calculate stored energy\n'
                errtxt += 'initalise with \'nTF\' keyword'
                raise ValueError(errtxt)
                
    def update(self):
        self.profile.load(obj=self.obj,nTF=self.nTF)
        if 'nTF' is not 'unset':
            self.tf = TF(self.profile) 
            
    def add_bound(self,x,side):
        for var in ['r','z']:
            self.bound[side][var] = np.append(self.bound[side][var],x[var])
            
    def clear_bound(self):
        for side in self.bound:
            for var in ['r','z']:
                self.bound[side][var] = np.array([])
            
    def plot_bounds(self):
        for side,marker in zip(['internal','interior','external'],
                               ['.','d','s']):
            pl.plot(self.bound[side]['r'],self.bound[side]['z'],
                    marker,markersize=6,color=sns.color_palette('Set2',10)[9])
  
    def minimise(self,verbose=False,ripple_limit=0.6,ripple=False,acc=0.002):
        tic = time.time()
        xnorm,bnorm = set_oppvar(self.loop.xo,self.loop.oppvar)  # normalize
        xnorm = fmin_slsqp(self.fit,xnorm,f_ieqcons=self.constraint_array,
                           bounds=bnorm,acc=acc,iprint=-1,
                           args=(False,ripple_limit))
        if ripple:  # re-solve with ripple constraint
            if self.nTF == 'unset':
                raise ValueError('requre \'nTF\' to solve ripple constraint')
            print('with ripple')
            xnorm = fmin_slsqp(self.fit,xnorm,f_ieqcons=self.constraint_array,
                               bounds=bnorm,acc=acc,iprint=-1,
                               args=(True,ripple_limit))     
        xo = get_oppvar(self.loop.xo,self.loop.oppvar,xnorm)  # de-normalize
        self.loop.set_input(x=xo)  # inner loop
        self.profile.write(nTF=self.nTF,obj=self.obj)  # store loop
        if verbose:
            self.toc(tic,xo)

    def toc(self,tic,xo):
        print('optimisation time {:1.1f}s'.format(time.time()-tic))
        print('noppvar {:1.0f}'.format(len(self.loop.oppvar)))
        if self.nTF is not 'unset':
            x = self.tf.get_loops(self.loop.draw(x=xo))
            self.cage.set_TFcoil(x['cl'],smooth=True)
            print('ripple',self.cage.get_ripple())
            print('energy {:1.2f}GJ'.format(1e-9*self.cage.energy()))

    def constraint_array(self,xnorm,*args):
        ripple,ripple_limit = args
        xo = get_oppvar(self.loop.xo,self.loop.oppvar,xnorm)  # de-normalize
        if ripple:  # constrain ripple contour
            x = self.tf.get_loops(self.loop.draw(x=xo))
            dot = np.array([])
            for side,key in zip(['internal','interior','external'],
                                ['in','in','out']):
                dot = np.append(dot,self.dot_diffrence(x[key],side))
            r,z = geom.rzInterp(x['cl']['r'],x['cl']['z'],npoints=50)
            self.cage.set_TFcoil({'r':r,'z':z})
            max_ripple = self.cage.get_ripple()
            edge_ripple = self.cage.edge_ripple(npoints=10)
            dot = np.append(dot,ripple_limit-edge_ripple)
            dot = np.append(dot,ripple_limit-max_ripple)
        else:  # without tf object (no ripple or energy)
            x = self.loop.draw(x=xo)
            dot = self.dot_diffrence(x,'internal')
            dot = np.append(dot,self.dot_diffrence(x,'interior'))
        return dot
        
    def fit(self,xnorm,*args):
        x = get_oppvar(self.loop.xo,self.loop.oppvar,xnorm)  # de-normalize
        x = self.loop.draw(x=x)
        if self.obj is 'L':  # coil length
            objF = geom.length(x['r'],x['z'],norm=False)[-1]
        elif self.obj is 'E':  # stored energy
            x = self.tf.get_loops(x=x)
            self.cage.set_TFcoil(x['cl'])
            objF = 1e-9*self.cage.energy()
        else:  # coil volume
            objF = geom.loop_vol(x['r'],x['z'])
        return objF
   
    def dot_diffrence(self,x,side):
        Rloop,Zloop = x['r'],x['z']  # inside coil loop
        switch = 1 if side is 'internal' else -1
        nRloop,nZloop,Rloop,Zloop = geom.normal(Rloop,Zloop)
        R,Z = self.bound[side]['r'],self.bound[side]['z']
        dot = np.zeros(len(R))
        for j,(r,z) in enumerate(zip(R,Z)):
            i = np.argmin((r-Rloop)**2+(z-Zloop)**2)
            dr = [Rloop[i]-r,Zloop[i]-z]  
            dn = [nRloop[i],nZloop[i]]
            dot[j] = switch*np.dot(dr,dn)
        return dot
        
if __name__ is '__main__': 


    config = {'TF':'SN','eq':'DEMO_SN'}
    #setup = Setup(config)
    #sf = SF(setup.filename)
    #sf.get_boundary(plot=True)
    
    profile = Profile(config['TF'],family='S',part='TF')
    shp = Shape(profile,obj='L',nTF=16,eqconf=config['eq'])
          
    #shp.cage.pattern(plot=True)

    demo = DEMO()
    #demo.fill_loops()
    demo.fill_part('Blanket')
    demo.fill_part('Vessel')
    vv = demo.parts['Vessel']['out']
    rvv,zvv = geom.rzSLine(vv['r'],vv['z'],30)
    rvv,zvv = geom.offset(rvv,zvv,0.2)
    rmin = np.min(rvv)
    rvv[rvv<=rmin+0.12] = rmin+0.12
    shp.add_bound({'r':rvv,'z':zvv},'internal')  # vessel
    shp.plot_bounds()
    shp.minimise()
    
    shp.update()
    shp.tf.fill()
    #shp.cage.plot_contours(variable='ripple',n=2e3,loop=demo.fw)
    
    plot_oppvar(shp.loop.xo,shp.loop.oppvar)


