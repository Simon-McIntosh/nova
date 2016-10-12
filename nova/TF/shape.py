import pylab as pl
import numpy as np
from amigo import geom
import seaborn as sns
from scipy.optimize import fmin_slsqp
import time
from DEMOxlsx import DEMO
from nova.coils import TF
from nova.loops import set_oppvar,get_oppvar,plot_oppvar
from nova.coil_cage import coil_cage

class shape(object):
    
    def __init__(self,config,tf,nTF=18,objective='L',**kwargs):
        self.tf = tf
        self.coil = tf.coil
        self.nTF = nTF
        self.objective = objective
        self.cage = coil_cage(nTF=nTF,rc=tf.rc,plasma={'config':config})
        self.objective = objective
        self.bound = {}  # initalise bounds
        for side in ['internal','external']:
            self.bound[side] = {'r':[],'z':[]}
            if side in kwargs:
                self.add_bound(kwargs[side],side)

    def add_bound(self,x,side):
        for var in ['r','z']:
            self.bound[side][var] = np.append(self.bound[side][var],x[var])
    '''
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
    ''' 
    def plot_bounds(self):
        for side,marker in zip(['internal','external'],['.','x']):
            pl.plot(self.bound[side]['r'],self.bound[side]['z'],
                    marker,markersize=6,color=sns.color_palette('Set2',10)[9])
  
    def minimise(self):
        tic = time.time()
        xnorm,bnorm = set_oppvar(self.coil.xo,self.coil.oppvar)  # normalized inputs             
        xnorm = fmin_slsqp(self.fit,xnorm,f_ieqcons=self.constraint_array,
                           bounds=bnorm,acc=0.005)
                         
        x = get_oppvar(self.coil.xo,self.coil.oppvar,xnorm)  # de-normalize
        self.coil.set_input(x=x)  # TFcoil inner loop
        x = self.tf.get_loops(self.coil.draw())
        self.tf.write(nTF=self.nTF,objective=self.objective)  # store tf coil
        
        print('optimisation time {:1.1f}s'.format(time.time()-tic))
        print('noppvar {:1.0f}'.format(len(self.coil.oppvar)))

        self.cage.set_TFcoil(x['cl'],smooth=True)
        print('ripple',self.cage.get_ripple())
        print('energy {:1.2f}GJ'.format(1e-9*self.cage.energy()))

    def constraint_array(self,xnorm,ripple_limit=0.6):
        x = get_oppvar(self.coil.xo,self.coil.oppvar,xnorm)  # de-normalize
        x = self.tf.get_loops(self.coil.draw(x=x))
        dot = np.array([])
        for side,key in zip(['internal','external'],['in','out']):
            dot = np.append(dot,self.dot_diffrence(x[key],side))

        self.cage.set_TFcoil(x['cl'])
        max_ripple = self.cage.get_ripple()
        edge_ripple = self.cage.edge_ripple(npoints=10)
        dot = np.append(dot,ripple_limit-edge_ripple)
        dot = np.append(dot,ripple_limit-max_ripple)
        return dot
        
    def fit(self,xnorm):
        x = get_oppvar(self.coil.xo,self.coil.oppvar,xnorm)  # de-normalize
        x = self.coil.draw(x=x)
        if self.objective is 'L':  # coil length
            objF = geom.length(x['r'],x['z'],norm=False)[-1]
        elif self.objective is 'E':  # stored energy
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
    config = 'DEMO_SN'
    tf = TF(config,coil_type='S',npoints=130)
    
    shp = shape(config,tf,nTF=16,objective='E')

    demo = DEMO()
    demo.fill_loops()
    demo.fill_part('Vessel')
    vv = demo.parts['Vessel']['out']
    rvv,zvv = geom.rzSLine(vv['r'],vv['z'],30)
    rvv,zvv = geom.offset(rvv,zvv,0.2)
    
    shp.add_bound({'r':rvv,'z':zvv},'internal')  # vessel
    shp.plot_bounds()
    shp.minimise()

    x = tf.coil.draw()
    tf.get_loops(x)
    tf.fill()
    #shp.cage.plot_contours(variable='ripple',n=2e3,loop=demo.fw)
    
    plot_oppvar(shp.coil.xo,shp.coil.oppvar)


