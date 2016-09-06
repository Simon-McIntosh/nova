import scipy as sp
import numpy as np
from scipy.special import binom as bn
from scipy.special import iv as besl  
from amigo import geom
import pylab as pl
from collections import OrderedDict
import seaborn as sns
from itertools import cycle

def get_input(oppvar=[],**kwargs):
    if 'xo' in kwargs:
        inputs = {}
        xo = kwargs.get('xo')
        try:
            for var,x in zip(oppvar,xo):
                inputs[var] = x
        except:
            errtxt = '\n'
            errtxt += 'Require self.variables'
            raise ValueError(errtxt)
    elif 'inputs' in kwargs:
        inputs = kwargs.get('inputs')
    else:
        inputs = {}
    return inputs
    
def close_loop(x,npoints):
    for var in ['r','z']:
        x[var] = np.append(x[var],x[var][0])
    x['r'],x['z'] = geom.rzSLine(x['r'],x['z'],npoints=npoints)
    return x
    
def set_limit(xo):
    if xo['value'] < xo['lb']:
        xo['value'] = xo['lb']
    if xo['value'] > xo['ub']:
        xo['value'] = xo['ub']
    return xo
                    

class Acoil(object):  # tripple arc coil
    def  __init__(self,npoints=200):
        self.npoints = npoints
        self.initalise_parameters()
        self.name = 'Acoil'
        
    def initalise_parameters(self):
        limit = 1e2
        self.xo = OrderedDict()
        self.xo['ro'] = {'value':4.486,'lb':3,'ub':5}  # r origin
        self.xo['zo'] = {'value':0,'lb':-limit,'ub':limit}  # z origin
        self.xo['sl'] = {'value':6.428,'lb':0.1,'ub':limit}  # straight length
        self.xo['f1'] = {'value':2,'lb':0,'ub':limit}  # rs == f1*z small
        self.xo['f2'] = {'value':5,'lb':0,'ub':limit}  # rm == f2*rs mid
        self.xo['a1'] = {'value':45,'lb':30,'ub':60}  # small arc angle, deg
        self.xo['a2'] = {'value':75,'lb':30,'ub':120}  # middle arc angle, deg
        self.oppvar = list(self.xo.keys())
        for rmvar in ['a1','a2']:  # remove arc angles from oppvar
            self.oppvar.remove(rmvar)
        self.oppvar.remove('ro')
           
    def set_input(self,**kwargs): 
        inputs = get_input(self.oppvar,**kwargs)     
        for key in inputs:
            if key in self.xo:
                try:  # dict
                    for k in inputs[key]:
                        self.xo[key][k] = inputs[key][k]
                except:  # single value
                    self.xo[key]['value'] = inputs[key]
                self.xo[key] = set_limit(self.xo[key])
            
    def get_xo(self):
        values = []
        for n in ['ro','zo','sl','f1','f2','a1','a2']:
            values.append(self.xo[n]['value'])
        return values
                   
    def draw(self,**kwargs): 
        self.npoints = kwargs.get('npoints',self.npoints)
        self.set_input(**kwargs)
        ro,zo,sl,f1,f2,a1,a2 = self.get_xo()
        a1 *= np.pi/180  # convert to radians
        a2 *= np.pi/180
        asum = a1+a2
          # straight section
        r = ro*np.ones(2)
        z = np.array([zo,zo+sl])
          # small arc
        theta = np.linspace(0,a1,round(0.5*self.npoints*a1/np.pi))
        r = np.append(r,r[-1]+f1*(1-np.cos(theta)))
        z = np.append(z,z[-1]+f1*np.sin(theta))
          # mid arc
        theta = np.linspace(theta[-1],asum,round(0.5*self.npoints*a2/np.pi))
        r = np.append(r,r[-1]+f2*(np.cos(a1)-np.cos(theta)))
        z = np.append(z,z[-1]+f2*(np.sin(theta)-np.sin(a1)))
          # large arc 
        rl = (z[-1]-zo)/np.sin(np.pi-asum)            
        theta = np.linspace(theta[-1],np.pi,60)
        r = np.append(r, r[-1]+rl*(np.cos(np.pi-theta)-np.cos(np.pi-asum)))
        z = np.append(z,z[-1]-rl*(np.sin(asum)-np.sin(np.pi-theta)))
        r = np.append(r,r[::-1])[::-1]
        z = np.append(z,-z[::-1]+2*zo)[::-1] 
        r,z = geom.rzSLine(r,z,self.npoints)  # distribute points
        x = {'r':r,'z':z}
        return x
        
    def plot(self,inputs={}):
        x = self.draw(inputs=inputs)
        pl.plot(x['r'],x['z'])

class Dcoil(object):  # Princton D
    def  __init__(self,npoints=100):
        self.npoints = npoints
        self.initalise_radii()
        self.name = 'Dcoil'
        
    def initalise_radii(self):
        self.xo = OrderedDict()
        self.xo['r1'] = {'value':4.486,'lb':3,'ub':5}  # inner radius
        self.xo['r2'] = {'value':15.708,'lb':10,'ub':20}  # outer radius
        self.xo['dz'] = {'value':0,'lb':-10,'ub':10}  # vertical offset
        self.oppvar = list(self.xo.keys())
                   
    def set_input(self,**kwargs): 
        inputs = get_input(self.oppvar,**kwargs)   
        for key in inputs:
            if key in self.xo:
                try:  # dict
                    for k in inputs[key]:
                        self.xo[key][k] = inputs[key][k]
                except:  # single value
                    self.xo[key]['value'] = inputs[key]
                self.xo[key] = set_limit(self.xo[key])
            
    def get_xo(self):
        values = []
        for n in ['r1','r2','dz']:
            values.append(self.xo[n]['value'])
        return values
   
    def draw(self,**kwargs):
        self.npoints = kwargs.get('npoints',self.npoints)
        self.set_input(**kwargs)
        r1,r2,dz = self.get_xo()
        ro=np.sqrt(r1*r2)
        k=0.5*np.log(r2/r1)
        theta=np.linspace(-0.5*np.pi,1.5*np.pi,2*self.npoints)
        r,z = np.zeros(2*self.npoints),np.zeros(2*self.npoints)
        s = np.zeros(2*self.npoints,dtype='complex128')
        for n in np.arange(1,20):  # sum convergent series
            ds = 1j/n*(np.exp(-1j*n*theta)-1)*(1+np.exp(1j*n*(theta+np.pi)))*\
            np.exp(1j*n*np.pi/2)*(besl(n-1,k)+besl(n+1,k))/2
            s += ds
            if np.max(abs(ds)) < 1e-14:
                break
        z=abs(ro*k*(besl(1,k)*theta+s))
        r=ro*np.exp(k*np.sin(theta))
        z -= np.mean(z)
        r,z = geom.space(r,z,self.npoints)
        z += dz  # vertical shift
        x = {'r':r,'z':z}
        x = close_loop(x,self.npoints)
        return x
        
    def plot(self,inputs={}):
        x = self.draw(inputs=inputs)
        pl.plot(x['r'],x['z'])
        
class Scoil(object):  # polybezier
    def  __init__(self,npoints=200,symetric=True,nl=8):
        self.npoints = npoints
        self.symetric = symetric
        self.check_tension_length(nl)
        self.initalise_nodes()
        self.name = 'Scoil'

    def initalise_nodes(self):
        self.xo = OrderedDict()
        self.xo['r1'] = {'value':4.486,'lb':4,'ub':6}  # inner radius
        self.xo['r2'] = {'value':15.708,'lb':10,'ub':25}  # outer radius 
        self.xo['z2'] = {'value':0,'lb':-0.3,'ub':0.3} # outer node vertical shift
        self.xo['height'] = {'value':17.367,'lb':0.1,'ub':50} # full coil height
        self.xo['top'] = {'value':0.33,'lb':0,'ub':1}  # horizontal shift
        self.xo['upper'] = {'value':0.62,'lb':0.55,'ub':1}  # vertical shift
        self.set_lower()  # lower coil parameters (bottom,lower)
        self.set_control_lengths({'value':0.8,'lb':0.6,'ub':1.5})  # 1/tesion
        self.xo['dz'] = {'value':0,'lb':-10,'ub':10}  # vertical offset
        self.oppvar = list(self.xo.keys())
        self.oppvar.remove('r1')
        
    def check_tension_length(self,nl):
        if nl in [1,2,4,8]:
            self.nl = nl
        else:
            errtxt = '\n'
            errtxt += 'Select Scoil tension length from [1,2,4,8]\n'
            raise ValueError(errtxt)
            
    def set_lower(self):
        if not self.symetric:
            for u,l in zip(['top','upper'],['bottom','lower']):
                self.xo[l] = {}
                for key in self.xo[u]:
                    self.xo[l][key] = self.xo[u][key]
                    
    def set_control_lengths(self,l):
        if self.nl == 1:
            self.xo['l'] = l.copy() 
            self.lkey = ['l','l','l','l','l','l','l','l']
        elif self.nl == 2:
            for i in range(2):
                self.xo['l{:1.0f}'.format(i)] = l.copy()  # inner / outer seg
            self.lkey = ['l0','l0','l1','l1','l1','l1','l0','l0']
        elif self.nl == 4:
            if self.symetric:
                for i in range(2):  # 4 controls, up/down symetric
                    self.xo['l{:1.0f}s'.format(i)] = l.copy()  # start
                    self.xo['l{:1.0f}e'.format(i)] = l.copy()  # end
                self.lkey = ['l0s','l0e','l1s','l1e','l1e','l1s','l0e','l0s']
            else:
                for i in range(4):
                    self.xo['l{:1.0f}'.format(i)] = l.copy()  # all segments
                self.lkey = ['l0','l0','l1','l1','l2','l2','l3','l3']
        elif self.nl == 8:
            for i in range(4):  # full control (8)
                self.xo['l{:1.0f}s'.format(i)] = l.copy()  # start
                self.xo['l{:1.0f}e'.format(i)] = l.copy()  # end
            self.lkey = ['l0s','l0e','l1s','l1e','l2s','l2e','l3s','l3e']
           
    def get_xo(self):
        values = []
        for var in ['r1','r2','z2','height','top',
                    'bottom','upper','lower','dz']:
            if var not in self.xo:
                if var == 'bottom':
                    var = 'top'
                if var == 'lower':
                    var = 'upper'
            values.append(self.xo[var]['value'])
        return values

    def get_l(self):
        ls,le = [],[]  # start,end
        for i in range(4):
            ls.append(self.xo[self.lkey[2*i]]['value'])
            le.append(self.xo[self.lkey[2*i+1]]['value'])
        return ls,le

    def basis(self,t,v):
        n = 3  # spline order
        return bn(n,v)*t**v*(1-t)**(n-v)

    def midpoints(p):  # convert polar to cartesian
        r = p['r']+p['l']*np.cos(p['t'])
        z = p['z']+p['l']*np.sin(p['t'])
        return r,z

    def control(p0,p3):  # add control points (length and theta or midpoint)
        p1,p2 = {},{}
        xm,ym = np.mean([p0['r'],p3['r']]),np.mean([p0['z'],p3['z']])
        dl = sp.linalg.norm([p3['r']-p0['r'],p3['z']-p0['z']])
        for p,pm in zip([p0,p3],[p1,p2]):
            if 'l' not in p:  # add midpoint length
                p['l'] = dl/2
            else:
                p['l'] *= dl/2
            if 't' not in p:  # add midpoint angle
                p['t'] = np.arctan2(ym-p['y'],xm-p['x'])
            pm['r'],pm['z'] = Scoil.midpoints(p)
        return p1,p2,dl
            
    def set_input(self,**kwargs): 
        inputs = get_input(self.oppvar,**kwargs)    
        for key in inputs:
            if key in self.xo:
                try:  # dict
                    for k in inputs[key]:
                        self.xo[key][k] = inputs[key][k]
                except:  # single value
                    self.xo[key]['value'] = inputs[key]
                self.xo[key] = set_limit(self.xo[key])
        for u,l in zip(['top','upper'],['bottom','lower']):
            if u in inputs and l not in inputs:
                self.xo[l] = self.xo[u]  # set lower equal to upper
            
  
    def verticies(self):
        r1,r2,z2,height,top,bottom,upper,lower,dz = self.get_xo()
        r,z,theta = np.zeros(5),np.zeros(5),np.zeros(5)
        r[0],z[0],theta[0] = r1,upper*height/2,np.pi/2  # upper sholder
        r[1],z[1],theta[1] = r1+top*(r2-r1),height/2,0  # top
        r[2],z[2],theta[2] = r2,z2*height/2,-np.pi/2  # outer
        r[3],z[3],theta[3] = r1+bottom*(r2-r1),-height/2,-np.pi  # bottom
        r[4],z[4],theta[4] = r1,-lower*height/2,np.pi/2  # lower sholder
        z += dz  # vertical coil offset
        return r,z,theta
            
    def linear_loop_length(self,r,z):
        self.L = 0
        for i in range(len(r)-1):
            self.L += sp.linalg.norm([r[i+1]-r[i],z[i+1]-z[i]])
            
    def segment(self,p,dl):
        n = int(np.ceil(self.npoints*dl/self.L))  # segment point number 
        t = np.linspace(0,1,n)
        curve = {'r':np.zeros(n),'z':np.zeros(n)}
        for i,pi in enumerate(p):
            for var in ['r','z']:
                curve[var] += self.basis(t,i)*pi[var]
        return curve

    def polybezier(self,r,z,theta):
        x = {'r':np.array([]),'z':np.array([])}
        self.p = []
        self.linear_loop_length(r,z)
        ls,le = self.get_l()
        for i in range(len(r)-1):
            p0 = {'r':r[i],'z':z[i],'t':theta[i],'l':ls[i]}
            p3 = {'r':r[i+1],'z':z[i+1],'t':theta[i+1]-np.pi,'l':le[i]}
            p1,p2,dl = Scoil.control(p0,p3)
            curve = self.segment([p0,p1,p2,p3],dl)
            self.p.append({'p0':p0,'p1':p1,'p2':p2,'p3':p3})
            for var in ['r','z']:
                x[var] = np.append(x[var],curve[var][:-1])
        for var in ['r','z']:
            x[var] = np.append(x[var],curve[var][-1])
            x[var] = x[var][::-1]
        return x
        
    def draw(self,**kwargs):
        self.npoints = kwargs.get('npoints',self.npoints)
        self.set_input(**kwargs)  
        r,z,theta = self.verticies()
        x = self.polybezier(r,z,theta)
        x = close_loop(x,self.npoints)
        return x    
        
    def plot(self,inputs={}):
        color = cycle(sns.color_palette('Set2',5))
        x = self.draw(inputs=inputs)
        r,z,theta = self.verticies()
        pl.plot(r,z,'s',color=next(color))
        pl.plot(x['r'],x['z'],color=next(color))
        c1,c2 = next(color),next(color)
        for p in self.p:
            pl.plot([p['p0']['r'],p['p1']['r']],
                    [p['p0']['z'],p['p1']['z']],color=c1)
            pl.plot(p['p1']['r'],p['p1']['z'],'o',color=c2)
            pl.plot([p['p3']['r'],p['p2']['r']],
                    [p['p3']['z'],p['p2']['z']],color=c1)
            pl.plot(p['p2']['r'],p['p2']['z'],'o',color=c2)

if __name__ is '__main__':  # plot coil classes
    #coil = Acoil()
    #x = coil.plot()
    
    coil = Scoil()
    x = coil.plot({'l1s':1.5,'top':0.8})
    '''
    coil = Dcoil()
    x = coil.draw()
    pl.plot(x['r'],x['z'])
    '''
    pl.axis('equal')

