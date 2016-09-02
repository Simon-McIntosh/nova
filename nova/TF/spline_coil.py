import scipy as sp
import numpy as np
from scipy.special import binom as bn
        
class Scoil(object):
    
    def  __init__(self,npoints=100):
        self.npoints = npoints
        self.initalise_nodes()
        
    def initalise_nodes(self):
        self.xo = {'ro':{'value':4.486,'lb':0,'ub':0},  # inner radius
                   'r1':{'value':15.708,'lb':10,'ub':20},  # outer radius 
                   'z1':{'value':0,'lb':-1,'ub':1}, # outer node vertical shift
                   'height':{'value':17.367,'lb':15,'ub':20}, # full coil height
                   'top':{'value':0.33,'lb':0.1,'ub':0.6},  # horizontal shift
                   'bottom':{'value':0.33,'lb':0.1,'ub':0.6},  # horizontal shift
                   'upper':{'value':0.62,'lb':0.4,'ub':0.8},  # vertical shift
                   'lower':{'value':0.62,'lb':0.4,'ub':0.8}}  # vertical shift

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
    
    def segment(self,p,dl):
        n = int(np.ceil(self.npoints*dl/self.L))  # segment point number 
        t = np.linspace(0,1,n)
        curve = {'r':np.zeros(n),'z':np.zeros(n)}
        for i,pi in enumerate(p):
            for var in ['r','z']:
                curve[var] += self.basis(t,i)*pi[var]
        return curve
        
    def set_input(self,**kwargs):    
        for key in kwargs:
            if key in self.xo:
                try:  # dict
                    for k in kwargs[key]:
                        self.xo[key][k] = kwargs[key][k]
                except:  # single value
                    self.xo[key]['value'] = kwargs[key]
        for u,l in zip(['top','upper'],['bottom','lower']):
            if u in kwargs and l not in kwargs:
                self.xo[l] = self.xo[u]  # set lower equal to upper
            
    def get_xo(self):
        values = []
        for n in ['ro','r1','z1','height','top','bottom','upper','lower']:
            values.append(self.xo[n]['value'])
        return values
        
    def verticies(self,**kwargs):
        self.set_input(**kwargs)
        ro,r1,z1,height,top,bottom,upper,lower = self.get_xo()
        r,z,theta = np.zeros(5),np.zeros(5),np.zeros(5)
        r[0],z[0],theta[0] = ro,upper*height/2,np.pi/2  # upper sholder
        r[1],z[1],theta[1] = ro+top*(r1-ro),height/2,0  # top
        r[2],z[2],theta[2] = r1,z1*height/2,-np.pi/2  # outer
        r[3],z[3],theta[3] = ro+bottom*(r1-ro),-height/2,-np.pi  # bottom
        r[4],z[4],theta[4] = ro,-lower*height/2,np.pi/2  # lower sholder
        return r,z,theta
            
    def linear_loop_length(self,r,z):
        self.L = 0
        for i in range(len(r)-1):
            self.L += sp.linalg.norm([r[i+1]-r[i],z[i+1]-z[i]])

    def polybezier(self,r,z,theta,l=0.8):
        x = {'r':np.array([]),'z':np.array([])}
        self.linear_loop_length(r,z)
        for i in range(len(r)-1):
            p0 = {'r':r[i],'z':z[i],'t':theta[i],'l':l}
            p3 = {'r':r[i+1],'z':z[i+1],'t':theta[i+1]-np.pi,'l':l}
            p1,p2,dl = Scoil.control(p0,p3)
            curve = self.segment([p0,p1,p2,p3],dl)
            for var in ['r','z']:
                x[var] = np.append(x[var],curve[var][:-1])
        for var in ['r','z']:
            x[var] = np.append(x[var],curve[var][-1])
        return x
        
    def generate(self,xo,keys):
        inputs = {}
        for x,key in zip(xo,keys):
            inputs[key] = x
        l = inputs.get('l',0.8)
        r,z,theta = self.verticies(**inputs)
        x = self.polybezier(r,z,theta,l=l)
        return x
