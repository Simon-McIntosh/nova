from scipy.special import binom as bn
import numpy as np
from scipy.linalg import lstsq
from scipy.interpolate import interp1d

class bernstein(object):
    
    def __init__(self,t,n=5):
        self.t = t
        self.n = n
        self.knots()
        
    def basis(self,n,v,t):
        return bn(n,v)*t**v*(1-t)**(n-v)
    
    def knots(self,v=[]):
        self.A = np.array(np.zeros((len(self.t),self.n+1)))
        if len(v) == 0: v = range(self.n+1)
        v = np.array(v)
        v[v>self.n] = self.n
        v[v<0] = 0
        print(v)
        for i,vv in enumerate(v):
            self.A[:,i] = self.basis(self.n,vv,self.t)
            
    def fit(self,data):
        self.data = data
        b = lstsq(self.A,data)[0]
        return self.gen(b),b
        
    def spline(self,b):
        return interp1d(self.t,self.gen(b),fill_value=0,bounds_error=False)
        
    def gen(self,b):
        if len(b) == 2*(self.n+1):
            self.knots(v=b[self.n+1:])
        return np.dot(self.A,b[:self.n+1])
        
    def error(self,b):
        data = self.data
        poly = self.gen(b)
        rms = np.sqrt(np.mean((data-poly)**2))
        return rms