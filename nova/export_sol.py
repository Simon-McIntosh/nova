import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

import pylab as pl
import shelve

path = './coil_data/'
file = 'Double_Decker_Coils'
ID = 'sol'

class spline(object):
    
    def __init__(self, path, file, ID):
        self.path = path
        self.file = file
        self.ID = ID
        
    def import_data(self):
        data = shelve.open(self.path+self.file+'_'+self.ID)
        x, y = (data['data-1'][:,0], data['data-1'][:,1])
        x, y = (1e3*x,1e3*y)  # convert to mm
        data.close()
        
        dz = ((x[1:]-x[:-1])**2+(y[1:]-y[:-1])**2)**0.5
        z = np.insert(np.cumsum(dz), 0, 0)
        z = z/z[-1]
        
        geom = np.zeros((len(x), 3))
        geom[:,0] = x
        geom[:,1] = y
        geom[:,2] = z
        
        print(min(y), max(x))
        
        return geom

    def interp(self, Np): 
        geom = self.import_data()
        x, y, z = geom[:,0], geom[:,1], geom[:,2]

        sx = UnivariateSpline(z, x, s=0)
        sy = UnivariateSpline(z, y, s=0)
        
        lin = 1  # linear interpolant
        sx_lin = interp1d(z, x)
        sy_lin = interp1d(z, y)
        
        sz = np.linspace(0,z[-1],Np)
        sol = np.zeros((len(sz),2))
        
        if lin == 1:
            sol[:,0] = sx_lin(sz)
            sol[:,1] = sy_lin(sz)
        else: 
            sol[:,0] = sx(sz)
            sol[:,1] = sy(sz)
                
        return sol

    def plot(self):
            sol = self.interp(100)
            fig = pl.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            #pl.plot(self.x, self.y, 'rx')
            #pl.plot(self.sx(sz), self.sy(sz), 'r-')
            pl.plot(sol[:,0], sol[:,1], 'k-')
            ax.axis('equal')

    def Jscript(self):
        import Jscript_functions as JS
        NPloop = 100  # number of points per loop

        dm_file = 'sol.js'
        f = open(path+dm_file, 'w')  # open file
        
        funcID, skID, psID = {}, {}, {}
        
        funcID['sol'], skID['sol'] = [], []  # sol sketches
 
        funcID['sol'].append(JS.open_function(f)) # sol sketches
        JS.set_plane_active(f)
        skID['sol'].append(JS.set_sketch(f))
        sol = self.interp(NPloop)
        JS.spline(f, skID['sol'][-1], sol[:,0], sol[:,1])
        JS.close_function(f)

        bp = JS.get_plane(f,'XY')  # base planes
        psID['sol'] = []
        offset_bp = JS.trans_plane(f, 'Y', bp, sol[0,1])  # Y-offset
        psID['sol'].append(JS.sketch(f, offset_bp, funcID['sol'][0]))

        JS.end_script(f)
        
sol = spline(path,file,ID)
sol.plot()
sol.Jscript()


