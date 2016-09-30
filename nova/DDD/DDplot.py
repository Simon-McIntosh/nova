import pylab as pl
import numpy as np
import shelve

class topo(object):
    
    def __init__(self, config):
        from eqlib import eq
        self.config = config
        self.path = './coil_data/'
        self.file = 'Double_Decker_Coils'
        self.coil_eq = eq(config).coil
        
    def sol(self):
        data = shelve.open(self.path+self.file+'_'+'sol')
        pl.plot(data['data-1'][:,0], data['data-1'][:,1], 'k')
        data.close()
        
    def plates(self):
        data = shelve.open(self.path+self.file+'_'+'sol')
        pl.plot(data['data-2'][:,0], data['data-2'][:,1], 'g')
        pl.plot(data['data-3'][:,0], data['data-3'][:,1], 'g')
        data.close()
        
    def TF(self):
        data = shelve.open(self.path+self.file+'_'+'TF')
        pl.plot(data['data-1'][:,0], data['data-1'][:,1], 'k-.')
        pl.plot(data['data-2'][:,0], data['data-2'][:,1], 'k-.')
        data.close()
    
    def plasma(self):
        plasma = 2*[0]  # plasma centre
        data = shelve.open(self.path+self.file+'_'+'plasma')
        pl.plot(data['data-1'][:,0], data['data-1'][:,1], 'k--')
        pl.plot(data['data-1'][[-1,0],0], data['data-1'][[-1,0],1], 'k--')
        for i in range(2):
            plasma[i] = sum(data['data-1'][:,i])/len(data['data-1'][:,i])
        pl.plot(plasma[0], plasma[1], 'rx')
        data.close()
        
    def coil(self):
        for name in self.coil_eq.keys():
            pl.plot(self.coil_eq[name]['r'], self.coil_eq[name]['z'], 'rx')  # coil centers
        '''
        from cross_coil import PFcoils
        coil = PFcoils(self.path, self.file, self.config)
        theta = np.linspace(0,2*np.pi)
        for i, name in enumerate(coil['name']):
            pl.plot(coil['r'][i], coil['z'][i], 'rx')  # coil centers
            x_circ = coil['rc'][i]*np.cos(theta)
            y_circ = coil['rc'][i]*np.sin(theta)
            if coil['I'][i] > 0:
                line = 'r-'
            else:
                line = 'b-'
            pl.plot(coil['r'][i]+x_circ, coil['z'][i]+y_circ, line)  # coil radius
        '''
    
    def coil_sheild(self):
        from cross_coil import PFcoils
        coil = PFcoils(self.path, self.file, self.config)
        theta = np.linspace(0,2*np.pi)
        for i, name in enumerate(coil['name']): 
            x_circ = (coil['rc'][i]+0.8)*np.cos(theta)
            y_circ = (coil['rc'][i]+0.8)*np.sin(theta)
            pl.plot(coil['r'][i]+x_circ, coil['z'][i]+y_circ, 'k--')  # sheilding
        
    def coil_label(self):
        from cross_coil import PFcoils
        coil = PFcoils(self.path, self.file, self.config)
        for i, name in enumerate(coil['name']):
            ax = pl.gca()
            ax.text(coil['rc'][i]+0.9+coil['r'][i],coil['z'][i], 
                    name+', '+str(coil['I'][i]/1e6),
                    horizontalalignment='left',verticalalignment='center')
        
        


