import pylab as pl
import numpy as np
import shelve

class topo(object):
    
    def __init__(self, config):
        from eqlib import eq
        self.config = config
        self.path = './coil_data/'
        self.file = 'Double_Decker_Coils'
        eqlib = eq(config)
        eqlib.swing(3)  # set current swing for rc calc
        self.coil = eqlib.coil
        
    def plasma_coils(self, coil):
        for name in coil.keys():
            if 'plasma' in name:
                pl.plot(coil[name]['r'], coil[name]['z'], 'bx')
        
    def sol(self):
        data = shelve.open(self.path+self.file+'_'+'sol')
        pl.plot(data['data-1'][:,0], data['data-1'][:,1], 'r')
        data.close()
        
    def P6_support(self, coil):
        data = shelve.open(self.path+self.file+'_'+'sol')
        r_sol = data['data-1'][:,0]
        z_sol = data['data-1'][:,1]
        data.close()
        
        # upper pinch
        r = [r for r,z in zip(r_sol,z_sol) if r>9 and z>-7.5]
        z = [z for r,z in zip(r_sol,z_sol) if r>9 and z>-7.5]
        ind = z.index(min(z))
        r_min = r[ind]
        z_min = z[ind]
        
        # lower pinch
        r = [r for r,z in zip(r_sol,z_sol) if r>9 and z<-7.5]
        z = [z for r,z in zip(r_sol,z_sol) if r>9 and z<-7.5]
        ind = z.index(max(z))
        r_max = r[ind]
        z_max = z[ind]
        
        Lpinch = 0.9*(z_max-z_min)
        
        r = np.mean([r_min,r_max])
        z = np.mean([z_min,z_max])

        pl.plot(r,z_min, 'rx')
        pl.plot(r,z_max, 'rx')
        pl.plot(r,z, 'ko')
        
        pl.plot([r,coil['P6']['r']],[z,coil['P6']['z']], 'k-')
        
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
        #pl.plot(plasma[0], plasma[1], 'rx')
        data.close()
        
    def coil_fill(self):
        fig = pl.gcf()
        for name in self.coil.keys():
            if self.coil[name]['I'] > 0:
                color = 'r'
            else:
                color = 'b'    
            circle=pl.Circle((self.coil[name]['r'],
                              self.coil[name]['z']),self.coil[name]['rc'],
                             color=color)
            fig.gca().add_artist(circle)
            
    def coil_loc(self):
        theta = np.linspace(0,2*np.pi)
        for name in self.coil.keys():
            #pl.plot(self.coil[name]['r'], self.coil[name]['z'], 'rx')  # coil centers
            x_circ = self.coil[name]['rc']*np.cos(theta)
            y_circ = self.coil[name]['rc']*np.sin(theta)
            if self.coil[name]['I'] > 0:
                line = 'r-'
            else:
                line = 'b-'
            pl.plot(self.coil[name]['r']+x_circ, self.coil[name]['z']+y_circ, line) 
    
    def coil_sheild(self):
        theta = np.linspace(0,2*np.pi)
        for name in self.coil.keys(): 
            x_circ = (self.coil[name]['rc']+0.8)*np.cos(theta)
            y_circ = (self.coil[name]['rc']+0.8)*np.sin(theta)
            pl.plot(self.coil[name]['r']+x_circ, self.coil[name]['z']+y_circ, 'k--') 
        
    def coil_label(self):
        for name in self.coil.keys():
            ax = pl.gca()
            ax.text(self.coil[name]['rc']+self.coil[name]['r'],
                    self.coil[name]['z'], name+', '+\
                    '{:1.1f}'.format(self.coil[name]['I']/1e6),
                    horizontalalignment='left',verticalalignment='center')
        
        


