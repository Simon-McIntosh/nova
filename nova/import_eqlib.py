import csv
import shelve
import numpy as np

class PEX(object):
    
    def __init__(self,path,config):
        self.path = path
        self.config = config
        self.Name = []
        self.coil = {}
        
        f = open(self.path+'coilID.vi')  # read current data
        data = csv.reader(f, delimiter=' ')
        for row in data:
            if len(row) > 0:
                name = row[0].replace('coil_','')
                self.Name.append(name)
                if row[0][0] is not '#':
                    self.coil[name] = {}
        f.close()    
        
    def get_current(self):
        f = open(self.path+'current.vi')  # coil current
        data = csv.reader(f, delimiter=' ')
        for row in (data):
            if len(row) > 1:
                name = row[0].replace('coil_','')
                self.coil[name]['I'] = float(row[1])*1e6
                self.coil[name]['rc'] = (abs(self.coil[name]['I'])/(self.Jmax*np.pi))**0.5
        f.close()
     
    def get_coil(self):
        file = 'coils'
        data = shelve.open(self.path+self.config+'_'+file)
        for index, name in zip(range(1,len(data.keys())+1), self.Name):
            key = 'data-'+str(index)  
            if name in self.coil.keys():
                self.coil[name]['r'] = np.mean(data[key][:,0])
                self.coil[name]['z'] = np.mean(data[key][:,1])
        data.close()
        
    def get_plasma(self):
        file = 'plasma'
        data = shelve.open(self.path+self.config+'_'+file)
        surface = ['sol','LCFS','centre']
        self.plasma = {}
        for index,name in zip(range(1,len(surface)+1), surface):
            key = 'data-'+str(index)  
            self.plasma[name] = {}
            self.plasma[name]['r'] = data[key][:,0]
            self.plasma[name]['z'] = data[key][:,1]
        self.plasma['rc'] = np.mean(self.plasma['centre']['r'])
        self.plasma['zc'] = np.mean(self.plasma['centre']['z'])
        data.close()