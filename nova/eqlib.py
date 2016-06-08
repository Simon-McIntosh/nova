import csv
import numpy as np

class eq(object):
    
    def __init__(self, config):
        self.path = './eqlib_data/'
        self.Name = []
        self.coil = {}
        self.config = config
        
        if self.config == 'DD':  ## double decker
            self.files = {'I':'DDcurrent', 'pFz':'DDplasmaFz',
                          'npFz':'DDnoplasmaFz'}
        elif self.config == 'Conv':  # convential config
            self.files = {'I':'Conv_current', 'pFz':'Conv_plasmaFz',
                          'npFz':'Conv_noplasmaFz'}
                          
        f = open(self.path+self.files['I'])  
        data = csv.reader(f, delimiter=' ')
        for row in data:
            name = row[0].replace('coil_','')
            self.Name.append(name)
            self.coil[name] = {}
            
        self.locate()
                          
    def store_current(self, swing_index):
        Jmax = 2e7  # [A/m^2]
        f = open(self.path+self.files['I'])  # coil current
        data = csv.reader(f, delimiter=' ')
        for name, row in zip(self.Name, data):
            self.coil[name]['I'] = float(row[swing_index])
            self.coil[name]['rc'] = (abs(self.coil[name]['I'])/(Jmax*np.pi))**0.5
        f.close()
    
    def store_force(self, ID, swing_index):
        f = open(self.path+self.files[ID])  # vertical force plasma on||off
        data = csv.reader(f, delimiter=' ')
        for name, row in zip(self.Name, data):
            row = list(filter(None, row))
            self.coil[name][ID] = float(row[swing_index-1])*1e9
        f.close()
        
    def swing(self, swing_index):
        self.store_current(swing_index)
        self.store_force('pFz', swing_index)
        self.store_force('npFz', swing_index)
        
    def locate(self):
        import shelve
        path = './coil_data/'
        file = 'Double_Decker_Coils_coil_centres'

        coil_sequence = ['CS3L','CS2L','CS1','CS2U','CS3U', 
                      'P1','P2','P3','P4','PS1','P6C','PS3','PS4','CS3LB',
                      'P6','P5','PS2','PS5','P5B','P6B']
                      
                      
        data = shelve.open(path+file)
        for index, name in zip(range(1,len(data.keys())+1), coil_sequence):
            key = 'data-'+str(index)  
            if name in self.coil.keys():
                self.coil[name]['r'] = np.mean(data[key][:,0])
                self.coil[name]['z'] = np.mean(data[key][:,1])
        data.close()
