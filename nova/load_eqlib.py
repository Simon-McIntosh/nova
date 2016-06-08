import csv
import shelve
import numpy as np
       
class PEX(object):
    
    def __init__(self,path,config,Jmax):
        self.path = path
        self.config = config
        self.Jmax = Jmax
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
        
        self.get_coil()  # store coil locations
        self.get_current()  # store current
        self.get_plasma()  # plasma shape / centre
        
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
        
class DTT(object):
    
    def __init__(self,config,Jmax):
        self.path = './eqlib_data/'
        self.config = config
        self.Jmax = Jmax
        self.Name = []
        self.coil = {}
        self.plasma = {'LCFS':{'r':np.array([]),'z':np.array([])},'rc':0,'zc':0}
        
        with open(self.path+self.config+'.vi','r') as fid:
            for row in fid:
                if 'coil_' in row.split()[0]:
                    name = row.split()[0].replace('coil_','')
                    self.coil[name] = {}
                    for i,var in enumerate(['r','z','I']):
                        self.coil[name][var] = float(row.split()[i+1])
                    self.coil[name]['rc'] = (abs(self.coil[name]['I'])/\
                    (self.Jmax*np.pi))**0.5
                if 'Reference' in row.split()[0]:
                    break
            for i in range(3):
                fid.readline()
            moment = fid.readline()  # magnetic moment
            self.Tm = {'r':moment.split()[0],'z':moment.split()[0],
                       'Bt':moment.split()[0]}
            for i in range(5):
                fid.readline()
            axis = fid.readline()  # magnetic axis
            self.plasma['rc'] = float(axis.split()[0])
            self.plasma['rc'] = float(axis.split()[1])
            for i in range(5):
                fid.readline()
            for row in fid:
                self.plasma['LCFS']['r'] = np.append(self.plasma['LCFS']['r'],
                                                     float(row.split()[0]))
                self.plasma['LCFS']['z'] = np.append(self.plasma['LCFS']['z'],
                                                     float(row.split()[1]))
                                                     
            r = self.plasma['LCFS']['r']
            z = self.plasma['LCFS']['z']
            min_loc = np.argmin(z)
            self.plasma['LCFS']['r'] = np.append(r[min_loc:],r[:min_loc+1])
            self.plasma['LCFS']['z'] = np.append(z[min_loc:],z[:min_loc+1])
