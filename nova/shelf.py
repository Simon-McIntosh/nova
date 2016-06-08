import pickle
from collections import OrderedDict as od

class PKL(object):
    
    def __init__(self,file,directory='./plot_data/'):
        self.file = directory+file+'.pkl'
        self.data = {'rb':None,'sf':None,'eq':None,'cc':None,'inv':None,
                     'ax':None}
        self.data = od(sorted(self.data.items(),key=lambda x:x[0]))
        self.inmemory = False
        
    def add(self,data):
        for key in data:
            if key in self.data.keys():
                if key == 'sf':
                    data[key].remove_contour()
                self.data[key] = data[key]

    def write(self,**kwargs):
        if 'data' in kwargs.keys(): self.add(kwargs.get('data'))  # add data
        with open(self.file, 'wb') as output:
            for key in self.data:
                pickle.dump(self.data[key],output,-1)

    def read(self):
        with open(self.file, 'rb') as input:
            for key in self.data:
                self.data[key] = pickle.load(input)
        self.inmemory = True
        self.cross_refernace()
    
    def notNone(self,key):
        return self.data[key] is not None
        
    def cross_refernace(self):
        if self.notNone('eq'):
            if self.notNone('sf'):
                self.data['eq'].sf = self.data['sf']
        if self.notNone('inv'):
            if self.notNone('sf'):
                self.data['inv'].sf = self.data['sf']
            if self.notNone('eq'):
                self.data['inv'].eq = self.data['eq']
           
    def fetch(self,keys):
        if not self.inmemory: self.read()
        data = []
        print(keys)
        for key in keys:
            if key in self.data.keys():
                if key == 'sf':
                    self.data[key].set_contour()
                data.append(self.data[key])
        return data
        