from multiprocessing import Pool

from etna.coil_geom import configure
from etna.coil_apdl import coil_map
from itertools import combinations_with_replacement as cr
import numpy as np
'''

cgeom = configure('TF',Ndp=0,Nloop=0)
cmap = coil_map(cgeom)
cmap.wrap()

a = cr(range(3),2)
X = cgeom.loop(np.mean(cmap.loops['x']),0) 
'''


class F(object):
    def __init__(self):
        self.a = 1
        
    def f(self,x):
        return x*x

if __name__ == '__main__':
    f = F()
    pool = Pool()
    L = pool.map(f.f,[1,2,3])
    pool.close()
    pool.join()
    print(L)
