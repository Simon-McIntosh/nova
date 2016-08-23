import h5py
import numpy as np

f = h5py.File("mytestfile2.hdf5", "w")
dset = f.create_dataset("mydataset", (100,), dtype='i')
    
    
    
h5py.File("mytestfile2.hdf5", "w")    
dset[...] = np.arange(100)