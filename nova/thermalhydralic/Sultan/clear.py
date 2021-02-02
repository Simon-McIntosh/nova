"""Remove all local data."""
import os
import shutil

from nova.definitions import root_dir

data_dir = os.path.join(root_dir, 'data/Sultan')

experiments = [experiment for experiment in os.listdir(data_dir)
               if experiment[0] != '.']
for experiment in experiments:
    local_dir = os.path.join(data_dir, experiment, 'local')
    if os.path.isdir(local_dir):
        #print(local_dir)
        #shutil.rmtree(local_dir)
        filename = os.path.join(local_dir, 'fluidresponse.h5')
        if os.path.isfile(filename):
            print(filename)
            os.remove(filename)
