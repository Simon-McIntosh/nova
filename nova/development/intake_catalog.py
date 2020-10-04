import subprocess
import os

import intake
from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry
import pandas as pd
import numpy as np
from intake import open_catalog

from nova.definitions import root_dir
from nova.utilities.IO import pythonIO 


intake_dir = os.path.join(root_dir, 'data/Intake')
tmp_file = os.path.join(intake_dir, 'tmp.parquet')

cat_file = os.path.join(intake_dir, 'DINA.yaml')

cat = intake.open_catalog(cat_file)
cat.close()

# create random data sources

random = 448

np.random.seed(random)
data = pd.DataFrame(np.random.random((24, 5)), 
                    columns=['rain', 'sun', 'wind', 'grass', 'trees'])
data.to_parquet(tmp_file)  # save dataframe as parquet file
md5_file = pythonIO.hash_file(tmp_file, algorithm='md5')  # hash
#if os.path.isfile(md5_file):  # remove if present
#    os.remove(md5_file)
#os.rename(tmp_file, md5_file)  # rename

# upload
subprocess.run(['scp', tmp_file,
                'hpc-login.iter.org:/work/imas/shared/external/'
                f'assets/nova/{md5_file}'])
md5_http = f'http://static.iter.org/imas/assets/nova/{md5_file}'

source = intake.open_parquet(md5_http)
source.name = f'random{random}'
source.description = f'random {random} dataset'
source.metadata= {'IDM': 2342, 'other': '55'}
source.direct_access = True

#if os.path.isfile(cat_file):
#    with intake.open()
#cat.autoreload = False

cat._load(reload=True)

#print(cat._entries.keys())
cat = cat.add(source)
#source.export('http://static.iter.org/imas/assets/nova/tmp_cat')

'''
entries = cat._entries.copy()
entries['s3'] = source

data = {'metadata': cat.metadata, 'sources': {}}
for e in entries:
    data['sources'][e] = list(entries[e].get()._yaml()['sources'].values())[0]

#cat.save(cat_file)
'''




"""
filename = '../../Data/Intake/tmp.csv'
data.to_csv(filename)

secure_hash = pythonIO.hash_file(filename, algorithm='md5')

data.to_csv(f'../../Data/Intake/{secure_hash}')

print(secure_hash)
# Intake file to ITER cluster'
'''
subprocess.run(['scp', filename,
                'hpc-login.iter.org:/work/imas/shared/external/'
                f'assets/nova/MD5/{secure_hash}'])

#source = intake.open_csv('http://static.iter.org/imas/assets/solps-iter/MD5/fadd1f3622808187793321c944886b91')
source = intake.open_csv('http://static.iter.org/imas/assets/nova/MD5/07ec4f9d9e44fd9dcd7fd95cfd4951f9')
'''

source = intake.open_csv(f'../../Data/Intake/{secure_hash}')
source.discover()
"""
