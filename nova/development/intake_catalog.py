import os

import intake
import pandas as pd
import numpy as np

from nova.definitions import root_dir
from nova.utilities.IO import pythonIO


intake_dir = os.path.join(root_dir, "data/Intake")
tmp_file = os.path.join(intake_dir, "tmp.nc")
cat_file = os.path.join(intake_dir, "DINA.yaml")

cat = intake.open_catalog(cat_file)
# cat.close()


# create random data sources
random = 500
np.random.seed(random)
data = pd.DataFrame(
    np.random.random((24, 5)), columns=["rain", "sun", "wind", "grass", "trees"]
)
data.to_csv(tmp_file)

md5_hash = pythonIO.hash_file(tmp_file, algorithm="md5")
md5_file = os.path.join(intake_dir, md5_hash)
md5_http = f"http://static.iter.org/imas/assets/nova/MD5/{md5_hash}"

"""
subprocess.run(['scp', '-r', tmp_file,
                'hpc-login.iter.org:/work/imas/shared/external/'
                f'assets/nova/MD5/{md5_hash}'])
"""

source = intake.open_csv(md5_http)
source.name = f"random{random}"
source.description = f"random {random} dataset"
source.metadata = {"IDM": 2342, "other": "55"}

print(source.discover())
print(source)
cat.add(source)

"""
md5_hash = hashlib.md5(
    pd.util.hash_pandas_object(data, index=True).values).hexdigest()


'''

# upload

source = intake.upload(data, md5_file)  # upload local md5 file
subprocess.run(['scp', '-r', md5_file,
                'hpc-login.iter.org:/work/imas/shared/external/'
                f'assets/nova/MD5/{md5_hash}'])
'''
#print(md5_http)



'''

#if os.path.isfile(cat_file):
#    with intake.open()
#cat.autoreload = False

#cat._load(reload=True)

#print(cat._entries.keys())
#cat = cat.add(source)
#source.export('http://static.iter.org/imas/assets/nova/tmp_cat')
'''

'''
entries = cat._entries.copy()
entries['s3'] = source

data = {'metadata': cat.metadata, 'sources': {}}
for e in entries:
    data['sources'][e] = list(entries[e].get()._yaml()['sources'].values())[0]

#cat.save(cat_file)
'''
"""


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
