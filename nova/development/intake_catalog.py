import subprocess

import intake
from intake.catalog import Catalog
from intake.catalog.local import LocalCatalogEntry
import pandas as pd
import numpy as np

from nova.utilities.IO import pythonIO 
#mycat = Catalog.from_dict({'source1': 
#                           LocalCatalogEntry('name', 'description', 'csv')})

data = pd.DataFrame(np.random.random((24, 5)), columns=['rain', 'sun', 'wind',
                                                      ' grass', 'trees'])

filename = './tmp.csv'
data.to_csv(filename)

secure_hash = pythonIO.hash_file(filename, algorithm='md5')


# upload file to ITER cluster'
subprocess.run(['scp', filename,
                'hpc-login.iter.org:/work/imas/shared/external/'
                f'assets/nova/MD5/{secure_hash}'])

source = intake.open_csv(f'http://static.iter.org/imas/assets/nova/MD5/{secure_hash}')

source.discover()