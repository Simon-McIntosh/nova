"""Demonstrate data access via codac uda client with synchronous script."""

import time

import numpy as np
import pandas as pd

from nova.utilities.importmanager import check_import

with check_import("codac_uda"):
    from uda_client_reader.uda_client_reader_python import UdaClientReaderPython

from nova.datachain.uda import UdaQuery

uda_query = UdaQuery(pulse_id=62, duration=5, sample_number=1, sample_type=1)

client = UdaClientReaderPython("10.153.0.254", 3090)
client.resetAll()

start_time = time.perf_counter()

data = []
variables = []
for variable in uda_query.generator:
    query = uda_query(variable)
    handle = client.fetchData(query)
    if handle < 0:
        continue
    data.append(np.array(client.getDataAsDouble(handle)))
    variables.append(variable)
    client.releaseData(handle)

print(f"run time {time.perf_counter() - start_time:1.3f}s")

data = np.stack(data, 1)
time = np.linspace(
    uda_query.relative_start_time, uda_query.relative_end_time, uda_query.sample_number
)
dataframe = pd.DataFrame(data, index=time, columns=variables)
print(dataframe.iloc[-1, :])
