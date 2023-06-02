import inspect

import h5py
from hdf5_udf import lib


def dynamic_dataset():
    """HDF5 user defined function."""
    from math import sin

    data = lib.getData("simple")
    dims = lib.getDims("Bv")
    for i in range(dims[0]):
        data[i] = sin(i)
    return data


with h5py.File("/tmp/data.h5", "w") as f:
    dataset = f.create_dataset("simple", (10,), dtype="float")

    print(dataset)

with open("/tmp/data.py", "w") as f:
    f.write(inspect.getsource(dynamic_dataset))

"""
    with UserDefinedFunction(
            hdf5_file='/tmp/data.h5', udf_file=tmp.name) as udf:
        udf.push_dataset(dict(name='simple',
                              datatype='float',
                              resolution=[10]))
        udf.compile()
        udf.store()
"""


#
#   f.write(inspect.getsource(dynamic_dataset))
