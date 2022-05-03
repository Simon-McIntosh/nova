from hdf5_udf import UserDefinedFunction

def dynamic_dataset():
   from math import sin
   data = lib.getData('simple')
   dims = lib.getDims('simple')
   for i in range(dims[0]):
      data[i] = sin(i)

import inspect
with open("/tmp/udf.py", "w") as f:
   f.write(inspect.getsource(dynamic_dataset))
