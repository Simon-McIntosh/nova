import imas
import pyuda

"""
entry = imas.DBEntry(
    "imas://localhost:56565/uda?path=/home/mcintos/Code/uda/data/105029/1&"
    "backend=hdf5",
    "r",
)
entry.open()
mag = entry.get("magnetics")
flux_loop = entry.partial_get("magnetics", "flux_loop(3)")
eq_slice = entry.get_slice(
    "equilibrium", time_requested=0.1, interpolation_method=imas.imasdef.CLOSEST_INTERP
)

"""


pyuda.Client.server = "localhost"
pyuda.Client.port = 56565
client = pyuda.Client()
result = client.get(
    "IMAS_JSON_MAP::get(path=magnetics/ids_properties/homogeneous_time, mapping=DRAFT)"
)

entry = imas.DBEntry(
    "imas://localhost:56565/uda?mapping=DRAFT&verbose=0&path=/",
    "r",
)
homogeneous_time = entry.partial_get("pf_active", "ids_properties/homogeneous_time")
print(homogeneous_time)
