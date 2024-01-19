import h5py

import matplotlib.pyplot as plt

from nova.imas import Database

# database = Database(135005, 4, name="equilibrium")
database = Database(150601, 2, name="ece")


# current = database.get_ids("time_slice(:)/global_quantities/ip")
# print(current)

itime = 0

with h5py.File(f"{database.ids_path}/equilibrium.h5") as f:
    # ip = f["equilibrium"]["time_slice[]&global_quantities&ip"][:]  # type:ignore
    radius = f["equilibrium"]["time_slice[]&profiles_2d[]&r"][itime, 0]
    height = f["equilibrium"]["time_slice[]&profiles_2d[]&z"][itime, 0]
    psi = f["equilibrium"]["time_slice[]&profiles_2d[]&psi"][itime, 0]


plt.contour(radius, height, psi, 51)
plt.axis("equal")
plt.axis("off")
