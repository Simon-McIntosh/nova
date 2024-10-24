"""Read JT-60SA coil data and map to IDSs."""

import itertools
from pathlib import Path

import appdirs
import pandas

import numpy as np
import matplotlib.pyplot as plt

plt.plot(x := np.linspace(0, 4 * np.pi), np.sin(x))

path = Path(appdirs.user_data_dir("machine_description/coil_geometry"))

print(path)

filepath = path / "coil_vv_OP1.dat"
with open(filepath, "r") as file:
    coil_number = file.readline()
    _ = file.readline()
    row_numbers = file.readline().split()
    coil_names = []
    for nrows in [int(number) for number in row_numbers]:
        coil_names.append(file.readline().split()[-1])
        _ = next(itertools.islice(file, nrows - 2, nrows - 1))


skiprows = 3
coil_data = {}
for name, nrows in zip(coil_names, [int(number) for number in row_numbers]):

    coil_data[name] = pandas.read_csv(
        filepath,
        header=None,
        usecols=[0, 1, 2, 3, 4],
        names=["nturn", "r", "z", "dr", "dz"],
        skiprows=skiprows,
        nrows=nrows,
        sep=r"\s+",
    )
    skiprows += nrows

print(coil_data)
