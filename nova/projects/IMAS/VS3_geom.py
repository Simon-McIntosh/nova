import os

from nova.definitions import root_dir
from nova.frame.itergeom import ITERgeom
from nova.utilities.png_tools import data_load
import matplotlib.pyplot as plt


path = os.path.join(root_dir, "input/ITER/")
# plot = data_mine(path, 'VS3_imas_dina', (5.72, 5.94), (4.82, 5.04), save=True)

imas, dina = data_load(path, "VS3_imas_dina", date="2022_01_27")[0]

plt.plot(imas["x"], imas["y"], "C2o", label="Masanari")
plt.plot(dina["x"], dina["y"], "C0o", label="DINA vertex")

coilset = ITERgeom()
index = coilset.subframe.frame == "VS3U"
index |= coilset.subframe.frame == "VS3Uj"

coilset.plot(index)

from matplotlib.patches import Patch

axes = coilset.subframe.polyplot.axes

handles = axes.get_legend_handles_labels()[0]
handles.extend(
    [Patch(facecolor="C1", label="Conductor"), Patch(facecolor="C7", label="Jacket")]
)

axes.legend(handles=handles, ncol=1, loc="center right", bbox_to_anchor=(1.3, 0.5))

plt.axis("equal")

# f_data, P_data = H['x'], H['y']
