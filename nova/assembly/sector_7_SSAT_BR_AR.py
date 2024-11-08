"""Compare SSAT-BR fit with SSAT-AR mesurment."""

import pandas as pd

from nova.assembly.fiducialfit import FiducialFit
from nova.assembly.fiducialplotter import FiducialPlotter
from nova.assembly.transform import Rotate

coil_index = 1

ssat_br = FiducialFit(phase="SSAT BR", sectors={7: [8, 9]}, fill=False)
plot_br = FiducialPlotter(ssat_br.data)

plot_br.target(coil_index)
plot_br.fiducial("fit", color="C0", coil_index=coil_index)
plot_br.centerline("fit", label="SSAT BR target", color="C0", coil_index=coil_index)

ssat_ar = FiducialFit(phase="SSAT AR", sectors={7: [8, 9]}, fill=False)
plot_ar = FiducialPlotter(ssat_ar.data, axes=plot_br.axes)
plot_ar.fiducial(color="C1", coil_index=coil_index)
plot_ar.centerline(label="SSAT AR mesurment", color="C1", coil_index=coil_index)

plot_br.axes[1].legend(loc="center", bbox_to_anchor=(0, 1, 1, 0.1), ncol=1)
plot_br.plt.suptitle(f"Coil {ssat_br.data.coil[coil_index].values}")


data = ssat_ar.data


# delta = ssat_ar.data.fiducial - ssat_br.data.fiducial_fit
#
def to_pandas(data, target="fiducial"):
    """Evaluate fit to fiducial target."""
    target_cyl = data.fiducial_target_cyl.copy()
    target_cyl.loc[..., "r"] += data.radial_offset
    delta = Rotate.to_cylindrical(data[target]) - target_cyl
    frames = []
    for coil in data.coil.values:
        frame = delta.sel(coil=coil).to_pandas()  # .sortby("target")
        frame.loc[["C", "D", "E", "F", "G"], "r"] = ""
        frame.loc[["A", "B", "G", "H"], "z"] = ""
        frame.columns = pd.MultiIndex.from_product(
            [[f"Coil {coil}"], ["dr", "drphi", "dz"]]
        )
        frames.append(frame)
    return pd.concat(frames, axis=1)


# print(to_pandas(ssat_br.data, target="fiducial_fit_gpr"))

print(to_pandas(ssat_br.data, target="fiducial_fit_gpr"))

print(to_pandas(ssat_ar.data, target="fiducial_gpr"))

ssat_br.plot_gpr_array(coil_index, 3)
ssat_br.plt.suptitle(f"Coil {ssat_br.data.coil.values[coil_index]}")

# ssat_br.plot_fit(0, "fit")

# to_pandas(ssat_ar.data, target="fiducial")
# ssat_ar.plotter("", stage=1, coil_index=0, axes=ssat_br.plotter.axes)
