import numpy as np
from scipy.spatial.transform import Rotation
import xarray


from nova.assembly.fiducialfit import FiducialFit

phase = "SSAT AR"

sectors = {5: [16, 5]}

sector = next(iter(sectors.keys()))
coils = sectors[sector]
transforms = []
for coil in coils:
    fiducial = FiducialFit(
        phase=phase,
        sectors={sector: [coil]},
        fill=False,
        infer=True,
        ilis=True,
        ilis_pcr=True,
    )
    fiducial.build()
    transforms.append(fiducial.data.opt_x)

opt_x = xarray.concat(transforms, dim="coil")

def single_coil_transform(opt_x: xarray.DataArray, free_coil: int):
    """Calculate single coil transform from two coil fits."""

    assert free_coil in opt_x.coil, f"coil not present in {opt_x.coil}"
    other_coil = [c for c in opt_x.coil.values if c != free_coil][0]

    opt_x_fixed = opt_x.sel(coil=other_coil)
    opt_x_free = opt_x.sel(coil=free_coil)

    # Translation difference
    dt = opt_x_free[:3] - opt_x_fixed[:3]

    # Rotation difference
    R_fixed = Rotation.from_euler("XYZ", opt_x_fixed[3:6], degrees=True)
    R_free = Rotation.from_euler("XYZ", opt_x_free[3:6], degrees=True)
    R_rel = R_free * R_fixed.inv()

    # Convert back to Euler angles
    euler_rel = R_rel.as_euler("XYZ", degrees=True)

    opt_x = opt_x.copy()
    opt_x.loc[free_coil, :] = np.concatenate([dt, euler_rel])
    opt_x.loc[other_coil, :] = 0

    return opt_x

single_opt_x = single_coil_transform(opt_x, 16)

print(single_opt_x)

fiducial = FiducialFit(
    phase=phase,
    sectors=sectors,
    fill=False,
    infer=True,
    ilis=True,
    ilis_pcr=True,
)

fiducial.write("SSAT AR target", opt_x=single_opt_x)
