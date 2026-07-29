# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:50:00 2024

@author: mcintos
"""

import appdirs
import intake
import matplotlib.pyplot as plt
import numpy as np

catalog = intake.open_catalog("https://mastapp.site/intake/catalog.yml")
storage_options = {
    "cache_storage": appdirs.user_cache_dir("fair-mast", False),
    "s3": {"anon": True, "endpoint_url": "https://s3.echo.stfc.ac.uk"},
}

sources_df = catalog.index.level1.sources().read()

shot_id = 30419
# shot_id = 30471
sources_df = sources_df.loc[sources_df.shot_id == shot_id]

efit_url = sources_df.loc[sources_df.name == "efm"].iloc[0].url
currents_url = sources_df.loc[sources_df.name == "amc"].iloc[0].url

currents = catalog.level1.sources(
    url=currents_url, storage_options=storage_options
).to_dask()


# url = sources_df.loc[sources_df.name == "efm"].url.iloc[-1]
dataset = catalog.level1.sources(url=efit_url, storage_options=storage_options)
dataset = dataset.to_dask()

time_index = 20

plasma_current = dataset["plasma_current_rz"].dropna(dim="time")
plasma_current = plasma_current.isel(time=time_index)

polodial_flux_rz = dataset["psirz"]
polodial_flux_rz = polodial_flux_rz.dropna(dim="profile_r")
polodial_flux_rz = polodial_flux_rz.isel(time=time_index)

lcfs_R = dataset["lcfs_r"].isel(time=time_index)
lcfs_Z = dataset["lcfs_z"].isel(time=time_index)

# Get the R and Z coordinates of the profiles.
r = dataset["r"]
z = dataset["z"]
R, Z = np.meshgrid(r, z)

# Get the x-point
xpoint_r = dataset["xpoint2_rc"][time_index]
xpoint_z = dataset["xpoint2_zc"][time_index]

# Get the current centre
mag_axis_r = dataset["current_centrd_r"][time_index]
mag_axis_z = dataset["current_centrd_z"][time_index]


# Get the last closed flux surface (LCFS)
lcfs_r = lcfs_R.values
lcfs_r = lcfs_r[~np.isnan(lcfs_r)]
lcfs_z = lcfs_Z.values
lcfs_z = lcfs_z[~np.isnan(lcfs_z)]

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.contourf(R, Z, polodial_flux_rz, cmap="magma", levels=50, label="Polodial Flux")
# ax1.scatter(xpoint_r, xpoint_z, marker="x", color="green", label="X Point")
# ax1.scatter(mag_axis_r, mag_axis_z, marker="o", color="purple", label="Current Centre")
# ax1.plot(lcfs_r, lcfs_z, c="blue", linestyle="--", label="LCFS")
ax1.set_title(f'Polodial Flux for Shot {polodial_flux_rz.attrs["shot_id"]}')
ax1.set_ylabel("Z (m)")
ax1.set_xlabel("R (m)")

ax2.contourf(R, Z, plasma_current, cmap="magma", levels=20, label="Plasma Current")
ax2.scatter(xpoint_r, xpoint_z, marker="x", color="green", label="X Point")
ax2.scatter(mag_axis_r, mag_axis_z, marker="o", color="purple", label="Current Centre")
ax2.plot(lcfs_r, lcfs_z, c="blue", linestyle="--", label="LCFS")
ax2.set_title(f'Plasma Current for Shot {plasma_current.attrs["shot_id"]}')
plt.ylabel("Z (m)")
plt.xlabel("R (m)")
plt.legend()
plt.tight_layout()
