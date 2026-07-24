import pathlib

import appdirs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import seaborn as sns
import xarray as xr

import iis

path = pathlib.Path(appdirs.user_data_dir()) / "fair-mast"
path.mkdir(exist_ok=True)


def to_dataset(shot_ids: pd.Series):
    """Return concatanated xarray Dataset for the list of input ids."""
    dataset = []
    for shot_index, shot_id in shot_ids.items():
        shot = iis.Shot(shot_id)
        target = shot.to_dask("equilibrium", "magnetic_flux")
        signal = []
        for group in ["magnetics", "dalpha", "soft_x_rays", "thomson_scattering"]:
            data = shot[group].interp({"time": target.time})
            if "major_radius" in data:
                data = data.interp({"major_radius": target.major_radius})
            for var in data.data_vars:
                data[var].attrs |= {"group": group}
            signal.append(data)
        signal = xr.merge(signal, combine_attrs="drop_conflicts")
        signal["shot_index"] = "time", shot_index * np.ones(target.sizes["time"])
        dataset.append(xr.merge([signal, target], combine_attrs="override"))
    return xr.concat(dataset, "time", join="override", combine_attrs="drop_conflicts")


def flux_contour(magnetic_flux: xr.DataArray, fig, axes, **kwargs) -> PIL.Image.Image:
    """Yield poloidal flux contour images."""
    contours = []
    levels = np.linspace(magnetic_flux.data.min(), magnetic_flux.data.max(), 51)
    for data in magnetic_flux.data:
        # update contour plot
        for contour in contours:
            contour.remove()
        contours = [
            axes.contour(
                magnetic_flux.major_radius,
                magnetic_flux.z,
                data,
                levels=levels,
                **kwargs,
            )
        ]
        contours.append(
            axes.contour(
                -magnetic_flux.major_radius[::-2],
                magnetic_flux.z,
                data[:, ::-2],
                levels=levels,
                **kwargs,
            )
        )
        fig.canvas.draw()
        yield PIL.Image.fromarray(np.array(fig.canvas.buffer_rgba()))


def make_gif(shot_id: int):
    """Make gif of dataset frames."""
    dataset = iis.Shot(shot_id).to_dask("equilibrium", "magnetic_flux")

    fig, axes = plt.subplots(figsize=(2, 2))
    axes.set_aspect("equal")
    axes.set_axis_off()
    fig.tight_layout(pad=0)

    imgs = [
        image
        for image in flux_contour(dataset, fig, axes, colors="gray", linestyles="-")
    ]
    imgs[0].save(
        path / "equilibrium.gif",
        save_all=True,
        append_images=imgs,
        duration=100,
        loop=0,
        minimize_size=True,
    )


def test_train_validate():
    """Save test train and validate datasets for camera data challange."""
    source_ids = np.array([15585, 15212, 15010, 14998, 30410, 30418, 30420])

    rng = np.random.default_rng(7)
    rng.shuffle(source_ids)
    source_ids = pd.Series(source_ids)

    split_ids = {
        "train": source_ids[:5],
        "test": source_ids[5:],
    }

    dataset = {mode: to_dataset(shot_ids) for mode, shot_ids in split_ids.items()}

    # extract solution
    magnetic_flux = dataset["test"].magnetic_flux.data.reshape(
        (dataset["test"].sizes["time"], -1)
    )
    solution = pd.DataFrame(magnetic_flux)
    solution.index.name = "index"
    shot_index = dataset["test"].shot_index.data
    solution["Usage"] = [{5: "Public", 6: "Private"}.get(index) for index in shot_index]
    # delete solution from test file
    dataset["test"] = dataset["test"].drop_vars("magnetic_flux")

    # write to file
    path = pathlib.Path(appdirs.user_data_dir()) / "fair-mast/equilibrium"
    path.mkdir(exist_ok=True)
    dataset["train"].to_netcdf(path / "train.nc")
    dataset["test"].to_netcdf(path / "test.nc")
    solution.to_csv(path / "solution.csv")


if __name__ == "__main__":

    # make_gif(source_ids.iloc[0])
    test_train_validate()

    # dataset = to_dataset(source_ids)
