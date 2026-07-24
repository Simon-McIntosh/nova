"""Generate plasma volume dataset."""

import pathlib
import warnings

import appdirs
import intake
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import PIL
from tqdm import tqdm
import xarray as xr

warnings.simplefilter(action="ignore", category=FutureWarning)

path = pathlib.Path(appdirs.user_data_dir()) / "fair-mast"
path.mkdir(exist_ok=True)


def load_catalog():
    """Return intake catalog."""
    return intake.open_catalog("https://mastapp.site/intake/catalog.yml")


def load_sources():
    """Return sources dataframe."""
    filename = "sources.json"
    try:
        return pd.read_json(path / filename)
    except FileNotFoundError:
        catalog = load_catalog()
        sources = catalog.index.level1.sources().read()
        sources.to_json(path / filename)
        return sources


def load_summary(campaign="M9"):
    """Return summary dataframe."""
    filename = "summary"
    if campaign is not None:
        filename += f"_{campaign}"
    filepath = (path / filename).with_suffix(".json")
    try:
        return pd.read_json(filepath)
    except FileNotFoundError:
        catalog = load_catalog()
        summary = pd.DataFrame(catalog.index.level1.shots().read())
        if campaign is not None:
            summary = summary.loc[summary.campaign == campaign]
        summary.to_json(filepath)
        return summary


def to_dask(shot_id: int, name: str, sources=None):
    """Return dask dataset."""
    if sources is None:
        sources = load_sources()
    shot_df = sources.loc[sources.shot_id == shot_id]
    url = shot_df.loc[shot_df.name == name].iloc[0].url
    catalog = load_catalog()
    return catalog.level1.sources(url=url).to_dask()


def extract(campaign: str | None = "M9", image_shape=(448, 640)):
    """Extract a set of camera images taken at the time of maximum plasma current."""
    summary = load_summary(campaign)

    dataset = {}
    for _, (shot_id, time_vol_max) in tqdm(
        summary.loc[:, ["shot_id", "cpf_tvol_max"]].iterrows(),
        total=summary.shape[0],
    ):
        try:
            if np.isnan(time_vol_max):
                continue
            frames = to_dask(shot_id, "rbb")
            frame = frames.sel(time=time_vol_max, method="nearest")
            frame.load()
            frame.coords["shot_id"] = shot_id
            key = tuple(frame.shape[1:])
            try:
                dataset[key].append(frame)
            except KeyError:
                dataset[key] = [frame]
        except IndexError:  # no camera data
            pass

    # concatanate datasets
    camera_data = {}
    for key, objs in dataset.items():
        camera_data[key] = xr.concat(objs, "shot_id", combine_attrs="drop_conflicts")
        del camera_data[key].attrs["mds_name"]
        del camera_data[key].attrs["CLASS"]

    sizes = [data.sizes["shot_id"] for data in camera_data.values()]
    plt.bar([str(key) for key in camera_data], sizes)
    return camera_data


def concatanate(filename: str, dropvars=True):
    """Combine camera data with summary information."""
    dataset = xr.open_dataset((path / filename).with_suffix(".nc"))
    if (dataset.sizes["width"] == 640) & (dataset.sizes["height"] == 448):
        dataset.attrs = {
            "IMAGE_SUBCLASS": "IMAGE_INDEXED",
            "IMAGE_VERSION": "1.2",
            "board_temp": 0.0,
            "camera": "",
            "ccd_temp": 0.0,
            "codex": "JP2",
            "depth": 8,
            "description": "Photron bullet camera B",
            "dims": ["time", "height", "width"],
            "file_format": "IPX-1",
            "filter": "",
            "format": "IPX",
            "hbin": 0,
            "height": 448,
            "is_color": 0,
            "left": 193,
            "lens": "",
            "name": "rbb",
            "offset": [0.0, 0.0],
            "orientation": 0,
            "pre_exp": 0.0,
            "quality": "Not Checked",
            "rank": 3,
            "right": 832,
            "signal_type": "Image",
            "source": "rbb",
            "strobe": 0,
            "taps": 0,
            "trigger": -0.10000000149011612,
            "uda_name": "RBB",
            "units": "pixels",
            "vbin": 0,
            "version": -1,
            "width": 640,
        }
    summary = load_summary()
    index = np.isin(summary.shot_id, dataset.shot_id)
    dataset["plasma_volume"] = "shot_id", summary.loc[index, "cpf_vol_max"]
    if dropvars:
        return dataset.drop_vars(["time", "shot_id"])
    # dataset.to_netcdf(path / "plasma_volume.nc")
    return dataset


def test_train_validate(filename: str, test_size=0.3):
    """Save test train and validate datasets for camera data challenge."""
    dataset = concatanate("rbb_camera_448_640", dropvars=True)
    dataset = dataset.rename({"data": "frame"})

    shot_index = np.arange(dataset.sizes["shot_id"], dtype=int)
    rng = np.random.default_rng(7)
    rng.shuffle(shot_index)
    test_split = int(np.floor(test_size * dataset.sizes["shot_id"]))

    train = dataset.isel(shot_id=shot_index[test_split:])
    test = dataset.isel(shot_id=shot_index[:test_split])
    solution = test.plasma_volume.to_pandas().to_frame()
    rng.random(len(solution))
    solution["Usage"] = np.where(rng.random(len(solution)) < 0.5, "Public", "Private")

    test = test.drop_vars("plasma_volume")

    # write to file
    (path / "plasma_volume").mkdir(exist_ok=True)
    train.to_netcdf(path / "plasma_volume/train.nc")
    test.to_netcdf(path / "plasma_volume/test.nc")
    solution.to_csv(path / "plasma_volume/solution.csv")

    return train, test, solution


def make_gif(filename: str):
    """Make gif of dataset frames."""
    dataset = concatanate(filename, dropvars=False)

    imgs = [PIL.Image.fromarray(img[::2, ::2]) for img in dataset.data.values]
    imgs[0].save(
        path / "proton_camera.gif",
        save_all=True,
        append_images=imgs[1::4],
        duration=50,
        loop=0,
        minimize_size=True,
    )


if __name__ == "__main__":

    """
    camera_data = extract("M9")
    for key, data in camera_data.items():
        print(f"writing {key} camera data to file")
        data.to_netcdf(path / f"visable_camera_{key[0]}_{key[1]}.nc")
    """

    train, test, solution = test_train_validate("rbb_camera_448_640")

    dataset = concatanate("rbb_camera_448_640", dropvars=False)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(train, x="shot_id", y="plasma_volume")

    # make_gif("rbb_camera_448_640")

    # PIL.Image.fromarray(dataset.data[400].values).save(path / "proton_camera.png")
