from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import xarray

from typing import NamedTuple


class MatrixData(NamedTuple):
    """EM coupling data for jax backed computations."""

    source_target: jnp.ndarray
    plasma_target: jnp.ndarray | None = None
    source_plasma: jnp.ndarray | None = None
    plasma_plasma: jnp.ndarray | None = None


@dataclass
class Matrix:
    """Generate EM coupling matricies."""

    data: xarray.Dataset = field(repr=False)

    def __getitem__(self, attr: str):
        """Retrun jax matrix dataset."""
        dataset = {"source_target": jnp.array(self.data[attr])}
        if source_plasma := self.data.source_plasma_index != -1:
            dataset["plasma_target"] = jnp.array(self.data[f"{attr}_"])
        if target_plasma := self.data.target_plasma_index != -1:
            dataset["source_plasma"] = jnp.array(self.data[f"_{attr}"])
        if source_plasma and target_plasma:
            dataset["plasma_plasma"] = jnp.array(self.data[f"_{attr}_"])
        return (
            MatrixData(**dataset),
            (self.data.source_plasma_index, self.data.target_plasma_index),
        )


@partial(jax.jit, static_argnums=1)
def update_plasma_turns(matrix, index, plasma_nturn):
    """Update plasma turns."""
    source_plasma_index, target_plasma_index = index
    source_target = matrix.source_target
    if update_source := source_plasma_index != -1:
        source_target = source_target.at[:, source_plasma_index].set(
            matrix.plasma_target @ plasma_nturn
        )
    if update_target := target_plasma_index != -1:
        source_target = source_target.at[target_plasma_index, :].set(
            plasma_nturn @ matrix.source_plasma
        )
    if update_source and update_target:
        source_target = source_target.at[target_plasma_index, source_plasma_index].set(
            plasma_nturn @ matrix.plasma_plasma @ plasma_nturn
        )
    return source_target


if __name__ == "__main__":

    plasmagrid = xarray.open_dataset("plasmagrid.nc")


'''

    aloc: ArrayLocIndexer
    saloc: ArrayLocIndexer
    classname: str
    index: np.ndarray
    dataset: InitVar[xarray.Dataset]

def __post_init__(self, dataset):
    """Extract matrix, plasma_matrices, and plasma indicies from dataset."""
    attr = list(dataset.data_vars)[0]
    self.matrix = dataset[attr].data
    self.source_plasma_index = dataset.attrs["source_plasma_index"]
    self.target_plasma_index = dataset.attrs["target_plasma_index"]
    if source_plasma := self.source_plasma_index != -1:
        self.matrix_ = dataset[f"{attr}_"].data
    if target_plasma := self.target_plasma_index != -1:
        self._matrix = dataset[f"_{attr}"].data
    if source_plasma and target_plasma:
        self._matrix_ = dataset[f"_{attr}_"].data
'''
