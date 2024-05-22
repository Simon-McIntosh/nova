"""Manage geometry Dataset accessors."""

import numpy as np
import xarray

from .extensions import register_dataset_accessor


@register_dataset_accessor("radius")
class Radius:
    """Extend xarray.Dataset to include radius attribute."""

    data: xarray.Dataset

    def __init__(self, data: xarray.Dataset):
        self.data = data
        self.data["radius"] = np.linalg.norm([self.data.x, self.data.y], axis=0)
