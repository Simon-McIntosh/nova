from dataclasses import dataclass

import numpy as np
import xarray

print("run accessor")


@xarray.register_dataset_accessor("radius")
@dataclass
class Radius:
    data: xarray.Dataset

    def __post_init__(self):
        self.data["radius"] = np.linalg.norm([self.data.x, self.data.y], axis=0)
