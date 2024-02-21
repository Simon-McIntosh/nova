import numpy as np
import xarray


@xarray.register_dataset_accessor("radius")
class Radius:
    data: xarray.Dataset

    def __init__(self, data: xarray.Dataset):
        print("init radius")
        self.data = data
        self.data["radius"] = np.linalg.norm([self.data.x, self.data.y], axis=0)
