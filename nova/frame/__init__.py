__all__ = [
    "pandas",
    "xarray",
    ]

import lazy_loader as lazy

pandas = lazy.load('pandas')
xarray = lazy.load('xarray')
