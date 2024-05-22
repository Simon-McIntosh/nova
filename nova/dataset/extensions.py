"""Extend xarray accessor method to bypass warning for multiple user imports."""

import os

import xarray

XARRAY_ACCESSOR_WARNING = os.environ.get("XARRAY_ACCESSOR_WARNING", False)


def register_dataset_accessor(name):
    """Extend xarray.register_dataset_accessor to skip AccessorRegistrationWarning."""
    if hasattr(xarray.Dataset, name) and not XARRAY_ACCESSOR_WARNING:

        def decorator(accessor):
            return accessor

        return decorator
    return xarray.register_dataset_accessor(name)
