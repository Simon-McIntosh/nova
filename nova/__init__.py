"""
Nova: An electromagnetic analysis package for Python.

Subpackages
-----------
Using any of these subpackages requires an explicit import. For example,
``import nova.imas``.

::

 imas            --- Methods for interfacing with the IMAS data model.


Public API in the main Nova namespace
--------------------------------------

::

 __version__       --- Nova version string

"""

__all__ = [
    "geometry",
]

import importlib
import os

from .dataset import geometry

try:
    __version__ = importlib.metadata.version("nova-stella")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
__all__.extend("__version__")

submodules = [
    "imas",
]
__all__.extend(submodules)

os.environ["NUMBA_THREADING_LAYER"] = "omp"
os.environ["IMAS_HOME"] = os.environ.get(
    "IMAS_HOME", os.path.join(os.path.expanduser("~"), "imas")
)
os.environ["HDF5_USE_FILE_LOCKING"] = os.environ.get("HDF5_USE_FILE_LOCKING", "False")

try:
    from numba import njit, prange
except (ModuleNotFoundError, ImportError):
    from functools import wraps

    def njit(**jit_kwargs):
        """Replicate njit decorator, accept and ignore decorator kwargs."""

        def decorator(method):
            """Return method evaluated with passed args and kwargs."""

            @wraps(method)
            def wrapper(*args, **kwargs):
                return method(*args, **kwargs)

            return wrapper

        return decorator

    prange = range

try:
    import vedo

    vedo.settings.default_backend = "vtk"
except ModuleNotFoundError:
    pass


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return importlib.import_module(f"nova.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'nova' has no attribute '{name}'")
