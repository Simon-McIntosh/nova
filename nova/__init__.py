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

from importlib import metadata, import_module


try:
    __version__ = metadata.version(__package__ or __name__)
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

submodules = [
    "imas",
]

__all__ = [
    "__version__",
]

__all__.extend(submodules)


try:
    from numba import njit
except ModuleNotFoundError:
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


def __dir__():
    return __all__


def __getattr__(name):
    if name in submodules:
        return import_module(f"nova.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'nova' has no attribute '{name}'")
