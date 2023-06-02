"""
Manage cupy / numpy import for CPU / GPU agnostic code.

Code adapted from:

License: Apache 2.0
Carl Kadie
https://github.com/CarlKCarlK/gpuoptional/

"""

import logging
import os
from types import ModuleType

import numpy as np

_warn = True


def array_module(module=None):
    """
    Return array module.

    Parameters
    ----------
    module: Union[ModuleType, str]
        Requested module type. If None query XPU environment, defaults to cupy.

    Returns
    -------
    module: ModuleType
        cupy or numpy.

    """
    module = module or os.environ.get("XPU", "numpy")

    if isinstance(module, ModuleType):
        return module

    if module == "numpy" or module == "np":
        return np

    if module == "cupy" or module == "cp":
        try:
            import cupy as cp

            return cp
        except ModuleNotFoundError as e:
            global _warn
            if _warn:
                logging.warning(f"cupy not installed, using numpy. ({e})")
                _warn = False
            return np

    raise NotImplementedError(f'Undefined ARRAY_MODULE "{module}".')


xp = array_module()


def asnumpy(array):
    """
    Return numpy array.

    Given an array created with any array module, return the equivalent
    numpy array. (Returns a numpy array unchanged.)
    >>> from pysnptools.util import asnumpy, array_module
    >>> xp = array_module('cupy')
    >>> zeros_xp = xp.zeros((3)) # will be cupy if available
    >>> zeros_np = asnumpy(zeros_xp) # will be numpy
    >>> zeros_np
    array([0., 0., 0.])

    """
    if isinstance(array, np.ndarray):
        return array
    if isinstance(array, xp.ndarray):
        return array.get()
    return array
