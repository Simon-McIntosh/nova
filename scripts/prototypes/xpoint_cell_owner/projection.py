"""Constrain a Bell quadratic to the measured saddle position and level."""

import numpy as np


def saddle_constrained_coefficients(coefficients, centre, pitch, saddle, level):
    """Keep the fitted Hessian and set value and gradient at the saddle."""
    x, y = (np.asarray(saddle) - np.asarray(centre)) / pitch
    _, _, _, radial_squared, cross, vertical_squared = coefficients
    radial = -2 * radial_squared * x - cross * y
    vertical = -cross * x - 2 * vertical_squared * y
    constant = level + radial_squared * x * x + cross * x * y + vertical_squared * y * y
    return np.asarray(
        (constant, radial, vertical, radial_squared, cross, vertical_squared)
    )
