"""Split the prescribed toroidal source on saved physical flux states.

Flux functions use COCOS 17 total flux: each density term contains -2*pi.
The shape verdict compares a flux-coordinate centre with an inner shoulder;
map-current contrasts use the separately declared geometric sampling rule.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import mu_0


def split_current(radius, psi_norm, closure, amplitude=1.0, support=None):
    """Evaluate the actual closure and retain both independently signed terms."""
    radius, coordinate = np.broadcast_arrays(radius, psi_norm)
    pressure = np.asarray(closure.pressure_gradient(radius, coordinate))
    diamagnetic = np.asarray(closure.ff_prime(coordinate))
    pressure_term = -2 * np.pi * radius * pressure
    field_term = -2 * np.pi * diamagnetic / (mu_0 * radius)
    density = np.asarray(closure.current_density(radius, coordinate))
    np.testing.assert_allclose(
        pressure_term + field_term, density, rtol=2e-13, atol=1e-8
    )
    selected = (
        np.ones_like(radius, dtype=bool)
        if support is None
        else np.broadcast_to(support, radius.shape)
    )
    return dict(
        psi_norm=coordinate,
        p_prime=pressure,
        ff_prime=diamagnetic,
        pressure_term=pressure_term,
        field_term=field_term,
        unscaled_density=density,
        density=np.where(selected, amplitude * density, 0.0),
        support=selected,
    )


def profile_shape(radius, closure):
    """Compare the axis value to the median on normalized flux 0.15–0.35."""
    coordinate = np.r_[0.0, np.linspace(0.15, 0.35, 41)]
    terms = split_current(np.full_like(coordinate, radius), coordinate, closure)
    centre = float(terms["unscaled_density"][0])
    shoulder = float(np.median(terms["unscaled_density"][1:]))
    if shoulder <= 0 or not np.isfinite(centre + shoulder):
        raise ValueError("shape requires a finite positive shoulder current")
    return dict(
        centre=centre,
        shoulder=shoulder,
        ratio=centre / shoulder,
        verdict="peaked" if centre >= shoulder else "hollow",
    )


def arrays_to_lists(record):
    """Return portable numeric arrays without discarding signed source terms."""
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in record.items()
    }


if __name__ == "__main__":
    from recover import main

    main()
