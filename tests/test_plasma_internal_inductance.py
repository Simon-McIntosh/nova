"""Direct agreement between the legacy and forward inductance instruments."""

from types import SimpleNamespace

import numpy as np

from nova.utilities.importmanager import skip_import

with skip_import("jax"):
    from nova.biot.plasma import Plasma
    from nova.equilibrium.observation import ClippedIntegralMeasure, observe_moments
    from nova.jax.config import configure_dtypes


def test_li_3_matches_forward_observation_on_the_same_state():
    """Both instruments use the state's geometric radius and field energy."""
    configure_dtypes()
    filament_volume = np.array([1.2, 0.8, 1.5])
    poloidal_field = np.array([0.2, 0.35, 0.5])
    plasma_current = 8.0e5
    geometric_radius = 0.83

    state = SimpleNamespace(
        aloc={
            ("ionize", "volume"): filament_volume,
            ("plasma", "ionize"): np.ones(3, dtype=bool),
        },
        grid=SimpleNamespace(bp=poloidal_field),
        i_plasma=plasma_current,
        lcfs=SimpleNamespace(geometric_radius=geometric_radius),
    )
    measure = ClippedIntegralMeasure(
        area=np.ones_like(filament_volume),
        volume=filament_volume,
        radial_volume=geometric_radius * filament_volume,
        cell_current=plasma_current * filament_volume / filament_volume.sum(),
        pressure_volume=np.zeros_like(filament_volume),
        field_volume=poloidal_field**2 * filament_volume,
        masks=None,
    )

    forward = observe_moments(measure, flux_span=np.asarray(1.0))
    legacy = Plasma.li_3.fget(state)

    np.testing.assert_allclose(legacy, forward.internal_inductance, rtol=1e-12)
