# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:59:05 2024

@author: mcintos
"""


'''
@dataclass
class Ensemble:

    shots: list[int]

    def target_times(self, group: str, path: str) -> dict[int, np.ndarray]:
        """Return target time vectors for shots."""
        return (Shot(shot_id).to_dask(group, path).time.data for shot_id in self.shots)

    # def to_target(self, str, path: str, time: np.ndarray | None = None):


Shot(30420).to_pandas("equilibrium", "magnetic_flux")

ensemble = Ensemble([30420, 30421]).target_times("equilibrium", "magnetic_flux")

np.concatenate(
    [
        Shot(shot_id).to_target("equilibrium", "magnetic_flux")
        for shot_id in [30420, 30421]
    ],
    axis=0,
)


@dataclass
class FluxMap(Shot):
    """Extract flux map target from shot."""

    @property
    def magnetic_flux(self):
        """Return equilibrium ML target."""
        return self["equilibrium"].magnetic_flux.dropna("time")

    @property
    def shape(self):
        """Return flux map shape."""
        return self.magnetic_flux.shape

    @property
    def target(self):
        """Return flux map target."""
        return self.magnetic_flux.data.reshape(self.magnetic_flux.sizes["time"], -1)


@dataclass
class Equilibrium(Shot):

    signals: list[str] = field(
        default_factory=lambda: [
            # "center_column",
            # "coil_currents",
            # "coil_voltages",
            "flux_loops",
            # "outer_discrete",
            "saddle_coils",
        ]
    )

    @property
    def magnetic_flux(self):
        """Return equilibrium ML target."""
        return self["equilibrium"].magnetic_flux.dropna("time")

    @property
    def time(self):
        """Return time vector."""
        return self.magnetic_flux.time

    def signal(self):
        """Return equilibrium ML signal."""
        return pd.concat(
            [
                self["magnetics"][group].interp({"time": self.time}).T.to_pandas()
                for group in self.signals
            ],
            axis=1,
        )


equilibrium = Equilibrium(30420)

fluxmap = FluxMap(30420)


magnetics = xr.open_zarr(store, group="magnetics")
equilibrium = xr.open_zarr(store, group="equilibrium")

magnetic_flux = equilibrium.magnetic_flux.dropna("time")
grid_shape = magnetic_flux[0].shape


signal =
target = magnetic_flux.data.reshape(magnetic_flux.sizes["time"], -1)
'''
