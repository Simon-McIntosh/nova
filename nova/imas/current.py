"""Manage access to current waveforms stored in pf_active and pf_passive."""

from dataclasses import dataclass

import numpy as np
from scipy import signal
from tqdm import tqdm
import xarray

from nova.graphics.plot import Plot
from nova.imas.database import Database
from nova.imas.pf_active import PF_Active
from nova.imas.pf_passive import PF_Passive


@dataclass
class Current(Plot, Database):
    """Analyze active and passive current waveforms."""

    def __post_init__(self):
        """Load current waveforms."""
        super().__post_init__()
        self.pf_active = PF_Active(**self.ids_attrs).data
        self.pf_passive = PF_Passive(**self.ids_attrs).data

        self.time = np.linspace(
            self.pf_active.time[0],
            self.pf_active.time[-1],
            2 * len(self.pf_active.time),
        )
        self.data = xarray.concat(
            [
                self.pf_active.current.interp(dict(time=self.time)).rename(
                    dict(coil_name="index")
                ),
                self.pf_passive.current.interp(dict(time=self.time)).rename(
                    dict(loop_name="index")
                ),
            ],
            dim="index",
        )

    def plot_active(self):
        """Plot all active coil currents."""
        active = self.mpl.collections.LineCollection(
            np.stack(
                np.broadcast_arrays(
                    self.pf_active.time.data[np.newaxis, :],
                    1e-3 * self.pf_active.current.T,
                ),
                axis=-1,
            ),
            color="gray",
            alpha=0.5,
            label="active",
            zorder=-9,
        )
        self.axes.add_collection(active)

    def plot_passive(self):
        """Plot all passive coil currents."""
        passive = self.mpl.collections.LineCollection(
            np.stack(
                np.broadcast_arrays(
                    self.pf_passive.time.data[np.newaxis, :],
                    1e-3 * self.pf_passive.current.T,
                ),
                axis=-1,
            ),
            color="C2",
            alpha=0.5,
            label="passive",
            zorder=-10,
        )
        self.axes.add_collection(passive)

    def plot(self, axes=None):
        """Plot waveforms."""
        self.set_axes(None, "1d")
        # self.plot_active()
        # self.plot_passive()
        self.axes.plot(
            self.pf_active.time,
            1e-3 * self.pf_active.current.sel(coil_name="VS3"),
            label="VS3",
            zorder=10,
        )
        self.axes.plot(
            self.pf_passive.time,
            1e-3 * self.pf_passive.current.sel(loop_name="TRI_SUPP"),
            label="TRI_SUPP",
        )
        self.axes.set_xlabel("time s")
        self.axes.set_ylabel(r"current rate kAs$^{-1}$")
        self.plt.tight_layout()
        self.legend(ncol=3)
        self.plt.title(f"{self.pulse}, {self.run}")
        self.axes.set_xlim([0, 30])

    def correlate(self, in1: str, in2: str, plot=False):
        """Return maximum absolute cross-correlation coefficent."""
        in1_data = self.data.sel(index=in1).differentiate("time")
        in2_data = self.data.sel(index=in2).differentiate("time")
        correlation = signal.correlate(1e-3 * in1_data, 1e-3 * in2_data, mode="same")
        correlation /= len(in1_data)
        if plot:
            self.set_axes(None, "1d")
            self.axes.plot(self.time - (self.time[-1] - self.time[0]) / 2, correlation)
        return correlation[np.argmax(abs(correlation))]

    def correlation_matrix(self):
        """Build correlation matrix."""
        length = current.data.shape[1]
        matrix = np.zeros((length, length))
        for i in tqdm(range(length)):
            for j in range(length):
                matrix[i, j] = self.correlate(
                    self.data.index[i].data, self.data.index[j].data
                )
        matrix /= np.max(matrix)
        self.set_axes(None, "2d")
        self.axes.matshow(matrix)
        return matrix


if __name__ == "__main__":
    pulse, run = 105027, 2
    # pulse, run = 105028, 1
    # pulse, run = 105027, 2

    current = Current(pulse, run)
    current.plot()
    current.correlate("VS3", "TRI_SUPP", True)
    # current.correleate('VES_1', 'VS3')

    # matrix = current.correlation_matrix()

    """
    plt.plot(matrix[11], label='VS3')
    plt.plot(matrix[12], label='TRI_SUPP')
    plt.legend()

    """

    """
    dina = np.array(
        [[105013, 105014, 105015, 105016, 105017, 105018, 105019, 105020,
          105021, 105022, 105023, 105024, 105025, 105026, 105027, 105028,
          105029, 105030, 105031, 105032, 105033, 135001, 135011, 135012,
          135013, 135014],
         [     1,      1,      1,      1,      1,      1,      1,      1,
               1,      1,      1,      1,      1,      1,      2,      1,
               1,      1,      1,      1,      1,      7,      7,      2,
               2,      1]])

    for i, (pulse, run) in tqdm(enumerate(zip(*dina))):
        current = Current(pulse, run)

        try:
            corr = current.correlate('VS3', 'TRI_SUPP')
            current.plot()
            current.plt.title(f'{pulse}, {run}, ({corr:1.1f})')
        except KeyError:
            pass
    """
