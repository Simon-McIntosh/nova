"""Manage ensemble plots of fourier gaps structural model dataset."""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import xarray

from nova.assembly.structural import Data, Fourier
import matplotlib.pyplot as plt


@dataclass
class Compose(Data):
    """Perform Fourier analysis on TFC deformations."""

    origin: tuple[int] = (0, 0, 0)
    wavenumber: list[int] = field(default_factory=lambda: range(10))

    prefix: ClassVar[str] = "k"
    ncoil: ClassVar[int] = 18
    cluster: ClassVar[int] = 1

    def build(self):
        """Build ensemble of ansys fourier component simulations."""
        mode = []
        for wavenumber in self.wavenumber:
            fourier = Fourier(f"{self.prefix}{wavenumber}")
            mode.append(fourier.data.copy(deep=True))
        self.data = xarray.concat(mode, "wavenumber", combine_attrs="drop_conflicts")
        self.data["wavenumber"] = self.data.mode.data
        self.store()

    def plot_wave(self, wavenumber: int, scenario="TFonly"):
        """Plot fft components."""
        index = dict(wavenumber=wavenumber, scenario=scenario)
        plt.figure()
        plt.bar(
            self.data.index.data,
            np.cos(wavenumber * self.data.points.sel(**index, dimensions="toroidal")),
            width=0.7,
            label="toroidal placment error 1.00",
            color="C3",
        )

        radial_amplitude = self.data.amplitude.sel(
            **index, mode=wavenumber, response="radial"
        ).data
        radial_phase = self.data.phase.sel(
            **index, mode=wavenumber, response="radial"
        ).data
        plt.bar(
            self.data.index.data,
            self.data.delta.sel(**index, dimensions="radial"),
            width=0.5,
            color="C0",
            label=f"radial misalignment {radial_amplitude:1.2f}",
        )
        phi = np.linspace(0, 2 * np.pi, 150)
        plt.plot(
            phi * 9 / np.pi + 1,
            radial_amplitude * np.cos(wavenumber * phi + radial_phase),
            color="gray",
        )
        plt.despine()
        plt.xlabel("coil index")
        plt.ylabel("misalignment, mm")
        plt.xticks(range(1, self.ncoil + 1))
        plt.legend(ncol=2, loc="upper center", bbox_to_anchor=[0.5, 1.14])
        plt.title(f"wavenumber={wavenumber}\n\n")

    def plot_amplitude(self):
        """Plot radial amplitude magnification."""
        amplitude = self.diag("amplitude")
        plt.figure()
        plt.bar(self.data.wavenumber, amplitude, label="ANSYS", width=0.75)
        plt.despine()
        plt.xticks(self.data.wavenumber.values)
        plt.xlabel("wavenumber")
        plt.ylabel("radial amplification factor " r"$\frac{\Delta r}{r \Delta \phi}$")
        plt.legend(frameon=False)
        plt.yscale("log")

    def diag(self, attr: str, scenario="TFonly", response="radial"):
        """Return attribute's diagonal components."""
        return np.diag(self.data[attr].sel(scenario=scenario, response=response).data)

    def plot_argand(self, scenario="TFonly", response="radial"):
        """Plot complex transform."""
        plt.figure()
        amplitude = self.diag("amplitude", scenario, response)
        phase = self.diag("phase", scenario, response).copy()
        coef = amplitude * np.exp(1j * phase)

        for i, mode in enumerate(coef):
            plt.plot(mode.real, mode.imag, "o")
            plt.text(mode.real, mode.imag, i, va="bottom", ha="center")
        phi = np.linspace(0, 2 * np.pi)
        plt.plot(np.cos(phi), np.sin(phi), "--", color="gray")
        plt.plot([-1, 1], [0, 0], "-.", color="gray")
        plt.plot([0, 0], [-1, 1], "-.", color="gray")
        plt.axis("equal")
        plt.axis("off")

    def plot_fit(self):
        """Fit model to observations."""
        H = np.array(
            [
                0.396,
                0.759,
                1.6871,
                0.934,
                0.6451,
                0.5373,
                0.4835,
                0.4085,
                0.6682,
                0.4012,
            ]
        )
        weights = np.ones(10)
        weights[1] = 0  # exclude n=1 mode
        amplitude = self.diag("amplitude")
        matrix = np.array([np.ones(10), amplitude]).T

        coef = np.linalg.lstsq(matrix * weights[:, np.newaxis], H * weights)[0]

        _amplitude = np.linspace(amplitude.min(), amplitude.max(), 51)
        _matrix = np.array([np.ones_like(_amplitude), _amplitude]).T

        plt.figure()
        plt.plot(amplitude, H, "o")
        plt.plot(_amplitude, _matrix @ coef, "-", color="gray")
        plt.despine()
        plt.xlabel(r"amplitude of radial mode, $\Delta r$ mm")
        plt.ylabel(r"peak to peak misalignment, $H$ mm")
        plt.title(rf"$H={{{coef[0]:1.2f}}}+{{{coef[1]:1.2f}}}\Delta r$")


if __name__ == "__main__":
    compose = Compose()

    compose.plot_wave(1)
    compose.plot_argand()
    compose.plot_amplitude()
    compose.plot_fit()
