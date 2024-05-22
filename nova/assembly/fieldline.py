"""Perform post-processing analysis on Fourier perterbed TFC dataset."""

from dataclasses import dataclass

import xarray

from nova.assembly import structural, electromagnetic
from nova.assembly.gap import Gap
import matplotlib.pyplot as plt


@dataclass
class FieldLine:
    """Combine structural and em proxy field line deviation models."""

    def __post_init__(self):
        """Load proxy models."""
        self.structural = structural.Model()
        self.electromagnetic = electromagnetic.Model()

    def predict(self, gap, roll, yaw, ndiv=360):
        """Return field line deviation waveform."""
        radial = self.structural.predict("radial", gap, roll, yaw)
        tangential = self.structural.predict("tangential", gap, roll, yaw)
        self.electromagnetic.predict(radial, tangential, ndiv)
        return self.electromagnetic.fieldline.data

    def peaktopeak(self, gap, ndiv=72):
        """Return predictions for peak to peak fieldline deviations."""
        deviation = self.predict(gap, ndiv)
        return deviation.max(axis=-1) - deviation.min(axis=-1)

    def plot_benchmark(self, simulation: str, title=True):
        """Plot combined structural+EM benchmark."""
        gap = Gap(simulation)
        dataset = self.electromagnetic.load_dataset(simulation)
        self.predict(
            gap.data["gap"], gap.data["roll"], gap.data["yaw"], dataset.sizes["phi"]
        )
        axes = plt.subplots(
            2, 1, sharex=False, sharey=False, gridspec_kw=dict(height_ratios=[1, 2])
        )[1]
        axes[0].bar(gap.data.index, gap.data.gap)
        axes[0].set_ylabel("gap")
        axes[0].set_xticks([])
        self.electromagnetic.plot_deviation(
            axes[1], dataset.phi, dataset.deviation.data
        )
        if title:
            axes[0].set_title(f"Vault+EM benchmark: {simulation}")
        plt.savefig("tmp.png", bbox_inches="tight")

    def plot_peaktopeak(self, simulations=None):
        """Plot peak to peak benchmark."""
        if simulations is None:
            simulations = ["a1", "a2", "c1", "c2", "v3"]
        data = xarray.Dataset(dict(simulations=simulations))
        data["benchmark"] = xarray.DataArray(0.0, coords=data.coords)
        data["model"] = xarray.DataArray(0.0, coords=data.coords)
        for i, simulation in enumerate(simulations):
            dataset = self.electromagnetic.load_dataset(simulation)
            data["benchmark"][i] = dataset.peaktopeak
            model = self.predict(Gap(simulation).gap)
            data["model"][i] = model.max() - model.min()
        plt.figure()
        plt.bar(simulations, data["benchmark"], label="ground truth")
        plt.bar(simulations, data["model"], width=0.5, label="inference")
        plt.despine()
        plt.xlabel("simulation")
        plt.ylabel(r"peak to peak deviation $H$, mm")
        plt.legend()


if __name__ == "__main__":
    feildline = FieldLine()

    feildline.plot_benchmark("v3", title=False)
    # feildline.plot_peaktopeak()
