"""Run Monte Carlo simulations for candidate vault assemblies."""
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from functools import cached_property
from time import time
from typing import ClassVar, Union

import numpy as np
import xarray
import xxhash

from nova.assembly import structural, electromagnetic, overlap
from nova.assembly.gap import WedgeGap
from nova.assembly.model import Dataset
import matplotlib.pyplot as plt


@dataclass
class TrialAttrs:
    """Manage trial attributes."""

    samples: int = 100_000
    component: list[str] = field(default_factory=list)
    theta: list[float] = field(default_factory=list)
    pdf: list[str] = field(default_factory=list)
    nominal_gap: float = 2.0
    sead: int = 2025

    ncoil: ClassVar[int] = 18

    @cached_property
    def field_names(self):
        """Return list of field names."""
        return [attr.name for attr in fields(TrialAttrs)]

    @property
    def attrs(self):
        """Return trial attrs."""
        attrs = {}
        for attr in self.field_names:
            value = getattr(self, attr)
            if not isinstance(value, list):
                attrs[attr] = value
        return attrs


@dataclass
class Trial(Dataset, TrialAttrs):
    """Run stastistical analysis on trial assemblies."""

    filename: str = "trial"
    xxh32: xxhash.xxh32 = field(repr=False, init=False, default_factory=xxhash.xxh32)

    def __post_init__(self):
        """Set dataset group for netCDF file load/store."""
        self.group = self.group_name
        self.rng = np.random.default_rng(self.sead)
        super().__post_init__()

    @property
    def group_name(self):
        """Return group name as xxh32 hex hash."""
        self.xxh32.reset()
        self.xxh32.update(np.array(list(self.attrs.values()) + self.theta + self.pdf))
        return self.xxh32.hexdigest()

    def normal(self, variance: float):
        """Return sample with normal distribution."""
        scale = np.sqrt(variance)
        return self.rng.normal(scale=scale, size=(self.samples, self.ncoil))

    def uniform(self, bound: float):
        """Return sample with uniform distribution."""
        return self.rng.uniform(-bound, bound, size=(self.samples, self.ncoil))

    def build_signal(self):
        """Build input distributions."""
        self.data = xarray.Dataset(attrs=self.attrs)
        self.data["sample"] = range(self.samples)
        self.data["index"] = range(self.ncoil)
        for component in self.component:
            self.data[component] = xarray.DataArray(0.0, self.data.coords)
        self.data["component"] = self.component
        self.data["signal"] = ["radial", "tangential"]
        self.data["coordinate"] = ["x", "y"]
        self.data["theta"] = "component", self.theta
        self.data["pdf"] = "component", self.pdf
        for i, component in enumerate(self.component):
            theta = self.theta[i]
            pdf = self.pdf[i]
            self.data[component] = ("sample", "index"), getattr(self, pdf)(theta)

    def build_positive_gap(self, nmax=20, eps=1e-3):
        """Built gap waveform via iterative loop."""
        self.build_gap()
        for i in range(nmax):
            gap = self.data.gap.sum(axis=-1).data + self.nominal_gap
            sample_index = (gap < -eps).any(axis=1)
            if sample_index.sum() == 0:
                print(f"positive gap iteration converged {i}")
                return
            offset = gap[sample_index]
            offset[offset >= 0] = 0
            self.data.tangential[sample_index] += offset
            self.build_gap()
        raise ValueError(
            f"gap itteration failure at iteration {nmax} "
            "negitive samples "
            f"{100*sample_index.sum()/len(gap):1.0f}%"
        )

    def build_gap(self):
        """Build vault gap from radial and toroidal waveforms."""
        self.data["gap"] = ("sample", "index", "signal"), np.zeros(
            (self.data.sizes["sample"], self.ncoil, self.data.sizes["signal"])
        )
        self.data.gap[..., 0] = np.pi / self.ncoil * self.data["radial"]
        self.data.gap[:, :-1, 0] += np.pi / self.ncoil * self.data["radial"][:, 1:].data
        self.data.gap[:, -1, 0] += np.pi / self.ncoil * self.data["radial"][:, 0].data
        self.data.gap[..., 1] = -self.data["tangential"]
        self.data.gap[:, :-1, 1] += self.data["tangential"][:, 1:].data
        self.data.gap[:, -1, 1] += self.data["tangential"][:, 0].data

    @contextmanager
    def timer(self):
        """Time build."""
        start_time = time()
        yield
        print(f"build time {time() - start_time:1.0f}s")

    def pdf_text(self, wall=False, fancy=False):
        """Return pdf text label."""
        text = ""
        for i, component in enumerate(self.component):
            if component == "wall" and not wall:
                continue
            if fancy:
                attr = component.split("_")[-1]
                if attr in ["radial", "tangential"]:
                    attr = attr[0]
                    if attr == "t":
                        text += r"$r$"
                        attr = r"\phi"
                    text += rf'$\Delta {attr}_{{{component.split("_")[0]}}}$'
                else:
                    text += component.split("_")[-1]
            else:
                text += component
            theta = self.data.theta[i].data
            if self.data.pdf[i] == "normal":
                pdf = rf"$\mathcal{{N}}\,(0, {theta:1.1f})$"
            elif self.data.pdf[i] == "uniform":
                pdf = rf"$\mathcal{{U}}\,(\pm{theta:1.1f})$"
            text += ": " + pdf
            text += "\n"
        text += "\n"
        text += f"samples: {self.samples:,}"
        plt.text(
            0.95,
            0.95,
            text,
            fontsize="x-small",
            transform=plt.gca().transAxes,
            ha="right",
            va="top",
            bbox=dict(facecolor="w", boxstyle="round, pad=0.5", linewidth=0.5),
        )

    def label_quantile(self, data, label: str, quantile=0.99, height=0.1, color="gray"):
        """Label quantile."""
        ylim = plt.gca().get_ylim()
        yline = ylim[0] + np.array([0, height * (ylim[1] - ylim[0])])
        quantile = np.quantile(data, quantile)
        plt.plot(quantile * np.ones(2), yline, "-", color="k", alpha=0.75)
        text = rf"q(0.99): ${label}={quantile:1.2f}$"
        plt.text(
            quantile,
            yline[1],
            text,
            ha="left",
            va="bottom",
            fontsize="small",
            color=color,
            bbox=dict(facecolor="w", edgecolor=color),
        )

    def plot_pdf(self, bins=51):
        """Plot pdf."""
        pdf, edges = np.histogram(self.data.peaktopeak, bins, density=True)
        plt.plot((edges[:-1] + edges[1:]) / 2, pdf)

    def sample(self, quantile, offset=True):
        """Return sample index closest to quantile."""
        label = "peaktopeak"
        if offset:
            label += "_offset"
        peaktopeak = np.quantile(self.data[label], quantile)
        return np.argmin((self.data[label].data - peaktopeak) ** 2)


@dataclass
class Vault(Trial):
    """Run vault assembly Monte Carlo trials."""

    filename: str = "vault_trial"
    component: list[str] = field(
        default_factory=lambda: [
            "radial",
            "tangential",
            "roll_length",
            "yaw_length",
            "radial_ccl",
            "tangential_ccl",
            "radial_wall",
        ]
    )
    theta: list[float] = field(default_factory=lambda: [1.5, 1.5, 3, 3, 2, 2, 5])
    pdf: list[str] = field(
        default_factory=lambda: [
            "uniform",
            "uniform",
            "uniform",
            "uniform",
            "normal",
            "normal",
            "uniform",
        ]
    )
    modes: int = 3
    energize: Union[int, bool] = True
    wall: bool = True

    def __post_init__(self):
        """Initialize model instances."""
        self.energize = int(self.energize)
        self.wall = int(self.wall)
        self.field_names += ["modes", "energize", "wall"]
        self.structural_model = structural.Model()
        self.electromagnetic_model = electromagnetic.Model()
        super().__post_init__()

    def build(self):
        """Build Monte Carlo dataset."""
        with self.timer():
            self.build_signal()
            self.build_positive_gap()
            self.predict_structure()
            self.predict_electromagnetic()
            if self.wall:
                self.predict_wall()
        return self.store()

    def predict_structure(self):
        """Run structural simulation."""
        self.data["structural"] = ("sample", "index", "signal"), np.zeros(
            (self.samples, self.ncoil, self.data.sizes["signal"])
        )
        if self.energize:
            gap = self.data.gap.sum(axis=-1)
            roll = self.data["roll_length"] - self.data["tangential"]
            yaw = self.data["yaw_length"] - self.data["tangential"]
            for i, signal in enumerate(self.data.signal.values):
                self.data["structural"][..., i] = self.structural_model.predict(
                    signal, gap, roll, yaw
                )

    def predict_electromagnetic(self):
        """Run electromagnetic simulation."""
        self.data["electromagnetic"] = self.data.structural.copy(deep=True)
        self.data.electromagnetic[..., 0] += self.data.radial
        self.data.electromagnetic[..., 1] += self.data.tangential
        self.data.electromagnetic[..., 0] += self.data.radial_ccl
        self.data.electromagnetic[..., 1] += self.data.tangential_ccl
        self.electromagnetic_model.predict(
            self.data.electromagnetic[..., 0], self.data.electromagnetic[..., 1]
        )
        self.data["peaktopeak"] = "sample", self.electromagnetic_model.peaktopeak(
            modes=self.modes
        )
        self.data["offset"] = ("sample", "coordinate"), np.zeros(
            (self.data.sizes["sample"], 2)
        )
        offset = self.electromagnetic_model.axis_offset
        self.data["offset"][..., 0] = offset.real
        self.data["offset"][..., 1] = -offset.imag
        self.data[
            "peaktopeak_offset"
        ] = "sample", self.electromagnetic_model.peaktopeak(
            modes=self.modes, axis_offset=True
        )

    def predict_wall(self):
        """Predict combined wall-fieldline deviations."""
        ndiv = self.electromagnetic_model.fieldline.shape[1]
        wall_hat = np.fft.rfft(self.data.radial_wall)
        firstwall = np.fft.irfft(wall_hat, ndiv) * ndiv / self.ncoil
        wall_hat[..., 1] += self.electromagnetic_model.axis_offset * (self.ncoil // 2)
        offset_firstwall = np.fft.irfft(wall_hat, ndiv) * ndiv / self.ncoil
        deviation = self.electromagnetic_model.fieldline.data - firstwall.data
        self.data["peaktopeak"] = "sample", self.electromagnetic_model.peaktopeak(
            deviation, modes=self.modes
        )
        offset_deviation = (
            self.electromagnetic_model.fieldline.data - offset_firstwall.data
        )
        self.data[
            "peaktopeak_offset"
        ] = "sample", self.electromagnetic_model.peaktopeak(
            offset_deviation, modes=self.modes
        )

    def plot(self, offset=True):
        """Plot peak to peak distribution."""
        plt.figure()
        plt.hist(
            self.data.peaktopeak,
            bins=51,
            density=True,
            rwidth=0.8,
            label="machine axis",
            color="C1",
        )
        if offset:
            plt.hist(
                self.data.peaktopeak_offset,
                bins=51,
                density=True,
                rwidth=0.8,
                alpha=0.85,
                color="C2",
                label="magnetic axis",
            )
            plt.legend(
                loc="center", bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize="small"
            )
            self.label_quantile(
                self.data.peaktopeak_offset, "H", color="C2", height=0.15
            )
        self.label_quantile(self.data.peaktopeak, "H", color="C1", height=0.04)
        plt.despine()
        axes = plt.gca()
        axes.set_yticks([])
        plt.xlabel(r"peak to peak deviation $H$, mm")
        plt.ylabel(r"$P(H)$")
        self.pdf_text()

    def plot_offset(self):
        """Plot pdf of field line axis offset."""
        offset = np.linalg.norm(self.data.offset, axis=-1)
        plt.figure()
        plt.hist(offset, bins=51, density=True, rwidth=0.8)
        plt.despine()
        axes = plt.gca()
        axes.set_yticks([])
        plt.xlabel(r"magnetic axis offset $\zeta$, mm")
        plt.ylabel(r"$P(\zeta)$")

        self.label_quantile(offset, r"\zeta")
        self.pdf_text()

    def plot_sample(self, quantile=0.99, offset=True, plot_deviation=False):
        """Plot waveforms from single sample."""
        sample = self.sample(quantile, offset)
        axes = plt.subplots(
            3, 1, sharex=False, sharey=False, gridspec_kw=dict(height_ratios=[1, 1, 2])
        )[1]
        width = 0.8

        signal_width = width / self.data.sizes["component"]
        for i, component in enumerate(self.data.component.values):
            signal = self.data[component]
            bar_offset = (i + 0.5) * signal_width - width / 2
            axes[0].bar(
                self.data.index + bar_offset,
                signal[sample],
                color=f"C{i+1}",
                width=signal_width,
                label=component,
            )
            # axes[0].plot(self.data.index,
            #             self.theta[0] * (-1)**i *
            #             np.ones_like(self.data.index), 'C7--', alpha=0.5,
            #             lw=1.5)
        axes[0].set_ylabel("vault")
        axes[0].legend(fontsize="xx-small", bbox_to_anchor=(1, 1))
        axes[0].set_xticks([])

        # signal_width = width / 3
        # for i, signal in ['gap', 'roll', 'yaw']
        axes[1].bar(
            self.data.index,
            self.data.gap[sample].sum(axis=-1) + self.data.nominal_gap,
            width=width,
            color="C0",
        )
        axes[1].set_ylabel("gap")
        axes[1].set_xticks([])

        fieldline = self.electromagnetic_model.predict(
            self.data.electromagnetic[sample, :, 0],
            self.data.electromagnetic[sample, :, 1],
        )[0]
        axes[2].plot(fieldline.phi, fieldline, "C6", label="fieldline")

        ndiv = len(fieldline)
        wall_hat = np.fft.rfft(self.data.radial_wall[sample, :])
        firstwall = np.fft.irfft(wall_hat, ndiv) * ndiv / self.ncoil
        wall_hat[1] += self.electromagnetic_model.axis_offset[0] * (self.ncoil // 2)
        offset_firstwall = np.fft.irfft(wall_hat, ndiv) * ndiv / self.ncoil
        axes[2].plot(fieldline.phi, firstwall, "-.", color="gray", label="wall")
        axes[2].plot(
            fieldline.phi, offset_firstwall, "-", color="gray", label="offset wall"
        )

        if plot_deviation:
            longwave = np.fft.irfft(
                np.fft.rfft(fieldline - firstwall)[: self.modes + 1], ndiv
            )
            offset_longwave = np.fft.irfft(
                np.fft.rfft(fieldline - offset_firstwall)[: self.modes + 1], ndiv
            )
            peaktopeak = self.electromagnetic_model.peaktopeak(longwave)
            offset_peaktopeak = self.electromagnetic_model.peaktopeak(offset_longwave)
            axes[2].plot(
                fieldline.phi, longwave, "-.C0", label=rf"$H_{{LW}}={peaktopeak:1.1f}$"
            )
            axes[2].plot(
                fieldline.phi,
                offset_longwave,
                "-C0",
                label=rf"offset $H_{{LW}}={offset_peaktopeak:1.1f}$",
            )

        axes[2].legend(fontsize="xx-small", bbox_to_anchor=(1, 1))
        axes[2].set_ylabel("deviation")
        axes[2].set_xlabel(r"$\phi$")
        plt.despine()
        plt.suptitle(f"quantile={quantile} offset={offset}")


@dataclass
class ErrorField(Trial):
    """Run Monte Carlo error field trials."""

    filename: str = "errorfield_trial"
    component: list[str] = field(
        default_factory=lambda: [
            "radial",
            "tangential",
            "vertical",
            "radial_ccl",
            "tangential_ccl",
            "vertical_ccl",
            "pitch_length",
            "roll_length",
            "yaw_length",
        ]
    )
    theta: list[float] = field(default_factory=lambda: [5, 5, 5, 2, 2, 2, 5, 10, 10])
    pdf: list[str] = field(
        default_factory=lambda: [
            "uniform",
            "uniform",
            "uniform",
            "normal",
            "normal",
            "normal",
            "uniform",
            "uniform",
            "uniform",
        ]
    )

    def __post_init__(self):
        """Initialize model instances."""
        self.model = overlap.Model()
        super().__post_init__()

    def build(self):
        """Build Monte Carlo dataset."""
        with self.timer():
            self.build_signal()
            self.predict()
        return self.store()

    def predict(self):
        """Predict overlap error field."""
        self.data["plasma"] = self.model.data.plasma
        self.data["overlap"] = ("sample", "plasma"), np.zeros(
            (self.samples, self.data.sizes["plasma"])
        )
        radial = self.data.radial + self.data.radial_ccl
        tangential = self.data.tangential + self.data.tangential_ccl
        vertical = self.data.vertical + self.data.vertical_ccl
        pitch = self.data.pitch_length / (1e3 * WedgeGap.length["pitch"])
        roll = self.data.roll_length / (1e3 * WedgeGap.length["roll"])
        yaw = self.data.yaw_length / (1e3 * WedgeGap.length["yaw"])
        for i, plasma in enumerate(self.data.plasma.values):
            self.data.overlap[:, i] = self.model.predict(
                plasma, radial, tangential, vertical, pitch, roll, yaw
            )

    def plot(self):
        """Plot overlap errorfield PDFs."""
        plt.figure()
        plt.hist(
            self.data.overlap,
            bins=51,
            density=True,
            rwidth=0.9,
            label=[f"plasma {i}" for i in self.data.plasma.values],
        )
        plt.legend(ncol=1, bbox_to_anchor=(0.27, 1), fontsize="x-small")
        plt.despine()
        axes = plt.gca()
        axes.set_yticks([])
        plt.xlabel(r"Overlap error field $B/B_{limit}$")
        plt.ylabel(r"$P(B/B_{limit})$")
        self.pdf_text()

        quantile_index = np.argmax(np.quantile(self.data.overlap, 0.99, axis=0))
        self.label_quantile(
            self.data.overlap[:, quantile_index],
            r"B/B_{limit}",
            color=f"C{quantile_index}",
        )

    def scan(self, quantile=0.99):
        """Run sensitivity scan."""
        if (
            "quantile_scan" in self.data
            and self.data.attrs.get("quantile", None) == quantile
        ):
            return self
        self.data["quantile_scan"] = ("component", "plasma"), np.ones(
            (self.data.sizes["component"], self.data.sizes["plasma"])
        )
        for i, pdf in enumerate(self.pdf):
            theta = list(np.zeros(len(self.pdf)))
            theta[i] = self.data.theta.values[i]
            error = ErrorField(
                self.samples, component=self.component, theta=theta, pdf=self.pdf
            )
            self.data["quantile_scan"][i] = np.quantile(
                error.data.overlap, quantile, axis=0
            )
        self.data.attrs["quantile"] = quantile
        return self.store()

    def plot_scan(self, quantile=0.99):
        """Plot sensitivity scan results."""
        self.scan(quantile)
        component = [
            component.replace("_", " ") for component in self.data.component.values
        ]
        for i, plasma in enumerate(self.data.plasma.values):
            plt.bar(
                component,
                self.data.quantile_scan[:, i],
                width=0.8 - i * 0.2,
                label=f"plasma {plasma}",
            )
            plt.xticks(rotation=90)
        plt.legend(fontsize="x-small")
        plt.despine()
        plt.ylabel(r"Overlap error field $B/B_{limit}$")


if __name__ == "__main__":
    # theta = [5, 5, 5, 10, 2, 2, 2.5]
    # theta = [0, 0, 0, 10, 0, 0, 0]
    theta = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3]
    vault = Vault(2_000_000, theta=theta)

    #'radial', 'tangential', 'roll_length',
    #'yaw_length', 'radial_ccl', 'tangential_ccl', 'radial_wall'

    vault.plot()
    vault.plot_offset()

    vault.plot_sample(0.99, False)

    # theta_error = [5, 5, 5, 2, 2, 2, 5, 10, 10]

    theta_error = [1.5, 1.5, 3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
    # theta_error = [np.sqrt(3), np.sqrt(3), np.sqrt(3),
    #               1, 1, 1,
    #               np.sqrt(3), np.sqrt(3), np.sqrt(3)]
    # theta_error = list(3*np.ones(9))
    error = ErrorField(2_000_000, theta=theta_error)

    # error.plot_scan()
    error.plot()

    # trial.plot_offset()

    # case -> 1.7/0.3, 2.1/0.8
    # roll -> 0.2/0.1
    # yaw -> 0.2/0.2
    # ccl -> 1.4/0.9, 1.7/0.8
    # wall -> 3.2 / 3.2

    # trial.plot_sample(0.99, False)
    # trial.plot_sample(0.99, True)
