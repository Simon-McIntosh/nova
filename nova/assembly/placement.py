"""Build windowed placement annimation."""
from dataclasses import dataclass, field
from typing import ClassVar

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np

import matplotlib.pyplot as plt


@dataclass
class Placement:
    """Illustrate coil placement within windows."""

    bound: float = 1.5
    ncoil: int = 18
    samples_per_second: float = 2.5
    sample_time: float = field(init=False, repr=False)
    amplitude: float = field(init=False, repr=False)
    phase: float = field(init=False, repr=False)

    sead: ClassVar[int] = 2025

    def __post_init__(self):
        """Load dataset."""
        self.index = np.arange(1, self.ncoil + 1)
        self.rng = np.random.default_rng(self.sead)
        self.sample_time = -1 / self.samples_per_second
        self.phi = np.linspace(0, 2 * np.pi, 100)

    def make_frame(self, t):
        """Plot waveforms."""
        last_sample = t - self.sample_time
        if last_sample >= 1 / self.samples_per_second:
            self.sample_time = t
            self.sample = self.rng.uniform(-self.bound, self.bound, size=self.ncoil)
            n1 = np.fft.rfft(self.sample)[1]
            self.amplitude = np.abs(n1) / (self.ncoil // 2)
            self.phase = np.angle(n1, False)

        plt.gca().clear()
        plt.bar(self.index, self.sample, color="C0", label="gap", zorder=-2, width=0.9)
        plt.bar(
            self.index,
            self.bound * np.cos((self.index - 1) * 2 * np.pi / self.ncoil),
            color="gray",
            zorder=-3,
            width=0.9,
        )

        plt.plot(
            1 + self.phi * self.ncoil / (2 * np.pi),
            self.amplitude * np.cos(self.phi + self.phase),
            "C3-.",
            zorder=1,
        )

        plt.plot(self.index, self.bound * np.ones(self.ncoil), "--", color="gray")
        plt.plot(self.index, -self.bound * np.ones(self.ncoil), "--", color="gray")
        axes = plt.gca()
        axes.xaxis.set_major_locator(MultipleLocator(4))
        axes.xaxis.set_major_formatter(FormatStrFormatter("%d"))
        axes.xaxis.set_minor_locator(MultipleLocator(1))
        plt.axis("off")

        return mplfig_to_npimage(plt.gcf())

    def animate(self):
        """Animate convolution."""
        animation = VideoClip(self.make_frame, duration=10)
        animation.write_gif("placement.gif", fps=20)


if __name__ == "__main__":
    Placement().animate()
