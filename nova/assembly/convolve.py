"""Build convolution animation."""
from dataclasses import dataclass, field

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import numpy as np

from nova.assembly import structural
import matplotlib.pyplot as plt


@dataclass
class Convolve:
    """Illustrate filter convolution in spacial domain."""

    simulation: str = 'v3'
    response: str = 'radial'
    ndiv: int = 360
    model: structural.Model = field(default_factory=structural.Model)

    def __post_init__(self):
        """Load dataset."""
        self.phi = np.linspace(0, 2*np.pi, self.ndiv)
        self.data = structural.Transform(self.simulation).data
        self.ncoil = self.data.ncoil

    def interpolate(self, waveform):
        """Return waveform interpolated to ndiv."""
        return np.fft.irfft(np.fft.rfft(waveform), n=self.ndiv) * \
            self.ndiv / self.ncoil

    @property
    def spectral_filter(self):
        """Return spectral filter."""
        return self.model.filter[self.response][:, 0]

    @property
    def spatial_filter(self):
        """Return spacial filter."""
        return np.fft.irfft(self.spectral_filter)
    #* \ self.ndiv / self.ncoil

    def make_frame(self, t):
        """Plot waveforms."""
        duration = 5
        index = t*self.ncoil / duration

        response = self.data.response_delta.sel(response=self.response)
        convolution = np.convolve(np.r_[self.data.gap, self.data.gap],
                                  self.spatial_filter, mode='valid')

        plt.gca().clear()
        plt.bar(self.data.index, self.data.gap, color='C0', label='gap',
                zorder=-2, width=0.9)
        plt.bar(self.data.index-self.data.ncoil, self.data.gap,
                color='gray', zorder=-2, width=0.9)
        plt.bar(self.data.index-self.data.ncoil+index,
                self.spatial_filter[::-1], color='C3', zorder=0,
                label='filter', width=0.9)
        i = int(np.ceil(index))
        plt.bar(self.data.index[:i], convolution[1:i+1], color='C2',
                zorder=-1, label='displacement', width=0.7)
        plt.legend(loc=1, ncol=3, bbox_to_anchor=(1, 1.15))

        ylim = np.max(np.abs([response.min(), response.max()]))
        plt.ylim([-ylim, ylim])

        axes = plt.gca()
        axes.xaxis.set_major_locator(MultipleLocator(4))
        axes.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        axes.xaxis.set_minor_locator(MultipleLocator(1))
        plt.axis('off')

        return mplfig_to_npimage(plt.gcf())

    def animate(self):
        """Animate convolution."""
        animation = VideoClip(self.make_frame, duration=5)
        animation.write_gif('convolve.gif', fps=20)


if __name__ == '__main__':

    conv = Convolve()
    conv.animate()
