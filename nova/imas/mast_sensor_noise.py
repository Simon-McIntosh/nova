"""MAST compatibility adapter for array-only sensor-noise kernels.

The MAST cohort supplies waveform records; all detrending, pooling and repeat
scatter calculations live in :mod:`nova.calibrate.noise`.
"""

from nova.calibrate.noise import (
    CAMPAIGN_SHOT_SPAN,
    MINIMUM_NOISE_SAMPLES,
    ChannelNoise,
    NoiseEnvelope,
    NoiseError,
    NoiseFit,
    RepeatScatter,
    WaveformArrays,
    measure_noise_envelope,
    measure_noise,
    measure_repeat_scatter,
    repeat_groups,
)

__all__ = [
    "CAMPAIGN_SHOT_SPAN",
    "MINIMUM_NOISE_SAMPLES",
    "ChannelNoise",
    "NoiseEnvelope",
    "NoiseError",
    "NoiseFit",
    "RepeatScatter",
    "WaveformArrays",
    "measure_noise_envelope",
    "measure_noise",
    "measure_repeat_scatter",
    "repeat_groups",
]
