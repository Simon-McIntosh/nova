from itertools import product
import pytest

import matplotlib.pylab
import numpy as np

from nova.graphics.plot import Animate


def test_fps():
    fps = 12
    animate = Animate(fps)
    assert animate.fps == fps


def test_duration():
    animate = Animate()
    animate.add_animation("elongation", 2, amplitude=0.2)
    assert np.isclose(animate.duration, 2)
    assert np.isclose(animate._segments["elongation"][-1, 0], 2)


def test_set_duration():
    animate = Animate()
    animate.add_animation("elongation", 2, amplitude=0.2)
    animate.add_animation("minor_radius", 3.5, amplitude=0.2)
    animate.duration = 1
    assert np.isclose(animate.duration, 1)
    assert np.isclose(animate._segments["elongation"][-1, 0], 2 / 5.5)


@pytest.mark.parametrize("append", [True, False])
def test_append(append):
    animate = Animate()
    animate.add_animation("elongation", 2, amplitude=0.2, offset=True)
    animate.add_animation("triangularity", 3, amplitude=0.2, offset=True, append=False)
    animate.add_animation("minor_radius", 2.5, amplitude=0.2, append=append)

    if append:
        assert np.isclose(animate.duration, 5.5)
        assert np.isclose(animate._segments["minor_radius"][-1, 0], 5.5)
    else:
        assert np.isclose(animate.duration, 3)
        assert np.isclose(animate._segments["minor_radius"][-1, 0], 2.5)


def test_make_frame():
    with pytest.raises(NotImplementedError):
        Animate().make_frame(0)


def test_plot_animation():
    animate = Animate()
    animate.add_animation("elongation", 2, amplitude=0.2, offset=True)
    animate.add_animation("triangularity", 3, amplitude=0.2, offset=True, append=False)
    animate.add_animation("minor_radius", 2.5, amplitude=0.2, append=True)
    with matplotlib.pylab.ioff():
        animate.plot_animation()
    assert True


@pytest.mark.parametrize(
    "amplitude,offset", product([0.2, 0.8], [-3.3, 4, True, False])
)
def test_sin(amplitude, offset):
    animate = Animate()
    animate.elongation = 7.2
    animate.add_animation("elongation", 2, amplitude=amplitude, offset=offset)
    if offset is True:
        offset = 7.2
    if offset is False:
        offset = 0
    waveform = animate._segments["elongation"][:, 1]
    assert np.isclose(np.nanmax(waveform), offset + amplitude, 1e-3)
    assert np.isclose(np.nanmin(waveform), offset - amplitude, 1e-3)


def test_scene():
    animate = Animate()
    animate.add_animation("elongation", 2 * np.pi, amplitude=3.4, offset=3)
    animate.add_animation("triangularity", 2 * np.pi, amplitude=1.1, repeat=2, num=200)
    assert np.isclose(animate.scene(np.pi / 2)["elongation"], 6.4, 1e-3)
    assert np.isclose(animate.scene(2 * np.pi + np.pi / 4)["triangularity"], 1.1, 1e-3)


@pytest.mark.parametrize("ramp,offset", product([-3.1, 2], [0, -0.5]))
def test_ramp(ramp, offset):
    animate = Animate()
    animate.add_animation("elongation", 3.5, ramp=ramp, offset=offset)
    assert np.isclose(animate.scene(0)["elongation"], offset, 1e-3)
    assert np.isclose(animate.scene(3.5)["elongation"], offset + ramp, 1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
