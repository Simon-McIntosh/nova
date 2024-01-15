import pytest

import numpy as np
import matplotlib

from nova.assembly.gap import UniformGap, GapData


def test_plot_gapdata():
    gap_data = GapData(
        ["c1", "c2"],
        np.array(
            [
                [0.741, 1.456],
                [1.663, 2.357],
                [1.941, 2.366],
                [2.396, 2.657],
                [2.205, 1.263],
                [2.253, 2.543],
                [2.94, 3.698],
                [2.401, 1.812],
                [0.976, 3.063],
                [1.718, 2.634],
                [2.236, 1.994],
                [2.887, 2.357],
                [1.337, 2.279],
                [1.483, 2.08],
                [1.687, 1.479],
                [1.142, 0.652],
                [2.007, 1.309],
                [3.986, 0.0],
            ]
        ).T,
    )
    with matplotlib.pylab.ioff():
        gap_data.plot("c1")


def test_k0_gap():
    gap = UniformGap("k0", dirname="root.data/Assembly")
    assert np.allclose(gap.data.sel(simulation="k0").gap, 2)


if __name__ == "__main__":
    pytest.main([__file__])
