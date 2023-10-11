import numpy as np
import pytest

from nova.imas.matrix import Benchmark

try:
    import imas
except ImportError:
    pytest.skip("IMAS module unavailable", allow_module_level=True)


try:
    benchmark = Benchmark(dplasma=-150, nfield=15, nforce=100, ngrid=None)
except imas.ids_base.ALException:
    pytest.skip(
        "Requisite IDSs unavailable. "
        "Unable to initiate matrix benchmark, skipping tests",
        allow_module_level=True,
    )


@pytest.mark.parametrize("itime", [0, 10, 200, 1000, 1800, -1])
def test_field(itime):
    benchmark.itime = itime
    field_index = [
        i
        for i, name in enumerate(benchmark.field.coil_name)
        if name in benchmark.profile.data.coil_name
        and benchmark.profile["b_field_max_timed"].data[i] != 0
    ]
    profile_index = [
        i
        for i, name in enumerate(benchmark.profile.data.coil_name.data)
        if name in benchmark.field.coil_name[field_index]
    ]
    assert np.allclose(
        benchmark.field.bp[field_index],
        benchmark.profile["b_field_max_timed"][profile_index],
        rtol=1e-1,
    )


@pytest.mark.parametrize("itime", [0, 10, 200, 1000, 1800])
def test_force(itime):
    benchmark.itime = itime
    force_index = [
        i
        for i, name in enumerate(benchmark.force.coil_name)
        if name in benchmark.profile.data.coil_name
    ]
    profile_index = [
        i
        for i, name in enumerate(benchmark.profile.data.coil_name.data)
        if name in benchmark.force.coil_name[force_index]
    ]
    assert np.allclose(
        benchmark.force.fr[force_index],
        benchmark.profile["radial_force"][profile_index],
        rtol=1e-1,
        atol=1e6,
    )
    assert np.allclose(
        benchmark.force.fz[force_index],
        benchmark.profile["vertical_force"][profile_index],
        rtol=1e-1,
        atol=1e6,
    )


if __name__ == "__main__":
    pytest.main([__file__])
