import tempfile

import numpy as np
import pytest

from nova.imas.database import Database
from nova.imas.equilibrium import EquilibriumData
from nova.imas.extrapolate import Extrapolate, Grid
from nova.imas.test_utilities import ids_attrs, load_ids, mark

ids_attrs["CORSICA"] = dict(pulse=130506, run=403, name="equilibrium")
mark["CORSICA"] = pytest.mark.skipif(
    not load_ids(**ids_attrs["CORSICA"]).is_valid, reason="CORSICA database unavalible"
)


def test_extrapolation_grid_relitive_to_coilset():
    grid_attrs = Grid(100, 0, "coil").grid_attrs
    assert grid_attrs == {"number": 100, "limit": 0, "index": "coil"}


@mark["CORSICA"]
def test_load_from_ids():
    equilibrium = Database(**ids_attrs["CORSICA"], lazy=False)
    kwargs = dict(
        ids=equilibrium.ids,
        limit="ids",
        ngrid=5,
        dplasma=-5,
        tplasma="hex",
        wall="iter_md",
    )
    with tempfile.NamedTemporaryFile() as tmp:
        extrapolate = Extrapolate(**kwargs, filename=tmp.name)
        extrapolate._clear()


@mark["equilibrium"]
def test_extrapolation_grid_relitive_to_ids():
    ids = EquilibriumData(
        ids_attrs["equilibrium"]["pulse"], ids_attrs["equilibrium"]["run"]
    ).ids
    grid = Grid(50, "ids", ids=ids)
    assert grid.grid_attrs == {
        "number": 50,
        "limit": [2.75, 8.9, -5.49, 5.51],
        "index": "plasma",
    }


@mark["equilibrium"]
def test_extrapolation_grid_exact_copy_of_ids():
    ids = EquilibriumData(
        ids_attrs["equilibrium"]["pulse"], ids_attrs["equilibrium"]["run"]
    ).ids
    grid = Grid("ids", "ids", ids=ids)
    assert grid.grid_attrs["number"] == 8385


def test_extrapolation_grid_raises():
    with pytest.raises(AttributeError) as error:
        Grid(1000, "ids", "coil")
    assert "Require IMAS ids when" in str(error.value)


@mark["CORSICA"]
def test_extrapolate_attrs():
    extrapolate = Extrapolate(
        **ids_attrs["CORSICA"],
        dplasma=-1,
        tplasma="hex",
        pf_active="iter_md",
        wall="iter_md",
    )
    assert extrapolate.pulse == ids_attrs["CORSICA"]["pulse"]
    assert extrapolate.run == ids_attrs["CORSICA"]["run"]
    assert extrapolate.ids.code.name == "CORSICA"


@mark["CORSICA"]
@pytest.mark.parametrize("itime", [5, 15, 20, 30, 35, 40])
def test_extrapolate_rms_error(itime):
    ids = EquilibriumData(**ids_attrs["CORSICA"], lazy=False).ids
    extrapolate = Extrapolate(
        ids=ids, limit="ids", ngrid=50, dplasma=-250, nturn=10, wall="iter_md"
    )
    extrapolate.itime = itime
    extrapolate_psi = extrapolate.grid.psi_.copy()

    radius = extrapolate.grid.data.x2d
    height = extrapolate.grid.data.z2d
    equilibrium_psi = extrapolate.psi_rbs(radius, height)

    extrapolate_psi_norm = extrapolate.normalize(-extrapolate_psi)
    equilibrium_psi_norm = extrapolate.normalize(equilibrium_psi)

    error = np.sqrt(np.mean((equilibrium_psi_norm - extrapolate_psi_norm) ** 2))
    maxmin = equilibrium_psi_norm.max() - equilibrium_psi_norm.min()
    assert error / maxmin < 0.05


test_extrapolate_rms_error(5)
assert False

if __name__ == "__main__":
    pytest.main([__file__])
