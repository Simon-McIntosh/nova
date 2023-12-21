import pytest

import imaspy
import numpy as np

from nova.imas.database import Database
from nova.imas.test_utilities import ids_attrs, mark


@mark["equilibrium"]
def test_lazy_ids_data():
    equilibrium = Database(**ids_attrs["equilibrium"])
    with equilibrium.load_ids as ids:
        assert isinstance(
            ids.time_slice[0].global_quantities.ip, imaspy.ids_primitive.IDSFloat0D
        )

    assert equilibrium.ids is None

    print(ids.time_slice[12].global_quantities.ip)
    assert False


if __name__ == "__main__":
    pytest.main([__file__])
