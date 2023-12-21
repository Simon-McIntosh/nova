import pytest

import imaspy
import numpy as np

from nova.imas.database import Database
from nova.imas.test_utilities import ids_attrs, mark


@mark["equilibrium"]
def test_lazy_ids_data():
    equilibrium = Database(**ids_attrs["equilibrium"])
    print(equilibrium.name)

    print(type(equilibrium.ids_data().time_slice[0].global_quantities.ip))
    with equilibrium.ids_data as ids_data:
        assert isinstance(
            ids_data.time_slice[0].global_quantities.ip, imaspy.ids_primitive.IDSFloat0D
        )
    assert equilibrium.ids is None


if __name__ == "__main__":
    pytest.main([__file__])
