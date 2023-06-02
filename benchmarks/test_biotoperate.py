import pytest

from biotoperate import PlasmaTurns


def test_plasma_turns():
    biot = PlasmaTurns()
    biot.setup_cache()
    biot.setup(75)
    biot.time_update_turns(75)
    biot.remove()


if __name__ == "__main__":
    pytest.main([__file__])
