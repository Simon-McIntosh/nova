import pytest

try:
    from nova.imas.database import IDS, IMASdb, Database
except ImportError:
    pytest.skip("IMAS installation unabalible", allow_module_level=True)


def load_pf_active():
    try:
        return IMASdb('public', 'iter_md').ids(111001, 1, 'pf_active')
    except Exception:
        return False


pf_active = pytest.mark.skipif(
    not load_pf_active(), reason='pf_active database unavalible')


def test_ids():
    ids = IDS(111001, 1, 'pf_active')
    assert ids.shot == 111001
    assert ids.run == 1
    assert ids.ids_name == 'pf_active'


@pf_active
def test_IMASdb():
    pf_active = IMASdb('public', 'iter_md').ids(111001, 1, 'pf_active')
    assert pf_active.coil.array[0].identifier == 'CS3U'


@pf_active
def test_database():
    database = Database(111001, 1, 'pf_active', 'public', 'iter_md')
    ids_data = database.load_ids_data()
    assert ids_data.ids_properties.source == 'ITER_D_33NHXN'


if __name__ == '__main__':

    pytest.main([__file__])
