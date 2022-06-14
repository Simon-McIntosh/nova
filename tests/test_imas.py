import pytest

try:
    from nova.imas.database import IDS, IMASdb, Database
except ImportError:
    pytest.skip("IMAS installation unabalible", allow_module_level=True)


pulse, run = 111001, 201


def load_pf_active():
    try:
        return IMASdb('public', 'iter_md').ids(pulse, run, 'pf_active', None)
    except Exception:
        return False


pf_active = pytest.mark.skipif(
    not load_pf_active(), reason='pf_active database unavalible')


def test_ids():
    ids = IDS(pulse, run, 'pf_active')
    assert ids.pulse == pulse
    assert ids.run == run
    assert ids.ids_name == 'pf_active'


@pf_active
def test_IMASdb():
    pf_active = IMASdb('public', 'iter_md').ids(pulse, run, 'pf_active', None)
    assert pf_active.coil.array[0].identifier == 'CS3U'


@pf_active
def test_database():
    database = Database(pulse, run, 'pf_active',
                        user='public', machine='iter_md')
    ids_data = database.load_ids_data()
    assert 'ITER_D_33NHXN' in ids_data.ids_properties.source


if __name__ == '__main__':

    pytest.main([__file__])
