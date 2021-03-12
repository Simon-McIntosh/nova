
import pytest

from nova.electromagnetic.superspace import SuperSpace
from nova.electromagnetic.superspace import SuperSpaceIndexError


def test_init():
    superspace = SuperSpace(link=True, metadata={'additional': ['link']})
    superspace.add_frame(4, [5, 7, 12], 0.1, 0.05)
    return superspace


def test_len():
    superspace = SuperSpace(Required=['x', 'z'])
    superspace.add_frame(4, range(30), link=True)
    superspace.add_frame(4, range(2), link=False)
    superspace.add_frame(4, range(4), link=True)
    assert len(superspace.subspace) == 4 and len(superspace) == 36


def test_getattr():
    superspace = SuperSpace(Required=['x', 'z'])
    superspace.add_frame(4, range(1), Ic=5, link=True)
    superspace.add_frame(4, range(2), link=False)
    assert superspace.Ic.to_list() == [5.0, 0.0, 0.0]


def test_getitem():
    superspace = SuperSpace(Required=['x', 'z'])
    superspace.add_frame(4, range(3), Ic=5, link=True)
    superspace.add_frame(4, range(2), link=False)
    assert superspace['Ic'].to_list() == [5.0, 0.0, 0.0]


def test_setattr():
    superspace = SuperSpace(Required=['x', 'z'])
    superspace.add_frame(4, range(7), Ic=5, link=True)
    superspace.add_frame(4, range(2), link=False)
    superspace.Ic = [3.6, 5.2, 10]
    assert superspace.Ic.to_list() == [3.6, 5.2, 10.0]


def test_setitem():
    superspace = SuperSpace(Required=['x', 'z'])
    superspace.add_frame(4, range(5), Ic=5, link=True)
    superspace.add_frame(4, range(2), link=False)
    superspace['Ic'] = [3.6, 5.2, 10]
    assert superspace.Ic.to_list() == [3.6, 5.2, 10.0]


def test_loc():
    superspace = SuperSpace(Required=['x', 'z'], Additional=[])
    superspace.add_frame(4, range(2), Ic=5, link=True)
    superspace.add_frame(4, range(2), Ic=0, link=False)
    superspace.subspace.loc[:, 'It'] = [3.6, 5.2, 0]
    assert superspace.It.to_list() == [3.6, 5.2, 0]


def test_loc_slice():
    superspace = SuperSpace(Required=['x', 'z'], Additional=[], label='Coil',
                  offset=15)
    superspace.add_frame(4, range(2), It=5, link=True)
    superspace.add_frame(4, range(2), It=7.3, link=False)
    superspace.subspace.loc['Coil15':'Coil17', 'It'] = [3.6, 5.2]
    assert superspace.It.to_list() == [3.6, 5.2, 7.3]


def test_loc_error():
    superspace = SuperSpace(Required=['x', 'z'], Additional=[])
    superspace.add_frame(4, range(2), Ic=5, link=True)
    superspace.add_frame(4, range(2), Ic=0, link=False)
    with pytest.raises(SuperSpaceIndexError):
        superspace.loc[:, 'It'] = [3.6, 5.2, 0]


def test_iloc():
    superspace = SuperSpace(Required=['x', 'z'], Additional=[])
    superspace.add_frame(4, range(7), Ic=5, link=True)
    superspace.add_frame(4, range(2), link=False)
    superspace.Ic = 0
    superspace.subspace.iloc[1, 0] = 3.6
    assert superspace.Ic.to_list() == [0, 3.6, 0]


def test_set_at():
    superspace = SuperSpace(Required=['x', 'z'], Additional=[])
    superspace.add_frame(4, range(7), Ic=5, link=True)
    superspace.add_frame(4, range(2), link=False)
    superspace.Ic = 0.0
    superspace.subspace.at['Coil7', 'Ic'] = 3.6
    assert superspace.Ic.to_list() == [0, 3.6, 0]


def test_set_iat():
    superspace = SuperSpace(Required=['x', 'z'], Additional=[])
    superspace.add_frame(4, range(7), Ic=5, link=True)
    superspace.add_frame(4, range(2), link=False)
    superspace.Ic = 0.0
    superspace.subspace.iat[-2, 0] = 3.6
    assert superspace.Ic.to_list() == [0, 3.6, 0]


def test_get_at():
    superspace = SuperSpace(Required=['x', 'z'], Additional=[])
    superspace.add_frame(4, range(7), Ic=5, link=True)
    superspace.add_frame(4, range(2), link=False)
    superspace.Ic = [7.4, 3.2, 6.666]
    assert superspace.at['Coil8', 'Ic'] == 6.666


def test_get_at_keyerror():
    superspace = SuperSpace(Required=['x', 'z'], Additional=[])
    superspace.add_frame(4, range(7), Ic=5, link=True)
    superspace.add_frame(4, range(2), link=False)
    superspace.Ic = [7.4, 3.2, 6.666]
    with pytest.raises(KeyError):
        _ = superspace.subspace.at['Coil6', 'Ic']


def test_get_iat_indexerror():
    superspace = SuperSpace(Required=['x', 'z'], Additional=[])
    superspace.add_frame(4, range(7), Ic=5, link=True)
    superspace.add_frame(4, range(2), link=False)
    superspace.Ic = [7.4, 3.2, 6.666]
    with pytest.raises(IndexError):
        _ = superspace.subspace.iat[7, 2]


def test_get_frame():
    superspace = SuperSpace(Required=['x', 'z'])
    superspace.add_frame(4, range(3), Ic=5.7, link=True)
    superspace.add_frame(4, range(2), Ic=3.2, link=False)
    assert superspace.get_frame('Ic').to_list() == [5.7, 5.7, 5.7, 3.2, 3.2]


def test_setattr_error():
    superspace = SuperSpace(Required=['x', 'z'])
    superspace.add_frame(4, range(3), Ic=5.7, link=True)
    with pytest.raises(IndexError):
        with superspace.metaframe.setlock('subspace', False):
            superspace.Ic = range(3)


def test_setitem_error():
    superspace = SuperSpace(Required=['x', 'z'])
    superspace.add_frame(4, range(3), Ic=5.7, link=True)
    with pytest.raises(IndexError):
        with superspace.metaframe.setlock('subspace', False):
            superspace['Ic'] = range(3)


def test_subspace_lock():
    superspace = SuperSpace(metadata={'Required': ['x'], 'Subspace': ['x']})
    assert superspace.in_subspace('x') and superspace.metaframe.lock


if __name__ == '__main__':

    pytest.main([__file__])
