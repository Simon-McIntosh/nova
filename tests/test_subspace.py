
import pytest

from nova.electromagnetic.coilframe import CoilFrame
from nova.electromagnetic.dataframe import SubSpaceError


def test_init():
    coilframe = CoilFrame(link=True, metadata={'additional': ['link']})
    coilframe.add_frame(4, [5, 7, 12], 0.1, 0.05)
    return coilframe


def test_len():
    coilframe = CoilFrame(Required=['x', 'z'])
    coilframe.add_frame(4, range(30), link=True)
    coilframe.add_frame(4, range(2), link=False)
    coilframe.add_frame(4, range(4), link=True)
    assert len(coilframe.subspace) == 4 and len(coilframe) == 36


def test_getattr():
    coilframe = CoilFrame(Required=['x', 'z'])
    coilframe.add_frame(4, range(1), Ic=5, link=True)
    coilframe.add_frame(4, range(2), link=False)
    assert coilframe.Ic.to_list() == [5.0, 0.0, 0.0]


def test_getitem():
    coilframe = CoilFrame(Required=['x', 'z'])
    coilframe.add_frame(4, range(3), Ic=5, link=True)
    coilframe.add_frame(4, range(2), link=False)
    assert coilframe['Ic'].to_list() == [5.0, 0.0, 0.0]


def test_setattr():
    coilframe = CoilFrame(Required=['x', 'z'])
    coilframe.add_frame(4, range(7), Ic=5, link=True)
    coilframe.add_frame(4, range(2), link=False)
    coilframe.Ic = [3.6, 5.2, 10]
    assert coilframe.Ic.to_list() == [3.6, 5.2, 10.0]


def test_setattr_energize():
    coilframe = CoilFrame(Required=['x', 'z'])
    coilframe.add_frame(4, range(7), Ic=5, Nt=3.6, link=True)
    coilframe.add_frame(4, range(2), Nt=5.2, link=False)
    coilframe.It = [3.6, 5.2, 10.4]
    assert coilframe.Ic.to_list() == [1, 1, 2]


def test_setitem():
    coilframe = CoilFrame(Required=['x', 'z'])
    coilframe.add_frame(4, range(5), Ic=5, link=True)
    coilframe.add_frame(4, range(2), link=False)
    coilframe['Ic'] = [3.6, 5.2, 10]
    assert coilframe.Ic.to_list() == [3.6, 5.2, 10.0]


def test_loc():
    coilframe = CoilFrame(Required=['x', 'z'], Additional=[])
    coilframe.add_frame(4, range(2), Ic=5, link=True)
    coilframe.add_frame(4, range(2), Ic=0, link=False)
    coilframe.subspace.loc[:, 'It'] = [3.6, 5.2, 0]
    assert coilframe.It.to_list() == [3.6, 5.2, 0]


def test_loc_slice():
    coilframe = CoilFrame(Required=['x', 'z'], Additional=[], label='Coil',
                          offset=15)
    coilframe.add_frame(4, range(2), It=5, link=True)
    coilframe.add_frame(4, range(2), It=7.3, link=False)
    coilframe.subspace.loc['Coil15':'Coil17', 'It'] = [3.6, 5.2]
    assert coilframe.It.to_list() == [3.6, 5.2, 7.3]


def test_loc_error():
    coilframe = CoilFrame(Required=['x', 'z'], Additional=[])
    coilframe.add_frame(4, range(2), Ic=5, link=True)
    coilframe.add_frame(4, range(2), Ic=0, link=False)
    with pytest.raises(SubSpaceError):
        coilframe.loc[:, 'It'] = [3.6, 5.2, 0]


def test_iloc():
    coilframe = CoilFrame(Required=['x', 'z'], Additional=[])
    coilframe.add_frame(4, range(7), Ic=5, link=True)
    coilframe.add_frame(4, range(2), link=False)
    #coilframe.Ic = 0
    coilframe.subspace.iloc[1, 0] = 3.6
    print(coilframe.subspace)
    print(coilframe.Ic)
    assert coilframe.Ic.to_list() == [0, 3.6, 0]


def test_set_at():
    coilframe = CoilFrame(Required=['x', 'z'], Additional=[])
    coilframe.add_frame(4, range(7), Ic=5, link=True)
    coilframe.add_frame(4, range(2), link=False)
    coilframe.Ic = 0.0
    coilframe.subspace.at['Coil7', 'Ic'] = 3.6
    assert coilframe.Ic.to_list() == [0, 3.6, 0]


def test_set_iat():
    coilframe = CoilFrame(Required=['x', 'z'], Additional=[])
    coilframe.add_frame(4, range(7), Ic=5, link=True)
    coilframe.add_frame(4, range(2), link=False)
    coilframe.Ic = 0.0
    coilframe.subspace.iat[-2, 0] = 3.6
    assert coilframe.Ic.to_list() == [0, 3.6, 0]


def test_get_at():
    coilframe = CoilFrame(Required=['x', 'z'], Additional=[])
    coilframe.add_frame(4, range(7), Ic=5, link=True)
    coilframe.add_frame(4, range(2), link=False)
    coilframe.Ic = [7.4, 3.2, 6.666]
    assert coilframe.at['Coil8', 'Ic'] == 6.666


def test_get_at_keyerror():
    coilframe = CoilFrame(Required=['x', 'z'], Additional=[])
    coilframe.add_frame(4, range(7), Ic=5, link=True)
    coilframe.add_frame(4, range(2), link=False)
    coilframe.Ic = [7.4, 3.2, 6.666]
    with pytest.raises(KeyError):
        _ = coilframe.subspace.at['Coil6', 'Ic']


def test_get_iat_indexerror():
    coilframe = CoilFrame(Required=['x', 'z'], Additional=[])
    coilframe.add_frame(4, range(7), Ic=5, link=True)
    coilframe.add_frame(4, range(2), link=False)
    coilframe.Ic = [7.4, 3.2, 6.666]
    with pytest.raises(IndexError):
        _ = coilframe.subspace.iat[7, 2]


def test_get_frame():
    coilframe = CoilFrame(Required=['x', 'z'])
    coilframe.add_frame(4, range(3), Ic=5.7, link=True)
    coilframe.add_frame(4, range(2), Ic=3.2, link=False)
    assert coilframe.get_frame('Ic').to_list() == [5.7, 5.7, 5.7, 3.2, 3.2]


def test_setattr_error():
    coilframe = CoilFrame(Required=['x', 'z'])
    coilframe.add_frame(4, range(3), Ic=5.7, link=True)
    with pytest.raises(IndexError):
        with coilframe.metaframe.setlock(False, 'subspace'):
            coilframe.Ic = range(3)


def test_setitem_error():
    coilframe = CoilFrame(Required=['x', 'z'])
    coilframe.add_frame(4, range(3), Ic=5.7, link=True)
    with pytest.raises(IndexError):
        with coilframe.metaframe.setlock(False, 'subspace'):
            coilframe['Ic'] = range(3)


def test_subspace_lock():
    coilframe = CoilFrame(metadata={'Required': ['x'], 'Subspace': ['x']})
    assert coilframe.in_field('x', 'subspace')
    assert coilframe.metaframe.lock('subspace')


if __name__ == '__main__':

    test_iloc()
    #pytest.main([__file__])
