
import pytest

from nova.electromagnetic.frame import Frame


def test_init():
    frame = Frame(link=True, metadata={'additional': ['link']})
    frame.add_frame(4, [5, 7, 12], 0.1, 0.05)
    return frame


def test_len():
    frame = Frame(Required=['x', 'z'])
    frame.add_frame(4, range(30), link=True)
    frame.add_frame(4, range(2), link=False)
    frame.add_frame(4, range(4), link=True)
    assert len(frame.subspace) == 4 and len(frame) == 36


def test_getattr():
    frame = Frame(Required=['x', 'z'])
    frame.add_frame(4, range(1), Ic=5, link=True)
    frame.add_frame(4, range(2), link=False)
    assert frame.Ic.to_list() == [5.0, 0.0, 0.0]


def test_getitem():
    frame = Frame(Required=['x', 'z'])
    frame.add_frame(4, range(3), Ic=5, link=True)
    frame.add_frame(4, range(2), link=False)
    assert frame['Ic'].to_list() == [5.0, 0.0, 0.0]


def test_setattr():
    frame = Frame(Required=['x', 'z'])
    frame.add_frame(4, range(7), Ic=5, link=True)
    frame.add_frame(4, range(2), link=False)
    frame.Ic = [3.6, 5.2, 10]
    assert frame.Ic.to_list() == [3.6, 5.2, 10.0]


def test_setitem():
    frame = Frame(Required=['x', 'z'])
    frame.add_frame(4, range(5), Ic=5, link=True)
    frame.add_frame(4, range(2), link=False)
    frame['Ic'] = [3.6, 5.2, 10]
    assert frame.Ic.to_list() == [3.6, 5.2, 10.0]


def test_loc():
    frame = Frame(Required=['x', 'z'], Additional=[])
    frame.add_frame(4, range(2), Ic=5.0, link=True)
    frame.add_frame(4, range(2), Ic=0.0, link=False)
    print(frame)
    print(frame.loc[:, 'It'], frame.It.to_list(), frame.It)
    frame.loc[:, 'It'] = [3.6, 5.2, 0]
    assert frame.It.to_list() == [3.6, 5.2, 0]


def test_iloc():
    frame = Frame(Required=['x', 'z'], Additional=[])
    frame.add_frame(4, range(7), Ic=5, link=True)
    frame.add_frame(4, range(2), link=False)
    frame.Ic = 0
    frame.Ic.iloc[1] = 3.6
    assert frame.Ic.to_list() == [0, 3.6, 0]


def test_set_at():
    frame = Frame(Required=['x', 'z'], Additional=[])
    frame.add_frame(4, range(7), Ic=5, link=True)
    frame.add_frame(4, range(2), link=False)
    frame.Ic = 0.0
    frame.at['Coil7', 'Ic'] = 3.6
    assert frame.Ic.to_list() == [0, 3.6, 0]


def test_set_iat():
    frame = Frame(Required=['x', 'z'], Additional=[])
    frame.add_frame(4, range(7), Ic=5, link=True)
    frame.add_frame(4, range(2), link=False)
    frame.Ic = 0.0
    frame.iat[-2, 2] = 3.6
    assert frame.Ic.to_list() == [0, 3.6, 0]


def test_get_at():
    frame = Frame(Required=['x', 'z'], Additional=[])
    frame.add_frame(4, range(7), Ic=5, link=True)
    frame.add_frame(4, range(2), link=False)
    frame.Ic = [7.4, 3.2, 6.666]
    assert frame.at['Coil8', 'Ic'] == 6.666


def test_get_at_keyerror():
    frame = Frame(Required=['x', 'z'], Additional=[])
    frame.add_frame(4, range(7), Ic=5, link=True)
    frame.add_frame(4, range(2), link=False)
    frame.Ic = [7.4, 3.2, 6.666]
    with pytest.raises(KeyError):
        _ = frame.at['Coil6', 'Ic']


def test_get_iat_indexerror():
    frame = Frame(Required=['x', 'z'], Additional=[])
    frame.add_frame(4, range(7), Ic=5, link=True)
    frame.add_frame(4, range(2), link=False)
    frame.Ic = [7.4, 3.2, 6.666]
    with pytest.raises(IndexError):
        _ = frame.iat[7, 2]


def test_get_frame():
    frame = Frame(Required=['x', 'z'])
    frame.add_frame(4, range(3), Ic=5.7, link=True)
    frame.add_frame(4, range(2), Ic=3.2, link=False)
    assert frame.get_frame('Ic').to_list() == [5.7, 5.7, 5.7, 3.2, 3.2]


def test_setattr_error():
    frame = Frame(Required=['x', 'z'])
    frame.add_frame(4, range(3), Ic=5.7, link=True)
    with pytest.raises(IndexError):
        with frame.metaframe.setlock(False):
            frame.Ic = range(3)


def test_setitem_error():
    frame = Frame(Required=['x', 'z'])
    frame.add_frame(4, range(3), Ic=5.7, link=True)
    with pytest.raises(IndexError):
        with frame.metaframe.setlock(False):
            frame['Ic'] = range(3)


if __name__ == '__main__':

    pytest.main([__file__])
