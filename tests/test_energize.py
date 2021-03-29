
import pytest

from nova.electromagnetic.frame import Frame


def test_in_energize():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    print(frame.metadata)

    assert frame.metaframe.hascol('energize', 'It')
test_in_energize()


def test_set_loc_Ic():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    frame.insert(0.5, [6, 8.3], Nt=0.5)
    frame.subspace.loc[:, 'Ic'] = 6.6
    assert frame.loc[:, 'It'].to_list() == [3.3, 3.3]


def test_set_loc_It():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    frame.insert(0.5, [6, 8.3], Nt=0.25)
    frame.subspace.loc[:, 'It'] = 6.6
    assert frame.loc[:, 'Ic'].to_list() == [26.4, 26.4]


def test_set_loc_Nt():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    frame.insert(0.5, [6, 8.3], Nt=1)
    frame.subspace.loc[:, 'It'] = 6.6
    frame.subspace.loc[:, 'Nt'] = 2.2
    assert frame.loc[:, 'It'].to_list() == [14.52, 14.52]


def test_set_item_Ic():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    frame.insert(0.5, [6, 8.3], Nt=0.5)
    frame['Ic'] = 6.6
    assert frame['It'].to_list() == [3.3, 3.3]


def test_set_item_It():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    frame.insert(0.5, [6, 8.3], Nt=0.25)
    frame['It'] = 6.6
    assert frame['Ic'].to_list() == [26.4, 26.4]


def test_set_item_Nt():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    frame.insert(0.5, [6, 8.3], Nt=1)
    frame['It'] = 6.6
    frame['Nt'] = 2.2
    assert frame['It'].to_list() == [14.52, 14.52]


def test_set_attr_Ic():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    frame.insert(0.5, [6, 8.3], Nt=0.5)
    frame.Ic = 6.6
    assert frame.It.to_list() == [3.3, 3.3]


def test_set_attr_It():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    frame.insert(0.5, [6, 8.3], Nt=0.25)
    frame.It = 6.6
    assert frame.Ic.to_list() == [26.4, 26.4]


def test_set_attr_Nt():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    frame.insert(0.5, [6, 8.3], Nt=1)
    frame.It = 6.6
    frame.Nt = 2.2
    assert frame.It.to_list() == [14.52, 14.52]


def test_subspace_Ic():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    frame.insert(0.5, [6, 8.3], Nt=0.5)
    frame.insert(0.5, range(10), Nt=3.5, link=True)
    frame.Ic = [6.6, 6.6, 1]
    assert frame.It.to_list() == [3.3, 3.3, 3.5]


def test_subspace_intersect_columns():
    frame = Frame(Required=['x', 'z'], Additional=['Ic'])
    frame.insert(0.5, [6, 8.3], Nt=0.5)
    frame.insert(0.5, range(10), Nt=3.5, link=True)
    frame.Ic = [6.6, 6.6, 1]
    frame.update_frame()


if __name__ == '__main__':

    pytest.main([__file__])
