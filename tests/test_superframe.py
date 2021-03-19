import pytest
import pandas

from nova.electromagnetic.dataframe import DataFrame


def test_Ic_unset():
    dataframe = DataFrame(Required=['x'])
    dataframe.add_frame([4, 5], It=6.5)
    assert dataframe.Ic.to_list() == [6.5, 6.5]


def test_data_It_Ic_unset():
    dataframe = DataFrame({'x': [1, 2], 'It': 5, 'Nt': 2.5}, Required=['x'])
    assert dataframe.Ic.to_list() == [2, 2]


def test_data_It_Ic_set():
    dataframe = DataFrame({'x': [1, 2], 'It': 5, 'Ic': 10, 'Nt': 2.5},
                          Required=['x'])
    assert dataframe.Ic.to_list() == [10, 10]
    assert dataframe.It.to_list() == [25, 25]


def test_Ic_unset_Additional():
    dataframe = DataFrame(Required=['x', 'z'], Additional=['Ic'])
    dataframe.add_frame(4, range(2), It=5)
    assert dataframe.Ic.to_list() == [5, 5]