import pytest
import numpy as np
import pandas

from nova.electromagnetic.coilframe import CoilFrame


def test_instance():
    frame = CoilFrame()
    assert isinstance(frame, CoilFrame)


def test_subclass():
    assert issubclass(CoilFrame, pandas.DataFrame)


def test_columns():
    frame = CoilFrame(columns=['x', 'z'])
    assert sorted(frame.columns) == sorted(['x', 'z'])


def test_index():
    frame = CoilFrame(index=['Coil0', 'Coil1'])
    assert sorted(frame.index) == sorted(['Coil0', 'Coil1'])


def test_default_columns():
    frame = CoilFrame(metadata={'required': ['x', 'z']})
    frame.add_coil(0, 4)
    print(frame)


def test_init():
    frame = pandas.DataFrame({'x': [1, 2], 'z': [3, 3]})
    #frame.attrs['metadata'] = MetaData()

    frame = CoilFrame(frame)
    print(frame.attrs)


if __name__ == '__main__':

    #pytest.main([__file__])
    test_default_columns()