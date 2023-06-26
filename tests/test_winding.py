import pytest

from nova.frame.coilset import CoilSet


# def test_circle_sweep():
coilset = CoilSet(additional=["vtk"])
coilset.coil.insert(5, 3, 0.1, 0.1)
coilset.plot()


# test_circle_sweep()
