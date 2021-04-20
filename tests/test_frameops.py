import pytest

from nova.electromagnetic.coilset import CoilSet


def test_get_current_frame():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[], required=['x'])
    coilset.coil.insert(1, Ic=[7.7])
    assert coilset.current.to_list() == [7.7, 7.7]


def test_get_current_subspace():
    coilset = CoilSet(dcoil=-2, subspace=['Ic'], array=[], required=['x'])
    coilset.coil.insert([1, 3, 7], Ic=[7.7, 8.3, 6.6])
    assert coilset.current.to_list() == [7.7, 8.3, 6.6]


def test_get_current_subspace_array():
    coilset = CoilSet(dcoil=-1, subspace=['Ic'], array=['Ic'], required=['x'])
    coilset.coil.insert([1, 3], Ic=[7.7, 6.6])
    assert list(coilset.current) == [7.7, 6.6]


def test_set_current_frame():
    coilset = CoilSet(dcoil=-2, subspace=[], array=[], required=['x'])
    coilset.coil.insert(1.5)
    coilset.current = [8.8, 7.7]
    assert coilset.current.to_list() == [8.8, 7.7]


def test_set_current_subspace():
    coilset = CoilSet(dcoil=-2, subspace=['Ic'], array=[], required=['x'])
    coilset.coil.insert(1.5)
    coilset.current = [8.8]
    assert coilset.current.to_list() == [8.8]


def test_set_current_subspace_error():
    coilset = CoilSet(dcoil=-2, subspace=['Ic'], array=[], required=['x'])
    coilset.coil.insert(1.5)
    with pytest.raises(ValueError):
        coilset.current = [8.8, 8.8]


def test_set_current_subspace_array():
    coilset = CoilSet(dcoil=-2, subspace=['Ic'], array=['Ic'], required=['x'])
    coilset.coil.insert(1.5)
    coilset.current = [8.8]
    assert list(coilset.current) == [8.8]


def test_get_current_subset():
    coilset = CoilSet(dcoil=-1, subspace=['Ic'], array=['Ic'], required=['x'])
    coilset.coil.insert(3, plasma=False)
    coilset.coil.insert(6.6, plasma=True)
    coilset.coil.insert([1.2, 2.2], active=False)
    coilset.link(['Coil0', 'Coil3'])
    with coilset.switch('active'):
        coilset.current = [8.8, 7.7]
    assert list(coilset.subframe.Ic) == [8.8, 7.7, 0, 8.8]


def test_get_current_insert_default():
    coilset = CoilSet(dcoil=-1, subspace=['Ic'], array=['Ic'], required=['x'])
    coilset.coil.insert(3)
    coilset.plasma.insert({'o': [1, 0, 0.5]})
    coilset.shell.insert([1.2, 2.2], [2.2, 2.2], -1, 0.5)
    coilset.current = 4.4
    with coilset.switch('plasma'):
        coilset.current = [9.9]
    with coilset.switch('active'):
        coilset.current = [3.3]
    with coilset.switch('coil'):
        coilset.current = [5.5]
    assert list(coilset.current) == [5.5, 3.3, 4.4]


if __name__ == '__main__':

    pytest.main([__file__])

