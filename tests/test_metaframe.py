import pytest

from nova.electromagnetic.frame import MetaFrame


def test_init_metaframe():
    MetaFrame()


def test_replace_required():
    metaframe = MetaFrame(required=['x', 'z'])
    metaframe.metadata = {'required': []}
    metaframe.metadata = {'required': ['dx', 'dz']}
    assert metaframe.required == ['dx', 'dz']


def test_clear_required():
    metaframe = MetaFrame(required=['x', 'z'])
    metaframe.clear('required')
    metaframe.metadata = {'required': ['dx', 'dz']}
    assert metaframe.required == ['dx', 'dz']


def test_extend_additional():
    metaframe = MetaFrame(additional=[], default={'mpc': '', 'dCoil': -1})
    metaframe.metadata = {'additional': ['mpc']}
    metaframe.metadata = {'additional': ['mpc', 'dCoil']}
    assert metaframe.additional == ['mpc', 'dCoil']


def test_required_default():
    metaframe = MetaFrame(default={'dCoil': -1}, additional=['dCoil'])
    with pytest.raises(ValueError):
        assert setattr(metaframe, 'metadata', {'additional': ['dShell']})


def test_metadata():
    metaframe = MetaFrame(required=['x', 'z'],
                          default={'dCoil': -1},
                          additional=['dCoil'])
    metadata = {attr: metaframe.metadata[attr]
                for attr in ['required', 'default', 'additional']}
    assert metadata == {'required': ['x', 'z'], 'default': {'dCoil': -1},
                        'additional': ['dCoil']}


def test_required_number():
    metaframe = MetaFrame(required=['x', 'z'])
    assert metaframe.required_number == 2


if __name__ == '__main__':

    pytest.main([__file__])
