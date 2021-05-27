import pytest

from nova.electromagnetic.metaframe import MetaFrame


def test_init_metaframe():
    MetaFrame()


def test_clear():
    metaframe = MetaFrame(required=['x', 'y', 'z'])
    metaframe.clear('required')
    assert metaframe.required == []


def test_type_error():
    with pytest.raises(TypeError):
        MetaFrame(required={'x': 3})


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
    metaframe = MetaFrame(additional=[], default={'link': '', 'dCoil': -1})
    metaframe.metadata = {'additional': ['link']}
    metaframe.metadata = {'additional': ['link', 'dCoil']}
    assert metaframe.additional == ['link', 'dCoil']


def test_metadata():
    metaframe = MetaFrame(required=['x', 'z'],
                          default={'dCoil': -1},
                          additional=['dCoil'])
    metadata = {attr: metaframe.metadata[attr]
                for attr in ['required', 'additional']}
    assert metadata == {'required': ['x', 'z'], 'additional': ['dCoil']}

def test_required_number():
    metaframe = MetaFrame(required=['x', 'z'])
    assert len(metaframe.required) == 2


if __name__ == '__main__':

    pytest.main([__file__])
