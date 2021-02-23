
import pytest

from nova.electromagnetic.frame import MetaFrame


def test_clear():
    metaframe = MetaFrame(required=['x', 'y', 'z'])
    metaframe.clear('required')
    assert metaframe.required == []


def test_type_error():
    with pytest.raises(TypeError):
        MetaFrame(required={'x': 3})


if __name__ == '__main__':

    pytest.main([__file__])
