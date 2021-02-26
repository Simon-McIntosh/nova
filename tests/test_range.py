
import pytest
import numpy as np
import pandas

from nova.electromagnetic.frame import Frame


def test_init():
    frame = Frame(link=True, metadata={'additional': ['link']})
    frame.add_frame(4, [5, 7, 12], 0.1, 0.05)
    return frame


if __name__ == '__main__':

    frame = test_init()

    print(frame)
    print(frame.range)