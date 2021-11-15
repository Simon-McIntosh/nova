import pytest
import vedo

from nova.electromagnetic.coilset import CoilSet

def test_insert():

    coilset = CoilSet()
    box = vedo.shapes.Box(pos=(5, 0, 0), length=1, width=2, height=3)

    print(coilset.ferritic.required, coilset.frame.metaframe.required)
    coilset.ferritic.insert(box)

    #print(coilset.Loc())

if __name__ == '__main__':

    test_insert()
