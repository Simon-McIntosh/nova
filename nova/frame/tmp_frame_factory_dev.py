#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 08:59:46 2022

@author: mcintos
"""

from nova import ImportManager

with ImportManager.deferred(False):
    from nova.biot import BiotPoint

    print(ImportManager().state, BiotPoint)
print(ImportManager().state, BiotPoint)

from nova.frame.frameset import FrameSet, frame_factory


class CoilSet(FrameSet):

    @frame_factory(BiotPoint)
    def biotpoint(self):
        """bp."""
        return dict(path='wgfr')


if __name__ == '__main__':
    pass
    #coilset = CoilSet()
    #coilset.biotpoint
    #coilset.biotpoint

    #print(isinstance(coilset.biotpoint, FrameData))
