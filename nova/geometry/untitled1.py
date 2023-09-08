#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:41:18 2023

@author: mcintos
"""
import numpy as np


class Sweep:
    """Sweep polygon along path."""

    poly: dict[str, list[float]] | list[float] | np.ndarray
    path: np.ndarray
    delta: int = 0
    cap: bool = False
    origin: np.ndarray = np.zeros(3, float)
    triad: np.ndarray = np.identity(3, float)

    def __init__(self):
        type(self).triad += 1


if __name__ == "__main__":
    sweep = Sweep()

    # sweep.triad *= -1

    print(Sweep().triad)
    print(Sweep().triad)

    print(Sweep().triad)

    print(sweep.triad)
