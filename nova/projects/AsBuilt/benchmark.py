#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:55:25 2022

@author: mcintos
"""

@dataclass
class BenchMark(Model):
    """Perfom benchmark tests on broadband ANSYS gap simulations."""

    dataset: dict[str, float] = field(
        default_factory=lambda: dict(v3=4.478, a1=0.8932, a2=0.7694,
                                     c1=0.9847, c2=1.7273))

    @property
    def simulation(self):
        """Return list of benchmark simulations."""
        return list(self.dataset.keys())
