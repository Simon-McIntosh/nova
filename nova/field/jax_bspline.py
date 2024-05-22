#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 18:10:18 2024

@author: mcintos
"""
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

term = 3
coordinate = jnp.linspace(0, 1, 51)
order = 3


def binom(x, y):
    """Return Binomial coefficient considered as a function of two real variables.

    Function adaptation using a Jax backed algorithum is taken from
    https://stackoverflow.com/questions/74759936"""

    return jnp.exp(
        jax.scipy.special.gammaln(x + 1)
        - jax.scipy.special.gammaln(y + 1)
        - jax.scipy.special.gammaln(x - y + 1)
    )


@jax.jit
def basis(term, order, coordinate):
    return binom(order, term) * coordinate**term * (1 - coordinate) ** (order - term)


plt.plot(coordinate, basis(term, order, coordinate))
