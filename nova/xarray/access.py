#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:13:16 2024

@author: mcintos
"""
import jax
import jax.numpy as jnp

from jax import config
import matplotlib.pyplot as plt

import numpy as np

# from nova.xarray import xarray


config.update("jax_debug_nans", True)


x, dx = jnp.linspace(0, 12, 35, retstep=True)

# def intergrate(current_density, delta_psi)


#  @jax.jit
def sum_width(r, R):
    x_prime = jnp.abs(x - R) / r
    current_density = jnp.where(x_prime <= 1.0, 200 * (1 - x_prime), 0.0)
    area = jnp.where(x_prime <= 1.0, dx, 0.0)
    area_sum = jnp.sum(area)
    factor = (2 * r) / jnp.where(area_sum > 0, area_sum, 2 * r)

    return jnp.sum(current_density * area) * factor


if __name__ == "__main__":
    grad = jax.grad(sum_width)
    r_vector = jnp.linspace(0.5, 6.0, 250)
    plt.plot(r_vector, [grad(r, 6.0) for r in r_vector])

    finite_grad = np.zeros_like(r_vector)
    delta_x = 0.01
    for i, r in enumerate(r_vector):
        finite_grad[i] = (sum_width(r + delta_x, 6.0) - sum_width(r - delta_x, 6.0)) / (
            2 * delta_x
        )

    plt.plot(r_vector, finite_grad)
