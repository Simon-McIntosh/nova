"""Fixed-shape morphology shared by host geometry and device kernels."""

import jax.numpy as jnp
import numpy as np


def _dilate4(mask: np.ndarray | jnp.ndarray) -> np.ndarray | jnp.ndarray:
    """Dilate a host or device mask within its bounds by four neighbours."""
    array = np if isinstance(mask, np.ndarray) else jnp
    padded = array.pad(mask, ((1, 1), (1, 1)), constant_values=False)
    return (
        padded[1:-1, 1:-1]
        | padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )
