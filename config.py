import os

USE_JAX = False

if USE_JAX:
    from jax.config import config
    config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    K = jnp
    dtype = jnp.float64
    print('Using JAX backend')
else:
    import numpy as np
    K = np
    dtype = np.float64
    print('Using NUMPY backend')
