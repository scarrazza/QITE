import os

if 'USE_JAX' in os.environ:
    import jax.numpy as jnp
    K = jnp
    dtype = jnp.float32
    print('Using JAX backend')
else:
    import numpy as np
    K = np
    dtype = np.float32
    print('Using NUMPY backend')
