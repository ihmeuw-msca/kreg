import jax
import jax.numpy as jnp


def cartesian_prod(x: jax.Array, y: jax.Array):
    """
    Computes Cartesian product of two arrays x,y
    """
    a, b = jnp.meshgrid(y, x)
    full_X = jnp.vstack([b.flatten(), a.flatten()]).T
    return full_X
