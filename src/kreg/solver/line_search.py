import jax.numpy as jnp

from kreg.typing import Callable, JAXArray


def armijo_line_search(
    x: JAXArray,
    p: JAXArray,
    g: JAXArray,
    objective: Callable,
    step_init: float = 1.0,
    alpha: float = 0.1,
    shrinkage: float = 0.2,
):
    step = step_init
    new_x = x - step * p
    val, new_val = objective(x), objective(new_x)
    while new_val - val >= step * alpha * jnp.dot(g, p) or jnp.isnan(new_val):
        step *= shrinkage
        new_x = x - step * p
        new_val = objective(new_x)

    return step
