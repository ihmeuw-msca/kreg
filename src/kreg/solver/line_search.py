import jax.numpy as jnp

from kreg.typing import Callable, JAXArray


def armijo_line_search(
    x: JAXArray,
    p: JAXArray,
    g: JAXArray,
    objective: Callable,
    step_init: float = 1.0,
    alpha: float = 0.6,
    shrinkage: float = 0.2,
):
    step = step_init
    new_x = x - step * p
    val, new_val = objective(x), objective(new_x)
    while new_val - val >= step * alpha * jnp.dot(g, p) or jnp.isnan(new_val):
        if step<=1e-15:
            raise RuntimeError(f"Line Search Failed,new_val = {new_val},prev_val = {val}")
        step *= shrinkage
        new_x = x - step * p
        new_val = objective(new_x)

    return step
