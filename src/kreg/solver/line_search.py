import jax.numpy as jnp

from kreg.typing import Callable, JAXArray


def armijo_line_search(
    x: JAXArray,
    p: JAXArray,
    g: JAXArray,
    objective: Callable,
    gradient: Callable,
    step_init: float = 1.0,
    alpha: float = 0.1,
    shrinkage: float = 0.5,
    grad_decrease=1.0,
):
    def sufficiently_improved(new_val, step):
        return (new_val - val <= -1 * alpha * step * jnp.dot(g, p)) and (
            not jnp.isnan(new_val)
        )

    step = step_init
    new_x = x - step * p
    val, new_val, new_grad = objective(x), objective(new_x), gradient(new_x)
    while (not sufficiently_improved(new_val, step)) and (
        (jnp.linalg.norm(new_grad) / jnp.linalg.norm(g)) > (1 - grad_decrease)
    ):
        if step <= 1e-15:
            raise RuntimeError(
                f"Line Search Failed, new_val = {new_val}, prev_val = {val}"
            )
        step *= shrinkage
        new_x = x - step * p
        new_val = objective(new_x)
    armijo_ratio = (val - new_val) / (step * jnp.dot(g, p))

    return step, armijo_ratio, (jnp.linalg.norm(new_grad) / jnp.linalg.norm(g))
