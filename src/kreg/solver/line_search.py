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
        new_val, new_grad = objective(new_x), gradient(new_x)
    armijo_ratio = (val - new_val) / (step * jnp.dot(g, p))

    return step, armijo_ratio, (jnp.linalg.norm(new_grad) / jnp.linalg.norm(g))


def armijo_line_search_other(
    gradient: Callable,
    x: JAXArray,
    dx: JAXArray,
    step_init: float = 1.0,
    step_const: float = 0.01,
    step_scale: float = 0.9,
    step_lb: float = 1e-3,
) -> float:
    """Armijo line search.

    Parameters
    ----------
    x
        A list a parameters, including x, s, and v, where s is the slackness
        variable and v is the dual variable for the constraints.
    dx
        A list of direction for the parameters.
    step_init
        Initial step size, by default 1.0.
    step_const
        Constant for the line search condition, the larger the harder, by
        default 0.01.
    step_scale
        Shrinkage factor for step size, by default 0.9.
    step_lb
        Lower bound of the step size when the step size is below this bound
        the line search will be terminated.

    Returns
    -------
    float
        The step size in the given direction.

    """
    step = step_init
    x_next = x + step * dx
    g_next = gradient(x_next)
    gnorm_curr = jnp.max(jnp.abs(gradient(x)))
    gnorm_next = jnp.max(jnp.abs(g_next))

    while gnorm_next > (1 - step_const * step) * gnorm_curr:
        if step * step_scale < step_lb:
            break
        step *= step_scale
        x_next = x + step * dx
        g_next = gradient(x_next)
        gnorm_next = jnp.max(jnp.abs(g_next))

    return step
