import jax.numpy as jnp
import jax

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


def build_armijo_linesearch(f, decrease_ratio=0.5, slope=0.05, max_iter=25):
    def armijo_linesearch(x, f_curr, d, g, t0=1.0):
        """
        x: current parameters (pytree)
        f_curr: f(x)
        d: descent direction (pytree)
        g: gradient at x (pytree)
        t0: initial step size
        a: Armijo constant
        """
        candidate = x - t0 * d  # tree_add(x, tree_scale(d, -t0))
        dec0 = f(candidate) - f_curr
        pred_dec0 = -t0 * jnp.dot(d, g)  # tree_dot(d, g)

        # The loop state: (iteration, t, current decrease, predicted decrease)
        init_state = (0, t0, dec0, pred_dec0)

        def cond_fun(state):
            i, t, dec, pred_dec = state
            # Continue while we haven't satisfied the Armijo condition and haven't exceeded max_iter iterations.
            not_enough_decrease = dec >= slope * pred_dec
            return jnp.logical_and(i < max_iter, not_enough_decrease)

        def body_fun(state):
            i, t, dec, pred_dec = state
            t_new = decrease_ratio * t
            candidate_new = x - t_new * d
            dec_new = f(candidate_new) - f_curr
            pred_dec_new = -t_new * jnp.dot(d, g)  # tree_dot(d, g)
            return (i + 1, t_new, dec_new, pred_dec_new)

        # Run the while loop
        i_final, t_final, dec_final, pred_dec_final = jax.lax.while_loop(
            cond_fun, body_fun, init_state
        )
        armijo_rat_final = dec_final / pred_dec_final
        candidate_final = x - t_final * d
        return candidate_final, t_final, armijo_rat_final

    return armijo_linesearch
