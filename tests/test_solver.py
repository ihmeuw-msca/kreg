import jax
import jax.numpy as jnp

from kreg.solver import newton
from kreg.solver.newton import NewtonCG

jax.config.update("jax_enable_x64", True)

A = jnp.array([[2.0, 0.5], [0.5, 1.0]])
b = jnp.array([1.0, -1.0])


def objective(x):
    return 0.5 * x @ A @ x - b @ x


def gradient(x):
    return A @ x - b


def hessian(x):
    return lambda v: A @ v


def test_newton_cg_budget_starts_at_cg_maxiter(monkeypatch):
    budgets = []
    real_cg = newton.cg

    def cg(*args, maxiter, **kwargs):
        budgets.append(maxiter)
        return real_cg(*args, maxiter=maxiter, **kwargs)

    monkeypatch.setattr(newton, "cg", cg)
    solver = NewtonCG(objective, gradient, hessian)
    solver.solve(
        jnp.zeros(2),
        max_iter=3,
        gtol=0.0,
        cg_maxiter=7,
        cg_maxiter_increment=3,
        verbose=False,
    )
    assert budgets[0] == 7
    assert budgets == [7, 10, 13][: len(budgets)]
