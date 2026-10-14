import jax
import jax.numpy as jnp
import pytest

from kreg.solver import newton
from kreg.solver.newton import NewtonCG, NewtonDirect

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


def hessian_mat(x):
    return A


def make_solver(solver_class):
    if solver_class is NewtonCG:
        return NewtonCG(objective, gradient, hessian)
    return NewtonDirect(objective, gradient, hessian_mat)


@pytest.mark.parametrize("solver_class", [NewtonCG, NewtonDirect])
def test_newton_solves_quadratic(solver_class):
    x, result = make_solver(solver_class).solve(
        jnp.zeros(2), gtol=1e-10, verbose=False
    )
    assert jnp.allclose(x, jnp.linalg.solve(A, b))
    assert result.converged
    assert result.convergence_criterion == "grad_norm"
    log = result.log.to_dataframe()
    assert {"iter", "obj_val", "gnorm2", "step"} <= set(log.columns)
    assert log["iter"].tolist() == list(range(len(log)))


@pytest.mark.parametrize("solver_class", [NewtonCG, NewtonDirect])
def test_newton_warns_when_not_converged(solver_class):
    with pytest.warns(UserWarning, match="Convergence wasn't achieved"):
        _, result = make_solver(solver_class).solve(
            jnp.zeros(2), max_iter=0, verbose=False
        )
    assert not result.converged
