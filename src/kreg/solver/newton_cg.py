import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax.scipy.sparse.linalg import cg
from tqdm.auto import tqdm

from kreg.solver.line_search import armijo_line_search
from kreg.typing import Callable, JAXArray


class NewtonCG:
    def __init__(
        self,
        objective: Callable,
        gradient: Callable,
        hessian: Callable,
        precon_builder: Callable | None = None,
    ) -> None:
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.precon_builder = precon_builder

    def solve(
        self,
        x0: JAXArray,
        max_iter: int = 25,
        gtol: float = 1e-3,
        cg_maxiter: int = 100,
        cg_maxiter_increment: int = 25,
        precon_build_freq: int = 10,
        disable_tqdm=False,
        grad_decrease=0.25,
    ) -> tuple[JAXArray, dict]:
        loss_vals = []
        grad_norms = []
        newton_decrements = []
        iterate_maxnorm_distances = []
        converged = False

        x = x0.copy()
        precon = None

        for i in tqdm(range(max_iter), disable=disable_tqdm):
            val, g, hess = self.objective(x), self.gradient(x), self.hessian(x)

            # Check for convergence
            if jnp.linalg.vector_norm(g) <= gtol:
                converged = True
                conv_crit = "grad_norm"
                break

            loss_vals.append(val)
            grad_norms.append(jnp.linalg.vector_norm(g))

            # update preconditioner
            if self.precon_builder is not None and i % precon_build_freq == 0:
                precon = self.precon_builder(x)

            cg_maxiter += cg_maxiter_increment
            p, info = cg(hess, g, M=precon, maxiter=cg_maxiter, tol=1e-16)

            newton_decrements.append(jnp.sqrt(jnp.dot(g, p)))
            # Hard coded line search
            step, armijo_ratio, gradnorm_ratio = armijo_line_search(
                x,
                p,
                g,
                self.objective,
                self.gradient,
                grad_decrease=grad_decrease,
            )
            iterate_maxnorm_distances.append(jnp.max(jnp.abs(step * p)))
            x = x - step * p
        if not converged:
            conv_crit = "Did not converge"
            print(f"Convergence wasn't achieved in {max_iter} iterations")

        val, g = self.objective(x), self.gradient(x)
        loss_vals.append(val)
        grad_norms.append(jnp.linalg.vector_norm(g))

        loss_vals = jnp.array(loss_vals)
        grad_norms = jnp.array(grad_norms)
        newton_decrements = jnp.array(newton_decrements)
        iterate_maxnorm_distances = jnp.array(iterate_maxnorm_distances)
        convergence_data = {
            "loss_vals": loss_vals,
            "gnorms": grad_norms,
            "converged": converged,
            "convergence_criterion": conv_crit,
            "newton_decrements": newton_decrements,
            "iterate_maxnorm_distances": iterate_maxnorm_distances,
        }

        return x, convergence_data


class NewtonDirect:
    def __init__(
        self,
        objective: Callable,
        gradient: Callable,
        hessian_mat: Callable,
    ) -> None:
        self.objective = objective
        self.gradient = gradient
        self.hessian_mat = hessian_mat

    def solve(
        self,
        x0: JAXArray,
        max_iter: int = 25,
        gtol: float = 1e-3,
        disable_tqdm=False,
        grad_decrease=0.25,
    ) -> tuple[JAXArray, dict]:
        loss_vals = []
        grad_norms = []
        newton_decrements = []
        iterate_maxnorm_distances = []
        armijo_ratios = []
        gradnorm_ratios = []
        stepsizes = []
        converged = False

        x = x0.copy()

        for i in tqdm(range(max_iter), disable=disable_tqdm):
            val, g, hess = (
                self.objective(x),
                self.gradient(x),
                self.hessian_mat(x),
            )

            # Check for convergence
            if jnp.linalg.vector_norm(g) <= gtol:
                converged = True
                conv_crit = "grad_norm"
                break

            loss_vals.append(val)
            grad_norms.append(jnp.linalg.vector_norm(g))

            p = solve(hess, g, assume_a="pos")

            newton_decrements.append(jnp.sqrt(jnp.dot(g, p)))
            # Hard coded line search
            step, armijo_ratio, gradnorm_ratio = armijo_line_search(
                x,
                p,
                g,
                self.objective,
                self.gradient,
                grad_decrease=grad_decrease,
            )
            iterate_maxnorm_distances.append(jnp.max(jnp.abs(step * p)))
            armijo_ratios.append(armijo_ratio)
            gradnorm_ratios.append(gradnorm_ratio)
            stepsizes.append(step)
            x = x - step * p
        if not converged:
            conv_crit = "Did not converge"
            print(f"Convergence wasn't achieved in {max_iter} iterations")

        val, g = self.objective(x), self.gradient(x)
        loss_vals.append(val)
        grad_norms.append(jnp.linalg.vector_norm(g))

        loss_vals = jnp.array(loss_vals)
        grad_norms = jnp.array(grad_norms)
        newton_decrements = jnp.array(newton_decrements)
        iterate_maxnorm_distances = jnp.array(iterate_maxnorm_distances)
        armijo_ratios = jnp.array(armijo_ratios)
        gradnorm_ratios = jnp.array(gradnorm_ratios)
        stepsizes.append(stepsizes)
        convergence_data = {
            "loss_vals": loss_vals,
            "gnorms": grad_norms,
            "converged": converged,
            "convergence_criterion": conv_crit,
            "newton_decrements": newton_decrements,
            "iterate_maxnorm_distances": iterate_maxnorm_distances,
            "armijo_ratios": armijo_ratios,
            "gradnorm_ratios": gradnorm_ratios,
            "stepsizes": stepsizes,
        }

        return x, convergence_data
