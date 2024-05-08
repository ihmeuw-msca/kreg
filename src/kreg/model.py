from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from tqdm.auto import tqdm

from kreg.kernel.kron_kernel import KroneckerKernel
from kreg.likelihood import Likelihood
from kreg.precon.nystroem import build_nystroem_precon
from kreg.typing import Callable, JAXArray

# TODO: Inexact solve, when to quit
jax.config.update("jax_enable_x64", True)


class KernelRegModel:
    def __init__(
        self,
        kernel: KroneckerKernel,
        likelihood: Likelihood,
        lam: float,
    ) -> None:
        self.kernel = kernel
        self.likelihood = likelihood
        self.lam = lam

    @partial(jax.jit, static_argnums=0)
    def objective(self, x: JAXArray) -> JAXArray:
        return (
            self.likelihood.objective(x)
            + 0.5 * self.lam * x.T @ self.kernel.op_p @ x
        )

    @partial(jax.jit, static_argnums=0)
    def gradient(self, x: JAXArray) -> JAXArray:
        return self.likelihood.gradient(x) + self.lam * self.kernel.op_p @ x

    def hessian(self, x: JAXArray) -> Callable:
        hess_diag = self.likelihood.hessian_diag(x)

        def op_hess(z: JAXArray) -> JAXArray:
            return hess_diag * z + self.lam * self.kernel.op_p @ z

        return op_hess

    def optimize(
        self,
        y0: JAXArray | None = None,
        max_newton_cg: int = 25,
        grad_tol: float = 1e-3,
        max_cg_iter: int = 100,
        scaling_cg_iter: int = 25,
        nystroem_rank: int = 25,
    ) -> tuple[JAXArray, dict]:
        """ """
        rng_key = jax.random.PRNGKey(101)
        rng_key, *split_keys = jax.random.split(rng_key, 2 * max_newton_cg)

        if y0 is None:
            y0 = jnp.zeros(len(self.kernel))

        y = y0.copy()

        loss_vals = []
        grad_norms = []
        newton_decrements = []
        iterate_maxnorm_distances = []
        converged = False
        # default preconditioner
        precon = self.kernel.dot
        for i in tqdm(range(max_newton_cg)):
            val, g, hess = self.objective(y), self.gradient(y), self.hessian(y)

            # Check for convergence
            if jnp.linalg.vector_norm(g) <= grad_tol:
                converged = True
                conv_crit = "grad_norm"
                break

            loss_vals.append(val)
            grad_norms.append(jnp.linalg.vector_norm(g))

            # update preconditioner
            if nystroem_rank > 0 and i % 10 == 0:
                precon = build_nystroem_precon(
                    self.likelihood.hessian_diag(y),
                    self.kernel,
                    self.lam,
                    split_keys[i],
                    rank=nystroem_rank,
                )

            num_cg_iter = max_cg_iter + scaling_cg_iter * i
            step, info = cg(hess, g, M=precon, maxiter=num_cg_iter, tol=1e-16)

            newton_decrements.append(jnp.sqrt(jnp.dot(g, step)))
            # Hard coded line search
            step_size = 1.0
            alpha = 0.1
            new_y = y - step_size * step
            new_val = self.objective(new_y)
            while new_val - val >= step_size * alpha * jnp.dot(
                g, step
            ) or jnp.isnan(new_val):
                step_size = 0.2 * step_size
                new_y = y - step_size * step
                new_val = self.objective(new_y)

            iterate_maxnorm_distances.append(jnp.max(jnp.abs(step_size * step)))
            y = new_y
        if not converged:
            conv_crit = "Did not converge"
            print(f"Convergence wasn't achieved in {max_newton_cg} iterations")

        val, g = self.objective(y), self.gradient(y)
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
        return y, convergence_data
